"""最小 Jev-style：冻结 backbone -> 候选条件特征 -> 共享标量头 -> softmax。"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .data import records


def validate(rows):
    seen_ids, seen_text = set(), set()
    groups = {}
    for row in rows:
        if row["id"] in seen_ids or row["state"] in seen_text:
            raise ValueError("重复 id 或文本，请先检查 split 泄漏")
        seen_ids.add(row["id"])
        seen_text.add(row["state"])
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError("同一来源 group 不能跨 split")
        if not 2 <= len(row["candidates"]) <= 32 or row["target"] not in row["candidates"]:
            raise ValueError("候选数量或 target 无效")
        if not all(isinstance(v, str) and v.strip() for v in row["candidates"].values()):
            raise ValueError("候选描述不能为空")


def prompt(row, description):
    # 不把 target、split、候选序号或候选 ID 放进模型输入。
    return (f"任务规则：{row['question']}\n用户输入：{row['state']}\n"
            f"候选动作：{description}\n判断这个候选是否符合用户当前意图。判断依据：")


class Encoder:
    def __init__(self, model, revision, device, local_only=False):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as error:
            if error.name != "transformers":
                raise
            raise RuntimeError(
                "语言模型课程需要额外依赖：python -m pip install -r requirements-models.txt。"
                "仅玩推箱子不需要这些依赖。"
            ) from error
        self.device = torch.device(device)
        options = {"revision": revision, "local_files_only": local_only}
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="right", **options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.backbone = AutoModel.from_pretrained(
            model, dtype=dtype, attn_implementation="sdpa", **options
        ).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, rows, batch_size=4, max_length=256):
        texts = [prompt(row, desc) for row in rows for desc in row["candidates"].values()]
        flat = self.encode_texts(texts, batch_size, max_length)
        return list(flat.split([len(row["candidates"]) for row in rows]))

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=4, max_length=256):
        """Expose the same frozen encoder for other explicit decision prompts."""
        features = []
        for start in range(0, len(texts), batch_size):
            # 宁可明确报错，也不静默截掉用户意图或候选。
            batch = self.tokenizer(texts[start:start + batch_size], padding=True, return_tensors="pt")
            if batch.input_ids.shape[1] > max_length:
                raise ValueError(f"输入超过 {max_length} tokens；缩短输入或明确调大上限")
            batch = batch.to(self.device)
            hidden = self.backbone(**batch, use_cache=False).last_hidden_state
            last = batch.attention_mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(len(last), device=self.device), last].float()
            # 固定、无可训练参数的归一化，使不同尺度更容易训练。
            features.append(F.layer_norm(pooled, (pooled.shape[-1],)).cpu())
        return torch.cat(features)


def make_head(width, seed=42):
    torch.manual_seed(seed)
    # bias 对所有候选相同，会被 softmax 抵消，因此不需要。
    return nn.Linear(width, 1, bias=False)


def fit_head(features, targets, dev_features, dev_targets, steps=200, lr=0.01, seed=42):
    head = make_head(features[0].shape[-1], seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    # 此处教学数据每题都是四选一；推理和指标函数支持变长候选。
    x, y = torch.stack(features), torch.tensor(targets)
    dx, dy = torch.stack(dev_features), torch.tensor(dev_targets)
    best_loss, best_state, best_step = float("inf"), None, 0
    history = []
    for step in range(steps + 1):
        with torch.no_grad():
            dev_loss = F.cross_entropy(head(dx).squeeze(-1), dy).item()
        if dev_loss < best_loss:
            best_loss, best_step = dev_loss, step
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if step % 25 == 0 or step == steps:
            history.append({"step": step, "dev_nll": dev_loss})
        if step == steps:
            break
        loss = F.cross_entropy(head(x).squeeze(-1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.load_state_dict(best_state)
    return head.eval(), {"selected_step": best_step, "dev_nll": best_loss, "history": history}


@torch.no_grad()
def metrics(head, features, rows):
    predictions = []
    correct, nll, brier, max_order_error = 0, 0.0, 0.0, 0.0
    for feature, row in zip(features, rows):
        keys = list(row["candidates"])
        probs = head(feature).squeeze(-1).softmax(-1)
        target = keys.index(row["target"])
        chosen = int(probs.argmax())
        correct += chosen == target
        nll -= float(probs[target].clamp_min(1e-12).log())
        truth = F.one_hot(torch.tensor(target), len(keys)).float()
        brier += float((probs - truth).square().sum())
        # 检查共享 head 的置换等变性；不是重新做 GPU 前向的一致性测试。
        reverse_probs = head(feature.flip(0)).squeeze(-1).softmax(-1).flip(0)
        max_order_error = max(max_order_error, float((probs - reverse_probs).abs().max()))
        predictions.append({"id": row["id"], "state": row["state"], "target": row["target"],
                            "choice": keys[chosen], "probabilities": dict(zip(keys, probs.tolist()))})
    count = len(rows)
    return {"count": count, "correct": correct, "accuracy": correct / count,
            "nll": nll / count, "brier_sum_per_question": brier / count,
            "head_permutation_max_error": max_order_error, "predictions": predictions}


def fingerprint(model):
    directory = Path(model)
    if not directory.is_dir():
        return {"model_id": model, "note": "使用 --revision 固定远端 commit"}
    paths = sorted(p for p in directory.iterdir() if p.name in {
        "config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"
    } or p.suffix == ".safetensors")
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/first"))
    parser.add_argument("--text", default="不要取消，只是暂停")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps 和 batch-size 必须为正数")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    if args.mode == "train" and args.out.exists():
        parser.error("输出目录已存在；换一个 --out，保留之前的结果")
    checkpoint = None
    if args.mode == "predict":
        checkpoint = torch.load(args.out / "head.pt", map_location="cpu", weights_only=True)
        args.model, args.revision = checkpoint["model"], checkpoint["revision"]
    print(f"加载冻结 backbone: {args.model}", flush=True)
    encoder = Encoder(args.model, args.revision, args.device, args.local_only)
    if checkpoint is not None:
        if fingerprint(args.model) != checkpoint["model_fingerprint"]:
            raise ValueError("模型文件身份改变，拒绝将旧 head 用于新 backbone")
        row = records()[0] | {"state": args.text}
        features = encoder.encode([row], args.batch_size)
        head = make_head(features[0].shape[-1])
        head.load_state_dict(checkpoint["state_dict"])
        print(json.dumps(metrics(head, features, [row])["predictions"][0] | {"target": None},
                         ensure_ascii=False, indent=2))
        return
    rows = records()
    validate(rows)
    started = time.perf_counter()
    features = encoder.encode(rows, args.batch_size)
    if args.device == "cuda":
        torch.cuda.synchronize()
    feature_seconds = time.perf_counter() - started
    print(f"已编码 {len(rows)} 题；开始只训练共享 head", flush=True)
    partitions = {split: [i for i, row in enumerate(rows) if row["split"] == split]
                  for split in ("train", "dev", "test", "challenge")}
    targets = [list(row["candidates"]).index(row["target"]) for row in rows]
    selected = lambda values, split: [values[i] for i in partitions[split]]
    random_head = make_head(features[0].shape[-1])
    started = time.perf_counter()
    head, training = fit_head(selected(features, "train"), selected(targets, "train"),
                             selected(features, "dev"), selected(targets, "dev"), args.steps)
    head_seconds = time.perf_counter() - started
    report = {"scope": "hand-authored teaching development set, not production or JevBench",
              "seed": 42, "model": args.model, "revision": args.revision,
              "model_fingerprint": fingerprint(args.model),
              "data_sha256": hashlib.sha256(json.dumps(rows, ensure_ascii=False,
                                                       sort_keys=True).encode()).hexdigest(),
              "device": args.device, "torch": torch.__version__, "python": platform.python_version(),
              "feature_seconds": feature_seconds, "head_train_seconds": head_seconds,
              "trainable_parameters": sum(p.numel() for p in head.parameters()),
              "backbone_trainable_parameters": sum(p.numel() for p in encoder.backbone.parameters()
                                                   if p.requires_grad),
              "training": training, "splits": {}}
    import transformers
    report["transformers"] = transformers.__version__
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    for split in partitions:
        xs, rs = selected(features, split), selected(rows, split)
        report["splits"][split] = {"random_head": metrics(random_head, xs, rs),
                                   "trained_head": metrics(head, xs, rs),
                                   "uniform_expected_accuracy": 0.25,
                                   "majority_accuracy": max(sum(r["target"] == label for r in rs)
                                                            for label in rs[0]["candidates"]) / len(rs)}
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"state_dict": head.state_dict(), "model": args.model, "revision": args.revision,
                "model_fingerprint": report["model_fingerprint"]}, args.out / "head.pt")
    for split, result in report["splits"].items():
        score = result["trained_head"]
        print(f"{split}: {score['correct']}/{score['count']} = {score['accuracy']:.1%}")
    print(f"结果保存在 {args.out.resolve()}")


if __name__ == "__main__":
    main()
