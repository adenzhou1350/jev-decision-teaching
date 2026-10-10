"""重新加载真实模型/head；检查改变候选 batch 后的实际差异，不替代质量评测。"""
import argparse
import json
from pathlib import Path

import torch

from .data import records
from .mini_jev import Encoder, fingerprint, make_head, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    destination = args.out / "runtime-check.json"
    if destination.exists():
        parser.error("runtime-check.json 已存在，请保留旧验证记录")
    torch.set_num_threads(4)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(args.out / "head.pt", weights_only=True, map_location="cpu")
    if fingerprint(checkpoint["model"]) != checkpoint["model_fingerprint"]:
        raise ValueError("backbone 身份不匹配")
    encoder = Encoder(checkpoint["model"], checkpoint["revision"], args.device, local_only=True)
    rows = [row for row in records() if row["id"] in {"test-07", "challenge-00"}]
    features = encoder.encode(rows, batch_size=4)
    head = make_head(features[0].shape[-1])
    head.load_state_dict(checkpoint["state_dict"])
    current = metrics(head, features, rows)["predictions"]
    # 重新做 backbone 前向，而不只是翻转缓存特征；同时把 batch_size 从4改为1。
    reversed_rows = [row | {"candidates": dict(reversed(list(row["candidates"].items())))} for row in rows]
    reversed_predictions = metrics(head, encoder.encode(reversed_rows, batch_size=1),
                                   reversed_rows)["predictions"]
    report = json.loads((args.out / "report.json").read_text(encoding="utf-8"))
    original = {p["id"]: p for result in report["splits"].values()
                for p in result["trained_head"]["predictions"]}
    checks = []
    for a, b in zip(current, reversed_predictions):
        old = original[a["id"]]
        checks.append({"id": a["id"], "reload_choice_equal": a["choice"] == old["choice"],
                       "reload_probability_max_error": max(abs(v - old["probabilities"][k])
                                                           for k, v in a["probabilities"].items()),
                       "new_batch_choice_equal": a["choice"] == b["choice"],
                       "new_batch_probability_max_error": max(abs(v - b["probabilities"][k])
                                                              for k, v in a["probabilities"].items())})
    result = {"scope": "two examples only; batch/padding changes under BF16, cause not isolated",
              "device": args.device, "checks": checks,
              "predictions": current, "reversed_batch1_predictions": reversed_predictions}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(c["reload_choice_equal"] and c["new_batch_choice_equal"] for c in checks):
        raise SystemExit("检测到选择翻转；记录已保存，不宣称运行一致")


if __name__ == "__main__":
    main()
