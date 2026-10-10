"""不下载权重、不使用 GPU 的数学和数据契约测试。"""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from jev_teaching.data import records
from jev_teaching.mini_jev import fit_head, make_head, metrics, prompt, validate


class TeachingTests(unittest.TestCase):
    def test_data(self):
        rows = records()
        validate(rows)
        self.assertEqual(len(rows), 56)
        self.assertEqual({r["split"] for r in rows}, {"train", "dev", "test", "challenge"})

    def test_leakage_rejected(self):
        rows = records()
        rows[-1]["group"] = rows[0]["group"]
        with self.assertRaises(ValueError):
            validate(rows)

    def test_answer_never_enters_prompt(self):
        row = records()[0]
        modified = row | {"target": "other", "split": "test", "id": "secret"}
        self.assertEqual(prompt(row, "暂停"), prompt(modified, "暂停"))

    def test_probability_and_permutation(self):
        row = records()[0]
        x = torch.randn(4, 8)
        head = make_head(8)
        original = metrics(head, [x], [row])
        reversed_row = row | {"candidates": dict(reversed(list(row["candidates"].items())))}
        reversed_result = metrics(head, [x.flip(0)], [reversed_row])
        a = original["predictions"][0]["probabilities"]
        b = reversed_result["predictions"][0]["probabilities"]
        self.assertAlmostEqual(sum(a.values()), 1, places=6)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=6)
        self.assertEqual(original["correct"], reversed_result["correct"])

    def test_variable_candidates(self):
        row = records()[0]
        row["candidates"].pop("cancel")
        validate([row])
        score = metrics(make_head(8), [torch.randn(3, 8)], [row])
        self.assertEqual(len(score["predictions"][0]["probabilities"]), 3)

    def test_training_and_reload(self):
        # 人工线性可分特征只验证优化代码，不代表语言能力。
        features = [torch.tensor([[2.0, 0.0], [-2.0, 0.0]])] * 4
        targets = [0] * 4
        initial = make_head(2)
        initial_loss = F.cross_entropy(initial(features[0]).view(1, 2), torch.tensor([0]))
        head, info = fit_head(features, targets, features, targets, steps=40)
        final_loss = F.cross_entropy(head(features[0]).view(1, 2), torch.tensor([0]))
        self.assertLess(final_loss.item(), initial_loss.item())
        self.assertGreater(info["selected_step"], 0)
        reloaded = make_head(2)
        reloaded.load_state_dict(copy.deepcopy(head.state_dict()))
        torch.testing.assert_close(head(features[0]), reloaded(features[0]), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save(head.state_dict(), path)
            reloaded.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(head(features[0]), reloaded(features[0]), rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
