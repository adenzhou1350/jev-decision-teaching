"""CPU-only regression tests for information boundaries and feature alignment."""

from copy import deepcopy
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from jev_teaching import browser_data, browser_train
from jev_teaching.browser_train import decision_prompt, encode, percentile, public_model_identity, score_head
from jev_teaching.mini_jev import Encoder


class TensorBatch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return TensorBatch({name: value.to(device) for name, value in self.items()})


class FakeTokenizer:
    """Fixed token IDs make padding and the expected final token inspectable."""
    sequences = {"long": [2, 5, 9], "short": [3], "medium": [4, 7]}

    def __call__(self, texts, padding, return_tensors):
        sequences = [self.sequences[text] for text in texts]
        width = max(map(len, sequences))
        return TensorBatch({
            "input_ids": torch.tensor([seq + [0] * (width - len(seq)) for seq in sequences]),
            "attention_mask": torch.tensor([[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences]),
        })


class FakeBackbone:
    def __call__(self, input_ids, attention_mask, use_cache):
        ids = input_ids.float()
        return SimpleNamespace(last_hidden_state=torch.stack(
            (ids, ids.square(), ids + 1, -ids), dim=-1))


class BrowserTrainTests(unittest.TestCase):
    def test_missing_local_model_or_hub_id_fails_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("Qwen/Qwen3.5-0.8B", str(root / "missing-snapshot"), None):
                with self.subTest(model=model):
                    output = root / "must-not-be-created"
                    arguments = ["browser_train", "--out", str(output), "--device", "cpu"]
                    if model is not None:
                        arguments += ["--model", model]
                    errors = io.StringIO()
                    with patch.object(sys, "argv", arguments), \
                            patch("jev_teaching.browser_train.Path.is_dir", autospec=True, return_value=False) as is_directory, \
                            patch.object(browser_train, "Encoder") as encoder, \
                            patch.object(browser_train, "fingerprint") as fingerprint, \
                            patch.object(browser_train, "encode") as encode_features, \
                            redirect_stderr(errors):
                        with self.assertRaises(SystemExit) as failure:
                            browser_train.main()
                    self.assertEqual(failure.exception.code, 2)
                    self.assertIn("local snapshot directory", errors.getvalue())
                    self.assertEqual(is_directory.call_args.args[0], Path(model or "models/Qwen3.5-0.8B"))
                    encoder.assert_not_called()
                    fingerprint.assert_not_called()
                    encode_features.assert_not_called()
                    self.assertFalse(output.exists())

    def test_latency_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile(list(range(1, 13)), .95), 12)
        self.assertEqual(percentile([5], .95), 5)
        with self.assertRaises(ValueError):
            percentile([], .95)

    def test_model_identity_uses_actual_model_without_local_directory(self):
        with patch("jev_teaching.browser_train.Path.is_dir", return_value=True):
            self.assertEqual(public_model_identity("private/models/a-different-model"), "a-different-model")
        with patch("jev_teaching.browser_train.Path.is_dir", return_value=False):
            self.assertEqual(public_model_identity("ExampleOrg/another-model"), "ExampleOrg/another-model")

    def test_prompt_cannot_observe_labels_ids_split_or_effects(self):
        row = browser_data.records()[0]
        description = next(iter(row["candidates"].values()))
        original = decision_prompt(row, description)
        changed = deepcopy(row)
        changed.update(id="SECRET_ID", group="SECRET_GROUP", split="SECRET_SPLIT",
                       target="SECRET_TARGET", candidates={"SECRET_KEY": description},
                       ui={"task": "SECRET_UI", "effect": {"correct": True}})
        self.assertEqual(decision_prompt(changed, description), original)
        self.assertIn(row["state"], original)
        self.assertIn(row["question"], original)
        self.assertIn(description, original)
        self.assertNotIn("SECRET", original)

    def test_flat_features_retain_row_and_candidate_alignment(self):
        rows = deepcopy(browser_data.records()[:2])
        rows[1]["candidates"] = dict(list(rows[1]["candidates"].items())[:2])

        class RecordingEncoder:
            def encode_texts(self, texts, batch_size, max_length):
                self.texts = texts
                return torch.arange(len(texts) * 3).reshape(len(texts), 3).float()

        encoder = RecordingEncoder()
        result = encode(encoder, rows, batch_size=2)
        self.assertEqual([tuple(x.shape) for x in result], [(4, 3), (2, 3)])
        self.assertEqual(encoder.texts, [decision_prompt(row, value)
                         for row in rows for value in row["candidates"].values()])
        self.assertTrue(torch.equal(torch.cat(result), torch.arange(18).reshape(6, 3).float()))

    def test_right_padding_pools_last_actual_token(self):
        encoder = Encoder.__new__(Encoder)
        encoder.device = torch.device("cpu")
        encoder.tokenizer = FakeTokenizer()
        encoder.backbone = FakeBackbone()
        texts = ["long", "short", "medium"]
        actual = encoder.encode_texts(texts, batch_size=2)
        final_ids = torch.tensor([9., 3., 7.])
        expected = F.layer_norm(torch.stack(
            (final_ids, final_ids.square(), final_ids + 1, -final_ids), dim=-1), (4,))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, encoder.encode_texts(texts, batch_size=1))
        with self.assertRaisesRegex(ValueError, "2 tokens"):
            encoder.encode_texts(["long"], max_length=2)

    def test_scoring_does_not_consult_expected_answer(self):
        row = browser_data.records()[0]
        feature = torch.tensor([[0.], [3.], [1.], [2.]])
        head = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            head.weight.fill_(1)
        predicted = score_head(head, feature, row)
        changed = {"candidates": row["candidates"], "target": "NOT_THE_ANSWER"}
        self.assertEqual(predicted, score_head(head, feature, changed))
        self.assertEqual(predicted["choice"], list(row["candidates"])[1])


if __name__ == "__main__":
    unittest.main()
