"""Dataset contract and simulator tests; these do not assert learned accuracy."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
import unittest

from jev_teaching import browser_data


class BrowserDataTests(unittest.TestCase):
    def setUp(self):
        self.rows = browser_data.records()

    def test_counts_and_distinct_template_groups(self):
        self.assertEqual(Counter(row["split"] for row in self.rows),
                         {"train": 192, "dev": 32, "test": 48, "challenge": 24})
        groups = defaultdict(set)
        for row in self.rows:
            groups[row["group"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual(len({row["id"] for row in self.rows}), len(self.rows))
        self.assertEqual(len({row["state"] for row in self.rows}), len(self.rows))

    def test_visible_contract_and_balanced_positions(self):
        for row in self.rows:
            self.assertEqual(row["state"], browser_data.visible_state(row["ui"]))
            self.assertEqual(len(row["candidates"]), 4)
            self.assertIn(row["target"], row["candidates"])
            self.assertEqual(row["candidates"],
                             {b["id"]: b["label"] for b in row["ui"]["buttons"]})
            self.assertNotIn("target", row["ui"])
            self.assertNotIn(row["id"], row["state"])
            self.assertNotIn(row["group"], row["state"])
            self.assertNotIn(row["target"], row["state"])
        for split in ("train", "dev", "test", "challenge"):
            counts = Counter(r["target"] for r in self.rows if r["split"] == split)
            self.assertEqual(len(set(counts.values())), 1)

    def test_each_entity_crosses_every_action_within_its_family(self):
        effects = defaultdict(set)
        counts = Counter()
        for row in self.rows:
            key = (row["group"], row["ui"]["title"])
            selected = next(button for button in row["ui"]["buttons"]
                            if button["id"] == row["target"])
            effects[key].add(json.dumps(selected["effect"], ensure_ascii=False, sort_keys=True))
            counts[key] += 1
        self.assertTrue(effects)
        self.assertTrue(all(count == 4 for count in counts.values()))
        self.assertTrue(all(len(choices) == 4 for choices in effects.values()))

    def test_effects_do_not_consult_labels_or_mutate_ui(self):
        for row in self.rows:
            before = deepcopy(row["ui"])
            for button in row["ui"]["buttons"]:
                expected = deepcopy(before["initial_state"])
                expected.update(button["effect"])
                self.assertEqual(browser_data.apply_button(row["ui"], button["id"]), expected)
            self.assertEqual(row["ui"], before)
            with self.assertRaises(KeyError):
                browser_data.apply_button(row["ui"], "not-a-button")

    def test_shuffling_preserves_semantics(self):
        for row in self.rows:
            shuffled = deepcopy(row)
            shuffled["candidates"] = dict(reversed(list(row["candidates"].items())))
            shuffled["ui"]["buttons"].reverse()
            self.assertEqual(row["candidates"][row["target"]],
                             shuffled["candidates"][shuffled["target"]])
            self.assertEqual(browser_data.apply_button(row["ui"], row["target"]),
                             browser_data.apply_button(shuffled["ui"], shuffled["target"]))

    def test_lexical_baseline_cannot_read_answer_or_split(self):
        for row in self.rows:
            minimal = {"ui": {"task": row["ui"]["task"]}, "candidates": row["candidates"]}
            self.assertEqual(browser_data.lexical_choice(row), browser_data.lexical_choice(minimal))
            self.assertIn(browser_data.lexical_choice(minimal), row["candidates"])

    def test_wrong_button_produces_distinct_effect(self):
        for row in self.rows:
            outcomes = [repr(browser_data.apply_button(row["ui"], key)) for key in row["candidates"]]
            self.assertEqual(len(set(outcomes)), 4)


if __name__ == "__main__":
    unittest.main()
