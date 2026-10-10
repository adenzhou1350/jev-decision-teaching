"""Game-rule, oracle, and split-isolation tests (no model or GPU needed)."""

from collections import deque
from dataclasses import replace
import unittest

from jev_teaching.sokoban_env import (
    ExactOracle, Layout, State, assert_disjoint_layouts, canonical_layout,
    encode_planes, example_from_dict, example_to_dict, is_solved, legal_actions,
    make_dataset, parse_ascii, render_ascii, step,
)


class SokobanRulesTest(unittest.TestCase):
    def test_push_onto_goal_and_wall_collision(self):
        board = "#####\n#@$.#\n#####"
        layout, state = parse_ascii(board)
        blocked, moved = step(layout, state, 0)
        self.assertFalse(moved)
        self.assertEqual(blocked, state)
        pushed, moved = step(layout, state, 1)
        self.assertTrue(moved)
        self.assertTrue(is_solved(layout, pushed))
        self.assertEqual(render_ascii(layout, pushed), "#####\n# @*#\n#####")
        self.assertEqual(step(layout, pushed, 1), (pushed, False))
        self.assertEqual(ExactOracle(layout).distances[state], 1)
        self.assertEqual(legal_actions(layout, state), (1,))

    def test_cannot_push_two_boxes(self):
        layout, state = parse_ascii("#######\n#@$$..#\n#######")
        self.assertEqual(step(layout, state, 1), (state, False))

    def test_no_wraparound(self):
        layout = Layout(3, 3, frozenset(), (8,))
        state = State(3, (8,))
        self.assertEqual(step(layout, state, 3), (state, False))

    def test_reject_malformed_game(self):
        with self.assertRaises(ValueError):
            parse_ascii("####\n#+$#\n# .#\n####")
        with self.assertRaises(ValueError):
            State(1, (2, 2))
        with self.assertRaises(ValueError):
            parse_ascii("####\n#@.#\n####")

    def test_corner_deadlock_is_unsolvable(self):
        layout, state = parse_ascii("#####\n#$  #\n# @.#\n#   #\n#####")
        self.assertNotIn(state, ExactOracle(layout).distances)
        self.assertEqual(ExactOracle(layout).optimal_actions(state), ())

    def test_ascii_roundtrip_and_binary_planes(self):
        board = "######\n# @  #\n# *$.#\n######"
        layout, state = parse_ascii(board)
        self.assertEqual(render_ascii(layout, state), board)
        planes = encode_planes(layout, state)
        self.assertEqual((len(planes), len(planes[0]), len(planes[0][0])), (4, 4, 6))
        self.assertEqual(sum(map(sum, planes[1])), 2)
        self.assertEqual(sum(map(sum, planes[2])), 2)
        self.assertEqual(sum(map(sum, planes[3])), 1)
        self.assertTrue(all(value in (0.0, 1.0)
                            for plane in planes for row in plane for value in row))


class OracleAndSplitTest(unittest.TestCase):
    @staticmethod
    def forward_distance(layout, state):
        """Independent forward BFS, checked against reverse-BFS labels."""
        queue = deque([(state, 0)])
        seen = {state}
        while queue:
            current, distance = queue.popleft()
            if is_solved(layout, current):
                return distance
            for action in range(4):
                following, moved = step(layout, current, action)
                if moved and following not in seen:
                    seen.add(following)
                    queue.append((following, distance + 1))
        return None

    def test_oracle_matches_forward_bfs_including_two_boxes(self):
        for board in ("#####\n#   #\n# $.#\n#@  #\n#####",
                      "######\n# .. #\n# $$ #\n# @  #\n######"):
            layout, state = parse_ascii(board)
            oracle = ExactOracle(layout)
            self.assertEqual(oracle.distances[state], self.forward_distance(layout, state))
            current = state
            for _ in range(oracle.distances[state]):
                self.assertFalse(is_solved(layout, current))
                optimal = oracle.optimal_actions(current)
                self.assertTrue(optimal)
                before = oracle.distances[current]
                current, moved = step(layout, current, optimal[0])
                self.assertTrue(moved)
                self.assertEqual(oracle.distances[current], before - 1)
            self.assertTrue(is_solved(layout, current))
            self.assertEqual(oracle.optimal_actions(current), ())

    def test_multiple_optimal_first_actions_are_retained(self):
        # Walk above OR below the box to get behind it, then push onto the goal.
        layout, state = parse_ascii("######\n#    #\n# $+ #\n#    #\n######")
        oracle = ExactOracle(layout)
        actions = oracle.optimal_actions(state)
        self.assertEqual(actions, (0, 2))
        self.assertEqual(oracle.distances[state], 5)
        self.assertEqual(sum(oracle.optimal_mask(state)), len(actions))
        for action in actions:
            self.assertEqual(self.forward_distance(layout, step(layout, state, action)[0]),
                             self.forward_distance(layout, state) - 1)

    def test_symmetries_share_layout_group_even_with_other_start(self):
        layout, _ = parse_ascii("######\n## . #\n# $  #\n# @  #\n#    #\n######")
        board = render_ascii(layout, State(3 * 6 + 2, (2 * 6 + 2,)))
        rows = board.splitlines()
        rotated = "\n".join("".join(row) for row in zip(*rows[::-1]))
        reflected = "\n".join(row[::-1] for row in rows)
        self.assertEqual(canonical_layout(layout), canonical_layout(parse_ascii(rotated)[0]))
        self.assertEqual(canonical_layout(layout), canonical_layout(parse_ascii(reflected)[0]))

    def test_deterministic_generation_and_no_split_overlap(self):
        options = dict(seed=7, train_layouts=4, dev_layouts=2, test_layouts=2,
                       ood_layouts=2, train_starts=8, eval_starts=3)
        dataset = make_dataset(**options)
        self.assertEqual(dataset, make_dataset(**options))
        self.assertLessEqual(len(dataset["train"]), 32)
        self.assertGreaterEqual(len(dataset["train"]), 12)
        self.assertEqual({name: len(rows) for name, rows in dataset.items() if name != "train"},
                         {"dev": 6, "test": 6, "ood": 6})
        assert_disjoint_layouts(dataset)
        for split, examples in dataset.items():
            for example in examples:
                self.assertGreaterEqual(example.distance, 1 if split == "train" else 4)
                self.assertLessEqual(example.distance, 30)
                self.assertEqual(len(example.state.boxes), 2 if split == "ood" else 1)
                self.assertEqual(example.distance,
                                 self.forward_distance(example.layout, example.state))
                self.assertTrue(example.optimal_actions)
                self.assertEqual(example_from_dict(example_to_dict(example)), example)
        self.assertTrue(any(example.distance == 1 for example in dataset["train"]))
        leaked = {**dataset, "test": [replace(dataset["train"][0], split="test")]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint_layouts(leaked)


if __name__ == "__main__":
    unittest.main()
