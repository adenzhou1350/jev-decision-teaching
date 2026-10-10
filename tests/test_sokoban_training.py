"""Check geometric augmentation and that closed-loop policy has no oracle."""

from dataclasses import replace
from unittest.mock import patch
import unittest

import torch

from jev_teaching.sokoban_env import ExactOracle, Example, Layout, State, encode_planes, layout_key, parse_ascii
from jev_teaching.sokoban_train import CandidatePolicy, augment, legal_mask, rollout


class FixedAugmentation:
    def __init__(self, rotations, reflect):
        self.rotations, self.reflect = rotations, reflect

    def randrange(self, _):
        return self.rotations

    def random(self):
        return 0.0 if self.reflect else 1.0


def from_planes(planes):
    cells = [tuple(torch.nonzero(plane.flatten(), as_tuple=True)[0].tolist())
             for plane in planes]
    return Layout(planes.shape[-1], planes.shape[-2], frozenset(cells[0]), cells[1]), \
        State(cells[3][0], cells[2])


class SokobanTrainingTest(unittest.TestCase):
    def test_all_eight_augmentations_preserve_exact_actions(self):
        layout, state = parse_ascii("######\n##   #\n# $. #\n# @  #\n#    #\n######")
        optimal = ExactOracle(layout).optimal_actions(state)
        self.assertTrue(optimal)
        x = torch.tensor([encode_planes(layout, state)])
        legal = torch.tensor([legal_mask(layout, state)])
        labels = torch.tensor([[float(a in optimal) for a in range(4)]])
        labels /= labels.sum(-1, keepdim=True)
        for rotations in range(4):
            for reflect in (False, True):
                xx, ll, yy = augment(x, legal, labels, FixedAugmentation(rotations, reflect))
                transformed_layout, transformed_state = from_planes(xx[0])
                expected = ExactOracle(transformed_layout).optimal_actions(transformed_state)
                self.assertEqual(ll[0].tolist(), legal_mask(transformed_layout, transformed_state))
                self.assertEqual(tuple(torch.nonzero(yy[0], as_tuple=True)[0].tolist()), expected)
                self.assertEqual(layout_key(layout), layout_key(transformed_layout))

    def test_shared_scorer_is_candidate_order_equivariant(self):
        torch.manual_seed(123)
        model = CandidatePolicy().eval()
        layout, state = parse_ascii("######\n#    #\n# $. #\n# @  #\n#    #\n######")
        x = torch.tensor([encode_planes(layout, state)])
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            before = model(x)
            model.candidates.copy_(model.candidates[permutation])
            after = model(x)
        torch.testing.assert_close(after, before[:, permutation])

    def test_rollout_uses_no_oracle_or_ground_truth_label(self):
        layout, state = parse_ascii("######\n#    #\n# $. #\n# @  #\n#    #\n######")
        oracle = ExactOracle(layout)
        example = Example("example", "test", layout_key(layout), layout, state,
                          oracle.distances[state], oracle.optimal_actions(state))
        altered = replace(example, distance=987, optimal_actions=(3,))
        model = CandidatePolicy().eval()
        # Labels may be used for final metric reporting; never to choose moves.
        with patch.object(ExactOracle, "__init__", side_effect=AssertionError("Oracle called online")):
            original = rollout(model, [example], max_steps=12)
            modified = rollout(model, [altered], max_steps=12)
        for field in ("actions", "outcome", "steps", "final_player", "final_boxes"):
            self.assertEqual(original["trajectories"][0][field], modified["trajectories"][0][field])


if __name__ == "__main__":
    unittest.main()
