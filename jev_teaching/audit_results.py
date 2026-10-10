"""Read-only Sokoban artifact audit: replay traces and independently check labels.

Use --run runs/sokoban-first for trainer output, or --artifacts artifacts/sokoban for the
repository's sokoban-*.json / sokoban-demo.pt files. No GPU is required.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .sokoban_env import ACTIONS, ExactOracle, Layout, State, is_solved, layout_key, step, validate_state


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(actual, expected, context):
    if expected is None:
        require(actual is None, f"{context}: expected null, got {actual}")
    else:
        require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"{context}: expected {expected}, got {actual}")


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(levels_path, report_path, checkpoint_path):
    started = time.perf_counter()
    levels = json.loads(levels_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(levels, sort_keys=True).encode()).hexdigest()
    require(report["dataset_sha256"] == digest, "Report dataset SHA256 does not match levels")
    # weights_only avoids arbitrary pickle execution; loading is always on CPU.
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["dataset_sha256"] == digest, "Checkpoint dataset SHA256 does not match levels")
    require(checkpoint["kind"] == "sokoban-shared-scorer-v1", "Unexpected checkpoint kind")
    require(checkpoint["actions"] == list(ACTIONS), "Checkpoint action order does not match environment")
    require("before" in checkpoint and "after" in checkpoint, "Checkpoint missing before/after weights")

    indexed, groups, oracle_cache = {}, {}, {}
    split_sizes, split_layouts = Counter(), {}
    for item in levels:
        name, split = item["id"], item["split"]
        require(name not in indexed, f"Duplicate level id: {name}")
        layout = Layout(item["width"], item["height"], frozenset(item["walls"]), tuple(item["goals"]))
        initial = State(item["player"], tuple(item["boxes"]))
        validate_state(layout, initial)
        require(layout.width == checkpoint["size"] == layout.height, f"{name}: checkpoint/board size mismatch")
        key = layout_key(layout)
        require(item["layout_key"] == key, f"{name}: incorrect canonical layout key")
        owner = groups.setdefault(key, split)
        require(owner == split, f"Canonical layout crosses {owner}/{split}: {key}")
        if layout not in oracle_cache:
            oracle_cache[layout] = ExactOracle(layout)
        exact = oracle_cache[layout].distances.get(initial)
        require(exact is not None and exact > 0, f"{name}: initial state unsolvable or already solved")
        require(exact == item["optimal_steps"], f"{name}: incorrect shortest-path length")
        indexed[name] = (item, layout, initial)
        split_sizes[split] += 1
        split_layouts.setdefault(split, set()).add(key)

    require(set(report["splits"]) == set(split_sizes), "Report and level split names differ")
    summaries, trajectory_count, replayed_steps = {}, 0, 0
    for split, block in report["splits"].items():
        require(block["states"] == split_sizes[split], f"{split}: state count mismatch")
        require(block["layouts"] == len(split_layouts[split]), f"{split}: layout count mismatch")
        expected_ids = {name for name, (item, _, _) in indexed.items() if item["split"] == split}
        summaries[split] = {"levels": split_sizes[split], "layouts": len(split_layouts[split])}
        if split == "train":
            continue
        summaries[split]["policies"] = {}
        for policy in ("before", "after", "random", "greedy"):
            metrics = block[policy]
            traces = metrics["trajectories"]
            ids = [trace["id"] for trace in traces]
            require(len(ids) == len(set(ids)) and set(ids) == expected_ids,
                    f"{split}/{policy}: missing, repeated, or unexpected trajectories")
            outcomes, solved_steps, solved_excess = Counter(), [], []
            for trace in traces:
                name = trace["id"]
                context = f"{split}/{policy}/{name}"
                item, layout, current = indexed[name]
                require(trace["layout_key"] == item["layout_key"], f"{context}: layout mismatch")
                require(trace["optimal_steps"] == item["optimal_steps"], f"{context}: optimal length mismatch")
                actions = trace["actions"]
                require(trace["steps"] == len(actions), f"{context}: step count mismatch")
                require(0 < len(actions) <= metrics["max_steps"], f"{context}: invalid trajectory length")
                seen, expected_outcome = {current}, None
                for index, action_name in enumerate(actions):
                    require(expected_outcome is None, f"{context}: actions continue after terminal condition")
                    require(action_name in ACTIONS, f"{context}: unknown action {action_name!r}")
                    current, moved = step(layout, current, ACTIONS.index(action_name))
                    if is_solved(layout, current):
                        expected_outcome = "solved"
                    elif not moved:
                        require(not any(step(layout, current, action)[1] for action in range(4)),
                                f"{context}: selected blocked move although legal alternatives exist")
                        expected_outcome = "no_legal_move"
                    elif policy != "random" and current in seen:
                        expected_outcome = "deterministic_loop"
                    seen.add(current)
                if expected_outcome is None:
                    require(len(actions) == metrics["max_steps"], f"{context}: unexplained early termination")
                    expected_outcome = "step_limit"
                require(trace["outcome"] == expected_outcome,
                        f"{context}: expected {expected_outcome}, got {trace['outcome']}")
                require(trace["final_player"] == current.player and tuple(trace["final_boxes"]) == current.boxes,
                        f"{context}: replayed final state differs")
                outcomes[expected_outcome] += 1
                trajectory_count += 1
                replayed_steps += len(actions)
                if expected_outcome == "solved":
                    require(len(actions) >= item["optimal_steps"], f"{context}: solved faster than exact optimum")
                    solved_steps.append(len(actions))
                    solved_excess.append(len(actions) - item["optimal_steps"])
            count, wins = len(traces), outcomes["solved"]
            require(metrics["count"] == count and metrics["solved"] == wins,
                    f"{split}/{policy}: aggregate counts differ")
            equal_number(metrics["success_rate"], wins / count, f"{split}/{policy}: success rate")
            equal_number(metrics["mean_steps_solved"], statistics.mean(solved_steps) if wins else None,
                         f"{split}/{policy}: mean steps")
            equal_number(metrics["mean_excess_steps_solved"], statistics.mean(solved_excess) if wins else None,
                         f"{split}/{policy}: mean excess steps")
            summaries[split]["policies"][policy] = {"count": count, "solved": wins, "outcomes": dict(outcomes)}

    return {
        "status": "passed", "dataset_sha256": digest,
        "levels_checked": len(levels), "canonical_layouts_checked": len(groups),
        "exact_oracle_layouts_checked": len(oracle_cache),
        "trajectories_replayed": trajectory_count, "environment_steps_replayed": replayed_steps,
        "split_layout_overlap": 0, "report_and_checkpoint_dataset_hash_match": True,
        "checks": ["canonical split isolation", "every initial state's exact shortest distance",
                   "all stored trajectory actions, termination and final states",
                   "all rollout aggregate counts, success rates and length metrics",
                   "dataset digest linked to report and before/after checkpoint"],
        "scope": "Artifact consistency and exact environment replay; this audit does not recompute policy logits. See artifacts/live-check.json for served policy replay.",
        "artifact_sha256": {path.name: file_sha256(path) for path in (levels_path, report_path, checkpoint_path)},
        "splits": summaries, "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Trainer directory containing levels.json/report.json/policy.pt")
    source.add_argument("--artifacts", type=Path, help="Directory containing sokoban-* artifacts, e.g. artifacts/sokoban")
    parser.add_argument("--out", type=Path, help="Optional new JSON audit file (existing files are not overwritten)")
    args = parser.parse_args()
    if args.out and args.out.exists():
        parser.error("--out already exists; choose a new audit file")
    if args.run:
        paths = tuple(args.run / name for name in ("levels.json", "report.json", "policy.pt"))
    else:
        paths = tuple(args.artifacts / name for name in ("sokoban-levels.json", "sokoban-results.json", "sokoban-demo.pt"))
    try:
        result = audit(*paths)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from error
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
