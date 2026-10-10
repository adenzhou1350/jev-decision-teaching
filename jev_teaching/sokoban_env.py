"""Small, exact Sokoban environment and offline supervised-data generator.

The policy only needs ``step`` and ``encode_planes``. ``ExactOracle`` is a
separate, exhaustive offline labeler; it is never called by game transitions.
Actions are up, right, down, left. A blocked move leaves the state unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import comb
import random


ACTIONS = ("up", "right", "down", "left")
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class State:
    player: int
    boxes: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "boxes", tuple(sorted(self.boxes)))
        if len(set(self.boxes)) != len(self.boxes):
            raise ValueError("Boxes must occupy distinct cells")
        if self.player in self.boxes:
            raise ValueError("Player cannot occupy a box cell")


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    walls: frozenset[int]
    goals: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "walls", frozenset(self.walls))
        object.__setattr__(self, "goals", tuple(sorted(self.goals)))
        if self.width < 3 or self.height < 3:
            raise ValueError("Board must be at least 3 by 3")
        cells = set(range(self.width * self.height))
        if not self.walls <= cells or not set(self.goals) <= cells:
            raise ValueError("Layout contains out-of-bounds cells")
        if not self.goals or len(set(self.goals)) != len(self.goals):
            raise ValueError("At least one distinct goal is required")
        if set(self.goals) & self.walls:
            raise ValueError("Goals cannot be walls")

    @property
    def floors(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.width * self.height) if i not in self.walls)


def validate_state(layout: Layout, state: State) -> None:
    floor = set(layout.floors)
    if state.player not in floor or not set(state.boxes) <= floor:
        raise ValueError("Player and boxes must occupy floor cells")
    if len(state.boxes) != len(layout.goals):
        raise ValueError("The number of boxes must match the number of goals")


def _neighbor(layout: Layout, cell: int, action: int) -> int | None:
    dx, dy = DELTAS[action]
    x, y = cell % layout.width + dx, cell // layout.width + dy
    if 0 <= x < layout.width and 0 <= y < layout.height:
        return y * layout.width + x
    return None


def step(layout: Layout, state: State, action: int) -> tuple[State, bool]:
    """Apply one move/push without an oracle; bool indicates a legal move.

    Walking/pushing each costs one environment step. Pushing two adjacent boxes
    is illegal. Out-of-bounds cells behave as walls, even on an unbordered map.
    """
    if action not in range(4):
        raise ValueError("Action must be an integer in [0, 3]")
    target = _neighbor(layout, state.player, action)
    if target is None or target in layout.walls:
        return state, False
    boxes = state.boxes
    if target in boxes:
        beyond = _neighbor(layout, target, action)
        if beyond is None or beyond in layout.walls or beyond in boxes:
            return state, False
        boxes = tuple(beyond if box == target else box for box in boxes)
    return State(target, boxes), True


def is_solved(layout: Layout, state: State) -> bool:
    return state.boxes == layout.goals


def legal_actions(layout: Layout, state: State) -> tuple[int, ...]:
    """Rule-only legality mask; it does not detect future deadlocks."""
    return tuple(action for action in range(4) if step(layout, state, action)[1])


def encode_planes(layout: Layout, state: State) -> list[list[list[float]]]:
    """Four binary planes: walls, goals, boxes, player; contains no labels."""
    validate_state(layout, state)
    occupied = (layout.walls, set(layout.goals), set(state.boxes), {state.player})
    return [
        [[float(y * layout.width + x in cells) for x in range(layout.width)]
         for y in range(layout.height)]
        for cells in occupied
    ]


def render_ascii(layout: Layout, state: State) -> str:
    validate_state(layout, state)
    chars = []
    for cell in range(layout.width * layout.height):
        if cell in layout.walls:
            char = "#"
        elif cell == state.player:
            char = "+" if cell in layout.goals else "@"
        elif cell in state.boxes:
            char = "*" if cell in layout.goals else "$"
        else:
            char = "." if cell in layout.goals else " "
        chars.append(char)
    return "\n".join("".join(chars[y * layout.width:(y + 1) * layout.width])
                     for y in range(layout.height))


def parse_ascii(board: str) -> tuple[Layout, State]:
    """Read # wall, . goal, $ box, @ player, * box/goal, + player/goal."""
    rows = board.strip("\n\r").splitlines()
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("Board must be a nonempty rectangle (preserve spaces)")
    width = len(rows[0])
    walls, goals, boxes, players = set(), [], [], []
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char not in "# .$@*+":
                raise ValueError(f"Unrecognized board character: {char!r}")
            cell = y * width + x
            if char == "#":
                walls.add(cell)
            if char in ".*+":
                goals.append(cell)
            if char in "$*":
                boxes.append(cell)
            if char in "@+":
                players.append(cell)
    if len(players) != 1:
        raise ValueError("Board must contain exactly one player")
    layout = Layout(width, len(rows), frozenset(walls), tuple(goals))
    state = State(players[0], tuple(boxes))
    validate_state(layout, state)
    return layout, state


def canonical_layout(layout: Layout) -> str:
    """Exact wall+goal signature modulo all eight square-grid symmetries.

    Player and box starts deliberately do not affect the group. Rectangular
    layouts are supported too: rotations also exchange width and height.
    """
    grid = tuple(tuple("#" if y * layout.width + x in layout.walls else
                       "." if y * layout.width + x in layout.goals else " "
                       for x in range(layout.width))
                 for y in range(layout.height))
    variants = []
    for _ in range(4):
        for candidate in (grid, tuple(tuple(reversed(row)) for row in grid)):
            variants.append(f"{len(candidate[0])}x{len(candidate)}:" +
                            "/".join("".join(row) for row in candidate))
        grid = tuple(tuple(row) for row in zip(*grid[::-1]))
    return min(variants)


def layout_key(layout: Layout) -> str:
    return sha256(canonical_layout(layout).encode("ascii")).hexdigest()


class ExactOracle:
    """All solvable states and exact *move* distances for a fixed layout.

    Enumerates every legal player/box assignment, reverses legal forward edges,
    and runs multi-source BFS from all solved assignments. A missing state is
    provably unsolvable within this finite board, not merely search-truncated.
    This is intentionally offline and bounded to small teaching boards.
    """

    def __init__(self, layout: Layout, max_states: int = 250_000):
        floors = layout.floors
        count = len(layout.goals)
        total = comb(len(floors), count) * (len(floors) - count)
        if total > max_states:
            raise ValueError(f"Oracle would enumerate {total} states; cap={max_states}")
        self.layout = layout
        self.enumerated_states = total
        reverse: dict[State, list[State]] = {}
        solved = []
        for boxes in combinations(floors, count):
            for player in floors:
                if player in boxes:
                    continue
                source = State(player, boxes)
                if is_solved(layout, source):
                    solved.append(source)
                for action in range(4):
                    target, moved = step(layout, source, action)
                    if moved:
                        reverse.setdefault(target, []).append(source)
        self.distances = {state: 0 for state in solved}
        queue = deque(solved)
        while queue:
            target = queue.popleft()
            for source in reverse.get(target, ()):
                if source not in self.distances:
                    self.distances[source] = self.distances[target] + 1
                    queue.append(source)

    def optimal_actions(self, state: State) -> tuple[int, ...]:
        distance = self.distances.get(state)
        if distance is None or distance == 0:
            return ()
        return tuple(action for action in range(4)
                     if self.distances.get(step(self.layout, state, action)[0])
                     == distance - 1)

    def optimal_mask(self, state: State) -> tuple[bool, bool, bool, bool]:
        actions = self.optimal_actions(state)
        return tuple(action in actions for action in range(4))


@dataclass(frozen=True)
class Example:
    level_id: str
    split: str
    layout_key: str
    layout: Layout
    state: State
    distance: int
    optimal_actions: tuple[int, ...]


def example_to_dict(example: Example) -> dict:
    """JSON-safe artifact record. Keep labels out of policy observations."""
    return {
        "level_id": example.level_id, "split": example.split,
        "layout_key": example.layout_key,
        "layout": {"width": example.layout.width, "height": example.layout.height,
                   "walls": sorted(example.layout.walls),
                   "goals": list(example.layout.goals)},
        "state": {"player": example.state.player, "boxes": list(example.state.boxes)},
        "distance": example.distance, "optimal_actions": list(example.optimal_actions),
        "board": render_ascii(example.layout, example.state),
    }


def example_from_dict(record: dict) -> Example:
    layout = Layout(**record["layout"])
    state = State(**record["state"])
    validate_state(layout, state)
    if layout_key(layout) != record["layout_key"]:
        raise ValueError("Serialized layout key does not match the board")
    return Example(record["level_id"], record["split"], record["layout_key"],
                   layout, state, record["distance"], tuple(record["optimal_actions"]))


def _connected(layout: Layout) -> bool:
    floors = set(layout.floors)
    reached = {next(iter(floors))}
    queue = deque(reached)
    while queue:
        cell = queue.popleft()
        for action in range(4):
            neighbor = _neighbor(layout, cell, action)
            if neighbor in floors and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached == floors


def make_dataset(
    seed: int = 20261009,
    train_layouts: int = 64,
    dev_layouts: int = 16,
    test_layouts: int = 24,
    ood_layouts: int = 24,
    train_starts: int = 24,
    eval_starts: int = 8,
    min_distance: int = 4,
    max_distance: int = 30,
    train_min_distance: int = 1,
    max_attempts_per_split: int = 10_000,
) -> dict[str, list[Example]]:
    """Deterministic 6x6 boards, grouped by wall+goal layout before sampling.

    Main splits use one box, with 0--3 random interior walls. The ``ood`` split
    has two boxes/goals and 0--2 interior walls. Every selected start is solvable
    and unsolved. Held-out starts have min_distance <= distance <= max_distance;
    training starts use train_min_distance (default 1), retaining supervision
    for the final moves that a rollout must make to reach a goal.
    train_starts is a maximum per layout (all eligible states if fewer); eval
    starts are exactly eval_starts per layout. Multiple starts from one
    layout stay together; no rotated/reflected wall+goal layout crosses splits.
    Held-out labels are provided for evaluation, never as policy observations.

    This procedural distribution is deliberately small. It is NOT Boxoban, a
    visual benchmark, or evidence of general browser/robot planning ability.
    """
    counts = (train_layouts, dev_layouts, test_layouts, ood_layouts)
    if any(count < 0 for count in counts):
        raise ValueError("Layout counts must be nonnegative")
    if train_starts < 1 or eval_starts < 1 or min_distance < 1:
        raise ValueError("Start counts and min_distance must be positive")
    if max_distance < min_distance:
        raise ValueError("max_distance must be at least min_distance")
    if not 1 <= train_min_distance <= max_distance:
        raise ValueError("train_min_distance must be in [1, max_distance]")
    rng = random.Random(seed)
    interior = tuple(y * 6 + x for y in range(1, 5) for x in range(1, 5))
    boundary = frozenset(cell for cell in range(36)
                         if cell % 6 in (0, 5) or cell // 6 in (0, 5))
    used: set[str] = set()
    dataset = {split: [] for split in ("train", "dev", "test", "ood")}
    for split, count in zip(dataset, counts):
        accepted, attempts = 0, 0
        starts = train_starts if split == "train" else eval_starts
        boxes = 2 if split == "ood" else 1
        while accepted < count:
            attempts += 1
            if attempts > max_attempts_per_split:
                raise RuntimeError(f"Could not generate {count} {split} layouts in "
                                   f"{max_attempts_per_split} attempts")
            wall_count = rng.randint(0, 2 if split == "ood" else 3)
            walls = boundary | frozenset(rng.sample(interior, wall_count))
            floor = tuple(cell for cell in interior if cell not in walls)
            layout = Layout(6, 6, walls, tuple(rng.sample(floor, boxes)))
            key = layout_key(layout)
            if key in used or not _connected(layout):
                continue
            oracle = ExactOracle(layout)
            lower_distance = train_min_distance if split == "train" else min_distance
            eligible = sorted((state for state, distance in oracle.distances.items()
                               if lower_distance <= distance <= max_distance),
                              key=lambda state: (state.boxes, state.player))
            minimum = min(starts, eval_starts) if split == "train" else starts
            if len(eligible) < minimum:
                continue
            used.add(key)
            for index, state in enumerate(rng.sample(eligible, min(starts, len(eligible)))):
                dataset[split].append(Example(
                    f"{split}-{accepted:03d}-{index:03d}", split, key, layout,
                    state, oracle.distances[state], oracle.optimal_actions(state)))
            accepted += 1
    assert_disjoint_layouts(dataset)
    return dataset


def assert_disjoint_layouts(dataset: dict[str, list[Example]]) -> None:
    """Fail loudly if a canonical wall+goal layout appears in two splits."""
    owner: dict[str, str] = {}
    for split, examples in dataset.items():
        for example in examples:
            actual = layout_key(example.layout)
            if actual != example.layout_key:
                raise ValueError("Example layout key does not match its board")
            previous = owner.setdefault(actual, split)
            if previous != split:
                raise ValueError(f"Layout overlap between {previous} and {split}")


if __name__ == "__main__":
    generated = make_dataset()
    for name, examples in generated.items():
        print(name, "layouts=", len({example.layout_key for example in examples}),
              "starts=", len(examples), "distance_range=",
              (min(example.distance for example in examples),
               max(example.distance for example in examples)))
    sample = generated["test"][0]
    print(render_ascii(sample.layout, sample.state))
    print("Exact distance:", sample.distance,
          "optimal first moves:", [ACTIONS[action] for action in sample.optimal_actions])
