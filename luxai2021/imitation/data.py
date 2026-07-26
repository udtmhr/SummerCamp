from __future__ import annotations

# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, S311
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from luxai2021.imitation.actions import (
    ACTION_SCHEMA,
    DIRECTIONS,
    RESOURCES,
    TARGET_NAMES,
)
from luxai2021.imitation.masking import (
    LEGAL_MASK_SUFFIX,
    build_legal_masks,
    sanitize_targets,
)
from luxai2021.imitation.schema import (
    BOARD_SIZE,
    FEATURE_INDEX,
    BoardSnapshot,
    UnitSnapshot,
    encode_snapshot,
    snapshot_from_updates,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

IGNORE_INDEX = -100
MAX_ENTITIES = BOARD_SIZE * BOARD_SIZE
_MINIMUM_SPLIT_GAMES = 3
_TEAM_COUNT = 2
_TAIL_BYTES = 128 * 1024
_STEP_PATTERN = re.compile(rb'"step"\s*:\s*(\d+)')
_DIRECTION_TO_DELTA = {"n": (-1, 0), "e": (0, 1), "s": (1, 0), "w": (0, -1)}
_DELTA_TO_DIRECTION = {value: key for key, value in _DIRECTION_TO_DELTA.items()}


@dataclass(frozen=True)
class SourceInfo:
    source_id: int
    lb: float
    metadata_path: Path


@dataclass(frozen=True)
class ReplayMetadata:
    turn_count: int
    winner: int | None
    source_id: int | None = None
    source_teams: tuple[int, ...] = ()


def discover_replays(replay_dir: str) -> list[Path]:
    root = Path(replay_dir)
    files = [root] if root.is_file() else list(root.rglob("*.json"))
    result = [path for path in sorted(files) if not path.stem.endswith("_info")]
    if not result:
        msg = f"No Kaggle replay JSON files found under {replay_dir}"
        raise ValueError(msg)
    return result


def _find_source_info(path: Path) -> SourceInfo:
    for parent in (path.parent, *path.parents):
        metadata_path = parent / "agent_info.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        try:
            source_id = int(metadata["agent_id"])
            lb = float(metadata["lb"])
        except (KeyError, TypeError, ValueError) as error:
            msg = f"Invalid source metadata in {metadata_path}"
            raise ValueError(msg) from error
        return SourceInfo(source_id=source_id, lb=lb, metadata_path=metadata_path)
    msg = f"No ancestor agent_info.json found for source replay {path}"
    raise ValueError(msg)


def discover_sources(replay_paths: Sequence[Path]) -> tuple[SourceInfo, ...]:
    sources: dict[int, SourceInfo] = {}
    for replay_path in replay_paths:
        source = _find_source_info(Path(replay_path))
        previous = sources.get(source.source_id)
        if previous is not None and (previous.lb != source.lb or previous.metadata_path != source.metadata_path):
            msg = f"Conflicting metadata for source {source.source_id}"
            raise ValueError(msg)
        sources[source.source_id] = source
    return tuple(sources[source_id] for source_id in sorted(sources))


def limit_replays_per_source(
    replay_paths: Sequence[Path],
    maximum: int,
) -> list[Path]:
    if maximum <= 0:
        return list(replay_paths)
    selected: dict[int, list[Path]] = {}
    for replay_path in sorted(Path(path) for path in replay_paths):
        source_id = _find_source_info(replay_path).source_id
        bucket = selected.setdefault(source_id, [])
        if len(bucket) < maximum:
            bucket.append(replay_path)
    return [path for source_id in sorted(selected) for path in selected[source_id]]


def split_replays(
    replay_paths: Sequence[Path],
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[Path]]:
    paths = list(replay_paths)
    random.Random(seed).shuffle(paths)
    if len(paths) < _MINIMUM_SPLIT_GAMES:
        return {"train": paths, "validation": paths[-1:], "test": paths[-1:]}
    validation_count = max(1, round(len(paths) * validation_fraction))
    test_count = max(1, round(len(paths) * test_fraction))
    if validation_count + test_count >= len(paths):
        validation_count = test_count = 1
    return {
        "train": paths[validation_count + test_count :],
        "validation": paths[:validation_count],
        "test": paths[validation_count : validation_count + test_count],
    }


class _ReplayCache:
    def __init__(self, max_size: int = 2) -> None:
        self.max_size = max_size
        self.data: OrderedDict[str, Mapping[str, object]] = OrderedDict()

    def get(self, path: Path) -> Mapping[str, object]:
        key = str(path)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        with path.open(encoding="utf-8") as replay_file:
            replay = json.load(replay_file)
        self.data[key] = replay
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)
        return replay


def _winner_from_rewards(rewards: Sequence[float | None]) -> int | None:
    if len(rewards) != _TEAM_COUNT or any(reward is None for reward in rewards) or rewards[0] == rewards[1]:
        return None
    return 0 if rewards[0] > rewards[1] else 1


def _replay_turn_count(path: Path) -> int | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as replay_file:
            replay_file.seek(max(0, size - _TAIL_BYTES))
            final_chunk = replay_file.read()
    except OSError:
        return None
    steps = _STEP_PATTERN.findall(final_chunk)
    return max(int(step) for step in steps) if steps else None


def _replay_metadata(
    path: Path,
    cache: _ReplayCache,
    *,
    include_source: bool = False,
) -> ReplayMetadata:
    info_path = path.with_name(f"{path.stem}_info.json")
    source = _find_source_info(path) if include_source else None
    if info_path.exists():
        with info_path.open(encoding="utf-8") as info_file:
            info = json.load(info_file)
        agents = sorted(info.get("agents", ()), key=lambda agent: agent["index"])
        rewards = [agent["reward"] for agent in agents]
        turn_count = _replay_turn_count(path)
        if len(rewards) == _TEAM_COUNT and turn_count is not None:
            source_teams = ()
            if source is not None:
                matches = [int(agent["index"]) for agent in agents if int(agent["submissionId"]) == source.source_id]
                if not matches:
                    msg = f"Expected at least one agent with submissionId={source.source_id} in {info_path}"
                    raise ValueError(msg)
                if len(set(matches)) != len(matches):
                    msg = f"Duplicate team indices for submissionId={source.source_id} in {info_path}"
                    raise ValueError(msg)
                source_teams = tuple(matches)
            return ReplayMetadata(
                turn_count=turn_count,
                winner=_winner_from_rewards(rewards),
                source_id=None if source is None else source.source_id,
                source_teams=source_teams,
            )
    if source is not None:
        msg = f"Source replay requires valid per-game metadata: {info_path}"
        raise ValueError(msg)
    replay = cache.get(path)
    return ReplayMetadata(
        turn_count=len(replay["steps"]) - 1,
        winner=_winner_from_rewards(replay.get("rewards") or ()),
    )


def _empty_targets() -> dict[str, np.ndarray]:
    targets = {name: np.full(MAX_ENTITIES, IGNORE_INDEX, dtype=np.int64) for name in TARGET_NAMES}
    targets.update(
        {f"{entity}_positions": np.full((MAX_ENTITIES, 2), -1, dtype=np.int64) for entity in ("worker", "cart", "city")}
    )
    return targets


def _direction_between(source: UnitSnapshot, destination: UnitSnapshot) -> str:
    delta = (destination.y - source.y, destination.x - source.x)
    if delta not in _DELTA_TO_DIRECTION:
        msg = f"Transfer units are not adjacent: {source.unit_id} -> {destination.unit_id}"
        raise ValueError(msg)
    return _DELTA_TO_DIRECTION[delta]


def build_targets(
    snapshot: BoardSnapshot,
    team: int,
    actions: Iterable[str],
) -> dict[str, np.ndarray]:
    targets = _empty_targets()
    unit_indices: dict[str, tuple[str, int]] = {}
    entity_counts = {"worker": 0, "cart": 0, "city": 0}
    for unit in snapshot.units.values():
        if unit.team != team or not unit.can_act:
            continue
        x, y = snapshot.padded_position(unit.x, unit.y)
        prefix = "worker" if unit.unit_type == 0 else "cart"
        entity_index = entity_counts[prefix]
        entity_counts[prefix] += 1
        unit_indices[unit.unit_id] = (prefix, entity_index)
        targets[f"{prefix}_positions"][entity_index] = (y, x)
        targets[f"{prefix}_type"][entity_index] = 0
    city_indices = {}
    for tile in snapshot.city_tiles:
        if tile.team == team and tile.can_act:
            x, y = snapshot.padded_position(tile.x, tile.y)
            entity_index = entity_counts["city"]
            entity_counts["city"] += 1
            city_indices[(tile.x, tile.y)] = entity_index
            targets["city_positions"][entity_index] = (y, x)
            targets["city"][entity_index] = 0

    for command in actions:
        parts = command.split()
        action = parts[0]
        if action in {"m", "bcity", "t", "p"}:
            source_id = parts[1]
            source = snapshot.units.get(source_id)
            if source is None or source.team != team or source_id not in unit_indices:
                continue
            prefix, entity_index = unit_indices[source_id]
            if action == "m":
                direction = parts[2]
                if direction == "c":
                    targets[f"{prefix}_type"][entity_index] = 0
                else:
                    targets[f"{prefix}_type"][entity_index] = 1
                    targets[f"{prefix}_move"][entity_index] = DIRECTIONS.index(direction)
            elif action == "bcity" and prefix == "worker":
                targets["worker_type"][entity_index] = 2
            elif action == "p" and prefix == "cart":
                targets["cart_type"][entity_index] = 2
            elif action == "t":
                destination = snapshot.units.get(parts[2])
                if destination is None:
                    continue
                targets[f"{prefix}_type"][entity_index] = 3
                targets[f"{prefix}_transfer_dir"][entity_index] = DIRECTIONS.index(
                    _direction_between(source, destination)
                )
                targets[f"{prefix}_resource"][entity_index] = RESOURCES.index(parts[3])
        elif action in {"bw", "bc", "r"}:
            raw_x, raw_y = int(parts[1]), int(parts[2])
            entity_index = city_indices.get((raw_x, raw_y))
            if entity_index is not None:
                targets["city"][entity_index] = {"bw": 1, "bc": 2, "r": 3}[action]
    legal_masks = build_legal_masks(snapshot, team)
    sanitize_targets(targets, legal_masks)
    targets.update(legal_masks)
    return targets


def _direction_remap(rotations: int, *, horizontal_flip: bool) -> dict[int, int]:
    remap = {}
    for index, direction in enumerate(DIRECTIONS):
        dy, dx = _DIRECTION_TO_DELTA[direction]
        for _ in range(rotations):
            dy, dx = -dx, dy
        if horizontal_flip:
            dx = -dx
        remap[index] = DIRECTIONS.index(_DELTA_TO_DIRECTION[(dy, dx)])
    return remap


def augment_sample(
    observation: np.ndarray,
    targets: dict[str, np.ndarray],
    rotations: int,
    *,
    horizontal_flip: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observation = np.rot90(observation, rotations, axes=(-2, -1)).copy()
    transformed = {name: target.copy() for name, target in targets.items()}
    if horizontal_flip:
        observation = observation[:, :, ::-1].copy()
    for entity in ("worker", "cart", "city"):
        positions = transformed[f"{entity}_positions"]
        valid_positions = positions[:, 0] >= 0
        for _ in range(rotations):
            old_y = positions[valid_positions, 0].copy()
            old_x = positions[valid_positions, 1].copy()
            positions[valid_positions, 0] = BOARD_SIZE - 1 - old_x
            positions[valid_positions, 1] = old_y
        if horizontal_flip:
            positions[valid_positions, 1] = BOARD_SIZE - 1 - positions[valid_positions, 1]
    remap = _direction_remap(rotations, horizontal_flip=horizontal_flip)
    for name in ("worker_move", "worker_transfer_dir", "cart_move", "cart_transfer_dir"):
        target = transformed[name]
        valid = target != IGNORE_INDEX
        values = target[valid].copy()
        target[valid] = np.asarray([remap[int(value)] for value in values], dtype=np.int64)
        mask_name = f"{name}{LEGAL_MASK_SUFFIX}"
        original_mask = transformed[mask_name].copy()
        transformed[mask_name][...] = False
        for old_index, new_index in remap.items():
            transformed[mask_name][:, new_index] = original_mask[:, old_index]

    mask = observation[FEATURE_INDEX["board_mask"]]
    ys, xs = np.nonzero(mask)
    observation[FEATURE_INDEX["x_coordinate"]] = 0
    observation[FEATURE_INDEX["y_coordinate"]] = 0
    if len(xs):
        x_values = np.linspace(-1, 1, xs.max() - xs.min() + 1, dtype=np.float32)
        y_values = np.linspace(-1, 1, ys.max() - ys.min() + 1, dtype=np.float32)
        observation[FEATURE_INDEX["x_coordinate"], ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = x_values[None, :]
        observation[FEATURE_INDEX["y_coordinate"], ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = y_values[:, None]
    return observation, transformed


class LuxReplayDataset(Dataset):
    def __init__(
        self,
        replay_paths: Sequence[Path],
        *,
        augment: bool = False,
        team_selection: str = "winner",
        winner_weight: float = 1.5,
        seed: int = 42,
        max_turns: int = 0,
        source_ids: Sequence[int] = (),
    ) -> None:
        self.replay_paths = [Path(path) for path in replay_paths]
        self.augment = augment
        if team_selection not in {"winner", "all", "source"}:
            msg = f"Unsupported team selection: {team_selection}"
            raise ValueError(msg)
        self.team_selection = team_selection
        self.source_ids = tuple(int(source_id) for source_id in source_ids)
        self.source_id_to_index = {source_id: index for index, source_id in enumerate(self.source_ids)}
        if self.team_selection == "source" and not self.source_ids:
            msg = "source_ids must be provided when team_selection='source'"
            raise ValueError(msg)
        self.winner_weight = winner_weight
        self.seed = seed
        self.cache = _ReplayCache()
        self.samples: list[tuple[Path, int, int, int]] = []
        self.sample_groups: list[list[int]] = []
        for path in self.replay_paths:
            metadata = _replay_metadata(path, self.cache, include_source=self.team_selection == "source")
            turn_count = metadata.turn_count
            if max_turns > 0:
                turn_count = min(turn_count, max_turns)
            teams = (metadata.winner,) if self.team_selection == "winner" and metadata.winner is not None else ()
            source_index = -1
            if self.team_selection == "all":
                teams = (0, 1)
            elif self.team_selection == "source":
                if metadata.source_id not in self.source_id_to_index:
                    msg = f"Source {metadata.source_id} is not present in source_ids"
                    raise ValueError(msg)
                teams = metadata.source_teams
                source_index = self.source_id_to_index[metadata.source_id]
            group_start = len(self.samples)
            self.samples.extend((path, turn, team, source_index) for turn in range(turn_count) for team in teams)
            if len(self.samples) > group_start:
                self.sample_groups.append(list(range(group_start, len(self.samples))))

    def __len__(self) -> int:
        return len(self.samples)

    def _snapshot_actions_team(
        self,
        index: int,
    ) -> tuple[BoardSnapshot, list[str], int, Mapping[str, object]]:
        path, turn, team, _ = self.samples[index]
        replay = self.cache.get(path)
        observation = replay["steps"][turn][0]["observation"]
        initial_observation = replay["steps"][0][0]["observation"]
        width = int(observation.get("width", initial_observation["width"]))
        height = int(observation.get("height", initial_observation["height"]))
        snapshot = snapshot_from_updates(observation["updates"], width, height, turn)
        actions = replay["steps"][turn + 1][team].get("action") or []
        return snapshot, actions, team, replay

    def targets_for_index(self, index: int) -> dict[str, np.ndarray]:
        snapshot, actions, team, _ = self._snapshot_actions_team(index)
        return build_targets(snapshot, team, actions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        snapshot, actions, team, replay = self._snapshot_actions_team(index)
        features = encode_snapshot(snapshot, team)
        targets = build_targets(snapshot, team, actions)
        if self.augment:
            rng = random.Random(self.seed + index + random.randint(0, 2**16))
            features, targets = augment_sample(
                features,
                targets,
                rng.randrange(4),
                horizontal_flip=bool(rng.randrange(2)),
            )

        winner = None
        if self.team_selection == "all":
            winner = _winner_from_rewards(replay.get("rewards") or ())
        result = {
            "observation": torch.from_numpy(features),
            "source_index": torch.tensor(self.samples[index][3], dtype=torch.long),
            "sample_weight": torch.tensor(
                self.winner_weight if self.team_selection == "all" and team == winner else 1.0,
                dtype=torch.float32,
            ),
        }
        result.update({name: torch.from_numpy(target) for name, target in targets.items()})
        return result


class ReplayBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: LuxReplayDataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.groups = dataset.sample_groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        group_order = list(range(len(self.groups)))
        if self.shuffle:
            rng.shuffle(group_order)
        for group_index in group_order:
            indices = self.groups[group_index].copy()
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


_CLASS_COUNT_OFFSETS = {}
_CLASS_COUNT_TOTAL = 0
for _target_name in TARGET_NAMES:
    _next_offset = _CLASS_COUNT_TOTAL + len(ACTION_SCHEMA[_target_name])
    _CLASS_COUNT_OFFSETS[_target_name] = slice(_CLASS_COUNT_TOTAL, _next_offset)
    _CLASS_COUNT_TOTAL = _next_offset


class _ClassCountDataset(Dataset):
    def __init__(self, dataset: LuxReplayDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        targets = self.dataset.targets_for_index(index)
        sample_counts = np.zeros(_CLASS_COUNT_TOTAL, dtype=np.int64)
        for name in TARGET_NAMES:
            values = targets[name]
            valid = values != IGNORE_INDEX
            if np.any(valid):
                sample_counts[_CLASS_COUNT_OFFSETS[name]] = np.bincount(
                    values[valid],
                    minlength=len(ACTION_SCHEMA[name]),
                )
        return torch.from_numpy(sample_counts)


def class_counts(
    dataset: LuxReplayDataset,
    *,
    show_progress: bool = False,
    num_workers: int = 0,
    prefetch_factor: int = 2,
) -> dict[str, torch.Tensor]:
    loader_options = {"num_workers": num_workers}
    if num_workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    loader = DataLoader(_ClassCountDataset(dataset), batch_size=256, **loader_options)
    batches = tqdm(
        loader,
        desc="Class statistics",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    counts = torch.zeros(_CLASS_COUNT_TOTAL, dtype=torch.int64)
    for batch in batches:
        counts += batch.sum(dim=0)
    return {name: counts[index_range].clone() for name, index_range in _CLASS_COUNT_OFFSETS.items()}
