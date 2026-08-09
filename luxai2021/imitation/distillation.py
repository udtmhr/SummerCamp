from __future__ import annotations

# ruff: noqa: PLR0913, SLF001
import os
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as nn_functional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from luxai2021.imitation.actions import FIRST_PLACE_ACTION_SCHEMA, first_place_action_remap
from luxai2021.imitation.data import IGNORE_INDEX, MAX_ENTITIES, LuxReplayDataset, _winner_from_rewards
from luxai2021.imitation.first_place import (
    FIRST_PLACE_TEACHER_SHA256,
    build_first_place_targets,
    first_place_city_legal_mask,
    first_place_unit_legal_mask,
)
from luxai2021.imitation.masking import apply_legal_action_mask
from luxai2021.imitation.model import _entity_logits
from luxai2021.imitation.schema import BOARD_SIZE, FEATURE_INDEX, BoardSnapshot, encode_snapshot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DISTILLATION_CACHE_VERSION = 2
DISTILLATION_PREPARED_CACHE_VERSION = 1
_D4_TRANSFORMS = tuple((rotations, horizontal_flip) for rotations in range(4) for horizontal_flip in (False, True))
_D4_ACTION_REMAPS = {
    entity: torch.tensor(
        [
            [
                first_place_action_remap(entity, rotations, horizontal_flip=horizontal_flip)[index]
                for index in range(len(actions))
            ]
            for rotations, horizontal_flip in _D4_TRANSFORMS
        ],
        dtype=torch.long,
    )
    for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items()
}
_BASE_SPATIAL_INDICES = torch.arange(BOARD_SIZE * BOARD_SIZE).reshape(BOARD_SIZE, BOARD_SIZE)
_D4_SPATIAL_SOURCE_INDICES = torch.stack(
    [
        torch.flip(torch.rot90(_BASE_SPATIAL_INDICES, rotations, dims=(-2, -1)), dims=(-1,))
        if horizontal_flip
        else torch.rot90(_BASE_SPATIAL_INDICES, rotations, dims=(-2, -1))
        for rotations, horizontal_flip in _D4_TRANSFORMS
    ]
).reshape(len(_D4_TRANSFORMS), -1)
_D4_POSITION_REMAPS = torch.argsort(_D4_SPATIAL_SOURCE_INDICES, dim=1)
_DEVICE_ACTION_REMAPS: dict[tuple[str, str, int | None], tuple[Tensor, Tensor]] = {}
_DEVICE_SPATIAL_REMAPS: dict[tuple[str, int | None], tuple[Tensor, Tensor]] = {}


def _device_action_remaps(entity: str, device: torch.device) -> tuple[Tensor, Tensor]:
    key = (entity, device.type, device.index)
    if key not in _DEVICE_ACTION_REMAPS:
        remaps = _D4_ACTION_REMAPS[entity].to(device)
        _DEVICE_ACTION_REMAPS[key] = remaps, torch.argsort(remaps, dim=1)
    return _DEVICE_ACTION_REMAPS[key]


def _device_spatial_remaps(device: torch.device) -> tuple[Tensor, Tensor]:
    key = (device.type, device.index)
    if key not in _DEVICE_SPATIAL_REMAPS:
        _DEVICE_SPATIAL_REMAPS[key] = (
            _D4_SPATIAL_SOURCE_INDICES.to(device),
            _D4_POSITION_REMAPS.to(device),
        )
    return _DEVICE_SPATIAL_REMAPS[key]


def distillation_cache_path(replay_path: Path, cache_dir: Path) -> Path:
    key = sha256(str(replay_path.resolve()).encode()).hexdigest()
    return cache_dir / f"{key}.pt"


def prepared_distillation_cache_path(
    replay_path: Path,
    cache_dir: Path,
    observation_dtype: str = "float16",
) -> Path:
    source = replay_fingerprint(replay_path)
    fingerprint = (
        f"{DISTILLATION_PREPARED_CACHE_VERSION}\0{source['path']}\0{source['size']}\0{source['mtime_ns']}"
        f"\0{FIRST_PLACE_TEACHER_SHA256}\0{observation_dtype}"
    )
    return cache_dir / f"{sha256(fingerprint.encode()).hexdigest()}.npz"


def replay_fingerprint(replay_path: Path) -> dict[str, object]:
    source = replay_path.resolve()
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_matches_replay(
    cache: Mapping[str, object],
    replay_path: Path,
    teacher_sha256: str,
    *,
    rot180: bool,
    cache_dtype: str | None = None,
    amp_dtype: str | None = None,
) -> bool:
    matches = (
        cache.get("cache_version") == DISTILLATION_CACHE_VERSION
        and cache.get("source") == replay_fingerprint(replay_path)
        and cache.get("teacher_sha256") == teacher_sha256
        and cache.get("rot180") is rot180
    )
    if cache_dtype is not None:
        matches = matches and cache.get("dtype") == cache_dtype
    if amp_dtype is not None:
        matches = matches and cache.get("amp_dtype") == amp_dtype
    return matches


def save_distillation_cache(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_distillation_cache(path: Path) -> Mapping[str, object]:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_prepared_distillation_cache(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepared_cache_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "cache_version": int(archive["cache_version"]),
            "turn_count": int(archive["turn_count"]),
            "winner": int(archive["winner"]),
            "observation_dtype": str(archive["observation_dtype"]),
        }


def extract_teacher_turn(
    snapshot: BoardSnapshot,
    output: Mapping[str, Tensor],
    batch_index: int,
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Extract both players' actionable logits from dense teacher output."""
    result = []
    for team in (0, 1):
        team_result = {}
        for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items():
            values = []
            if entity in {"worker", "cart"}:
                unit_type = 0 if entity == "worker" else 1
                for unit in snapshot.units.values():
                    if unit.team != team or unit.unit_type != unit_type or not unit.can_act:
                        continue
                    logits = output[entity][batch_index, team, :, unit.x, unit.y].clone()
                    legal = torch.from_numpy(first_place_unit_legal_mask(snapshot, unit))
                    logits[~legal] = torch.finfo(logits.dtype).min
                    values.append(logits.to(dtype))
            else:
                legal = torch.from_numpy(first_place_city_legal_mask(snapshot, team))
                for tile in snapshot.city_tiles:
                    if tile.team != team or not tile.can_act:
                        continue
                    logits = output[entity][batch_index, team, :, tile.x, tile.y].clone()
                    logits[~legal] = torch.finfo(logits.dtype).min
                    values.append(logits.to(dtype))
            team_result[entity] = torch.stack(values) if values else torch.empty((0, len(actions)), dtype=dtype)
        result.append(team_result)
    return result[0], result[1]


class _DistillationCache:
    def __init__(self, cache_dir: Path, max_size: int = 2) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.data: OrderedDict[str, Mapping[str, object]] = OrderedDict()

    def get(self, replay_path: Path) -> Mapping[str, object]:
        key = str(replay_path.resolve())
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        path = distillation_cache_path(replay_path, self.cache_dir)
        if not path.exists():
            message = f"Missing teacher cache for {replay_path}: {path}"
            raise FileNotFoundError(message)
        value = load_distillation_cache(path)
        self.data[key] = value
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)
        return value


class _PreparedDistillationCache:
    def __init__(self, cache_dir: Path, observation_dtype: str, max_size: int = 2) -> None:
        self.cache_dir = Path(cache_dir)
        self.observation_dtype = observation_dtype
        self.max_size = max_size
        self.data: OrderedDict[str, Mapping[str, np.ndarray]] = OrderedDict()

    def path_for(self, replay_path: Path) -> Path:
        return prepared_distillation_cache_path(replay_path, self.cache_dir, self.observation_dtype)

    def get(self, replay_path: Path) -> Mapping[str, np.ndarray]:
        key = str(replay_path.resolve())
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        path = self.path_for(replay_path)
        if not path.exists():
            message = (
                f"Missing prepared distillation cache for {replay_path}: {path}. "
                "Run examples/precompute_distillation_dataset.py first."
            )
            raise FileNotFoundError(message)
        with np.load(path, allow_pickle=False) as archive:
            value = {name: archive[name] for name in archive.files}
        self.data[key] = value
        while len(self.data) > self.max_size:
            self.data.popitem(last=False)
        return value


class LuxDistillationDataset(Dataset):
    def __init__(
        self,
        replay_paths: Sequence[Path],
        teacher_cache_dir: Path,
        *,
        winner_weight: float = 1.5,
        seed: int = 42,
        max_turns: int = 0,
        replay_cache_dir: Path | None = None,
        prepared_cache_dir: Path | None = None,
        prepared_observation_dtype: str = "float16",
        prepared_cache_size: int = 2,
    ) -> None:
        self.winner_weight = winner_weight
        self.samples: list[tuple[Path, int, int]] = []
        self.sample_groups: list[list[int]] = []
        self.prepared_cache = None
        self.base = None
        self.teacher_cache = None
        if prepared_cache_dir is not None:
            if prepared_cache_size < 1:
                raise ValueError("Prepared distillation cache size must be positive")
            self.prepared_cache = _PreparedDistillationCache(
                prepared_cache_dir,
                prepared_observation_dtype,
                max_size=prepared_cache_size,
            )
            for replay_path in replay_paths:
                path = self.prepared_cache.path_for(replay_path)
                if not path.exists():
                    message = (
                        f"Missing prepared distillation cache for {replay_path}: {path}. "
                        "Run examples/precompute_distillation_dataset.py first."
                    )
                    raise FileNotFoundError(message)
                metadata = prepared_cache_metadata(path)
                if metadata["cache_version"] != DISTILLATION_PREPARED_CACHE_VERSION:
                    message = f"Incompatible prepared distillation cache: {path}"
                    raise ValueError(message)
                turn_count = int(metadata["turn_count"])
                if max_turns > 0:
                    turn_count = min(turn_count, max_turns)
                group_start = len(self.samples)
                self.samples.extend((Path(replay_path), turn, team) for turn in range(turn_count) for team in (0, 1))
                self.sample_groups.append(list(range(group_start, len(self.samples))))
        else:
            self.base = LuxReplayDataset(
                replay_paths,
                augment=False,
                team_selection="all",
                winner_weight=winner_weight,
                seed=seed,
                max_turns=max_turns,
                replay_cache_dir=replay_cache_dir,
            )
            self.samples = self.base.samples
            self.sample_groups = self.base.sample_groups
            self.teacher_cache = _DistillationCache(teacher_cache_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if self.prepared_cache is not None:
            replay_path, turn, team = self.samples[index]
            cached = self.prepared_cache.get(replay_path)
            sample_index = turn * 2 + team
            result = {
                "observation": torch.from_numpy(cached["observation"][turn, team]),
                "sample_weight": torch.tensor(
                    self.winner_weight if team == int(cached["winner"]) else 1.0,
                    dtype=torch.float32,
                ),
            }
            for entity in FIRST_PLACE_ACTION_SCHEMA:
                offsets = cached[f"{entity}_offsets"]
                start, stop = int(offsets[sample_index]), int(offsets[sample_index + 1])
                result[f"{entity}_positions"] = torch.from_numpy(cached[f"{entity}_positions"][start:stop]).long()
                result[f"{entity}_flat"] = torch.from_numpy(cached[f"{entity}_flat"][start:stop]).long()
                result[f"{entity}_legal_mask"] = torch.from_numpy(cached[f"{entity}_legal_mask"][start:stop])
                result[f"{entity}_teacher_logits"] = torch.from_numpy(cached[f"{entity}_teacher_logits"][start:stop])
            return result

        if self.base is None or self.teacher_cache is None:
            raise RuntimeError("Distillation dataset caches were not initialized")
        snapshot, actions, team, replay = self.base._snapshot_actions_team(index)
        replay_path, turn, _ = self.samples[index]
        targets = build_first_place_targets(snapshot, team, actions)
        cached = self.teacher_cache.get(replay_path)
        if not cache_matches_replay(
            cached,
            replay_path,
            FIRST_PLACE_TEACHER_SHA256,
            rot180=True,
        ):
            message = f"Teacher cache is stale or incompatible: {replay_path}"
            raise ValueError(message)
        cached_turns = cached["turns"]
        if turn >= len(cached_turns):
            message = f"Teacher cache ends at turn {len(cached_turns) - 1}, requested turn {turn}: {replay_path}"
            raise ValueError(message)
        cached_logits = cached_turns[turn][team]
        result: dict[str, Tensor] = {}
        for entity in FIRST_PLACE_ACTION_SCHEMA:
            values = cached_logits[entity]
            if len(values) > MAX_ENTITIES:
                message = f"Too many cached {entity} entities: {len(values)}"
                raise ValueError(message)
            expected = int(np.count_nonzero(targets[f"{entity}_flat"] != IGNORE_INDEX))
            if expected != len(values):
                message = (
                    f"Teacher cache entity mismatch for {replay_path} turn={turn} team={team} "
                    f"entity={entity}: expected={expected} cached={len(values)}"
                )
                raise ValueError(message)
            result[f"{entity}_positions"] = torch.from_numpy(targets[f"{entity}_positions"][:expected])
            result[f"{entity}_flat"] = torch.from_numpy(targets[f"{entity}_flat"][:expected])
            result[f"{entity}_legal_mask"] = torch.from_numpy(targets[f"{entity}_legal_mask"][:expected])
            result[f"{entity}_teacher_logits"] = values

        observation = encode_snapshot(snapshot, team)
        rewards = replay.get("rewards") or ()
        winner = _winner_from_rewards(rewards)
        result["observation"] = torch.from_numpy(observation)
        result["sample_weight"] = torch.tensor(self.winner_weight if team == winner else 1.0, dtype=torch.float32)
        return result


def compact_distillation_collate(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate only actionable entities instead of padding every entity type to 32x32."""
    if not samples:
        raise ValueError("Cannot collate an empty distillation batch")
    result = {
        "observation": torch.stack([sample["observation"] for sample in samples]),
        "sample_weight": torch.stack([sample["sample_weight"] for sample in samples]),
    }
    for entity in FIRST_PLACE_ACTION_SCHEMA:
        result[f"{entity}_positions"] = pad_sequence(
            [sample[f"{entity}_positions"] for sample in samples],
            batch_first=True,
            padding_value=-1,
        )
        result[f"{entity}_flat"] = pad_sequence(
            [sample[f"{entity}_flat"] for sample in samples],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        for suffix in ("legal_mask", "teacher_logits"):
            key = f"{entity}_{suffix}"
            result[key] = pad_sequence(
                [sample[key] for sample in samples],
                batch_first=True,
                padding_value=0,
            )
    return result


def augment_distillation_batch(
    batch: Mapping[str, Tensor],
    transform_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    """Apply independent D4 transforms to a compact batch on its current device."""
    observation = batch["observation"]
    batch_size = observation.shape[0]
    if transform_ids is None:
        transform_ids = torch.randint(len(_D4_TRANSFORMS), (batch_size,), device=observation.device)
    else:
        if transform_ids.shape != (batch_size,):
            raise ValueError("transform_ids must contain one D4 transform index per sample")
        if transform_ids.device.type == "cpu" and torch.any(
            (transform_ids < 0) | (transform_ids >= len(_D4_TRANSFORMS))
        ):
            raise ValueError("transform_ids must contain one D4 transform index per sample")
        transform_ids = transform_ids.to(device=observation.device, dtype=torch.long)

    transformed = dict(batch)
    spatial_remaps, position_remaps = _device_spatial_remaps(observation.device)
    spatial_indices = spatial_remaps.index_select(0, transform_ids)
    transformed_observation = torch.gather(
        observation.flatten(2),
        2,
        spatial_indices[:, None].expand(-1, observation.shape[1], -1),
    ).reshape_as(observation)
    # Coordinates describe the transformed board frame, not rotated scalar images.
    transformed_observation[:, FEATURE_INDEX["x_coordinate"]] = observation[:, FEATURE_INDEX["x_coordinate"]]
    transformed_observation[:, FEATURE_INDEX["y_coordinate"]] = observation[:, FEATURE_INDEX["y_coordinate"]]
    transformed["observation"] = transformed_observation

    for entity in FIRST_PLACE_ACTION_SCHEMA:
        remaps, inverse_remaps = _device_action_remaps(entity, observation.device)
        positions = batch[f"{entity}_positions"]
        labels = batch[f"{entity}_flat"]
        valid_positions = positions[..., 0] >= 0
        safe_positions = positions.clamp_min(0)
        flat_positions = safe_positions[..., 0] * BOARD_SIZE + safe_positions[..., 1]
        sample_position_remaps = position_remaps.index_select(0, transform_ids)
        transformed_flat_positions = torch.gather(sample_position_remaps, 1, flat_positions)
        transformed_positions = torch.stack(
            (transformed_flat_positions // BOARD_SIZE, transformed_flat_positions % BOARD_SIZE),
            dim=-1,
        )
        transformed[f"{entity}_positions"] = torch.where(
            valid_positions[..., None],
            transformed_positions,
            positions,
        )

        valid_labels = labels != IGNORE_INDEX
        sample_action_remaps = remaps.index_select(0, transform_ids)
        transformed_labels = torch.gather(sample_action_remaps, 1, labels.clamp_min(0))
        transformed[f"{entity}_flat"] = torch.where(valid_labels, transformed_labels, labels)
        inverse = inverse_remaps.index_select(0, transform_ids)
        for suffix in ("legal_mask", "teacher_logits"):
            values = batch[f"{entity}_{suffix}"]
            transformed[f"{entity}_{suffix}"] = torch.gather(
                values,
                -1,
                inverse[:, None].expand(-1, values.shape[1], -1),
            )
    return transformed


def distillation_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    temperature: float = 2.0,
    distill_weight: float = 0.75,
    hard_label_weight: float = 0.0,
    illegal_weight: float = 0.1,
) -> dict[str, Tensor]:
    sample_weight = batch["sample_weight"]
    losses = {}
    for entity in FIRST_PLACE_ACTION_SCHEMA:
        positions = batch[f"{entity}_positions"]
        student_raw = _entity_logits(output[entity], positions)
        legal = batch[f"{entity}_legal_mask"].permute(0, 2, 1)
        
        student_probs = nn_functional.softmax(student_raw, dim=1)
        p_legal = (student_probs * legal).sum(dim=1)
        
        labels = batch[f"{entity}_flat"]
        valid = labels != IGNORE_INDEX
        weights = sample_weight[:, None].expand_as(labels) * valid
        
        illegal_loss_tensor = -torch.log(p_legal.clamp_min(1e-8))
        illegal = (illegal_loss_tensor * weights).sum() / weights.sum().clamp_min(1)
        losses[f"{entity}_illegal_loss"] = illegal

        student = apply_legal_action_mask(student_raw, legal, action_dim=1)
        
        if hard_label_weight > 0:
            hard = nn_functional.cross_entropy(
                student,
                labels,
                ignore_index=IGNORE_INDEX,
                reduction="none",
            )
            hard = (hard * weights).sum() / weights.sum().clamp_min(1)
            losses[f"{entity}_hard_loss"] = hard

        teacher = batch[f"{entity}_teacher_logits"].permute(0, 2, 1).to(dtype=student.dtype)
        teacher = apply_legal_action_mask(teacher, legal, action_dim=1)
        student_log_probs = nn_functional.log_softmax(student / temperature, dim=1)
        teacher_probs = nn_functional.softmax(teacher / temperature, dim=1)
        kl = nn_functional.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
        kl = (kl * weights).sum() / weights.sum().clamp_min(1) * temperature**2
        losses[f"{entity}_distill_loss"] = kl
        
    hard_total = torch.stack([value for name, value in losses.items() if name.endswith("_hard_loss")]).sum() if hard_label_weight > 0 else torch.tensor(0.0, device=sample_weight.device)
    distill_total = torch.stack([value for name, value in losses.items() if name.endswith("_distill_loss")]).sum()
    illegal_total = torch.stack([value for name, value in losses.items() if name.endswith("_illegal_loss")]).sum()
    
    total = hard_label_weight * hard_total + distill_weight * distill_total + illegal_weight * illegal_total
    
    result = {"loss": total, "distill_loss": distill_total, "illegal_loss": illegal_total, **losses}
    if hard_label_weight > 0:
        result["hard_loss"] = hard_total
    return result


def distillation_metrics(output: Mapping[str, Tensor], batch: Mapping[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    metrics = {}
    for entity in FIRST_PLACE_ACTION_SCHEMA:
        student_raw = _entity_logits(output[entity], batch[f"{entity}_positions"])
        legal = batch[f"{entity}_legal_mask"].permute(0, 2, 1)
        
        student_probs = nn_functional.softmax(student_raw, dim=1)
        p_illegal = (student_probs * ~legal).sum(dim=1)
        
        student_raw_actions = student_raw.argmax(dim=1)
        is_legal_argmax = legal.gather(1, student_raw_actions.unsqueeze(1)).squeeze(1)
        is_illegal_argmax = ~is_legal_argmax
        
        student = apply_legal_action_mask(student_raw, legal, action_dim=1)
        teacher = batch[f"{entity}_teacher_logits"].permute(0, 2, 1).to(student.dtype)
        teacher = apply_legal_action_mask(teacher, legal, action_dim=1)
        labels = batch[f"{entity}_flat"]
        valid = labels != IGNORE_INDEX
        student_actions = student.argmax(dim=1)
        teacher_actions = teacher.argmax(dim=1)
        metrics[f"{entity}_teacher_agreement"] = ((student_actions == teacher_actions) & valid).sum(), valid.sum()
        metrics[f"{entity}_hard_accuracy"] = ((student_actions == labels) & valid).sum(), valid.sum()
        metrics[f"{entity}_illegal_top1"] = (is_illegal_argmax & valid).sum(), valid.sum()
        metrics[f"{entity}_illegal_prob_mass"] = (p_illegal * valid).sum(), valid.sum()
    return metrics
