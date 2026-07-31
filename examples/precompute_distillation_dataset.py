from __future__ import annotations

# ruff: noqa: INP001
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from luxai2021.imitation.actions import FIRST_PLACE_ACTION_SCHEMA
from luxai2021.imitation.data import IGNORE_INDEX, _load_replay, _winner_from_rewards, discover_replays
from luxai2021.imitation.distillation import (
    DISTILLATION_PREPARED_CACHE_VERSION,
    cache_matches_replay,
    distillation_cache_path,
    load_distillation_cache,
    prepared_cache_metadata,
    prepared_distillation_cache_path,
    save_prepared_distillation_cache,
)
from luxai2021.imitation.first_place import FIRST_PLACE_TEACHER_SHA256, build_first_place_targets
from luxai2021.imitation.schema import BOARD_SIZE, FEATURE_NAMES, encode_snapshot, snapshot_from_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute compact student-ready observations and targets for distillation."
    )
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--teacher-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-cache-dir")
    parser.add_argument("--observation-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _empty_entity_arrays(entity: str, teacher_dtype: np.dtype) -> dict[str, np.ndarray]:
    action_count = len(FIRST_PLACE_ACTION_SCHEMA[entity])
    return {
        f"{entity}_positions": np.empty((0, 2), dtype=np.int16),
        f"{entity}_flat": np.empty(0, dtype=np.int8),
        f"{entity}_legal_mask": np.empty((0, action_count), dtype=np.bool_),
        f"{entity}_teacher_logits": np.empty((0, action_count), dtype=teacher_dtype),
    }


def _prepare_one(arguments: tuple[str, str, str, str | None, str]) -> dict[str, object]:
    replay_name, teacher_cache_name, output_name, replay_cache_name, observation_dtype = arguments
    replay_path = Path(replay_name)
    output_dir = Path(output_name)
    output_path = prepared_distillation_cache_path(replay_path, output_dir, observation_dtype)
    if output_path.exists():
        metadata = prepared_cache_metadata(output_path)
        if (
            metadata["cache_version"] == DISTILLATION_PREPARED_CACHE_VERSION
            and metadata["observation_dtype"] == observation_dtype
        ):
            return {
                "created": False,
                "turn_count": metadata["turn_count"],
                "size_bytes": output_path.stat().st_size,
            }

    teacher_path = distillation_cache_path(replay_path, Path(teacher_cache_name))
    teacher = load_distillation_cache(teacher_path)
    if not cache_matches_replay(teacher, replay_path, FIRST_PLACE_TEACHER_SHA256, rot180=True):
        message = f"Teacher cache is stale or incompatible: {replay_path}"
        raise ValueError(message)
    replay_cache_dir = Path(replay_cache_name) if replay_cache_name else None
    replay = _load_replay(replay_path, replay_cache_dir)
    turn_count = min(len(replay["steps"]) - 1, len(teacher["turns"]))
    initial = replay["steps"][0][0]["observation"]
    width, height = int(initial["width"]), int(initial["height"])
    numpy_observation_dtype = np.float16 if observation_dtype == "float16" else np.float32
    observations = np.empty(
        (turn_count, 2, len(FEATURE_NAMES), BOARD_SIZE, BOARD_SIZE),
        dtype=numpy_observation_dtype,
    )
    entity_values = {
        entity: {suffix: [] for suffix in ("positions", "flat", "legal_mask", "teacher_logits")}
        for entity in FIRST_PLACE_ACTION_SCHEMA
    }
    offsets = {entity: [0] for entity in FIRST_PLACE_ACTION_SCHEMA}
    for turn in range(turn_count):
        observation = replay["steps"][turn][0]["observation"]
        snapshot = snapshot_from_updates(observation["updates"], width, height, turn)
        for team in (0, 1):
            observations[turn, team] = encode_snapshot(snapshot, team).astype(numpy_observation_dtype, copy=False)
            actions = replay["steps"][turn + 1][team].get("action") or []
            targets = build_first_place_targets(snapshot, team, actions)
            cached_logits = teacher["turns"][turn][team]
            for entity in FIRST_PLACE_ACTION_SCHEMA:
                logits = cached_logits[entity].numpy()
                count = int(np.count_nonzero(targets[f"{entity}_flat"] != IGNORE_INDEX))
                if count != len(logits):
                    message = (
                        f"Teacher cache entity mismatch for {replay_path} turn={turn} team={team} "
                        f"entity={entity}: expected={count} cached={len(logits)}"
                    )
                    raise ValueError(message)
                entity_values[entity]["positions"].append(targets[f"{entity}_positions"][:count].astype(np.int16))
                entity_values[entity]["flat"].append(targets[f"{entity}_flat"][:count].astype(np.int8))
                entity_values[entity]["legal_mask"].append(targets[f"{entity}_legal_mask"][:count])
                entity_values[entity]["teacher_logits"].append(logits)
                offsets[entity].append(offsets[entity][-1] + count)

    winner = _winner_from_rewards(replay.get("rewards") or ())
    arrays: dict[str, np.ndarray] = {
        "cache_version": np.asarray(DISTILLATION_PREPARED_CACHE_VERSION, dtype=np.int64),
        "turn_count": np.asarray(turn_count, dtype=np.int64),
        "winner": np.asarray(-1 if winner is None else winner, dtype=np.int8),
        "observation_dtype": np.asarray(observation_dtype),
        "observation": observations,
    }
    for entity in FIRST_PLACE_ACTION_SCHEMA:
        teacher_dtype = teacher["turns"][0][0][entity].numpy().dtype if turn_count else np.dtype(np.float16)
        empty = _empty_entity_arrays(entity, teacher_dtype)
        arrays[f"{entity}_offsets"] = np.asarray(offsets[entity], dtype=np.int64)
        for suffix, parts in entity_values[entity].items():
            key = f"{entity}_{suffix}"
            arrays[key] = np.concatenate(parts) if parts else empty[key]
    save_prepared_distillation_cache(output_path, arrays)
    return {"created": True, "turn_count": turn_count, "size_bytes": output_path.stat().st_size}


def main() -> None:
    args = parse_args()
    if args.num_workers < -1:
        raise ValueError("--num-workers must be non-negative, or -1 for automatic workers")
    replay_paths = discover_replays(args.replay_dir)
    worker_count = min(4, max(1, (os.cpu_count() or 2) // 4)) if args.num_workers < 0 else args.num_workers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.replay_cache_dir:
        Path(args.replay_cache_dir).mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(path),
            args.teacher_cache_dir,
            str(output_dir),
            args.replay_cache_dir,
            args.observation_dtype,
        )
        for path in replay_paths
    ]
    started = time.perf_counter()
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                tqdm(
                    executor.map(_prepare_one, tasks, chunksize=1),
                    total=len(tasks),
                    desc="Prepared distillation cache",
                    unit="replay",
                    dynamic_ncols=True,
                    disable=args.no_progress,
                )
            )
    else:
        results = [
            _prepare_one(task)
            for task in tqdm(
                tasks,
                desc="Prepared distillation cache",
                unit="replay",
                dynamic_ncols=True,
                disable=args.no_progress,
            )
        ]
    duration = time.perf_counter() - started
    manifest = {
        "cache_version": DISTILLATION_PREPARED_CACHE_VERSION,
        "observation_dtype": args.observation_dtype,
        "replay_count": len(results),
        "created_count": sum(bool(result["created"]) for result in results),
        "turn_count": sum(int(result["turn_count"]) for result in results),
        "size_bytes": sum(int(result["size_bytes"]) for result in results),
        "duration_seconds": duration,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
