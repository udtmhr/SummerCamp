from __future__ import annotations

# ruff: noqa: C901, INP001, PLR0915
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from luxai2021.imitation.data import _load_replay, discover_replays
from luxai2021.imitation.distillation import (
    DISTILLATION_CACHE_VERSION,
    cache_matches_replay,
    distillation_cache_path,
    extract_teacher_turn,
    load_distillation_cache,
    replay_fingerprint,
    save_distillation_cache,
)
from luxai2021.imitation.first_place import (
    FIRST_PLACE_TEACHER_SHA256,
    FIRST_PLACE_UPSTREAM_COMMIT,
    load_first_place_teacher,
    predict_first_place,
)
from luxai2021.imitation.schema import snapshot_from_updates


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute first-place teacher logits for Lux replays.")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay-cache-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--rot180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    teacher_path = Path(args.teacher_checkpoint)
    teacher_sha = sha256_file(teacher_path)
    if teacher_sha != FIRST_PLACE_TEACHER_SHA256:
        message = f"Unexpected teacher SHA-256: {teacher_sha}"
        raise ValueError(message)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_cache_dir = Path(args.replay_cache_dir) if args.replay_cache_dir else None
    if replay_cache_dir is not None:
        replay_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dtype = {"float16": torch.float16, "float32": torch.float32}[args.cache_dtype]
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.amp_dtype]
    model = load_first_place_teacher(teacher_path, device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    replay_paths = discover_replays(args.replay_dir)
    created = skipped = total_turns = 0
    started = time.perf_counter()
    progress = tqdm(replay_paths, desc="Teacher cache", unit="replay", disable=args.no_progress, dynamic_ncols=True)
    for replay_path in progress:
        cache_path = distillation_cache_path(replay_path, output_dir)
        if cache_path.exists():
            cached = load_distillation_cache(cache_path)
            if cache_matches_replay(
                cached,
                replay_path,
                teacher_sha,
                rot180=args.rot180,
                cache_dtype=args.cache_dtype,
                amp_dtype=args.amp_dtype,
            ):
                skipped += 1
                total_turns += len(cached["turns"])
                continue
        replay = _load_replay(replay_path, replay_cache_dir)
        turn_count = len(replay["steps"]) - 1
        if args.max_turns > 0:
            turn_count = min(turn_count, args.max_turns)
        first_observation = replay["steps"][0][0]["observation"]
        width, height = int(first_observation["width"]), int(first_observation["height"])
        turns = []
        for start in range(0, turn_count, args.batch_size):
            stop = min(turn_count, start + args.batch_size)
            snapshots = []
            for turn in range(start, stop):
                observation = replay["steps"][turn][0]["observation"]
                snapshots.append(snapshot_from_updates(observation["updates"], width, height, turn))
            output = predict_first_place(
                model,
                snapshots,
                device=device,
                rot180=args.rot180,
                amp_dtype=amp_dtype,
            )
            turns.extend(
                extract_teacher_turn(snapshot, output, index, dtype=cache_dtype)
                for index, snapshot in enumerate(snapshots)
            )
        save_distillation_cache(
            cache_path,
            {
                "cache_version": DISTILLATION_CACHE_VERSION,
                "source": replay_fingerprint(replay_path),
                "teacher_sha256": teacher_sha,
                "upstream_commit": FIRST_PLACE_UPSTREAM_COMMIT,
                "rot180": args.rot180,
                "dtype": args.cache_dtype,
                "amp_dtype": args.amp_dtype,
                "turns": turns,
            },
        )
        created += 1
        total_turns += turn_count
        progress.set_postfix(created=created, skipped=skipped, turns=total_turns)
    duration = time.perf_counter() - started
    manifest = {
        "cache_version": DISTILLATION_CACHE_VERSION,
        "teacher_sha256": teacher_sha,
        "upstream_commit": FIRST_PLACE_UPSTREAM_COMMIT,
        "rot180": args.rot180,
        "cache_dtype": args.cache_dtype,
        "amp_dtype": args.amp_dtype,
        "replay_count": len(replay_paths),
        "created_count": created,
        "skipped_count": skipped,
        "turn_count": total_turns,
        "duration_seconds": duration,
        "turns_per_second": total_turns / max(duration, 1e-9),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
