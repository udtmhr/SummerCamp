from __future__ import annotations

# ruff: noqa: INP001, PLR0915
import argparse
import copy
import hashlib
import json
import time
from contextlib import suppress
from pathlib import Path

import torch
from tqdm import tqdm

from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.imitation import FirstPlaceAgent
from luxai2021.imitation.first_place import FIRST_PLACE_TEACHER_SHA256
from luxai2021.imitation.selfplay import KaggleReplayRecorder

DEFAULT_TEACHER = "models/teachers/lux_2021_first_place/062179520_weights.pt"


class RecordingFirstPlaceAgent(FirstPlaceAgent):
    def __init__(
        self,
        checkpoint_path: str,
        *,
        recorder: KaggleReplayRecorder,
        device: str,
        tta: str,
    ) -> None:
        super().__init__(checkpoint_path, device=device, tta=tta)
        self.recorder = recorder
        self.records_turns = True

    def post_turn(self, game: object, actions: list[object]) -> bool:
        if self.records_turns:
            self.recorder.record_turn(game, actions)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Kaggle-compatible replay data from first-place teacher self-play."
    )
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--output-dir", default="replay_datasets/first_place_selfplay")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tta", choices=("auto", "none", "rot180"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_path_for_seed(output_dir: Path, seed: int) -> Path:
    return output_dir / f"first_place_selfplay_seed{seed:010d}.json"


def existing_replay_matches(path: Path, *, seed: int, teacher_sha256: str, tta: str) -> bool:
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
        info = replay["info"]
        return bool(
            replay["configuration"]["seed"] == seed
            and info["source"] == "first-place-selfplay"
            and info["teacher_sha256"] == teacher_sha256
            and info["tta"] == tta
            and replay["steps"]
        )
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.games < 1:
        raise ValueError("--games must be positive")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be zero or greater")

    checkpoint_path = Path(args.teacher_checkpoint)
    teacher_sha = sha256_file(checkpoint_path)
    if teacher_sha != FIRST_PLACE_TEACHER_SHA256:
        message = f"Unexpected teacher SHA-256: expected={FIRST_PLACE_TEACHER_SHA256} actual={teacher_sha}"
        raise ValueError(message)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_tta = "rot180" if args.tta == "auto" else args.tta
    seeds = list(range(args.seed_start, args.seed_start + args.games))

    existing = []
    pending = []
    for seed in seeds:
        path = output_path_for_seed(output_dir, seed)
        if path.exists() and existing_replay_matches(path, seed=seed, teacher_sha256=teacher_sha, tta=resolved_tta):
            existing.append(seed)
        elif path.exists() and not args.overwrite:
            message = f"Existing replay is incompatible or incomplete (use --overwrite to replace it): {path}"
            raise ValueError(message)
        else:
            pending.append(seed)

    recorder = KaggleReplayRecorder()
    agent_a = RecordingFirstPlaceAgent(
        str(checkpoint_path),
        recorder=recorder,
        device=args.device,
        tta=args.tta,
    )
    # Both agents are stateless during inference. A shallow copy gives each side
    # independent team/controller fields while sharing the one loaded teacher model.
    agent_b = copy.copy(agent_a)
    agent_b.records_turns = False
    teacher_name = checkpoint_path.name
    agent_a.replay_name = teacher_name
    agent_b.replay_name = teacher_name
    if agent_a.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    initial_seed = pending[0] if pending else seeds[0]
    config = dict(LuxMatchConfigs_Default)
    config["seed"] = initial_seed
    env = LuxEnvironment(config, agent_a, agent_b)
    created = []
    turn_count = 0
    started = time.perf_counter()
    progress = tqdm(pending, desc="First-place self-play", unit="game", disable=args.no_progress, dynamic_ncols=True)
    for seed in progress:
        recorder.reset()
        with suppress(StopIteration):
            env.reset(seed=seed)
        replay = recorder.build_replay(
            env.game,
            seed=seed,
            teacher_sha256=teacher_sha,
            tta=resolved_tta,
            team_names=(teacher_name, teacher_name),
        )
        write_json_atomic(output_path_for_seed(output_dir, seed), replay)
        created.append(seed)
        game_turns = len(recorder.turns)
        turn_count += game_turns
        progress.set_postfix(seed=seed, turns=game_turns)

    duration = time.perf_counter() - started
    manifest = {
        "created_count": len(created),
        "created_seeds": created,
        "device": str(agent_a.device),
        "duration_seconds": duration,
        "existing_count": len(existing),
        "existing_seeds": existing,
        "game_count": len(seeds),
        "output_dir": str(output_dir.resolve()),
        "seed_start": args.seed_start,
        "source": "first-place-selfplay",
        "teacher_checkpoint": str(checkpoint_path.resolve()),
        "teacher_sha256": teacher_sha,
        "tta": resolved_tta,
        "turn_count_created": turn_count,
    }
    write_json_atomic(output_dir / "first_place_selfplay_info.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
