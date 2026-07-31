from __future__ import annotations

import argparse
import json
import re
import secrets
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.imitation import BehaviorCloningAgent


def parse_seed(value: str) -> int | None:
    if value.lower() == "random":
        return None
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer or 'random'") from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be zero or greater")
    return seed


def model_label(model_path: str) -> str:
    path = Path(model_path)
    parent = path.parent.name
    label = f"{parent}_{path.stem}" if parent else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label)


def rename_replay(replay_file: str, winner_name: str) -> Path:
    source = Path(replay_file)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_winner-{model_label(winner_name)}"
    target = source.with_name(f"{stem}.json")
    suffix = 2
    while target.exists():
        target = source.with_name(f"{stem}_{suffix}.json")
        suffix += 1

    source.rename(target)
    replay = json.loads(target.read_text(encoding="utf-8"))
    replay["results"]["replayFile"] = str(target)
    target.write_text(json.dumps(replay), encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a local match between two behavior-cloning models.")
    parser.add_argument(
        "--seed",
        type=parse_seed,
        default=None,
        help="Map and team-assignment seed. Use an integer to reproduce a match; default: random.",
    )
    parser.add_argument("--model-a", default="models/bc_v2/best.pt", help="First model checkpoint.")
    parser.add_argument("--model-b", default="models/bc_v2/best.pt", help="Second model checkpoint.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Inference device.")
    parser.add_argument(
        "--tta-a",
        default="auto",
        choices=("auto", "none", "rot180"),
        help="Inference augmentation for model A; auto reads the checkpoint metadata.",
    )
    parser.add_argument(
        "--tta-b",
        default="auto",
        choices=("auto", "none", "rot180"),
        help="Inference augmentation for model B; auto reads the checkpoint metadata.",
    )
    parser.add_argument("--replay-dir", default="replays", help="Replay output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)

    config = dict(LuxMatchConfigs_Default)
    config["seed"] = seed

    agent_a = BehaviorCloningAgent(args.model_a, device=args.device, tta=args.tta_a)
    agent_a.replay_name = args.model_a
    agent_b = BehaviorCloningAgent(args.model_b, device=args.device, tta=args.tta_b)
    agent_b.replay_name = args.model_b

    match_name = f"{model_label(args.model_a)}-vs-{model_label(args.model_b)}_seed{seed}"
    env = LuxEnvironment(
        config,
        agent_a,
        agent_b,
        replay_folder=args.replay_dir,
        replay_prefix=match_name,
    )

    with suppress(StopIteration):
        env.reset(seed=seed)

    team_names = {
        agent_a.team: agent_a.replay_name,
        agent_b.team: agent_b.replay_name,
    }
    winner = env.game.last_winning_team
    replay_path = rename_replay(env.game.last_replay_file, team_names[winner])
    env.game.last_replay_file = str(replay_path)

    print(f"Seed: {seed}")
    print(f"試合: Team 0={team_names[0]} vs Team 1={team_names[1]}")
    print(f"勝者: Team {winner} ({team_names[winner]})")
    print("終了ターン:", env.game.state["turn"])
    print("Replay:", env.game.last_replay_file)


if __name__ == "__main__":
    main()
