from __future__ import annotations

import argparse
import secrets
from contextlib import suppress

from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.imitation import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.imitation.matches import model_label, rename_replay


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a local match between two Lux AI 2021 models.")
    parser.add_argument(
        "--seed",
        type=parse_seed,
        default=None,
        help="Map and team-assignment seed. Use an integer to reproduce a match; default: random.",
    )
    parser.add_argument("--model-a", default="models/bc_v2/best.pt", help="First model checkpoint.")
    parser.add_argument("--model-b", default="models/bc_v2/best.pt", help="Second model checkpoint.")
    parser.add_argument(
        "--type-a",
        default="bc",
        choices=("bc", "first-place"),
        help="Model A checkpoint type.",
    )
    parser.add_argument(
        "--type-b",
        default="bc",
        choices=("bc", "first-place"),
        help="Model B checkpoint type.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Inference device.")
    parser.add_argument(
        "--tta-a",
        default="auto",
        choices=("auto", "none", "rot180"),
        help="Inference augmentation for model A; first-place auto uses rot180.",
    )
    parser.add_argument(
        "--tta-b",
        default="auto",
        choices=("auto", "none", "rot180"),
        help="Inference augmentation for model B; first-place auto uses rot180.",
    )
    parser.add_argument("--replay-dir", default="replays", help="Replay output directory.")
    return parser


def create_agent(checkpoint: str, model_type: str, device: str, tta: str) -> BehaviorCloningAgent:
    if model_type == "first-place":
        return FirstPlaceAgent(checkpoint, device=device, tta=tta)
    return BehaviorCloningAgent(checkpoint, device=device, tta=tta)


def main() -> None:
    args = build_parser().parse_args()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)

    config = dict(LuxMatchConfigs_Default)
    config["seed"] = seed

    agent_a = create_agent(args.model_a, args.type_a, args.device, args.tta_a)
    agent_a.replay_name = args.model_a
    agent_b = create_agent(args.model_b, args.type_b, args.device, args.tta_b)
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
