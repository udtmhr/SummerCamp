# ruff: noqa: ANN202, PLR0913

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from luxai2021.rl.batched_rollout import ActorCriticBatcher, BatchedOpponentPool
from luxai2021.rl.policy import FullTurnActorCritic
from luxai2021.rl.ppo import aggregate_episode_timings, collect_episodes_batched
from luxai2021.rl.reward import default_reward_program

NATIVE_CPU_SHARE_GATE = 0.25
NATIVE_CPU_MIN_DECISIONS = 40_000


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        message = "Checkpoint must use NAME=PATH"
        raise argparse.ArgumentTypeError(message)
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_file():
        message = f"Checkpoint does not exist: {path}"
        raise argparse.ArgumentTypeError(message)
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lockstep/threaded evolutionary RL rollout throughput.")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rollout-envs", action="append", type=int)
    parser.add_argument("--backend", action="append", choices=("lockstep", "threaded"))
    parser.add_argument("--precision", action="append", choices=("auto", "fp32", "bf16", "fp16"))
    parser.add_argument("--compile", dest="compile_modes", action="append", choices=("auto", "on", "off"))
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--decisions", type=int, help="Minimum rollout decisions per repetition (default: 40000).")
    budget.add_argument("--seconds", type=float, help="Minimum rollout wall-clock seconds per repetition.")
    parser.add_argument("--max-turns", type=int, default=360)
    parser.add_argument("--warmup-turns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_batchers(
    checkpoint: Path,
    *,
    device: torch.device,
    rollout_envs: int,
    backend: str,
    precision: str,
    compile_mode: str,
    wait_seconds: float,
) -> tuple[FullTurnActorCritic, ActorCriticBatcher, BatchedOpponentPool]:
    actor = FullTurnActorCritic.from_checkpoint(checkpoint, device).eval()
    snapshot = copy.deepcopy(actor).eval().requires_grad_(requires_grad=False)
    pad_batches = backend == "lockstep"
    candidate = ActorCriticBatcher(
        actor,
        device,
        rollout_envs,
        name="benchmark-candidate",
        wait_seconds=wait_seconds,
        precision=precision,
        compile_mode=compile_mode,
        pad_batches=pad_batches,
    )
    opponent = ActorCriticBatcher(
        snapshot,
        device,
        rollout_envs,
        name="benchmark-opponent",
        wait_seconds=wait_seconds,
        precision=precision,
        compile_mode=compile_mode,
        pad_batches=pad_batches,
    )
    return actor, candidate, BatchedOpponentPool({"snapshot": opponent})


def run_once(
    checkpoint: Path,
    *,
    device: torch.device,
    rollout_envs: int,
    backend: str,
    precision: str,
    compile_mode: str,
    wait_seconds: float,
    decisions: int,
    seconds: float | None,
    max_turns: int,
    warmup_turns: int,
    seed: int,
) -> dict[str, object]:
    actor, candidate, opponents = make_batchers(
        checkpoint,
        device=device,
        rollout_envs=rollout_envs,
        backend=backend,
        precision=precision,
        compile_mode=compile_mode,
        wait_seconds=wait_seconds,
    )
    opponent_factory = opponents.factory("snapshot")

    def collect(seed_start: int, turn_limit: int):
        specs = [(opponent_factory, seed_start + offset, "snapshot") for offset in range(rollout_envs)]
        return collect_episodes_batched(
            actor,
            specs,
            default_reward_program(),
            device=device,
            inference_backend=candidate.submit,
            max_turns=turn_limit,
            rollout_backend=backend,
        )

    try:
        collect(seed + 9_000_000, warmup_turns)
        candidate.reset_metrics()
        opponents.reset_metrics()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()
        decision_count = 0
        turn_count = 0
        game_count = 0
        episode_timings: dict[str, float] = {}
        wave = 0
        while decision_count < decisions or (seconds is not None and time.perf_counter() - started_at < seconds):
            episodes = collect(seed + wave * rollout_envs, max_turns)
            wave += 1
            game_count += len(episodes)
            turn_count += sum(len(episode.records) for episode in episodes)
            decision_count += sum(len(record.decisions) for episode in episodes for record in episode.records)
            for name, duration in aggregate_episode_timings(episodes).items():
                episode_timings[name] = episode_timings.get(name, 0.0) + duration
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started_at
        native_cpu_seconds = sum(
            episode_timings.get(name, 0.0) for name in ("snapshot", "encode", "game_step", "reward_metrics")
        )
        return {
            "elapsed_seconds": elapsed,
            "decisions": decision_count,
            "turns": turn_count,
            "games": game_count,
            "budget": {"decisions": decisions or None, "seconds": seconds},
            "decisions_per_second": decision_count / elapsed,
            "turns_per_second": turn_count / elapsed,
            "games_per_second": game_count / elapsed,
            "candidate_inference": candidate.metrics(),
            "opponent_inference": opponents.metrics()["snapshot"],
            "episode_stage_seconds": episode_timings,
            "native_cpu_gate": {
                "stage_seconds": native_cpu_seconds,
                "estimated_end_to_end_share": min(native_cpu_seconds / elapsed, 1.0),
                "sample_sufficient": decision_count >= NATIVE_CPU_MIN_DECISIONS,
                "eligible": (
                    decision_count >= NATIVE_CPU_MIN_DECISIONS
                    and native_cpu_seconds / elapsed >= NATIVE_CPU_SHARE_GATE
                ),
            },
        }
    finally:
        candidate.close()
        opponents.close()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    rollout_env_values = args.rollout_envs or [1, 2, 4, 8]
    backends = args.backend or ["lockstep"]
    precisions = args.precision or ["auto"]
    compile_modes = args.compile_modes or ["off"]
    decision_budget = 40_000 if args.decisions is None and args.seconds is None else int(args.decisions or 0)
    if (
        min(rollout_env_values) < 1
        or decision_budget < 0
        or (args.seconds is None and decision_budget < 1)
        or (args.seconds is not None and args.seconds <= 0)
        or args.repeats < 1
    ):
        raise ValueError("Rollout environments, decisions, and repeats must be positive")
    configurations = []
    for architecture, checkpoint in args.checkpoint:
        for rollout_envs in rollout_env_values:
            for backend in backends:
                for precision in precisions:
                    for compile_mode in compile_modes:
                        runs = [
                            run_once(
                                checkpoint,
                                device=device,
                                rollout_envs=rollout_envs,
                                backend=backend,
                                precision=precision,
                                compile_mode=compile_mode,
                                wait_seconds=args.batch_wait_ms / 1000.0,
                                decisions=decision_budget,
                                seconds=args.seconds,
                                max_turns=args.max_turns,
                                warmup_turns=args.warmup_turns,
                                seed=args.seed + repeat * 1_000_000,
                            )
                            for repeat in range(args.repeats)
                        ]
                        configurations.append(
                            {
                                "architecture": architecture,
                                "checkpoint": str(checkpoint),
                                "rollout_envs": rollout_envs,
                                "backend": backend,
                                "precision": precision,
                                "compile": compile_mode,
                                "median_decisions_per_second": statistics.median(
                                    float(run["decisions_per_second"]) for run in runs
                                ),
                                "runs": runs,
                            }
                        )
    baseline = float(configurations[0]["median_decisions_per_second"])
    for configuration in configurations:
        configuration["speedup_vs_first"] = float(configuration["median_decisions_per_second"]) / baseline
    result = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "configurations": configurations,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
