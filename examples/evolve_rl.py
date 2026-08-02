from __future__ import annotations

# ruff: noqa: C901, INP001, PLR0912, PLR0913, PLR0915, PLR2004, S311, S607
import argparse
import atexit
import copy
import gzip
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from luxai2021.imitation.agent import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.imitation.data import ReplayBatchSampler
from luxai2021.imitation.distillation import LuxDistillationDataset, compact_distillation_collate
from luxai2021.imitation.model import load_bc_checkpoint
from luxai2021.rl.batched_rollout import (
    ActorCriticBatcher,
    BatchedOpponentPool,
    BehaviorCloningBatcher,
    FirstPlaceBatcher,
)
from luxai2021.rl.evaluation import LeagueMember, acceptance_report, evaluate_against_league
from luxai2021.rl.evolution import (
    CandidateResult,
    CodexCandidateGenerator,
    EvolutionCandidate,
    EvolutionJob,
    EvolutionStore,
    FilesystemJobQueue,
    add_candidate_reflection,
    approximate_ast_distance,
    initial_candidate,
    mutate_candidate,
    select_elites,
)
from luxai2021.rl.job_api import JOB_API_VERSION, JobApiClient, JobApiServer, extract_artifact_directory
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent
from luxai2021.rl.ppo import (
    PPOTrainer,
    apply_reward_program,
    collect_episodes_batched,
    value_head_calibration_loss,
    warmup_value_head,
)
from luxai2021.rl.reward import RewardProgram, calibrate_reward_scale

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from luxai2021.env.agent import Agent

DEFAULT_UNET = "models/distilled/unet_v3/best.pt"
DEFAULT_RESATTN8 = "models/distilled/resattn8_v2_selfplay_ft/best.pt"
DEFAULT_TEACHER = "models/teachers/lux_2021_first_place/062179520_weights.pt"
DEFAULT_TEACHER_CACHE = "models/teachers/lux_2021_first_place/cache"
DEFAULT_PREPARED_CACHE = "models/teachers/lux_2021_first_place/prepared"

_FATAL_CUDA_ERROR_MARKERS = (
    "unspecified launch failure",
    "device-side assert triggered",
    "illegal memory access",
    "device is lost",
    "driver shutting down",
)

_RUN_MANIFEST_ARGUMENTS = (
    "unet_checkpoint",
    "resattn8_checkpoint",
    "teacher_checkpoint",
    "teacher_cache_dir",
    "prepared_cache_dir",
    "seed",
    "islands",
    "initial_per_island",
    "generations",
    "short_seconds",
    "medium_seconds",
    "final_seconds",
    "short_decisions",
    "medium_decisions",
    "final_decisions",
    "decisions_per_update",
    "rollout_envs",
    "medium_count",
    "final_count",
    "episodes_per_update",
    "max_turns",
    "screening_seeds",
    "medium_seeds",
    "final_seeds",
    "screening_seed_start",
    "final_seed_start",
    "bc_batch_size",
    "bc_replays",
    "no_bc_anchor",
    "recover_stale_job_seconds",
    "critic_warmup_episodes",
    "unet_probe_every",
    "resattn8_only",
)


def _is_fatal_cuda_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda" in message and any(marker in message for marker in _FATAL_CUDA_ERROR_MARKERS)


def _synchronize_cuda(device: torch.device, phase: str) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        message = f"CUDA failure during {phase}: {error}"
        raise RuntimeError(message) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex-guided evolutionary RL for distilled Lux policies.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--unet-checkpoint", default=DEFAULT_UNET)
    parser.add_argument("--resattn8-checkpoint", default=DEFAULT_RESATTN8)
    parser.add_argument("--teacher-checkpoint", default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-cache-dir", default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--prepared-cache-dir", default=DEFAULT_PREPARED_CACHE)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--initial-per-island", type=int, default=2)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--short-seconds", type=int, default=20 * 60)
    parser.add_argument("--medium-seconds", type=int, default=90 * 60)
    parser.add_argument("--final-seconds", type=int, default=6 * 60 * 60)
    parser.add_argument("--short-decisions", type=int, default=550_000)
    parser.add_argument("--medium-decisions", type=int, default=1_925_000)
    parser.add_argument("--final-decisions", type=int, default=9_900_000)
    parser.add_argument("--decisions-per-update", type=int, default=40_000)
    parser.add_argument("--rollout-envs", type=int, default=4)
    parser.add_argument("--medium-count", type=int, default=8)
    parser.add_argument("--final-count", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=360)
    parser.add_argument("--screening-seeds", type=int, default=4)
    parser.add_argument("--medium-seeds", type=int, default=8)
    parser.add_argument("--final-seeds", type=int, default=50)
    parser.add_argument("--screening-seed-start", type=int, default=100_000)
    parser.add_argument("--final-seed-start", type=int, default=200_000)
    parser.add_argument("--bc-batch-size", type=int, default=8)
    parser.add_argument("--bc-replays", type=int, default=32)
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--no-codex", action="store_true")
    parser.add_argument("--no-bc-anchor", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id")
    parser.add_argument("--worker-idle-seconds", type=float, default=300.0)
    parser.add_argument("--job-poll-seconds", type=float, default=2.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=24 * 60 * 60)
    parser.add_argument("--recover-stale-job-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--coordinator-only", action="store_true")
    parser.add_argument("--job-api-listen", help="Coordinator API listen address, for example 127.0.0.1:8765")
    parser.add_argument("--job-api-url", help="Worker API URL reached through SSH, for example http://127.0.0.1:18765")
    parser.add_argument("--job-api-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--job-heartbeat-seconds", type=float, default=600.0)
    parser.add_argument("--critic-warmup-episodes", type=int, default=8)
    parser.add_argument("--unet-probe-every", type=int, default=2)
    parser.add_argument(
        "--resattn8-only",
        action="store_true",
        help="Disable all UNet training, probing, opponents, and final evaluation.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _active_base_names(args: argparse.Namespace) -> tuple[str, ...]:
    return ("resattn8",) if getattr(args, "resattn8_only", False) else ("unet", "resattn8")


def _evaluation_anchors(args: argparse.Namespace) -> list[LeagueMember]:
    anchors = []
    if not getattr(args, "resattn8_only", False):
        anchors.append(LeagueMember("unet-base", Path(args.unet_checkpoint)))
    anchors.extend(
        [
            LeagueMember("resattn8-base", Path(args.resattn8_checkpoint)),
            LeagueMember("first-place", Path(args.teacher_checkpoint), "first-place"),
        ]
    )
    return anchors


def _checkpoint_pair(args: argparse.Namespace, base_name: str) -> tuple[Path, Path]:
    if base_name not in _active_base_names(args):
        message = f"Architecture {base_name!r} is disabled for this run"
        raise ValueError(message)
    resattn8_checkpoint = Path(args.resattn8_checkpoint)
    if base_name == "resattn8":
        other = resattn8_checkpoint if args.resattn8_only else Path(args.unet_checkpoint)
        return resattn8_checkpoint, other
    return Path(args.unet_checkpoint), resattn8_checkpoint


def git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def build_anchor_provider(
    base_checkpoint: Path,
    *,
    teacher_cache_dir: Path,
    prepared_cache_dir: Path,
    batch_size: int,
    replay_count: int,
    seed: int,
) -> Callable[[], Mapping[str, torch.Tensor]]:
    _, checkpoint = load_bc_checkpoint(str(base_checkpoint), "cpu")
    replay_paths = [Path(path) for path in checkpoint["split"]["train"][:replay_count]]
    dataset = LuxDistillationDataset(
        replay_paths,
        teacher_cache_dir,
        winner_weight=1.0,
        seed=seed,
        max_turns=64,
        prepared_cache_dir=prepared_cache_dir,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=ReplayBatchSampler(dataset, batch_size, shuffle=True, seed=seed),
        collate_fn=compact_distillation_collate,
        num_workers=0,
        pin_memory=True,
    )
    iterator: Iterator[Mapping[str, torch.Tensor]] = iter(loader)

    def next_batch() -> Mapping[str, torch.Tensor]:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    return next_batch


def opponent_factories(
    *,
    base_checkpoint: Path,
    other_checkpoint: Path,
    teacher_checkpoint: Path,
    snapshot: FullTurnActorCritic,
    device: torch.device,
) -> dict[str, tuple[str, Callable[[], Agent]]]:
    device_name = str(device)
    return {
        "self_base": (
            base_checkpoint.parent.name,
            lambda: BehaviorCloningAgent(str(base_checkpoint), device=device_name, tta="none"),
        ),
        "other_base": (
            other_checkpoint.parent.name,
            lambda: BehaviorCloningAgent(str(other_checkpoint), device=device_name, tta="none"),
        ),
        "teacher": (
            "first-place",
            lambda: FirstPlaceAgent(str(teacher_checkpoint), device=device_name, tta="none"),
        ),
        "snapshot": (
            "initial-snapshot",
            lambda: RolloutAgent(snapshot, device=device, deterministic=True),
        ),
    }


def batched_opponent_pool(
    *,
    base_checkpoint: Path,
    other_checkpoint: Path,
    teacher_checkpoint: Path,
    snapshot: FullTurnActorCritic,
    device: torch.device,
    rollout_envs: int,
) -> tuple[BatchedOpponentPool, dict[str, tuple[str, Callable[[], Agent]]]]:
    device_name = str(device)
    self_base = BehaviorCloningBatcher(
        BehaviorCloningAgent(str(base_checkpoint), device=device_name, tta="none"),
        rollout_envs,
        name="lux-self-base-inference",
    )
    other_base = BehaviorCloningBatcher(
        BehaviorCloningAgent(str(other_checkpoint), device=device_name, tta="none"),
        rollout_envs,
        name="lux-other-base-inference",
    )
    teacher = FirstPlaceBatcher(
        FirstPlaceAgent(str(teacher_checkpoint), device=device_name, tta="none"),
        rollout_envs,
        name="lux-teacher-inference",
    )
    snapshot_backend = ActorCriticBatcher(snapshot, device, rollout_envs, name="lux-snapshot-inference")
    resources = BatchedOpponentPool(
        {
            "self_base": self_base,
            "other_base": other_base,
            "teacher": teacher,
            "snapshot": snapshot_backend,
        }
    )
    factories = {
        "self_base": (base_checkpoint.parent.name, resources.factory("self_base")),
        "other_base": (other_checkpoint.parent.name, resources.factory("other_base")),
        "teacher": ("first-place", resources.factory("teacher")),
        "snapshot": ("initial-snapshot", resources.factory("snapshot")),
    }
    return resources, factories


def train_candidate(
    candidate: EvolutionCandidate,
    *,
    base_name: str,
    base_checkpoint: Path,
    other_checkpoint: Path,
    teacher_checkpoint: Path,
    teacher_cache_dir: Path,
    prepared_cache_dir: Path,
    output_dir: Path,
    seconds: int,
    decision_budget: int | None,
    decisions_per_update: int,
    rollout_envs: int,
    episodes_per_update: int,
    bc_batch_size: int,
    bc_replays: int,
    use_bc_anchor: bool,
    device: torch.device,
    seed: int,
    max_turns: int,
    inherit_from: Path | None = None,
    parent_reward_program: RewardProgram | None = None,
    parent_effective_scale: float | None = None,
    critic_warmup_episodes: int = 8,
    constraint_decay_decisions: int | None = None,
    resume_from: Path | None = None,
    resume_budget_progress: bool = True,
    checkpoint_callback: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> tuple[Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_critic = FullTurnActorCritic.from_checkpoint(base_checkpoint, device)
    inherited_modules: list[str] = []
    inherited_hash = None
    if inherit_from is not None and resume_from is None:
        inherited = torch.load(inherit_from, map_location=device, weights_only=False)
        actor_critic.policy.load_state_dict(inherited["policy"])
        inherited_modules.append("policy")
        if candidate.inheritance_mode == "policy_value":
            actor_critic.value_head.load_state_dict(inherited["value_head"])
            inherited_modules.append("value_head")
        digest = hashlib.sha256()
        with inherit_from.open("rb") as checkpoint_input:
            while chunk := checkpoint_input.read(1024 * 1024):
                digest.update(chunk)
        inherited_hash = digest.hexdigest()
    if device.type == "cuda":
        actor_critic.policy.to(memory_format=torch.channels_last)
    reference_policy = copy.deepcopy(actor_critic.policy).to(device)
    parent_value_reference = (
        {name: value.detach().clone() for name, value in actor_critic.value_head.named_parameters()}
        if inherit_from is not None and candidate.inheritance_mode == "policy_value"
        else None
    )
    bc_provider = None
    if use_bc_anchor and candidate.ppo_config.bc_coefficient > 0:
        bc_provider = build_anchor_provider(
            base_checkpoint,
            teacher_cache_dir=teacher_cache_dir,
            prepared_cache_dir=prepared_cache_dir,
            batch_size=bc_batch_size,
            replay_count=bc_replays,
            seed=seed,
        )
    actor_critic.eval()
    start_update = 0
    cumulative_decisions = 0
    cumulative_turns = 0
    episode_index = 0
    previous_elapsed_seconds = 0.0
    resumed_metrics: dict[str, object] | None = None
    rng = random.Random(seed)
    constraint_progress = 0
    joint_update = 0
    resume_metadata: dict[str, object] | None = None

    def make_trainer() -> PPOTrainer:
        return PPOTrainer(
            actor_critic,
            reference_policy,
            candidate.ppo_config,
            device,
            bc_batch_provider=bc_provider,
            parameter_constraint_coefficient=(
                candidate.parameter_constraint_coefficient if inherit_from is not None else 0.0
            ),
            parameter_constraint_decay_decisions=max(
                1,
                constraint_decay_decisions or (decision_budget or decisions_per_update) // 2,
            ),
            constrain_value_head=candidate.inheritance_mode == "policy_value",
            value_parameter_reference=parent_value_reference,
        )

    trainer = make_trainer()
    if resume_from is not None:
        resume_state = trainer.load_training_state(
            resume_from,
            source_checkpoint=str(base_checkpoint),
            reward_program=candidate.reward_program,
            legacy_target_decisions=decision_budget,
            legacy_stage_seconds=seconds,
            allow_compatible_source_checkpoint=True,
        )
        resume_metadata = {
            "checkpoint": str(resume_from),
            "stored_source_checkpoint": resume_state.source_checkpoint,
            "requested_source_checkpoint": str(base_checkpoint),
            "source_checkpoint_mismatch": resume_state.source_checkpoint_mismatch,
            "budget_progress_resumed": resume_budget_progress,
        }
        start_update = resume_state.next_update
        resumed_metrics = resume_state.metrics
        constraint_progress = resume_state.constraint_progress
        joint_update = resume_state.joint_update
        if resume_budget_progress:
            cumulative_decisions = resume_state.cumulative_decisions
            cumulative_turns = resume_state.cumulative_turns
            episode_index = resume_state.cumulative_episodes
            previous_elapsed_seconds = resume_state.elapsed_seconds
        if resume_budget_progress and resume_state.python_random_state is not None:
            rng.setstate(resume_state.python_random_state)
    snapshot = copy.deepcopy(actor_critic).eval().requires_grad_(requires_grad=False)
    candidate_batcher = ActorCriticBatcher(
        actor_critic,
        device,
        rollout_envs,
        name="lux-candidate-inference",
    )
    opponent_resources, pool = batched_opponent_pool(
        base_checkpoint=base_checkpoint,
        other_checkpoint=other_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        snapshot=snapshot,
        device=device,
        rollout_envs=rollout_envs,
    )
    effective_reward_program = candidate.reward_program
    reward_calibration: dict[str, object] = {
        "proposed_scale": candidate.reward_program.reward_scale,
        "effective_scale": candidate.reward_program.reward_scale,
        "fallback": True,
        "reason": "initial_or_resume",
    }
    critic_warmup: dict[str, float] | None = None
    critic_calibration: dict[str, float | bool] | None = None
    if resume_from is not None and isinstance(resumed_metrics, dict):
        resumed_scale = float(resumed_metrics.get("effective_reward_scale", candidate.reward_program.reward_scale))
        effective_reward_program = RewardProgram(
            candidate.reward_program.components,
            candidate.reward_program.derived_metrics,
            reward_scale=resumed_scale,
            gamma=candidate.reward_program.gamma,
            version=candidate.reward_program.version,
        )
    elif resume_from is None:
        calibration_specs = []
        for offset, opponent_key in enumerate(("self_base", "other_base", "teacher", "snapshot")):
            opponent_name, factory = pool[opponent_key]
            calibration_specs.append((factory, seed + 9_000_000 + offset, opponent_name))
        calibration_episodes = collect_episodes_batched(
            actor_critic,
            calibration_specs,
            candidate.reward_program,
            device=device,
            inference_backend=candidate_batcher.submit,
            max_turns=max_turns,
        )
        if parent_reward_program is not None:
            transitions = [
                (
                    record.metrics,
                    episode.records[index + 1].metrics if index + 1 < len(episode.records) else episode.final_metrics,
                )
                for episode in calibration_episodes
                for index, record in enumerate(episode.records)
            ]
            effective_reward_program, reward_calibration = calibrate_reward_scale(
                parent_reward_program,
                candidate.reward_program,
                transitions,
                parent_effective_scale=parent_effective_scale,
            )
        for episode in calibration_episodes:
            apply_reward_program(episode, effective_reward_program)
        needs_warmup = candidate.inheritance_mode != "policy_value"
        if candidate.inheritance_mode == "policy_value":
            critic_calibration = value_head_calibration_loss(
                actor_critic,
                calibration_episodes,
                candidate.ppo_config,
                device,
            )
            needs_warmup = bool(critic_calibration["requires_warmup"])
        if needs_warmup:
            warmup_episodes = list(calibration_episodes)
            warmup_turns = sum(len(episode.records) for episode in warmup_episodes)
            next_key = 0
            while len(warmup_episodes) < critic_warmup_episodes or warmup_turns < 1024:
                opponent_key = ("self_base", "other_base", "teacher", "snapshot")[next_key % 4]
                opponent_name, factory = pool[opponent_key]
                wave = collect_episodes_batched(
                    actor_critic,
                    [(factory, seed + 9_100_000 + next_key, opponent_name)],
                    effective_reward_program,
                    device=device,
                    inference_backend=candidate_batcher.submit,
                    max_turns=max_turns,
                )
                warmup_episodes.extend(wave)
                warmup_turns += sum(len(episode.records) for episode in wave)
                next_key += 1
            critic_warmup = warmup_value_head(actor_critic, warmup_episodes, candidate.ppo_config, device)
            trainer = make_trainer()
    deadline = time.monotonic() + max(0.0, seconds - previous_elapsed_seconds)
    history = []
    diagnostic_events: list[dict[str, object]] = []
    update = start_update
    started_at = time.monotonic()
    try:
        while (
            cumulative_decisions < decision_budget
            if decision_budget is not None
            else time.monotonic() < deadline or update == 0
        ):
            episodes = []
            update_decisions = 0
            target_update_decisions = decisions_per_update if decision_budget is not None else None
            while (
                update_decisions < target_update_decisions
                if target_update_decisions is not None
                else len(episodes) < episodes_per_update
            ):
                wave_size = rollout_envs
                if target_update_decisions is None:
                    wave_size = min(wave_size, episodes_per_update - len(episodes))
                specs = []
                opponent_key = candidate.opponent_mix.choose(rng)
                opponent_name, factory = pool[opponent_key]
                for _ in range(wave_size):
                    episode_seed = seed + episode_index
                    episode_index += 1
                    specs.append((factory, episode_seed, opponent_name))
                wave = collect_episodes_batched(
                    actor_critic,
                    specs,
                    effective_reward_program,
                    device=device,
                    inference_backend=candidate_batcher.submit,
                    max_turns=max_turns,
                )
                episodes.extend(wave)
                update_decisions += sum(len(record.decisions) for episode in wave for record in episode.records)
            _synchronize_cuda(device, "rollout collection")
            actor_critic.train()
            trainer.set_schedule_state(constraint_progress=constraint_progress, joint_update=joint_update)
            metrics = trainer.update(episodes)
            _synchronize_cuda(device, "PPO update")
            actor_critic.eval()
            cumulative_decisions += int(metrics["decisions"])
            constraint_progress += int(metrics["decisions"])
            joint_update += 1
            cumulative_turns += int(metrics["turns"])
            diagnostic_events.extend(event for episode in episodes for event in episode.diagnostic_events)
            metrics.update(
                {
                    "update": update,
                    "elapsed_seconds": previous_elapsed_seconds + time.monotonic() - started_at,
                    "cumulative_decisions": cumulative_decisions,
                    "cumulative_turns": cumulative_turns,
                    "cumulative_episodes": episode_index,
                    "decisions_per_second": cumulative_decisions
                    / max(previous_elapsed_seconds + time.monotonic() - started_at, 1e-6),
                    "candidate_inference": candidate_batcher.metrics(),
                    "opponent_inference": opponent_resources.metrics(),
                    "rollout_envs": rollout_envs,
                    "effective_reward_scale": effective_reward_program.reward_scale,
                    "constraint_progress": constraint_progress,
                    "joint_update": joint_update,
                    "actor_lr_multiplier": trainer.actor_lr_multiplier,
                }
            )
            history.append(metrics)
            trainer.save_training_checkpoint(
                output_dir / "latest_rl.pt",
                source_checkpoint=str(base_checkpoint),
                reward_program=candidate.reward_program,
                update=update,
                metrics=metrics,
                training_state={
                    "cumulative_decisions": cumulative_decisions,
                    "cumulative_turns": cumulative_turns,
                    "cumulative_episodes": episode_index,
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "python_random_state": rng.getstate(),
                    "constraint_progress": constraint_progress,
                    "joint_update": joint_update,
                },
            )
            if checkpoint_callback is not None:
                checkpoint_callback(output_dir, metrics)
            update += 1
            if decision_budget is None and seconds <= 0:
                break
    finally:
        candidate_batcher.close()
        opponent_resources.close()
    final_metrics = history[-1] if history else resumed_metrics
    if final_metrics is None:
        raise RuntimeError("Training completed without a checkpoint or a PPO update")
    _, source = load_bc_checkpoint(str(base_checkpoint), "cpu")
    summary = {
        "candidate_id": candidate.candidate_id,
        "base_name": base_name,
        "base_checkpoint": str(base_checkpoint),
        "reward_program": candidate.reward_program.to_dict(),
        "effective_reward_program": effective_reward_program.to_dict(),
        "reward_calibration": reward_calibration,
        "critic_warmup": critic_warmup,
        "critic_calibration": critic_calibration,
        "inheritance": {
            "mode": candidate.inheritance_mode,
            "checkpoint": str(inherit_from) if inherit_from is not None else None,
            "checkpoint_sha256": inherited_hash,
            "modules": inherited_modules,
        },
        "resume": resume_metadata,
        "ppo_config": asdict(candidate.ppo_config),
        "opponent_mix": asdict(candidate.opponent_mix),
        "decision_budget": decision_budget,
        "decisions_per_update": decisions_per_update,
        "rollout_envs": rollout_envs,
        "history": history,
        "diagnostic_events": diagnostic_events,
    }
    actor_critic.export_policy(
        output_dir / "best.pt",
        epoch=max(0, update - 1),
        metrics={"validation": final_metrics, "ppo": final_metrics},
        split=source["split"],
        metadata=summary,
    )
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir / "best.pt", summary


def _artifact_checkpoint(run_dir: Path, candidate_id: str, base_name: str) -> Path | None:
    stages = (
        (f"final-{base_name}", f"probe-{base_name}", f"medium-{base_name}", f"short-{base_name}")
        if base_name == "unet"
        else (f"final-{base_name}", f"medium-{base_name}", f"short-{base_name}")
    )
    for stage in stages:
        checkpoint = run_dir / "artifacts" / candidate_id / stage / base_name / "latest_rl.pt"
        if checkpoint.exists():
            return checkpoint
    return None


def _resolve_parent_checkpoint(
    run_dir: Path,
    candidate: EvolutionCandidate,
    candidates: Mapping[str, EvolutionCandidate],
    base_name: str,
) -> tuple[Path, EvolutionCandidate] | None:
    if candidate.inheritance_mode == "base" or candidate.mutation_kind == "restart":
        return None
    pending = [candidate.primary_parent_id or (candidate.parent_ids[0] if candidate.parent_ids else None)]
    visited = set()
    while pending:
        candidate_id = pending.pop(0)
        if candidate_id is None or candidate_id in visited:
            continue
        visited.add(candidate_id)
        parent = candidates.get(candidate_id)
        checkpoint = _artifact_checkpoint(run_dir, candidate_id, base_name)
        if checkpoint is not None and parent is not None:
            return checkpoint, parent
        if parent is not None:
            pending.extend((parent.primary_parent_id, *parent.parent_ids))
    if base_name == "unet":
        return None
    message = f"No inherited {base_name} checkpoint for {candidate.candidate_id}"
    raise FileNotFoundError(message)


def _effective_scale_from_artifact(checkpoint: Path, fallback: float) -> float:
    metrics_path = checkpoint.with_name("metrics.json")
    if not metrics_path.exists():
        return fallback
    value = json.loads(metrics_path.read_text(encoding="utf-8"))
    effective = value.get("effective_reward_program", {})
    return float(effective.get("reward_scale", fallback)) if isinstance(effective, dict) else fallback


def candidate_result(
    candidate: EvolutionCandidate,
    *,
    stage: str,
    base_name: str,
    base_checkpoint: Path,
    other_checkpoint: Path,
    args: argparse.Namespace,
    device: torch.device,
    seconds: int,
    decision_budget: int | None,
    eval_seeds: int,
    eval_seed_start: int,
    candidates: Mapping[str, EvolutionCandidate],
    checkpoint_callback: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> CandidateResult:
    started_at = time.monotonic()
    output_dir = Path(args.run_dir) / "artifacts" / candidate.candidate_id / stage / base_name
    current_checkpoint = output_dir / "latest_rl.pt"
    prior_short_checkpoint = (
        Path(args.run_dir) / "artifacts" / candidate.candidate_id / "short-resattn8" / "resattn8" / "latest_rl.pt"
    )
    resume_from = current_checkpoint if current_checkpoint.exists() else None
    predecessor_paths = [prior_short_checkpoint] if stage == "medium-resattn8" else []
    if stage == "final-resattn8":
        predecessor_paths.extend(
            [
                Path(args.run_dir)
                / "artifacts"
                / candidate.candidate_id
                / "medium-resattn8"
                / "resattn8"
                / "latest_rl.pt",
                prior_short_checkpoint,
            ]
        )
    if stage == "final-unet":
        predecessor_paths.append(
            Path(args.run_dir) / "artifacts" / candidate.candidate_id / "probe-unet" / "unet" / "latest_rl.pt"
        )
    if resume_from is None:
        resume_from = next((path for path in predecessor_paths if path.exists()), None)
    inheritance = (
        None
        if resume_from is not None
        else _resolve_parent_checkpoint(Path(args.run_dir), candidate, candidates, base_name)
    )
    inherit_from = inheritance[0] if inheritance is not None else None
    parent = inheritance[1] if inheritance is not None else None
    parent_effective_scale = (
        _effective_scale_from_artifact(inherit_from, parent.reward_program.reward_scale)
        if inherit_from is not None and parent is not None
        else None
    )
    try:
        checkpoint, training = train_candidate(
            candidate,
            base_name=base_name,
            base_checkpoint=base_checkpoint,
            other_checkpoint=other_checkpoint,
            teacher_checkpoint=Path(args.teacher_checkpoint),
            teacher_cache_dir=Path(args.teacher_cache_dir),
            prepared_cache_dir=Path(args.prepared_cache_dir),
            output_dir=output_dir,
            seconds=seconds,
            decision_budget=decision_budget,
            decisions_per_update=args.decisions_per_update,
            rollout_envs=args.rollout_envs,
            episodes_per_update=args.episodes_per_update,
            bc_batch_size=args.bc_batch_size,
            bc_replays=args.bc_replays,
            use_bc_anchor=not args.no_bc_anchor,
            device=device,
            seed=args.seed + candidate.generation * 10_000 + candidate.island * 100,
            max_turns=args.max_turns,
            inherit_from=inherit_from,
            parent_reward_program=parent.reward_program if parent is not None else None,
            parent_effective_scale=parent_effective_scale,
            critic_warmup_episodes=args.critic_warmup_episodes,
            constraint_decay_decisions=max(1, args.short_decisions // 2),
            resume_from=resume_from,
            resume_budget_progress=resume_from == current_checkpoint,
            checkpoint_callback=checkpoint_callback,
        )
        anchors = _evaluation_anchors(args)
        evaluation = evaluate_against_league(
            LeagueMember(f"{candidate.candidate_id}-{base_name}", checkpoint),
            anchors,
            seed_start=eval_seed_start,
            seed_count=eval_seeds,
            device=str(device),
            max_turns=args.max_turns,
        )
        score_rate = float(evaluation["totals"]["score_rate"])
        teacher_games = [
            game for game in evaluation["games"] if game.get("anchor") == "first-place" and "outcome" in game
        ]
        teacher_score_rate = sum((float(game["outcome"]) + 1.0) * 0.5 for game in teacher_games) / len(teacher_games)
        kl = float(training["history"][-1]["kl"])
        diagnostics_path = output_dir / "diagnostics.json.gz"
        with gzip.open(diagnostics_path, "wt", encoding="utf-8") as output:
            json.dump(
                {
                    "training_diagnostic_events": training.get("diagnostic_events", []),
                    "evaluation_games": [
                        {
                            "anchor": game.get("anchor"),
                            "seed": game.get("seed"),
                            "orientation": game.get("orientation"),
                            "candidate_team": game.get("candidate_team"),
                            "candidate_inference_seconds": game.get("candidate_inference_seconds", []),
                            "diagnostic_events": game.get("diagnostic_events", []),
                        }
                        for game in evaluation["games"]
                        if "outcome" in game
                    ],
                },
                output,
                separators=(",", ":"),
            )
        metrics = {
            "training": training,
            "evaluation": evaluation,
            "checkpoint": str(checkpoint),
            "diagnostics_artifact": diagnostics_path.name,
        }
        return CandidateResult(
            candidate.candidate_id,
            stage,
            "completed",
            score_rate,
            teacher_score_rate,
            kl,
            time.monotonic() - started_at,
            metrics,
        )
    except Exception as error:
        if _is_fatal_cuda_error(error):
            raise
        return CandidateResult(
            candidate.candidate_id,
            stage,
            "failed",
            0.0,
            0.0,
            float("inf"),
            time.monotonic() - started_at,
            {
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(limit=20),
                }
            },
            error=f"{type(error).__name__}: {error}",
        )


def execute_evolution_job(
    job: EvolutionJob,
    *,
    args: argparse.Namespace,
    device: torch.device,
    store: EvolutionStore,
    checkpoint_callback: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> CandidateResult:
    candidates = {candidate.candidate_id: candidate for candidate in store.candidates()}
    candidate = candidates[job.candidate_id]
    prior_results = store.results()
    existing = next(
        (result for result in prior_results if result.candidate_id == job.candidate_id and result.stage == job.stage),
        None,
    )
    if existing is not None and existing.status == "completed":
        return existing
    base_checkpoint, other_checkpoint = _checkpoint_pair(args, job.base_name)
    decision_budget = job.decision_budget
    if decision_budget is None:
        if job.seconds <= 0:
            decision_budget = 1
        elif job.stage.startswith("short-"):
            decision_budget = args.short_decisions
        elif job.stage.startswith("medium-"):
            decision_budget = args.medium_decisions
        elif job.stage.startswith("final-"):
            decision_budget = args.final_decisions
    result = candidate_result(
        candidate,
        stage=job.stage,
        base_name=job.base_name,
        base_checkpoint=base_checkpoint,
        other_checkpoint=other_checkpoint,
        args=args,
        device=device,
        seconds=job.seconds,
        decision_budget=decision_budget,
        eval_seeds=job.eval_seeds,
        eval_seed_start=job.eval_seed_start,
        candidates=candidates,
        checkpoint_callback=checkpoint_callback,
    )
    result = add_candidate_reflection(result, candidate, candidates, prior_results)
    store.save_result(result)
    return result


def _apply_coordinator_manifest(args: argparse.Namespace, manifest: Mapping[str, object]) -> None:
    coordinator_args = manifest.get("arguments", {})
    if not isinstance(coordinator_args, dict):
        raise TypeError("Coordinator manifest arguments are invalid")
    for name in _RUN_MANIFEST_ARGUMENTS:
        if name in coordinator_args:
            setattr(args, name, coordinator_args[name])


def _sync_api_claim(
    store: EvolutionStore,
    claim: Mapping[str, object],
    api: JobApiClient | None = None,
) -> tuple[EvolutionJob, str]:
    if int(claim.get("api_version", 0)) not in {1, JOB_API_VERSION}:
        raise ValueError("Coordinator Job API version is incompatible")
    manifest = claim["manifest"]
    if not isinstance(manifest, dict):
        raise TypeError("Coordinator manifest is invalid")
    store.save_manifest(manifest)
    for value in claim.get("candidates", []):
        store.save_candidate(EvolutionCandidate.from_dict(value))
    coordinator_result_keys = set()
    for value in claim.get("results", []):
        result = CandidateResult(**value)
        coordinator_result_keys.add((result.candidate_id, result.stage))
        store.save_result(result)
    candidate = EvolutionCandidate.from_dict(claim["candidate"])
    store.save_candidate(candidate)
    job = EvolutionJob.from_dict(claim["job"])
    if (job.candidate_id, job.stage) not in coordinator_result_keys:
        stale_result = store.result_dir / f"{job.candidate_id}-{job.stage}.json"
        if stale_result.exists():
            local_result = CandidateResult(**json.loads(stale_result.read_text(encoding="utf-8")))
            if local_result.status != "completed":
                stale_result.unlink()
    input_artifact = claim.get("input_artifact")
    if input_artifact is not None:
        if not isinstance(input_artifact, dict):
            raise TypeError("Input artifact descriptor is invalid")
        destination = (
            store.run_dir
            / "artifacts"
            / str(input_artifact.get("candidate_id", job.candidate_id))
            / str(input_artifact["stage"])
            / str(input_artifact["base_name"])
        )
        extract_artifact_directory(str(input_artifact["zip_base64"]), destination)
    for descriptor in claim.get("input_artifacts", []):
        if not isinstance(descriptor, dict) or api is None:
            raise TypeError("Streaming input artifact requires a Job API client")
        destination = (
            store.run_dir
            / "artifacts"
            / str(descriptor["candidate_id"])
            / str(descriptor["stage"])
            / str(descriptor["base_name"])
        )
        api.download_artifact(descriptor, store.run_dir / "artifact-cache", destination)
    return job, str(claim["lease_id"])


def run_worker(args: argparse.Namespace) -> None:
    if args.overwrite_run:
        raise ValueError("Distributed workers cannot overwrite the run directory")
    run_dir = Path(args.run_dir)
    if args.job_api_url is None and not (run_dir / "manifest.json").exists():
        raise ValueError("Worker run directory does not contain a coordinator manifest")
    if (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        _apply_coordinator_manifest(args, manifest)
    device = resolve_device(args.device)
    store = EvolutionStore(run_dir)
    queue = None if args.job_api_url else FilesystemJobQueue(run_dir)
    api = (
        JobApiClient(
            args.job_api_url,
            token=os.environ.get("LUX_EVOLUTION_JOB_TOKEN"),
            timeout_seconds=args.job_api_timeout_seconds,
        )
        if args.job_api_url
        else None
    )
    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    idle_started = time.monotonic()
    idle_announced = False
    last_api_error: OSError | None = None
    while True:
        if api is not None:
            try:
                claim = api.claim(worker_id)
            except OSError as error:
                if last_api_error is None:
                    print(
                        json.dumps(
                            {
                                "worker_id": worker_id,
                                "job_api": args.job_api_url,
                                "status": "waiting",
                                "error": str(error),
                            },
                            sort_keys=True,
                        )
                    )
                last_api_error = error
                if args.worker_idle_seconds > 0 and time.monotonic() - idle_started >= args.worker_idle_seconds:
                    message = f"Job API remained unreachable: {args.job_api_url}"
                    raise ConnectionError(message) from error
                time.sleep(max(0.1, args.job_poll_seconds))
                continue
            if last_api_error is not None:
                print(
                    json.dumps(
                        {"worker_id": worker_id, "job_api": args.job_api_url, "status": "connected"}, sort_keys=True
                    )
                )
                last_api_error = None
            if claim is None:
                claimed = None
            else:
                job, lease_id = _sync_api_claim(store, claim, api)
                _apply_coordinator_manifest(args, claim["manifest"])
                claimed = (job, lease_id)
        else:
            claimed = queue.claim(worker_id)
        if claimed is None:
            if not idle_announced:
                print(
                    json.dumps(
                        {"worker_id": worker_id, "status": "waiting_for_job"},
                        sort_keys=True,
                    )
                )
                idle_announced = True
            if args.worker_idle_seconds > 0 and time.monotonic() - idle_started >= args.worker_idle_seconds:
                print(
                    json.dumps(
                        {"worker_id": worker_id, "status": "idle_timeout", "seconds": args.worker_idle_seconds},
                        sort_keys=True,
                    )
                )
                return
            time.sleep(max(0.1, args.job_poll_seconds))
            continue
        idle_started = time.monotonic()
        idle_announced = False
        job, lease = claimed
        print(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "job_id": job.job_id,
                    "status": "claimed",
                    "training_seconds": job.seconds,
                    "evaluation_seeds": job.eval_seeds,
                },
                sort_keys=True,
            )
        )
        last_checkpoint_upload = time.monotonic()
        current_job = job
        current_lease = lease

        def checkpoint_callback(
            artifact_dir: Path,
            metrics: Mapping[str, object],
            current_lease: str | Path = current_lease,
            current_job: EvolutionJob = current_job,
        ) -> None:
            nonlocal last_checkpoint_upload
            if api is None or time.monotonic() - last_checkpoint_upload < args.job_heartbeat_seconds:
                return
            try:
                api.heartbeat(lease_id=current_lease, job=current_job, artifact_dir=artifact_dir)
            except (OSError, RuntimeError) as error:
                print(
                    json.dumps(
                        {
                            "worker_id": worker_id,
                            "job_id": current_job.job_id,
                            "status": "heartbeat_failed",
                            "error": str(error),
                        },
                        sort_keys=True,
                    )
                )
                return
            last_checkpoint_upload = time.monotonic()
            print(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "job_id": current_job.job_id,
                        "status": "checkpoint_uploaded",
                        "cumulative_decisions": metrics.get("cumulative_decisions"),
                    },
                    sort_keys=True,
                )
            )

        lease_stop = threading.Event()
        stale_seconds = float(args.recover_stale_job_seconds)
        lease_interval = min(60.0, stale_seconds / 3.0) if stale_seconds > 0 else 60.0
        lease_interval = max(0.1, lease_interval)

        def keep_lease_alive(
            stop: threading.Event = lease_stop,
            interval: float = lease_interval,
            active_lease: str | Path = current_lease,
            active_job: EvolutionJob = current_job,
            active_api: JobApiClient | None = api,
            active_queue: FilesystemJobQueue | None = queue,
        ) -> None:
            while not stop.is_set():
                try:
                    if active_api is None:
                        if active_queue is None or not isinstance(active_lease, Path):
                            return
                        active_queue.heartbeat(active_lease)
                    else:
                        active_api.heartbeat(lease_id=str(active_lease), job=active_job)
                except OSError as error:
                    print(
                        json.dumps(
                            {
                                "worker_id": worker_id,
                                "job_id": active_job.job_id,
                                "status": "lease_heartbeat_retry",
                                "error": str(error),
                            },
                            sort_keys=True,
                        )
                    )
                    stop.wait(min(interval, 10.0))
                    continue
                except (RuntimeError, ValueError) as error:
                    print(
                        json.dumps(
                            {
                                "worker_id": worker_id,
                                "job_id": active_job.job_id,
                                "status": "lease_heartbeat_failed",
                                "error": str(error),
                            },
                            sort_keys=True,
                        )
                    )
                    return
                stop.wait(interval)

        lease_thread = threading.Thread(
            target=keep_lease_alive,
            name=f"lux-lease-{job.job_id}",
            daemon=True,
        )
        lease_thread.start()
        try:
            result = execute_evolution_job(
                job,
                args=args,
                device=device,
                store=store,
                checkpoint_callback=checkpoint_callback,
            )
        except BaseException:
            lease_stop.set()
            lease_thread.join(timeout=5)
            if api is None:
                queue.release(lease)
            else:
                artifact_dir = run_dir / "artifacts" / job.candidate_id / job.stage / job.base_name
                try:
                    api.heartbeat(lease_id=lease, job=job, artifact_dir=artifact_dir)
                    api.release(lease_id=lease, job=job)
                except (OSError, RuntimeError):
                    pass
            raise
        if api is None:
            queue.complete(lease, result)
        else:
            artifact_dir = run_dir / "artifacts" / job.candidate_id / job.stage / job.base_name
            upload_deadline = time.monotonic() + args.job_timeout_seconds
            while True:
                try:
                    api.complete(lease_id=lease, job=job, result=result, artifact_dir=artifact_dir)
                    break
                except OSError:
                    if time.monotonic() >= upload_deadline:
                        raise
                    time.sleep(max(0.1, args.job_poll_seconds))
        lease_stop.set()
        lease_thread.join(timeout=5)
        print(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "job_id": job.job_id,
                    "status": result.status,
                    "error": result.error,
                },
                sort_keys=True,
            )
        )


def _best_parent(
    candidates: dict[str, EvolutionCandidate],
    results: list[CandidateResult],
    island: int,
) -> EvolutionCandidate:
    island_candidates = {key: value for key, value in candidates.items() if value.island == island}
    elites = select_elites(island_candidates, results, count=1)
    if elites:
        return elites[0]
    return max(island_candidates.values(), key=lambda candidate: candidate.generation)


def _legacy_stage_selection(run_dir: Path, stage: str, count: int) -> list[str]:
    evidence: list[tuple[float, str]] = []
    for directory in ("completed", "pending", "running"):
        for path in (run_dir / "jobs" / directory).glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("stage") != stage:
                continue
            timestamp = float(value.get("completed_at", path.stat().st_mtime))
            evidence.append((timestamp, str(value["candidate_id"])))
    if not evidence:
        for path in (run_dir / "results").glob(f"*-{stage}.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            evidence.append((path.stat().st_mtime, str(value["candidate_id"])))
    selected = []
    for _, candidate_id in sorted(evidence):
        if candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) >= count:
            break
    return selected


def _select_completed_stage(
    candidates: Mapping[str, EvolutionCandidate],
    results: list[CandidateResult],
    *,
    stage: str,
    count: int,
) -> list[str]:
    ranked = sorted(
        (
            result
            for result in results
            if result.stage == stage and result.status == "completed" and result.candidate_id in candidates
        ),
        key=lambda result: result.fitness,
        reverse=True,
    )
    return [result.candidate_id for result in ranked[:count]]


def _load_or_create_stage_selection(
    store: EvolutionStore,
    candidates: Mapping[str, EvolutionCandidate],
    results: list[CandidateResult],
    *,
    name: str,
    target_stage: str,
    source_stage: str,
    count: int,
) -> list[EvolutionCandidate]:
    path = store.run_dir / "selections" / f"{name}.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        candidate_ids = [str(candidate_id) for candidate_id in value.get("candidate_ids", [])]
    else:
        candidate_ids = _legacy_stage_selection(store.run_dir, target_stage, count)
        source = "legacy_jobs" if candidate_ids else source_stage
        if not candidate_ids:
            candidate_ids = _select_completed_stage(candidates, results, stage=source_stage, count=count)
        path.parent.mkdir(parents=True, exist_ok=True)
        EvolutionStore.write_json(
            path,
            {
                "schema_version": 1,
                "name": name,
                "source": source,
                "source_stage": source_stage,
                "target_stage": target_stage,
                "candidate_ids": candidate_ids,
                "created_at": time.time(),
            },
        )
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
    if missing:
        message = f"Selection {name!r} references missing candidates: {missing}"
        raise ValueError(message)
    return [candidates[candidate_id] for candidate_id in candidate_ids]


def _archive_unselected_stage_jobs(run_dir: Path, stage: str, selected_ids: set[str]) -> list[Path]:
    archived = []
    destination_dir = run_dir / "jobs" / "obsolete"
    for directory in (run_dir / "jobs" / "pending", run_dir / "jobs" / "running"):
        for path in sorted(directory.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("stage") != stage or value.get("candidate_id") in selected_ids:
                continue
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = _unused_history_path(destination_dir, path.name)
            path.replace(destination)
            archived.append(destination)
    return archived


def _archive_failed_results(store: EvolutionStore, jobs: list[EvolutionJob]) -> None:
    retry_keys = {(job.candidate_id, job.stage) for job in jobs}
    history_dir = store.result_dir / "failed_history"
    for path in sorted(store.result_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        key = (str(value.get("candidate_id")), str(value.get("stage")))
        if key not in retry_keys or value.get("status") == "completed":
            continue
        history_dir.mkdir(parents=True, exist_ok=True)
        destination = _unused_history_path(history_dir, path.name)
        path.replace(destination)


def _unused_history_path(directory: Path, filename: str) -> Path:
    timestamp = int(time.time())
    destination = directory / f"{timestamp}-{filename}"
    suffix = 1
    while destination.exists():
        destination = directory / f"{timestamp}-{suffix}-{filename}"
        suffix += 1
    return destination


def _archive_stale_summary(run_dir: Path) -> None:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return
    history = run_dir / "summaries"
    history.mkdir(parents=True, exist_ok=True)
    summary.replace(_unused_history_path(history, "summary.json"))


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    if args.job_api_url:
        raise ValueError("--job-api-url is only valid with --worker")
    run_dir = Path(args.run_dir)
    if args.overwrite_run and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() and not args.overwrite_run
        else None
    )
    if existing_manifest is not None:
        _apply_coordinator_manifest(args, existing_manifest)
    if args.islands < 1 or args.initial_per_island < 1 or args.generations < 0:
        raise ValueError("Population sizes must be positive")
    if args.rollout_envs < 1 or args.decisions_per_update < 1:
        raise ValueError("Rollout environment and decision budgets must be positive")
    repository = Path(__file__).resolve().parents[1]
    device = resolve_device(args.device)
    store = EvolutionStore(run_dir)
    if existing_manifest is None:
        manifest = {
            "schema_version": 1,
            "created_at": time.time(),
            "git_revision": git_revision(repository),
            "arguments": vars(args),
            "device": str(device),
            "codex_available": shutil.which(args.codex_executable) is not None,
            "bases": {name: getattr(args, f"{name}_checkpoint") for name in _active_base_names(args)},
        }
        store.save_manifest(manifest)
    else:
        manifest = existing_manifest
    generator = None
    if not args.no_codex and not args.dry_run:
        generator = CodexCandidateGenerator(
            repository=repository,
            run_dir=run_dir,
            executable=args.codex_executable,
            model=args.codex_model,
            timeout_seconds=args.codex_timeout,
        )
    candidates = {candidate.candidate_id: candidate for candidate in store.candidates()}
    results = store.results()
    queue = FilesystemJobQueue(run_dir) if args.distributed or args.job_api_listen else None
    job_api_server = None
    if args.job_api_listen and not args.dry_run:
        if queue is None:
            raise RuntimeError("Job API requires a coordinator queue")
        job_api_server = JobApiServer(
            args.job_api_listen,
            run_dir=run_dir,
            queue=queue,
            token=os.environ.get("LUX_EVOLUTION_JOB_TOKEN"),
        )
        job_api_server.start()
        atexit.register(job_api_server.close)
        host, port = job_api_server.address
        print(json.dumps({"job_api": f"http://{host}:{port}"}, sort_keys=True))

    def register(candidate: EvolutionCandidate) -> None:
        candidates[candidate.candidate_id] = candidate
        store.save_candidate(candidate)

    def refresh_results() -> None:
        latest = {(result.candidate_id, result.stage): result for result in results}
        for stored in store.results():
            key = (stored.candidate_id, stored.stage)
            latest[key] = stored
        results[:] = list(latest.values())

    def evaluate_jobs(jobs: list[EvolutionJob]) -> None:
        if args.dry_run or not jobs:
            return
        refresh_results()
        completed_keys = {(result.candidate_id, result.stage) for result in results if result.status == "completed"}
        jobs = [job for job in jobs if (job.candidate_id, job.stage) not in completed_keys]
        if not jobs:
            return
        _archive_stale_summary(run_dir)
        _archive_failed_results(store, jobs)
        retry_keys = {(job.candidate_id, job.stage) for job in jobs}
        results[:] = [
            result
            for result in results
            if (result.candidate_id, result.stage) not in retry_keys or result.status == "completed"
        ]
        if queue is None:
            for job in jobs:
                result = execute_evolution_job(job, args=args, device=device, store=store)
                results.append(result)
                print(json.dumps(asdict(result), sort_keys=True))
                if result.status != "completed":
                    message = f"Evolution job failed: {job.job_id}: {result.error}"
                    raise RuntimeError(message)
            return
        for job in jobs:
            queue.enqueue(job)
            print(json.dumps({"job_id": job.job_id, "status": "enqueued"}, sort_keys=True))
        expected = {(job.candidate_id, job.stage) for job in jobs}
        deadline = time.monotonic() + args.job_timeout_seconds
        worker_id = f"coordinator-{socket.gethostname()}-{os.getpid()}"
        while True:
            refresh_results()
            completed = {(result.candidate_id, result.stage) for result in results if result.status == "completed"}
            if expected <= completed:
                return
            failed = [
                result
                for result in results
                if (result.candidate_id, result.stage) in expected and result.status != "completed"
            ]
            if failed:
                detail = ", ".join(f"{result.candidate_id}--{result.stage}: {result.error}" for result in failed)
                message = f"Evolution jobs failed; fix the cause and resume: {detail}"
                raise RuntimeError(message)
            queue.recover_stale(args.recover_stale_job_seconds)
            if time.monotonic() >= deadline:
                missing = sorted(expected - completed)
                message = f"Timed out waiting for distributed jobs: {missing}"
                raise TimeoutError(message)
            claimed = None if args.coordinator_only else queue.claim(worker_id)
            if claimed is not None:
                job, claimed_path = claimed
                try:
                    result = execute_evolution_job(job, args=args, device=device, store=store)
                except BaseException:
                    queue.release(claimed_path)
                    raise
                queue.complete(claimed_path, result)
                refresh_results()
                print(json.dumps(asdict(result), sort_keys=True))
                continue
            time.sleep(max(0.1, args.job_poll_seconds))

    def short_job(candidate: EvolutionCandidate) -> EvolutionJob:
        return EvolutionJob(
            candidate.candidate_id,
            "short-resattn8",
            "resattn8",
            args.short_seconds,
            args.screening_seeds,
            args.screening_seed_start,
            1 if args.short_seconds <= 0 else args.short_decisions,
        )

    island_parents: dict[int, EvolutionCandidate] = {}
    initial_by_island: dict[int, list[EvolutionCandidate]] = {}
    base_wave = []
    for island in range(args.islands):
        existing_initial = sorted(
            (
                candidate
                for candidate in candidates.values()
                if candidate.island == island and candidate.generation == 0
            ),
            key=lambda candidate: (bool(candidate.parent_ids), candidate.candidate_id),
        )
        initial_by_island[island] = existing_initial
        if existing_initial:
            base = existing_initial[0]
        else:
            base = initial_candidate(island=island, seed=args.seed)
            register(base)
        island_parents[island] = base
        base_wave.append(short_job(base))
    evaluate_jobs(base_wave)

    for initial_index in range(1, args.initial_per_island):
        wave = []
        for island in range(args.islands):
            existing_initial = initial_by_island[island]
            parent = island_parents[island]
            if initial_index < len(existing_initial):
                child = existing_initial[initial_index]
            elif generator is not None:
                try:
                    child = generator.generate([parent], results, generation=0, island=island)
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                    print(f"Codex fallback for island {island}: {error}")
                    child = mutate_candidate(
                        parent,
                        generation=0,
                        island=island,
                        seed=args.seed + island * 100 + initial_index,
                    )
            else:
                child = mutate_candidate(
                    parent,
                    generation=0,
                    island=island,
                    seed=args.seed + island * 100 + initial_index,
                )
            register(child)
            wave.append(short_job(child))
            island_parents[island] = child
        evaluate_jobs(wave)

    for generation in range(1, args.generations + 1):
        global_elites = select_elites(candidates, results, count=max(4, args.islands * 2))
        champion = global_elites[0] if global_elites else None
        best_by_generation = {
            prior_generation: max(
                (
                    result.score_rate
                    for result in results
                    if result.status == "completed"
                    and candidates.get(result.candidate_id) is not None
                    and candidates[result.candidate_id].generation == prior_generation
                ),
                default=float("-inf"),
            )
            for prior_generation in range(generation)
        }
        earlier_best = max(
            (score for prior_generation, score in best_by_generation.items() if prior_generation < generation - 2),
            default=float("-inf"),
        )
        recent_best = max(
            (score for prior_generation, score in best_by_generation.items() if prior_generation >= generation - 2),
            default=float("-inf"),
        )
        stagnated = generation >= 3 and recent_best <= earlier_best
        wave = []
        for island in range(args.islands):
            existing_generation = [
                candidate
                for candidate in candidates.values()
                if candidate.island == island and candidate.generation == generation
            ]
            if existing_generation:
                child = existing_generation[0]
            else:
                if island in {0, 1} and champion is not None:
                    parent = champion
                elif island == 2 and champion is not None and len(global_elites) > 1:
                    parent = max(global_elites[1:], key=lambda item: approximate_ast_distance(champion, item))
                else:
                    parent = _best_parent(candidates, results, island)
                parents = [parent]
                if island == 3:
                    parents.extend(elite for elite in global_elites if elite.candidate_id != parent.candidate_id)
                    parents = parents[:3]
                if generator is not None:
                    try:
                        child = generator.generate(parents, results, generation=generation, island=island)
                    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                        print(f"Codex fallback for generation {generation} island {island}: {error}")
                        child = mutate_candidate(
                            parent,
                            generation=generation,
                            island=island,
                            seed=args.seed + generation * 1000 + island,
                            secondary_parents=tuple(parents[1:]),
                            stagnated=stagnated,
                        )
                else:
                    child = mutate_candidate(
                        parent,
                        generation=generation,
                        island=island,
                        seed=args.seed + generation * 1000 + island,
                        secondary_parents=tuple(parents[1:]),
                        stagnated=stagnated,
                    )
                register(child)
            wave.append(short_job(child))
        evaluate_jobs(wave)
        if not args.resattn8_only and args.unet_probe_every > 0 and generation % args.unet_probe_every == 0:
            refreshed_elites = select_elites(candidates, results, count=1)
            if refreshed_elites:
                probe = refreshed_elites[0]
                evaluate_jobs(
                    [
                        EvolutionJob(
                            probe.candidate_id,
                            "probe-unet",
                            "unet",
                            args.short_seconds,
                            args.screening_seeds,
                            args.screening_seed_start + 50_000 + generation * 100,
                            args.short_decisions,
                        )
                    ]
                )

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "candidates": len(candidates), "dry_run": True}))
        return

    medium = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="medium",
        target_stage="medium-resattn8",
        source_stage="short-resattn8",
        count=args.medium_count,
    )
    _archive_unselected_stage_jobs(
        run_dir,
        "medium-resattn8",
        {candidate.candidate_id for candidate in medium},
    )
    evaluate_jobs(
        [
            EvolutionJob(
                candidate.candidate_id,
                "medium-resattn8",
                "resattn8",
                max(0, args.medium_seconds - args.short_seconds),
                args.medium_seeds,
                args.screening_seed_start + 10_000,
                args.medium_decisions,
            )
            for candidate in medium
        ]
    )

    finalists = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="final",
        target_stage="final-resattn8",
        source_stage="medium-resattn8",
        count=args.final_count,
    )
    for base_name in _active_base_names(args):
        _archive_unselected_stage_jobs(
            run_dir,
            f"final-{base_name}",
            {candidate.candidate_id for candidate in finalists},
        )
    final_jobs = [
        EvolutionJob(
            candidate.candidate_id,
            f"final-{base_name}",
            base_name,
            args.final_seconds,
            args.final_seeds,
            args.final_seed_start,
            args.final_decisions,
        )
        for candidate in finalists
        for base_name in _active_base_names(args)
    ]
    evaluate_jobs(final_jobs)
    anchors = _evaluation_anchors(args)
    baseline_dir = run_dir / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_evaluations = {}
    for base_name, checkpoint in (
        ((name, Path(getattr(args, f"{name}_checkpoint"))) for name in _active_base_names(args)) if finalists else ()
    ):
        baseline_path = baseline_dir / f"final-{base_name}.json"
        if baseline_path.exists():
            baseline_evaluations[base_name] = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            baseline_evaluations[base_name] = evaluate_against_league(
                LeagueMember(f"{base_name}-baseline-eval", checkpoint),
                anchors,
                seed_start=args.final_seed_start,
                seed_count=args.final_seeds,
                device=str(device),
                max_turns=args.max_turns,
            )
            store.write_json(baseline_path, baseline_evaluations[base_name])
    final_keys = {(job.candidate_id, job.stage) for job in final_jobs}
    final_results = [result for result in results if (result.candidate_id, result.stage) in final_keys]
    grouped = {
        candidate.candidate_id: [result for result in final_results if result.candidate_id == candidate.candidate_id]
        for candidate in finalists
    }
    ranking_rows = []
    for candidate_id, values in grouped.items():
        if not values or not all(result.status == "completed" for result in values):
            continue
        evaluations = {result.stage.removeprefix("final-"): result.metrics["evaluation"] for result in values}
        acceptance = acceptance_report(evaluations, baseline_evaluations, seed=args.seed)
        illegal_action_count = sum(
            int(result.metrics.get("reflection", {}).get("diagnostics", {}).get("illegal_action_count", 0))
            for result in values
        )
        acceptance["illegal_action_count"] = illegal_action_count
        acceptance["promote"] = bool(acceptance["promote"] and illegal_action_count == 0)
        ranking_rows.append(
            {
                "candidate_id": candidate_id,
                "mean_score_rate": sum(result.score_rate for result in values) / len(values),
                "worst_score_rate": min(result.score_rate for result in values),
                "architectures": {result.stage.removeprefix("final-"): asdict(result) for result in values},
                "acceptance": acceptance,
            }
        )
    ranking = sorted(
        ranking_rows,
        key=lambda row: (row["mean_score_rate"], row["worst_score_rate"]),
        reverse=True,
    )
    promoted = next((row["candidate_id"] for row in ranking if row["acceptance"]["promote"]), None)
    summary = {
        "status": "completed",
        "medium_selection": [candidate.candidate_id for candidate in medium],
        "final_selection": [candidate.candidate_id for candidate in finalists],
        "ranking": ranking,
        "promoted_candidate": promoted,
    }
    store.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if job_api_server is not None:
        job_api_server.close()
        atexit.unregister(job_api_server.close)


if __name__ == "__main__":
    main()
