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
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Sampler

from luxai2021.imitation.agent import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.imitation.data import ReplayBatchSampler
from luxai2021.imitation.distillation import (
    LuxDistillationDataset,
    compact_distillation_collate,
    prepared_distillation_cache_path,
)
from luxai2021.imitation.model import load_bc_checkpoint
from luxai2021.rl.batched_rollout import (
    ActorCriticBatcher,
    BatchedOpponentPool,
    BehaviorCloningBatcher,
    FirstPlaceBatcher,
    configure_rollout_determinism,
    resolve_rollout_precision,
)
from luxai2021.rl.evaluation import (
    LeagueMember,
    acceptance_report,
    evaluate_against_league,
    paired_seed_deltas,
)
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
    lux_s1_rules_context,
    mutate_candidate,
    select_elites,
    training_curriculum,
)
from luxai2021.rl.job_api import JOB_API_VERSION, JobApiClient, JobApiServer, extract_artifact_directory
from luxai2021.rl.notifications import EvolutionNotifier
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent
from luxai2021.rl.ppo import (
    PPOTrainer,
    aggregate_episode_timings,
    apply_reward_program,
    collect_episodes_batched,
    resolve_rollout_backend,
    value_head_calibration_loss,
    warmup_value_head,
)
from luxai2021.rl.reward import RewardProgram, calibrate_reward_scale

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

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
_AUTOMATIC_INFRASTRUCTURE_RETRIES = 2
_METRIC_SCHEMA_VERSION = 2
_RUN_MANIFEST_SCHEMA_VERSION = 4
_NATIVE_CPU_GATE_MIN_DECISIONS = 40_000
_NATIVE_CPU_GATE_SHARE = 0.25
_RETRYABLE_INFRASTRUCTURE_ERROR_MARKERS = (
    "filenotfounderror",
    "no such file or directory",
    "connectionerror",
    "connection reset",
    "connection refused",
    "timed out",
    "cuda error",
    "cuda failure",
    "device is lost",
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
    "rollout_backend",
    "rollout_precision",
    "rollout_compile",
    "rollout_batch_wait_ms",
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
    "bc_anchor_max_turns",
    "bc_anchor_sampling",
    "curriculum_profile",
    "teacher_noninferiority_margin",
    "no_bc_anchor",
    "recover_stale_job_seconds",
    "critic_warmup_episodes",
    "unet_probe_every",
    "resattn8_only",
    "codex_executable",
    "codex_model",
    "codex_timeout",
    "codex_validation_retries",
    "no_codex",
    "allow_codex_fallback",
    "fixed_candidate",
)


def _is_fatal_cuda_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "cuda" in message and any(marker in message for marker in _FATAL_CUDA_ERROR_MARKERS)


def _is_retryable_infrastructure_failure(result: CandidateResult) -> bool:
    if result.status == "completed" or not result.error:
        return False
    message = result.error.lower()
    return any(marker in message for marker in _RETRYABLE_INFRASTRUCTURE_ERROR_MARKERS)


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
    parser.add_argument("--rollout-envs", type=int, default=16)
    parser.add_argument("--rollout-backend", choices=("auto", "lockstep", "threaded"), default="auto")
    parser.add_argument("--rollout-precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--rollout-compile", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--rollout-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--medium-count", type=int, default=8)
    parser.add_argument("--final-count", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=360)
    parser.add_argument("--screening-seeds", type=int, default=12)
    parser.add_argument("--medium-seeds", type=int, default=24)
    parser.add_argument("--final-seeds", type=int, default=100)
    parser.add_argument("--screening-seed-start", type=int, default=100_000)
    parser.add_argument("--final-seed-start", type=int, default=200_000)
    parser.add_argument("--bc-batch-size", type=int, default=8)
    parser.add_argument("--bc-replays", type=int, default=128)
    parser.add_argument("--bc-anchor-max-turns", type=int, default=0)
    parser.add_argument(
        "--bc-anchor-sampling",
        choices=("phase-balanced", "replay"),
        default="phase-balanced",
    )
    parser.add_argument(
        "--curriculum-profile",
        choices=("dense_shaping", "teacher_guarded_near_sparse", "terminal_only_ablation", "legacy"),
        default="dense_shaping",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("potential_linear", "potential_tanh", "direct_step"),
        default=None,
        help="Override candidate reward mode (default: potential_linear from candidate).",
    )
    parser.add_argument("--teacher-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument(
        "--codex-validation-retries",
        type=int,
        default=2,
        help="Retry schema-valid Codex proposals rejected by the mutation validator (default: 2).",
    )
    parser.add_argument("--no-codex", action="store_true")
    parser.add_argument(
        "--fixed-candidate",
        help="Train one immutable candidate proposal through short, medium, and final stages.",
    )
    parser.add_argument(
        "--allow-codex-fallback",
        action="store_true",
        help="Explicitly allow deterministic candidates after a Codex proposal failure.",
    )
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_descriptors(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    paths = {name: Path(getattr(args, f"{name}_checkpoint")) for name in _active_base_names(args)}
    paths["teacher"] = Path(args.teacher_checkpoint)
    descriptors = {}
    for name, path in paths.items():
        if not path.is_file():
            message = f"Required checkpoint does not exist: {path}"
            raise FileNotFoundError(message)
        descriptors[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return descriptors


def _validate_checkpoint_descriptors(args: argparse.Namespace, manifest: Mapping[str, object]) -> None:
    expected = manifest.get("checkpoint_descriptors")
    if expected is None:
        return
    if not isinstance(expected, dict):
        raise TypeError("Run manifest checkpoint descriptors are invalid")
    if not expected:
        return
    actual = _checkpoint_descriptors(args)
    for name, descriptor in expected.items():
        if name not in actual or not isinstance(descriptor, dict):
            message = f"Run manifest checkpoint descriptor is invalid: {name}"
            raise ValueError(message)
        for key in ("path", "size", "sha256"):
            if actual[name][key] != descriptor.get(key):
                message = (
                    f"Checkpoint integrity mismatch for {name}: {key} expected={descriptor.get(key)!r} "
                    f"actual={actual[name][key]!r}"
                )
                raise ValueError(message)


def _candidate_provenance_path(run_dir: Path, candidate_id: str) -> Path:
    return run_dir / "provenance" / f"{candidate_id}.json"


def _load_fixed_candidate(path: Path) -> EvolutionCandidate:
    if not path.is_file():
        message = f"Fixed candidate does not exist: {path}"
        raise FileNotFoundError(message)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Fixed candidate must be a JSON object")
    allowed = {"reward_program", "ppo_config", "opponent_mix", "rationale"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        message = f"Fixed candidate contains unsupported fields: {unknown}"
        raise ValueError(message)
    proposal = {
        **value,
        "mutation_kind": "initial",
        "primary_parent_id": None,
        "secondary_parent_ids": [],
        "inheritance_mode": "base",
        "mutation_manifest": {
            "changed_paths": [],
            "summary": "User-selected immutable fixed candidate",
        },
        "parameter_constraint_coefficient": 0.0,
    }
    return EvolutionCandidate.from_proposal(proposal, generation=0, island=0, parent_ids=())


def _validate_fixed_candidate_descriptor(path: Path, manifest: Mapping[str, object]) -> None:
    expected = manifest.get("fixed_candidate_descriptor")
    if expected is None:
        return
    if not isinstance(expected, dict):
        raise TypeError("Fixed candidate descriptor is invalid")
    actual = {"path": str(path), "size": path.stat().st_size, "sha256": _sha256_file(path)}
    if actual != expected:
        raise ValueError("Fixed candidate changed; preserve this run and start a new run directory")


def _save_candidate_provenance(run_dir: Path, candidate: EvolutionCandidate, value: Mapping[str, object]) -> None:
    path = _candidate_provenance_path(run_dir, candidate.candidate_id)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(value):
            message = f"Candidate provenance changed for {candidate.candidate_id}"
            raise ValueError(message)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    EvolutionStore.write_json(path, value)


def _candidate_provenance(run_dir: Path, candidate_id: str) -> dict[str, object] | None:
    path = _candidate_provenance_path(run_dir, candidate_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _validate_candidate_provenance(
    run_dir: Path,
    candidates: Mapping[str, EvolutionCandidate],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    generation = manifest.get("candidate_generation", {})
    if not isinstance(generation, dict):
        raise TypeError("Run manifest candidate-generation policy is invalid")
    expected_mode = str(generation.get("mode", "legacy"))
    allow_fallback = bool(generation.get("allow_fallback", False))
    counts: dict[str, int] = {}
    errors = []
    for candidate in candidates.values():
        provenance = _candidate_provenance(run_dir, candidate.candidate_id)
        if provenance is None:
            errors.append(f"{candidate.candidate_id}: missing provenance")
            continue
        source = str(provenance.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
        is_initial = candidate.mutation_kind == "initial" and not candidate.parent_ids
        allowed_sources = {"initial"} if is_initial else {expected_mode}
        if expected_mode == "codex" and allow_fallback and not is_initial:
            allowed_sources.add("codex_fallback")
        if source not in allowed_sources:
            errors.append(f"{candidate.candidate_id}: source={source!r}, expected={sorted(allowed_sources)}")
        if source == "codex":
            proposal_name = str(provenance.get("proposal_path", ""))
            metadata_name = str(provenance.get("proposal_metadata", ""))
            if Path(proposal_name).name != proposal_name or Path(metadata_name).name != metadata_name:
                errors.append(f"{candidate.candidate_id}: unsafe Codex provenance path")
                continue
            proposal_path = run_dir / proposal_name
            metadata_path = run_dir / metadata_name
            if not proposal_path.is_file() or not metadata_path.is_file():
                errors.append(f"{candidate.candidate_id}: Codex proposal or metadata file is missing")
                continue
            proposal_sha256 = _sha256_file(proposal_path)
            if proposal_sha256 != provenance.get("proposal_sha256"):
                errors.append(f"{candidate.candidate_id}: Codex proposal SHA-256 changed")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("candidate_id") != candidate.candidate_id
                or metadata.get("proposal_sha256") != proposal_sha256
            ):
                errors.append(f"{candidate.candidate_id}: Codex metadata does not match candidate/proposal")
            if metadata.get("model") != generation.get("model"):
                errors.append(f"{candidate.candidate_id}: Codex model does not match run manifest")
            raw_name = metadata.get("raw_proposal_path")
            prompt_name = metadata.get("prompt_path")
            for artifact_name, hash_key in (
                (raw_name, "raw_proposal_sha256"),
                (prompt_name, None),
            ):
                if artifact_name is None:
                    continue
                if Path(str(artifact_name)).name != artifact_name:
                    errors.append(f"{candidate.candidate_id}: unsafe Codex audit artifact path")
                    continue
                artifact_path = run_dir / str(artifact_name)
                if not artifact_path.is_file():
                    errors.append(f"{candidate.candidate_id}: Codex audit artifact is missing: {artifact_name}")
                elif hash_key and _sha256_file(artifact_path) != metadata.get(hash_key):
                    errors.append(f"{candidate.candidate_id}: Codex raw proposal SHA-256 changed")
            if prompt_name is not None and (run_dir / str(prompt_name)).is_file():
                prompt_bytes = gzip.decompress((run_dir / str(prompt_name)).read_bytes())
                prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
                if len(prompt_bytes) != metadata.get("prompt_bytes") or prompt_hash != metadata.get("prompt_sha256"):
                    errors.append(f"{candidate.candidate_id}: Codex prompt audit does not match metadata")
            if "canonical_proposal_sha256" in metadata and metadata["canonical_proposal_sha256"] != proposal_sha256:
                errors.append(f"{candidate.candidate_id}: Codex canonical proposal SHA-256 changed")
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            reconstructed = EvolutionCandidate.from_proposal(
                proposal,
                generation=candidate.generation,
                island=candidate.island,
                parent_ids=candidate.parent_ids,
            )
            if reconstructed.candidate_id != candidate.candidate_id:
                errors.append(f"{candidate.candidate_id}: Codex proposal no longer reconstructs the candidate")
    return {
        "expected_mode": expected_mode,
        "allow_fallback": allow_fallback,
        "counts": counts,
        "errors": errors,
        "valid": not errors,
        "fully_codex_guided": expected_mode == "codex" and counts.get("codex_fallback", 0) == 0 and not errors,
    }


def _validate_run_kind(manifest: Mapping[str, object], *, dry_run: bool) -> None:
    run_kind = manifest.get("run_kind")
    requested_kind = "dry-run" if dry_run else "training"
    if run_kind is not None and run_kind != requested_kind:
        message = f"Cannot resume a {run_kind!r} run as {requested_kind!r}; use a new --run-dir or --overwrite-run"
        raise ValueError(message)


class PhaseBalancedBatchSampler(Sampler[list[int]]):
    """Sample early/late and day/night turns evenly from prepared replay data."""

    def __init__(self, dataset: LuxDistillationDataset, batch_size: int, *, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("BC anchor batch size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.strata: list[list[int]] = [[] for _ in range(8)]
        for index, (_, turn, _) in enumerate(dataset.samples):
            phase = min(max(int(turn), 0) // 90, 3)
            night = int(turn) % 40 >= 30
            self.strata[phase * 2 + int(night)].append(index)
        if any(not values for values in self.strata):
            missing = [index for index, values in enumerate(self.strata) if not values]
            message = f"BC anchor dataset is missing phase/day-night strata: {missing}"
            raise ValueError(message)
        self.batch_count = (len(dataset) + batch_size - 1) // batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = [values.copy() for values in self.strata]
        for values in pools:
            rng.shuffle(values)
        offsets = [0] * len(pools)
        for batch_index in range(self.batch_count):
            batch = []
            for local_index in range(self.batch_size):
                stratum = (batch_index * self.batch_size + local_index) % len(pools)
                if offsets[stratum] >= len(pools[stratum]):
                    rng.shuffle(pools[stratum])
                    offsets[stratum] = 0
                batch.append(pools[stratum][offsets[stratum]])
                offsets[stratum] += 1
            yield batch

    def __len__(self) -> int:
        return self.batch_count


def build_anchor_provider(
    base_checkpoint: Path,
    *,
    teacher_cache_dir: Path,
    prepared_cache_dir: Path,
    batch_size: int,
    replay_count: int,
    seed: int,
    max_turns: int = 0,
    sampling: str = "phase-balanced",
) -> Callable[[], Mapping[str, torch.Tensor]]:
    _, checkpoint = load_bc_checkpoint(str(base_checkpoint), "cpu")
    train_paths = [Path(path) for path in checkpoint["split"]["train"]]
    rng = random.Random(seed)
    rng.shuffle(train_paths)
    replay_paths = train_paths[:replay_count]
    missing_replays = [path for path in replay_paths if not path.exists()]
    if missing_replays:
        preview = ", ".join(str(path) for path in missing_replays[:8])
        message = f"BC anchor replay files are missing ({len(missing_replays)}): {preview}"
        raise FileNotFoundError(message)
    missing_cache = [
        path
        for path in replay_paths
        if not prepared_distillation_cache_path(path, prepared_cache_dir).exists()
    ]
    if missing_cache:
        preview = ", ".join(str(path) for path in missing_cache[:8])
        replay_root = Path(os.path.commonpath([str(path.parent) for path in replay_paths]))
        message = (
            f"Prepared BC anchor caches are missing for {len(missing_cache)} replay(s): {preview}. "
            "Run: uv run --locked python examples/precompute_distillation_dataset.py "
            f"--replay-dir {replay_root} --teacher-cache-dir {teacher_cache_dir} "
            f"--output-dir {prepared_cache_dir}"
        )
        raise FileNotFoundError(message)
    dataset = LuxDistillationDataset(
        replay_paths,
        teacher_cache_dir,
        winner_weight=1.0,
        seed=seed,
        max_turns=max_turns,
        prepared_cache_dir=prepared_cache_dir,
    )
    if sampling == "phase-balanced":
        batch_sampler: Sampler[list[int]] = PhaseBalancedBatchSampler(dataset, batch_size, seed=seed)
    elif sampling == "replay":
        batch_sampler = ReplayBatchSampler(dataset, batch_size, shuffle=True, seed=seed)
    else:
        message = f"Unknown BC anchor sampling mode: {sampling}"
        raise ValueError(message)
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
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
    rollout_backend: str = "auto",
    rollout_precision: str = "auto",
    rollout_compile: str = "auto",
    rollout_batch_wait_ms: float = 2.0,
) -> tuple[BatchedOpponentPool, dict[str, tuple[str, Callable[[], Agent]]]]:
    device_name = str(device)
    wait_seconds = rollout_batch_wait_ms / 1000.0
    pad_batches = resolve_rollout_backend(rollout_backend) == "lockstep"
    self_base = BehaviorCloningBatcher(
        BehaviorCloningAgent(str(base_checkpoint), device=device_name, tta="none"),
        rollout_envs,
        name="lux-self-base-inference",
        wait_seconds=wait_seconds,
        precision=rollout_precision,
        compile_mode=rollout_compile,
        pad_batches=pad_batches,
    )
    other_base = BehaviorCloningBatcher(
        BehaviorCloningAgent(str(other_checkpoint), device=device_name, tta="none"),
        rollout_envs,
        name="lux-other-base-inference",
        wait_seconds=wait_seconds,
        precision=rollout_precision,
        compile_mode=rollout_compile,
        pad_batches=pad_batches,
    )
    teacher = FirstPlaceBatcher(
        FirstPlaceAgent(str(teacher_checkpoint), device=device_name, tta="none"),
        rollout_envs,
        name="lux-teacher-inference",
        wait_seconds=wait_seconds,
        precision=rollout_precision,
        compile_mode=rollout_compile,
        pad_batches=pad_batches,
    )
    snapshot_backend = ActorCriticBatcher(
        snapshot,
        device,
        rollout_envs,
        name="lux-snapshot-inference",
        wait_seconds=wait_seconds,
        precision=rollout_precision,
        compile_mode=rollout_compile,
        pad_batches=pad_batches,
    )
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
    eval_seeds: int = 0,
    eval_seed_start: int = 0,
    prepared_cache_dir: Path,
    output_dir: Path,
    seconds: int,
    decision_budget: int | None,
    decisions_per_update: int,
    rollout_envs: int,
    episodes_per_update: int,
    bc_batch_size: int,
    bc_replays: int,
    bc_anchor_max_turns: int,
    bc_anchor_sampling: str,
    bc_anchor_seed: int | None,
    use_bc_anchor: bool,
    device: torch.device,
    seed: int,
    max_turns: int,
    curriculum_profile: str = "legacy",
    curriculum_total_decisions: int | None = None,
    curriculum_start_decisions: int = 0,
    inherit_from: Path | None = None,
    parent_reward_program: RewardProgram | None = None,
    parent_effective_scale: float | None = None,
    critic_warmup_episodes: int = 8,
    resume_from: Path | None = None,
    resume_budget_progress: bool = True,
    checkpoint_callback: Callable[[Path, Mapping[str, object]], None] | None = None,
    rollout_backend: str = "auto",
    rollout_precision: str = "auto",
    rollout_compile: str = "auto",
    rollout_batch_wait_ms: float = 2.0,
    reward_mode: str | None = None,
) -> tuple[Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_sha256 = _sha256_file(base_checkpoint)
    _, base_source = load_bc_checkpoint(str(base_checkpoint), "cpu")
    actor_critic = FullTurnActorCritic.from_checkpoint(base_checkpoint, device)
    
    reference_policy = None
    if candidate.ppo_config.kl_coefficient > 0:
        reference_policy = FullTurnActorCritic.from_checkpoint(base_checkpoint, device).policy
        reference_policy.eval()
        for param in reference_policy.parameters():
            param.requires_grad = False
            
    inherited_modules: list[str] = []
    inherited_hash = None
    if inherit_from is not None and resume_from is None:
        inherited = torch.load(inherit_from, map_location=device, weights_only=False)
        if "policy" in inherited:
            inherited_policy = inherited["policy"]
        else:
            inherited_policy, _ = load_bc_checkpoint(str(inherit_from), str(device))
            inherited_policy = inherited_policy.state_dict()
        actor_critic.policy.load_state_dict(inherited_policy)
        inherited_modules.append("policy")
        if candidate.inheritance_mode == "policy_value" and "value_head" in inherited:
            actor_critic.value_head.load_state_dict(inherited["value_head"])
            inherited_modules.append("value_head")
        digest = hashlib.sha256()
        with inherit_from.open("rb") as checkpoint_input:
            while chunk := checkpoint_input.read(1024 * 1024):
                digest.update(chunk)
        inherited_hash = digest.hexdigest()
    if device.type == "cuda":
        actor_critic.policy.to(memory_format=torch.channels_last)
    bc_provider = None
    if use_bc_anchor and candidate.ppo_config.bc_coefficient > 0:
        bc_provider = build_anchor_provider(
            base_checkpoint,
            teacher_cache_dir=teacher_cache_dir,
            prepared_cache_dir=prepared_cache_dir,
            batch_size=bc_batch_size,
            replay_count=bc_replays,
            seed=seed if bc_anchor_seed is None else bc_anchor_seed,
            max_turns=bc_anchor_max_turns,
            sampling=bc_anchor_sampling,
        )
    curriculum = training_curriculum(curriculum_profile)
    curriculum_total = max(1, int(curriculum_total_decisions or decision_budget or decisions_per_update))
    actor_critic.eval()
    start_update = 0
    cumulative_decisions = 0
    cumulative_turns = 0
    episode_index = 0
    previous_elapsed_seconds = 0.0
    resumed_metrics: dict[str, object] | None = None
    rng = random.Random(seed)
    curriculum_progress_decisions = max(0, int(curriculum_start_decisions))
    joint_update = 0
    resume_metadata: dict[str, object] | None = None

    def make_trainer() -> PPOTrainer:
        return PPOTrainer(
            actor_critic,
            candidate.ppo_config,
            device,
            reference_policy=reference_policy,
            bc_batch_provider=bc_provider,
        )

    trainer = make_trainer()
    if resume_from is not None:
        resume_state = trainer.load_training_state(
            resume_from,
            source_checkpoint=str(base_checkpoint),
            source_checkpoint_sha256=base_checkpoint_sha256,
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
            "stored_source_checkpoint_sha256": resume_state.source_checkpoint_sha256,
            "requested_source_checkpoint_sha256": base_checkpoint_sha256,
            "source_checkpoint_sha256_mismatch": resume_state.source_checkpoint_sha256_mismatch,
            "budget_progress_resumed": resume_budget_progress,
        }
        start_update = resume_state.next_update
        resumed_metrics = resume_state.metrics
        curriculum_progress_decisions = max(
            resume_state.curriculum_progress_decisions,
            int(curriculum_start_decisions),
        )
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
        wait_seconds=rollout_batch_wait_ms / 1000.0,
        precision=rollout_precision,
        compile_mode=rollout_compile,
        pad_batches=resolve_rollout_backend(rollout_backend) == "lockstep",
    )
    opponent_resources, pool = batched_opponent_pool(
        base_checkpoint=base_checkpoint,
        other_checkpoint=other_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        snapshot=snapshot,
        device=device,
        rollout_envs=rollout_envs,
        rollout_backend=rollout_backend,
        rollout_precision=rollout_precision,
        rollout_compile=rollout_compile,
        rollout_batch_wait_ms=rollout_batch_wait_ms,
    )
    if reward_mode is not None:
        candidate_reward = RewardProgram(
            candidate.reward_program.components,
            candidate.reward_program.derived_metrics,
            reward_scale=candidate.reward_program.reward_scale,
            gamma=candidate.reward_program.gamma,
            version=candidate.reward_program.version,
            mode=reward_mode,
            terminal_reward_scale=candidate.reward_program.terminal_reward_scale,
            normalize_total=candidate.reward_program.normalize_total,
        )
        candidate = replace(candidate, reward_program=candidate_reward)
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
        resumed_scale = float(
            resumed_metrics.get(
                "calibrated_reward_scale",
                resumed_metrics.get("effective_reward_scale", candidate.reward_program.reward_scale),
            )
        )
        effective_reward_program = RewardProgram(
            candidate.reward_program.components,
            candidate.reward_program.derived_metrics,
            reward_scale=resumed_scale,
            gamma=candidate.reward_program.gamma,
            version=candidate.reward_program.version,
            mode=candidate.reward_program.mode,
            terminal_reward_scale=candidate.reward_program.terminal_reward_scale,
            normalize_total=candidate.reward_program.normalize_total,
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
            rollout_backend=rollout_backend,
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
                    rollout_backend=rollout_backend,
                )
                warmup_episodes.extend(wave)
                warmup_turns += sum(len(episode.records) for episode in wave)
                next_key += 1
            critic_warmup = warmup_value_head(actor_critic, warmup_episodes, candidate.ppo_config, device)
            trainer = make_trainer()
    deadline = time.monotonic() + max(0.0, seconds - previous_elapsed_seconds)
    history = []
    diagnostic_events: list[dict[str, object]] = []
    milestone_dir = output_dir / "milestones"
    milestone_dir.mkdir(exist_ok=True)
    milestone_points = (0.20, 0.40, 0.60, 0.80, 1.00)
    best_teacher_score_rate = -1.0
    epochs_without_improvement = 0
    saved_milestones = {
        point for point in milestone_points if (milestone_dir / f"p{round(point * 100):03d}.pt").exists()
    }
    update = start_update
    started_at = time.monotonic()
    opponent_episode_counts = dict.fromkeys(("self_base", "other_base", "teacher", "snapshot"), 0)
    should_early_stop = False
    try:
        while not should_early_stop and (
            cumulative_decisions < decision_budget
            if decision_budget is not None
            else time.monotonic() < deadline or update == 0
        ):
            candidate_batcher.reset_metrics()
            opponent_resources.reset_metrics()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            rollout_started = time.monotonic()
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
                curriculum_progress = min(max(curriculum_progress_decisions / curriculum_total, 0.0), 1.0)
                active_opponent_mix = curriculum.opponent_mix(candidate.opponent_mix, curriculum_progress)
                shaping_multiplier = curriculum.shaping_multiplier(curriculum_progress)
                bc_coefficient_multiplier = curriculum.bc_coefficient_multiplier(curriculum_progress)
                scheduled_reward_program = RewardProgram(
                    effective_reward_program.components,
                    effective_reward_program.derived_metrics,
                    reward_scale=effective_reward_program.reward_scale * shaping_multiplier,
                    gamma=effective_reward_program.gamma,
                    version=effective_reward_program.version,
                    mode=effective_reward_program.mode,
                    terminal_reward_scale=effective_reward_program.terminal_reward_scale,
                    normalize_total=effective_reward_program.normalize_total,
                )
                allocated_opponents = active_opponent_mix.allocate(wave_size, rng)
                for opponent_key in allocated_opponents:
                    opponent_name, factory = pool[opponent_key]
                    episode_seed = seed + episode_index
                    episode_index += 1
                    specs.append((factory, episode_seed, opponent_name))
                    opponent_episode_counts[opponent_key] += 1
                wave = collect_episodes_batched(
                    actor_critic,
                    specs,
                    scheduled_reward_program,
                    device=device,
                    inference_backend=candidate_batcher.submit,
                    max_turns=max_turns,
                    rollout_backend=rollout_backend,
                )
                episodes.extend(wave)
                update_decisions += sum(len(record.decisions) for episode in wave for record in episode.records)
            _synchronize_cuda(device, "rollout collection")
            rollout_seconds = time.monotonic() - rollout_started
            actor_critic.train()
            trainer.set_schedule_state(joint_update=joint_update, bc_coefficient_multiplier=bc_coefficient_multiplier)
            ppo_update_started = time.monotonic()
            is_first_update = (update == 0)
            is_last_update = (decision_budget is not None and target_update_decisions is not None and cumulative_decisions + target_update_decisions >= decision_budget)
            record_grad_norms = is_first_update or is_last_update
            metrics = trainer.update(episodes, record_grad_norms=record_grad_norms)
            _synchronize_cuda(device, "PPO update")
            ppo_update_seconds = time.monotonic() - ppo_update_started
            actor_critic.eval()
            cumulative_decisions += int(metrics["decisions"])
            curriculum_progress_decisions += int(metrics["decisions"])
            joint_update += 1
            cumulative_turns += int(metrics["turns"])
            diagnostic_events.extend(event for episode in episodes for event in episode.diagnostic_events)
            reward_values = [record.reward for episode in episodes for record in episode.records]
            terminal_values = [episode.outcome for episode in episodes]
            shaping_values = []
            for episode in episodes:
                for record_index, record in enumerate(episode.records):
                    terminal = episode.outcome if record_index + 1 == len(episode.records) else 0.0
                    shaping_values.append(record.reward - terminal)
            rollout_stage_seconds = aggregate_episode_timings(episodes)
            native_cpu_stage_seconds = sum(
                rollout_stage_seconds.get(name, 0.0) for name in ("snapshot", "encode", "game_step", "reward_metrics")
            )
            native_cpu_estimated_share = min(native_cpu_stage_seconds / max(rollout_seconds, 1e-6), 1.0)
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
                    "rollout_backend": resolve_rollout_backend(rollout_backend),
                    "rollout_precision_requested": rollout_precision,
                    "rollout_compile_requested": rollout_compile,
                    "rollout_batch_wait_ms": rollout_batch_wait_ms,
                    "rollout_seconds": rollout_seconds,
                    "ppo_update_seconds": ppo_update_seconds,
                    "rollout_games_per_second": len(episodes) / max(rollout_seconds, 1e-6),
                    "rollout_turns_per_second": float(metrics["turns"]) / max(rollout_seconds, 1e-6),
                    "rollout_decisions_per_second": float(metrics["decisions"]) / max(rollout_seconds, 1e-6),
                    "rollout_stage_seconds": rollout_stage_seconds,
                    "native_cpu_gate": {
                        "stage_seconds": native_cpu_stage_seconds,
                        "estimated_end_to_end_share": native_cpu_estimated_share,
                        "sample_sufficient": int(metrics["decisions"]) >= _NATIVE_CPU_GATE_MIN_DECISIONS,
                        "eligible": (
                            int(metrics["decisions"]) >= _NATIVE_CPU_GATE_MIN_DECISIONS
                            and native_cpu_estimated_share >= _NATIVE_CPU_GATE_SHARE
                        ),
                    },
                    "calibrated_reward_scale": effective_reward_program.reward_scale,
                    "effective_reward_scale": scheduled_reward_program.reward_scale,
                    "reward_shaping_multiplier": shaping_multiplier,
                    "reward_total_abs_mean": sum(abs(value) for value in reward_values) / max(len(reward_values), 1),
                    "reward_terminal_abs_mean": sum(abs(value) for value in terminal_values)
                    / max(len(terminal_values), 1),
                    "reward_shaping_abs_mean": sum(abs(value) for value in shaping_values)
                    / max(len(shaping_values), 1),
                    "curriculum_progress": curriculum_progress,
                    "effective_opponent_mix": asdict(active_opponent_mix),
                    "opponent_episode_counts": dict(opponent_episode_counts),
                    "curriculum_progress_decisions": curriculum_progress_decisions,
                    "joint_update": joint_update,
                    "actor_lr_multiplier": trainer.actor_lr_multiplier,
                    "bc_coefficient_multiplier": bc_coefficient_multiplier,
                }
            )
            history.append(metrics)
            trainer.save_training_checkpoint(
                output_dir / "latest_rl.pt",
                source_checkpoint=str(base_checkpoint),
                source_checkpoint_sha256=base_checkpoint_sha256,
                reward_program=candidate.reward_program,
                update=update,
                metrics=metrics,
                training_state={
                    "cumulative_decisions": cumulative_decisions,
                    "cumulative_turns": cumulative_turns,
                    "cumulative_episodes": episode_index,
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "python_random_state": rng.getstate(),
                    "curriculum_progress_decisions": curriculum_progress_decisions,
                    "joint_update": joint_update,
                },
            )
            completed_progress = min(max(curriculum_progress_decisions / curriculum_total, 0.0), 1.0)
            for milestone in milestone_points:
                if milestone not in saved_milestones and completed_progress >= milestone:
                    actor_critic.export_policy(
                        milestone_dir / f"p{round(milestone * 100):03d}.pt",
                        epoch=update,
                        metrics={"validation": metrics, "ppo": metrics},
                        split=base_source["split"],
                        metadata={
                            "candidate_id": candidate.candidate_id,
                            "curriculum_progress": completed_progress,
                            "curriculum_profile": curriculum.name,
                            "source_checkpoint": str(base_checkpoint),
                        },
                    )
                    saved_milestones.add(milestone)
                    
                    if eval_seeds > 0:
                        from luxai2021.rl.evaluation import evaluate_against_league, LeagueMember
                        import shutil
                        milestone_path = milestone_dir / f"p{round(milestone * 100):03d}.pt"
                        teacher = LeagueMember("first-place", teacher_checkpoint, model_type="first-place")
                        evaluation = evaluate_against_league(
                            LeagueMember("milestone", milestone_path),
                            [teacher],
                            seed_start=eval_seed_start,
                            seed_count=eval_seeds,
                            device=str(device),
                            max_turns=max_turns,
                        )
                        score_rate = float(evaluation["totals"]["score_rate"])
                        if score_rate > best_teacher_score_rate:
                            best_teacher_score_rate = score_rate
                            shutil.copyfile(milestone_path, output_dir / "best.pt")
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                        
                        if epochs_without_improvement >= 2:
                            print(f"Early stopping at milestone {milestone}. Best: {best_teacher_score_rate:.3f}, Current: {score_rate:.3f}")
                            should_early_stop = True
                            break
            if checkpoint_callback is not None:
                checkpoint_callback(output_dir, metrics)
            
            if update > 0 and update % 10 == 0:
                snapshot.load_state_dict(actor_critic.state_dict())
                
            update += 1
            if decision_budget is None and seconds <= 0:
                break
    finally:
        candidate_batcher.close()
        opponent_resources.close()
    final_metrics = history[-1] if history else resumed_metrics
    if final_metrics is None:
        raise RuntimeError("Training completed without a checkpoint or a PPO update")
    candidate_runtime_value = final_metrics.get("candidate_inference", {})
    opponent_runtime_value = final_metrics.get("opponent_inference", {})
    candidate_runtime = candidate_runtime_value if isinstance(candidate_runtime_value, Mapping) else {}
    opponent_runtime = opponent_runtime_value if isinstance(opponent_runtime_value, Mapping) else {}
    rollout_runtime = {
        "backend_requested": rollout_backend,
        "backend_effective": resolve_rollout_backend(rollout_backend),
        "backend_fallback_reason": (
            "lockstep_acceptance_not_met" if rollout_backend == "auto" else None
        ),
        "precision_requested": rollout_precision,
        "precision_effective": candidate_runtime.get("precision"),
        "compile_requested": rollout_compile,
        "compile_effective": candidate_runtime.get("compile_effective"),
        "compile_fallback_reason": candidate_runtime.get("compile_fallback_reason"),
        "candidate": candidate_runtime,
        "opponents": opponent_runtime,
    }
    summary = {
        "candidate_id": candidate.candidate_id,
        "base_name": base_name,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_checkpoint_sha256,
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
        "training_curriculum": curriculum.to_dict(),
        "curriculum_total_decisions": curriculum_total,
        "curriculum_start_decisions": max(0, int(curriculum_start_decisions)),
        "curriculum_milestones": [f"p{round(point * 100):03d}.pt" for point in sorted(saved_milestones)],
        "bc_anchor": {
            "replays": bc_replays,
            "max_turns": bc_anchor_max_turns,
            "sampling": bc_anchor_sampling,
            "seed": seed if bc_anchor_seed is None else bc_anchor_seed,
        },
        "decision_budget": decision_budget,
        "decisions_per_update": decisions_per_update,
        "rollout_envs": rollout_envs,
        "rollout_backend": resolve_rollout_backend(rollout_backend),
        "rollout_precision": rollout_precision,
        "rollout_compile": rollout_compile,
        "rollout_batch_wait_ms": rollout_batch_wait_ms,
        "rollout_runtime": rollout_runtime,
        "history": history,
        "final_metrics": final_metrics,
        "diagnostic_events": diagnostic_events,
    }
    (output_dir / "rollout_runtime.json").write_text(
        json.dumps(rollout_runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not (output_dir / "best.pt").exists():
        actor_critic.export_policy(
            output_dir / "best.pt",
            epoch=max(0, update - 1),
            metrics={"validation": final_metrics, "ppo": final_metrics},
            split=base_source["split"],
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
    final_metrics = value.get("final_metrics", {})
    if isinstance(final_metrics, dict) and "calibrated_reward_scale" in final_metrics:
        return float(final_metrics["calibrated_reward_scale"])
    effective = value.get("effective_reward_program", {})
    return float(effective.get("reward_scale", fallback)) if isinstance(effective, dict) else fallback


def _final_training_metrics(training: Mapping[str, object]) -> Mapping[str, object]:
    final_metrics = training.get("final_metrics")
    if isinstance(final_metrics, dict):
        return final_metrics
    history = training.get("history")
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        return history[-1]
    raise RuntimeError("Training result does not contain final PPO metrics")


def _curriculum_start_decisions(stage: str, args: argparse.Namespace) -> int:
    if stage.startswith("medium-"):
        return int(args.short_decisions)
    if stage.startswith("final-"):
        return int(args.short_decisions) + int(args.medium_decisions)
    return 0


def _select_teacher_milestone(
    output_dir: Path,
    *,
    candidate_id: str,
    teacher_checkpoint: Path,
    seed_start: int,
    seed_count: int,
    device: str,
    max_turns: int,
) -> tuple[Path, dict[str, object]]:
    checkpoints = sorted((output_dir / "milestones").glob("p*.pt"))
    if not checkpoints:
        return output_dir / "best.pt", {"enabled": True, "reason": "no_milestones"}
    teacher = LeagueMember("first-place", teacher_checkpoint, model_type="first-place")
    evaluations = []
    for index, checkpoint in enumerate(checkpoints):
        evaluation = evaluate_against_league(
            LeagueMember(f"{candidate_id}-{checkpoint.stem}", checkpoint),
            [teacher],
            seed_start=seed_start,
            seed_count=seed_count,
            device=device,
            max_turns=max_turns,
        )
        evaluations.append(
            {
                "checkpoint": str(checkpoint),
                "teacher_score_rate": float(evaluation["totals"]["score_rate"]),
                "order": index,
            }
        )
    selected = max(evaluations, key=lambda value: (value["teacher_score_rate"], value["order"]))
    report = {
        "enabled": True,
        "seed_start": seed_start,
        "seed_count": seed_count,
        "selection_policy": "highest_teacher_score_rate_then_latest",
        "selected_checkpoint": selected["checkpoint"],
        "evaluations": evaluations,
    }
    (output_dir / "milestone_selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return Path(str(selected["checkpoint"])), report


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
    prior_short_best = (
        Path(args.run_dir)
        / "artifacts"
        / candidate.candidate_id
        / "short-resattn8"
        / "resattn8"
        / "best.pt"
    )
    resume_from = current_checkpoint if current_checkpoint.exists() else None
    predecessor_paths = [prior_short_best] if stage == "medium-resattn8" else []
    if stage == "final-resattn8":
        predecessor_paths.extend(
            [
                Path(args.run_dir)
                / "artifacts"
                / candidate.candidate_id
                / "medium-resattn8"
                / "resattn8"
                / "best.pt",
                prior_short_best,
            ]
        )
    if stage == "final-unet":
        predecessor_paths.append(
            Path(args.run_dir) / "artifacts" / candidate.candidate_id / "probe-unet" / "unet" / "best.pt"
        )
    stage_inherit_from = (
        None if resume_from is not None else next((path for path in predecessor_paths if path.exists()), None)
    )
    inheritance = None
    if resume_from is None and stage_inherit_from is None:
        inheritance = _resolve_parent_checkpoint(Path(args.run_dir), candidate, candidates, base_name)
    inherit_from = stage_inherit_from if stage_inherit_from is not None else (
        inheritance[0] if inheritance is not None else None
    )
    parent = candidate if stage_inherit_from is not None else (inheritance[1] if inheritance is not None else None)
    parent_effective_scale = (
        _effective_scale_from_artifact(inherit_from, parent.reward_program.reward_scale)
        if inherit_from is not None and parent is not None
        else None
    )
    curriculum_total_decisions = args.short_decisions + args.medium_decisions + args.final_decisions
    try:
        checkpoint, training = train_candidate(
            candidate,
            base_name=base_name,
            base_checkpoint=base_checkpoint,
            other_checkpoint=other_checkpoint,
            teacher_checkpoint=Path(args.teacher_checkpoint),
            eval_seeds=min(eval_seeds, args.screening_seeds),
            eval_seed_start=eval_seed_start,
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
            bc_anchor_max_turns=args.bc_anchor_max_turns,
            bc_anchor_sampling=args.bc_anchor_sampling,
            bc_anchor_seed=args.seed,
            use_bc_anchor=not args.no_bc_anchor,
            device=device,
            seed=args.seed + candidate.generation * 10_000 + candidate.island * 100,
            max_turns=args.max_turns,
            curriculum_profile=args.curriculum_profile,
            curriculum_total_decisions=curriculum_total_decisions,
            curriculum_start_decisions=_curriculum_start_decisions(stage, args),
            inherit_from=inherit_from,
            parent_reward_program=parent.reward_program if parent is not None else None,
            parent_effective_scale=parent_effective_scale,
            critic_warmup_episodes=args.critic_warmup_episodes,
            reward_mode=args.reward_mode,
            resume_from=resume_from,
            resume_budget_progress=resume_from == current_checkpoint,
            checkpoint_callback=checkpoint_callback,
            rollout_backend=args.rollout_backend,
            rollout_precision=args.rollout_precision,
            rollout_compile=args.rollout_compile,
            rollout_batch_wait_ms=args.rollout_batch_wait_ms,
        )
        milestone_selection: dict[str, object] = {"enabled": False}
        if args.curriculum_profile != "legacy" and stage.startswith("final-"):
            checkpoint, milestone_selection = _select_teacher_milestone(
                output_dir,
                candidate_id=candidate.candidate_id,
                teacher_checkpoint=Path(args.teacher_checkpoint),
                seed_start=eval_seed_start,
                seed_count=min(eval_seeds, args.screening_seeds),
                device=str(device),
                max_turns=args.max_turns,
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
        kl = 0.0
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
            "milestone_selection": milestone_selection,
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
    if int(manifest.get("metric_schema_version", 1)) != _METRIC_SCHEMA_VERSION:
        raise ValueError("Reward metric schema changed; preserve this run and start a new run directory")
    coordinator_args = manifest.get("arguments", {})
    if not isinstance(coordinator_args, dict):
        raise TypeError("Coordinator manifest arguments are invalid")
    for name in _RUN_MANIFEST_ARGUMENTS:
        if name in coordinator_args:
            setattr(args, name, coordinator_args[name])
    if "curriculum_profile" not in coordinator_args and hasattr(args, "curriculum_profile"):
        args.curriculum_profile = "legacy"
        args.bc_anchor_max_turns = 64
        args.bc_anchor_sampling = "replay"
    stored_curriculum = manifest.get("training_curriculum")
    if stored_curriculum is not None and hasattr(args, "curriculum_profile"):
        current_curriculum = training_curriculum(args.curriculum_profile).to_dict()
        if stored_curriculum != current_curriculum:
            raise ValueError("Training curriculum changed; preserve this run and start a new run directory")
    # Runs created before rollout runtime controls were recorded must resume with
    # the historical eager FP32 threaded path, rather than silently changing
    # trajectories because the new CLI defaults are auto-selected.
    if "rollout_backend" not in coordinator_args and hasattr(args, "rollout_backend"):
        args.rollout_backend = "threaded"
        args.rollout_precision = "fp32"
        args.rollout_compile = "off"
        args.rollout_batch_wait_ms = 2.0
    expected_rules = manifest.get("lux_s1_rules")
    if expected_rules is not None:
        current_rules = lux_s1_rules_context()
        rules_changed = (
            not isinstance(expected_rules, dict)
            or expected_rules.get("summary_sha256") != current_rules["summary_sha256"]
        )
        if rules_changed:
            raise ValueError("Lux S1 rules summary changed; start a new run directory")


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
    validated_manifests: set[str] = set()

    def apply_manifest(value: Mapping[str, object]) -> None:
        _apply_coordinator_manifest(args, value)
        signature = hashlib.sha256(
            json.dumps(value.get("checkpoint_descriptors", {}), sort_keys=True).encode()
        ).hexdigest()
        if signature not in validated_manifests:
            _validate_checkpoint_descriptors(args, value)
            validated_manifests.add(signature)

    if args.job_api_url is None and not (run_dir / "manifest.json").exists():
        raise ValueError("Worker run directory does not contain a coordinator manifest")
    if (run_dir / "manifest.json").exists():
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        apply_manifest(manifest)
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
                apply_manifest(claim["manifest"])
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
    baseline: Mapping[str, object] | None = None,
) -> list[str]:
    def ranking_key(result: CandidateResult) -> tuple[float, ...]:
        if baseline is None or result.status != "completed":
            return result.fitness
        evaluation = result.metrics.get("evaluation", {}) if isinstance(result.metrics, Mapping) else {}
        if not isinstance(evaluation, dict):
            return result.fitness
        seed_deltas, anchor_deltas = paired_seed_deltas(evaluation, dict(baseline))
        teacher_delta = float(anchor_deltas.get("first-place", -1.0))
        base_delta = max(
            (float(value) for name, value in anchor_deltas.items() if name != "first-place"),
            default=-1.0,
        )
        combined_delta = sum(seed_deltas.values()) / max(len(seed_deltas), 1)
        valid = float(result.fitness[0] > 0.0)
        return valid, float(teacher_delta >= 0.0), teacher_delta, combined_delta, base_delta

    ranked = sorted(
        (
            result
            for result in results
            if result.stage == stage and result.status == "completed" and result.candidate_id in candidates
        ),
        key=ranking_key,
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
    allow_legacy_unverified: bool = False,
    baseline: Mapping[str, object] | None = None,
) -> list[EvolutionCandidate]:
    path = store.run_dir / "selections" / f"{name}.json"
    eligible = {
        result.candidate_id
        for result in results
        if result.stage == source_stage and result.status == "completed" and result.candidate_id in candidates
    }
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        candidate_ids = [str(candidate_id) for candidate_id in value.get("candidate_ids", [])]
        selection_was_verified = value.get("source_stage_verified") is True
    else:
        candidate_ids = _legacy_stage_selection(store.run_dir, target_stage, count)
        source = "legacy_jobs" if candidate_ids else source_stage
        invalid_legacy = [candidate_id for candidate_id in candidate_ids if candidate_id not in eligible]
        if invalid_legacy and not allow_legacy_unverified:
            candidate_ids = []
            source = source_stage
        if not candidate_ids:
            candidate_ids = _select_completed_stage(
                candidates,
                results,
                stage=source_stage,
                count=count,
                baseline=baseline,
            )
        unverified = [candidate_id for candidate_id in candidate_ids if candidate_id not in eligible]
        path.parent.mkdir(parents=True, exist_ok=True)
        EvolutionStore.write_json(
            path,
            {
                "schema_version": 2,
                "name": name,
                "source": source,
                "source_stage": source_stage,
                "target_stage": target_stage,
                "candidate_ids": candidate_ids,
                "source_stage_verified": not unverified,
                "unverified_candidate_ids": unverified,
                "selection_policy": "teacher_guarded_paired" if baseline is not None else "legacy_fitness",
                "baseline_context": baseline.get("_context") if baseline is not None else None,
                "created_at": time.time(),
            },
        )
        selection_was_verified = not unverified
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in candidates]
    if missing:
        message = f"Selection {name!r} references missing candidates: {missing}"
        raise ValueError(message)
    unverified = (
        []
        if selection_was_verified
        else [candidate_id for candidate_id in candidate_ids if candidate_id not in eligible]
    )
    if unverified and not allow_legacy_unverified:
        message = f"Selection {name!r} contains candidates without completed {source_stage}: {unverified}"
        raise ValueError(message)
    return [candidates[candidate_id] for candidate_id in candidate_ids]


def _load_or_create_stage_baseline(
    store: EvolutionStore,
    args: argparse.Namespace,
    *,
    name: str,
    seed_start: int,
    seed_count: int,
    device: torch.device,
) -> dict[str, object]:
    path = store.run_dir / "baselines" / f"{name}.json"
    context = {
        "schema_version": 2,
        "metric_schema_version": _METRIC_SCHEMA_VERSION,
        "base_name": "resattn8",
        "checkpoint_sha256": _sha256_file(Path(args.resattn8_checkpoint)),
        "seed_start": seed_start,
        "seed_count": seed_count,
        "max_turns": args.max_turns,
    }
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("_context") != context:
            message = f"Stage baseline context changed: {path}"
            raise ValueError(message)
        return value
    path.parent.mkdir(exist_ok=True)
    value = evaluate_against_league(
        LeagueMember("resattn8-baseline-eval", Path(args.resattn8_checkpoint)),
        _evaluation_anchors(args),
        seed_start=seed_start,
        seed_count=seed_count,
        device=str(device),
        max_turns=args.max_turns,
    )
    value["_context"] = context
    store.write_json(path, value)
    return value


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


def _job_retry_state_path(run_dir: Path, job: EvolutionJob) -> Path:
    return run_dir / "jobs" / "retries" / f"{job.job_id}.json"


def _job_retry_count(run_dir: Path, job: EvolutionJob) -> int:
    path = _job_retry_state_path(run_dir, job)
    if not path.exists():
        return 0
    value = json.loads(path.read_text(encoding="utf-8"))
    return int(value.get("retry_count", 0))


def _record_job_retry(run_dir: Path, job: EvolutionJob, result: CandidateResult) -> int:
    path = _job_retry_state_path(run_dir, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    retry_count = _job_retry_count(run_dir, job) + 1
    EvolutionStore.write_json(
        path,
        {
            "candidate_id": job.candidate_id,
            "stage": job.stage,
            "retry_count": retry_count,
            "last_error": result.error,
            "updated_at": time.time(),
        },
    )
    return retry_count


def _record_skipped_job(run_dir: Path, job: EvolutionJob, result: CandidateResult) -> None:
    path = run_dir / "jobs" / "skipped" / f"{job.job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    EvolutionStore.write_json(
        path,
        {
            "candidate_id": job.candidate_id,
            "stage": job.stage,
            "status": "skipped_after_infrastructure_retries",
            "retry_count": _job_retry_count(run_dir, job),
            "error": result.error,
            "skipped_at": time.time(),
        },
    )


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


def _run_integrity_report(
    run_dir: Path,
    manifest: Mapping[str, object],
    candidates: Mapping[str, EvolutionCandidate],
    results: list[CandidateResult],
    medium: list[EvolutionCandidate],
    finalists: list[EvolutionCandidate],
) -> dict[str, object]:
    manifest_schema = int(manifest.get("schema_version", 1))
    issues = []
    if manifest_schema < 2:
        issues.append("legacy manifest has no immutable checkpoint/proposal provenance lock")
        legacy_counts: dict[str, int] = {}
        for candidate in candidates.values():
            if candidate.mutation_kind == "initial" and not candidate.parent_ids:
                source = "initial"
            elif candidate.rationale.startswith("Deterministic island-"):
                source = "deterministic_fallback"
            else:
                source = "codex_unverified"
            legacy_counts[source] = legacy_counts.get(source, 0) + 1
        provenance: dict[str, object] = {
            "expected_mode": "legacy",
            "counts": legacy_counts,
            "errors": ["candidate provenance was not recorded"],
            "valid": False,
            "fully_codex_guided": False,
        }
    else:
        provenance = _validate_candidate_provenance(run_dir, candidates, manifest)
        issues.extend(str(error) for error in provenance["errors"])
        if provenance["expected_mode"] == "codex" and not provenance["fully_codex_guided"]:
            issues.append("Codex-mode run contains non-Codex candidate mutations")
    completed_medium = {
        result.candidate_id for result in results if result.stage == "medium-resattn8" and result.status == "completed"
    }
    final_without_medium = [
        candidate.candidate_id for candidate in finalists if candidate.candidate_id not in completed_medium
    ]
    if final_without_medium:
        issues.append(f"final candidates without completed medium stage: {final_without_medium}")
    final_lineages: dict[str, str] = {}
    for result in results:
        if result.status != "completed" or not result.stage.startswith("final-"):
            continue
        training = result.metrics.get("training", {}) if isinstance(result.metrics, dict) else {}
        resume = training.get("resume") if isinstance(training, dict) else None
        lineage = resume.get("stored_source_checkpoint") if isinstance(resume, dict) else None
        if lineage is None and isinstance(training, dict):
            lineage = training.get("base_checkpoint")
        if lineage is not None:
            final_lineages[result.candidate_id] = str(lineage)
    if len(set(final_lineages.values())) > 1:
        issues.append(f"final candidates have mixed base-checkpoint lineages: {final_lineages}")
    medium_without_short = [
        candidate.candidate_id
        for candidate in medium
        if not any(
            result.candidate_id == candidate.candidate_id
            and result.stage == "short-resattn8"
            and result.status == "completed"
            for result in results
        )
    ]
    if medium_without_short:
        issues.append(f"medium candidates without completed short stage: {medium_without_short}")
    descriptors = manifest.get("checkpoint_descriptors", {})
    checkpoint_lock_valid = manifest_schema >= 2 and isinstance(descriptors, dict) and bool(descriptors)
    if not checkpoint_lock_valid:
        issues.append("checkpoint SHA-256 descriptors are unavailable")
    elif isinstance(descriptors, dict):
        for result in results:
            if result.status != "completed" or not result.stage.startswith("final-"):
                continue
            base_name = result.stage.removeprefix("final-")
            descriptor = descriptors.get(base_name)
            training = result.metrics.get("training", {}) if isinstance(result.metrics, dict) else {}
            actual_sha256 = training.get("base_checkpoint_sha256") if isinstance(training, dict) else None
            expected_sha256 = descriptor.get("sha256") if isinstance(descriptor, dict) else None
            if actual_sha256 != expected_sha256:
                issues.append(
                    f"{result.candidate_id} {result.stage} base SHA-256 mismatch: "
                    f"expected={expected_sha256!r} actual={actual_sha256!r}"
                )
    return {
        "valid_for_promotion": not issues,
        "issues": issues,
        "manifest_schema_version": manifest_schema,
        "checkpoint_lock_valid": checkpoint_lock_valid,
        "candidate_provenance": provenance,
        "medium_without_short": medium_without_short,
        "final_without_medium": final_without_medium,
        "final_checkpoint_lineages": final_lineages,
    }


def _notify_run_event(
    notifier: EvolutionNotifier,
    run_dir: Path,
    event_id: str,
    title: str,
    detail: str,
    *,
    priority: int = 3,
    tags: tuple[str, ...] = (),
) -> None:
    notifier.notify_once(
        event_id,
        title=title,
        message=f"{detail}\nhost: {socket.gethostname()}\nrun_dir: {run_dir}",
        priority=priority,
        tags=tags,
    )


def main(
    args: argparse.Namespace | None = None,
    notifier: EvolutionNotifier | None = None,
) -> None:
    args = parse_args() if args is None else args
    run_dir = Path(args.run_dir)
    notifier = notifier or EvolutionNotifier.from_environment(run_dir)
    for warning in notifier.configuration_warnings:
        print(json.dumps({"notification_configuration_warning": warning}, sort_keys=True), file=sys.stderr)
    if args.worker:
        run_worker(args)
        return
    if args.job_api_url:
        raise ValueError("--job-api-url is only valid with --worker")
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
        _validate_run_kind(existing_manifest, dry_run=args.dry_run)
        _apply_coordinator_manifest(args, existing_manifest)
    fixed_candidate_path = Path(args.fixed_candidate) if args.fixed_candidate else None
    fixed_candidate = _load_fixed_candidate(fixed_candidate_path) if fixed_candidate_path is not None else None
    if existing_manifest is not None and fixed_candidate_path is not None:
        _validate_fixed_candidate_descriptor(fixed_candidate_path, existing_manifest)
    if fixed_candidate is not None:
        args.islands = 1
        args.initial_per_island = 1
        args.generations = 0
        args.medium_count = 1
        args.final_count = 1
        args.no_codex = True
    if args.islands < 1 or args.initial_per_island < 1 or args.generations < 0:
        raise ValueError("Population sizes must be positive")
    if args.rollout_envs < 1 or args.decisions_per_update < 1:
        raise ValueError("Rollout environment and decision budgets must be positive")
    if args.rollout_batch_wait_ms < 0:
        raise ValueError("Rollout batch wait must be non-negative")
    if not args.no_bc_anchor and (args.bc_replays < 1 or args.bc_batch_size < 1):
        raise ValueError("BC anchor replay and batch counts must be positive")
    if args.bc_anchor_max_turns < 0:
        raise ValueError("BC anchor max turns must be non-negative")
    if not 0.0 <= args.teacher_noninferiority_margin <= 0.1:
        raise ValueError("Teacher noninferiority margin must be in [0, 0.1]")
    repository = Path(__file__).resolve().parents[1]
    device = resolve_device(args.device)
    configure_rollout_determinism(device)
    if not args.no_codex and not args.dry_run and shutil.which(args.codex_executable) is None:
        message = f"Codex executable is unavailable: {args.codex_executable}"
        raise FileNotFoundError(message)
    store = EvolutionStore(run_dir)
    if existing_manifest is None:
        generation_mode = (
            "fixed"
            if fixed_candidate is not None
            else "dry_run"
            if args.dry_run
            else "deterministic"
            if args.no_codex
            else "codex"
        )
        effective_precision, _ = resolve_rollout_precision(args.rollout_precision, device)
        effective_backend = resolve_rollout_backend(args.rollout_backend)
        compile_can_calibrate = device.type == "cuda" and (
            args.rollout_compile == "on" or (args.rollout_compile == "auto" and effective_backend == "lockstep")
        )
        manifest = {
            "schema_version": _RUN_MANIFEST_SCHEMA_VERSION,
            "metric_schema_version": _METRIC_SCHEMA_VERSION,
            "created_at": time.time(),
            "git_revision": git_revision(repository),
            "arguments": vars(args),
            "device": str(device),
            "rollout_runtime": {
                "backend_requested": args.rollout_backend,
                "backend_effective": effective_backend,
                "backend_fallback_reason": (
                    "lockstep_acceptance_not_met" if args.rollout_backend == "auto" else None
                ),
                "precision_requested": args.rollout_precision,
                "precision_effective": effective_precision,
                "compile_requested": args.rollout_compile,
                "compile_effective": "pending_runtime_calibration" if compile_can_calibrate else "off",
                "compile_fallback_reason": (
                    None
                    if compile_can_calibrate or args.rollout_compile == "off"
                    else "compile_requires_cuda"
                    if device.type != "cuda"
                    else "auto_compile_requires_static_batches"
                ),
            },
            "codex_available": shutil.which(args.codex_executable) is not None,
            "bases": {name: getattr(args, f"{name}_checkpoint") for name in _active_base_names(args)},
            "run_kind": "dry-run" if args.dry_run else "training",
            "candidate_generation": {
                "mode": generation_mode,
                "allow_fallback": bool(args.allow_codex_fallback),
                "model": args.codex_model,
                "executable": args.codex_executable,
            },
            "lux_s1_rules": lux_s1_rules_context(),
            "training_curriculum": training_curriculum(args.curriculum_profile).to_dict(),
            "checkpoint_descriptors": {} if args.dry_run else _checkpoint_descriptors(args),
            "fixed_candidate_descriptor": (
                {
                    "path": str(fixed_candidate_path),
                    "size": fixed_candidate_path.stat().st_size,
                    "sha256": _sha256_file(fixed_candidate_path),
                }
                if fixed_candidate_path is not None
                else None
            ),
        }
        store.save_manifest(manifest)
    else:
        manifest = existing_manifest
    if not args.dry_run:
        _notify_run_event(
            notifier,
            run_dir,
            f"run-started:{manifest.get('created_at', 'legacy')}",
            "Lux evolution coordinator active",
            f"Coordinator started or resumed.\ndevice: {device}\ngeneration target: {args.generations}",
            tags=("rocket",),
        )
    _validate_checkpoint_descriptors(args, manifest)
    generator = None
    if not args.no_codex and not args.dry_run:
        generator = CodexCandidateGenerator(
            repository=repository,
            run_dir=run_dir,
            executable=args.codex_executable,
            model=args.codex_model,
            timeout_seconds=args.codex_timeout,
            validation_retries=args.codex_validation_retries,
        )
    candidates = {candidate.candidate_id: candidate for candidate in store.candidates()}
    results = store.results()
    if int(manifest.get("schema_version", 1)) >= 2 and candidates:
        provenance_report = _validate_candidate_provenance(run_dir, candidates, manifest)
        if not provenance_report["valid"]:
            message = f"Candidate provenance audit failed: {provenance_report['errors']}"
            raise ValueError(message)
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

    def register(candidate: EvolutionCandidate, provenance: Mapping[str, object]) -> None:
        candidates[candidate.candidate_id] = candidate
        store.save_candidate(candidate)
        _save_candidate_provenance(run_dir, candidate, provenance)

    def candidate_provenance(
        candidate: EvolutionCandidate,
        source: str,
        *,
        generation: int,
        island: int,
        error: Exception | None = None,
    ) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": 1,
            "candidate_id": candidate.candidate_id,
            "generation": generation,
            "island": island,
            "parent_ids": list(candidate.parent_ids),
            "source": source,
            "created_at": time.time(),
        }
        if source == "codex":
            metadata_path = generator.metadata_path(generation, island) if generator is not None else None
            if metadata_path is None or not metadata_path.exists():
                message = f"Accepted Codex proposal metadata is missing: {metadata_path}"
                raise FileNotFoundError(message)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            value.update(
                {
                    "proposal_metadata": metadata_path.name,
                    "proposal_path": metadata["proposal_path"],
                    "proposal_sha256": metadata["proposal_sha256"],
                    "raw_proposal_path": metadata.get("raw_proposal_path"),
                    "raw_proposal_sha256": metadata.get("raw_proposal_sha256"),
                    "prompt_path": metadata.get("prompt_path"),
                    "prompt_sha256": metadata.get("prompt_sha256"),
                    "model": metadata.get("model"),
                }
            )
        if error is not None:
            value.update({"fallback_error_type": type(error).__name__, "fallback_error": str(error)})
        return value

    def refresh_results() -> None:
        latest = {(result.candidate_id, result.stage): result for result in results}
        for stored in store.results():
            key = (stored.candidate_id, stored.stage)
            latest[key] = stored
        results[:] = list(latest.values())

    def jobs_are_pending(jobs: list[EvolutionJob]) -> bool:
        if args.dry_run or not jobs:
            return False
        refresh_results()
        completed = {(result.candidate_id, result.stage) for result in results if result.status == "completed"}
        return any((job.candidate_id, job.stage) not in completed for job in jobs)

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
        original_expected = set(expected)
        job_by_key = {(job.candidate_id, job.stage): job for job in jobs}
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
                retried = []
                exhausted = []
                terminal = []
                for result in failed:
                    key = result.candidate_id, result.stage
                    job = job_by_key[key]
                    if not _is_retryable_infrastructure_failure(result):
                        terminal.append(result)
                        continue
                    retry_count = _job_retry_count(run_dir, job)
                    if retry_count < _AUTOMATIC_INFRASTRUCTURE_RETRIES:
                        retry_count = _record_job_retry(run_dir, job, result)
                        _archive_failed_results(store, [job])
                        results[:] = [
                            item
                            for item in results
                            if (item.candidate_id, item.stage) != key or item.status == "completed"
                        ]
                        queue.enqueue(job)
                        retried.append((job, result, retry_count))
                    else:
                        _record_skipped_job(run_dir, job, result)
                        expected.discard(key)
                        exhausted.append((job, result))
                for job, result, retry_count in retried:
                    print(
                        json.dumps(
                            {
                                "job_id": job.job_id,
                                "status": "infrastructure_retry",
                                "retry": retry_count,
                                "max_retries": _AUTOMATIC_INFRASTRUCTURE_RETRIES,
                                "error": result.error,
                            },
                            sort_keys=True,
                        )
                    )
                if retried:
                    continue
                if exhausted:
                    skipped_ids = [job.job_id for job, _ in exhausted]
                    _notify_run_event(
                        notifier,
                        run_dir,
                        f"infrastructure-jobs-skipped:{time.time_ns()}",
                        "Lux evolution jobs skipped after retries",
                        f"Skipped jobs after {_AUTOMATIC_INFRASTRUCTURE_RETRIES} retries: {skipped_ids}",
                        priority=4,
                        tags=("warning",),
                    )
                    print(json.dumps({"job_ids": skipped_ids, "status": "infrastructure_skipped"}, sort_keys=True))
                    if expected:
                        continue
                    if original_expected & completed:
                        return
                    detail = ", ".join(f"{job.job_id}: {result.error}" for job, result in exhausted)
                    message = f"All evolution jobs failed from infrastructure errors after retries: {detail}"
                    raise RuntimeError(message)
                if terminal:
                    failed = terminal
                detail = ", ".join(f"{result.candidate_id}--{result.stage}: {result.error}" for result in failed)
                message = f"Evolution jobs failed; fix the cause and resume: {detail}"
                raise RuntimeError(message)
            recovered_jobs = queue.recover_stale(args.recover_stale_job_seconds)
            if recovered_jobs:
                _notify_run_event(
                    notifier,
                    run_dir,
                    f"stale-jobs-recovered:{time.time_ns()}",
                    "Lux evolution worker lease expired",
                    f"Recovered and requeued jobs: {recovered_jobs}",
                    priority=4,
                    tags=("warning",),
                )
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
            base = fixed_candidate if fixed_candidate is not None else initial_candidate(island=island, seed=args.seed)
            register(base, candidate_provenance(base, "initial", generation=0, island=island))
        island_parents[island] = base
        base_wave.append(short_job(base))
    generation_zero_progress = jobs_are_pending(base_wave)
    evaluate_jobs(base_wave)

    for initial_index in range(1, args.initial_per_island):
        wave = []
        for island in range(args.islands):
            existing_initial = initial_by_island[island]
            parent = island_parents[island]
            provenance = None
            if initial_index < len(existing_initial):
                child = existing_initial[initial_index]
            elif generator is not None:
                try:
                    child = generator.generate([parent], results, generation=0, island=island)
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                    if not args.allow_codex_fallback:
                        message = f"Codex proposal failed for generation 0 island {island}; fallback is disabled"
                        raise RuntimeError(message) from error
                    print(f"Codex fallback for island {island}: {error}")
                    child = mutate_candidate(
                        parent,
                        generation=0,
                        island=island,
                        seed=args.seed + island * 100 + initial_index,
                    )
                    provenance = candidate_provenance(
                        child,
                        "codex_fallback",
                        generation=0,
                        island=island,
                        error=error,
                    )
                else:
                    provenance = candidate_provenance(child, "codex", generation=0, island=island)
            else:
                child = mutate_candidate(
                    parent,
                    generation=0,
                    island=island,
                    seed=args.seed + island * 100 + initial_index,
                )
                source = "dry_run" if args.dry_run else "deterministic"
                provenance = candidate_provenance(child, source, generation=0, island=island)
            if provenance is not None:
                register(child, provenance)
            wave.append(short_job(child))
            island_parents[island] = child
        generation_zero_progress = jobs_are_pending(wave) or generation_zero_progress
        evaluate_jobs(wave)

    if generation_zero_progress:
        _notify_run_event(
            notifier,
            run_dir,
            "generation-00-completed",
            "Lux evolution generation 0 completed",
            f"Initial population completed.\ncandidates: {len(candidates)}",
            tags=("white_check_mark",),
        )

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
                        if not args.allow_codex_fallback:
                            message = (
                                f"Codex proposal failed for generation {generation} island {island}; "
                                "fallback is disabled"
                            )
                            raise RuntimeError(message) from error
                        print(f"Codex fallback for generation {generation} island {island}: {error}")
                        child = mutate_candidate(
                            parent,
                            generation=generation,
                            island=island,
                            seed=args.seed + generation * 1000 + island,
                            secondary_parents=tuple(parents[1:]),
                            stagnated=stagnated,
                        )
                        provenance = candidate_provenance(
                            child,
                            "codex_fallback",
                            generation=generation,
                            island=island,
                            error=error,
                        )
                    else:
                        provenance = candidate_provenance(child, "codex", generation=generation, island=island)
                else:
                    child = mutate_candidate(
                        parent,
                        generation=generation,
                        island=island,
                        seed=args.seed + generation * 1000 + island,
                        secondary_parents=tuple(parents[1:]),
                        stagnated=stagnated,
                    )
                    source = "dry_run" if args.dry_run else "deterministic"
                    provenance = candidate_provenance(child, source, generation=generation, island=island)
                register(child, provenance)
            wave.append(short_job(child))
        generation_progress = jobs_are_pending(wave)
        evaluate_jobs(wave)
        if not args.resattn8_only and args.unet_probe_every > 0 and generation % args.unet_probe_every == 0:
            refreshed_elites = select_elites(candidates, results, count=1)
            if refreshed_elites:
                probe = refreshed_elites[0]
                probe_jobs = [
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
                generation_progress = jobs_are_pending(probe_jobs) or generation_progress
                evaluate_jobs(probe_jobs)
        if generation_progress:
            generation_results = [
                result
                for result in results
                if result.status == "completed"
                and candidates.get(result.candidate_id) is not None
                and candidates[result.candidate_id].generation == generation
                and result.stage == "short-resattn8"
            ]
            best_score = max((result.score_rate for result in generation_results), default=float("nan"))
            _notify_run_event(
                notifier,
                run_dir,
                f"generation-{generation:02d}-completed",
                f"Lux evolution generation {generation} completed",
                f"Short-stage candidates: {len(generation_results)}\nbest score rate: {best_score:.4f}",
                tags=("white_check_mark",),
            )

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "candidates": len(candidates), "dry_run": True}))
        return

    screening_baseline = (
        _load_or_create_stage_baseline(
            store,
            args,
            name="short-resattn8",
            seed_start=args.screening_seed_start,
            seed_count=args.screening_seeds,
            device=device,
        )
        if args.curriculum_profile != "legacy"
        else None
    )
    medium = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="medium",
        target_stage="medium-resattn8",
        source_stage="short-resattn8",
        count=args.medium_count,
        allow_legacy_unverified=int(manifest.get("schema_version", 1)) < 2,
        baseline=screening_baseline,
    )
    _archive_unselected_stage_jobs(
        run_dir,
        "medium-resattn8",
        {candidate.candidate_id for candidate in medium},
    )
    medium_jobs = [
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
    medium_progress = jobs_are_pending(medium_jobs)
    if medium_progress:
        _notify_run_event(
            notifier,
            run_dir,
            "medium-stage-started",
            "Lux evolution medium stage started",
            f"Selected candidates: {', '.join(candidate.candidate_id for candidate in medium)}",
            tags=("hourglass_flowing_sand",),
        )
    evaluate_jobs(medium_jobs)
    if medium_progress:
        _notify_run_event(
            notifier,
            run_dir,
            "medium-stage-completed",
            "Lux evolution medium stage completed",
            f"Completed candidates: {', '.join(candidate.candidate_id for candidate in medium)}",
            tags=("white_check_mark",),
        )

    medium_baseline = (
        _load_or_create_stage_baseline(
            store,
            args,
            name="medium-resattn8",
            seed_start=args.screening_seed_start + 10_000,
            seed_count=args.medium_seeds,
            device=device,
        )
        if args.curriculum_profile != "legacy"
        else None
    )
    finalists = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="final",
        target_stage="final-resattn8",
        source_stage="medium-resattn8",
        count=args.final_count,
        allow_legacy_unverified=int(manifest.get("schema_version", 1)) < 2,
        baseline=medium_baseline,
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
    final_progress = jobs_are_pending(final_jobs)
    if final_progress:
        _notify_run_event(
            notifier,
            run_dir,
            "final-stage-started",
            "Lux evolution final stage started",
            f"Finalists: {', '.join(candidate.candidate_id for candidate in finalists)}",
            priority=4,
            tags=("hourglass_flowing_sand",),
        )
    evaluate_jobs(final_jobs)
    anchors = _evaluation_anchors(args)
    baseline_dir = run_dir / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_evaluations = {}
    for base_name, checkpoint in (
        ((name, Path(getattr(args, f"{name}_checkpoint"))) for name in _active_base_names(args)) if finalists else ()
    ):
        baseline_path = baseline_dir / f"final-{base_name}.json"
        baseline_context = {
            "schema_version": 2,
            "metric_schema_version": _METRIC_SCHEMA_VERSION,
            "base_name": base_name,
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_descriptors": manifest.get("checkpoint_descriptors", {}),
            "seed_start": args.final_seed_start,
            "seed_count": args.final_seeds,
            "max_turns": args.max_turns,
        }
        if baseline_path.exists():
            baseline_evaluations[base_name] = json.loads(baseline_path.read_text(encoding="utf-8"))
            if (
                int(manifest.get("schema_version", 1)) >= 2
                and baseline_evaluations[base_name].get("_context") != baseline_context
            ):
                message = f"Baseline evaluation context does not match the run manifest: {baseline_path}"
                raise ValueError(message)
        else:
            baseline_evaluations[base_name] = evaluate_against_league(
                LeagueMember(f"{base_name}-baseline-eval", checkpoint),
                anchors,
                seed_start=args.final_seed_start,
                seed_count=args.final_seeds,
                device=str(device),
                max_turns=args.max_turns,
            )
            baseline_evaluations[base_name]["_context"] = baseline_context
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
        guarded = args.curriculum_profile != "legacy"
        acceptance = acceptance_report(
            evaluations,
            baseline_evaluations,
            seed=args.seed,
            enforce_teacher_guard=guarded,
            teacher_noninferiority_margin=args.teacher_noninferiority_margin,
            require_survival=guarded,
            require_stranded_fuel=guarded,
        )
        illegal_action_count = sum(
            int(result.metrics.get("reflection", {}).get("diagnostics", {}).get("illegal_action_count", 0))
            for result in values
        )
        acceptance["illegal_action_count"] = illegal_action_count
        acceptance["promote"] = bool(acceptance["promote"] and illegal_action_count == 0)
        acceptance["failure_reasons"] = [
            reason
            for reason, failed in (
                (
                    "paired_score_or_latency",
                    not all(report["score_latency_passes"] for report in acceptance["architectures"].values()),
                ),
                (
                    "teacher_guard",
                    not all(report["teacher_guard_passes"] for report in acceptance["architectures"].values()),
                ),
                (
                    "city_survival",
                    not all(report["survival_passes"] for report in acceptance["architectures"].values()),
                ),
                (
                    "stranded_fuel_regression",
                    not all(report["stranded_fuel_passes"] for report in acceptance["architectures"].values()),
                ),
                ("illegal_action", illegal_action_count != 0),
            )
            if failed
        ]
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
    integrity = _run_integrity_report(run_dir, manifest, candidates, results, medium, finalists)
    promoted = (
        next((row["candidate_id"] for row in ranking if row["acceptance"]["promote"]), None)
        if integrity["valid_for_promotion"]
        else None
    )
    summary = {
        "status": "completed",
        "medium_selection": [candidate.candidate_id for candidate in medium],
        "final_selection": [candidate.candidate_id for candidate in finalists],
        "ranking": ranking,
        "integrity": integrity,
        "promoted_candidate": promoted,
    }
    store.write_json(run_dir / "summary.json", summary)
    _notify_run_event(
        notifier,
        run_dir,
        "run-completed",
        "Lux evolution completed",
        (
            f"Promoted candidate: {promoted or 'none'}\n"
            f"Integrity valid: {integrity['valid_for_promotion']}\n"
            f"Finalists: {', '.join(candidate.candidate_id for candidate in finalists)}"
        ),
        priority=4,
        tags=("tada", "white_check_mark"),
    )
    print(json.dumps(summary, sort_keys=True))
    if job_api_server is not None:
        job_api_server.close()
        atexit.unregister(job_api_server.close)


if __name__ == "__main__":
    parsed_args = parse_args()
    notification_run_dir = Path(parsed_args.run_dir)
    evolution_notifier = EvolutionNotifier.from_environment(notification_run_dir)
    try:
        main(parsed_args, evolution_notifier)
    except KeyboardInterrupt:
        _notify_run_event(
            evolution_notifier,
            notification_run_dir,
            f"run-interrupted:{time.time_ns()}",
            "Lux evolution interrupted",
            "The process received Ctrl+C or another keyboard interrupt.",
            priority=3,
            tags=("warning",),
        )
        raise
    except Exception as error:
        failure_traceback = traceback.format_exc()[-6000:]
        _notify_run_event(
            evolution_notifier,
            notification_run_dir,
            f"run-failed:{time.time_ns()}",
            "Lux evolution failed",
            f"{type(error).__name__}: {error}\n\n{failure_traceback}",
            priority=5,
            tags=("rotating_light", "x"),
        )
        raise
