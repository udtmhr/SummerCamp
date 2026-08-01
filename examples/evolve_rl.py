from __future__ import annotations

# ruff: noqa: BLE001, C901, INP001, PLR0912, PLR0913, PLR0915, S311, S607
import argparse
import atexit
import copy
import gzip
import json
import os
import random
import shutil
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from luxai2021.imitation.agent import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.imitation.data import ReplayBatchSampler
from luxai2021.imitation.distillation import LuxDistillationDataset, compact_distillation_collate
from luxai2021.imitation.model import load_bc_checkpoint
from luxai2021.rl.evaluation import LeagueMember, acceptance_report, evaluate_against_league
from luxai2021.rl.evolution import (
    CandidateResult,
    CodexCandidateGenerator,
    EvolutionCandidate,
    EvolutionJob,
    EvolutionStore,
    FilesystemJobQueue,
    add_candidate_reflection,
    initial_candidate,
    mutate_candidate,
    select_elites,
)
from luxai2021.rl.job_api import JOB_API_VERSION, JobApiClient, JobApiServer, extract_artifact_directory
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent
from luxai2021.rl.ppo import PPOTrainer, collect_episode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from luxai2021.env.agent import Agent

DEFAULT_UNET = "models/distilled/unet_v3/best.pt"
DEFAULT_RESATTN8 = "models/distilled/resattn8_v2/best.pt"
DEFAULT_TEACHER = "models/teachers/lux_2021_first_place/062179520_weights.pt"
DEFAULT_TEACHER_CACHE = "models/teachers/lux_2021_first_place/cache"
DEFAULT_PREPARED_CACHE = "models/teachers/lux_2021_first_place/prepared"


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
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


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
    episodes_per_update: int,
    bc_batch_size: int,
    bc_replays: int,
    use_bc_anchor: bool,
    device: torch.device,
    seed: int,
    max_turns: int,
    resume_from: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    actor_critic = FullTurnActorCritic.from_checkpoint(base_checkpoint, device)
    if device.type == "cuda":
        actor_critic.policy.to(memory_format=torch.channels_last)
    reference_policy = copy.deepcopy(actor_critic.policy).to(device)
    snapshot = copy.deepcopy(actor_critic).eval().requires_grad_(requires_grad=False)
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
    trainer = PPOTrainer(
        actor_critic,
        reference_policy,
        candidate.ppo_config,
        device,
        bc_batch_provider=bc_provider,
    )
    start_update = 0
    if resume_from is not None:
        start_update = trainer.load_training_checkpoint(
            resume_from,
            source_checkpoint=str(base_checkpoint),
            reward_program=candidate.reward_program,
        )
    pool = opponent_factories(
        base_checkpoint=base_checkpoint,
        other_checkpoint=other_checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        snapshot=snapshot,
        device=device,
    )
    rng = random.Random(seed)
    deadline = time.monotonic() + seconds
    history = []
    diagnostic_events: list[dict[str, object]] = []
    update = start_update
    while time.monotonic() < deadline or update == 0:
        episodes = []
        for episode_index in range(episodes_per_update):
            opponent_key = candidate.opponent_mix.choose(rng)
            opponent_name, factory = pool[opponent_key]
            episode_seed = seed + update * episodes_per_update + episode_index
            episodes.append(
                collect_episode(
                    actor_critic,
                    factory,
                    candidate.reward_program,
                    device=device,
                    seed=episode_seed,
                    opponent_name=opponent_name,
                    max_turns=max_turns,
                )
            )
        metrics = trainer.update(episodes)
        diagnostic_events.extend(event for episode in episodes for event in episode.diagnostic_events)
        metrics["update"] = update
        metrics["elapsed_seconds"] = seconds - max(0.0, deadline - time.monotonic())
        history.append(metrics)
        trainer.save_training_checkpoint(
            output_dir / "latest_rl.pt",
            source_checkpoint=str(base_checkpoint),
            reward_program=candidate.reward_program,
            update=update,
            metrics=metrics,
        )
        update += 1
        if seconds <= 0:
            break
    _, source = load_bc_checkpoint(str(base_checkpoint), "cpu")
    summary = {
        "candidate_id": candidate.candidate_id,
        "base_name": base_name,
        "base_checkpoint": str(base_checkpoint),
        "reward_program": candidate.reward_program.to_dict(),
        "ppo_config": asdict(candidate.ppo_config),
        "opponent_mix": asdict(candidate.opponent_mix),
        "history": history,
        "diagnostic_events": diagnostic_events,
    }
    actor_critic.export_policy(
        output_dir / "best.pt",
        epoch=max(0, update - 1),
        metrics={"validation": history[-1], "ppo": history[-1]},
        split=source["split"],
        metadata=summary,
    )
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir / "best.pt", summary


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
    eval_seeds: int,
    eval_seed_start: int,
) -> CandidateResult:
    started_at = time.monotonic()
    output_dir = Path(args.run_dir) / "artifacts" / candidate.candidate_id / stage / base_name
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
            episodes_per_update=args.episodes_per_update,
            bc_batch_size=args.bc_batch_size,
            bc_replays=args.bc_replays,
            use_bc_anchor=not args.no_bc_anchor,
            device=device,
            seed=args.seed + candidate.generation * 10_000 + candidate.island * 100,
            max_turns=args.max_turns,
            resume_from=(
                Path(args.run_dir)
                / "artifacts"
                / candidate.candidate_id
                / "short-resattn8"
                / "resattn8"
                / "latest_rl.pt"
                if stage == "medium-resattn8"
                else None
            ),
        )
        anchors = [
            LeagueMember("unet-base", Path(args.unet_checkpoint)),
            LeagueMember("resattn8-base", Path(args.resattn8_checkpoint)),
            LeagueMember("first-place", Path(args.teacher_checkpoint), "first-place"),
        ]
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
        return CandidateResult(
            candidate.candidate_id,
            stage,
            "failed",
            0.0,
            0.0,
            float("inf"),
            time.monotonic() - started_at,
            {},
            error=f"{type(error).__name__}: {error}",
        )


def execute_evolution_job(
    job: EvolutionJob,
    *,
    args: argparse.Namespace,
    device: torch.device,
    store: EvolutionStore,
) -> CandidateResult:
    candidates = {candidate.candidate_id: candidate for candidate in store.candidates()}
    candidate = candidates[job.candidate_id]
    prior_results = store.results()
    existing = next(
        (result for result in prior_results if result.candidate_id == job.candidate_id and result.stage == job.stage),
        None,
    )
    if existing is not None:
        return existing
    checkpoints = {
        "unet": (Path(args.unet_checkpoint), Path(args.resattn8_checkpoint)),
        "resattn8": (Path(args.resattn8_checkpoint), Path(args.unet_checkpoint)),
    }
    base_checkpoint, other_checkpoint = checkpoints[job.base_name]
    result = candidate_result(
        candidate,
        stage=job.stage,
        base_name=job.base_name,
        base_checkpoint=base_checkpoint,
        other_checkpoint=other_checkpoint,
        args=args,
        device=device,
        seconds=job.seconds,
        eval_seeds=job.eval_seeds,
        eval_seed_start=job.eval_seed_start,
    )
    result = add_candidate_reflection(result, candidate, candidates, prior_results)
    store.save_result(result)
    return result


def _apply_coordinator_manifest(args: argparse.Namespace, manifest: Mapping[str, object]) -> None:
    coordinator_args = manifest.get("arguments", {})
    if not isinstance(coordinator_args, dict):
        raise TypeError("Coordinator manifest arguments are invalid")
    for name in ("seed", "episodes_per_update", "bc_batch_size", "bc_replays", "no_bc_anchor", "max_turns"):
        if name in coordinator_args:
            setattr(args, name, coordinator_args[name])


def _sync_api_claim(store: EvolutionStore, claim: Mapping[str, object]) -> tuple[EvolutionJob, str]:
    if int(claim.get("api_version", 0)) != JOB_API_VERSION:
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
        stale_result.unlink(missing_ok=True)
    input_artifact = claim.get("input_artifact")
    if input_artifact is not None:
        if not isinstance(input_artifact, dict):
            raise TypeError("Input artifact descriptor is invalid")
        destination = (
            store.run_dir
            / "artifacts"
            / job.candidate_id
            / str(input_artifact["stage"])
            / str(input_artifact["base_name"])
        )
        extract_artifact_directory(str(input_artifact["zip_base64"]), destination)
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
                job, lease_id = _sync_api_claim(store, claim)
                _apply_coordinator_manifest(args, claim["manifest"])
                claimed = (job, lease_id)
        else:
            claimed = queue.claim(worker_id)
        if claimed is None:
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
        result = execute_evolution_job(job, args=args, device=device, store=store)
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
        print(json.dumps({"worker_id": worker_id, "job_id": job.job_id, "status": result.status}, sort_keys=True))


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


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    if args.job_api_url:
        raise ValueError("--job-api-url is only valid with --worker")
    if args.islands < 1 or args.initial_per_island < 1 or args.generations < 0:
        raise ValueError("Population sizes must be positive")
    run_dir = Path(args.run_dir)
    if args.overwrite_run and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    device = resolve_device(args.device)
    store = EvolutionStore(run_dir)
    manifest = {
        "schema_version": 1,
        "created_at": time.time(),
        "git_revision": git_revision(repository),
        "arguments": vars(args),
        "device": str(device),
        "codex_available": shutil.which(args.codex_executable) is not None,
        "bases": {"unet": args.unet_checkpoint, "resattn8": args.resattn8_checkpoint},
    }
    store.save_manifest(manifest)
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
        known = {(result.candidate_id, result.stage) for result in results}
        for stored in store.results():
            key = (stored.candidate_id, stored.stage)
            if key not in known:
                results.append(stored)
                known.add(key)

    def evaluate_jobs(jobs: list[EvolutionJob]) -> None:
        if args.dry_run or not jobs:
            return
        refresh_results()
        completed_keys = {(result.candidate_id, result.stage) for result in results}
        jobs = [job for job in jobs if (job.candidate_id, job.stage) not in completed_keys]
        if not jobs:
            return
        if queue is None:
            for job in jobs:
                result = execute_evolution_job(job, args=args, device=device, store=store)
                results.append(result)
                print(json.dumps(asdict(result), sort_keys=True))
            return
        for job in jobs:
            queue.enqueue(job)
            print(json.dumps({"job_id": job.job_id, "status": "enqueued"}, sort_keys=True))
        expected = {(job.candidate_id, job.stage) for job in jobs}
        deadline = time.monotonic() + args.job_timeout_seconds
        worker_id = f"coordinator-{socket.gethostname()}-{os.getpid()}"
        while True:
            refresh_results()
            completed = {(result.candidate_id, result.stage) for result in results}
            if expected <= completed:
                return
            queue.recover_stale(args.recover_stale_job_seconds)
            if time.monotonic() >= deadline:
                missing = sorted(expected - completed)
                message = f"Timed out waiting for distributed jobs: {missing}"
                raise TimeoutError(message)
            claimed = None if args.coordinator_only else queue.claim(worker_id)
            if claimed is not None:
                job, claimed_path = claimed
                result = execute_evolution_job(job, args=args, device=device, store=store)
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
        global_elites = select_elites(candidates, results, count=2)
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
                parent = _best_parent(candidates, results, island)
                parents = [parent]
                if generation % 2 == 0:
                    parents.extend(elite for elite in global_elites if elite.candidate_id != parent.candidate_id)
                    parents = parents[:2]
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
                        )
                else:
                    child = mutate_candidate(
                        parent,
                        generation=generation,
                        island=island,
                        seed=args.seed + generation * 1000 + island,
                    )
                register(child)
            wave.append(short_job(child))
        evaluate_jobs(wave)

    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "candidates": len(candidates), "dry_run": True}))
        return

    medium = select_elites(candidates, results, count=args.medium_count)
    evaluate_jobs(
        [
            EvolutionJob(
                candidate.candidate_id,
                "medium-resattn8",
                "resattn8",
                max(0, args.medium_seconds - args.short_seconds),
                args.medium_seeds,
                args.screening_seed_start + 10_000,
            )
            for candidate in medium
        ]
    )

    finalists = select_elites(candidates, results, count=args.final_count)
    final_jobs = [
        EvolutionJob(
            candidate.candidate_id,
            f"final-{base_name}",
            base_name,
            args.final_seconds,
            args.final_seeds,
            args.final_seed_start,
        )
        for candidate in finalists
        for base_name in ("unet", "resattn8")
    ]
    evaluate_jobs(final_jobs)
    anchors = [
        LeagueMember("unet-base", Path(args.unet_checkpoint)),
        LeagueMember("resattn8-base", Path(args.resattn8_checkpoint)),
        LeagueMember("first-place", Path(args.teacher_checkpoint), "first-place"),
    ]
    baseline_dir = run_dir / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_evaluations = {}
    for base_name, checkpoint in (
        (
            ("unet", Path(args.unet_checkpoint)),
            ("resattn8", Path(args.resattn8_checkpoint)),
        )
        if finalists
        else ()
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
    summary = {"ranking": ranking, "promoted_candidate": promoted}
    store.write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if job_api_server is not None:
        job_api_server.close()
        atexit.unregister(job_api_server.close)


if __name__ == "__main__":
    main()
