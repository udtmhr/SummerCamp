# ruff: noqa: ANN001, ANN003, ANN201, ANN202, PLR2004, PT011, S101
from __future__ import annotations

import copy
import json
import threading

import numpy as np
import pytest
import torch

from examples.evolve_rl import _sync_api_claim
from luxai2021.env.agent import Agent
from luxai2021.game.actions import MoveAction
from luxai2021.game.game import Game
from luxai2021.game.match_controller import MatchController
from luxai2021.game.position import Position
from luxai2021.imitation.masking import monotonically_tighten_legal_mask
from luxai2021.imitation.model import (
    POLICY_SCHEMA_FIRST_PLACE_FLAT,
    LuxBehaviorCloningModel,
    ModelConfig,
    load_bc_checkpoint,
)
from luxai2021.rl.batched_rollout import ActorCriticBatcher, InferenceBatcher
from luxai2021.rl.evaluation import acceptance_report, paired_seed_deltas
from luxai2021.rl.evolution import (
    CandidateResult,
    EvolutionCandidate,
    EvolutionJob,
    EvolutionStore,
    FilesystemJobQueue,
    OpponentMix,
    add_candidate_reflection,
    build_codex_prompt,
    initial_candidate,
    mutate_candidate,
    proposal_schema,
)
from luxai2021.rl.job_api import JobApiClient, JobApiServer, extract_artifact_directory
from luxai2021.rl.metrics import GameMetrics
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent
from luxai2021.rl.ppo import PPOConfig, PPOTrainer, collect_episode, collect_episodes_batched
from luxai2021.rl.reward import RewardProgram, default_reward_program


def _small_policy():
    return LuxBehaviorCloningModel(
        ModelConfig(
            base_channels=4,
            feature_channels=8,
            cycle_embedding_dim=2,
            phase_embedding_dim=2,
            board_size_embedding_dim=2,
            policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT,
        )
    )


def _metrics(**updates):
    values = {
        "turn": 0.0,
        "night": 0.0,
        "cycle": 0.0,
        "city_tiles": 0.0,
        "units": 0.0,
        "workers": 0.0,
        "carts": 0.0,
        "research": 0.0,
        "city_fuel": 0.0,
        "city_survival": 0.0,
        "cargo_fuel": 0.0,
        "collected_fuel": 0.0,
        "fuel_generated": 0.0,
        "city_tiles_built": 0.0,
    }
    values.update(updates)
    return GameMetrics(0, values)


def test_reward_program_is_bounded_and_preserves_terminal_reward():
    program = default_reward_program()
    previous = _metrics(city_tiles=-0.5, city_survival=-0.5)
    following = _metrics(city_tiles=0.5, city_survival=0.5)

    breakdown = program.reward(previous, following, terminal_outcome=1.0)

    assert breakdown.terminal == 1.0
    assert 1.0 < breakdown.total <= 1.5
    assert set(breakdown.components) == {component.name for component in program.components}


@pytest.mark.parametrize(
    "expression",
    [
        {"op": "metric", "name": "secret"},
        {"op": "import", "name": "os"},
        {"op": "constant", "value": float("nan")},
    ],
)
def test_reward_program_rejects_unsafe_expressions(expression):
    with pytest.raises(ValueError):
        RewardProgram.from_dict(
            {
                "components": [{"name": "unsafe", "expression": expression, "weight": 1.0}],
                "reward_scale": 0.2,
                "gamma": 0.995,
            }
        )


def test_candidate_round_trip_and_resume_store(tmp_path):
    candidate = mutate_candidate(initial_candidate(island=1, seed=7), generation=2, island=1, seed=8)
    store = EvolutionStore(tmp_path)
    store.save_candidate(candidate)

    loaded = store.candidates()

    assert loaded == [candidate]
    assert EvolutionCandidate.from_dict(candidate.to_dict()) == candidate
    assert proposal_schema()["additionalProperties"] is False
    assert sum(vars(OpponentMix()).values()) == pytest.approx(1.0)


def test_candidate_content_hash_detects_manual_edit():
    value = initial_candidate(island=0, seed=42).to_dict()
    value["reward_program"]["reward_scale"] = 0.3

    with pytest.raises(ValueError, match="content hash"):
        EvolutionCandidate.from_dict(value)


def test_actor_critic_exports_behavior_cloning_compatible_policy(tmp_path):
    actor = FullTurnActorCritic(_small_policy())
    path = tmp_path / "best.pt"
    actor.export_policy(
        path,
        epoch=0,
        metrics={"validation": {"loss": 0.0}},
        split={"train": [], "validation": [], "test": []},
        metadata={"candidate_id": "test"},
    )

    loaded, checkpoint = load_bc_checkpoint(str(path))

    assert loaded.config.policy_schema == POLICY_SCHEMA_FIRST_PLACE_FLAT
    assert checkpoint["rl_training"]["candidate_id"] == "test"
    assert not any(name.startswith("value_head") for name in checkpoint["model"])


def test_short_full_turn_ppo_smoke_updates_finite_parameters(tmp_path):
    actor = FullTurnActorCritic(_small_policy())
    snapshot = copy.deepcopy(actor).eval()
    before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    episode = collect_episode(
        actor,
        lambda: RolloutAgent(snapshot, device="cpu", deterministic=True),
        default_reward_program(),
        device=torch.device("cpu"),
        seed=7,
        opponent_name="small",
        max_turns=4,
    )
    trainer = PPOTrainer(
        actor,
        copy.deepcopy(actor.policy),
        PPOConfig(update_epochs=1, minibatch_turns=4, bc_coefficient=0.0),
        torch.device("cpu"),
    )

    metrics = trainer.update([episode])

    assert episode.records
    assert metrics["decisions"] > 0
    assert all(torch.isfinite(value).all() for value in actor.parameters())
    assert any(not torch.equal(before[name], value) for name, value in actor.named_parameters())
    checkpoint_path = tmp_path / "latest_rl.pt"
    trainer.save_training_checkpoint(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        update=0,
        metrics=metrics,
    )
    resumed_actor = FullTurnActorCritic(_small_policy())
    resumed = PPOTrainer(
        resumed_actor,
        copy.deepcopy(resumed_actor.policy),
        PPOConfig(update_epochs=1, minibatch_turns=4, bc_coefficient=0.0),
        torch.device("cpu"),
    )
    next_update = resumed.load_training_checkpoint(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
    )
    assert next_update == 1
    assert all(
        torch.equal(value, dict(resumed_actor.named_parameters())[name]) for name, value in actor.named_parameters()
    )


def test_training_checkpoint_v2_restores_budget_progress_and_legacy_estimate(tmp_path):
    actor = FullTurnActorCritic(_small_policy())
    trainer = PPOTrainer(actor, copy.deepcopy(actor.policy), PPOConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "latest_rl.pt"
    trainer.save_training_checkpoint(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        update=3,
        metrics={"elapsed_seconds": 25.0},
        training_state={
            "cumulative_decisions": 1234,
            "cumulative_turns": 456,
            "cumulative_episodes": 7,
            "elapsed_seconds": 25.0,
        },
    )

    restored = trainer.load_training_state(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
    )

    assert restored.next_update == 4
    assert restored.cumulative_decisions == 1234
    assert restored.cumulative_turns == 456
    assert restored.cumulative_episodes == 7

    legacy = torch.load(checkpoint_path, weights_only=False)
    legacy["schema_version"] = 1
    legacy.pop("training_state")
    legacy.pop("torch_rng_state")
    legacy.pop("cuda_rng_state_all")
    legacy["metrics"] = {"elapsed_seconds": 50.0}
    legacy_path = tmp_path / "legacy_rl.pt"
    torch.save(legacy, legacy_path)
    restored_legacy = trainer.load_training_state(
        legacy_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        legacy_target_decisions=1000,
        legacy_stage_seconds=100,
    )
    assert restored_legacy.cumulative_decisions == 500


def test_inference_batcher_and_parallel_rollout_batch_requests():
    barrier = threading.Barrier(2)

    def infer(values):
        return [value * 2 for value in values]

    batcher = InferenceBatcher(infer, max_batch_size=2, wait_seconds=0.05, name="test-inference")
    outputs = [None, None]

    def submit(index):
        barrier.wait()
        outputs[index] = batcher.submit(index + 1)

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    metrics = batcher.metrics()
    batcher.close()
    assert outputs == [2, 4]
    assert metrics["max_batch_size"] == 2

    actor = FullTurnActorCritic(_small_policy()).eval()
    snapshot = copy.deepcopy(actor).eval()
    actor_batcher = ActorCriticBatcher(actor, torch.device("cpu"), 2, name="test-rollout-inference")
    try:
        episodes = collect_episodes_batched(
            actor,
            [
                (lambda: RolloutAgent(snapshot, device="cpu", deterministic=True), 31, "snapshot"),
                (lambda: RolloutAgent(snapshot, device="cpu", deterministic=True), 32, "snapshot"),
            ],
            default_reward_program(),
            device=torch.device("cpu"),
            inference_backend=actor_batcher.submit,
            max_turns=4,
        )
    finally:
        rollout_metrics = actor_batcher.metrics()
        actor_batcher.close()
    assert len(episodes) == 2
    assert all(episode.records for episode in episodes)
    assert rollout_metrics["samples"] == sum(len(episode.records) for episode in episodes)


def test_dry_run_archive_is_json_serializable():
    candidate = initial_candidate(island=0, seed=42)
    encoded = json.dumps(candidate.to_dict(), sort_keys=True)
    assert candidate.candidate_id in encoded


def _league_evaluation(outcomes, p95=0.1):
    games = []
    for seed, outcome in enumerate(outcomes, start=10):
        games.append({"anchor": "anchor", "seed": seed, "orientation": 0, "outcome": outcome})
    return {"games": games, "candidate_inference_p95_seconds": p95}


def test_paired_acceptance_requires_positive_both_architectures():
    baseline = _league_evaluation([-1.0, 0.0, 0.0])
    improved = _league_evaluation([0.0, 1.0, 1.0])

    seed_deltas, anchor_deltas = paired_seed_deltas(improved, baseline)
    report = acceptance_report(
        {"unet": improved, "resattn8": improved},
        {"unet": baseline, "resattn8": baseline},
        seed=3,
    )

    assert all(delta > 0 for delta in seed_deltas.values())
    assert anchor_deltas["anchor"] > 0
    assert report["combined_bootstrap_lcb"] > 0
    assert report["promote"] is True


def test_legal_mask_tightening_never_reenables_existing_illegal_actions():
    existing = np.array([True, False, True, False])
    additional = np.array([True, True, False, False])

    tightened = monotonically_tighten_legal_mask(existing, additional)

    assert tightened.tolist() == [True, False, False, False]
    assert not np.any(tightened & ~existing)


def test_rollout_mask_does_not_inspect_offboard_actions():
    game = Game({"seed": 17})
    unit = next(iter(game.state["teamStates"][0]["units"].values()))
    game.map.get_cell_by_pos(unit.pos).units.pop(unit.id)
    unit.pos = Position(0, 0)
    game.map.get_cell(0, 0).units[unit.id] = unit
    agent = RolloutAgent(FullTurnActorCritic(_small_policy()), device="cpu", deterministic=True)
    agent.set_team(0)
    agent.game_start(game)

    agent.process_turn(game, 0)

    assert agent.records


def test_engine_records_night_city_loss_and_invalid_action_turns():
    game = Game({"seed": 13})
    tile = game.spawn_city_tile(0, 0, 0)
    city = game.cities[tile.city_id]
    city.fuel = 0.0
    game.state["turn"] = 30

    game.handle_night()

    city_event = game.diagnostic_events[-1]
    assert city_event["event"] == "city_destroyed_night_fuel"
    assert city_event["turn"] == 30
    assert city_event["city_tiles_lost"] == 1
    assert city_event["fuel_deficit"] > 0

    controller = MatchController(game, [Agent(), Agent()])
    controller.take_action(MoveAction(0, "missing", "n"))
    illegal = game.diagnostic_events[-1]
    assert illegal["event"] == "illegal_action"
    assert illegal["turn"] == 30
    assert illegal["details"]["unit_id"] == "missing"


def test_reflection_contains_parent_changes_improvement_and_diagnostic_turns():
    parent = initial_candidate(island=0, seed=3)
    child = mutate_candidate(parent, generation=1, island=0, seed=4)
    parent_result = CandidateResult(parent.candidate_id, "short-resattn8", "completed", 0.4, 0.2, 0.02, 2.0, {})
    child_result = CandidateResult(
        child.candidate_id,
        "short-resattn8",
        "completed",
        0.6,
        0.3,
        0.01,
        3.0,
        {
            "training": {
                "diagnostic_events": [
                    {"event": "city_destroyed_night_fuel", "turn": 31, "city_tiles_lost": 2},
                    {"event": "illegal_action", "turn": 32, "action_class": "MoveAction"},
                ]
            }
        },
    )

    reflected = add_candidate_reflection(
        child_result,
        child,
        {parent.candidate_id: parent, child.candidate_id: child},
        [parent_result],
    )
    reflection = reflected.metrics["reflection"]

    assert reflection["diagnostics"]["city_tile_loss_turns"] == [31]
    assert reflection["diagnostics"]["illegal_action_turns"] == [32]
    assert reflected.metrics["training"]["diagnostic_event_count"] == 2
    assert "diagnostic_events" not in reflected.metrics["training"]
    comparison = reflection["parent_comparisons"][0]
    assert comparison["changes"]
    assert comparison["improvement"]["score_rate_delta"] == pytest.approx(0.2)
    prompt = build_codex_prompt([child], [reflected], island=0, generation=2)
    assert '"city_tile_loss_turns": [' in prompt
    assert "31" in prompt
    assert '"score_rate_delta": 0.199' in prompt


def test_filesystem_job_queue_claim_and_complete(tmp_path):
    queue = FilesystemJobQueue(tmp_path)
    job = EvolutionJob("candidate", "short-resattn8", "resattn8", 0, 1, 100)
    queue.enqueue(job)

    claimed = queue.claim("other-pc")

    assert claimed is not None
    claimed_job, claimed_path = claimed
    assert claimed_job == job
    result = CandidateResult("candidate", "short-resattn8", "completed", 0.5, 0.5, 0.0, 1.0, {})
    queue.complete(claimed_path, result)
    assert queue.outstanding_ids() == set()
    assert (tmp_path / "jobs" / "completed" / f"{job.job_id}.json").exists()


def test_filesystem_job_completion_recovers_a_requeued_lease(tmp_path):
    queue = FilesystemJobQueue(tmp_path)
    job = EvolutionJob("candidate", "short-resattn8", "resattn8", 1200, 4, 100)
    queue.enqueue(job)
    claimed_job, claimed_path = queue.claim("pc1")
    assert claimed_job == job
    claimed_path.replace(queue.pending_dir / f"{job.job_id}.json")

    result = CandidateResult("candidate", job.stage, "completed", 0.5, 0.5, 0.0, 10.0, {})
    queue.complete(claimed_path, result)

    assert queue.outstanding_ids() == set()
    assert (queue.completed_dir / f"{job.job_id}.json").exists()


def test_job_api_claim_context_and_upload_artifacts(tmp_path):  # noqa: PLR0915
    coordinator_dir = tmp_path / "coordinator"
    store = EvolutionStore(coordinator_dir)
    candidate = initial_candidate(island=0, seed=7)
    store.save_candidate(candidate)
    store.save_manifest({"schema_version": 1, "arguments": {"seed": 7}})
    queue = FilesystemJobQueue(coordinator_dir)
    job = EvolutionJob(candidate.candidate_id, "short-resattn8", "resattn8", 0, 1, 100)
    queue.enqueue(job)
    token = f"test-{candidate.candidate_id}"
    server = JobApiServer("127.0.0.1:0", run_dir=coordinator_dir, queue=queue, token=token)
    server.start()
    try:
        host, port = server.address
        health = JobApiClient(f"http://{host}:{port}").health()
        assert health == {"status": "ok", "api_version": 1}
        unauthorized = JobApiClient(f"http://{host}:{port}", token=f"wrong-{candidate.candidate_id}")
        with pytest.raises(RuntimeError, match="HTTP 401"):
            unauthorized.claim("intruder")
        client = JobApiClient(f"http://{host}:{port}", token=token)
        claim = client.claim("ws3")

        assert claim is not None
        assert claim["job"]["candidate_id"] == candidate.candidate_id
        assert claim["candidate"]["candidate_id"] == candidate.candidate_id
        assert claim["manifest"]["arguments"]["seed"] == 7

        worker_artifacts = tmp_path / "worker-artifacts"
        worker_artifacts.mkdir()
        (worker_artifacts / "best.pt").write_bytes(b"policy")
        (worker_artifacts / "latest_rl.pt").write_bytes(b"training")
        partial_artifacts = tmp_path / "partial-artifacts"
        partial_artifacts.mkdir()
        (partial_artifacts / "latest_rl.pt").write_bytes(b"partial-training")
        client.heartbeat(
            lease_id=claim["lease_id"],
            job=job,
            artifact_dir=partial_artifacts,
        )
        partial_checkpoint = (
            coordinator_dir / "artifacts" / candidate.candidate_id / job.stage / job.base_name / "latest_rl.pt"
        )
        assert partial_checkpoint.read_bytes() == b"partial-training"
        result = CandidateResult(
            candidate.candidate_id,
            job.stage,
            "completed",
            0.6,
            0.5,
            0.01,
            2.0,
            {"checkpoint": "/worker/path/best.pt"},
        )
        client.complete(
            lease_id=claim["lease_id"],
            job=job,
            result=result,
            artifact_dir=worker_artifacts,
        )

        medium_job = EvolutionJob(candidate.candidate_id, "medium-resattn8", "resattn8", 0, 1, 200)
        queue.enqueue(medium_job)
        medium_claim = client.claim("other-worker")
        assert medium_claim is not None
        assert medium_claim["input_artifact"]["stage"] == "short-resattn8"
        downloaded = tmp_path / "downloaded-input"
        extract_artifact_directory(medium_claim["input_artifact"]["zip_base64"], downloaded)
        assert (downloaded / "latest_rl.pt").read_bytes() == b"training"
        medium_partial = tmp_path / "medium-partial"
        medium_partial.mkdir()
        (medium_partial / "latest_rl.pt").write_bytes(b"medium-partial-training")
        client.heartbeat(
            lease_id=medium_claim["lease_id"],
            job=medium_job,
            artifact_dir=medium_partial,
        )
        client.release(lease_id=medium_claim["lease_id"], job=medium_job)
        medium_claim = client.claim("replacement-worker")
        assert medium_claim is not None
        assert medium_claim["input_artifact"]["stage"] == "medium-resattn8"
        resumed_medium = tmp_path / "resumed-medium"
        extract_artifact_directory(medium_claim["input_artifact"]["zip_base64"], resumed_medium)
        assert (resumed_medium / "latest_rl.pt").read_bytes() == b"medium-partial-training"
        failed = CandidateResult(
            candidate.candidate_id,
            medium_job.stage,
            "failed",
            0.0,
            0.0,
            float("inf"),
            0.0,
            {},
            "test completion",
        )
        client.complete(
            lease_id=medium_claim["lease_id"],
            job=medium_job,
            result=failed,
            artifact_dir=tmp_path / "missing-artifacts",
        )
    finally:
        server.close()

    coordinator_artifacts = coordinator_dir / "artifacts" / candidate.candidate_id / job.stage / job.base_name
    assert (coordinator_artifacts / "best.pt").read_bytes() == b"policy"
    assert (coordinator_artifacts / "latest_rl.pt").read_bytes() == b"training"
    assert queue.outstanding_ids() == set()
    stored = next(item for item in store.results() if item.stage == job.stage)
    assert stored.metrics["checkpoint"] == str(coordinator_artifacts / "best.pt")


def test_api_claim_discards_stale_local_result(tmp_path):
    store = EvolutionStore(tmp_path)
    candidate = initial_candidate(island=0, seed=9)
    stale = CandidateResult(
        candidate.candidate_id,
        "short-resattn8",
        "failed",
        0.0,
        0.0,
        float("inf"),
        0.0,
        {},
        "missing model from an older coordinator run",
    )
    store.save_result(stale)
    job = EvolutionJob(candidate.candidate_id, stale.stage, "resattn8", 0, 1, 100)
    claim = {
        "api_version": 1,
        "lease_id": "ws3--job.json",
        "job": job.to_dict(),
        "candidate": candidate.to_dict(),
        "candidates": [candidate.to_dict()],
        "results": [],
        "manifest": {"schema_version": 1, "arguments": {}},
        "input_artifact": None,
    }

    claimed_job, _ = _sync_api_claim(store, claim)

    assert claimed_job == job
    assert store.results() == []
