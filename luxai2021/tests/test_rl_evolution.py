# ruff: noqa: ANN001, ANN003, ANN201, ANN202, PLR2004, PT011, S101, SLF001
from __future__ import annotations

import copy
import gzip
import json
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.evolve_rl import (
    _active_base_names,
    _apply_coordinator_manifest,
    _archive_unselected_stage_jobs,
    _checkpoint_descriptors,
    _checkpoint_pair,
    _evaluation_anchors,
    _final_training_metrics,
    _is_fatal_cuda_error,
    _load_or_create_stage_selection,
    _save_candidate_provenance,
    _sync_api_claim,
    _validate_candidate_provenance,
    _validate_checkpoint_descriptors,
    _validate_run_kind,
    execute_evolution_job,
)
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
    CodexCandidateGenerator,
    EvolutionCandidate,
    EvolutionJob,
    EvolutionStore,
    FilesystemJobQueue,
    OpponentMix,
    add_candidate_reflection,
    approximate_ast_distance,
    build_codex_prompt,
    canonicalize_candidate_proposal,
    compress_turn_ranges,
    initial_candidate,
    mutate_candidate,
    proposal_schema,
    select_codex_feedback_results,
    validate_candidate_mutation,
)
from luxai2021.rl.job_api import JobApiClient, JobApiServer
from luxai2021.rl.metrics import GameMetrics, MetricContext, metrics_from_game
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent
from luxai2021.rl.ppo import (
    PPOConfig,
    PPOTrainer,
    _checkpoint_cuda_rng_state,
    collect_episode,
    collect_episodes_batched,
    warmup_value_head,
)
from luxai2021.rl.reward import (
    DIRECT_REWARD_METRIC_NAMES,
    GATING_METRIC_NAMES,
    LOWER_IS_BETTER_METRIC_NAMES,
    RewardProgram,
    calibrate_reward_scale,
    default_reward_program,
)


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
        "turns_until_night": 0.0,
        "night_turns_remaining": 0.0,
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
        "min_city_survival": 0.0,
        "city_tiles_at_risk": 0.0,
        "night_fuel_deficit": 0.0,
        "fuel_delivery_coverage": 0.0,
        "city_tile_loss": 0.0,
        "night_fuel_shortage": 0.0,
        "worker_resource_access": 0.0,
        "worker_cargo_fullness": 0.0,
        "unit_capacity_utilization": 0.0,
        "coal_unlocked": 0.0,
        "uranium_unlocked": 0.0,
        "own_min_city_survival": 0.0,
        "own_city_tiles_at_risk": 0.0,
        "own_night_fuel_deficit": 0.0,
        "own_fuel_delivery_coverage": 0.0,
        "own_city_tiles_lost": 0.0,
        "own_night_fuel_shortage": 0.0,
    }
    values.update(updates)
    return GameMetrics(0, values)


def _candidate_proposal(parent, *, mutation_kind, reward_program=None, secondary_parent_ids=()):
    return {
        "reward_program": copy.deepcopy(reward_program or parent.reward_program.to_dict()),
        "ppo_config": copy.deepcopy(vars(parent.ppo_config)),
        "opponent_mix": copy.deepcopy(vars(parent.opponent_mix)),
        "mutation_kind": mutation_kind,
        "primary_parent_id": None if mutation_kind == "restart" else parent.candidate_id,
        "secondary_parent_ids": list(secondary_parent_ids),
        "inheritance_mode": "base" if mutation_kind == "restart" else "policy",
        "mutation_manifest": {"changed_paths": ["reward_program"], "summary": "test proposal"},
        "parameter_constraint_coefficient": 0.0
        if mutation_kind == "restart"
        else parent.parameter_constraint_coefficient,
        "rationale": "test proposal",
    }


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


def test_metric_dsl_evaluates_bounded_board_queries():
    context = MetricContext(
        width=12,
        height=12,
        positions={"own_workers": ((0, 0),), "uranium_tiles": ((11, 11),)},
        sums={},
    )
    metrics = _metrics()
    object.__setattr__(metrics, "context", context)
    program = RewardProgram.from_dict(
        {
            "version": 2,
            "derived_metrics": [
                {
                    "name": "uranium_distance",
                    "expression": {
                        "op": "distance",
                        "source": "own_workers",
                        "target": "uranium_tiles",
                        "reduce": "min",
                    },
                }
            ],
            "components": [
                {
                    "name": "distance",
                    "expression": {"op": "derived", "name": "uranium_distance"},
                    "weight": -1.0,
                }
            ],
            "reward_scale": 0.2,
            "gamma": 0.995,
        }
    )

    potential, values = program.potential(metrics)

    assert -1.0 <= potential <= 1.0
    assert values["distance"] == pytest.approx(1.0)


def test_metric_dsl_exposes_phase_local_city_risk_and_delivery():
    game = Game({"seed": 13})
    own_city = next(city for city in game.cities.values() if city.team == 0)
    opponent_city = next(city for city in game.cities.values() if city.team == 1)
    own_upkeep = float(own_city.get_light_upkeep())
    opponent_upkeep = float(opponent_city.get_light_upkeep())
    own_city.fuel = own_upkeep * 10.0 - 10.0
    opponent_city.fuel = opponent_upkeep * 10.0 + 10.0
    own_worker = next(iter(game.state["teamStates"][0]["units"].values()))
    own_worker.cargo["wood"] = 20
    game.state["turn"] = 29

    metrics = metrics_from_game(game, 0)

    assert metrics.get("turns_until_night") == pytest.approx(1.0 / 30.0)
    assert metrics.get("night_turns_remaining") == 0.0
    assert metrics.get("own_city_tiles_at_risk") == 1.0
    assert metrics.get("city_tiles_at_risk") == -1.0
    assert 0.0 < metrics.get("own_night_fuel_deficit") < 1.0
    assert 0.0 <= metrics.get("own_fuel_delivery_coverage") <= 1.0
    assert metrics.context is not None
    assert metrics.context.positions["own_at_risk_city_tiles"]
    assert metrics.context.positions["own_fuel_carrying_workers"]
    assert metrics.context.sums["own_night_fuel_required"] == pytest.approx(own_upkeep * 10.0)
    assert metrics.context.sums["own_night_fuel_deficit"] == pytest.approx(10.0)
    program = RewardProgram.from_dict(
        {
            "version": 2,
            "derived_metrics": [
                {
                    "name": "delivery_distance",
                    "expression": {
                        "op": "distance",
                        "source": "own_fuel_carrying_workers",
                        "target": "own_at_risk_city_tiles",
                        "reduce": "min",
                    },
                }
            ],
            "components": [
                {
                    "name": "delivery_distance",
                    "expression": {"op": "derived", "name": "delivery_distance"},
                    "weight": -1.0,
                },
                {
                    "name": "raw_deficit",
                    "expression": {"op": "sum", "name": "own_night_fuel_deficit"},
                    "weight": -0.5,
                },
            ],
            "reward_scale": 0.2,
            "gamma": 0.995,
        }
    )
    potential, components = program.potential(metrics)
    assert -1.0 <= potential <= 1.0
    assert set(components) == {"delivery_distance", "raw_deficit"}


def test_city_loss_metric_is_cumulative_for_potential_difference_reward():
    game = Game({"seed": 13})
    own_city = next(city for city in game.cities.values() if city.team == 0)
    opponent_city = next(city for city in game.cities.values() if city.team == 1)
    own_city.fuel = 0.0
    opponent_city.fuel = float(opponent_city.get_light_upkeep()) * 10.0
    game.state["turn"] = 30
    before = metrics_from_game(game, 0)

    game.handle_night()
    after = metrics_from_game(game, 0)

    assert before.get("own_city_tiles_lost") == 0.0
    assert after.get("own_city_tiles_lost") > 0.0
    assert after.get("own_night_fuel_shortage") > 0.0
    assert after.get("city_tile_loss") < 0.0
    program = RewardProgram.from_dict(
        {
            "version": 2,
            "derived_metrics": [],
            "components": [
                {
                    "name": "avoid_city_loss",
                    "expression": {"op": "metric", "name": "own_city_tiles_lost"},
                    "weight": -1.0,
                }
            ],
            "reward_scale": 0.2,
            "gamma": 1.0,
        }
    )
    assert program.reward(before, after).shaping < 0.0
    assert program.reward(after, after).shaping == pytest.approx(0.0)


def test_sparse_component_is_not_used_for_inverse_rms_calibration():
    parent = default_reward_program()
    value = parent.to_dict()
    value["components"].append(
        {"name": "sparse_signal", "expression": {"op": "metric", "name": "cargo_fuel"}, "weight": 1.0}
    )
    child = RewardProgram.from_dict(value)
    transitions = []
    for index in range(300):
        previous = _metrics(city_tiles=-0.5 if index % 2 else 0.5, cargo_fuel=0.0)
        following = _metrics(city_tiles=0.5 if index % 2 else -0.5, cargo_fuel=1.0 if index == 0 else 0.0)
        transitions.append((previous, following))

    calibrated, report = calibrate_reward_scale(parent, child, transitions)

    assert report["child_component_activity"]["sparse_signal"] < 0.01
    assert "sparse_signal" not in report["child_dense_components"]
    assert 0.01 <= calibrated.reward_scale <= 0.5


def test_candidate_round_trip_and_resume_store(tmp_path):
    candidate = mutate_candidate(initial_candidate(island=1, seed=7), generation=2, island=1, seed=8)
    store = EvolutionStore(tmp_path)
    store.save_candidate(candidate)

    loaded = store.candidates()

    assert loaded == [candidate]
    assert EvolutionCandidate.from_dict(candidate.to_dict()) == candidate
    assert proposal_schema()["additionalProperties"] is False
    assert sum(vars(OpponentMix()).values()) == pytest.approx(1.0)


def test_v1_candidate_id_round_trip_is_unchanged():
    proposal = {
        "reward_program": default_reward_program().to_dict(),
        "ppo_config": vars(PPOConfig()),
        "opponent_mix": vars(OpponentMix()),
        "rationale": "legacy",
    }
    proposal["reward_program"]["version"] = 1
    proposal["reward_program"].pop("derived_metrics", None)
    candidate = EvolutionCandidate.from_proposal(
        proposal,
        generation=1,
        island=0,
        parent_ids=(),
        schema_version=1,
    )

    restored = EvolutionCandidate.from_dict(candidate.to_dict())

    assert restored.candidate_id == candidate.candidate_id
    assert restored.to_dict()["schema_version"] == 1


def test_parameter_mutation_keeps_ast_shape_and_validates_island_contract():
    parent = initial_candidate(island=0, seed=12)
    child = mutate_candidate(parent, generation=1, island=0, seed=13)

    validate_candidate_mutation([parent], child)

    assert child.mutation_kind == "parameter"
    assert child.inheritance_mode == "policy_value"
    assert approximate_ast_distance(parent, child) <= 0.2


def test_existing_feature_mutation_uses_safe_phase_and_risk_signs_in_every_generation():
    parent = initial_candidate(island=2, seed=12)
    additions = 0
    for generation in (1, 2):
        for seed in range(100):
            child = mutate_candidate(parent, generation=generation, island=2, seed=seed)
            validate_candidate_mutation([parent], child)
            assert child.mutation_kind == "feature_existing"
            assert child.reward_program.derived_metrics == parent.reward_program.derived_metrics
            added = [
                component for component in child.reward_program.components if component.name.startswith("feature_")
            ]
            for component in added:
                additions += 1
                metric = str(component.expression["name"])
                assert metric in DIRECT_REWARD_METRIC_NAMES
                assert metric not in GATING_METRIC_NAMES
                if metric in LOWER_IS_BETTER_METRIC_NAMES:
                    assert component.weight < 0.0
                else:
                    assert component.weight > 0.0
    assert additions > 0


def test_feature_generated_remains_loadable_but_is_not_proposable():
    parent = initial_candidate(island=2, seed=12)
    proposal = _candidate_proposal(parent, mutation_kind="feature_generated")
    proposal["reward_program"]["derived_metrics"] = [
        {"name": "legacy_generated", "expression": {"op": "count", "selector": "own_workers"}}
    ]
    proposal["reward_program"]["components"].append(
        {
            "name": "legacy_generated",
            "expression": {"op": "derived", "name": "legacy_generated"},
            "weight": 0.2,
        }
    )
    candidate = EvolutionCandidate.from_proposal(
        proposal,
        generation=2,
        island=2,
        parent_ids=(parent.candidate_id,),
    )

    restored = EvolutionCandidate.from_dict(candidate.to_dict())
    mutation_enum = proposal_schema()["properties"]["mutation_kind"]["enum"]

    assert restored == candidate
    assert restored.candidate_id == candidate.candidate_id
    assert "feature_generated" not in mutation_enum
    validate_candidate_mutation([parent], candidate)


@pytest.mark.parametrize(
    ("stagnated", "expected"),
    [
        (False, {"structural": 0.50, "crossover": 0.30, "restart": 0.20}),
        (True, {"structural": 0.40, "crossover": 0.20, "restart": 0.40}),
    ],
)
def test_island3_fallback_has_no_generated_feature_and_expected_distribution(stagnated, expected):
    parent = initial_candidate(island=3, seed=1)
    donor = initial_candidate(island=0, seed=99)
    counts = Counter(
        mutate_candidate(
            parent,
            generation=1,
            island=3,
            seed=seed,
            secondary_parents=(donor,),
            stagnated=stagnated,
        ).mutation_kind
        for seed in range(1000)
    )

    assert "feature_generated" not in counts
    for kind, ratio in expected.items():
        assert counts[kind] / 1000 == pytest.approx(ratio, abs=0.04)


def test_island3_without_secondary_parent_reassigns_crossover_to_structural():
    parent = initial_candidate(island=3, seed=1)
    counts = Counter(mutate_candidate(parent, generation=1, island=3, seed=seed).mutation_kind for seed in range(500))

    assert counts["crossover"] == 0
    assert counts["structural"] / 500 == pytest.approx(0.80, abs=0.05)
    assert counts["restart"] / 500 == pytest.approx(0.20, abs=0.05)


def test_island3_structural_soft_accepts_compound_training_setting_changes():
    parent = initial_candidate(island=3, seed=1)
    structural = next(
        child
        for seed in range(100)
        if (child := mutate_candidate(parent, generation=1, island=3, seed=seed)).mutation_kind == "structural"
    )
    proposal = _candidate_proposal(
        parent,
        mutation_kind="structural",
        reward_program=structural.reward_program.to_dict(),
    )
    proposal["ppo_config"]["learning_rate"] *= 1.1
    proposal["parameter_constraint_coefficient"] = 0.05
    proposal["mutation_manifest"]["changed_paths"].extend(
        ("ppo_config.learning_rate", "parameter_constraint_coefficient")
    )
    ppo_child = EvolutionCandidate.from_proposal(
        proposal,
        generation=1,
        island=3,
        parent_ids=(parent.candidate_id,),
    )
    validate_candidate_mutation([parent], ppo_child)

    zero_constraint = copy.deepcopy(proposal)
    zero_constraint["parameter_constraint_coefficient"] = 0.0
    zero_canonical, zero_constraint_child, zero_report = canonicalize_candidate_proposal(
        zero_constraint,
        [parent],
        generation=1,
        island=3,
    )
    assert zero_constraint_child.inheritance_mode == "policy"
    assert zero_constraint_child.parameter_constraint_coefficient == 0.05
    assert "parameter_constraint_coefficient" in zero_report["corrected_fields"]
    assert zero_canonical["mutation_manifest"]["changed_paths"]

    opponent_proposal = _candidate_proposal(
        parent,
        mutation_kind="structural",
        reward_program=structural.reward_program.to_dict(),
    )
    opponent_proposal["opponent_mix"]["self_base"] -= 0.05
    opponent_proposal["opponent_mix"]["other_base"] += 0.05
    opponent_proposal["parameter_constraint_coefficient"] = 0.05
    opponent_proposal["mutation_manifest"]["changed_paths"].extend(("opponent_mix", "parameter_constraint_coefficient"))
    opponent_child = EvolutionCandidate.from_proposal(
        opponent_proposal,
        generation=1,
        island=3,
        parent_ids=(parent.candidate_id,),
    )
    validate_candidate_mutation([parent], opponent_child)

    both = copy.deepcopy(proposal)
    both["opponent_mix"]["self_base"] -= 0.05
    both["opponent_mix"]["other_base"] += 0.05
    both["mutation_manifest"]["changed_paths"].append("opponent_mix")
    _, both_child, both_report = canonicalize_candidate_proposal(
        both,
        [parent],
        generation=1,
        island=3,
    )
    assert both_child.mutation_kind == "structural"
    assert both_report["mutation_scale"] == "large"
    validate_candidate_mutation([parent], both_child)


def test_island3_structural_soft_accepts_small_and_large_ast_changes():
    parent = initial_candidate(island=3, seed=1)
    too_small_reward = parent.reward_program.to_dict()
    too_small_reward["components"][0]["weight"] *= 1.01
    _, too_small, small_report = canonicalize_candidate_proposal(
        _candidate_proposal(parent, mutation_kind="structural", reward_program=too_small_reward),
        [parent],
        generation=1,
        island=3,
    )
    assert too_small.mutation_kind == "parameter"
    assert too_small.inheritance_mode == "policy_value"
    assert small_report["ast_distance"] < 0.20

    large_reward = {
        "version": 2,
        "derived_metrics": [],
        "components": [
            {
                "name": f"large_{index}",
                "expression": {"op": "square", "value": {"op": "metric", "name": metric}},
                "weight": -1.0 if index % 2 else 1.0,
            }
            for index, metric in enumerate(sorted(DIRECT_REWARD_METRIC_NAMES)[:16])
        ],
        "reward_scale": 0.4,
        "gamma": 0.91,
    }
    _, too_large, large_report = canonicalize_candidate_proposal(
        _candidate_proposal(parent, mutation_kind="structural", reward_program=large_reward),
        [parent],
        generation=1,
        island=3,
    )
    assert approximate_ast_distance(parent, too_large) > 0.65
    assert too_large.mutation_kind == "structural"
    assert too_large.inheritance_mode == "policy"
    assert large_report["mutation_scale"] == "large"
    validate_candidate_mutation([parent], too_large)


@pytest.mark.parametrize("distance", [0.05, 0.19, 0.20, 0.65, 0.90])
def test_ast_distance_boundaries_are_advisory(distance, monkeypatch):
    parent = initial_candidate(island=3, seed=1)
    reward = parent.reward_program.to_dict()
    reward["components"][0]["expression"] = {
        "op": "tanh",
        "value": reward["components"][0]["expression"],
    }
    proposal = _candidate_proposal(parent, mutation_kind="structural", reward_program=reward)
    monkeypatch.setattr("luxai2021.rl.evolution.approximate_ast_distance", lambda *_: distance)

    _, candidate, report = canonicalize_candidate_proposal(proposal, [parent], generation=1, island=3)

    assert report["ast_distance"] == distance
    assert candidate.mutation_kind == "structural"
    validate_candidate_mutation([parent], candidate)


def test_canonicalization_classifies_feature_and_pure_crossover():
    parent = initial_candidate(island=3, seed=3)
    donor = initial_candidate(island=0, seed=99)
    feature = _candidate_proposal(parent, mutation_kind="structural")
    feature["reward_program"]["components"].append(
        {"name": "added_units", "expression": {"op": "metric", "name": "units"}, "weight": 0.1}
    )
    _, feature_candidate, feature_report = canonicalize_candidate_proposal(
        feature, [parent], generation=1, island=3
    )
    assert feature_candidate.mutation_kind == "feature_existing"
    assert feature_report["mutation_scale"] == "feature"

    crossover = _candidate_proposal(
        parent,
        mutation_kind="crossover",
        secondary_parent_ids=(donor.candidate_id,),
    )
    crossover["reward_program"]["components"][0] = copy.deepcopy(donor.reward_program.to_dict()["components"][0])
    _, crossover_candidate, crossover_report = canonicalize_candidate_proposal(
        crossover, [parent, donor], generation=1, island=3
    )
    assert crossover_candidate.mutation_kind == "crossover"
    assert crossover_candidate.secondary_parent_ids == (donor.candidate_id,)
    assert crossover_report["mutation_scale"] == "recombined"


def test_island3_crossover_is_reclassified_from_actual_parent_contribution():
    parent = initial_candidate(island=3, seed=1)
    donor = initial_candidate(island=0, seed=99)
    crossover = next(
        child
        for seed in range(100)
        if (
            child := mutate_candidate(
                parent,
                generation=1,
                island=3,
                seed=seed,
                secondary_parents=(donor,),
            )
        ).mutation_kind
        == "crossover"
    )
    validate_candidate_mutation([parent, donor], crossover)
    assert crossover.secondary_parent_ids == (donor.candidate_id,)

    unchanged = _candidate_proposal(
        parent,
        mutation_kind="crossover",
        secondary_parent_ids=(donor.candidate_id,),
    )
    _, unchanged_child, unchanged_report = canonicalize_candidate_proposal(
        unchanged,
        [parent, donor],
        generation=1,
        island=3,
    )
    assert unchanged_child.mutation_kind == "structural"
    assert unchanged_report["mutation_scale"] == "large"

    changed_ppo = _candidate_proposal(
        parent,
        mutation_kind="crossover",
        reward_program=crossover.reward_program.to_dict(),
        secondary_parent_ids=(donor.candidate_id,),
    )
    changed_ppo["ppo_config"]["learning_rate"] *= 1.1
    changed_ppo["mutation_manifest"]["changed_paths"].append("ppo_config.learning_rate")
    _, changed_ppo_child, changed_report = canonicalize_candidate_proposal(
        changed_ppo,
        [parent, donor],
        generation=1,
        island=3,
    )
    assert changed_ppo_child.mutation_kind == "parameter"
    assert changed_report["mutation_scale"] == "numeric"


def test_island3_restart_accepts_arbitrary_safe_reward_and_compound_settings():
    parent = initial_candidate(island=3, seed=1)
    reward = default_reward_program().to_dict()
    reward["version"] = 2
    reward["derived_metrics"] = []
    reward["components"][0]["expression"] = {
        "op": "gate",
        "condition": {"op": "metric", "name": "night"},
        "when_true": {"op": "metric", "name": "min_city_survival"},
        "when_false": {"op": "metric", "name": "worker_resource_access"},
    }
    proposal = _candidate_proposal(parent, mutation_kind="restart", reward_program=reward)
    proposal["ppo_config"]["learning_rate"] *= 1.1
    proposal["mutation_manifest"]["changed_paths"].extend(
        ("ppo_config.learning_rate", "parameter_constraint_coefficient")
    )
    candidate = EvolutionCandidate.from_proposal(
        proposal,
        generation=1,
        island=3,
        parent_ids=(parent.candidate_id,),
    )
    validate_candidate_mutation([parent], candidate)
    assert candidate.primary_parent_id is None
    assert candidate.inheritance_mode == "base"
    assert candidate.parameter_constraint_coefficient == 0.0

    both = copy.deepcopy(proposal)
    both["opponent_mix"]["self_base"] -= 0.05
    both["opponent_mix"]["other_base"] += 0.05
    both["mutation_manifest"]["changed_paths"].append("opponent_mix")
    _, both_candidate, report = canonicalize_candidate_proposal(
        both,
        [parent],
        generation=1,
        island=3,
    )
    assert both_candidate.mutation_kind == "restart"
    assert both_candidate.inheritance_mode == "base"
    assert both_candidate.parameter_constraint_coefficient == 0.0
    assert report["mutation_scale"] == "restart"


def test_codex_proposal_schema_uses_supported_structured_output_constructs():
    schema = proposal_schema()

    def visit(value):
        if isinstance(value, dict):
            assert "oneOf" not in value
            if value.get("type") == "object":
                assert set(value.get("required", ())) == set(value.get("properties", ()))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    assert schema["properties"]["reward_program"]["properties"]["version"]["type"] == "integer"
    encoded = json.dumps(schema)
    assert "own_at_risk_city_tiles" in encoded
    assert "own_night_fuel_deficit" in encoded

    island3_prompt = build_codex_prompt([initial_candidate(island=3, seed=1)], [], island=3, generation=1)
    assert "coordinated edits across multiple components" in island3_prompt
    assert "Use only structural, crossover, or restart" in island3_prompt
    assert "prefer the single targeted change" not in island3_prompt


def test_codex_generator_records_all_failure_types(tmp_path, monkeypatch):
    generator = CodexCandidateGenerator(repository=tmp_path, run_dir=tmp_path, executable="codex", model="test")
    parent = initial_candidate(island=0, seed=1)

    monkeypatch.setattr(
        "luxai2021.rl.evolution.subprocess.run",
        lambda *_, **__: SimpleNamespace(returncode=1, stdout="stdout", stderr="authentication failed"),
    )

    with pytest.raises(RuntimeError, match="authentication failed"):
        generator.generate([parent], [], generation=1, island=0)
    error = json.loads((tmp_path / "codex-g01-i00.error.json").read_text())
    assert error["status"] == "failed"
    assert error["error_type"] == "RuntimeError"
    assert error["model"] == "test"
    assert error["parent_ids"] == [parent.candidate_id]


def test_codex_generator_records_accepted_proposal_hash(tmp_path, monkeypatch):
    generator = CodexCandidateGenerator(repository=tmp_path, run_dir=tmp_path, executable="codex", model="test")
    parent = initial_candidate(island=0, seed=1)
    proposal = _candidate_proposal(parent, mutation_kind="parameter")
    proposal["inheritance_mode"] = "policy_value"
    proposal["reward_program"]["components"][0]["weight"] += 0.1
    proposal["mutation_manifest"] = {
        "changed_paths": ["reward_program.components[0].weight"],
        "summary": "one numeric mutation",
    }

    def fake_run(command, **_):
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps(proposal))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("luxai2021.rl.evolution.subprocess.run", fake_run)

    candidate = generator.generate([parent], [], generation=1, island=0)
    metadata = json.loads(generator.metadata_path(1, 0).read_text())

    assert metadata["status"] == "accepted"
    assert metadata["candidate_id"] == candidate.candidate_id
    assert metadata["proposal_sha256"]
    assert metadata["raw_proposal_sha256"]
    assert metadata["canonical_proposal_sha256"] == metadata["proposal_sha256"]
    assert gzip.decompress((tmp_path / metadata["prompt_path"]).read_bytes()).decode() == build_codex_prompt(
        [parent], [], island=0, generation=1
    )
    assert not (tmp_path / "codex-g01-i00.error.json").exists()
    _save_candidate_provenance(
        tmp_path,
        candidate,
        {
            "candidate_id": candidate.candidate_id,
            "source": "codex",
            "proposal_path": metadata["proposal_path"],
            "proposal_metadata": generator.metadata_path(1, 0).name,
            "proposal_sha256": metadata["proposal_sha256"],
        },
    )
    audit = _validate_candidate_provenance(
        tmp_path,
        {candidate.candidate_id: candidate},
        {"candidate_generation": {"mode": "codex", "allow_fallback": False, "model": "test"}},
    )
    assert audit["valid"] is True
    assert audit["fully_codex_guided"] is True


def test_codex_generator_soft_accepts_and_reclassifies_semantic_deviation(tmp_path, monkeypatch):
    generator = CodexCandidateGenerator(
        repository=tmp_path,
        run_dir=tmp_path,
        executable="codex",
        model="test",
        validation_retries=2,
    )
    parent = initial_candidate(island=3, seed=1)
    invalid = _candidate_proposal(parent, mutation_kind="structural")
    invalid["reward_program"]["reward_scale"] *= 1.1
    invalid["mutation_manifest"] = {
        "changed_paths": [
            "reward_program.reward_scale",
            "parameter_constraint_coefficient",
        ],
        "summary": "too small structural mutation",
    }
    invalid["parameter_constraint_coefficient"] = 0.05
    prompts = []

    def fake_run(command, **kwargs):
        prompts.append(kwargs["input"])
        assert command[-1] == "-"
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps(invalid))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("luxai2021.rl.evolution.subprocess.run", fake_run)

    candidate = generator.generate([parent], [], generation=1, island=3)
    metadata = json.loads(generator.metadata_path(1, 3).read_text())

    assert candidate.mutation_kind == "parameter"
    assert candidate.inheritance_mode == "policy_value"
    assert len(prompts) == 1
    assert metadata["attempts"] == 1
    assert metadata["rejected_attempts"] == []
    assert metadata["normalization"]["declared_mutation_kind"] == "structural"
    assert metadata["normalization"]["effective_mutation_kind"] == "parameter"
    assert not generator.error_path(1, 3).exists()


def test_codex_generator_streams_large_prompt_through_stdin(tmp_path, monkeypatch):
    generator = CodexCandidateGenerator(repository=tmp_path, run_dir=tmp_path, executable="codex", model="test")
    parent = initial_candidate(island=0, seed=1)
    proposal = _candidate_proposal(parent, mutation_kind="parameter")
    proposal["inheritance_mode"] = "policy_value"
    proposal["reward_program"]["components"][0]["weight"] += 0.1
    large_prompt = "P" * (200 * 1024 + 17)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, prompt=kwargs["input"])
        Path(command[command.index("-o") + 1]).write_text(json.dumps(proposal))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("luxai2021.rl.evolution.build_codex_prompt", lambda *_, **__: large_prompt)
    monkeypatch.setattr("luxai2021.rl.evolution.subprocess.run", fake_run)

    generator.generate([parent], [], generation=1, island=0)
    metadata = json.loads(generator.metadata_path(1, 0).read_text())

    assert captured["command"][-1] == "-"
    assert large_prompt not in captured["command"]
    assert captured["prompt"] == large_prompt
    assert metadata["prompt_bytes"] == len(large_prompt.encode())
    assert gzip.decompress((tmp_path / metadata["prompt_path"]).read_bytes()) == large_prompt.encode()


def test_codex_generator_retries_only_hard_invalid_proposal(tmp_path, monkeypatch):
    generator = CodexCandidateGenerator(
        repository=tmp_path, run_dir=tmp_path, executable="codex", model="test", validation_retries=1
    )
    parent = initial_candidate(island=0, seed=1)
    unsafe = _candidate_proposal(parent, mutation_kind="parameter")
    unsafe["reward_program"]["components"][0]["expression"] = {"op": "metric", "name": "not_a_metric"}
    repaired = _candidate_proposal(parent, mutation_kind="parameter")
    repaired["reward_program"]["components"][0]["weight"] += 0.1
    repaired["inheritance_mode"] = "policy_value"
    proposals = [unsafe, repaired]
    prompts = []

    def fake_run(command, **kwargs):
        prompts.append(kwargs["input"])
        Path(command[command.index("-o") + 1]).write_text(json.dumps(proposals[len(prompts) - 1]))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("luxai2021.rl.evolution.subprocess.run", fake_run)

    candidate = generator.generate([parent], [], generation=1, island=0)

    assert candidate.mutation_kind == "parameter"
    assert len(prompts) == 2
    assert "VALIDATOR REPAIR REQUIRED" in prompts[1]


def test_turn_ranges_are_lossless_and_feedback_selection_is_bounded():
    turns = [30, 31, 32, 39, 40, 90]
    ranges = compress_turn_ranges(turns)
    expanded = [turn for start, end in ranges for turn in range(start, end + 1)]
    assert expanded == turns

    parents = [initial_candidate(island=island, seed=1) for island in range(3)]
    results = []
    for index, parent in enumerate(parents):
        results.extend(
            (
                CandidateResult(parent.candidate_id, "short-resattn8", "completed", 0.1 + index, 0.1, 0.0, 1.0, {}),
                CandidateResult(parent.candidate_id, "medium-resattn8", "failed", 0.0, 0.0, 0.0, 1.0, {}, "boom"),
            )
        )
    for island in range(3, 7):
        candidate = initial_candidate(island=island, seed=2)
        results.append(CandidateResult(candidate.candidate_id, "final-resattn8", "completed", island, 0.1, 0, 1, {}))

    selected = select_codex_feedback_results(parents, results)

    assert len(selected) == 8
    non_parent_ids = {
        result.candidate_id for result in selected if result.candidate_id not in {p.candidate_id for p in parents}
    }
    assert non_parent_ids == {
        results[-1].candidate_id,
        results[-2].candidate_id,
    }


def test_candidate_provenance_rejects_silent_codex_fallback(tmp_path):
    initial = initial_candidate(island=0, seed=1)
    child = mutate_candidate(initial, generation=1, island=0, seed=2)
    candidates = {candidate.candidate_id: candidate for candidate in (initial, child)}
    _save_candidate_provenance(
        tmp_path,
        initial,
        {"candidate_id": initial.candidate_id, "source": "initial"},
    )
    _save_candidate_provenance(
        tmp_path,
        child,
        {"candidate_id": child.candidate_id, "source": "codex_fallback"},
    )

    strict = _validate_candidate_provenance(
        tmp_path,
        candidates,
        {"candidate_generation": {"mode": "codex", "allow_fallback": False}},
    )
    explicit = _validate_candidate_provenance(
        tmp_path,
        candidates,
        {"candidate_generation": {"mode": "codex", "allow_fallback": True}},
    )

    assert strict["valid"] is False
    assert strict["fully_codex_guided"] is False
    assert explicit["valid"] is True
    assert explicit["fully_codex_guided"] is False
    assert explicit["counts"] == {"initial": 1, "codex_fallback": 1}


def test_checkpoint_descriptors_detect_content_replacement(tmp_path):
    base = tmp_path / "base.pt"
    teacher = tmp_path / "teacher.pt"
    base.write_bytes(b"base-a")
    teacher.write_bytes(b"teacher")
    args = SimpleNamespace(
        resattn8_only=True,
        resattn8_checkpoint=str(base),
        teacher_checkpoint=str(teacher),
    )
    descriptors = _checkpoint_descriptors(args)
    manifest = {"checkpoint_descriptors": descriptors}
    _validate_checkpoint_descriptors(args, manifest)

    base.write_bytes(b"base-b")

    with pytest.raises(ValueError, match=r"SHA-256|sha256"):
        _validate_checkpoint_descriptors(args, manifest)


def test_dry_run_cannot_be_resumed_as_training():
    manifest = {"run_kind": "dry-run"}

    _validate_run_kind(manifest, dry_run=True)
    with pytest.raises(ValueError, match="new --run-dir"):
        _validate_run_kind(manifest, dry_run=False)


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
    assert 0.0 <= metrics["illegal_action_mass_mean"] <= 1.0
    assert metrics["illegal_action_loss"] >= 0.0
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


def test_parameter_constraint_is_relative_and_cosine_decays_to_zero():
    actor = FullTurnActorCritic(_small_policy())
    trainer = PPOTrainer(
        actor,
        copy.deepcopy(actor.policy),
        PPOConfig(),
        torch.device("cpu"),
        parameter_constraint_coefficient=0.05,
        parameter_constraint_decay_decisions=100,
        constrain_value_head=True,
    )
    source = torch.zeros((), requires_grad=True)
    assert float(trainer._parameter_constraint_loss(source)) == pytest.approx(0.0)
    with torch.no_grad():
        next(actor.policy.parameters()).add_(0.01)
    assert float(trainer._parameter_constraint_loss(source)) > 0.0
    trainer.set_schedule_state(constraint_progress=100, joint_update=3)
    assert trainer.current_parameter_constraint_coefficient() == pytest.approx(0.0)


def test_critic_warmup_never_changes_policy_parameters():
    actor = FullTurnActorCritic(_small_policy())
    snapshot = copy.deepcopy(actor).eval()
    episode = collect_episode(
        actor,
        lambda: RolloutAgent(snapshot, device="cpu", deterministic=True),
        default_reward_program(),
        device=torch.device("cpu"),
        seed=19,
        opponent_name="small",
        max_turns=4,
    )
    before = {name: value.detach().clone() for name, value in actor.policy.named_parameters()}

    metrics = warmup_value_head(actor, [episode], PPOConfig(), torch.device("cpu"), max_epochs=2)

    assert metrics["epochs"] >= 2
    assert torch.isfinite(torch.tensor(metrics["final_validation_loss"]))
    assert metrics["final_validation_loss"] <= metrics["initial_validation_loss"]
    assert all(torch.equal(before[name], value) for name, value in actor.policy.named_parameters())


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
            "constraint_progress": 789,
            "joint_update": 2,
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
    assert restored.constraint_progress == 789
    assert restored.joint_update == 2
    trainer.set_schedule_state(
        constraint_progress=restored.constraint_progress,
        joint_update=restored.joint_update,
    )
    assert trainer.actor_lr_multiplier == 1.0

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


def test_training_checkpoint_can_resume_compatible_weights_from_an_older_base_path(tmp_path):
    actor = FullTurnActorCritic(_small_policy())
    trainer = PPOTrainer(actor, copy.deepcopy(actor.policy), PPOConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "latest_rl.pt"
    trainer.save_training_checkpoint(
        checkpoint_path,
        source_checkpoint="models/old-base.pt",
        source_checkpoint_sha256="old-sha256",
        reward_program=default_reward_program(),
        update=0,
        metrics={},
    )

    with pytest.raises(ValueError, match="source checkpoint"):
        trainer.load_training_state(
            checkpoint_path,
            source_checkpoint="models/new-base.pt",
            reward_program=default_reward_program(),
        )
    restored = trainer.load_training_state(
        checkpoint_path,
        source_checkpoint="models/new-base.pt",
        reward_program=default_reward_program(),
        allow_compatible_source_checkpoint=True,
    )

    assert restored.source_checkpoint == "models/old-base.pt"
    assert restored.source_checkpoint_mismatch is True
    assert restored.source_checkpoint_sha256 == "old-sha256"
    with pytest.raises(ValueError, match="SHA-256"):
        trainer.load_training_state(
            checkpoint_path,
            source_checkpoint="models/new-base.pt",
            source_checkpoint_sha256="new-sha256",
            reward_program=default_reward_program(),
            allow_compatible_source_checkpoint=True,
        )


def test_cuda_rng_resume_is_portable_across_visible_gpu_counts():
    first = torch.tensor([1, 2], dtype=torch.uint8)
    second = torch.tensor([3, 4], dtype=torch.uint8)

    assert torch.equal(_checkpoint_cuda_rng_state({"cuda_rng_state_all": [first, second]}, 0), first)
    assert torch.equal(_checkpoint_cuda_rng_state({"cuda_rng_state_all": [first, second]}, 7), first)
    assert torch.equal(
        _checkpoint_cuda_rng_state({"cuda_rng_state": second, "cuda_rng_state_all": [first]}, 0),
        second,
    )


def test_final_training_metrics_supports_an_inference_only_resume():
    resumed = {"kl": 0.125, "cumulative_decisions": 2_000_000}

    assert _final_training_metrics({"history": [], "final_metrics": resumed}) is resumed
    assert _final_training_metrics({"history": [resumed]}) is resumed
    with pytest.raises(RuntimeError, match="final PPO metrics"):
        _final_training_metrics({"history": []})


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
    assert reflected.fitness[0] == 0.0
    prompt = build_codex_prompt([child], [reflected], island=0, generation=2)
    assert '"city_tile_loss_turn_ranges":[[31,31]]' in prompt
    assert "31" in prompt
    assert '"score_rate_delta":0.199' in prompt


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


def test_filesystem_job_queue_retries_failed_completion(tmp_path):
    queue = FilesystemJobQueue(tmp_path)
    job = EvolutionJob("candidate", "medium-resattn8", "resattn8", 1, 1, 100)
    queue.enqueue(job)
    _, claimed_path = queue.claim("pc1")
    failed = CandidateResult("candidate", job.stage, "failed", 0.0, 0.0, float("inf"), 1.0, {}, "boom")
    queue.complete(claimed_path, failed)

    queue.enqueue(job)

    assert (queue.pending_dir / f"{job.job_id}.json").exists()
    assert not (queue.completed_dir / f"{job.job_id}.json").exists()


def test_execute_job_retries_an_existing_failed_result(tmp_path, monkeypatch):
    store = EvolutionStore(tmp_path)
    candidate = initial_candidate(island=0, seed=17)
    store.save_candidate(candidate)
    job = EvolutionJob(candidate.candidate_id, "medium-resattn8", "resattn8", 1, 1, 100)
    failed = CandidateResult(candidate.candidate_id, job.stage, "failed", 0.0, 0.0, 1.0, 1.0, {}, "old")
    store.save_result(failed)
    completed = CandidateResult(candidate.candidate_id, job.stage, "completed", 1.0, 1.0, 0.0, 1.0, {})
    monkeypatch.setattr("examples.evolve_rl._checkpoint_pair", lambda *_: (tmp_path / "base.pt", None))
    monkeypatch.setattr("examples.evolve_rl.candidate_result", lambda *_, **__: completed)
    monkeypatch.setattr("examples.evolve_rl.add_candidate_reflection", lambda result, *_: result)

    result = execute_evolution_job(
        job,
        args=SimpleNamespace(medium_decisions=100),
        device=torch.device("cpu"),
        store=store,
    )

    assert result.status == "completed"
    assert store.results()[0].status == "completed"


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
        assert health == {"status": "ok", "api_version": 2}
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
        assert medium_claim["input_artifacts"][0]["stage"] == "short-resattn8"
        downloaded = tmp_path / "downloaded-input"
        client.download_artifact(medium_claim["input_artifacts"][0], tmp_path / "cache", downloaded)
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
        assert medium_claim["input_artifacts"][0]["stage"] == "medium-resattn8"
        resumed_medium = tmp_path / "resumed-medium"
        client.download_artifact(medium_claim["input_artifacts"][0], tmp_path / "cache", resumed_medium)
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
        legacy_job = EvolutionJob(candidate.candidate_id, "legacy-check", "resattn8", 0, 1, 300)
        queue.enqueue(legacy_job)
        legacy_claim = client._post("/v1/claim", {"worker_id": "legacy-worker"})
        assert legacy_claim["api_version"] == 1
        assert legacy_claim["input_artifacts"] == []
        client.complete(
            lease_id=legacy_claim["lease_id"],
            job=legacy_job,
            result=CandidateResult(
                candidate.candidate_id,
                legacy_job.stage,
                "failed",
                0.0,
                0.0,
                float("inf"),
                0.0,
                {},
                "legacy endpoint check",
            ),
            artifact_dir=tmp_path / "missing-legacy-artifacts",
        )
    finally:
        server.close()

    coordinator_artifacts = coordinator_dir / "artifacts" / candidate.candidate_id / job.stage / job.base_name
    assert (coordinator_artifacts / "best.pt").read_bytes() == b"policy"
    assert (coordinator_artifacts / "latest_rl.pt").read_bytes() == b"training"
    assert queue.outstanding_ids() == set()
    stored = next(item for item in store.results() if item.stage == job.stage)
    assert stored.metrics["checkpoint"] == str(coordinator_artifacts / "best.pt")


def test_job_api_v2_streams_and_caches_parent_checkpoint(tmp_path):
    coordinator_dir = tmp_path / "coordinator"
    store = EvolutionStore(coordinator_dir)
    parent = initial_candidate(island=0, seed=21)
    child = mutate_candidate(parent, generation=1, island=0, seed=22)
    store.save_candidate(parent)
    store.save_candidate(child)
    store.save_manifest({"schema_version": 1, "arguments": {}})
    parent_dir = coordinator_dir / "artifacts" / parent.candidate_id / "short-resattn8" / "resattn8"
    parent_dir.mkdir(parents=True)
    (parent_dir / "latest_rl.pt").write_bytes(b"parent-training-checkpoint")
    queue = FilesystemJobQueue(coordinator_dir)
    job = EvolutionJob(child.candidate_id, "short-resattn8", "resattn8", 0, 1, 100)
    queue.enqueue(job)
    server = JobApiServer("127.0.0.1:0", run_dir=coordinator_dir, queue=queue)
    server.start()
    try:
        host, port = server.address
        client = JobApiClient(f"http://{host}:{port}")
        claim = client.claim("ws3")
        descriptor = claim["input_artifacts"][0]
        destination = tmp_path / "worker" / "parent"
        cache = tmp_path / "worker" / "cache"

        client.download_artifact(descriptor, cache, destination)
        client.download_artifact(descriptor, cache, destination)

        assert descriptor["kind"] == "parent"
        assert descriptor["candidate_id"] == parent.candidate_id
        assert (destination / "latest_rl.pt").read_bytes() == b"parent-training-checkpoint"
        assert len(list(cache.glob("*.zip"))) == 1
    finally:
        server.close()


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


def test_api_claim_preserves_completed_local_result_for_retry_upload(tmp_path):
    store = EvolutionStore(tmp_path)
    candidate = initial_candidate(island=0, seed=10)
    completed = CandidateResult(
        candidate.candidate_id,
        "short-resattn8",
        "completed",
        0.5,
        0.5,
        0.01,
        2.0,
        {"checkpoint": "local/best.pt"},
    )
    store.save_result(completed)
    job = EvolutionJob(candidate.candidate_id, completed.stage, "resattn8", 1200, 4, 100)
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

    _sync_api_claim(store, claim)

    assert store.results() == [completed]


def test_resattn8_only_removes_all_unet_runtime_inputs():
    args = SimpleNamespace(
        resattn8_only=True,
        unet_checkpoint="models/unet.pt",
        resattn8_checkpoint="models/resattn8.pt",
        teacher_checkpoint="models/teacher.pt",
    )

    assert _active_base_names(args) == ("resattn8",)
    assert [anchor.name for anchor in _evaluation_anchors(args)] == ["resattn8-base", "first-place"]
    assert _checkpoint_pair(args, "resattn8") == (
        Path("models/resattn8.pt"),
        Path("models/resattn8.pt"),
    )
    with pytest.raises(ValueError, match="disabled"):
        _checkpoint_pair(args, "unet")


def test_worker_inherits_resattn8_only_from_coordinator_manifest():
    worker_args = SimpleNamespace(resattn8_only=False, resattn8_checkpoint="new.pt")

    _apply_coordinator_manifest(
        worker_args,
        {"arguments": {"resattn8_only": True, "resattn8_checkpoint": "original.pt"}},
    )

    assert worker_args.resattn8_only is True
    assert worker_args.resattn8_checkpoint == "original.pt"


def test_stage_selection_is_frozen_and_uses_only_completed_source_stage(tmp_path):
    store = EvolutionStore(tmp_path)
    candidates = {
        candidate.candidate_id: candidate
        for candidate in (
            initial_candidate(island=0, seed=1),
            initial_candidate(island=1, seed=2),
            initial_candidate(island=2, seed=3),
        )
    }
    for candidate in candidates.values():
        store.save_candidate(candidate)
    ordered = list(candidates.values())
    results = [
        CandidateResult(ordered[0].candidate_id, "short-resattn8", "completed", 0.9, 0.0, 0.0, 1.0, {}),
        CandidateResult(ordered[1].candidate_id, "short-resattn8", "completed", 0.8, 0.0, 0.0, 1.0, {}),
        CandidateResult(ordered[2].candidate_id, "short-resattn8", "failed", 1.0, 1.0, 0.0, 1.0, {}),
        CandidateResult(ordered[2].candidate_id, "final-resattn8", "completed", 1.0, 1.0, 0.0, 1.0, {}),
    ]

    first = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="medium",
        target_stage="medium-resattn8",
        source_stage="short-resattn8",
        count=2,
    )
    changed_results = [CandidateResult(ordered[2].candidate_id, "short-resattn8", "completed", 1.0, 1.0, 0.0, 1.0, {})]
    resumed = _load_or_create_stage_selection(
        store,
        candidates,
        changed_results,
        name="medium",
        target_stage="medium-resattn8",
        source_stage="short-resattn8",
        count=2,
    )

    assert [candidate.candidate_id for candidate in first] == [
        ordered[0].candidate_id,
        ordered[1].candidate_id,
    ]
    assert [candidate.candidate_id for candidate in resumed] == [
        ordered[0].candidate_id,
        ordered[1].candidate_id,
    ]


def test_final_selection_ignores_legacy_jobs_without_completed_medium(tmp_path):
    store = EvolutionStore(tmp_path)
    candidates = {
        candidate.candidate_id: candidate
        for candidate in (
            initial_candidate(island=0, seed=1),
            initial_candidate(island=1, seed=2),
            initial_candidate(island=2, seed=3),
        )
    }
    ordered = list(candidates.values())
    for candidate in ordered:
        store.save_candidate(candidate)
    results = [
        CandidateResult(ordered[0].candidate_id, "medium-resattn8", "completed", 0.8, 0.0, 0.0, 1.0, {}),
        CandidateResult(ordered[1].candidate_id, "medium-resattn8", "completed", 0.7, 0.0, 0.0, 1.0, {}),
        CandidateResult(ordered[2].candidate_id, "short-resattn8", "completed", 1.0, 0.0, 0.0, 1.0, {}),
    ]
    queue = FilesystemJobQueue(tmp_path)
    legacy_final = EvolutionJob(ordered[2].candidate_id, "final-resattn8", "resattn8", 1, 1, 100)
    queue.enqueue(legacy_final)

    selected = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="final",
        target_stage="final-resattn8",
        source_stage="medium-resattn8",
        count=2,
    )

    assert [candidate.candidate_id for candidate in selected] == [
        ordered[0].candidate_id,
        ordered[1].candidate_id,
    ]
    saved = json.loads((tmp_path / "selections" / "final.json").read_text())
    assert saved["source_stage_verified"] is True


def test_existing_final_selection_without_completed_medium_is_rejected(tmp_path):
    store = EvolutionStore(tmp_path)
    candidate = initial_candidate(island=0, seed=1)
    store.save_candidate(candidate)
    selection = tmp_path / "selections" / "final.json"
    selection.parent.mkdir()
    EvolutionStore.write_json(selection, {"candidate_ids": [candidate.candidate_id]})

    with pytest.raises(ValueError, match="without completed medium-resattn8"):
        _load_or_create_stage_selection(
            store,
            {candidate.candidate_id: candidate},
            [],
            name="final",
            target_stage="final-resattn8",
            source_stage="medium-resattn8",
            count=1,
        )


def test_unselected_running_stage_job_is_archived(tmp_path):
    queue = FilesystemJobQueue(tmp_path)
    selected = EvolutionJob("selected", "medium-resattn8", "resattn8", 1, 1, 100)
    extra = EvolutionJob("extra", "medium-resattn8", "resattn8", 1, 1, 100)
    queue.enqueue(selected)
    queue.enqueue(extra)
    queue.claim("coordinator-host-123")
    queue.claim("coordinator-host-123")

    archived = _archive_unselected_stage_jobs(tmp_path, "medium-resattn8", {"selected"})

    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["candidate_id"] == "extra"
    assert queue.outstanding_ids() == {selected.job_id}


def test_fatal_cuda_context_errors_are_distinguished_from_oom():
    assert _is_fatal_cuda_error(RuntimeError("CUDA error: unspecified launch failure"))
    assert _is_fatal_cuda_error(RuntimeError("CUDA error: an illegal memory access was encountered"))
    assert not _is_fatal_cuda_error(RuntimeError("CUDA out of memory"))
