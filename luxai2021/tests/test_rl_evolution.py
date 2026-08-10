# ruff: noqa: ANN001, ANN003, ANN201, ANN202, PLR2004, PT011, S101, SLF001
from __future__ import annotations

import copy
import gzip
import json
import random
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch._inductor import config as inductor_config
from torch.distributions import Categorical

from examples.evolve_rl import (
    AnchorBatchProvider,
    PhaseBalancedBatchSampler,
    _active_base_names,
    _apply_coordinator_manifest,
    _archive_unselected_stage_jobs,
    _checkpoint_descriptors,
    _checkpoint_pair,
    _curriculum_start_decisions,
    _curriculum_start_games,
    _evaluation_anchors,
    _final_training_metrics,
    _fixed_candidate_paths,
    _is_fatal_cuda_error,
    _is_retryable_infrastructure_failure,
    _job_retry_count,
    _load_fixed_candidate,
    _load_inherited_modules,
    _load_or_create_stage_selection,
    _record_job_retry,
    _record_skipped_job,
    _save_candidate_provenance,
    _save_stage_inheritance_checkpoint,
    _select_completed_stage,
    _select_teacher_milestone,
    _select_update_evaluation,
    _stage_budget,
    _stage_checkpoint_sources,
    _sync_api_claim,
    _update_safety_failures,
    _validate_candidate_provenance,
    _validate_checkpoint_descriptors,
    _validate_fixed_candidate_descriptor,
    _validate_run_kind,
    execute_evolution_job,
)
from luxai2021.env.agent import Agent
from luxai2021.game.actions import MoveAction
from luxai2021.game.game import Game
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.match_controller import MatchController
from luxai2021.game.position import Position
from luxai2021.imitation.actions import FIRST_PLACE_ACTION_SCHEMA
from luxai2021.imitation.agent import BehaviorCloningAgent
from luxai2021.imitation.masking import monotonically_tighten_legal_mask
from luxai2021.imitation.model import (
    POLICY_SCHEMA_FIRST_PLACE_FLAT,
    LuxBehaviorCloningModel,
    ModelConfig,
    load_bc_checkpoint,
)
from luxai2021.imitation.schema import FEATURE_INDEX
from luxai2021.rl.batched_rollout import ActorCriticBatcher, InferenceBatcher, _CompiledInference
from luxai2021.rl.evaluation import (
    _survival_summary,
    acceptance_report,
    paired_seed_deltas,
    stage_advancement_report,
)
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
    lux_s1_rules_context,
    mutate_candidate,
    proposal_schema,
    select_codex_feedback_results,
    training_curriculum,
    validate_candidate_mutation,
)
from luxai2021.rl.job_api import JobApiClient, JobApiServer
from luxai2021.rl.metrics import GameMetrics, MetricContext, metrics_from_game
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent, _action_statistics
from luxai2021.rl.ppo import (
    ActorLRScheduleConfig,
    PPOConfig,
    PPOTrainer,
    _actionwise_clipped_surrogate,
    _checkpoint_cuda_rng_state,
    calculate_gae,
    collect_episode,
    collect_episodes_batched,
    resolve_rollout_backend,
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
        "safe_city_tiles": 0.0,
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
        "stranded_fuel": 0.0,
        "fuel_delivery_coverage": 0.0,
        "city_tile_loss": 0.0,
        "city_tile_loss_linear": 0.0,
        "night_fuel_shortage": 0.0,
        "worker_resource_access": 0.0,
        "worker_cargo_fullness": 0.0,
        "unit_capacity_utilization": 0.0,
        "coal_unlocked": 0.0,
        "uranium_unlocked": 0.0,
        "own_min_city_survival": 0.0,
        "own_city_tiles_at_risk": 0.0,
        "own_night_fuel_deficit": 0.0,
        "own_stranded_fuel": 0.0,
        "own_fuel_delivery_coverage": 0.0,
        "own_city_tiles_lost": 0.0,
        "own_city_tiles_lost_linear": 0.0,
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
    assert 1.0 < breakdown.total <= 3.0
    assert set(breakdown.components) == {component.name for component in program.components}


def test_default_reward_penalizes_absolute_fuel_deficit_and_city_loss():
    program = default_reward_program()
    safe = _metrics(own_night_fuel_deficit=0.0, own_city_tiles_lost=0.0)
    unsafe = _metrics(own_night_fuel_deficit=0.5, own_city_tiles_lost=0.25)

    breakdown = program.reward(safe, unsafe)
    weights = {component.name: component.weight for component in program.components}

    assert program.mode == "potential_linear"
    assert program.reward_scale == pytest.approx(0.35)
    assert program.gamma == pytest.approx(0.999)
    assert program.terminal_reward_scale == pytest.approx(10.0)
    assert program.normalize_total is True
    assert sum(abs(weight) for weight in weights.values()) == pytest.approx(5.2)
    assert weights["own_night_fuel_deficit"] == pytest.approx(-1.0)
    assert weights["own_city_tiles_lost"] == pytest.approx(-1.2)
    assert "night_fuel_deficit" not in weights
    assert "city_tile_loss" not in weights
    assert breakdown.shaping < 0.0
    assert set(breakdown.component_shaping) == set(weights)


def test_normalized_terminal_reward_cannot_be_reversed_by_potential_shaping():
    program = RewardProgram.from_dict(
        {
            "components": [{"name": "signal", "expression": {"op": "metric", "name": "city_tiles"}, "weight": 5}],
            "reward_scale": 0.5,
            "gamma": 0.995,
            "terminal_reward_scale": 10.0,
            "normalize_total": True,
        }
    )
    high = _metrics(city_tiles=1.0)
    low = _metrics(city_tiles=-1.0)

    win = program.reward(high, low, terminal_outcome=1.0)
    loss = program.reward(low, high, terminal_outcome=-1.0)

    assert win.total > 0.0
    assert loss.total < 0.0
    assert win.total == pytest.approx(1.0 + win.shaping)
    assert loss.total == pytest.approx(-1.0 + loss.shaping)
    assert max(abs(win.total), abs(loss.total)) < 1.5


def test_normalized_reward_rejects_terminal_scale_that_shaping_can_reverse():
    with pytest.raises(ValueError, match="must exceed"):
        RewardProgram.from_dict(
            {
                "components": [{"name": "signal", "expression": {"op": "metric", "name": "city_tiles"}, "weight": 5}],
                "reward_scale": 0.5,
                "gamma": 0.995,
                "terminal_reward_scale": 4.0,
                "normalize_total": True,
            }
        )


def test_survival_linear_fixed_candidate_matches_safe_defaults():
    path = Path("configs/rl_candidates/survival_linear_v1.json")
    candidate = _load_fixed_candidate(path)
    weights = {component.name: component.weight for component in candidate.reward_program.components}

    assert candidate.mutation_kind == "initial"
    assert candidate.reward_program.mode == "potential_linear"
    assert candidate.reward_program.reward_scale == pytest.approx(0.5)
    assert candidate.reward_program.terminal_reward_scale == pytest.approx(10.0)
    assert candidate.reward_program.normalize_total is True
    assert weights["own_night_fuel_deficit"] == pytest.approx(-1.0)
    assert weights["own_city_tiles_lost"] == pytest.approx(-1.2)
    assert candidate.ppo_config.bc_coefficient == pytest.approx(0.025)
    assert candidate.ppo_config.gae_lambda == pytest.approx(0.98)
    assert candidate.opponent_mix.teacher == pytest.approx(0.25)


def test_survival_v2_ab_candidates_share_credit_settings_and_differ_only_in_reward_components():
    credit = _load_fixed_candidate(Path("configs/rl_candidates/survival_credit_v2.json"), island=0)
    safe_city = _load_fixed_candidate(Path("configs/rl_candidates/survival_safe_city_v2.json"), island=1)
    safe_weights = {component.name: component.weight for component in safe_city.reward_program.components}

    assert credit.ppo_config == safe_city.ppo_config
    assert credit.opponent_mix == safe_city.opponent_mix
    assert credit.ppo_config.gamma == pytest.approx(0.999)
    assert credit.ppo_config.gae_lambda == pytest.approx(0.995)
    assert credit.ppo_config.bc_coefficient == pytest.approx(0.05)
    assert credit.ppo_config.kl_coefficient == 0.0
    assert credit.ppo_config.illegal_action_coefficient == pytest.approx(0.01)
    assert safe_city.reward_program.reward_scale == pytest.approx(0.35)
    assert safe_weights == {
        "safe_city_tiles": 1.2,
        "own_min_city_survival": 0.8,
        "units": 0.4,
        "research": 0.1,
        "own_night_fuel_deficit": -1.1,
        "own_stranded_fuel": -0.8,
        "own_city_tiles_lost_linear": -0.8,
    }


def test_fixed_candidate_descriptor_rejects_changed_file(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text("{}", encoding="utf-8")
    manifest = {
        "fixed_candidate_descriptor": {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": "changed",
        }
    }

    with pytest.raises(ValueError, match="Fixed candidate changed"):
        _validate_fixed_candidate_descriptor(path, manifest)


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


def _spawn_disconnected_city(game, team):
    own_positions = {
        (int(cell.pos.x), int(cell.pos.y))
        for city in game.cities.values()
        if city.team == team
        for cell in city.city_cells
    }
    for y in range(game.map.height):
        for x in range(game.map.width):
            cell = game.map.get_cell(x, y)
            if not cell.is_city_tile() and all(abs(x - px) + abs(y - py) > 1 for px, py in own_positions):
                return game.spawn_city_tile(team, x, y)
    raise AssertionError("No disconnected City location is available")


def test_stranded_fuel_tracks_only_simultaneous_surplus_and_deficit_across_cities():
    game = Game({"seed": 13})
    _spawn_disconnected_city(game, 0)
    own_cities = [city for city in game.cities.values() if city.team == 0]
    assert len(own_cities) == 2
    game.state["turn"] = 29
    required = [float(city.get_light_upkeep()) * 10.0 for city in own_cities]
    own_cities[0].fuel = required[0] + required[1]
    own_cities[1].fuel = 0.0
    for city in game.cities.values():
        if city.team == 1:
            city.fuel = float(city.get_light_upkeep()) * 10.0

    stranded = metrics_from_game(game, 0)
    expected = required[1] / sum(required)
    assert stranded.get("own_stranded_fuel") == pytest.approx(expected)
    assert stranded.get("stranded_fuel") == pytest.approx(-expected)
    assert 0.0 <= stranded.get("own_stranded_fuel") <= 1.0

    for city, requirement in zip(own_cities, required):
        city.fuel = requirement * 0.5
    assert metrics_from_game(game, 0).get("own_stranded_fuel") == 0.0

    for city, requirement in zip(own_cities, required):
        city.fuel = requirement
    assert metrics_from_game(game, 0).get("own_stranded_fuel") == 0.0


def test_stranded_fuel_is_zero_for_no_city_and_one_connected_city():
    game = Game({"seed": 19})
    own_city = next(city for city in game.cities.values() if city.team == 0)
    origin = own_city.city_cells[0].pos
    adjacent = next(
        cell for cell in game.map.get_adjacent_cells(game.map.get_cell(origin.x, origin.y)) if not cell.is_city_tile()
    )
    game.spawn_city_tile(0, adjacent.pos.x, adjacent.pos.y)
    assert len([city for city in game.cities.values() if city.team == 0]) == 1
    assert metrics_from_game(game, 0).get("own_stranded_fuel") == 0.0

    for city_id in [city.id for city in game.cities.values() if city.team == 0]:
        game.destroy_city(0, city_id)
    assert metrics_from_game(game, 0).get("own_stranded_fuel") == 0.0


def test_own_stranded_fuel_can_produce_negative_potential_shaping():
    program = RewardProgram.from_dict(
        {
            "version": 2,
            "derived_metrics": [],
            "components": [
                {
                    "name": "avoid_stranded_fuel",
                    "expression": {"op": "metric", "name": "own_stranded_fuel"},
                    "weight": -1.0,
                }
            ],
            "reward_scale": 0.2,
            "gamma": 1.0,
        }
    )
    before = _metrics(own_stranded_fuel=0.0)
    after = _metrics(own_stranded_fuel=0.5)

    assert program.reward(before, after).shaping < 0.0


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
    assert after.get("own_city_tiles_lost_linear") == pytest.approx(
        min(after.context.sums["own_city_tiles_lost"] / 64.0, 1.0)
    )
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


def test_safe_city_tiles_rewards_only_cities_funded_for_the_next_night():
    game = Game({"seed": 17})
    own_city = next(city for city in game.cities.values() if city.team == 0)
    opponent_city = next(city for city in game.cities.values() if city.team == 1)
    game.state["turn"] = 30
    own_city.fuel = float(own_city.get_light_upkeep()) * 10.0
    opponent_city.fuel = 0.0

    metrics = metrics_from_game(game, 0)

    assert metrics.get("safe_city_tiles") > 0.0
    assert metrics.get("own_night_fuel_deficit") == pytest.approx(0.0)


def test_reward_and_ppo_gamma_must_match():
    parent = initial_candidate(island=0, seed=1)
    proposal = _candidate_proposal(parent, mutation_kind="parameter")
    proposal["reward_program"]["gamma"] = 0.995

    with pytest.raises(ValueError, match="gamma must match"):
        EvolutionCandidate.from_proposal(proposal, generation=1, island=0, parent_ids=(parent.candidate_id,))


@pytest.mark.parametrize("horizon", [1, 40, 160, 360])
def test_discounted_terminal_outcome_dominates_extreme_potential_shaping(horizon):
    program = RewardProgram.from_dict(
        {
            "components": [{"name": "signal", "expression": {"op": "metric", "name": "city_tiles"}, "weight": 5.0}],
            "reward_scale": 0.35,
            "gamma": 0.999,
            "terminal_reward_scale": 10.0,
            "normalize_total": True,
        }
    )
    low, high = _metrics(city_tiles=-1.0), _metrics(city_tiles=1.0)

    def discounted_total(start, finish, outcome):
        total = 0.0
        previous = start
        for step in range(horizon):
            following = finish if step + 1 == horizon else start
            terminal = outcome if step + 1 == horizon else 0.0
            total += (program.gamma**step) * program.reward(previous, following, terminal_outcome=terminal).total
            previous = following
        return total

    assert discounted_total(high, low, 1.0) > 0.0
    assert discounted_total(low, high, -1.0) < 0.0


@pytest.mark.parametrize("horizon", [1, 360])
def test_v3_terminal_zero_potential_telescopes_to_initial_potential(horizon):
    program = default_reward_program()
    states = [_metrics(city_tiles=(-1.0 if step % 2 else 1.0)) for step in range(horizon + 1)]
    initial_potential = program.potential(states[0])[0]
    discounted_shaping = 0.0
    for step in range(horizon):
        breakdown = program.reward(
            states[step],
            states[step + 1],
            terminal_outcome=0.0,
            terminal=step + 1 == horizon,
        )
        discounted_shaping += (program.gamma**step) * breakdown.shaping
    assert discounted_shaping == pytest.approx(
        -program.reward_scale * initial_potential / program.terminal_reward_scale,
        abs=1e-6,
    )


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
    assert 0.01 <= calibrated.reward_scale <= 2.0


def test_candidate_round_trip_and_resume_store(tmp_path):
    initial = initial_candidate(island=1, seed=7)
    candidate = mutate_candidate(initial, generation=2, island=1, seed=8)
    store = EvolutionStore(tmp_path)
    store.save_candidate(candidate)

    loaded = store.candidates()

    assert loaded == [candidate]
    assert initial.ppo_config.bc_coefficient == pytest.approx(0.05)
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
    assert "own_stranded_fuel" in DIRECT_REWARD_METRIC_NAMES
    assert "own_stranded_fuel" in LOWER_IS_BETTER_METRIC_NAMES
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
    proposal["ppo_config"]["kl_coefficient"] = 0.5
    proposal["parameter_constraint_coefficient"] = 0.05
    proposal["mutation_manifest"]["changed_paths"].extend(
        ("ppo_config.learning_rate", "ppo_config.kl_coefficient", "parameter_constraint_coefficient")
    )
    ppo_child = EvolutionCandidate.from_proposal(
        proposal,
        generation=1,
        island=3,
        parent_ids=(parent.candidate_id,),
    )
    validate_candidate_mutation([parent], ppo_child)

    canonical, canonical_child, report = canonicalize_candidate_proposal(
        proposal,
        [parent],
        generation=1,
        island=3,
    )
    assert canonical_child.inheritance_mode == "policy"
    assert canonical_child.ppo_config.kl_coefficient == 0.5
    assert canonical_child.parameter_constraint_coefficient == 0.0
    assert {"parameter_constraint_coefficient"} <= set(report["corrected_fields"])
    assert canonical["mutation_manifest"]["changed_paths"]

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
    large_proposal = _candidate_proposal(parent, mutation_kind="structural", reward_program=large_reward)
    large_proposal["ppo_config"]["gamma"] = 0.91
    _, too_large, large_report = canonicalize_candidate_proposal(
        large_proposal,
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
    _, feature_candidate, feature_report = canonicalize_candidate_proposal(feature, [parent], generation=1, island=3)
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
    assert schema["properties"]["ppo_config"]["properties"]["kl_coefficient"]["const"] == 0.0
    assert schema["properties"]["ppo_config"]["properties"]["actionwise_clipping"]["const"] is True
    assert schema["properties"]["parameter_constraint_coefficient"]["const"] == 0.0
    encoded = json.dumps(schema)
    assert "own_at_risk_city_tiles" in encoded
    assert "own_night_fuel_deficit" in encoded
    assert "own_stranded_fuel" in encoded

    island3_prompt = build_codex_prompt([initial_candidate(island=3, seed=1)], [], island=3, generation=1)
    assert "coordinated edits across multiple components" in island3_prompt
    assert "Use only structural, crossover, or restart" in island3_prompt
    assert "prefer the single targeted change" not in island3_prompt
    assert "https://www.lux-ai.org/specs-2021#Background" in island3_prompt
    assert "360 turns" in island3_prompt
    assert "Teacher non-regression is a hard objective" in island3_prompt
    assert "simultaneous deficit" in island3_prompt


def test_rules_context_is_versioned_and_complete():
    context = lux_s1_rules_context()

    assert len(context["summary_sha256"]) == 64
    assert "30 day turns followed by 10 night turns" in context["summary"]
    assert "23 - 5 * adjacent_friendly_city_tiles" in context["summary"]
    assert "Wood, Coal, and Uranium are worth 1, 10, and 40" in context["summary"]
    parameters = GAME_CONSTANTS["PARAMETERS"]
    assert parameters["MAX_DAYS"] == 360
    assert (parameters["DAY_LENGTH"], parameters["NIGHT_LENGTH"]) == (30, 10)
    assert parameters["RESEARCH_REQUIREMENTS"] == {"COAL": 50, "URANIUM": 200}
    assert parameters["RESOURCE_TO_FUEL_RATE"] == {"WOOD": 1, "COAL": 10, "URANIUM": 40}


def test_teacher_guarded_curriculum_anneals_reward_and_increases_teacher():
    curriculum = training_curriculum("teacher_guarded_near_sparse")
    proposed = OpponentMix()

    assert curriculum.shaping_multiplier(0.0) == pytest.approx(1.0)
    assert curriculum.shaping_multiplier(0.6) == pytest.approx(0.5)
    assert curriculum.shaping_multiplier(1.0) == pytest.approx(0.05)
    mixes = [curriculum.opponent_mix(proposed, progress) for progress in (0.0, 0.3, 0.65, 1.0)]
    assert [mix.teacher for mix in mixes] == pytest.approx([0.25, 0.30, 0.40, 0.50])
    assert all(sum(vars(mix).values()) == pytest.approx(1.0) for mix in mixes)
    assert all(mix.snapshot >= 0.1 for mix in mixes)


def test_dense_shaping_curriculum_maintains_shaping_and_decays_bc():
    curriculum = training_curriculum("dense_shaping")
    proposed = OpponentMix()

    assert curriculum.shaping_multiplier(0.0) == pytest.approx(1.0)
    assert curriculum.shaping_multiplier(0.24) == pytest.approx(0.8)
    assert curriculum.shaping_multiplier(0.5) == pytest.approx(0.6304347826)
    assert curriculum.shaping_multiplier(1.0) == pytest.approx(0.25)

    assert curriculum.bc_coefficient_multiplier(0.0) == pytest.approx(1.0)
    assert curriculum.bc_coefficient_multiplier(0.24) == pytest.approx(0.9)
    assert curriculum.bc_coefficient_multiplier(0.5) == pytest.approx(0.8434782609)
    assert curriculum.bc_coefficient_multiplier(0.7) == pytest.approx(0.8)
    assert curriculum.bc_coefficient_multiplier(1.0) == pytest.approx(0.2)

    mixes = [curriculum.opponent_mix(proposed, progress) for progress in (0.0, 0.5, 1.0)]
    assert all(mix.teacher == pytest.approx(0.25) for mix in mixes)
    assert all(sum(vars(mix).values()) == pytest.approx(1.0) for mix in mixes)


def test_curriculum_stage_offsets_keep_final_phase_after_cross_stage_resume():
    args = SimpleNamespace(short_decisions=100, medium_decisions=300)

    assert _curriculum_start_decisions("short-resattn8", args) == 0
    assert _curriculum_start_decisions("probe-unet", args) == 0
    assert _curriculum_start_decisions("medium-resattn8", args) == 100
    assert _curriculum_start_decisions("final-resattn8", args) == 400
    assert _curriculum_start_decisions("final-unet", args) == 400


def test_game_budget_stage_offsets_and_budgets():
    args = SimpleNamespace(
        budget_unit="games",
        short_games=384,
        medium_games=1536,
        final_games=6144,
    )

    assert _curriculum_start_games("short-resattn8", args) == 0
    assert _curriculum_start_games("medium-resattn8", args) == 384
    assert _curriculum_start_games("final-resattn8", args) == 1920
    assert _stage_budget(args, "short-resattn8") == 384
    assert _stage_budget(args, "medium-resattn8") == 1536
    assert _stage_budget(args, "final-resattn8") == 6144


def test_stage_checkpoint_sources_separate_inference_inheritance_from_exact_resume(tmp_path):
    candidate = initial_candidate(island=0, seed=7)
    candidates = {candidate.candidate_id: candidate}
    short_best = tmp_path / "artifacts" / candidate.candidate_id / "short-resattn8" / "resattn8" / "best.pt"
    short_best.parent.mkdir(parents=True)
    short_best.write_bytes(b"inference-only")

    resume, inherit, parent = _stage_checkpoint_sources(
        tmp_path,
        candidate,
        candidates,
        stage="medium-resattn8",
        base_name="resattn8",
    )
    assert resume is None
    assert inherit == short_best
    assert parent == candidate

    short_best_rl = short_best.with_name("best_rl.pt")
    short_best_rl.write_bytes(b"policy-and-value")
    resume, inherit, parent = _stage_checkpoint_sources(
        tmp_path,
        candidate,
        candidates,
        stage="medium-resattn8",
        base_name="resattn8",
    )
    assert resume is None
    assert inherit == short_best_rl
    assert parent == candidate

    medium_latest = tmp_path / "artifacts" / candidate.candidate_id / "medium-resattn8" / "resattn8" / "latest_rl.pt"
    medium_latest.parent.mkdir(parents=True)
    medium_latest.write_bytes(b"optimizer-resume")
    resume, inherit, parent = _stage_checkpoint_sources(
        tmp_path,
        candidate,
        candidates,
        stage="medium-resattn8",
        base_name="resattn8",
    )
    assert resume == medium_latest
    assert inherit is None
    assert parent is None


def test_stage_best_rl_inherits_policy_and_value_without_optimizer_resume(tmp_path):
    source = FullTurnActorCritic(_small_policy())
    with torch.no_grad():
        for parameter in source.policy.parameters():
            parameter.fill_(0.125)
        for parameter in source.value_head.parameters():
            parameter.fill_(0.75)
    checkpoint_path = tmp_path / "best_rl.pt"
    _save_stage_inheritance_checkpoint(
        source,
        checkpoint_path,
        update=3,
        metrics={},
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_kind"] == "stage_inheritance"
    assert "optimizer" not in checkpoint
    assert "torch_rng_state" not in checkpoint
    target = FullTurnActorCritic(_small_policy())

    modules = _load_inherited_modules(
        target,
        checkpoint_path,
        device=torch.device("cpu"),
        inheritance_mode="base",
    )

    assert modules == ["policy", "value_head"]
    assert all(
        torch.equal(value, target.policy.state_dict()[name]) for name, value in source.policy.state_dict().items()
    )
    assert all(
        torch.equal(value, target.value_head.state_dict()[name])
        for name, value in source.value_head.state_dict().items()
    )


def test_repeated_fixed_candidate_paths_are_distinct_and_ordered():
    paths = _fixed_candidate_paths(["configs/rl_candidates/survival_credit_v2.json", "safe.json"])
    assert paths == [Path("configs/rl_candidates/survival_credit_v2.json"), Path("safe.json")]
    with pytest.raises(ValueError, match="distinct"):
        _fixed_candidate_paths(["same.json", "same.json"])


def test_teacher_milestone_selection_prefers_teacher_score_then_latest(tmp_path, monkeypatch):
    milestone_dir = tmp_path / "milestones"
    milestone_dir.mkdir()
    for name in ("p060.pt", "p080.pt", "p100.pt"):
        (milestone_dir / name).touch()
    scores = {"p060": 0.4, "p080": 0.5, "p100": 0.5}

    def fake_evaluate(candidate, anchors, **kwargs):
        assert anchors[0].model_type == "first-place"
        assert kwargs["seed_count"] == 12
        return {"totals": {"score_rate": scores[candidate.checkpoint.stem]}}

    monkeypatch.setattr("examples.evolve_rl.evaluate_against_league", fake_evaluate)
    selected, report = _select_teacher_milestone(
        tmp_path,
        candidate_id="candidate",
        teacher_checkpoint=tmp_path / "teacher.pt",
        seed_start=123,
        seed_count=12,
        device="cpu",
        max_turns=360,
    )

    assert selected == milestone_dir / "p100.pt"
    assert report["selection_policy"] == "highest_teacher_score_rate_then_latest"
    assert json.loads((tmp_path / "milestone_selection.json").read_text())["selected_checkpoint"] == str(selected)


def test_legacy_manifest_requires_a_new_run_for_metric_schema_change():
    args = SimpleNamespace(
        curriculum_profile="teacher_guarded_near_sparse",
        bc_anchor_max_turns=0,
        bc_anchor_sampling="phase-balanced",
    )

    with pytest.raises(ValueError, match="Reward metric schema changed"):
        _apply_coordinator_manifest(args, {"schema_version": 3, "arguments": {}})


def test_phase_balanced_sampler_covers_all_turn_strata():
    class Dataset:
        def __init__(self) -> None:
            self.samples = [(Path("r.json"), turn, 0) for turn in (0, 30, 90, 110, 180, 190, 280, 310)]

        def __len__(self) -> int:
            return len(self.samples)

    dataset = Dataset()
    sampler = PhaseBalancedBatchSampler(dataset, 8, seed=7)
    batch = next(iter(sampler))
    strata = {
        min(dataset.samples[index][1] // 90, 3) * 2 + int(dataset.samples[index][1] % 40 >= 30) for index in batch
    }

    assert strata == set(range(8))


def test_anchor_batch_provider_restores_exact_sampler_position():
    class EpochSampler:
        def __init__(self) -> None:
            self.epoch = 0

        def __iter__(self) -> object:
            order = list(range(6))
            if self.epoch % 2:
                order.reverse()
            self.epoch += 1
            yield from ([index] for index in order)

        def __len__(self) -> int:
            return 6

    dataset = [{"value": torch.tensor(index)} for index in range(6)]
    first_sampler = EpochSampler()
    first = AnchorBatchProvider(
        torch.utils.data.DataLoader(dataset, batch_sampler=first_sampler),
        first_sampler,
        sampling="test",
    )
    for _ in range(8):
        first()
    state = first.state_dict()
    expected = first()["value"]

    resumed_sampler = EpochSampler()
    resumed = AnchorBatchProvider(
        torch.utils.data.DataLoader(dataset, batch_sampler=resumed_sampler),
        resumed_sampler,
        sampling="test",
    )
    resumed.load_state_dict(state)

    assert torch.equal(resumed()["value"], expected)


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
    assert metadata["rules_source_url"] == "https://www.lux-ai.org/specs-2021#Background"
    assert metadata["rules_summary_sha256"] == lux_s1_rules_context()["summary_sha256"]
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
    assert candidate.parameter_constraint_coefficient == 0.0
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
    assert "kl" in metrics
    assert "reference_kl" in metrics
    assert "approx_kl" in metrics
    assert abs(metrics["approx_kl"]) <= 1e-6
    assert metrics["joint_clip_fraction"] == pytest.approx(0.0)
    assert metrics["action_clip_fraction"] == pytest.approx(0.0)
    assert "parameter_constraint_loss" not in metrics
    assert "parameter_constraint_coefficient" not in metrics
    checkpoint_path = tmp_path / "latest_rl.pt"
    trainer.save_training_checkpoint(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        update=0,
        metrics=metrics,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "reference_policy" not in checkpoint
    assert "parameter_reference" not in checkpoint
    assert "parameter_constraint_coefficient" not in checkpoint
    checkpoint["reference_policy"] = copy.deepcopy(actor.policy).state_dict()
    checkpoint["parameter_reference"] = {"legacy": torch.ones(1)}
    checkpoint["parameter_constraint_coefficient"] = 0.05
    torch.save(checkpoint, checkpoint_path)
    resumed_actor = FullTurnActorCritic(_small_policy())
    resumed = PPOTrainer(
        resumed_actor,
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


def test_ppo_logs_approximate_kl_before_early_stopping():
    actor = FullTurnActorCritic(_small_policy())
    snapshot = copy.deepcopy(actor).eval()
    episode = collect_episode(
        actor,
        lambda: RolloutAgent(snapshot, device="cpu", deterministic=True),
        default_reward_program(),
        device=torch.device("cpu"),
        seed=31,
        opponent_name="small",
        max_turns=8,
    )
    for record in episode.records:
        for decision in record.decisions:
            decision.old_log_prob += 0.5
        record.old_joint_log_prob += 0.5 * len(record.decisions)
    trainer = PPOTrainer(
        actor,
        PPOConfig(update_epochs=1, minibatch_turns=1, bc_coefficient=0.0, target_kl=1e-6),
        torch.device("cpu"),
    )

    metrics = trainer.update([episode])

    assert metrics["approx_kl"] > 0.0
    assert metrics["stable_approx_kl"] > 0.0
    assert metrics["early_stopped"] == 1.0
    assert metrics["early_stop_reason"] == "stable_per_action_kl"
    assert metrics["epochs_completed"] == 1.0
    assert metrics["minibatches_completed"] < metrics["minibatches_planned"]
    assert metrics["early_stop_kl"] > metrics["early_stop_threshold"]


def test_actionwise_clipping_avoids_joint_ratio_amplification():
    new_log_probs = torch.full((32,), float(np.log(1.01)), requires_grad=True)
    old_log_probs = torch.zeros(32)
    turn_indices = torch.zeros(32, dtype=torch.long)
    advantages = torch.ones(1)

    loss, log_ratios, ratios = _actionwise_clipped_surrogate(
        new_log_probs,
        old_log_probs,
        turn_indices,
        advantages,
        clip_range=0.2,
    )

    assert torch.exp(log_ratios.sum()) > 1.2
    assert not bool(((ratios - 1.0).abs() > 0.2).any())
    assert loss.item() == pytest.approx(-1.01)
    loss.backward()
    assert torch.isfinite(new_log_probs.grad).all()


def test_actionwise_surrogate_averages_over_all_valid_minibatch_factors():
    loss, _, _ = _actionwise_clipped_surrogate(
        torch.zeros(4),
        torch.zeros(4),
        torch.tensor([0, 1, 1, 1]),
        torch.tensor([1.0, 3.0]),
        clip_range=0.2,
    )

    assert loss.item() == pytest.approx(-(1.0 + 3.0 + 3.0 + 3.0) / 4.0)


def test_decision_factors_reconstruct_joint_log_probability_with_priority():
    actor = FullTurnActorCritic(_small_policy())
    snapshot = copy.deepcopy(actor).eval()
    episode = collect_episode(
        actor,
        lambda: RolloutAgent(snapshot, device="cpu", deterministic=False),
        default_reward_program(),
        device=torch.device("cpu"),
        seed=43,
        opponent_name="small",
        max_turns=4,
    )
    records = [record for record in episode.records if record.decisions]
    observations = torch.stack([record.observation for record in records])
    output, values = actor.forward_tta(observations)
    trainer = PPOTrainer(actor, PPOConfig(), torch.device("cpu"))

    statistics = trainer._vectorized_turn_statistics(output, values, records, None)
    reconstructed = values.new_zeros(len(records)).scatter_add(
        0,
        statistics["decision_turn_indices"],
        statistics["decision_log_probs"],
    )
    old_reconstructed = values.new_zeros(len(records)).scatter_add(
        0,
        statistics["decision_turn_indices"],
        statistics["decision_old_log_probs"],
    )

    assert torch.allclose(reconstructed, statistics["joint_log_prob"], atol=1e-6)
    assert torch.allclose(
        old_reconstructed,
        torch.tensor([record.old_joint_log_prob for record in records]),
        atol=1e-6,
    )


def test_online_teacher_kl_calibrates_once_and_stays_in_bounds():
    actor = FullTurnActorCritic(_small_policy())
    snapshot = copy.deepcopy(actor).eval()
    episode = collect_episode(
        actor,
        lambda: RolloutAgent(snapshot, device="cpu", deterministic=True),
        default_reward_program(),
        device=torch.device("cpu"),
        seed=37,
        opponent_name="small",
        max_turns=4,
    )
    for record in episode.records:
        for decision in record.decisions:
            decision.teacher_logits = torch.zeros(len(decision.legal_mask))
    config = PPOConfig(update_epochs=1, minibatch_turns=4, bc_coefficient=0.0)
    trainer = PPOTrainer(actor, config, torch.device("cpu"))

    metrics = trainer.update([episode], record_grad_norms=True)
    calibrated = trainer.effective_teacher_kl_coefficient

    assert calibrated is not None
    assert config.teacher_kl_coefficient_min <= calibrated <= config.teacher_kl_coefficient_max
    assert torch.isfinite(torch.tensor(metrics["online_teacher_kl"]))
    assert metrics["grad_norm_samples"] == 1.0


def test_parent_kl_and_parameter_constraint_are_absent_from_trainer():
    actor = FullTurnActorCritic(_small_policy())
    trainer = PPOTrainer(
        actor,
        PPOConfig(kl_coefficient=0.8),
        torch.device("cpu"),
    )
    assert hasattr(trainer, "reference_policy")
    assert not hasattr(trainer, "parameter_reference")
    assert not hasattr(trainer, "parameter_constraint_coefficient")
    trainer.set_schedule_state(joint_update=3)


def test_actor_lr_cosine_schedule_uses_global_progress_and_stable_kl_feedback():
    schedule = ActorLRScheduleConfig(mode="cosine", floor_ratio=0.1, warmup_updates=2)
    trainer = PPOTrainer(
        FullTurnActorCritic(_small_policy()),
        PPOConfig(),
        torch.device("cpu"),
        actor_lr_schedule=schedule,
    )

    trainer.set_schedule_state(joint_update=0, training_progress=0.0)
    assert trainer.actor_lr_multiplier == pytest.approx(0.25)
    groups = {group["group_name"]: group for group in trainer.optimizer.param_groups}
    assert groups["policy"]["lr"] == pytest.approx(0.25 * trainer.config.learning_rate)
    assert groups["value"]["lr"] == pytest.approx(trainer.config.learning_rate)

    short_progress = 384 / (384 + 1536 + 6144)
    trainer.set_schedule_state(joint_update=2, training_progress=short_progress)
    assert trainer.actor_lr_multiplier > 0.99

    trainer.set_schedule_state(joint_update=2, training_progress=0.5)
    assert trainer.actor_lr_multiplier == pytest.approx(0.55)

    trainer.set_schedule_state(
        joint_update=3,
        training_progress=0.5,
        previous_stable_kl=0.02,
        previous_joint_clip_fraction=0.10,
    )
    assert trainer.actor_lr_feedback_multiplier == pytest.approx(0.5)
    assert trainer.actor_lr_multiplier == pytest.approx(0.275)

    reference_diagnostic = PPOTrainer(
        FullTurnActorCritic(_small_policy()),
        PPOConfig(),
        torch.device("cpu"),
        actor_lr_schedule=schedule,
    )
    reference_diagnostic.set_schedule_state(
        joint_update=2,
        training_progress=0.0,
        previous_stable_kl=0.0002,
        previous_reference_kl=0.007,
        previous_joint_clip_fraction=0.01,
    )
    assert reference_diagnostic.actor_lr_feedback_multiplier == pytest.approx(1.0)
    assert reference_diagnostic.actor_lr_multiplier == pytest.approx(1.0)
    assert reference_diagnostic.actor_lr_feedback_reason == "hold"


def test_update_selection_uses_win_rates_not_base_drift_diagnostics():
    evaluations = [
        {
            "update": 3,
            "teacher_score_rate": 0.125,
            "teacher_game_count": 16,
            "overall_score_rate": 0.34375,
            "base_action_agreement": 0.984,
        },
        {
            "update": 7,
            "teacher_score_rate": 0.1875,
            "teacher_game_count": 16,
            "overall_score_rate": 0.28125,
            "base_action_agreement": 0.981,
        },
        {
            "update": 9,
            "teacher_score_rate": 0.25,
            "teacher_game_count": 16,
            "overall_score_rate": 0.5,
            "base_action_agreement": 0.97,
        },
    ]

    selected = _select_update_evaluation(evaluations)

    assert selected is not None
    assert selected["update"] == 9


def test_online_update_safety_rejects_score_regression_without_joint_drift():
    accepted = {
        "teacher_score_rate": 0.25,
        "overall_score_rate": 0.4375,
    }
    current = {
        "teacher_score_rate": 0.0,
        "teacher_game_count": 16,
        "overall_score_rate": 0.21875,
        "overall_game_count": 32,
        "base_action_agreement": 0.9776,
        "reference_kl": 0.0055,
    }

    failures = _update_safety_failures(
        current,
        accepted,
        joint_kl_high=0.01,
        joint_log_ratio_p95_high=0.20,
        teacher_regression_wins=1,
        overall_regression_wins=2,
    )

    assert failures == ["teacher_regression", "overall_regression"]


def test_online_update_safety_ignores_base_agreement_and_reference_kl_drift():
    accepted = {"teacher_score_rate": 0.25, "overall_score_rate": 0.4375}
    current = {
        "teacher_score_rate": 0.25,
        "teacher_game_count": 16,
        "overall_score_rate": 0.4375,
        "overall_game_count": 32,
        "base_action_agreement": 0.5,
        "reference_kl": 1.0,
    }

    assert not _update_safety_failures(
        current,
        accepted,
        joint_kl_high=0.01,
        joint_log_ratio_p95_high=0.20,
        teacher_regression_wins=1,
        overall_regression_wins=2,
    )


def test_online_update_safety_allows_bounded_matched_noise():
    accepted = {"teacher_score_rate": 0.25, "overall_score_rate": 0.4375}
    current = {
        "teacher_score_rate": 0.1875,
        "teacher_game_count": 16,
        "overall_score_rate": 0.375,
        "overall_game_count": 32,
        "base_action_agreement": 0.996,
        "reference_kl": 0.0015,
    }

    assert not _update_safety_failures(
        current,
        accepted,
        joint_kl_high=0.01,
        joint_log_ratio_p95_high=0.20,
        teacher_regression_wins=1,
        overall_regression_wins=2,
    )


def test_joint_policy_drift_decays_lr_and_rejects_update():
    schedule = ActorLRScheduleConfig(
        floor_ratio=0.1,
        joint_kl_high=0.01,
        joint_log_ratio_p95_high=0.20,
    )
    trainer = PPOTrainer(
        FullTurnActorCritic(_small_policy()),
        PPOConfig(learning_rate=1e-6),
        torch.device("cpu"),
        actor_lr_schedule=schedule,
    )
    trainer.set_schedule_state(
        joint_update=3,
        previous_joint_kl=0.013,
        previous_joint_log_ratio_p95=0.32,
    )

    assert trainer.actor_lr_feedback_multiplier == pytest.approx(0.5)
    assert trainer.actor_lr_feedback_reason == "decay:joint_kl+joint_log_ratio_p95"
    policy_group = next(group for group in trainer.optimizer.param_groups if group["group_name"] == "policy")
    assert policy_group["lr"] <= 0.5e-6

    failures = _update_safety_failures(
        {
            "joint_kl": 0.013,
            "joint_log_ratio_p95": 0.32,
            "teacher_score_rate": 0.25,
            "teacher_game_count": 16,
            "overall_score_rate": 0.5,
            "overall_game_count": 32,
        },
        None,
        joint_kl_high=0.01,
        joint_log_ratio_p95_high=0.20,
        teacher_regression_wins=1,
        overall_regression_wins=2,
    )
    assert failures == ["joint_kl", "joint_log_ratio_p95"]


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
    trainer = PPOTrainer(actor, PPOConfig(), torch.device("cpu"))
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
            "history": [{"update": 2}, {"update": 3}],
            "bc_batch_provider_state": {"schema_version": 1, "total_batches": 17},
        },
        training_contract_hash="contract-a",
    )

    restored = trainer.load_training_state(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        training_contract_hash="contract-a",
    )

    assert restored.next_update == 4
    assert restored.cumulative_decisions == 1234
    assert restored.cumulative_turns == 456
    assert restored.cumulative_episodes == 7
    assert restored.curriculum_progress_decisions == 789
    assert restored.joint_update == 2
    assert restored.history == [{"update": 2}, {"update": 3}]
    assert restored.bc_batch_provider_state == {"schema_version": 1, "total_batches": 17}
    trainer.set_schedule_state(
        joint_update=restored.joint_update,
    )
    assert trainer.actor_lr_multiplier == 1.0
    with pytest.raises(ValueError, match="training contract"):
        trainer.load_training_state(
            checkpoint_path,
            source_checkpoint="base.pt",
            reward_program=default_reward_program(),
            training_contract_hash="contract-b",
        )

    legacy = torch.load(checkpoint_path, weights_only=False)
    legacy["schema_version"] = 1
    legacy.pop("training_state")
    legacy.pop("torch_rng_state")
    legacy.pop("cuda_rng_state_all")
    legacy["ppo_config"].pop("illegal_action_coefficient")
    legacy["metrics"] = {"elapsed_seconds": 50.0}
    legacy_path = tmp_path / "legacy_rl.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match="cannot resume optimizer state"):
        trainer.load_training_state(
            legacy_path,
            source_checkpoint="base.pt",
            reward_program=default_reward_program(),
            legacy_target_decisions=1000,
            legacy_stage_seconds=100,
        )


def test_training_checkpoint_can_resume_compatible_weights_from_an_older_base_path(tmp_path):
    actor = FullTurnActorCritic(_small_policy())
    trainer = PPOTrainer(actor, PPOConfig(), torch.device("cpu"))
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


def test_training_checkpoint_rejects_mixed_budget_units_and_restores_teacher_coefficient(tmp_path):
    actor = FullTurnActorCritic(_small_policy())
    trainer = PPOTrainer(actor, PPOConfig(), torch.device("cpu"))
    trainer.effective_teacher_kl_coefficient = 0.0125
    checkpoint_path = tmp_path / "latest_rl.pt"
    trainer.save_training_checkpoint(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        update=0,
        metrics={},
        training_state={"budget_unit": "games", "curriculum_progress_games": 64},
    )

    resumed = PPOTrainer(FullTurnActorCritic(_small_policy()), PPOConfig(), torch.device("cpu"))
    state = resumed.load_training_state(
        checkpoint_path,
        source_checkpoint="base.pt",
        reward_program=default_reward_program(),
        budget_unit="games",
    )

    assert state.curriculum_progress_games == 64
    assert resumed.effective_teacher_kl_coefficient == pytest.approx(0.0125)
    with pytest.raises(ValueError, match="Cannot mix game and decision budgets"):
        resumed.load_training_state(
            checkpoint_path,
            source_checkpoint="base.pt",
            reward_program=default_reward_program(),
            budget_unit="decisions",
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
            rollout_backend="lockstep",
        )
    finally:
        rollout_metrics = actor_batcher.metrics()
        actor_batcher.close()
    assert len(episodes) == 2
    assert all(episode.records for episode in episodes)
    assert rollout_metrics["samples"] == sum(len(episode.records) for episode in episodes)
    assert rollout_metrics["mean_batch_fill_ratio"] == 1.0
    assert rollout_metrics["mean_batch_size"] == 2.0


def test_inference_batcher_async_submissions_batch_from_one_thread():
    batcher = InferenceBatcher(
        lambda values: [value * 3 for value in values],
        max_batch_size=2,
        wait_seconds=0.05,
        name="test-async-inference",
    )
    try:
        first = batcher.submit_async(2)
        second = batcher.submit_async(4)
        assert first.result() == 6
        assert second.result() == 12
        assert batcher.metrics()["max_batch_size"] == 2
    finally:
        batcher.close()


def test_rollout_overlaps_teacher_and_candidate_without_changing_stored_logits():
    class ImmediateFuture:
        def __init__(self, value) -> None:
            self.value = value

        def result(self):
            return self.value

    class AsyncTeacher:
        def __init__(self) -> None:
            self.async_calls = 0
            self.sync_calls = 0

        def submit_team(self, _snapshot, _team):
            self.sync_calls += 1
            raise AssertionError("synchronous teacher path should not run")

        def submit_team_async(self, _snapshot, _team):
            self.async_calls += 1
            return ImmediateFuture(
                {entity: torch.zeros(1, len(actions), 32, 32) for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items()}
            )

    actor = FullTurnActorCritic(_small_policy()).eval()
    snapshot = copy.deepcopy(actor).eval()
    candidate = ActorCriticBatcher(actor, torch.device("cpu"), 1, name="test-async-candidate")
    teacher = AsyncTeacher()
    try:
        episode = collect_episode(
            actor,
            lambda: RolloutAgent(snapshot, device="cpu", deterministic=True),
            default_reward_program(),
            device=torch.device("cpu"),
            seed=71,
            opponent_name="snapshot",
            max_turns=4,
            inference_backend=candidate.submit,
            teacher_inference_backend=teacher.submit_team,
        )
    finally:
        candidate.close()
    assert teacher.async_calls == len(episode.records)
    assert teacher.sync_calls == 0
    assert all(decision.teacher_logits is not None for record in episode.records for decision in record.decisions)


def test_batched_joint_margins_match_scalar_reference():
    logits = torch.tensor([[0.2, -0.3, 1.1], [2.0, -1.0, 0.5]])
    masks = torch.tensor([[True, False, True], [True, False, False]])

    actual = BehaviorCloningAgent._joint_margins(logits, masks)
    expected = torch.tensor([BehaviorCloningAgent._joint_margin(row, mask) for row, mask in zip(logits, masks)])

    assert torch.equal(actual, expected)


def test_auto_compile_skips_variable_batches_without_calling_torch_compile(monkeypatch):
    compile_calls = []
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: compile_calls.append(kwargs) or module)
    inference = _CompiledInference(
        torch.nn.Identity(),
        torch.device("cuda"),
        "auto",
        auto_eligible=False,
    )

    output = inference(torch.ones(1))

    assert torch.equal(output, torch.ones(1))
    assert compile_calls == []
    assert inference.compile_attempts == 0
    assert inference.fallback_reason == "auto_compile_requires_static_batches"


def test_compile_runs_once_per_instance_and_repairs_fork_worker_start(monkeypatch):
    compile_calls = []
    monkeypatch.setattr(inductor_config, "worker_start_method", "fork")
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: compile_calls.append(kwargs) or module)
    inference = _CompiledInference(torch.nn.Identity(), torch.device("cuda"), "on")

    inference(torch.ones(1))
    inference(torch.ones(1))

    assert compile_calls == [{"mode": "reduce-overhead"}]
    assert inference.compile_attempts == 1
    assert inference.compile_seconds >= 0.0
    assert inductor_config.worker_start_method == "subprocess"


def test_rollout_action_statistics_match_categorical_reference():
    logits = torch.tensor([0.25, -0.5, 1.0, 0.75])
    legal = torch.tensor([True, False, True, True])
    reference = Categorical(logits=logits.masked_fill(~legal, torch.finfo(logits.dtype).min))

    log_probabilities, probabilities, entropy = _action_statistics(logits, legal)

    assert torch.equal(log_probabilities, reference.logits)
    assert torch.equal(probabilities, reference.probs)
    assert torch.equal(entropy, reference.entropy())


def test_sampled_plackett_luce_priority_log_probability_matches_manual_sum():
    agent = RolloutAgent(FullTurnActorCritic(_small_policy()), device="cpu")
    agent.generator.manual_seed(123)
    entries = [
        {"identity": "a", "margin": 0.2},
        {"identity": "b", "margin": 0.8},
        {"identity": "c", "margin": -0.1},
    ]

    ordered = agent._joint_priority_order(entries)
    remaining = list(entries)
    manual = 0.0
    for selected, _ in ordered:
        margins = torch.tensor([float(entry["margin"]) for entry in remaining])
        selected_index = next(
            index for index, entry in enumerate(remaining) if entry["identity"] == selected["identity"]
        )
        manual += float(torch.log_softmax(margins, dim=0)[selected_index])
        remaining.pop(selected_index)

    assert sum(log_probability for _, log_probability in ordered) == pytest.approx(manual)


def test_lockstep_inference_drops_finished_participants_without_partial_batches():
    barrier = threading.Barrier(3)
    batcher = InferenceBatcher(lambda values: values, max_batch_size=3, wait_seconds=0.001, name="test-lockstep")
    outputs = [[], [], []]

    def submit(index, steps):
        try:
            barrier.wait()
            for step in range(steps):
                outputs[index].append(batcher.submit((index, step)))
        finally:
            batcher.participant_done()

    try:
        with batcher.batch_scope(3):
            threads = [
                threading.Thread(target=submit, args=(0, 1)),
                threading.Thread(target=submit, args=(1, 2)),
                threading.Thread(target=submit, args=(2, 2)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        metrics = batcher.metrics()
    finally:
        batcher.close()

    assert outputs == [[(0, 0)], [(1, 0), (1, 1)], [(2, 0), (2, 1)]]
    assert metrics["samples"] == 5
    assert metrics["batches"] == 2
    assert metrics["mean_batch_fill_ratio"] == 1.0


def test_lockstep_fp32_preserves_threaded_rollout_trajectory():
    torch.manual_seed(1234)
    prototype = FullTurnActorCritic(_small_policy()).eval()

    def collect(backend):
        actor = copy.deepcopy(prototype).eval()
        opponent = copy.deepcopy(prototype).eval()
        batcher = ActorCriticBatcher(
            actor,
            torch.device("cpu"),
            2,
            name=f"test-{backend}",
            wait_seconds=0.0,
            precision="fp32",
            pad_batches=backend == "lockstep",
        )
        try:
            return collect_episodes_batched(
                actor,
                [
                    (lambda: RolloutAgent(opponent, device="cpu", deterministic=True), 71, "snapshot"),
                    (lambda: RolloutAgent(opponent, device="cpu", deterministic=True), 72, "snapshot"),
                ],
                default_reward_program(),
                device=torch.device("cpu"),
                inference_backend=batcher.submit,
                max_turns=8,
                rollout_backend=backend,
            )
        finally:
            batcher.close()

    def signature(episodes):
        return [
            (
                episode.team,
                episode.outcome,
                episode.final_metrics.values,
                [[(decision.identity, decision.action) for decision in record.decisions] for record in episode.records],
            )
            for episode in episodes
        ]

    assert signature(collect("lockstep")) == signature(collect("threaded"))


def test_dry_run_archive_is_json_serializable():
    candidate = initial_candidate(island=0, seed=42)
    encoded = json.dumps(candidate.to_dict(), sort_keys=True)
    assert candidate.candidate_id in encoded


def _league_evaluation(outcomes, p95=0.1):
    games = []
    for seed, outcome in enumerate(outcomes, start=10):
        games.append({"anchor": "anchor", "seed": seed, "orientation": 0, "outcome": outcome})
    return {"games": games, "candidate_inference_p95_seconds": p95}


def _guarded_evaluation(outcome, *, normalized_city_loss=0.2, stranded=0.1):
    survival = {
        "final_city_zero": False,
        "last_night_survival": True,
        "min_night_fuel_margin": 0.1,
        "night_start_fuel_margin_mean": 0.2,
        "night_start_fuel_margin_p10": 0.0,
        "max_night_start_stranded_fuel_fraction": stranded,
        "normalized_city_tile_loss": normalized_city_loss,
        "city_destroyed_night_fuel_count": 1,
        "city_tiles_lost": 2,
    }
    games = [
        {
            "anchor": anchor,
            "seed": seed,
            "orientation": orientation,
            "outcome": outcome,
            "survival": survival,
        }
        for anchor in ("resattn8-base", "first-place")
        for seed in (10, 11)
        for orientation in (0, 1)
    ]
    return {"games": games, "candidate_inference_p95_seconds": 0.1}


def test_stage_advancement_requires_score_and_survival_non_regression():
    baseline = _guarded_evaluation(0.0)
    passing = stage_advancement_report(baseline, baseline, score_margin=0.02)
    score_failure = stage_advancement_report(_guarded_evaluation(-1.0), baseline, score_margin=0.02)
    survival_failure = stage_advancement_report(
        _guarded_evaluation(0.0, normalized_city_loss=0.23),
        baseline,
        score_margin=0.02,
    )

    assert passing["passes"] is True
    assert score_failure["checks"]["overall_score"] is False
    assert survival_failure["checks"]["normalized_city_loss"] is False


def test_short_selection_excludes_candidates_that_fail_the_advancement_gate(tmp_path):
    passing_candidate = initial_candidate(island=0, seed=41)
    failing_candidate = initial_candidate(island=1, seed=42)
    candidates = {
        passing_candidate.candidate_id: passing_candidate,
        failing_candidate.candidate_id: failing_candidate,
    }
    store = EvolutionStore(tmp_path)
    for candidate in candidates.values():
        store.save_candidate(candidate)
    baseline = _guarded_evaluation(0.0)
    results = [
        CandidateResult(
            passing_candidate.candidate_id,
            "short-resattn8",
            "completed",
            0.5,
            0.5,
            0.0,
            1.0,
            {"evaluation": baseline},
        ),
        CandidateResult(
            failing_candidate.candidate_id,
            "short-resattn8",
            "completed",
            0.0,
            0.0,
            0.0,
            1.0,
            {"evaluation": _guarded_evaluation(-1.0)},
        ),
    ]

    selected = _load_or_create_stage_selection(
        store,
        candidates,
        results,
        name="medium",
        target_stage="medium-resattn8",
        source_stage="short-resattn8",
        count=1,
        baseline=baseline,
    )

    assert [candidate.candidate_id for candidate in selected] == [passing_candidate.candidate_id]
    report = json.loads((tmp_path / "selections" / "medium.json").read_text())
    assert report["advancement_reports"][passing_candidate.candidate_id]["passes"] is True
    assert report["advancement_reports"][failing_candidate.candidate_id]["passes"] is False


def test_short_game_budget_selection_enforces_teacher_kl_and_entropy_guards():
    candidate = initial_candidate(island=0, seed=43)
    baseline = _guarded_evaluation(0.0)
    result = CandidateResult(
        candidate.candidate_id,
        "short-resattn8",
        "completed",
        0.5,
        0.5,
        0.0,
        1.0,
        {
            "evaluation": baseline,
            "training": {
                "budget_unit": "games",
                "history": [
                    {"online_teacher_kl": 0.02, "entropy": 0.10},
                    {"online_teacher_kl": 0.031, "entropy": 0.10},
                ],
            },
        },
    )

    selected = _select_completed_stage(
        {candidate.candidate_id: candidate},
        [result],
        stage="short-resattn8",
        count=1,
        baseline=baseline,
    )

    assert selected == []


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


def test_teacher_guard_rejects_base_gain_that_regresses_teacher():
    def evaluation(base_outcomes, teacher_outcomes):
        games = []
        survival = {
            "final_city_zero": False,
            "last_night_survival": True,
            "min_night_fuel_margin": 0.1,
            "night_start_fuel_margin_p10": 0.0,
            "normalized_city_tile_loss": 0.2,
        }
        for anchor, outcomes in (("resattn8-base", base_outcomes), ("first-place", teacher_outcomes)):
            for seed, outcome in enumerate(outcomes, start=10):
                games.append(
                    {
                        "anchor": anchor,
                        "seed": seed,
                        "orientation": 0,
                        "outcome": outcome,
                        "survival": survival,
                    }
                )
        return {"games": games, "candidate_inference_p95_seconds": 0.1}

    baseline = evaluation([-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 1.0, 1.0])
    teacher_regression = evaluation([1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0])
    report = acceptance_report(
        {"resattn8": teacher_regression},
        {"resattn8": baseline},
        enforce_teacher_guard=True,
        require_survival=True,
        seed=4,
    )

    architecture = report["architectures"]["resattn8"]
    assert architecture["base_score_rate_delta"] > 0
    assert architecture["teacher_score_rate_delta"] < 0
    assert architecture["teacher_guard_passes"] is False
    assert report["promote"] is False


@pytest.mark.parametrize(
    ("candidate_stranded", "passes"),
    [(0.119, True), (0.121, False)],
)
def test_stranded_fuel_promotion_gate_uses_two_percent_noninferiority_margin(candidate_stranded, passes):
    def evaluation(outcomes, stranded):
        games = [
            {
                "anchor": "resattn8-base",
                "seed": seed,
                "orientation": 0,
                "outcome": outcome,
                "survival": {"max_night_start_stranded_fuel_fraction": stranded},
            }
            for seed, outcome in enumerate(outcomes, start=10)
        ]
        return {"games": games, "candidate_inference_p95_seconds": 0.1}

    baseline = evaluation([-1.0, -1.0, -1.0], 0.1)
    candidate = evaluation([1.0, 1.0, 1.0], candidate_stranded)
    report = acceptance_report(
        {"resattn8": candidate},
        {"resattn8": baseline},
        require_stranded_fuel=True,
        seed=5,
    )

    architecture = report["architectures"]["resattn8"]
    assert architecture["stranded_fuel_delta"] == pytest.approx(candidate_stranded - 0.1)
    assert architecture["stranded_fuel_passes"] is passes
    assert report["promote"] is passes


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


def test_joint_decoder_deterministic_rollout_matches_exported_evaluation_agent(tmp_path):
    actor = FullTurnActorCritic(_small_policy()).eval()
    checkpoint = tmp_path / "joint.pt"
    actor.export_policy(checkpoint, epoch=0, metrics={}, split={}, metadata={})
    game = Game({"seed": 23})
    rollout = RolloutAgent(actor, device="cpu", deterministic=True)
    rollout.set_team(0)
    rollout.game_start(game)
    evaluation = BehaviorCloningAgent(str(checkpoint), device="cpu", tta="auto")
    evaluation.set_team(0)

    rollout_actions = [action.to_message(game) for action in rollout.process_turn(game, 0)]
    evaluation_actions = [action.to_message(game) for action in evaluation.process_turn(game, 0)]

    assert rollout_actions == evaluation_actions
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["decoder_schema"] == "joint_sequential_v2"
    assert saved["inference_augmentation"] == "rot180"


def test_actor_critic_rot180_tta_is_equivariant():
    actor = FullTurnActorCritic(_small_policy()).eval()
    observation = torch.randn(1, actor.policy.config.input_channels, 32, 32)
    rotated = torch.rot90(observation, 2, dims=(-2, -1)).clone()
    rotated[:, FEATURE_INDEX["x_coordinate"]].neg_()
    rotated[:, FEATURE_INDEX["y_coordinate"]].neg_()

    output, value = actor.forward_tta(observation)
    rotated_output, rotated_value = actor.forward_tta(rotated)
    restored = actor._restore_rot180(rotated_output)

    assert torch.allclose(value, rotated_value, atol=1e-6, rtol=1e-6)
    assert all(torch.allclose(output[name], restored[name], atol=1e-6, rtol=1e-6) for name in output)


def test_engine_records_night_city_loss_and_invalid_action_turns():
    game = Game({"seed": 13})
    tile = game.spawn_city_tile(0, 0, 0)
    city = game.cities[tile.city_id]
    city.fuel = 0.0
    game.state["turn"] = 30

    game.record_night_fuel_diagnostics()
    fuel_event = game.diagnostic_events[-2]
    assert fuel_event["event"] == "night_fuel_snapshot"
    assert fuel_event["team"] == 0
    assert fuel_event["night_start"] is True
    assert fuel_event["city_tiles"] >= 1
    assert fuel_event["fuel_margin"] < 0
    assert 0.0 <= fuel_event["stranded_fuel_fraction"] <= 1.0

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


def test_survival_summary_uses_maximum_night_start_stranded_fuel():
    game = Game({"seed": 13})
    game.diagnostic_events.extend(
        [
            {
                "event": "night_fuel_snapshot",
                "turn": 30,
                "team": 0,
                "night_start": True,
                "stranded_fuel_fraction": 0.1,
            },
            {
                "event": "night_fuel_snapshot",
                "turn": 31,
                "team": 0,
                "night_start": False,
                "stranded_fuel_fraction": 0.9,
            },
            {
                "event": "night_fuel_snapshot",
                "turn": 70,
                "team": 0,
                "night_start": True,
                "stranded_fuel_fraction": 0.4,
            },
        ]
    )

    summary = _survival_summary(game, 0)

    assert summary["max_night_start_stranded_fuel_fraction"] == pytest.approx(0.4)
    assert summary["max_night_start_stranded_fuel_turn"] == 70


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
                    {
                        "event": "night_fuel_snapshot",
                        "turn": 30,
                        "night_start": True,
                        "stranded_fuel_fraction": 0.4,
                    },
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
    assert reflection["diagnostics"]["max_night_start_stranded_fuel_fraction"] == pytest.approx(0.4)
    assert reflection["diagnostics"]["max_night_start_stranded_fuel_turn"] == 30
    assert reflection["diagnostics"]["illegal_action_turns"] == [32]
    assert reflected.metrics["training"]["diagnostic_event_count"] == 3
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
        (worker_artifacts / "best_rl.pt").write_bytes(b"policy-and-value")
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
        assert (downloaded / "best_rl.pt").read_bytes() == b"policy-and-value"
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
    assert (coordinator_artifacts / "best_rl.pt").read_bytes() == b"policy-and-value"
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
        {
            "schema_version": 4,
            "metric_schema_version": 3,
            "arguments": {"resattn8_only": True, "resattn8_checkpoint": "original.pt"},
        },
    )

    assert worker_args.resattn8_only is True
    assert worker_args.resattn8_checkpoint == "original.pt"


def test_worker_uses_legacy_rollout_runtime_for_old_manifest():
    worker_args = SimpleNamespace(
        rollout_backend="auto",
        rollout_precision="auto",
        rollout_compile="auto",
        rollout_batch_wait_ms=9.0,
    )

    _apply_coordinator_manifest(
        worker_args,
        {
            "schema_version": 4,
            "metric_schema_version": 3,
            "arguments": {},
        },
    )

    assert worker_args.rollout_backend == "threaded"
    assert worker_args.rollout_precision == "fp32"
    assert worker_args.rollout_compile == "off"
    assert worker_args.rollout_batch_wait_ms == 2.0


def test_manifest_rejects_changed_training_curriculum():
    worker_args = SimpleNamespace(
        curriculum_profile="dense_shaping",
        bc_anchor_max_turns=0,
        bc_anchor_sampling="phase-balanced",
    )
    stale_curriculum = training_curriculum("dense_shaping").to_dict()
    stale_curriculum["bc_coefficient_points"] = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]

    with pytest.raises(ValueError, match="Training curriculum changed"):
        _apply_coordinator_manifest(
            worker_args,
            {
                "schema_version": 4,
                "metric_schema_version": 3,
                "arguments": {"curriculum_profile": "dense_shaping"},
                "training_curriculum": stale_curriculum,
            },
        )


def test_auto_rollout_backend_stays_threaded_until_acceptance_gate_passes():
    assert resolve_rollout_backend("auto") == "threaded"
    assert resolve_rollout_backend("lockstep") == "lockstep"


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


def test_infrastructure_failures_are_retryable_but_candidate_errors_are_not(tmp_path):
    job = EvolutionJob("candidate", "short-resattn8", "resattn8", 0, 1, 1)
    missing_replay = CandidateResult(
        "candidate",
        "short-resattn8",
        "failed",
        0.0,
        0.0,
        float("inf"),
        1.0,
        {},
        "FileNotFoundError: /app/replay_datasets/missing.json",
    )
    invalid_reward = CandidateResult(
        "candidate",
        "short-resattn8",
        "failed",
        0.0,
        0.0,
        float("inf"),
        1.0,
        {},
        "ValueError: derived reward metric is invalid",
    )

    assert _is_retryable_infrastructure_failure(missing_replay)
    assert not _is_retryable_infrastructure_failure(invalid_reward)
    assert _record_job_retry(tmp_path, job, missing_replay) == 1
    assert _record_job_retry(tmp_path, job, missing_replay) == 2
    assert _job_retry_count(tmp_path, job) == 2
    _record_skipped_job(tmp_path, job, missing_replay)
    skipped = json.loads((tmp_path / "jobs" / "skipped" / f"{job.job_id}.json").read_text())
    assert skipped["status"] == "skipped_after_infrastructure_retries"
    assert skipped["retry_count"] == 2


def test_opponent_mix_allocate_counts_and_shuffles():
    mix = OpponentMix(self_base=0.20, other_base=0.05, teacher=0.25, snapshot=0.50)
    rng = random.Random(42)  # noqa: S311
    allocated = [name for _ in range(1000) for name in mix.allocate(8, rng)]

    assert len(allocated) == 8000
    assert allocated.count("self_base") / len(allocated) == pytest.approx(0.20, abs=0.015)
    assert allocated.count("other_base") / len(allocated) == pytest.approx(0.05, abs=0.015)
    assert allocated.count("teacher") / len(allocated) == pytest.approx(0.25, abs=0.015)
    assert allocated.count("snapshot") / len(allocated) == pytest.approx(0.50, abs=0.015)


def test_reward_program_modes():
    previous = _metrics(city_tiles=-0.5, city_survival=-0.5)
    following = _metrics(city_tiles=0.5, city_survival=0.5)

    linear_prog = default_reward_program(mode="potential_linear")
    tanh_prog = default_reward_program(mode="potential_tanh")
    direct_prog = default_reward_program(mode="direct_step")

    linear_breakdown = linear_prog.reward(previous, following, terminal_outcome=1.0)
    tanh_breakdown = tanh_prog.reward(previous, following, terminal_outcome=1.0)
    direct_breakdown = direct_prog.reward(previous, following, terminal_outcome=1.0)

    assert linear_breakdown.shaping > tanh_breakdown.shaping
    assert direct_breakdown.shaping == pytest.approx(
        linear_prog.reward_scale * (1.5 * 1.0 + 0.8 * 1.0) / linear_prog.terminal_reward_scale
    )


def test_gae_return_is_td_lambda_target():
    first = SimpleNamespace(reward=0.1, value=0.2, advantage=0.0, return_value=0.0)
    second = SimpleNamespace(reward=1.0, value=0.4, advantage=0.0, return_value=0.0)
    episode = SimpleNamespace(records=[first, second])
    config = PPOConfig(gamma=0.9, gae_lambda=0.8)

    calculate_gae([episode], config)

    expected_last = 1.0
    expected_first = (1.0 - config.gae_lambda) * (0.1 + config.gamma * 0.4) + config.gae_lambda * (
        0.1 + config.gamma * expected_last
    )
    assert second.return_value == pytest.approx(expected_last)
    assert first.return_value == pytest.approx(expected_first)
