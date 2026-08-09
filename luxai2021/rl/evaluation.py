from __future__ import annotations

# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, PLR2004, S311, TC003
import copy
import random
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.imitation.agent import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.rl.policy import deterministic_outcome


def _survival_summary(game: object, team: int) -> dict[str, float | bool | int | None]:
    events = [
        event
        for event in getattr(game, "diagnostic_events", ())
        if int(event.get("team", team)) == team
    ]
    snapshots = [event for event in events if event.get("event") == "night_fuel_snapshot"]
    city_losses = [event for event in events if event.get("event") == "city_destroyed_night_fuel"]
    final_city_tiles = sum(len(city.city_cells) for city in game.cities.values() if int(city.team) == team)
    peak_city_tiles = max(
        [int(event.get("peak_city_tiles", 0)) for event in snapshots] + [final_city_tiles, 1]
    )
    last_night = [event for event in snapshots if int(event.get("turn", -1)) >= 350]
    last_night_start = min(last_night, key=lambda event: int(event.get("turn", 0))) if last_night else None
    night_starts = [event for event in snapshots if bool(event.get("night_start", False))]
    night_start_margins = sorted(float(event.get("min_city_fuel_margin", -1.0)) for event in night_starts)
    max_stranded = (
        max(night_starts, key=lambda event: float(event.get("stranded_fuel_fraction", 0.0)))
        if night_starts
        else None
    )
    return {
        "final_city_tiles": float(final_city_tiles),
        "final_city_zero": final_city_tiles == 0,
        "last_night_survival": (
            bool(int(last_night_start.get("city_tiles", 0)) > 0 and final_city_tiles > 0)
            if last_night_start is not None
            else None
        ),
        "min_night_fuel_margin": (
            min(float(event.get("min_city_fuel_margin", -1.0)) for event in snapshots)
            if snapshots
            else None
        ),
        "night_start_fuel_margin_mean": (
            sum(night_start_margins) / len(night_start_margins) if night_start_margins else None
        ),
        "night_start_fuel_margin_p10": (
            night_start_margins[int(0.10 * (len(night_start_margins) - 1))]
            if night_start_margins
            else None
        ),
        "max_night_start_stranded_fuel_fraction": (
            float(max_stranded.get("stranded_fuel_fraction", 0.0)) if max_stranded is not None else None
        ),
        "max_night_start_stranded_fuel_turn": (
            int(max_stranded.get("turn", 0)) if max_stranded is not None else None
        ),
        "normalized_city_tile_loss": sum(int(event.get("city_tiles_lost", 0)) for event in city_losses)
        / peak_city_tiles,
        "city_destroyed_night_fuel_count": len(city_losses),
        "city_tiles_lost": sum(int(event.get("city_tiles_lost", 0)) for event in city_losses),
    }


@dataclass(frozen=True)
class LeagueMember:
    name: str
    checkpoint: Path
    model_type: Literal["bc", "first-place"] = "bc"


class TimedAgent(Agent):
    def __init__(self, inner: Agent) -> None:
        super().__init__()
        self.inner = inner
        self.turn_seconds: list[float] = []

    def set_team(self, team: int) -> None:
        super().set_team(team)
        self.inner.set_team(team)

    def set_controller(self, controller: object) -> None:
        super().set_controller(controller)
        self.inner.set_controller(controller)

    def game_start(self, game: object) -> None:
        self.turn_seconds = []
        self.inner.game_start(game)

    def process_turn(self, game: object, team: int) -> list[object]:
        started_at = perf_counter()
        actions = self.inner.process_turn(game, team)
        self.turn_seconds.append(perf_counter() - started_at)
        return actions


def create_league_agent(member: LeagueMember, device: str) -> BehaviorCloningAgent:
    if member.model_type == "first-place":
        return FirstPlaceAgent(str(member.checkpoint), device=device, tta="rot180")
    return BehaviorCloningAgent(str(member.checkpoint), device=device, tta="auto")


def evaluate_against_league(
    candidate: LeagueMember,
    anchors: list[LeagueMember],
    *,
    seed_start: int,
    seed_count: int,
    device: str,
    max_turns: int | None = None,
) -> dict[str, object]:
    if seed_count < 1:
        raise ValueError("League evaluation requires at least one seed")
    games = []
    totals = {"wins": 0, "losses": 0, "draws": 0, "points": 0.0}
    started_at = perf_counter()
    for anchor in anchors:
        pair = {member.name: {"wins": 0, "losses": 0, "draws": 0, "points": 0.0} for member in (candidate, anchor)}
        for seed in range(seed_start, seed_start + seed_count):
            for orientation in (0, 1):
                first, second = (candidate, anchor) if orientation == 0 else (anchor, candidate)
                first_agent = create_league_agent(first, device)
                second_agent = create_league_agent(second, device)
                timed_candidate = TimedAgent(first_agent if first.name == candidate.name else second_agent)
                if first.name == candidate.name:
                    first_agent = timed_candidate
                else:
                    second_agent = timed_candidate
                config = copy.deepcopy(LuxMatchConfigs_Default)
                if max_turns is not None:
                    config["parameters"]["MAX_DAYS"] = max_turns
                environment = LuxEnvironment(config, first_agent, second_agent)
                game_started = perf_counter()
                with suppress(StopIteration):
                    environment.reset(seed=seed)
                candidate_team = first_agent.team if first.name == candidate.name else second_agent.team
                outcome = deterministic_outcome(environment.game, candidate_team)
                survival = _survival_summary(environment.game, candidate_team)
                if outcome > 0:
                    totals["wins"] += 1
                    totals["points"] += 1.0
                    pair[candidate.name]["wins"] += 1
                    pair[candidate.name]["points"] += 1.0
                    pair[anchor.name]["losses"] += 1
                    winner = candidate.name
                elif outcome < 0:
                    totals["losses"] += 1
                    pair[candidate.name]["losses"] += 1
                    pair[anchor.name]["wins"] += 1
                    pair[anchor.name]["points"] += 1.0
                    winner = anchor.name
                else:
                    totals["draws"] += 1
                    totals["points"] += 0.5
                    for member in (candidate, anchor):
                        pair[member.name]["draws"] += 1
                        pair[member.name]["points"] += 0.5
                    winner = None
                games.append(
                    {
                        "anchor": anchor.name,
                        "seed": seed,
                        "orientation": orientation,
                        "candidate_team": candidate_team,
                        "winner": winner,
                        "outcome": outcome,
                        "end_turn": int(environment.game.state["turn"]),
                        "survival": survival,
                        "duration_seconds": perf_counter() - game_started,
                        "candidate_inference_seconds": timed_candidate.turn_seconds,
                        "diagnostic_events": [
                            dict(event)
                            for event in getattr(environment.game, "diagnostic_events", ())
                            if int(event.get("team", candidate_team)) == candidate_team
                        ],
                    }
                )
                del environment, first_agent, second_agent, timed_candidate
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        pair_games = seed_count * 2
        for value in pair.values():
            value["score_rate"] = value["points"] / pair_games
        pair["games"] = pair_games
        games.append({"pair_summary": pair, "anchor": anchor.name})
    game_count = len(anchors) * seed_count * 2
    inference_seconds = [value for game in games for value in game.get("candidate_inference_seconds", ())]
    inference_seconds.sort()
    percentile_index = min(len(inference_seconds) - 1, int(len(inference_seconds) * 0.95))
    p95 = inference_seconds[percentile_index] if inference_seconds else float("nan")
    by_orientation = {}
    for orientation in (0, 1):
        orientation_games = [game for game in games if game.get("orientation") == orientation and "outcome" in game]
        points = sum((float(game["outcome"]) + 1.0) * 0.5 for game in orientation_games)
        by_orientation[str(orientation)] = {
            "games": len(orientation_games),
            "score_rate": points / max(len(orientation_games), 1),
        }
    return {
        "candidate": candidate.name,
        "anchors": [anchor.name for anchor in anchors],
        "seed_start": seed_start,
        "seed_count": seed_count,
        "games": games,
        "totals": {**totals, "games": game_count, "score_rate": totals["points"] / game_count},
        "by_orientation": by_orientation,
        "duration_seconds": perf_counter() - started_at,
        "candidate_inference_p95_seconds": p95,
    }


def paired_seed_deltas(
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> tuple[dict[int, float], dict[str, float]]:
    def outcomes(evaluation: dict[str, object]) -> dict[tuple[str, int, int], float]:
        return {
            (str(game["anchor"]), int(game["seed"]), int(game["orientation"])): (float(game["outcome"]) + 1.0) * 0.5
            for game in evaluation["games"]
            if "outcome" in game
        }

    candidate_outcomes = outcomes(candidate)
    baseline_outcomes = outcomes(baseline)
    if candidate_outcomes.keys() != baseline_outcomes.keys():
        raise ValueError("Candidate and baseline league games do not have matching seeds")
    seed_values: dict[int, list[float]] = {}
    anchor_values: dict[str, list[float]] = {}
    for (anchor, seed, orientation), value in candidate_outcomes.items():
        delta = value - baseline_outcomes[(anchor, seed, orientation)]
        seed_values.setdefault(seed, []).append(delta)
        anchor_values.setdefault(anchor, []).append(delta)
    seed_deltas = {seed: sum(values) / len(values) for seed, values in seed_values.items()}
    anchor_deltas = {anchor: sum(values) / len(values) for anchor, values in anchor_values.items()}
    return seed_deltas, anchor_deltas


def paired_anchor_seed_deltas(
    candidate: dict[str, object], baseline: dict[str, object]
) -> dict[str, dict[int, float]]:
    def outcomes(evaluation: dict[str, object]) -> dict[tuple[str, int, int], float]:
        return {
            (str(game["anchor"]), int(game["seed"]), int(game["orientation"])): (float(game["outcome"]) + 1.0)
            * 0.5
            for game in evaluation["games"]
            if "outcome" in game
        }

    candidate_outcomes = outcomes(candidate)
    baseline_outcomes = outcomes(baseline)
    if candidate_outcomes.keys() != baseline_outcomes.keys():
        raise ValueError("Candidate and baseline league games do not have matching seeds")
    grouped: dict[str, dict[int, list[float]]] = {}
    for (anchor, seed, orientation), value in candidate_outcomes.items():
        grouped.setdefault(anchor, {}).setdefault(seed, []).append(
            value - baseline_outcomes[(anchor, seed, orientation)]
        )
    return {
        anchor: {seed: sum(values) / len(values) for seed, values in by_seed.items()}
        for anchor, by_seed in grouped.items()
    }


def _paired_survival_deltas(
    candidate: dict[str, object], baseline: dict[str, object]
) -> dict[str, float | bool]:
    def games(evaluation: dict[str, object]) -> dict[tuple[str, int, int], dict[str, object]]:
        return {
            (str(game["anchor"]), int(game["seed"]), int(game["orientation"])): game
            for game in evaluation["games"]
            if "outcome" in game and isinstance(game.get("survival"), dict)
        }

    candidate_games = games(candidate)
    baseline_games = games(baseline)
    if not candidate_games or candidate_games.keys() != baseline_games.keys():
        return {"available": False}

    def mean_delta(name: str) -> float | None:
        values = []
        for key, game in candidate_games.items():
            candidate_value = game["survival"].get(name)
            baseline_value = baseline_games[key]["survival"].get(name)
            if candidate_value is not None and baseline_value is not None:
                values.append(float(candidate_value) - float(baseline_value))
        return sum(values) / len(values) if values else None

    return {
        "available": True,
        "final_city_zero_delta": mean_delta("final_city_zero"),
        "last_night_survival_delta": mean_delta("last_night_survival"),
        "night_fuel_margin_delta": mean_delta("min_night_fuel_margin"),
        "night_start_fuel_margin_mean_delta": mean_delta("night_start_fuel_margin_mean"),
        "night_start_fuel_margin_p10_delta": mean_delta("night_start_fuel_margin_p10"),
        "stranded_fuel_delta": mean_delta("max_night_start_stranded_fuel_fraction"),
        "normalized_city_tile_loss_delta": mean_delta("normalized_city_tile_loss"),
        "city_destroyed_night_fuel_count_delta": mean_delta("city_destroyed_night_fuel_count"),
        "city_tiles_lost_delta": mean_delta("city_tiles_lost"),
    }


def stage_advancement_report(
    candidate: dict[str, object],
    baseline: dict[str, object],
    *,
    score_margin: float,
    survival_margin: float = 0.02,
) -> dict[str, object]:
    """Return the paired short/medium non-regression gate used before expensive stages."""
    if not 0.0 <= score_margin <= 1.0 or not 0.0 <= survival_margin <= 1.0:
        raise ValueError("Stage advancement margins must be in [0, 1]")
    seed_deltas, anchor_deltas = paired_seed_deltas(candidate, baseline)
    overall_delta = sum(seed_deltas.values()) / max(len(seed_deltas), 1)
    teacher_delta = anchor_deltas.get("first-place")
    base_deltas = [float(value) for name, value in anchor_deltas.items() if name != "first-place"]
    base_delta = max(base_deltas, default=None)
    survival = _paired_survival_deltas(candidate, baseline)
    extinction_delta = survival.get("final_city_zero_delta")
    last_night_delta = survival.get("last_night_survival_delta")
    city_loss_delta = survival.get("normalized_city_tile_loss_delta")
    stranded_delta = survival.get("stranded_fuel_delta")
    fuel_p10_delta = survival.get("night_start_fuel_margin_p10_delta")
    checks = {
        "overall_score": overall_delta >= -score_margin,
        "base_score": base_delta is not None and base_delta >= -score_margin,
        "teacher_score": teacher_delta is not None and teacher_delta >= -score_margin,
        "extinction": extinction_delta is not None and float(extinction_delta) <= 0.0,
        "last_night_survival": last_night_delta is not None and float(last_night_delta) >= 0.0,
        "normalized_city_loss": city_loss_delta is not None and float(city_loss_delta) <= survival_margin,
        "stranded_fuel": stranded_delta is not None and float(stranded_delta) <= survival_margin,
        "night_start_fuel_margin_p10": fuel_p10_delta is not None and float(fuel_p10_delta) >= -survival_margin,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "score_margin": score_margin,
        "survival_margin": survival_margin,
        "overall_score_delta": overall_delta,
        "teacher_score_delta": teacher_delta,
        "base_score_delta": base_delta,
        "survival": survival,
    }


def bootstrap_lower_bound(values: list[float], *, seed: int = 42, samples: int = 10_000) -> float:
    if not values:
        raise ValueError("Bootstrap requires at least one paired value")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return means[int(0.025 * (samples - 1))]


def acceptance_report(
    candidates: dict[str, dict[str, object]],
    baselines: dict[str, dict[str, object]],
    *,
    seed: int = 42,
    enforce_teacher_guard: bool = False,
    teacher_noninferiority_margin: float = 0.02,
    require_survival: bool = False,
    require_stranded_fuel: bool = False,
    stranded_fuel_noninferiority_margin: float = 0.02,
) -> dict[str, object]:
    if candidates.keys() != baselines.keys():
        raise ValueError("Acceptance requires matching candidate and baseline architectures")
    if not 0.0 <= stranded_fuel_noninferiority_margin <= 1.0:
        raise ValueError("Stranded-fuel noninferiority margin must be in [0, 1]")
    architecture_reports = {}
    combined = []
    for architecture in sorted(candidates):
        seed_deltas, anchor_deltas = paired_seed_deltas(candidates[architecture], baselines[architecture])
        anchor_seed_deltas = paired_anchor_seed_deltas(candidates[architecture], baselines[architecture])
        combined.extend(seed_deltas.values())
        point_delta = sum(seed_deltas.values()) / len(seed_deltas)
        candidate_p95 = float(candidates[architecture]["candidate_inference_p95_seconds"])
        baseline_p95 = float(baselines[architecture]["candidate_inference_p95_seconds"])
        teacher_name = next((name for name in anchor_deltas if name == "first-place"), None)
        base_name = next((name for name in anchor_deltas if name != teacher_name), None)
        teacher_delta = anchor_deltas.get(teacher_name, 0.0) if teacher_name is not None else None
        base_delta = anchor_deltas.get(base_name, 0.0) if base_name is not None else None
        teacher_lcb = (
            bootstrap_lower_bound(list(anchor_seed_deltas[teacher_name].values()), seed=seed)
            if teacher_name is not None
            else None
        )
        survival = _paired_survival_deltas(candidates[architecture], baselines[architecture])
        final_zero_delta = survival.get("final_city_zero_delta")
        last_night_delta = survival.get("last_night_survival_delta")
        fuel_margin_delta = survival.get("night_fuel_margin_delta")
        fuel_p10_delta = survival.get("night_start_fuel_margin_p10_delta")
        stranded_fuel_delta = survival.get("stranded_fuel_delta")
        city_loss_delta = survival.get("normalized_city_tile_loss_delta")
        survival_passes = (
            not require_survival
            or (
                bool(survival.get("available"))
                and final_zero_delta is not None
                and float(final_zero_delta) <= 0.0
                and last_night_delta is not None
                and float(last_night_delta) >= 0.0
                and fuel_margin_delta is not None
                and float(fuel_margin_delta) >= -0.02
                and fuel_p10_delta is not None
                and float(fuel_p10_delta) >= -0.02
                and city_loss_delta is not None
                and float(city_loss_delta) <= 0.02
            )
        )
        stranded_fuel_passes = (
            not require_stranded_fuel
            or (
                bool(survival.get("available"))
                and stranded_fuel_delta is not None
                and float(stranded_fuel_delta) <= stranded_fuel_noninferiority_margin
            )
        )
        teacher_passes = (
            not enforce_teacher_guard
            or (
                teacher_delta is not None
                and teacher_delta >= 0.0
                and teacher_lcb is not None
                and teacher_lcb >= -teacher_noninferiority_margin
                and base_delta is not None
                and base_delta >= 0.0
            )
        )
        score_latency_passes = (
            point_delta >= 0.0
            and min(anchor_deltas.values()) >= -0.05
            and candidate_p95 < 1.0
            and candidate_p95 <= baseline_p95 * 1.1
        )
        architecture_reports[architecture] = {
            "score_rate_delta": point_delta,
            "bootstrap_lcb": bootstrap_lower_bound(list(seed_deltas.values()), seed=seed),
            "anchor_deltas": anchor_deltas,
            "candidate_inference_p95_seconds": candidate_p95,
            "baseline_inference_p95_seconds": baseline_p95,
            "teacher_score_rate_delta": teacher_delta,
            "teacher_bootstrap_lcb": teacher_lcb,
            "base_score_rate_delta": base_delta,
            "survival": survival,
            "stranded_fuel_delta": stranded_fuel_delta,
            "teacher_guard_passes": teacher_passes,
            "survival_passes": survival_passes,
            "stranded_fuel_passes": stranded_fuel_passes,
            "score_latency_passes": score_latency_passes,
            "passes": (
                score_latency_passes
                and teacher_passes
                and survival_passes
                and stranded_fuel_passes
            ),
        }
    combined_lcb = bootstrap_lower_bound(combined, seed=seed)
    return {
        "combined_score_rate_delta": sum(combined) / len(combined),
        "combined_bootstrap_lcb": combined_lcb,
        "architectures": architecture_reports,
        "promote": combined_lcb > 0.0 and all(report["passes"] for report in architecture_reports.values()),
    }
