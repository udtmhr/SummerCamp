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
    return {
        "candidate": candidate.name,
        "anchors": [anchor.name for anchor in anchors],
        "seed_start": seed_start,
        "seed_count": seed_count,
        "games": games,
        "totals": {**totals, "games": game_count, "score_rate": totals["points"] / game_count},
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
) -> dict[str, object]:
    if candidates.keys() != baselines.keys():
        raise ValueError("Acceptance requires matching candidate and baseline architectures")
    architecture_reports = {}
    combined = []
    for architecture in sorted(candidates):
        seed_deltas, anchor_deltas = paired_seed_deltas(candidates[architecture], baselines[architecture])
        combined.extend(seed_deltas.values())
        point_delta = sum(seed_deltas.values()) / len(seed_deltas)
        candidate_p95 = float(candidates[architecture]["candidate_inference_p95_seconds"])
        baseline_p95 = float(baselines[architecture]["candidate_inference_p95_seconds"])
        architecture_reports[architecture] = {
            "score_rate_delta": point_delta,
            "bootstrap_lcb": bootstrap_lower_bound(list(seed_deltas.values()), seed=seed),
            "anchor_deltas": anchor_deltas,
            "candidate_inference_p95_seconds": candidate_p95,
            "baseline_inference_p95_seconds": baseline_p95,
            "passes": (
                point_delta >= 0.0
                and min(anchor_deltas.values()) >= -0.05
                and candidate_p95 < 1.0
                and candidate_p95 <= baseline_p95 * 1.1
            ),
        }
    combined_lcb = bootstrap_lower_bound(combined, seed=seed)
    return {
        "combined_score_rate_delta": sum(combined) / len(combined),
        "combined_bootstrap_lcb": combined_lcb,
        "architectures": architecture_reports,
        "promote": combined_lcb > 0.0 and all(report["passes"] for report in architecture_reports.values()),
    }
