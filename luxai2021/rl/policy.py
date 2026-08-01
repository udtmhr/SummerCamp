from __future__ import annotations

# ruff: noqa: C901, PLR0912, PLR0913, PLR0915, TC003
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from luxai2021.env.agent import Agent
from luxai2021.game.actions import ResearchAction, SpawnCartAction, SpawnWorkerAction
from luxai2021.imitation.actions import CITY_ACTIONS, FIRST_PLACE_ACTION_SCHEMA
from luxai2021.imitation.agent import BehaviorCloningAgent, _Candidate, _Choice
from luxai2021.imitation.first_place import first_place_city_legal_mask, first_place_unit_legal_mask
from luxai2021.imitation.masking import apply_legal_action_mask, monotonically_tighten_legal_mask
from luxai2021.imitation.model import (
    POLICY_SCHEMA_FIRST_PLACE_FLAT,
    LuxBehaviorCloningModel,
    load_bc_checkpoint,
    save_bc_checkpoint,
)
from luxai2021.imitation.schema import encode_snapshot, snapshot_from_game
from luxai2021.rl.metrics import GameMetrics, metrics_from_game

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass
class ActionDecision:
    entity: str
    position: tuple[int, int]
    action: int
    legal_mask: Tensor
    old_log_prob: float
    entropy: float
    identity: str


@dataclass
class TurnRecord:
    observation: Tensor
    metrics: GameMetrics
    decisions: list[ActionDecision]
    value: float
    reward: float = 0.0
    advantage: float = 0.0
    return_value: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeTrajectory:
    team: int
    records: list[TurnRecord]
    final_metrics: GameMetrics
    outcome: float
    seed: int
    opponent: str
    diagnostic_events: list[dict[str, object]] = field(default_factory=list)


class FullTurnActorCritic(nn.Module):
    """Existing distilled spatial policy plus a training-only global value head."""

    def __init__(self, policy: LuxBehaviorCloningModel) -> None:
        super().__init__()
        if policy.config.policy_schema != POLICY_SCHEMA_FIRST_PLACE_FLAT:
            raise ValueError("Full-turn RL requires first_place_flat_v1 checkpoints")
        self.policy = policy
        global_channels = policy.encoder.global_output_channels
        hidden = min(256, max(64, global_channels // 2))
        self.value_head = nn.Sequential(
            nn.LayerNorm(global_channels),
            nn.Linear(global_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> FullTurnActorCritic:
        policy, _ = load_bc_checkpoint(str(path), str(device))
        return cls(policy).to(device)

    def forward(self, observation: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        output = self.policy(observation, return_features=True)
        global_features = output.pop("global_features")
        output.pop("features")
        return output, self.value_head(global_features).squeeze(-1)

    def export_policy(
        self,
        path: Path,
        *,
        epoch: int,
        metrics: Mapping[str, object],
        split: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> None:
        save_bc_checkpoint(
            path,
            self.policy,
            None,
            epoch,
            metrics,
            split,
            extra_metadata={"inference_augmentation": "rot180", "rl_training": dict(metadata)},
        )


def _distribution(logits: Tensor, legal_mask: np.ndarray | Tensor) -> Categorical:
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    masked = apply_legal_action_mask(logits[None], mask[None])[0]
    return Categorical(logits=masked.float())


class RolloutAgent(BehaviorCloningAgent):
    """Samples full-turn actions while retaining the existing conflict decoder."""

    def __init__(
        self,
        actor_critic: FullTurnActorCritic,
        *,
        device: str | torch.device,
        deterministic: bool = False,
        inference_backend: Callable[[Tensor], tuple[dict[str, Tensor], Tensor]] | None = None,
        record_trajectory: bool = True,
    ) -> None:
        Agent.__init__(self)
        self.actor_critic = actor_critic
        self.model = actor_critic.policy
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.inference_backend = inference_backend
        self.record_trajectory = record_trajectory
        self.checkpoint = {"inference_augmentation": "none"}
        self.tta = "none"
        self.records: list[TurnRecord] = []
        self.generator = torch.Generator(device="cpu")

    def game_start(self, game: object) -> None:
        self.records = []
        seed = int(getattr(game, "configs", {}).get("seed", 0))
        self.generator.manual_seed(seed * 2 + self.team)

    def _choose_action(self, distribution: Categorical) -> Tensor:
        if self.deterministic:
            return distribution.logits.argmax()
        probabilities = distribution.probs.float().cpu()
        return torch.multinomial(probabilities, 1, generator=self.generator).squeeze(0)

    def _sample_units(
        self,
        game: object,
        team: int,
        snapshot: object,
        output: dict[str, Tensor],
        x_offset: int,
        y_offset: int,
    ) -> tuple[list[_Choice], list[ActionDecision]]:
        reserved_destinations: set[tuple[int, int]] = set()
        reserved_capacity: dict[str, int] = {}
        choices = []
        decisions = []
        decision_by_unit: dict[str, ActionDecision] = {}
        for unit in sorted(game.state["teamStates"][team]["units"].values(), key=lambda item: item.id):
            if not unit.can_act():
                continue
            unit_snapshot = snapshot.units[unit.id]
            entity = "worker" if unit.is_worker() else "cart"
            action_names = FIRST_PLACE_ACTION_SCHEMA[entity]
            existing_legal = first_place_unit_legal_mask(snapshot, unit_snapshot)
            additional_allow = np.ones_like(existing_legal)
            for index, action_name in enumerate(action_names):
                if not existing_legal[index]:
                    continue
                if action_name.startswith("move_"):
                    dx, dy = self._direction_delta(action_name[-1])
                    destination = (unit.pos.x + dx, unit.pos.y + dy)
                    cell = game.map.get_cell(*destination)
                    if not cell.is_city_tile() and destination in reserved_destinations:
                        additional_allow[index] = False
                elif action_name.startswith("transfer_"):
                    _, _, direction = action_name.split("_")
                    if not self._transfer_targets(game, unit, direction, reserved_capacity):
                        additional_allow[index] = False
            legal = monotonically_tighten_legal_mask(existing_legal, additional_allow)
            y, x = unit.pos.y + y_offset, unit.pos.x + x_offset
            distribution = _distribution(output[entity][0, :, y, x], legal)
            action_tensor = self._choose_action(distribution).to(distribution.logits.device)
            action_index = int(action_tensor)
            action_name = action_names[action_index]
            candidate = _Candidate("stay", float(distribution.logits[action_index]))
            destination = None
            transfer_target = None
            transfer_amount = 0
            if action_name.startswith("move_"):
                direction = action_name[-1]
                dx, dy = self._direction_delta(direction)
                destination = (unit.pos.x + dx, unit.pos.y + dy)
                candidate = _Candidate("move", float(distribution.logits[action_index]), direction)
                if not game.map.get_cell(*destination).is_city_tile():
                    reserved_destinations.add(destination)
            elif action_name.startswith("transfer_"):
                _, resource, direction = action_name.split("_")
                targets = self._transfer_targets(game, unit, direction, reserved_capacity)
                targets.sort(
                    key=lambda target: (
                        target.get_cargo_space_left() - reserved_capacity.get(target.id, 0),
                        int(target.is_cart()),
                        target.id,
                    ),
                    reverse=True,
                )
                if targets:
                    transfer_target = targets[0]
                    available = transfer_target.get_cargo_space_left() - reserved_capacity.get(transfer_target.id, 0)
                    transfer_amount = min(unit.cargo[resource], available)
                    reserved_capacity[transfer_target.id] = (
                        reserved_capacity.get(transfer_target.id, 0) + transfer_amount
                    )
                    candidate = _Candidate("transfer", float(distribution.logits[action_index]), direction, resource)
            elif action_name in {"build_city", "pillage"}:
                candidate = _Candidate(action_name, float(distribution.logits[action_index]))
            choice = _Choice(unit, candidate, destination, transfer_target, transfer_amount)
            choices.append(choice)
            decision = ActionDecision(
                entity=entity,
                position=(y, x),
                action=action_index,
                legal_mask=torch.from_numpy(legal),
                old_log_prob=float(distribution.log_prob(action_tensor)),
                entropy=float(distribution.entropy()),
                identity=unit.id,
            )
            decisions.append(decision)
            decision_by_unit[unit.id] = decision

        choices = self._remove_blocked_moves(game, choices)
        for choice in choices:
            if choice.candidate.kind != "stay":
                continue
            decision = decision_by_unit[choice.unit.id]
            if decision.action == 0:
                continue
            logits = output[decision.entity][0, :, decision.position[0], decision.position[1]]
            distribution = _distribution(logits, decision.legal_mask)
            action = torch.zeros((), device=logits.device, dtype=torch.long)
            decision.action = 0
            decision.old_log_prob = float(distribution.log_prob(action))
        return choices, decisions

    @staticmethod
    def _direction_delta(direction: str) -> tuple[int, int]:
        return {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}[direction]

    def _sample_cities(
        self,
        game: object,
        team: int,
        snapshot: object,
        output: dict[str, Tensor],
        x_offset: int,
        y_offset: int,
    ) -> tuple[list[object], list[ActionDecision]]:
        available_units = sum(len(city.city_cells) for city in game.cities.values() if city.team == team) - len(
            game.state["teamStates"][team]["units"]
        )
        available_research = max(0, 200 - snapshot.research_points[team])
        base_legal = first_place_city_legal_mask(snapshot, team)
        actions = []
        decisions = []
        tiles = sorted(
            (
                cell.city_tile
                for city in game.cities.values()
                if city.team == team
                for cell in city.city_cells
                if cell.city_tile.can_act()
            ),
            key=lambda tile: tile.get_tile_id(),
        )
        for tile in tiles:
            additional_allow = np.ones_like(base_legal)
            if available_units <= 0:
                additional_allow[1:3] = False
            if available_research <= 0:
                additional_allow[3] = False
            legal = monotonically_tighten_legal_mask(base_legal, additional_allow)
            y, x = tile.pos.y + y_offset, tile.pos.x + x_offset
            distribution = _distribution(output["city_tile"][0, :, y, x], legal)
            action_tensor = self._choose_action(distribution).to(distribution.logits.device)
            action_index = int(action_tensor)
            action_name = CITY_ACTIONS[action_index]
            if action_name in {"build_worker", "build_cart"}:
                action_class = SpawnWorkerAction if action_name == "build_worker" else SpawnCartAction
                actions.append(action_class(team, None, tile.pos.x, tile.pos.y))
                available_units -= 1
            elif action_name == "research":
                actions.append(ResearchAction(team, tile.pos.x, tile.pos.y, None))
                available_research -= 1
            decisions.append(
                ActionDecision(
                    entity="city_tile",
                    position=(y, x),
                    action=action_index,
                    legal_mask=torch.from_numpy(legal),
                    old_log_prob=float(distribution.log_prob(action_tensor)),
                    entropy=float(distribution.entropy()),
                    identity=tile.get_tile_id(),
                )
            )
        return actions, decisions

    def process_turn(self, game: object, team: int) -> list[object]:
        snapshot = snapshot_from_game(game)
        observation_cpu = torch.from_numpy(encode_snapshot(snapshot, team))
        if self.inference_backend is None:
            observation = observation_cpu[None].to(self.device)
            with torch.inference_mode():
                output, value = self.actor_critic(observation)
        else:
            output, value = self.inference_backend(observation_cpu)
        x_offset, y_offset = snapshot.padding
        unit_choices, unit_decisions = self._sample_units(
            game,
            team,
            snapshot,
            output,
            x_offset,
            y_offset,
        )
        city_actions, city_decisions = self._sample_cities(
            game,
            team,
            snapshot,
            output,
            x_offset,
            y_offset,
        )
        if self.record_trajectory:
            self.records.append(
                TurnRecord(
                    observation=observation_cpu,
                    metrics=metrics_from_game(game, team),
                    decisions=unit_decisions + city_decisions,
                    value=float(value.reshape(-1)[0]),
                )
            )
        return [*self._choices_to_actions(unit_choices, team), *city_actions]


def deterministic_outcome(game: object, team: int) -> float:
    city_tiles = {0: 0, 1: 0}
    for city in game.cities.values():
        city_tiles[city.team] += len(city.city_cells)
    scores = {
        candidate: (
            city_tiles[candidate],
            len(game.get_teams_units(candidate)),
            float(game.stats["teamStats"][candidate]["fuelGenerated"]),
        )
        for candidate in (0, 1)
    }
    if scores[0] == scores[1]:
        return 0.0
    winner = 0 if scores[0] > scores[1] else 1
    return 1.0 if winner == team else -1.0
