from __future__ import annotations

# ruff: noqa: C901, PLR0912, PLR0913
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as nn_functional

from luxai2021.env.agent import Agent
from luxai2021.game.actions import (
    MoveAction,
    PillageAction,
    ResearchAction,
    SpawnCartAction,
    SpawnCityAction,
    SpawnWorkerAction,
    TransferAction,
)
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.imitation.actions import (
    CART_ACTIONS,
    CITY_ACTIONS,
    DIRECTIONS,
    FIRST_PLACE_ACTION_SCHEMA,
    RESOURCES,
    WORKER_ACTIONS,
    first_place_action_remap,
)
from luxai2021.imitation.first_place import first_place_city_legal_mask, first_place_unit_legal_mask
from luxai2021.imitation.masking import (
    LEGAL_MASK_SUFFIX,
    apply_legal_action_mask,
    city_legal_mask,
    unit_legal_masks,
)
from luxai2021.imitation.model import POLICY_SCHEMA_FIRST_PLACE_FLAT, load_bc_checkpoint
from luxai2021.imitation.schema import (
    CYCLE_LENGTH,
    FEATURE_INDEX,
    BoardSnapshot,
    UnitSnapshot,
    encode_snapshot,
    snapshot_from_game,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_DIRECTION_DELTAS = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}
_MAX_RESEARCH = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]


@dataclass(frozen=True)
class _Candidate:
    kind: str
    score: float
    direction: str | None = None
    resource: str | None = None


@dataclass
class _Choice:
    unit: object
    candidate: _Candidate
    destination: tuple[int, int] | None = None
    transfer_target: object | None = None
    transfer_amount: int = 0


class BehaviorCloningAgent(Agent):
    def __init__(self, checkpoint_path: str, device: str = "auto", tta: str = "auto") -> None:
        super().__init__()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.checkpoint = load_bc_checkpoint(checkpoint_path, str(self.device))
        self.model.eval()
        if tta == "auto":
            tta = str(self.checkpoint.get("inference_augmentation", "none"))
        if tta not in {"none", "rot180"}:
            message = f"Unsupported inference augmentation: {tta}"
            raise ValueError(message)
        self.tta = tta

    def _restore_rot180(self, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        restored = {}
        flat_policy = self.model.config.policy_schema == POLICY_SCHEMA_FIRST_PLACE_FLAT
        directional_heads = {
            "worker_move",
            "worker_transfer_dir",
            "cart_move",
            "cart_transfer_dir",
        }
        for name, source_logits in output.items():
            restored_logits = torch.rot90(source_logits, 2, dims=(-2, -1))
            if flat_policy:
                remap = first_place_action_remap(name, 2)
                restored_logits = restored_logits[:, [remap[index] for index in range(len(remap))]]
            elif name in directional_heads:
                restored_logits = restored_logits[:, (2, 3, 0, 1)]
            restored[name] = restored_logits
        return restored

    def _predict(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.tta == "none":
            return self.model(observation)
        batch_size = observation.shape[0]
        rotated_observation = torch.rot90(observation, 2, dims=(-2, -1)).clone()
        rotated_observation[:, FEATURE_INDEX["x_coordinate"]].neg_()
        rotated_observation[:, FEATURE_INDEX["y_coordinate"]].neg_()
        augmented = torch.cat((observation, rotated_observation), dim=0)
        paired_output = self.model(augmented)
        rotated_output = {name: logits[batch_size:] for name, logits in paired_output.items()}
        restored = self._restore_rot180(rotated_output)
        return {name: (logits[:batch_size] + restored[name]) * 0.5 for name, logits in paired_output.items()}

    @staticmethod
    def _cell_logits(
        output: dict[str, torch.Tensor],
        name: str,
        x: int,
        y: int,
        legal_mask: object,
    ) -> torch.Tensor:
        logits = output[name][0, :, y, x][None]
        mask = torch.as_tensor(legal_mask, device=logits.device, dtype=torch.bool)[None]
        return nn_functional.log_softmax(apply_legal_action_mask(logits, mask)[0], dim=0)

    @staticmethod
    def _friendly_units_at(game: object, team: int, x: int, y: int) -> list[object]:
        return [unit for unit in game.map.get_cell(x, y).units.values() if unit.team == team]

    def _transfer_targets(
        self,
        game: object,
        unit: object,
        direction: str,
        reserved_capacity: dict[str, int],
    ) -> list[object]:
        dx, dy = _DIRECTION_DELTAS[direction]
        x, y = unit.pos.x + dx, unit.pos.y + dy
        if x < 0 or y < 0 or x >= game.map.width or y >= game.map.height:
            return []
        targets = []
        for target in self._friendly_units_at(game, unit.team, x, y):
            available = target.get_cargo_space_left() - reserved_capacity.get(target.id, 0)
            if target.id != unit.id and available > 0:
                targets.append(target)
        return targets

    def _unit_candidates(
        self,
        game: object,
        unit: object,
        snapshot: BoardSnapshot,
        unit_snapshot: UnitSnapshot,
        output: dict[str, torch.Tensor],
        x: int,
        y: int,
    ) -> list[_Candidate]:
        if self.model.config.policy_schema == POLICY_SCHEMA_FIRST_PLACE_FLAT:
            return self._flat_unit_candidates(game, unit, snapshot, unit_snapshot, output, x, y)
        prefix = "worker" if unit.is_worker() else "cart"
        action_names = WORKER_ACTIONS if unit.is_worker() else CART_ACTIONS
        legal_masks = unit_legal_masks(snapshot, unit_snapshot)
        type_scores = self._cell_logits(
            output,
            f"{prefix}_type",
            x,
            y,
            legal_masks[f"{prefix}_type{LEGAL_MASK_SUFFIX}"],
        )
        move_scores = self._cell_logits(
            output,
            f"{prefix}_move",
            x,
            y,
            legal_masks[f"{prefix}_move{LEGAL_MASK_SUFFIX}"],
        )
        transfer_scores = self._cell_logits(
            output,
            f"{prefix}_transfer_dir",
            x,
            y,
            legal_masks[f"{prefix}_transfer_dir{LEGAL_MASK_SUFFIX}"],
        )
        resource_scores = self._cell_logits(
            output,
            f"{prefix}_resource",
            x,
            y,
            legal_masks[f"{prefix}_resource{LEGAL_MASK_SUFFIX}"],
        )
        type_mask = legal_masks[f"{prefix}_type{LEGAL_MASK_SUFFIX}"]
        move_mask = legal_masks[f"{prefix}_move{LEGAL_MASK_SUFFIX}"]
        transfer_mask = legal_masks[f"{prefix}_transfer_dir{LEGAL_MASK_SUFFIX}"]
        resource_mask = legal_masks[f"{prefix}_resource{LEGAL_MASK_SUFFIX}"]
        candidates = [_Candidate("stay", float(type_scores[action_names.index("stay")]))]

        for direction_index, direction in enumerate(DIRECTIONS):
            if move_mask[direction_index]:
                candidates.append(
                    _Candidate(
                        "move",
                        float(type_scores[action_names.index("move")] + move_scores[direction_index]),
                        direction,
                    )
                )
        if unit.is_worker() and type_mask[action_names.index("build_city")]:
            candidates.append(_Candidate("build_city", float(type_scores[action_names.index("build_city")])))
        if unit.is_cart() and type_mask[action_names.index("pillage")]:
            candidates.append(_Candidate("pillage", float(type_scores[action_names.index("pillage")])))

        for direction_index, direction in enumerate(DIRECTIONS):
            if not transfer_mask[direction_index]:
                continue
            targets = self._transfer_targets(game, unit, direction, {})
            if not targets:
                continue
            for resource_index, resource in enumerate(RESOURCES):
                if not resource_mask[resource_index]:
                    continue
                score = (
                    type_scores[action_names.index("transfer")]
                    + transfer_scores[direction_index]
                    + resource_scores[resource_index]
                )
                candidates.append(_Candidate("transfer", float(score), direction, resource))
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _flat_unit_candidates(
        self,
        game: object,
        unit: object,
        snapshot: BoardSnapshot,
        unit_snapshot: UnitSnapshot,
        output: dict[str, torch.Tensor],
        x: int,
        y: int,
    ) -> list[_Candidate]:
        entity = "worker" if unit.is_worker() else "cart"
        action_names = FIRST_PLACE_ACTION_SCHEMA[entity]
        legal_mask = first_place_unit_legal_mask(snapshot, unit_snapshot)
        scores = self._cell_logits(output, entity, x, y, legal_mask)
        candidates = [_Candidate("stay", float(scores[0]))]
        for action_index, action_name in enumerate(action_names[1:], start=1):
            if not legal_mask[action_index]:
                continue
            if action_name.startswith("move_"):
                candidates.append(_Candidate("move", float(scores[action_index]), action_name[-1]))
            elif action_name.startswith("transfer_"):
                _, resource, direction = action_name.split("_")
                if self._transfer_targets(game, unit, direction, {}):
                    candidates.append(_Candidate("transfer", float(scores[action_index]), direction, resource))
            elif action_name == "pillage":
                candidates.append(_Candidate("pillage", float(scores[action_index])))
            elif action_name == "build_city":
                candidates.append(_Candidate("build_city", float(scores[action_index])))
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _choose_units(
        self,
        game: object,
        team: int,
        snapshot: BoardSnapshot,
        output: dict[str, torch.Tensor],
        x_offset: int,
        y_offset: int,
    ) -> list[_Choice]:
        entries = []
        for unit in game.state["teamStates"][team]["units"].values():
            if not unit.can_act():
                continue
            candidates = self._unit_candidates(
                game,
                unit,
                snapshot,
                snapshot.units[unit.id],
                output,
                unit.pos.x + x_offset,
                unit.pos.y + y_offset,
            )
            margin = candidates[0].score - candidates[1].score if len(candidates) > 1 else float("inf")
            entries.append((margin, unit.id, unit, candidates))
        entries.sort(key=lambda item: (-item[0], item[1]))

        reserved_destinations: set[tuple[int, int]] = set()
        reserved_capacity: dict[str, int] = {}
        choices = []
        for _, _, unit, candidates in entries:
            selected = _Candidate("stay", 0)
            destination = None
            transfer_target = None
            transfer_amount = 0
            for candidate in candidates:
                if candidate.kind == "move":
                    dx, dy = _DIRECTION_DELTAS[candidate.direction]
                    proposed = (unit.pos.x + dx, unit.pos.y + dy)
                    cell = game.map.get_cell(*proposed)
                    if not cell.is_city_tile() and proposed in reserved_destinations:
                        continue
                    selected = candidate
                    destination = proposed
                    if not cell.is_city_tile():
                        reserved_destinations.add(proposed)
                    break
                if candidate.kind == "transfer":
                    targets = self._transfer_targets(game, unit, candidate.direction, reserved_capacity)
                    if not targets:
                        continue
                    targets.sort(
                        key=lambda target: (
                            target.get_cargo_space_left() - reserved_capacity.get(target.id, 0),
                            int(target.is_cart()),
                            target.id,
                        ),
                        reverse=True,
                    )
                    target = targets[0]
                    available = target.get_cargo_space_left() - reserved_capacity.get(target.id, 0)
                    amount = min(unit.cargo[candidate.resource], available)
                    if amount <= 0:
                        continue
                    selected = candidate
                    transfer_target = target
                    transfer_amount = amount
                    reserved_capacity[target.id] = reserved_capacity.get(target.id, 0) + amount
                    break
                selected = candidate
                break
            choices.append(
                _Choice(
                    unit=unit,
                    candidate=selected,
                    destination=destination,
                    transfer_target=transfer_target,
                    transfer_amount=transfer_amount,
                )
            )
        return self._remove_blocked_moves(game, choices)

    @staticmethod
    def _remove_blocked_moves(game: object, choices: list[_Choice]) -> list[_Choice]:
        by_unit = {choice.unit.id: choice for choice in choices}
        changed = True
        while changed:
            changed = False
            moving = {
                choice.unit.id
                for choice in choices
                if choice.candidate.kind == "move" and choice.destination is not None
            }
            for choice in choices:
                if choice.candidate.kind != "move" or choice.destination is None:
                    continue
                destination_cell = game.map.get_cell(*choice.destination)
                if destination_cell.is_city_tile():
                    continue
                blockers = [
                    unit
                    for unit in destination_cell.units.values()
                    if unit.team == choice.unit.team and unit.id != choice.unit.id
                ]
                blocked = any(blocker.id not in moving for blocker in blockers)
                swapped = any(
                    blocker.id in by_unit and by_unit[blocker.id].destination == (choice.unit.pos.x, choice.unit.pos.y)
                    for blocker in blockers
                )
                if blocked or swapped:
                    choice.candidate = _Candidate("stay", choice.candidate.score)
                    choice.destination = None
                    changed = True
        return choices

    @staticmethod
    def _choices_to_actions(choices: Sequence[_Choice], team: int) -> list[object]:
        actions = []
        for choice in choices:
            candidate = choice.candidate
            unit = choice.unit
            if candidate.kind == "move":
                actions.append(MoveAction(team, unit.id, candidate.direction))
            elif candidate.kind == "build_city":
                actions.append(SpawnCityAction(team, unit.id))
            elif candidate.kind == "pillage":
                actions.append(PillageAction(team, unit.id))
            elif candidate.kind == "transfer" and choice.transfer_target is not None:
                actions.append(
                    TransferAction(
                        team,
                        unit.id,
                        choice.transfer_target.id,
                        candidate.resource,
                        choice.transfer_amount,
                    )
                )
        return actions

    def _city_actions(
        self,
        game: object,
        team: int,
        snapshot: BoardSnapshot,
        output: dict[str, torch.Tensor],
        x_offset: int,
        y_offset: int,
    ) -> list[object]:
        available_units = sum(len(city.city_cells) for city in game.cities.values() if city.team == team) - len(
            game.state["teamStates"][team]["units"]
        )
        available_research = max(0, _MAX_RESEARCH - snapshot.research_points[team])
        flat_policy = self.model.config.policy_schema == POLICY_SCHEMA_FIRST_PLACE_FLAT
        legal_mask = first_place_city_legal_mask(snapshot, team) if flat_policy else city_legal_mask(snapshot, team)
        output_name = "city_tile" if flat_policy else "city"
        entries = []
        for city in game.cities.values():
            if city.team != team:
                continue
            for cell in city.city_cells:
                tile = cell.city_tile
                if not tile.can_act():
                    continue
                scores = self._cell_logits(
                    output,
                    output_name,
                    tile.pos.x + x_offset,
                    tile.pos.y + y_offset,
                    legal_mask,
                )
                order = torch.argsort(scores, descending=True).tolist()
                margin = float(scores[order[0]] - scores[order[1]])
                entries.append((margin, tile.get_tile_id(), tile, order))
        entries.sort(key=lambda item: (-item[0], item[1]))

        actions = []
        for _, _, tile, order in entries:
            for action_index in order:
                action_name = CITY_ACTIONS[action_index]
                if action_name == "no_action":
                    if flat_policy and snapshot.turn >= CYCLE_LENGTH and available_research > 0:
                        continue
                    break
                if action_name in {"build_worker", "build_cart"}:
                    if flat_policy and action_name == "build_cart":
                        continue
                    if available_units <= 0:
                        continue
                    action_class = SpawnWorkerAction if action_name == "build_worker" else SpawnCartAction
                    actions.append(action_class(team, None, tile.pos.x, tile.pos.y))
                    available_units -= 1
                    break
                if action_name == "research":
                    if available_research <= 0:
                        continue
                    actions.append(ResearchAction(team, tile.pos.x, tile.pos.y, None))
                    available_research -= 1
                    break
        return actions

    def process_turn(self, game: object, team: int) -> list[object]:
        snapshot = snapshot_from_game(game)
        observation = torch.from_numpy(encode_snapshot(snapshot, team))[None].to(self.device)
        with torch.inference_mode():
            output = self._predict(observation)
        x_offset, y_offset = snapshot.padding
        unit_choices = self._choose_units(game, team, snapshot, output, x_offset, y_offset)
        actions = self._choices_to_actions(unit_choices, team)
        actions.extend(self._city_actions(game, team, snapshot, output, x_offset, y_offset))
        return actions
