from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.imitation.actions import (
    ACTION_SCHEMA,
    CART_ACTIONS,
    CITY_ACTIONS,
    DIRECTIONS,
    RESOURCES,
    TARGET_NAMES,
    WORKER_ACTIONS,
)
from luxai2021.imitation.schema import BOARD_SIZE, BoardSnapshot, UnitSnapshot

MAX_ENTITIES = BOARD_SIZE * BOARD_SIZE
LEGAL_MASK_SCHEMA_VERSION = 1
LEGAL_MASK_SUFFIX = "_legal_mask"
_DIRECTION_DELTAS = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}
_CITY_BUILD_COST = GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]
_MAX_RESEARCH = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
_CAPACITY = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]


def _units_by_position(snapshot: BoardSnapshot) -> dict[tuple[int, int], list[UnitSnapshot]]:
    result: dict[tuple[int, int], list[UnitSnapshot]] = {}
    for unit in snapshot.units.values():
        result.setdefault((unit.x, unit.y), []).append(unit)
    return result


def _city_teams(snapshot: BoardSnapshot) -> dict[tuple[int, int], int]:
    return {(tile.x, tile.y): tile.team for tile in snapshot.city_tiles}


def _cargo_space_left(unit: UnitSnapshot) -> int:
    kind = "WORKER" if unit.unit_type == 0 else "CART"
    return _CAPACITY[kind] - sum(unit.cargo.values())


def _unit_legal_masks(
    snapshot: BoardSnapshot,
    unit: UnitSnapshot,
    units_at: dict[tuple[int, int], list[UnitSnapshot]],
    city_teams: dict[tuple[int, int], int],
) -> dict[str, np.ndarray]:
    prefix = "worker" if unit.unit_type == 0 else "cart"
    action_names = WORKER_ACTIONS if prefix == "worker" else CART_ACTIONS
    move_mask = np.zeros(len(DIRECTIONS), dtype=np.bool_)
    transfer_dir_mask = np.zeros(len(DIRECTIONS), dtype=np.bool_)
    resource_mask = np.zeros(len(RESOURCES), dtype=np.bool_)
    transfer_pairs = np.zeros((len(DIRECTIONS), len(RESOURCES)), dtype=np.bool_)

    for direction_index, direction in enumerate(DIRECTIONS):
        dx, dy = _DIRECTION_DELTAS[direction]
        destination = (unit.x + dx, unit.y + dy)
        in_bounds = 0 <= destination[0] < snapshot.width and 0 <= destination[1] < snapshot.height
        if not in_bounds:
            continue

        destination_city_team = city_teams.get(destination)
        destination_units = units_at.get(destination, ())
        move_mask[direction_index] = destination_city_team in {None, unit.team} and not any(
            other.cooldown > 0 for other in destination_units
        )

        valid_targets = [
            target
            for target in destination_units
            if target.team == unit.team and target.unit_id != unit.unit_id and _cargo_space_left(target) > 0
        ]
        if valid_targets:
            for resource_index, resource in enumerate(RESOURCES):
                transfer_pairs[direction_index, resource_index] = unit.cargo[resource] > 0

    transfer_dir_mask[:] = transfer_pairs.any(axis=1)
    resource_mask[:] = transfer_pairs.any(axis=0)
    type_mask = np.zeros(len(action_names), dtype=np.bool_)
    type_mask[action_names.index("stay")] = True
    type_mask[action_names.index("move")] = bool(move_mask.any())
    type_mask[action_names.index("transfer")] = bool(transfer_pairs.any())

    if prefix == "worker":
        on_city = (unit.x, unit.y) in city_teams
        has_resource = (unit.x, unit.y) in snapshot.resources
        has_build_cargo = sum(unit.cargo.values()) >= _CITY_BUILD_COST
        type_mask[action_names.index("build_city")] = has_build_cargo and not has_resource and not on_city
    else:
        # Season 1 pillage is a worker action. The existing BC schema keeps this
        # cart output for checkpoint compatibility, but it must never be chosen.
        type_mask[action_names.index("pillage")] = False

    return {
        f"{prefix}_type{LEGAL_MASK_SUFFIX}": type_mask,
        f"{prefix}_move{LEGAL_MASK_SUFFIX}": move_mask,
        f"{prefix}_transfer_dir{LEGAL_MASK_SUFFIX}": transfer_dir_mask,
        f"{prefix}_resource{LEGAL_MASK_SUFFIX}": resource_mask,
    }


def unit_legal_masks(snapshot: BoardSnapshot, unit: UnitSnapshot) -> dict[str, np.ndarray]:
    """Return factorized viable-action masks for one actionable unit.

    The conditions mirror the action masking in IsaiahPressman's Lux 2021 agent:
    off-board/enemy-city/cooldown-blocked moves and impossible transfers are
    removed before softmax. Build, pillage and city constraints are represented
    in the action-type mask.
    """
    return _unit_legal_masks(snapshot, unit, _units_by_position(snapshot), _city_teams(snapshot))


def city_legal_mask(snapshot: BoardSnapshot, team: int) -> np.ndarray:
    mask = np.ones(len(CITY_ACTIONS), dtype=np.bool_)
    own_units = sum(unit.team == team for unit in snapshot.units.values())
    own_city_tiles = sum(tile.team == team for tile in snapshot.city_tiles)
    can_build = own_units < own_city_tiles
    mask[CITY_ACTIONS.index("build_worker")] = can_build
    mask[CITY_ACTIONS.index("build_cart")] = can_build
    mask[CITY_ACTIONS.index("research")] = snapshot.research_points[team] < _MAX_RESEARCH
    return mask


def build_legal_masks(snapshot: BoardSnapshot, team: int) -> dict[str, np.ndarray]:
    masks = {
        f"{name}{LEGAL_MASK_SUFFIX}": np.zeros((MAX_ENTITIES, len(ACTION_SCHEMA[name])), dtype=np.bool_)
        for name in TARGET_NAMES
    }
    units_at = _units_by_position(snapshot)
    city_teams = _city_teams(snapshot)
    entity_counts = {"worker": 0, "cart": 0}
    for unit in snapshot.units.values():
        if unit.team != team or not unit.can_act:
            continue
        prefix = "worker" if unit.unit_type == 0 else "cart"
        entity_index = entity_counts[prefix]
        entity_counts[prefix] += 1
        for name, mask in _unit_legal_masks(snapshot, unit, units_at, city_teams).items():
            masks[name][entity_index] = mask

    city_mask = city_legal_mask(snapshot, team)
    city_index = 0
    for tile in snapshot.city_tiles:
        if tile.team == team and tile.can_act:
            masks[f"city{LEGAL_MASK_SUFFIX}"][city_index] = city_mask
            city_index += 1
    return masks


def apply_legal_action_mask(
    logits: Tensor,
    legal_mask: Tensor,
    *,
    action_dim: int = -1,
) -> Tensor:
    if logits.shape != legal_mask.shape:
        msg = f"Legal-mask shape mismatch: logits={tuple(logits.shape)}, mask={tuple(legal_mask.shape)}"
        raise ValueError(msg)
    legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
    logits = logits.movedim(action_dim, -1)
    legal_mask = legal_mask.movedim(action_dim, -1)
    # Padded/non-entity rows have no legal actions. Make class zero a harmless
    # fallback so log_softmax remains finite; their targets use IGNORE_INDEX.
    no_legal_action = ~legal_mask.any(dim=-1)
    legal_mask = legal_mask.clone()
    legal_mask[..., 0] |= no_legal_action
    masked = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    return masked.movedim(-1, action_dim)


def sanitize_targets(
    targets: dict[str, np.ndarray],
    legal_masks: dict[str, np.ndarray],
) -> None:
    """Convert replay commands rejected by the viability mask to effective no-op."""
    for prefix in ("worker", "cart"):
        action_names = WORKER_ACTIONS if prefix == "worker" else CART_ACTIONS
        type_name = f"{prefix}_type"
        type_target = targets[type_name]
        type_mask = legal_masks[f"{type_name}{LEGAL_MASK_SUFFIX}"]
        for entity_index in np.flatnonzero(type_target >= 0):
            action_type = int(type_target[entity_index])
            invalid = not type_mask[entity_index, action_type]
            if action_type == action_names.index("move"):
                direction = int(targets[f"{prefix}_move"][entity_index])
                invalid |= (
                    direction < 0 or not legal_masks[f"{prefix}_move{LEGAL_MASK_SUFFIX}"][entity_index, direction]
                )
            elif action_type == action_names.index("transfer"):
                direction = int(targets[f"{prefix}_transfer_dir"][entity_index])
                resource = int(targets[f"{prefix}_resource"][entity_index])
                invalid |= direction < 0 or resource < 0
                if direction >= 0:
                    invalid |= not legal_masks[f"{prefix}_transfer_dir{LEGAL_MASK_SUFFIX}"][entity_index, direction]
                if resource >= 0:
                    invalid |= not legal_masks[f"{prefix}_resource{LEGAL_MASK_SUFFIX}"][entity_index, resource]
            if invalid:
                type_target[entity_index] = 0
                targets[f"{prefix}_move"][entity_index] = -100
                targets[f"{prefix}_transfer_dir"][entity_index] = -100
                targets[f"{prefix}_resource"][entity_index] = -100

    city_target = targets["city"]
    city_mask = legal_masks[f"city{LEGAL_MASK_SUFFIX}"]
    for entity_index in np.flatnonzero(city_target >= 0):
        action = int(city_target[entity_index])
        if not city_mask[entity_index, action]:
            city_target[entity_index] = 0
