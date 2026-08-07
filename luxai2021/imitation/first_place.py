"""Dependency-free adapter for Isaiah Pressman's MIT-licensed Lux AI 2021 teacher."""

from __future__ import annotations

# ruff: noqa: C901, PLR0912, PLR0915, PLR2004
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.imitation.actions import (
    FIRST_PLACE_ACTION_SCHEMA,
    FIRST_PLACE_CART_ACTIONS,
    FIRST_PLACE_CITY_ACTIONS,
    FIRST_PLACE_WORKER_ACTIONS,
    RESOURCES,
    first_place_action_remap,
)
from luxai2021.imitation.data import IGNORE_INDEX, MAX_ENTITIES
from luxai2021.imitation.schema import BOARD_SIZE, FEATURE_INDEX, BoardSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

FIRST_PLACE_TEACHER_SHA256 = "40248f0fbc9b8e1e1b1f7cc6fc674c041d8dac43b964ae45bd976d927cdffd22"
FIRST_PLACE_TEACHER_PARAMETER_COUNT = 19_868_201
FIRST_PLACE_UPSTREAM_COMMIT = "973a6c6c63211b6c7ab6fdf50e026e458d1f6e4e"
FIRST_PLACE_POLICY_PREFIXES = ("base_model.", "actor_base.", "actor.")

_MAX_RESOURCE = {"wood": 500.0, "coal": 425.0, "uranium": 350.0}
_MAX_FUEL = 30 * 10 * 9
_CYCLE_LENGTH = 40
_MAX_TURNS = 360
_MAP_SIZES = (12, 16, 24, 32)
_PLAYER_CATEGORICAL = (
    "worker",
    "cart",
    "worker_cargo_full",
    "cart_cargo_full",
    "city_tile",
    "researched_coal",
    "researched_uranium",
)
# Gym 0.19's Dict space sorted plain-dict keys. These orders define the
# checkpoint's learned merger-channel layout and must not be changed.
_EMBEDDING_INPUT_ORDER = (
    "board_size",
    "cart",
    "cart_cargo_full",
    "city_tile",
    "day_night_cycle",
    "night",
    "phase",
    "researched_coal",
    "researched_uranium",
    "worker",
    "worker_cargo_full",
)
_CONTINUOUS_INPUT_ORDER = (
    "cart_cargo_coal",
    "cart_cargo_uranium",
    "cart_cargo_wood",
    "cart_cooldown",
    "city_tile_cooldown",
    "city_tile_cost",
    "city_tile_fuel",
    "coal",
    "dist_from_center_x",
    "dist_from_center_y",
    "research_points",
    "road_level",
    "turn",
    "uranium",
    "wood",
    "worker_cargo_coal",
    "worker_cargo_uranium",
    "worker_cargo_wood",
    "worker_cooldown",
)
_CONTINUOUS = (
    "worker_cooldown",
    "cart_cooldown",
    "worker_cargo_wood",
    "worker_cargo_coal",
    "worker_cargo_uranium",
    "cart_cargo_wood",
    "cart_cargo_coal",
    "cart_cargo_uranium",
    "city_tile_fuel",
    "city_tile_cost",
    "city_tile_cooldown",
    "road_level",
    "wood",
    "coal",
    "uranium",
    "dist_from_center_x",
    "dist_from_center_y",
    "research_points",
    "turn",
)


@dataclass(frozen=True)
class FirstPlaceEncodedObservation:
    values: dict[str, Tensor]
    input_mask: Tensor


def _empty_spatial(players: int, *, categorical: bool = False) -> np.ndarray:
    dtype = np.int64 if categorical else np.float32
    return np.zeros((1, players, BOARD_SIZE, BOARD_SIZE), dtype=dtype)


def encode_first_place_observation(snapshot: BoardSnapshot) -> FirstPlaceEncodedObservation:
    """Encode one state exactly in the first-place model's top-left convention."""
    values: dict[str, np.ndarray] = OrderedDict()
    for name in ("worker", "cart", "worker_cargo_full", "cart_cargo_full", "city_tile"):
        values[name] = _empty_spatial(2, categorical=True)
    values["worker_COUNT"] = _empty_spatial(2)
    values["cart_COUNT"] = _empty_spatial(2)
    for name in _CONTINUOUS[:11]:
        values[name] = _empty_spatial(2)
    for name in _CONTINUOUS[11:17]:
        values[name] = _empty_spatial(1)
    values["research_points"] = np.zeros((1, 2), dtype=np.float32)
    values["researched_coal"] = np.zeros((1, 2), dtype=np.int64)
    values["researched_uranium"] = np.zeros((1, 2), dtype=np.int64)
    values["night"] = np.zeros((1, 1), dtype=np.int64)
    values["day_night_cycle"] = np.zeros((1, 1), dtype=np.int64)
    values["phase"] = np.zeros((1, 1), dtype=np.int64)
    values["turn"] = np.zeros((1, 1), dtype=np.float32)
    values["board_size"] = np.zeros((1, 1), dtype=np.int64)

    capacities = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]
    cooldowns = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"]
    for unit in snapshot.units.values():
        x, y, team = unit.x, unit.y, unit.team
        kind = "worker" if unit.unit_type == 0 else "cart"
        capacity = capacities[kind.upper()]
        max_cooldown = cooldowns[kind.upper()] * 2.0 - 1.0
        values[kind][0, team, x, y] = 1
        values[f"{kind}_COUNT"][0, team, x, y] += 1
        values[f"{kind}_cooldown"][0, team, x, y] = unit.cooldown / max_cooldown
        values[f"{kind}_cargo_full"][0, team, x, y] = sum(unit.cargo.values()) >= capacity
        for resource in RESOURCES:
            values[f"{kind}_cargo_{resource}"][0, team, x, y] = unit.cargo[resource] / capacity

    city_tiles_by_id: dict[str, list[object]] = {}
    for tile in snapshot.city_tiles:
        city_tiles_by_id.setdefault(tile.city_id, []).append(tile)
    city_cooldown = GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]
    for city_id, tiles in city_tiles_by_id.items():
        team, fuel, upkeep = snapshot.cities[city_id]
        tile_count = len(tiles)
        for tile in tiles:
            values["city_tile"][0, team, tile.x, tile.y] = 1
            values["city_tile_fuel"][0, team, tile.x, tile.y] = fuel / _MAX_FUEL / tile_count
            values["city_tile_cost"][0, team, tile.x, tile.y] = upkeep / 23.0 / tile_count
            values["city_tile_cooldown"][0, team, tile.x, tile.y] = tile.cooldown / city_cooldown

    max_road = GAME_CONSTANTS["PARAMETERS"]["MAX_ROAD"]
    for (x, y), road in snapshot.roads.items():
        values["road_level"][0, 0, x, y] = road / max_road
    for (x, y), (resource, amount) in snapshot.resources.items():
        values[resource][0, 0, x, y] = amount / _MAX_RESOURCE[resource]

    x_distance = np.abs(1 - np.linspace(0, 2, snapshot.height, dtype=np.float32))[None, :]
    y_distance = np.abs(1 - np.linspace(0, 2, snapshot.width, dtype=np.float32))[:, None]
    values["dist_from_center_x"][0, 0, : snapshot.width, : snapshot.height] = np.repeat(
        x_distance,
        snapshot.width,
        axis=0,
    )
    values["dist_from_center_y"][0, 0, : snapshot.width, : snapshot.height] = np.repeat(
        y_distance,
        snapshot.height,
        axis=1,
    )
    for team in (0, 1):
        research = snapshot.research_points[team]
        values["research_points"][0, team] = min(research / 200.0, 1.0)
        values["researched_coal"][0, team] = research >= 50
        values["researched_uranium"][0, team] = research >= 200
    values["night"][0, 0] = snapshot.turn % _CYCLE_LENGTH >= 30
    values["day_night_cycle"][0, 0] = snapshot.turn % _CYCLE_LENGTH
    values["phase"][0, 0] = min(snapshot.turn // _CYCLE_LENGTH, 8)
    values["turn"][0, 0] = snapshot.turn / _MAX_TURNS
    values["board_size"][0, 0] = _MAP_SIZES.index(snapshot.width)
    input_mask = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)
    input_mask[:, : snapshot.width, : snapshot.height] = True
    return FirstPlaceEncodedObservation(
        values={name: torch.from_numpy(value) for name, value in values.items()},
        input_mask=torch.from_numpy(input_mask),
    )


def stack_first_place_observations(
    observations: Sequence[FirstPlaceEncodedObservation],
) -> FirstPlaceEncodedObservation:
    return FirstPlaceEncodedObservation(
        values={
            name: torch.stack([observation.values[name] for observation in observations])
            for name in observations[0].values
        },
        input_mask=torch.stack([observation.input_mask for observation in observations]),
    )


def rotate_first_place_observation_180(observation: FirstPlaceEncodedObservation) -> FirstPlaceEncodedObservation:
    values = {
        name: torch.rot90(value, 2, dims=(-2, -1)) if value.ndim == 5 else value
        for name, value in observation.values.items()
    }
    return FirstPlaceEncodedObservation(values=values, input_mask=torch.rot90(observation.input_mask, 2, (-2, -1)))


class _FirstPlaceInputLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding_sizes = {
            "worker": 3,
            "cart": 3,
            "worker_cargo_full": 3,
            "cart_cargo_full": 3,
            "city_tile": 3,
            "researched_coal": 3,
            "researched_uranium": 3,
            "night": 2,
            "day_night_cycle": 40,
            "phase": 9,
            "board_size": 4,
        }
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, 32, padding_idx=0 if name in _PLAYER_CATEGORICAL else None)
                for name, size in embedding_sizes.items()
            }
        )
        self.continuous_space_embedding = nn.Sequential(nn.Conv2d(31, 128, 1), nn.LeakyReLU())
        self.embedding_merger = nn.Sequential(nn.Conv2d(576, 128, 1), nn.LeakyReLU())
        self.merger = nn.Sequential(nn.Conv2d(256, 128, 1))

    @staticmethod
    def _relative(value: Tensor) -> Tensor:
        swapped = value.flip(2) if value.shape[2] == 2 else value
        return torch.stack((value, swapped), dim=1).flatten(0, 1)

    def _embed(self, name: str, value: Tensor, mask: Tensor) -> Tensor:
        relative = self._relative(value).squeeze(1)
        if relative.shape[1] == 2:
            relative = relative.clone()
            relative[:, 1] = torch.where(relative[:, 1] != 0, relative[:, 1] + 1, relative[:, 1])
        embedded = self.embeddings[name](relative.long())
        if embedded.ndim == 3:
            embedded = embedded.unsqueeze(-2).unsqueeze(-2)
        embedded = embedded.permute(0, 1, 4, 2, 3).flatten(1, 2)
        return embedded * mask

    def forward(self, inputs: tuple[Mapping[str, Tensor], Tensor]) -> tuple[Tensor, Tensor]:
        values, input_mask = inputs
        mask = torch.repeat_interleave(input_mask, 2, dim=0)
        embeddings: OrderedDict[str, Tensor] = OrderedDict()
        for name in _EMBEDDING_INPUT_ORDER:
            embedded = self._embed(name, values[name], mask)
            count_name = f"{name}_COUNT"
            if count_name in values:
                count = self._relative(values[count_name]).squeeze(1)
                embedded = (
                    embedded.view(embedded.shape[0], 2, 32, BOARD_SIZE, BOARD_SIZE) * count[:, :, None]
                ).flatten(1, 2)
            embeddings[name] = embedded

        continuous = []
        for name in _CONTINUOUS_INPUT_ORDER:
            relative = self._relative(values[name]).flatten(1, 2)
            if relative.ndim == 2:
                relative = relative[:, :, None, None]
            continuous.append(relative * mask)
        continuous_output = self.continuous_space_embedding(torch.cat(continuous, dim=1))
        embedding_output = self.embedding_merger(torch.cat(tuple(embeddings.values()), dim=1))
        return self.merger(torch.cat((continuous_output, embedding_output), dim=1)), mask


class _FirstPlaceSELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(8, 128, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        pooled = inputs.flatten(-2).sum(-1) / mask.flatten(-2).sum(-1).clamp_min(1)
        scale = self.fc(pooled).view(inputs.shape[0], inputs.shape[1], 1, 1)
        return inputs * scale


class _FirstPlaceResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 5, padding=2)
        self.norm1 = nn.Identity()
        self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.norm2 = nn.Identity()
        self.final_act = nn.LeakyReLU()
        self.change_n_channels = nn.Identity()
        self.squeeze_excitation = _FirstPlaceSELayer()

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        value, mask = inputs
        residual = self.act1(self.norm1(self.conv1(value) * mask))
        residual = self.squeeze_excitation(self.norm2(self.conv2(residual) * mask), mask)
        return self.final_act(residual + value) * mask, mask


class _FirstPlaceActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actors = nn.ModuleDict(
            {name: nn.Conv2d(128, len(actions), 1) for name, actions in FIRST_PLACE_ACTION_SCHEMA.items()}
        )


class FirstPlaceTeacherModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = nn.Sequential(_FirstPlaceInputLayer(), *(_FirstPlaceResidualBlock() for _ in range(24)))
        self.actor_base = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(128, 128, 1)), nn.ReLU())
        self.actor = _FirstPlaceActor()

    def forward(self, observation: FirstPlaceEncodedObservation) -> dict[str, Tensor]:
        features, _ = self.base_model((observation.values, observation.input_mask))
        features = self.actor_base(features)
        batch_size = observation.input_mask.shape[0]
        return {
            name: actor(features).reshape(batch_size, 2, len(FIRST_PLACE_ACTION_SCHEMA[name]), BOARD_SIZE, BOARD_SIZE)
            for name, actor in self.actor.actors.items()
        }


def load_first_place_teacher(path: str | Path, device: torch.device | str = "cpu") -> FirstPlaceTeacherModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    policy_state = {name: value for name, value in state.items() if name.startswith(FIRST_PLACE_POLICY_PREFIXES)}
    model = FirstPlaceTeacherModel()
    model.load_state_dict(policy_state, strict=True)
    model.eval().to(device)
    return model


def inverse_rotate_first_place_policy_180(output: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result = {}
    for entity, logits in output.items():
        spatial = torch.rot90(logits, 2, dims=(-2, -1))
        remap = first_place_action_remap(entity, 2)
        indices = torch.tensor([remap[index] for index in range(len(remap))], device=logits.device)
        result[entity] = spatial.index_select(2, indices)
    return result


def predict_first_place(
    model: FirstPlaceTeacherModel,
    snapshots: Sequence[BoardSnapshot],
    *,
    device: torch.device,
    rot180: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Tensor]:
    encoded = stack_first_place_observations([encode_first_place_observation(snapshot) for snapshot in snapshots])
    encoded = FirstPlaceEncodedObservation(
        values={name: value.to(device, non_blocking=True) for name, value in encoded.values.items()},
        input_mask=encoded.input_mask.to(device, non_blocking=True),
    )
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=autocast_enabled):
        if rot180:
            rotated = rotate_first_place_observation_180(encoded)
            combined = FirstPlaceEncodedObservation(
                values={
                    name: torch.cat((value, rotated.values[name]), dim=0) for name, value in encoded.values.items()
                },
                input_mask=torch.cat((encoded.input_mask, rotated.input_mask), dim=0),
            )
            combined_output = model(combined)
            batch_size = len(snapshots)
            output = {name: value[:batch_size] for name, value in combined_output.items()}
            rotated_output = {name: value[batch_size:] for name, value in combined_output.items()}
            restored = inverse_rotate_first_place_policy_180(rotated_output)
            output = {name: (output[name] + restored[name]) * 0.5 for name in output}
        else:
            output = model(encoded)
    return {name: value.float().cpu() for name, value in output.items()}


def first_place_unit_legal_mask(snapshot: BoardSnapshot, unit: object) -> np.ndarray:
    actions = FIRST_PLACE_WORKER_ACTIONS if unit.unit_type == 0 else FIRST_PLACE_CART_ACTIONS
    mask = np.zeros(len(actions), dtype=np.bool_)
    mask[0] = True
    units_at = snapshot.units_by_position
    city_teams = snapshot.city_teams_by_position
    capacities = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]
    for direction_index, direction in enumerate(("n", "e", "s", "w")):
        dx, dy = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}[direction]
        destination = (unit.x + dx, unit.y + dy)
        if not (0 <= destination[0] < snapshot.width and 0 <= destination[1] < snapshot.height):
            continue
        destination_units = units_at.get(destination, ())
        if city_teams.get(destination) in {None, unit.team} and not any(
            other.cooldown > 0 for other in destination_units
        ):
            mask[1 + direction_index] = True
        targets = [
            target
            for target in destination_units
            if target.team == unit.team
            and target.unit_id != unit.unit_id
            and sum(target.cargo.values()) < capacities["WORKER" if target.unit_type == 0 else "CART"]
        ]
        if targets:
            for resource_index, resource in enumerate(RESOURCES):
                if unit.cargo[resource] > 0:
                    mask[5 + resource_index * 4 + direction_index] = True
    if unit.unit_type == 0:
        on_city = (unit.x, unit.y) in city_teams
        mask[17] = snapshot.roads.get((unit.x, unit.y), 0) > 0 and not on_city
        mask[18] = (
            sum(unit.cargo.values()) >= GAME_CONSTANTS["PARAMETERS"]["CITY_BUILD_COST"]
            and (unit.x, unit.y) not in snapshot.resources
            and not on_city
        )
    return mask


def first_place_city_legal_mask(snapshot: BoardSnapshot, team: int) -> np.ndarray:
    mask = np.ones(len(FIRST_PLACE_CITY_ACTIONS), dtype=np.bool_)
    unit_count = sum(unit.team == team for unit in snapshot.units.values())
    city_count = sum(tile.team == team for tile in snapshot.city_tiles)
    mask[1:3] = unit_count < city_count
    mask[3] = snapshot.research_points[team] < 200
    return mask


def build_first_place_targets(
    snapshot: BoardSnapshot,
    team: int,
    actions: Iterable[str],
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    counts = {"worker": 0, "cart": 0, "city_tile": 0}
    unit_indices: dict[str, tuple[str, int]] = {}
    for entity, action_names in FIRST_PLACE_ACTION_SCHEMA.items():
        targets[f"{entity}_flat"] = np.full(MAX_ENTITIES, IGNORE_INDEX, dtype=np.int64)
        targets[f"{entity}_positions"] = np.full((MAX_ENTITIES, 2), -1, dtype=np.int64)
        targets[f"{entity}_legal_mask"] = np.zeros((MAX_ENTITIES, len(action_names)), dtype=np.bool_)
    for unit in snapshot.units.values():
        if unit.team != team or not unit.can_act:
            continue
        entity = "worker" if unit.unit_type == 0 else "cart"
        index = counts[entity]
        counts[entity] += 1
        unit_indices[unit.unit_id] = (entity, index)
        x, y = snapshot.padded_position(unit.x, unit.y)
        targets[f"{entity}_positions"][index] = (y, x)
        targets[f"{entity}_flat"][index] = 0
        targets[f"{entity}_legal_mask"][index] = first_place_unit_legal_mask(snapshot, unit)
    city_indices = {}
    for tile in snapshot.city_tiles:
        if tile.team != team or not tile.can_act:
            continue
        index = counts["city_tile"]
        counts["city_tile"] += 1
        city_indices[(tile.x, tile.y)] = index
        x, y = snapshot.padded_position(tile.x, tile.y)
        targets["city_tile_positions"][index] = (y, x)
        targets["city_tile_flat"][index] = 0
        targets["city_tile_legal_mask"][index] = first_place_city_legal_mask(snapshot, team)

    for command in actions:
        parts = command.split()
        if not parts:
            continue
        action = parts[0]
        if action in {"m", "bcity", "t", "p"} and parts[1] in unit_indices:
            source = snapshot.units[parts[1]]
            entity, index = unit_indices[parts[1]]
            if action == "m" and parts[2] != "c":
                target_name = f"move_{parts[2]}"
            elif action == "bcity" and entity == "worker":
                target_name = "build_city"
            elif action == "p" and entity == "worker":
                target_name = "pillage"
            elif action == "t":
                destination = snapshot.units.get(parts[2])
                if destination is None:
                    continue
                dx, dy = destination.x - source.x, destination.y - source.y
                direction = {(0, -1): "n", (1, 0): "e", (0, 1): "s", (-1, 0): "w"}.get((dx, dy))
                if direction is None:
                    continue
                target_name = f"transfer_{parts[3]}_{direction}"
            else:
                target_name = "no_action"
            targets[f"{entity}_flat"][index] = FIRST_PLACE_ACTION_SCHEMA[entity].index(target_name)
        elif action in {"bw", "bc", "r"}:
            index = city_indices.get((int(parts[1]), int(parts[2])))
            if index is not None:
                targets["city_tile_flat"][index] = {"bw": 1, "bc": 2, "r": 3}[action]

    for entity in FIRST_PLACE_ACTION_SCHEMA:
        labels = targets[f"{entity}_flat"]
        masks = targets[f"{entity}_legal_mask"]
        for index in np.flatnonzero(labels != IGNORE_INDEX):
            if not masks[index, labels[index]]:
                labels[index] = 0
    return targets


def augment_first_place_sample(
    observation: np.ndarray,
    targets: dict[str, np.ndarray],
    rotations: int,
    *,
    horizontal_flip: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    observation = np.rot90(observation, rotations, axes=(-2, -1)).copy()
    transformed = {name: value.copy() for name, value in targets.items()}
    if horizontal_flip:
        observation = observation[:, :, ::-1].copy()
    for entity in FIRST_PLACE_ACTION_SCHEMA:
        positions = transformed[f"{entity}_positions"]
        valid_positions = positions[:, 0] >= 0
        for _ in range(rotations % 4):
            old_y = positions[valid_positions, 0].copy()
            old_x = positions[valid_positions, 1].copy()
            positions[valid_positions, 0] = BOARD_SIZE - 1 - old_x
            positions[valid_positions, 1] = old_y
        if horizontal_flip:
            positions[valid_positions, 1] = BOARD_SIZE - 1 - positions[valid_positions, 1]
        remap = first_place_action_remap(entity, rotations, horizontal_flip=horizontal_flip)
        labels = transformed[f"{entity}_flat"]
        valid = labels != IGNORE_INDEX
        labels[valid] = np.asarray([remap[int(value)] for value in labels[valid]], dtype=np.int64)
        for suffix in ("legal_mask", "teacher_logits"):
            key = f"{entity}_{suffix}"
            if key not in transformed:
                continue
            original = transformed[key].copy()
            for old_index, new_index in remap.items():
                transformed[key][:, new_index] = original[:, old_index]

    mask = observation[FEATURE_INDEX["board_mask"]]
    ys, xs = np.nonzero(mask)
    observation[FEATURE_INDEX["x_coordinate"]] = 0
    observation[FEATURE_INDEX["y_coordinate"]] = 0
    if len(xs):
        x_values = np.linspace(-1, 1, xs.max() - xs.min() + 1, dtype=np.float32)
        y_values = np.linspace(-1, 1, ys.max() - ys.min() + 1, dtype=np.float32)
        observation[FEATURE_INDEX["x_coordinate"], ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = x_values[None]
        observation[FEATURE_INDEX["y_coordinate"], ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = y_values[:, None]
    return observation, transformed
