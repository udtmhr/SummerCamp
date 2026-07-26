from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from luxai2021.game.game_constants import GAME_CONSTANTS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

BOARD_SIZE = 32
BOARD_SIZES = (12, 16, 24, 32)
FEATURE_SCHEMA_VERSION = 2
_DAY_LENGTH = GAME_CONSTANTS["PARAMETERS"]["DAY_LENGTH"]
_NIGHT_LENGTH = GAME_CONSTANTS["PARAMETERS"]["NIGHT_LENGTH"]
CYCLE_LENGTH = _DAY_LENGTH + _NIGHT_LENGTH
_MAX_TURNS = GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
GAME_PHASE_COUNT = _MAX_TURNS // CYCLE_LENGTH
_COAL_RESEARCH = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["COAL"]
_URANIUM_RESEARCH = GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"]
CATEGORICAL_FEATURE_NAMES = (
    "day_night_cycle",
    "game_phase",
    "board_size",
)
SPATIAL_FEATURE_NAMES = (
    "board_mask",
    "x_coordinate",
    "y_coordinate",
    "wood",
    "coal",
    "uranium",
    "road",
    "own_worker",
    "own_worker_count",
    "own_worker_cooldown",
    "own_worker_wood",
    "own_worker_coal",
    "own_worker_uranium",
    "own_worker_cargo_full",
    "own_cart",
    "own_cart_count",
    "own_cart_cooldown",
    "own_cart_wood",
    "own_cart_coal",
    "own_cart_uranium",
    "own_cart_cargo_full",
    "enemy_worker",
    "enemy_worker_count",
    "enemy_worker_cooldown",
    "enemy_worker_wood",
    "enemy_worker_coal",
    "enemy_worker_uranium",
    "enemy_worker_cargo_full",
    "enemy_cart",
    "enemy_cart_count",
    "enemy_cart_cooldown",
    "enemy_cart_wood",
    "enemy_cart_coal",
    "enemy_cart_uranium",
    "enemy_cart_cargo_full",
    "own_city",
    "own_city_cooldown",
    "own_city_night_fuel",
    "own_city_upkeep_per_tile",
    "enemy_city",
    "enemy_city_cooldown",
    "enemy_city_night_fuel",
    "enemy_city_upkeep_per_tile",
    "own_research_points",
    "own_coal_researched",
    "own_uranium_researched",
    "enemy_research_points",
    "enemy_coal_researched",
    "enemy_uranium_researched",
    "turn",
    "is_night",
    "turns_to_phase_change",
)
FEATURE_NAMES = (*SPATIAL_FEATURE_NAMES, *CATEGORICAL_FEATURE_NAMES)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


@dataclass(frozen=True)
class UnitSnapshot:
    unit_id: str
    unit_type: int
    team: int
    x: int
    y: int
    cooldown: float
    cargo: Mapping[str, int]

    @property
    def can_act(self) -> bool:
        return self.cooldown < 1


@dataclass(frozen=True)
class CityTileSnapshot:
    team: int
    city_id: str
    x: int
    y: int
    cooldown: float

    @property
    def can_act(self) -> bool:
        return self.cooldown < 1


@dataclass
class BoardSnapshot:
    width: int
    height: int
    turn: int
    research_points: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    resources: dict[tuple[int, int], tuple[str, int]] = field(default_factory=dict)
    roads: dict[tuple[int, int], float] = field(default_factory=dict)
    units: dict[str, UnitSnapshot] = field(default_factory=dict)
    cities: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    city_tiles: list[CityTileSnapshot] = field(default_factory=list)

    @property
    def padding(self) -> tuple[int, int]:
        return (BOARD_SIZE - self.width) // 2, (BOARD_SIZE - self.height) // 2

    def padded_position(self, x: int, y: int) -> tuple[int, int]:
        x_offset, y_offset = self.padding
        return x + x_offset, y + y_offset


def snapshot_from_updates(
    updates: Iterable[str],
    width: int,
    height: int,
    turn: int,
) -> BoardSnapshot:
    snapshot = BoardSnapshot(width=width, height=height, turn=turn)
    for update in updates:
        parts = update.split()
        if not parts:
            continue
        identifier = parts[0]
        if identifier == "rp":
            snapshot.research_points[int(parts[1])] = int(parts[2])
        elif identifier == "r":
            snapshot.resources[(int(parts[2]), int(parts[3]))] = (parts[1], int(float(parts[4])))
        elif identifier == "u":
            unit = UnitSnapshot(
                unit_id=parts[3],
                unit_type=int(parts[1]),
                team=int(parts[2]),
                x=int(parts[4]),
                y=int(parts[5]),
                cooldown=float(parts[6]),
                cargo={"wood": int(parts[7]), "coal": int(parts[8]), "uranium": int(parts[9])},
            )
            snapshot.units[unit.unit_id] = unit
        elif identifier == "c":
            snapshot.cities[parts[2]] = (int(parts[1]), float(parts[3]), float(parts[4]))
        elif identifier == "ct":
            snapshot.city_tiles.append(
                CityTileSnapshot(
                    team=int(parts[1]),
                    city_id=parts[2],
                    x=int(parts[3]),
                    y=int(parts[4]),
                    cooldown=float(parts[5]),
                )
            )
        elif identifier == "ccd":
            snapshot.roads[(int(parts[1]), int(parts[2]))] = float(parts[3])
    return snapshot


def snapshot_from_game(game: object) -> BoardSnapshot:
    snapshot = BoardSnapshot(width=game.map.width, height=game.map.height, turn=game.state["turn"])
    for team in (0, 1):
        snapshot.research_points[team] = game.state["teamStates"][team]["researchPoints"]
        for unit in game.state["teamStates"][team]["units"].values():
            snapshot.units[unit.id] = UnitSnapshot(
                unit_id=unit.id,
                unit_type=unit.type,
                team=unit.team,
                x=unit.pos.x,
                y=unit.pos.y,
                cooldown=unit.cooldown,
                cargo=dict(unit.cargo),
            )
    for cell in game.map.resources:
        if cell.has_resource():
            snapshot.resources[(cell.pos.x, cell.pos.y)] = (cell.resource.type, cell.resource.amount)
    for row in game.map.map:
        for cell in row:
            if cell.road > 0:
                snapshot.roads[(cell.pos.x, cell.pos.y)] = cell.road
    for city in game.cities.values():
        upkeep = city.get_light_upkeep()
        snapshot.cities[city.id] = (city.team, city.fuel, upkeep)
        for cell in city.city_cells:
            snapshot.city_tiles.append(
                CityTileSnapshot(
                    team=city.team,
                    city_id=city.id,
                    x=cell.pos.x,
                    y=cell.pos.y,
                    cooldown=cell.city_tile.cooldown,
                )
            )
    return snapshot


def _fill_unit_features(
    features: np.ndarray,
    snapshot: BoardSnapshot,
    unit: UnitSnapshot,
    prefix: str,
) -> None:
    x, y = snapshot.padded_position(unit.x, unit.y)
    kind = "worker" if unit.unit_type == 0 else "cart"
    base = f"{prefix}_{kind}"
    capacity = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][kind.upper()]
    action_cooldown = GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"][kind.upper()]
    max_cooldown = action_cooldown * 2.0 - 1.0
    features[FEATURE_INDEX[base], y, x] = 1
    features[FEATURE_INDEX[f"{base}_count"], y, x] += 1
    features[FEATURE_INDEX[f"{base}_cooldown"], y, x] = max(
        features[FEATURE_INDEX[f"{base}_cooldown"], y, x],
        min(unit.cooldown / max_cooldown, 1),
    )
    for resource_type in ("wood", "coal", "uranium"):
        features[FEATURE_INDEX[f"{base}_{resource_type}"], y, x] += unit.cargo[resource_type] / capacity
    features[FEATURE_INDEX[f"{base}_cargo_full"], y, x] = max(
        features[FEATURE_INDEX[f"{base}_cargo_full"], y, x],
        float(sum(unit.cargo.values()) >= capacity),
    )


def _fill_city_features(
    features: np.ndarray,
    snapshot: BoardSnapshot,
    tile: CityTileSnapshot,
    prefix: str,
) -> None:
    x, y = snapshot.padded_position(tile.x, tile.y)
    _, fuel, upkeep = snapshot.cities[tile.city_id]
    tile_count = sum(city_tile.city_id == tile.city_id for city_tile in snapshot.city_tiles)
    base = f"{prefix}_city"
    features[FEATURE_INDEX[base], y, x] = 1
    features[FEATURE_INDEX[f"{base}_cooldown"], y, x] = min(tile.cooldown / 10.0, 1)
    features[FEATURE_INDEX[f"{base}_night_fuel"], y, x] = min(fuel / max(upkeep * 10.0, 1.0), 1)
    features[FEATURE_INDEX[f"{base}_upkeep_per_tile"], y, x] = min(
        upkeep / max(23.0 * tile_count, 1.0),
        1,
    )


def encode_snapshot(snapshot: BoardSnapshot, team: int) -> np.ndarray:
    if snapshot.width > BOARD_SIZE or snapshot.height > BOARD_SIZE:
        msg = f"Map {snapshot.width}x{snapshot.height} exceeds {BOARD_SIZE}x{BOARD_SIZE}"
        raise ValueError(msg)
    if snapshot.width != snapshot.height or snapshot.width not in BOARD_SIZES:
        msg = f"Unsupported Season 1 map size: {snapshot.width}x{snapshot.height}"
        raise ValueError(msg)
    features = np.zeros((len(FEATURE_NAMES), BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    x_offset, y_offset = snapshot.padding
    board_slice = np.s_[y_offset : y_offset + snapshot.height, x_offset : x_offset + snapshot.width]
    features[FEATURE_INDEX["board_mask"]][board_slice] = 1
    if snapshot.width > 1:
        x_coordinates = np.linspace(-1, 1, snapshot.width, dtype=np.float32)
        features[FEATURE_INDEX["x_coordinate"]][board_slice] = x_coordinates[None, :]
    if snapshot.height > 1:
        y_coordinates = np.linspace(-1, 1, snapshot.height, dtype=np.float32)
        features[FEATURE_INDEX["y_coordinate"]][board_slice] = y_coordinates[:, None]

    for (raw_x, raw_y), (resource_type, amount) in snapshot.resources.items():
        x, y = snapshot.padded_position(raw_x, raw_y)
        features[FEATURE_INDEX[resource_type], y, x] = min(amount / 800.0, 1)
    for (raw_x, raw_y), road in snapshot.roads.items():
        x, y = snapshot.padded_position(raw_x, raw_y)
        features[FEATURE_INDEX["road"], y, x] = min(road / 6.0, 1)
    for unit in snapshot.units.values():
        _fill_unit_features(features, snapshot, unit, "own" if unit.team == team else "enemy")
    for tile in snapshot.city_tiles:
        _fill_city_features(features, snapshot, tile, "own" if tile.team == team else "enemy")

    enemy = (team + 1) % 2
    own_research = snapshot.research_points[team]
    enemy_research = snapshot.research_points[enemy]
    scalar_features = {
        "own_research_points": min(own_research / _URANIUM_RESEARCH, 1),
        "own_coal_researched": float(own_research >= _COAL_RESEARCH),
        "own_uranium_researched": float(own_research >= _URANIUM_RESEARCH),
        "enemy_research_points": min(enemy_research / _URANIUM_RESEARCH, 1),
        "enemy_coal_researched": float(enemy_research >= _COAL_RESEARCH),
        "enemy_uranium_researched": float(enemy_research >= _URANIUM_RESEARCH),
        "turn": min(snapshot.turn / _MAX_TURNS, 1),
        "is_night": float(snapshot.turn % CYCLE_LENGTH >= _DAY_LENGTH),
        "turns_to_phase_change": (
            (CYCLE_LENGTH - snapshot.turn % CYCLE_LENGTH)
            if snapshot.turn % CYCLE_LENGTH >= _DAY_LENGTH
            else (_DAY_LENGTH - snapshot.turn % CYCLE_LENGTH)
        )
        / _DAY_LENGTH,
        "day_night_cycle": snapshot.turn % CYCLE_LENGTH,
        "game_phase": min(snapshot.turn // CYCLE_LENGTH, GAME_PHASE_COUNT - 1),
        "board_size": BOARD_SIZES.index(snapshot.width),
    }
    mask = features[FEATURE_INDEX["board_mask"]]
    for name, value in scalar_features.items():
        features[FEATURE_INDEX[name]] = mask * value
    return features


def find_adjacent_units(
    snapshot: BoardSnapshot,
    source: UnitSnapshot,
    team: int | None = None,
) -> list[UnitSnapshot]:
    result = []
    for unit in snapshot.units.values():
        if unit.unit_id == source.unit_id:
            continue
        if team is not None and unit.team != team:
            continue
        if abs(unit.x - source.x) + abs(unit.y - source.y) == 1:
            result.append(unit)
    return result
