from __future__ import annotations

# ruff: noqa: EM102, PLR2004, TC003
from collections.abc import Mapping
from dataclasses import dataclass
from math import log1p, tanh
from types import MappingProxyType

from luxai2021.game.game_constants import GAME_CONSTANTS

_MAX_TURNS = 360.0
_MAX_RESEARCH = float(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"])
_FUEL_RATE = {"wood": 1.0, "coal": 10.0, "uranium": 40.0}


@dataclass(frozen=True)
class MetricContext:
    """Immutable, candidate-independent board data exposed to the metric DSL."""

    width: int
    height: int
    positions: Mapping[str, tuple[tuple[int, int], ...]]
    sums: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "positions",
            MappingProxyType({str(name): tuple(value) for name, value in self.positions.items()}),
        )
        object.__setattr__(
            self, "sums", MappingProxyType({str(name): float(value) for name, value in self.sums.items()})
        )


@dataclass(frozen=True)
class GameMetrics:
    """Normalized, team-relative state exposed to evolved reward programs."""

    turn: int
    values: Mapping[str, float]
    context: MetricContext | None = None

    def __post_init__(self) -> None:
        clean = {str(name): float(value) for name, value in self.values.items()}
        if not all(-1.000001 <= value <= 1.000001 for value in clean.values()):
            raise ValueError("Game metrics must be normalized to [-1, 1]")
        object.__setattr__(self, "values", MappingProxyType(clean))

    def get(self, name: str) -> float:
        if name not in self.values:
            raise ValueError(f"Unknown reward metric: {name}")
        return self.values[name]


def _symmetric_ratio(own: float, opponent: float, scale: float) -> float:
    return max(-1.0, min(1.0, (own - opponent) / max(scale, own + opponent, 1.0)))


def _team_values(game: object, team: int) -> dict[str, float]:
    city_tiles = 0
    city_fuel = 0.0
    city_upkeep = 0.0
    for city in game.cities.values():
        if city.team == team:
            city_tiles += len(city.city_cells)
            city_fuel += float(city.fuel)
            city_upkeep += float(city.get_light_upkeep())

    units = tuple(game.state["teamStates"][team]["units"].values())
    workers = sum(unit.is_worker() for unit in units)
    carts = len(units) - workers
    cargo_fuel = sum(
        sum(float(amount) * _FUEL_RATE[resource] for resource, amount in unit.cargo.items()) for unit in units
    )
    stats = game.stats["teamStats"][team]
    collected_fuel = sum(
        float(amount) * _FUEL_RATE[resource] for resource, amount in stats["resourcesCollected"].items()
    )
    return {
        "city_tiles": float(city_tiles),
        "units": float(len(units)),
        "workers": float(workers),
        "carts": float(carts),
        "research": float(game.state["teamStates"][team]["researchPoints"]),
        "city_fuel": city_fuel,
        "city_upkeep": city_upkeep,
        "cargo_fuel": cargo_fuel,
        "collected_fuel": collected_fuel,
        "fuel_generated": float(stats["fuelGenerated"]),
        "city_tiles_built": float(stats["cityTilesBuilt"]),
    }


def metrics_from_game(game: object, team: int) -> GameMetrics:
    own = _team_values(game, team)
    opponent = _team_values(game, 1 - team)
    turn = int(game.state["turn"])
    cycle = turn % 40
    survival_own = tanh(log1p(own["city_fuel"]) - log1p(own["city_upkeep"] * 10.0))
    survival_opponent = tanh(log1p(opponent["city_fuel"]) - log1p(opponent["city_upkeep"] * 10.0))
    values = {
        "turn": min(turn / _MAX_TURNS, 1.0),
        "night": 1.0 if cycle >= 30 else 0.0,
        "cycle": cycle / 39.0,
        "city_tiles": _symmetric_ratio(own["city_tiles"], opponent["city_tiles"], 8.0),
        "units": _symmetric_ratio(own["units"], opponent["units"], 8.0),
        "workers": _symmetric_ratio(own["workers"], opponent["workers"], 8.0),
        "carts": _symmetric_ratio(own["carts"], opponent["carts"], 4.0),
        "research": max(-1.0, min(1.0, (own["research"] - opponent["research"]) / _MAX_RESEARCH)),
        "city_fuel": _symmetric_ratio(log1p(own["city_fuel"]), log1p(opponent["city_fuel"]), 4.0),
        "city_survival": max(-1.0, min(1.0, (survival_own - survival_opponent) * 0.5)),
        "cargo_fuel": _symmetric_ratio(log1p(own["cargo_fuel"]), log1p(opponent["cargo_fuel"]), 3.0),
        "collected_fuel": _symmetric_ratio(
            log1p(own["collected_fuel"]),
            log1p(opponent["collected_fuel"]),
            3.0,
        ),
        "fuel_generated": _symmetric_ratio(
            log1p(own["fuel_generated"]),
            log1p(opponent["fuel_generated"]),
            3.0,
        ),
        "city_tiles_built": _symmetric_ratio(
            own["city_tiles_built"],
            opponent["city_tiles_built"],
            8.0,
        ),
    }
    own_units = tuple(game.state["teamStates"][team]["units"].values())
    opponent_units = tuple(game.state["teamStates"][1 - team]["units"].values())
    own_city_cells = tuple(cell for city in game.cities.values() if city.team == team for cell in city.city_cells)
    opponent_city_cells = tuple(
        cell for city in game.cities.values() if city.team == 1 - team for cell in city.city_cells
    )
    resources = tuple(cell for cell in game.map.resources if cell.has_resource())

    def unit_positions(units: tuple[object, ...], kind: str | None = None) -> tuple[tuple[int, int], ...]:
        selected = units
        if kind == "worker":
            selected = tuple(unit for unit in units if unit.is_worker())
        elif kind == "cart":
            selected = tuple(unit for unit in units if unit.is_cart())
        return tuple((int(unit.pos.x), int(unit.pos.y)) for unit in selected)

    def cell_positions(cells: tuple[object, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((int(cell.pos.x), int(cell.pos.y)) for cell in cells)

    resource_positions = {
        resource: cell_positions(tuple(cell for cell in resources if cell.resource.type == resource))
        for resource in _FUEL_RATE
    }
    positions = {
        "own_units": unit_positions(own_units),
        "own_workers": unit_positions(own_units, "worker"),
        "own_carts": unit_positions(own_units, "cart"),
        "opponent_units": unit_positions(opponent_units),
        "opponent_workers": unit_positions(opponent_units, "worker"),
        "opponent_carts": unit_positions(opponent_units, "cart"),
        "own_city_tiles": cell_positions(own_city_cells),
        "opponent_city_tiles": cell_positions(opponent_city_cells),
        "wood_tiles": resource_positions["wood"],
        "coal_tiles": resource_positions["coal"],
        "uranium_tiles": resource_positions["uranium"],
        "resource_tiles": cell_positions(resources),
    }
    sums = {
        "own_city_fuel": own["city_fuel"],
        "opponent_city_fuel": opponent["city_fuel"],
        "own_city_upkeep": own["city_upkeep"],
        "opponent_city_upkeep": opponent["city_upkeep"],
        "own_cargo_fuel": own["cargo_fuel"],
        "opponent_cargo_fuel": opponent["cargo_fuel"],
        "wood_amount": sum(float(cell.resource.amount) for cell in resources if cell.resource.type == "wood"),
        "coal_amount": sum(float(cell.resource.amount) for cell in resources if cell.resource.type == "coal"),
        "uranium_amount": sum(float(cell.resource.amount) for cell in resources if cell.resource.type == "uranium"),
    }
    context = MetricContext(
        width=int(game.map.width),
        height=int(game.map.height),
        positions=positions,
        sums=sums,
    )
    return GameMetrics(turn=turn, values=values, context=context)
