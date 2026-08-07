from __future__ import annotations

# ruff: noqa: EM102, PLR0915, PLR2004, TC003
from collections.abc import Mapping
from dataclasses import dataclass
from math import log1p, tanh
from types import MappingProxyType

from luxai2021.game.game_constants import GAME_CONSTANTS

_MAX_TURNS = 360.0
_MAX_RESEARCH = float(GAME_CONSTANTS["PARAMETERS"]["RESEARCH_REQUIREMENTS"]["URANIUM"])
_FUEL_RATE = {"wood": 1.0, "coal": 10.0, "uranium": 40.0}
_CITY_LOSS_NORMALIZER = 32.0
_SHORTAGE_NORMALIZER = 8.0
_DELIVERY_RADIUS = 3


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


def _strategic_team_values(
    game: object,
    team: int,
    *,
    required_night_turns: int,
    night_length: int,
    resources: tuple[object, ...],
) -> tuple[dict[str, float], dict[str, tuple[tuple[int, int], ...]], dict[str, float]]:
    """Return bounded city-risk signals and selector data for one team."""
    cities = tuple(city for city in game.cities.values() if city.team == team)
    units = tuple(game.state["teamStates"][team]["units"].values())
    workers = tuple(unit for unit in units if unit.is_worker())
    research = float(game.state["teamStates"][team]["researchPoints"])
    parameters = game.configs["parameters"]
    coal_requirement = float(parameters["RESEARCH_REQUIREMENTS"]["COAL"])
    uranium_requirement = float(parameters["RESEARCH_REQUIREMENTS"]["URANIUM"])
    harvestable_types = {"wood"}
    if research >= coal_requirement:
        harvestable_types.add("coal")
    if research >= uranium_requirement:
        harvestable_types.add("uranium")
    harvestable_cells = tuple(cell for cell in resources if cell.resource.type in harvestable_types)
    harvestable_positions = tuple((int(cell.pos.x), int(cell.pos.y)) for cell in harvestable_cells)

    city_tiles = sum(len(city.city_cells) for city in cities)
    required_fuel = 0.0
    fuel_deficit = 0.0
    fuel_surplus = 0.0
    survival_turns = []
    at_risk_cells = []
    safe_cells = []
    for city in cities:
        upkeep = max(float(city.get_light_upkeep()), 0.0)
        fuel = max(float(city.fuel), 0.0)
        requirement = upkeep * required_night_turns
        deficit = max(requirement - fuel, 0.0)
        surplus = max(fuel - requirement, 0.0)
        required_fuel += requirement
        fuel_deficit += deficit
        fuel_surplus += surplus
        survival_turns.append(fuel / upkeep if upkeep > 1e-6 else float(night_length))
        (at_risk_cells if deficit > 1e-6 else safe_cells).extend(city.city_cells)

    if cities:
        min_survival = min(min(survival_turns) / max(night_length, 1), 1.0)
        at_risk_fraction = len(at_risk_cells) / max(city_tiles, 1)
        deficit_fraction = fuel_deficit / max(required_fuel, 1e-6)
        stranded_fraction = min(fuel_deficit, fuel_surplus) / max(required_fuel, 1e-6)
    else:
        # A destroyed city must not look safer than a surviving city.
        min_survival = 0.0
        at_risk_fraction = 1.0
        deficit_fraction = 1.0
        stranded_fraction = 0.0

    def position(item: object) -> tuple[int, int]:
        return int(item.pos.x), int(item.pos.y)

    at_risk_positions = tuple(position(cell) for cell in at_risk_cells)
    fuel_carrying_workers = tuple(unit for unit in workers if float(unit.get_cargo_fuel_value()) > 0.0)
    full_workers = tuple(unit for unit in workers if int(unit.get_cargo_space_left()) <= 0)
    actionable_workers = tuple(unit for unit in workers if unit.can_act())
    delivery_fuel = sum(
        float(unit.get_cargo_fuel_value())
        for unit in fuel_carrying_workers
        if any(
            abs(int(unit.pos.x) - city_x) + abs(int(unit.pos.y) - city_y) <= _DELIVERY_RADIUS
            for city_x, city_y in at_risk_positions
        )
    )
    delivery_coverage = (
        0.0 if not cities else min(delivery_fuel / max(fuel_deficit, 1e-6), 1.0) if fuel_deficit > 0 else 1.0
    )
    resource_access = (
        sum(
            any(abs(int(unit.pos.x) - x) + abs(int(unit.pos.y) - y) <= 1 for x, y in harvestable_positions)
            for unit in workers
        )
        / len(workers)
        if workers
        else 0.0
    )
    cargo_fullness = (
        sum(
            1.0 - max(float(unit.get_cargo_space_left()), 0.0) / float(parameters["RESOURCE_CAPACITY"]["WORKER"])
            for unit in workers
        )
        / len(workers)
        if workers
        else 0.0
    )
    capacity_utilization = min(len(units) / max(city_tiles, 1), 1.0) if city_tiles else 0.0
    loss_events = tuple(
        event
        for event in getattr(game, "diagnostic_events", ())
        if event.get("event") == "city_destroyed_night_fuel" and int(event.get("team", -1)) == team
    )
    city_tiles_lost = sum(int(event.get("city_tiles_lost", 0)) for event in loss_events)

    values = {
        "min_city_survival": min_survival,
        "city_tiles_at_risk": min(max(at_risk_fraction, 0.0), 1.0),
        "night_fuel_deficit": min(max(deficit_fraction, 0.0), 1.0),
        "stranded_fuel": min(max(stranded_fraction, 0.0), 1.0),
        "fuel_delivery_coverage": delivery_coverage,
        "city_tiles_lost": city_tiles_lost / (city_tiles_lost + _CITY_LOSS_NORMALIZER),
        "night_fuel_shortage": len(loss_events) / (len(loss_events) + _SHORTAGE_NORMALIZER),
        "worker_resource_access": min(max(resource_access, 0.0), 1.0),
        "worker_cargo_fullness": min(max(cargo_fullness, 0.0), 1.0),
        "unit_capacity_utilization": capacity_utilization,
        "coal_unlocked": 1.0 if research >= coal_requirement else 0.0,
        "uranium_unlocked": 1.0 if research >= uranium_requirement else 0.0,
    }
    positions = {
        "at_risk_city_tiles": at_risk_positions,
        "safe_city_tiles": tuple(position(cell) for cell in safe_cells),
        "fuel_carrying_workers": tuple(position(unit) for unit in fuel_carrying_workers),
        "full_workers": tuple(position(unit) for unit in full_workers),
        "actionable_workers": tuple(position(unit) for unit in actionable_workers),
        "harvestable_resource_tiles": harvestable_positions,
    }
    sums = {
        "night_fuel_required": required_fuel,
        "night_fuel_deficit": fuel_deficit,
        "at_risk_delivery_fuel": delivery_fuel,
        "city_tiles_lost": float(city_tiles_lost),
    }
    return values, positions, sums


def metrics_from_game(game: object, team: int) -> GameMetrics:
    own = _team_values(game, team)
    opponent = _team_values(game, 1 - team)
    turn = int(game.state["turn"])
    parameters = game.configs["parameters"]
    day_length = int(parameters["DAY_LENGTH"])
    night_length = int(parameters["NIGHT_LENGTH"])
    cycle_length = day_length + night_length
    cycle = turn % cycle_length
    turns_until_night = max(day_length - cycle, 0) if cycle < day_length else 0
    night_turns_remaining = cycle_length - cycle if cycle >= day_length else 0
    required_night_turns = night_length if cycle < day_length else night_turns_remaining
    resources = tuple(cell for cell in game.map.resources if cell.has_resource())
    own_strategy, own_strategy_positions, own_strategy_sums = _strategic_team_values(
        game,
        team,
        required_night_turns=required_night_turns,
        night_length=night_length,
        resources=resources,
    )
    opponent_strategy, opponent_strategy_positions, opponent_strategy_sums = _strategic_team_values(
        game,
        1 - team,
        required_night_turns=required_night_turns,
        night_length=night_length,
        resources=resources,
    )
    survival_own = tanh(log1p(own["city_fuel"]) - log1p(own["city_upkeep"] * night_length))
    survival_opponent = tanh(log1p(opponent["city_fuel"]) - log1p(opponent["city_upkeep"] * night_length))
    values = {
        "turn": min(turn / _MAX_TURNS, 1.0),
        "night": 1.0 if cycle >= day_length else 0.0,
        "cycle": cycle / max(cycle_length - 1, 1),
        "turns_until_night": turns_until_night / max(day_length, 1),
        "night_turns_remaining": night_turns_remaining / max(night_length, 1),
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
        "min_city_survival": own_strategy["min_city_survival"] - opponent_strategy["min_city_survival"],
        "city_tiles_at_risk": opponent_strategy["city_tiles_at_risk"] - own_strategy["city_tiles_at_risk"],
        "night_fuel_deficit": opponent_strategy["night_fuel_deficit"] - own_strategy["night_fuel_deficit"],
        "stranded_fuel": opponent_strategy["stranded_fuel"] - own_strategy["stranded_fuel"],
        "fuel_delivery_coverage": (
            own_strategy["fuel_delivery_coverage"] - opponent_strategy["fuel_delivery_coverage"]
        ),
        "city_tile_loss": opponent_strategy["city_tiles_lost"] - own_strategy["city_tiles_lost"],
        "night_fuel_shortage": (opponent_strategy["night_fuel_shortage"] - own_strategy["night_fuel_shortage"]),
        "worker_resource_access": (
            own_strategy["worker_resource_access"] - opponent_strategy["worker_resource_access"]
        ),
        "worker_cargo_fullness": (own_strategy["worker_cargo_fullness"] - opponent_strategy["worker_cargo_fullness"]),
        "unit_capacity_utilization": (
            own_strategy["unit_capacity_utilization"] - opponent_strategy["unit_capacity_utilization"]
        ),
        "coal_unlocked": own_strategy["coal_unlocked"] - opponent_strategy["coal_unlocked"],
        "uranium_unlocked": own_strategy["uranium_unlocked"] - opponent_strategy["uranium_unlocked"],
        "own_min_city_survival": own_strategy["min_city_survival"],
        "own_city_tiles_at_risk": own_strategy["city_tiles_at_risk"],
        "own_night_fuel_deficit": own_strategy["night_fuel_deficit"],
        "own_stranded_fuel": own_strategy["stranded_fuel"],
        "own_fuel_delivery_coverage": own_strategy["fuel_delivery_coverage"],
        "own_city_tiles_lost": own_strategy["city_tiles_lost"],
        "own_night_fuel_shortage": own_strategy["night_fuel_shortage"],
    }
    own_units = tuple(game.state["teamStates"][team]["units"].values())
    opponent_units = tuple(game.state["teamStates"][1 - team]["units"].values())
    own_city_cells = tuple(cell for city in game.cities.values() if city.team == team for cell in city.city_cells)
    opponent_city_cells = tuple(
        cell for city in game.cities.values() if city.team == 1 - team for cell in city.city_cells
    )

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
        **{f"own_{name}": value for name, value in own_strategy_positions.items()},
        **{f"opponent_{name}": value for name, value in opponent_strategy_positions.items()},
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
        **{f"own_{name}": value for name, value in own_strategy_sums.items()},
        **{f"opponent_{name}": value for name, value in opponent_strategy_sums.items()},
    }
    context = MetricContext(
        width=int(game.map.width),
        height=int(game.map.height),
        positions=positions,
        sums=sums,
    )
    return GameMetrics(turn=turn, values=values, context=context)
