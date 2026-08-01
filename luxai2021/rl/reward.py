from __future__ import annotations

# ruff: noqa: C901, EM102, PLR0911, PLR2004, TC001, TRY004
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, tanh
from typing import Any

from luxai2021.rl.metrics import GameMetrics

REWARD_PROGRAM_VERSION = 1
METRIC_NAMES = frozenset(
    {
        "turn",
        "night",
        "cycle",
        "city_tiles",
        "units",
        "workers",
        "carts",
        "research",
        "city_fuel",
        "city_survival",
        "cargo_fuel",
        "collected_fuel",
        "fuel_generated",
        "city_tiles_built",
    }
)
_UNARY_OPS = frozenset({"abs", "neg", "tanh"})
_BINARY_OPS = frozenset({"add", "sub", "mul", "safe_div", "min", "max"})
_MAX_DEPTH = 12
_MAX_NODES = 128


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    terminal: float
    shaping: float
    previous_potential: float
    next_potential: float
    components: Mapping[str, float]


@dataclass(frozen=True)
class RewardComponent:
    name: str
    expression: Mapping[str, Any]
    weight: float


@dataclass(frozen=True)
class RewardProgram:
    components: tuple[RewardComponent, ...]
    reward_scale: float = 0.2
    gamma: float = 0.995
    version: int = REWARD_PROGRAM_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RewardProgram:
        if int(value.get("version", REWARD_PROGRAM_VERSION)) != REWARD_PROGRAM_VERSION:
            raise ValueError("Unsupported reward program version")
        raw_components = value.get("components")
        if not isinstance(raw_components, list) or not 1 <= len(raw_components) <= 16:
            raise ValueError("Reward program requires 1 to 16 components")
        components = []
        names = set()
        for raw in raw_components:
            if not isinstance(raw, Mapping):
                raise ValueError("Reward components must be objects")
            name = str(raw.get("name", ""))
            if not name or name in names or not name.replace("_", "").isalnum():
                raise ValueError(f"Invalid or duplicate reward component name: {name!r}")
            expression = raw.get("expression")
            _validate_expression(expression)
            weight = float(raw.get("weight", 1.0))
            if not isfinite(weight) or not -5.0 <= weight <= 5.0:
                raise ValueError("Reward component weights must be finite and in [-5, 5]")
            names.add(name)
            components.append(RewardComponent(name, expression, weight))
        reward_scale = float(value.get("reward_scale", 0.2))
        gamma = float(value.get("gamma", 0.995))
        if not isfinite(reward_scale) or not 0.0 <= reward_scale <= 0.5:
            raise ValueError("reward_scale must be in [0, 0.5]")
        if not isfinite(gamma) or not 0.9 <= gamma <= 1.0:
            raise ValueError("reward gamma must be in [0.9, 1.0]")
        return cls(tuple(components), reward_scale=reward_scale, gamma=gamma)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "components": [
                {"name": item.name, "expression": dict(item.expression), "weight": item.weight}
                for item in self.components
            ],
            "reward_scale": self.reward_scale,
            "gamma": self.gamma,
        }

    def potential(self, metrics: GameMetrics) -> tuple[float, dict[str, float]]:
        values = {
            component.name: max(-1.0, min(1.0, _evaluate_expression(component.expression, metrics)))
            for component in self.components
        }
        potential = tanh(sum(component.weight * values[component.name] for component in self.components))
        if not isfinite(potential):
            raise ValueError("Reward program produced a non-finite potential")
        return potential, values

    def reward(
        self,
        previous: GameMetrics,
        following: GameMetrics,
        *,
        terminal_outcome: float = 0.0,
    ) -> RewardBreakdown:
        if terminal_outcome not in {-1.0, 0.0, 1.0}:
            raise ValueError("Terminal outcome must be -1, 0, or 1")
        previous_potential, _ = self.potential(previous)
        next_potential, component_values = self.potential(following)
        shaping = self.reward_scale * (self.gamma * next_potential - previous_potential)
        total = max(-1.5, min(1.5, terminal_outcome + shaping))
        return RewardBreakdown(
            total=total,
            terminal=terminal_outcome,
            shaping=shaping,
            previous_potential=previous_potential,
            next_potential=next_potential,
            components=component_values,
        )


def _validate_expression(expression: object) -> None:
    nodes = 0

    def visit(node: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("Reward expression is too large")
        if not isinstance(node, Mapping):
            raise ValueError("Reward expression nodes must be objects")
        op = node.get("op")
        if op == "constant":
            value = float(node.get("value"))
            if not isfinite(value) or not -5.0 <= value <= 5.0:
                raise ValueError("Reward constants must be finite and in [-5, 5]")
            return
        if op == "metric":
            if node.get("name") not in METRIC_NAMES:
                raise ValueError(f"Unknown reward metric: {node.get('name')}")
            return
        if op in _UNARY_OPS:
            visit(node.get("value"), depth + 1)
            return
        if op in _BINARY_OPS:
            visit(node.get("left"), depth + 1)
            visit(node.get("right"), depth + 1)
            return
        if op == "clip":
            visit(node.get("value"), depth + 1)
            low, high = float(node.get("low", -1.0)), float(node.get("high", 1.0))
            if not isfinite(low) or not isfinite(high) or low >= high or low < -5.0 or high > 5.0:
                raise ValueError("Invalid reward clip bounds")
            return
        if op == "gate":
            visit(node.get("condition"), depth + 1)
            visit(node.get("when_true"), depth + 1)
            visit(node.get("when_false"), depth + 1)
            return
        raise ValueError(f"Unsupported reward operation: {op!r}")

    visit(expression, 0)


def _evaluate_expression(expression: Mapping[str, Any], metrics: GameMetrics) -> float:
    op = expression["op"]
    if op == "constant":
        return float(expression["value"])
    if op == "metric":
        return metrics.get(str(expression["name"]))
    if op in _UNARY_OPS:
        value = _evaluate_expression(expression["value"], metrics)
        return {"abs": abs, "neg": lambda item: -item, "tanh": tanh}[op](value)
    if op in _BINARY_OPS:
        left = _evaluate_expression(expression["left"], metrics)
        right = _evaluate_expression(expression["right"], metrics)
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        if op == "safe_div":
            return left / (right if abs(right) >= 1e-6 else (1e-6 if right >= 0 else -1e-6))
        return min(left, right) if op == "min" else max(left, right)
    if op == "clip":
        value = _evaluate_expression(expression["value"], metrics)
        return max(float(expression["low"]), min(float(expression["high"]), value))
    condition = _evaluate_expression(expression["condition"], metrics)
    branch = "when_true" if condition > 0 else "when_false"
    return _evaluate_expression(expression[branch], metrics)


def default_reward_program() -> RewardProgram:
    return RewardProgram.from_dict(
        {
            "version": REWARD_PROGRAM_VERSION,
            "components": [
                {"name": "city_tiles", "expression": {"op": "metric", "name": "city_tiles"}, "weight": 1.5},
                {
                    "name": "city_survival",
                    "expression": {"op": "metric", "name": "city_survival"},
                    "weight": 0.8,
                },
                {"name": "units", "expression": {"op": "metric", "name": "units"}, "weight": 0.5},
                {"name": "research", "expression": {"op": "metric", "name": "research"}, "weight": 0.2},
            ],
            "reward_scale": 0.2,
            "gamma": 0.995,
        }
    )
