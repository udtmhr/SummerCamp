from __future__ import annotations

# ruff: noqa: C901, EM102, PLR0911, PLR0912, PLR0913, PLR0915, PLR2004, TC001, TRY004
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import copysign, exp, isclose, isfinite, log1p, sqrt, tanh
from typing import Any

from luxai2021.rl.metrics import GameMetrics

REWARD_PROGRAM_VERSION = 2
SUPPORTED_REWARD_PROGRAM_VERSIONS = frozenset({1, REWARD_PROGRAM_VERSION})
METRIC_NAMES = frozenset(
    {
        "turn",
        "night",
        "cycle",
        "turns_until_night",
        "night_turns_remaining",
        "city_tiles",
        "safe_city_tiles",
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
        "min_city_survival",
        "city_tiles_at_risk",
        "night_fuel_deficit",
        "stranded_fuel",
        "fuel_delivery_coverage",
        "city_tile_loss",
        "city_tile_loss_linear",
        "night_fuel_shortage",
        "worker_resource_access",
        "worker_cargo_fullness",
        "unit_capacity_utilization",
        "coal_unlocked",
        "uranium_unlocked",
        "own_min_city_survival",
        "own_city_tiles_at_risk",
        "own_night_fuel_deficit",
        "own_stranded_fuel",
        "own_fuel_delivery_coverage",
        "own_city_tiles_lost",
        "own_city_tiles_lost_linear",
        "own_night_fuel_shortage",
    }
)
GATING_METRIC_NAMES = frozenset({"turn", "night", "cycle", "turns_until_night", "night_turns_remaining"})
LOWER_IS_BETTER_METRIC_NAMES = frozenset(
    {
        "own_city_tiles_at_risk",
        "own_night_fuel_deficit",
        "own_stranded_fuel",
        "own_city_tiles_lost",
        "own_city_tiles_lost_linear",
        "own_night_fuel_shortage",
    }
)
DIRECT_REWARD_METRIC_NAMES = METRIC_NAMES - GATING_METRIC_NAMES
METRIC_SELECTORS = frozenset(
    {
        "own_units",
        "own_workers",
        "own_carts",
        "opponent_units",
        "opponent_workers",
        "opponent_carts",
        "own_city_tiles",
        "opponent_city_tiles",
        "wood_tiles",
        "coal_tiles",
        "uranium_tiles",
        "resource_tiles",
        "own_at_risk_city_tiles",
        "opponent_at_risk_city_tiles",
        "own_safe_city_tiles",
        "opponent_safe_city_tiles",
        "own_fuel_carrying_workers",
        "opponent_fuel_carrying_workers",
        "own_full_workers",
        "opponent_full_workers",
        "own_actionable_workers",
        "opponent_actionable_workers",
        "own_harvestable_resource_tiles",
        "opponent_harvestable_resource_tiles",
    }
)
METRIC_SUM_NAMES = frozenset(
    {
        "own_city_fuel",
        "opponent_city_fuel",
        "own_city_upkeep",
        "opponent_city_upkeep",
        "own_cargo_fuel",
        "opponent_cargo_fuel",
        "wood_amount",
        "coal_amount",
        "uranium_amount",
        "own_night_fuel_required",
        "opponent_night_fuel_required",
        "own_night_fuel_deficit",
        "opponent_night_fuel_deficit",
        "own_at_risk_delivery_fuel",
        "opponent_at_risk_delivery_fuel",
        "own_city_tiles_lost",
        "opponent_city_tiles_lost",
    }
)
_UNARY_OPS = frozenset({"abs", "neg", "tanh", "exp_decay", "log1p_abs", "square"})
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
    component_shaping: Mapping[str, float]


@dataclass(frozen=True)
class RewardComponent:
    name: str
    expression: Mapping[str, Any]
    weight: float


@dataclass(frozen=True)
class DerivedMetric:
    name: str
    expression: Mapping[str, Any]


REWARD_MODES = ("potential_linear", "potential_tanh", "direct_step")


@dataclass(frozen=True)
class RewardProgram:
    components: tuple[RewardComponent, ...]
    derived_metrics: tuple[DerivedMetric, ...] = ()
    reward_scale: float = 1.0
    gamma: float = 0.995
    version: int = 1
    mode: str = "potential_linear"
    terminal_reward_scale: float = 1.0
    normalize_total: bool = False

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward mode: {self.mode}")
        if not isfinite(self.terminal_reward_scale) or not 1.0 <= self.terminal_reward_scale <= 100.0:
            raise ValueError("terminal_reward_scale must be in [1, 100]")
        if not isinstance(self.normalize_total, bool):
            raise TypeError("normalize_total must be a boolean")
        if self.mode == "direct_step":
            maximum_shaping = self.reward_scale * 2.0 * sum(abs(component.weight) for component in self.components)
        elif self.mode == "potential_tanh":
            maximum_shaping = self.reward_scale * (1.0 + self.gamma)
        else:
            maximum_shaping = self.reward_scale * 5.0 * (1.0 + self.gamma)
        if self.normalize_total and self.terminal_reward_scale <= maximum_shaping:
            raise ValueError("terminal_reward_scale must exceed the maximum potential shaping magnitude")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RewardProgram:
        version = int(value.get("version", 1))
        if version not in SUPPORTED_REWARD_PROGRAM_VERSIONS:
            raise ValueError("Unsupported reward program version")
        raw_derived = value.get("derived_metrics", [])
        if version == 1 and raw_derived:
            raise ValueError("Reward program v1 cannot contain derived metrics")
        if not isinstance(raw_derived, list) or len(raw_derived) > 16:
            raise ValueError("Reward program allows at most 16 derived metrics")
        derived_metrics = []
        derived_names: set[str] = set()
        for raw in raw_derived:
            if not isinstance(raw, Mapping):
                raise ValueError("Derived metrics must be objects")
            name = str(raw.get("name", ""))
            if not name or name in derived_names or name in METRIC_NAMES or not name.replace("_", "").isalnum():
                raise ValueError(f"Invalid or duplicate derived metric name: {name!r}")
            expression = raw.get("expression")
            _validate_expression(expression, derived_names=derived_names)
            derived_names.add(name)
            derived_metrics.append(DerivedMetric(name, expression))
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
            _validate_expression(expression, derived_names=derived_names)
            weight = float(raw.get("weight", 1.0))
            if not isfinite(weight) or not -5.0 <= weight <= 5.0:
                raise ValueError("Reward component weights must be finite and in [-5, 5]")
            names.add(name)
            components.append(RewardComponent(name, expression, weight))
        reward_scale = float(value.get("reward_scale", 1.0))
        gamma = float(value.get("gamma", 0.995))
        mode = str(value.get("mode", "potential_linear"))
        terminal_reward_scale = float(value.get("terminal_reward_scale", 1.0))
        normalize_total = value.get("normalize_total", False)
        if mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward mode: {mode}")
        if not isfinite(reward_scale) or not 0.0 <= reward_scale <= 2.0:
            raise ValueError("reward_scale must be in [0, 2.0]")
        if not isfinite(gamma) or not 0.9 <= gamma <= 1.0:
            raise ValueError("reward gamma must be in [0.9, 1.0]")
        return cls(
            tuple(components),
            tuple(derived_metrics),
            reward_scale=reward_scale,
            gamma=gamma,
            version=version,
            mode=mode,
            terminal_reward_scale=terminal_reward_scale,
            normalize_total=normalize_total,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "version": self.version,
            "mode": self.mode,
            "components": [
                {"name": item.name, "expression": dict(item.expression), "weight": item.weight}
                for item in self.components
            ],
            "reward_scale": self.reward_scale,
            "gamma": self.gamma,
        }
        if self.version >= 2:
            value["derived_metrics"] = [
                {"name": item.name, "expression": dict(item.expression)} for item in self.derived_metrics
            ]
        if self.terminal_reward_scale != 1.0:
            value["terminal_reward_scale"] = self.terminal_reward_scale
        if self.normalize_total:
            value["normalize_total"] = True
        return value

    def potential(self, metrics: GameMetrics) -> tuple[float, dict[str, float]]:
        derived_values: dict[str, float] = {}
        for derived in self.derived_metrics:
            derived_values[derived.name] = max(
                -1.0,
                min(1.0, _evaluate_expression(derived.expression, metrics, derived_values)),
            )
        values = {
            component.name: max(
                -1.0,
                min(1.0, _evaluate_expression(component.expression, metrics, derived_values)),
            )
            for component in self.components
        }
        raw_sum = sum(component.weight * values[component.name] for component in self.components)
        potential = tanh(raw_sum) if self.mode == "potential_tanh" else max(-5.0, min(5.0, raw_sum))
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
        previous_potential, prev_component_values = self.potential(previous)
        next_potential, component_values = self.potential(following)

        if self.mode == "direct_step":
            raw_shaping = self.reward_scale * sum(
                component.weight * (component_values[component.name] - prev_component_values[component.name])
                for component in self.components
            )
        else:
            raw_shaping = self.reward_scale * (self.gamma * next_potential - previous_potential)

        component_shaping = {
            component.name: self.reward_scale
            * component.weight
            * (
                component_values[component.name] - prev_component_values[component.name]
                if self.mode == "direct_step"
                else self.gamma * component_values[component.name] - prev_component_values[component.name]
            )
            for component in self.components
        }

        if self.normalize_total:
            terminal = terminal_outcome
            shaping = raw_shaping / self.terminal_reward_scale
            total = terminal + shaping
            component_shaping = {
                name: value / self.terminal_reward_scale for name, value in component_shaping.items()
            }
        else:
            terminal = terminal_outcome
            shaping = raw_shaping
            total = max(-3.0, min(3.0, terminal_outcome * self.terminal_reward_scale + shaping))
        return RewardBreakdown(
            total=total,
            terminal=terminal,
            shaping=shaping,
            previous_potential=previous_potential,
            next_potential=next_potential,
            components=component_values,
            component_shaping=component_shaping,
        )


def _validate_expression(expression: object, *, derived_names: set[str] | None = None) -> None:
    nodes = 0
    known_derived = derived_names or set()

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
        if op == "derived":
            if node.get("name") not in known_derived:
                raise ValueError(f"Unknown or forward-referenced derived metric: {node.get('name')}")
            return
        if op == "count":
            if node.get("selector") not in METRIC_SELECTORS:
                raise ValueError(f"Unknown metric selector: {node.get('selector')}")
            return
        if op == "sum":
            if node.get("name") not in METRIC_SUM_NAMES:
                raise ValueError(f"Unknown metric sum: {node.get('name')}")
            return
        if op == "distance":
            if node.get("source") not in METRIC_SELECTORS or node.get("target") not in METRIC_SELECTORS:
                raise ValueError("Unknown distance selector")
            if node.get("reduce") not in {"min", "mean", "max"}:
                raise ValueError("Distance reduction must be min, mean, or max")
            return
        if op == "density":
            if node.get("source") not in METRIC_SELECTORS or node.get("target") not in METRIC_SELECTORS:
                raise ValueError("Unknown density selector")
            radius = int(node.get("radius", 0))
            if not 1 <= radius <= 8:
                raise ValueError("Density radius must be in [1, 8]")
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


def _evaluate_expression(
    expression: Mapping[str, Any],
    metrics: GameMetrics,
    derived_values: Mapping[str, float] | None = None,
) -> float:
    op = expression["op"]
    if op == "constant":
        return float(expression["value"])
    if op == "metric":
        return metrics.get(str(expression["name"]))
    if op == "derived":
        values = derived_values or {}
        return float(values[str(expression["name"])])
    context = metrics.context
    if op in {"count", "sum", "distance", "density"} and context is None:
        return 0.0
    if op == "count":
        return min(len(context.positions[str(expression["selector"])]) / 32.0, 1.0)
    if op == "sum":
        return tanh(log1p(max(0.0, context.sums[str(expression["name"])])) / 8.0)
    if op == "distance":
        source = context.positions[str(expression["source"])]
        target = context.positions[str(expression["target"])]
        if not source or not target:
            return 1.0
        distances = [abs(ax - bx) + abs(ay - by) for ax, ay in source for bx, by in target]
        reduction = str(expression["reduce"])
        distance = (
            min(distances)
            if reduction == "min"
            else max(distances)
            if reduction == "max"
            else sum(distances) / len(distances)
        )
        return min(float(distance) / max(context.width + context.height - 2, 1), 1.0)
    if op == "density":
        source = context.positions[str(expression["source"])]
        target = context.positions[str(expression["target"])]
        if not source:
            return 0.0
        radius = int(expression["radius"])
        counts = [sum(abs(ax - bx) + abs(ay - by) <= radius for bx, by in target) for ax, ay in source]
        return min((sum(counts) / len(counts)) / max((2 * radius + 1) ** 2, 1), 1.0)
    if op in _UNARY_OPS:
        value = _evaluate_expression(expression["value"], metrics, derived_values)
        return {
            "abs": abs,
            "neg": lambda item: -item,
            "tanh": tanh,
            "exp_decay": lambda item: exp(-abs(item)),
            "log1p_abs": lambda item: copysign(log1p(abs(item)), item),
            "square": lambda item: item * item,
        }[op](value)
    if op in _BINARY_OPS:
        left = _evaluate_expression(expression["left"], metrics, derived_values)
        right = _evaluate_expression(expression["right"], metrics, derived_values)
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
        value = _evaluate_expression(expression["value"], metrics, derived_values)
        return max(float(expression["low"]), min(float(expression["high"]), value))
    condition = _evaluate_expression(expression["condition"], metrics, derived_values)
    branch = "when_true" if condition > 0 else "when_false"
    return _evaluate_expression(expression[branch], metrics, derived_values)


def default_reward_program(mode: str = "potential_linear") -> RewardProgram:
    # The component L1 envelope grows from 3.0 to 5.2 with the two survival
    # penalties. Potential itself remains clipped to [-5, 5]. Raw win/loss is
    # +/-10 and the complete reward is divided by 10, so normalized shaping is
    # below 0.35 and cannot reverse the terminal result.
    return RewardProgram.from_dict(
        {
            "version": REWARD_PROGRAM_VERSION,
            "mode": mode,
            "components": [
                {"name": "city_tiles", "expression": {"op": "metric", "name": "city_tiles"}, "weight": 1.5},
                {
                    "name": "city_survival",
                    "expression": {"op": "metric", "name": "city_survival"},
                    "weight": 0.8,
                },
                {"name": "units", "expression": {"op": "metric", "name": "units"}, "weight": 0.5},
                {"name": "research", "expression": {"op": "metric", "name": "research"}, "weight": 0.2},
                {
                    "name": "own_night_fuel_deficit",
                    "expression": {"op": "metric", "name": "own_night_fuel_deficit"},
                    "weight": -1.0,
                },
                {
                    "name": "own_city_tiles_lost",
                    "expression": {"op": "metric", "name": "own_city_tiles_lost"},
                    "weight": -1.2,
                },
            ],
            "reward_scale": 0.35,
            "gamma": 0.999,
            "terminal_reward_scale": 10.0,
            "normalize_total": True,
        }
    )


def calibrate_reward_scale(
    parent: RewardProgram,
    child: RewardProgram,
    transitions: Iterable[tuple[GameMetrics, GameMetrics]],
    *,
    parent_effective_scale: float | None = None,
    minimum_samples: int = 256,
    activity_threshold: float = 0.01,
) -> tuple[RewardProgram, dict[str, Any]]:
    """Match dense shaping RMS without amplifying unseen sparse components."""
    pairs = list(transitions)
    if not pairs:
        raise ValueError("Reward calibration requires transitions")

    def traces(program: RewardProgram) -> tuple[dict[str, list[float]], list[dict[str, float]], list[dict[str, float]]]:
        deltas: dict[str, list[float]] = defaultdict(list)
        previous_values = []
        following_values = []
        for previous, following in pairs:
            _, before = program.potential(previous)
            _, after = program.potential(following)
            previous_values.append(before)
            following_values.append(after)
            for component in program.components:
                deltas[component.name].append(program.gamma * after[component.name] - before[component.name])
        return deltas, previous_values, following_values

    def dense_rms(program: RewardProgram) -> tuple[float | None, dict[str, float], list[str], float]:
        deltas, before, after = traces(program)
        rates = {name: sum(abs(value) > 1e-6 for value in values) / len(values) for name, values in deltas.items()}
        dense = [component for component in program.components if rates[component.name] >= activity_threshold]
        if len(pairs) < minimum_samples or not dense:
            return None, rates, [], 0.0
        potential_before = [
            tanh(sum(component.weight * values[component.name] for component in dense)) for values in before
        ]
        potential_after = [
            tanh(sum(component.weight * values[component.name] for component in dense)) for values in after
        ]
        shaping = [
            program.gamma * following - previous for previous, following in zip(potential_before, potential_after)
        ]
        absolute = sorted(abs(value) for value in shaping)
        cap = absolute[min(len(absolute) - 1, int(0.99 * (len(absolute) - 1)))]
        winsorized_fraction = sum(abs(value) > cap for value in shaping) / len(shaping)
        rms = sqrt(sum(min(abs(value), cap) ** 2 for value in shaping) / len(shaping))
        return rms, rates, [component.name for component in dense], winsorized_fraction

    parent_rms, parent_rates, parent_dense, parent_winsorized = dense_rms(parent)
    child_rms, child_rates, child_dense, child_winsorized = dense_rms(child)
    parent_scale = float(parent_effective_scale if parent_effective_scale is not None else parent.reward_scale)
    nominal_ratio = max(0.8, min(1.25, child.reward_scale / max(parent.reward_scale, 1e-6)))
    fallback = parent_rms is None or child_rms is None or parent_rms < 1e-8 or child_rms < 1e-8
    unbounded = parent_scale * nominal_ratio if fallback else parent_scale * parent_rms * nominal_ratio / child_rms
    effective = max(0.01, min(2.0, max(parent_scale * 0.5, min(parent_scale * 2.0, unbounded))))
    if not isfinite(effective):
        raise ValueError("Reward calibration produced a non-finite scale")
    calibrated = RewardProgram(
        child.components,
        child.derived_metrics,
        reward_scale=effective,
        gamma=child.gamma,
        version=child.version,
        mode=child.mode,
        terminal_reward_scale=child.terminal_reward_scale,
        normalize_total=child.normalize_total,
    )
    return calibrated, {
        "parent_effective_scale": parent_scale,
        "proposed_scale": child.reward_scale,
        "effective_scale": effective,
        "unbounded_scale": unbounded,
        "clipped": not isclose(effective, unbounded, rel_tol=1e-9, abs_tol=1e-12),
        "fallback": fallback,
        "sample_count": len(pairs),
        "activity_threshold": activity_threshold,
        "parent_dense_components": parent_dense,
        "child_dense_components": child_dense,
        "parent_sparse_components": [name for name in parent_rates if name not in parent_dense],
        "child_sparse_components": [name for name in child_rates if name not in child_dense],
        "parent_component_activity": parent_rates,
        "child_component_activity": child_rates,
        "parent_winsorized_fraction": parent_winsorized,
        "child_winsorized_fraction": child_winsorized,
        "parent_dense_rms": parent_rms,
        "child_dense_rms": child_rms,
    }
