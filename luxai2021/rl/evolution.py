from __future__ import annotations

# ruff: noqa: C901, EM102, PLR0912, PLR0913, PLR0915, PLR2004, S311, S603
import copy
import gzip
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from luxai2021.rl.ppo import PPOConfig
from luxai2021.rl.reward import (
    DIRECT_REWARD_METRIC_NAMES,
    LOWER_IS_BETTER_METRIC_NAMES,
    METRIC_NAMES,
    METRIC_SELECTORS,
    METRIC_SUM_NAMES,
    REWARD_MODES,
    RewardProgram,
    default_reward_program,
)

EVOLUTION_SCHEMA_VERSION = 3
LEGACY_EVOLUTION_SCHEMA_VERSION = 1
SUPPORTED_EVOLUTION_SCHEMA_VERSIONS = frozenset({1, 2, EVOLUTION_SCHEMA_VERSION})
OPPONENT_KEYS = ("self_base", "other_base", "teacher", "snapshot")
LUX_S1_RULES_SOURCE_URL = "https://www.lux-ai.org/specs-2021#Background"
LUX_S1_RULES_PATH = Path(__file__).with_name("lux_s1_rules.md")
MUTATION_KINDS = frozenset(
    {"initial", "legacy", "parameter", "structural", "feature_existing", "feature_generated", "crossover", "restart"}
)
PROPOSABLE_MUTATION_KINDS = MUTATION_KINDS - {"legacy", "initial", "feature_generated"}
INHERITANCE_MODES = frozenset({"base", "policy", "policy_value"})
_STRUCTURAL_WRAPPERS = frozenset({"abs", "neg", "tanh", "exp_decay", "log1p_abs", "square"})
_STRUCTURAL_BINARY_OPS = frozenset({"add", "sub", "mul", "safe_div", "min", "max"})
_I03_STRUCTURAL_DISTANCE_MIN = 0.20
_I03_STRUCTURAL_DISTANCE_MAX = 0.65
_I03_STRUCTURAL_ATTEMPTS = 16


def lux_s1_rules_context() -> dict[str, str]:
    summary = LUX_S1_RULES_PATH.read_text(encoding="utf-8").strip()
    return {
        "source_url": LUX_S1_RULES_SOURCE_URL,
        "summary": summary,
        "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
    }


@dataclass(frozen=True)
class OpponentMix:
    self_base: float = 0.20
    other_base: float = 0.05
    teacher: float = 0.25
    snapshot: float = 0.50

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("Opponent weights must be in [0, 1]")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Opponent weights must sum to 1")

    def choose(self, rng: random.Random) -> str:
        draw = rng.random()
        cumulative = 0.0
        for name, weight in asdict(self).items():
            cumulative += weight
            if draw <= cumulative:
                return name
        return "snapshot"

    def allocate(self, count: int, rng: random.Random) -> list[str]:
        if count <= 0:
            return []
        values = asdict(self)
        raw_counts = {name: round(weight * count) for name, weight in values.items()}
        current_sum = sum(raw_counts.values())
        if current_sum != count:
            diff = count - current_sum
            sorted_names = sorted(values.keys(), key=lambda k: values[k], reverse=True)
            for i in range(abs(diff)):
                target_name = sorted_names[i % len(sorted_names)]
                raw_counts[target_name] += 1 if diff > 0 else -1
        allocated = []
        for name, n in raw_counts.items():
            allocated.extend([name] * max(0, n))
        rng.shuffle(allocated)
        return allocated


def _piecewise_linear(points: tuple[tuple[float, float], ...], progress: float) -> float:
    value = min(max(float(progress), 0.0), 1.0)
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if value <= right_x:
            fraction = (value - left_x) / max(right_x - left_x, 1e-9)
            return left_y + fraction * (right_y - left_y)
    return points[-1][1]


@dataclass(frozen=True)
class TrainingCurriculum:
    name: str
    teacher_floor_points: tuple[tuple[float, float], ...]
    shaping_multiplier_points: tuple[tuple[float, float], ...]
    snapshot_floor: float = 0.10
    bc_coefficient_points: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 1.0))

    def shaping_multiplier(self, progress: float) -> float:
        return _piecewise_linear(self.shaping_multiplier_points, progress)

    def bc_coefficient_multiplier(self, progress: float) -> float:
        return _piecewise_linear(self.bc_coefficient_points, progress)

    def opponent_mix(self, proposed: OpponentMix, progress: float) -> OpponentMix:
        if self.name == "legacy":
            return proposed
        snapshot = max(proposed.snapshot, self.snapshot_floor)
        teacher = min(
            max(proposed.teacher, _piecewise_linear(self.teacher_floor_points, progress)),
            1.0 - snapshot,
        )
        if teacher + snapshot > 1.0 + 1e-9:
            snapshot = self.snapshot_floor
            teacher = 1.0 - snapshot
        remaining = max(0.0, 1.0 - teacher - snapshot)
        base_total = proposed.self_base + proposed.other_base
        self_fraction = proposed.self_base / base_total if base_total > 0 else 0.5
        return OpponentMix(
            self_base=remaining * self_fraction,
            other_base=remaining * (1.0 - self_fraction),
            teacher=teacher,
            snapshot=snapshot,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "teacher_floor_points": [list(point) for point in self.teacher_floor_points],
            "shaping_multiplier_points": [list(point) for point in self.shaping_multiplier_points],
            "snapshot_floor": self.snapshot_floor,
            "bc_coefficient_points": [list(point) for point in self.bc_coefficient_points],
        }


def training_curriculum(name: str) -> TrainingCurriculum:
    teacher_points = ((0.0, 0.25), (0.30, 0.30), (0.65, 0.40), (0.85, 0.50), (1.0, 0.50))
    if name == "legacy":
        return TrainingCurriculum("legacy", ((0.0, 0.0), (1.0, 0.0)), ((0.0, 1.0), (1.0, 1.0)), 0.0)
    if name == "teacher_guarded_near_sparse":
        return TrainingCurriculum(
            name,
            teacher_points,
            ((0.0, 1.0), (0.30, 1.0), (0.60, 0.50), (0.80, 0.20), (1.0, 0.05)),
        )
    if name == "terminal_only_ablation":
        return TrainingCurriculum(
            name,
            teacher_points,
            ((0.0, 1.0), (0.30, 1.0), (0.60, 0.50), (0.80, 0.0), (1.0, 0.0)),
        )
    if name == "dense_shaping":
        return TrainingCurriculum(
            name,
            ((0.0, 0.10), (1.0, 0.10)),
            ((0.0, 1.0), (0.20, 1.0), (0.70, 0.50), (1.0, 0.25)),
            snapshot_floor=0.10,
            bc_coefficient_points=((0.0, 1.0), (0.20, 1.0), (0.70, 0.80), (1.0, 0.20)),
        )
    raise ValueError(f"Unknown curriculum profile: {name}")


@dataclass(frozen=True)
class EvolutionCandidate:
    candidate_id: str
    generation: int
    island: int
    parent_ids: tuple[str, ...]
    reward_program: RewardProgram
    ppo_config: PPOConfig
    opponent_mix: OpponentMix
    rationale: str
    mutation_kind: str = "legacy"
    primary_parent_id: str | None = None
    secondary_parent_ids: tuple[str, ...] = ()
    inheritance_mode: str = "base"
    mutation_manifest: Mapping[str, Any] = field(default_factory=dict)
    parameter_constraint_coefficient: float = 0.0
    schema_version: int = EVOLUTION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionCandidate:
        schema_version = int(value.get("schema_version", 0))
        if schema_version not in SUPPORTED_EVOLUTION_SCHEMA_VERSIONS:
            raise ValueError("Unsupported evolution candidate schema")
        candidate = cls.from_proposal(
            value,
            generation=int(value["generation"]),
            island=int(value["island"]),
            parent_ids=tuple(value.get("parent_ids", ())),
            schema_version=schema_version,
        )
        if candidate.candidate_id != value["candidate_id"]:
            raise ValueError("Evolution candidate content hash does not match its id")
        return candidate

    @classmethod
    def from_proposal(
        cls,
        proposal: Mapping[str, Any],
        *,
        generation: int,
        island: int,
        parent_ids: tuple[str, ...],
        schema_version: int = EVOLUTION_SCHEMA_VERSION,
    ) -> EvolutionCandidate:
        reward = RewardProgram.from_dict(proposal["reward_program"])
        raw_ppo = dict(proposal["ppo_config"])
        ppo = PPOConfig(**raw_ppo)
        if not math.isclose(reward.gamma, ppo.gamma, abs_tol=1e-12):
            raise ValueError("Reward-program gamma must match PPO gamma")
        opponent = OpponentMix(**proposal["opponent_mix"])
        rationale = str(proposal.get("rationale", ""))[:4000]
        mutation_kind = str(proposal.get("mutation_kind", "legacy" if schema_version == 1 else "parameter"))
        primary = proposal.get("primary_parent_id")
        primary_parent_id = str(primary) if primary is not None else None
        secondary_parent_ids = tuple(str(item) for item in proposal.get("secondary_parent_ids", ()))
        inheritance_mode = str(proposal.get("inheritance_mode", "base" if not parent_ids else "policy"))
        mutation_manifest = dict(proposal.get("mutation_manifest", {}))
        parameter_constraint_coefficient = float(proposal.get("parameter_constraint_coefficient", 0.0))
        if mutation_kind not in MUTATION_KINDS:
            raise ValueError(f"Unsupported mutation kind: {mutation_kind}")
        if inheritance_mode not in INHERITANCE_MODES:
            raise ValueError(f"Unsupported inheritance mode: {inheritance_mode}")
        if not 0.0 <= parameter_constraint_coefficient <= 1.0:
            raise ValueError("parameter_constraint_coefficient must be in [0, 1]")
        if primary_parent_id is not None and primary_parent_id not in parent_ids:
            raise ValueError("primary_parent_id must be one of parent_ids")
        if any(item not in parent_ids or item == primary_parent_id for item in secondary_parent_ids):
            raise ValueError("secondary_parent_ids must be distinct parent_ids")
        canonical_ppo = asdict(ppo)
        if schema_version < 3:
            canonical_ppo.pop("illegal_action_coefficient")
        canonical_value: dict[str, Any] = {
            "generation": generation,
            "island": island,
            "parents": parent_ids,
            "reward": reward.to_dict(),
            "ppo": canonical_ppo,
            "opponent": asdict(opponent),
        }
        if schema_version >= 2:
            canonical_value.update(
                {
                    "mutation_kind": mutation_kind,
                    "primary_parent_id": primary_parent_id,
                    "secondary_parent_ids": secondary_parent_ids,
                    "inheritance_mode": inheritance_mode,
                    "mutation_manifest": mutation_manifest,
                    "parameter_constraint_coefficient": parameter_constraint_coefficient,
                }
            )
        canonical = json.dumps(
            canonical_value,
            sort_keys=True,
            separators=(",", ":"),
        )
        candidate_id = f"g{generation:02d}-i{island:02d}-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"
        return cls(
            candidate_id,
            generation,
            island,
            parent_ids,
            reward,
            ppo,
            opponent,
            rationale,
            mutation_kind,
            primary_parent_id,
            secondary_parent_ids,
            inheritance_mode,
            mutation_manifest,
            parameter_constraint_coefficient,
            schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        schema_version = self.schema_version
        ppo_config = asdict(self.ppo_config)
        if schema_version < 3:
            ppo_config.pop("illegal_action_coefficient")
        value = {
            "schema_version": schema_version,
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "island": self.island,
            "parent_ids": list(self.parent_ids),
            "reward_program": self.reward_program.to_dict(),
            "ppo_config": ppo_config,
            "opponent_mix": asdict(self.opponent_mix),
            "rationale": self.rationale,
        }
        if schema_version >= 2:
            value.update(
                {
                    "mutation_kind": self.mutation_kind,
                    "primary_parent_id": self.primary_parent_id,
                    "secondary_parent_ids": list(self.secondary_parent_ids),
                    "inheritance_mode": self.inheritance_mode,
                    "mutation_manifest": dict(self.mutation_manifest),
                    "parameter_constraint_coefficient": self.parameter_constraint_coefficient,
                }
            )
        return value


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    stage: str
    status: str
    score_rate: float
    teacher_score_rate: float
    kl: float
    duration_seconds: float
    metrics: Mapping[str, Any]
    error: str | None = None

    @property
    def fitness(self) -> tuple[float, float, float]:
        reflection = self.metrics.get("reflection", {}) if isinstance(self.metrics, Mapping) else {}
        diagnostics = reflection.get("diagnostics", {}) if isinstance(reflection, Mapping) else {}
        illegal_count = int(diagnostics.get("illegal_action_count", 0)) if isinstance(diagnostics, Mapping) else 0
        valid = 1.0 if self.status == "completed" and illegal_count == 0 else 0.0
        return valid, self.teacher_score_rate, self.score_rate


class EvolutionStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.candidate_dir = run_dir / "candidates"
        self.result_dir = run_dir / "results"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def save_manifest(self, value: Mapping[str, Any]) -> None:
        self.write_json(self.run_dir / "manifest.json", value)

    def save_candidate(self, candidate: EvolutionCandidate) -> None:
        self.write_json(self.candidate_dir / f"{candidate.candidate_id}.json", candidate.to_dict())

    def save_result(self, result: CandidateResult) -> None:
        self.write_json(self.result_dir / f"{result.candidate_id}-{result.stage}.json", asdict(result))

    def results(self) -> list[CandidateResult]:
        return [
            CandidateResult(**json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.result_dir.glob("*.json"))
        ]

    def candidates(self) -> list[EvolutionCandidate]:
        return [
            EvolutionCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.candidate_dir.glob("*.json"))
        ]


@dataclass(frozen=True)
class EvolutionJob:
    candidate_id: str
    stage: str
    base_name: str
    seconds: int
    eval_seeds: int
    eval_seed_start: int
    decision_budget: int | None = None

    @property
    def job_id(self) -> str:
        return f"{self.candidate_id}--{self.stage}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionJob:
        job = cls(
            candidate_id=str(value["candidate_id"]),
            stage=str(value["stage"]),
            base_name=str(value["base_name"]),
            seconds=int(value["seconds"]),
            eval_seeds=int(value["eval_seeds"]),
            eval_seed_start=int(value["eval_seed_start"]),
            decision_budget=(int(value["decision_budget"]) if value.get("decision_budget") is not None else None),
        )
        if (
            job.base_name not in {"unet", "resattn8"}
            or job.seconds < 0
            or job.eval_seeds < 1
            or (job.decision_budget is not None and job.decision_budget < 1)
        ):
            raise ValueError("Invalid distributed evolution job")
        return job


class FilesystemJobQueue:
    """Atomic shared-filesystem queue for independent candidate training jobs."""

    def __init__(self, run_dir: Path) -> None:
        self.pending_dir = run_dir / "jobs" / "pending"
        self.running_dir = run_dir / "jobs" / "running"
        self.completed_dir = run_dir / "jobs" / "completed"
        for path in (self.pending_dir, self.running_dir, self.completed_dir):
            path.mkdir(parents=True, exist_ok=True)

    def enqueue(self, job: EvolutionJob) -> None:
        filename = f"{job.job_id}.json"
        pending = self.pending_dir / filename
        completed = self.completed_dir / filename
        if pending.exists():
            return
        if completed.exists():
            payload = json.loads(completed.read_text(encoding="utf-8"))
            if payload.get("result_status") == "completed":
                return
            completed.unlink()
        if any(self.running_dir.glob(f"*--{filename}")):
            return
        EvolutionStore.write_json(pending, job.to_dict())

    def claim(self, worker_id: str) -> tuple[EvolutionJob, Path] | None:
        safe_worker = "".join(character if character.isalnum() or character in "-_" else "_" for character in worker_id)
        for pending in sorted(self.pending_dir.glob("*.json")):
            claimed = self.running_dir / f"{safe_worker}--{pending.name}"
            try:
                pending.replace(claimed)
            except FileNotFoundError:
                continue
            return EvolutionJob.from_dict(json.loads(claimed.read_text(encoding="utf-8"))), claimed
        return None

    def complete(self, claimed_path: Path, result: CandidateResult) -> None:
        job_id = f"{result.candidate_id}--{result.stage}"
        completed = self.completed_dir / f"{job_id}.json"
        if completed.exists():
            claimed_path.unlink(missing_ok=True)
            return
        source_path = claimed_path if claimed_path.exists() else self.pending_dir / f"{job_id}.json"
        if source_path.exists():
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        else:
            payload = {"candidate_id": result.candidate_id, "stage": result.stage}
        payload.update({"result_status": result.status, "completed_at": time.time()})
        EvolutionStore.write_json(completed, payload)
        source_path.unlink(missing_ok=True)
        claimed_path.unlink(missing_ok=True)
        for duplicate in self.running_dir.glob(f"*--{job_id}.json"):
            duplicate.unlink(missing_ok=True)

    def heartbeat(self, claimed_path: Path) -> None:
        if not claimed_path.exists():
            raise ValueError("Unknown or expired lease")
        claimed_path.touch()

    def release(self, claimed_path: Path) -> None:
        if not claimed_path.exists():
            return
        job = EvolutionJob.from_dict(json.loads(claimed_path.read_text(encoding="utf-8")))
        pending = self.pending_dir / f"{job.job_id}.json"
        if not pending.exists():
            claimed_path.replace(pending)

    def recover_stale(self, stale_seconds: float) -> int:
        if stale_seconds <= 0:
            return 0
        recovered = 0
        cutoff = time.time() - stale_seconds
        for running in sorted(self.running_dir.glob("*.json")):
            if running.stat().st_mtime >= cutoff:
                continue
            job = EvolutionJob.from_dict(json.loads(running.read_text(encoding="utf-8")))
            pending = self.pending_dir / f"{job.job_id}.json"
            if not pending.exists():
                running.replace(pending)
                recovered += 1
        return recovered

    def outstanding_ids(self) -> set[str]:
        paths = [*self.pending_dir.glob("*.json"), *self.running_dir.glob("*.json")]
        return {EvolutionJob.from_dict(json.loads(path.read_text(encoding="utf-8"))).job_id for path in paths}


def proposal_schema() -> dict[str, Any]:
    def expression_object(properties: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": dict(properties),
            "required": list(properties),
            "additionalProperties": False,
        }

    expression_ref = {"$ref": "#/$defs/expression"}
    expression: dict[str, Any] = {
        "anyOf": [
            expression_object(
                {
                    "op": {"type": "string", "const": "constant"},
                    "value": {"type": "number", "minimum": -5.0, "maximum": 5.0},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "metric"},
                    "name": {"type": "string", "enum": sorted(METRIC_NAMES)},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "derived"},
                    "name": {"type": "string"},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "count"},
                    "selector": {"type": "string", "enum": sorted(METRIC_SELECTORS)},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "sum"},
                    "name": {"type": "string", "enum": sorted(METRIC_SUM_NAMES)},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "distance"},
                    "source": {"type": "string", "enum": sorted(METRIC_SELECTORS)},
                    "target": {"type": "string", "enum": sorted(METRIC_SELECTORS)},
                    "reduce": {"type": "string", "enum": ["min", "mean", "max"]},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "density"},
                    "source": {"type": "string", "enum": sorted(METRIC_SELECTORS)},
                    "target": {"type": "string", "enum": sorted(METRIC_SELECTORS)},
                    "radius": {"type": "integer", "minimum": 1, "maximum": 8},
                }
            ),
            expression_object(
                {
                    "op": {
                        "type": "string",
                        "enum": sorted({"abs", "neg", "tanh", "exp_decay", "log1p_abs", "square"}),
                    },
                    "value": expression_ref,
                }
            ),
            expression_object(
                {
                    "op": {
                        "type": "string",
                        "enum": sorted({"add", "sub", "mul", "safe_div", "min", "max"}),
                    },
                    "left": expression_ref,
                    "right": expression_ref,
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "clip"},
                    "value": expression_ref,
                    "low": {"type": "number", "minimum": -5.0, "maximum": 5.0},
                    "high": {"type": "number", "minimum": -5.0, "maximum": 5.0},
                }
            ),
            expression_object(
                {
                    "op": {"type": "string", "const": "gate"},
                    "condition": expression_ref,
                    "when_true": expression_ref,
                    "when_false": expression_ref,
                }
            ),
        ]
    }
    ppo_properties = {
        field: (
            {"type": "number", "const": 0.0}
            if field == "kl_coefficient"
            else {"type": "boolean"}
            if field in {"joint_action_policy", "online_teacher_kl"}
            else {
                "type": "integer"
                if field in {"update_epochs", "minibatch_turns", "joint_loss_reference_actions"}
                else "number"
            }
        )
        for field in asdict(PPOConfig())
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"expression": expression},
        "type": "object",
        "properties": {
            "reward_program": {
                "type": "object",
                "properties": {
                    "version": {"type": "integer", "const": 3},
                    "mode": {"type": "string", "enum": list(REWARD_MODES)},
                    "derived_metrics": {
                        "type": "array",
                        "maxItems": 16,
                        "items": expression_object(
                            {
                                "name": {"type": "string"},
                                "expression": {"$ref": "#/$defs/expression"},
                            }
                        ),
                    },
                    "components": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "expression": {"$ref": "#/$defs/expression"},
                                "weight": {"type": "number"},
                            },
                            "required": ["name", "expression", "weight"],
                            "additionalProperties": False,
                        },
                    },
                    "reward_scale": {"type": "number"},
                    "gamma": {"type": "number"},
                    "terminal_reward_scale": {"type": "number"},
                    "normalize_total": {"type": "boolean"},
                    "terminal_potential_zero": {"type": "boolean", "const": True},
                },
                "required": [
                    "version",
                    "mode",
                    "derived_metrics",
                    "components",
                    "reward_scale",
                    "gamma",
                    "terminal_reward_scale",
                    "normalize_total",
                    "terminal_potential_zero",
                ],
                "additionalProperties": False,
            },
            "ppo_config": {
                "type": "object",
                "properties": ppo_properties,
                "required": list(ppo_properties),
                "additionalProperties": False,
            },
            "opponent_mix": {
                "type": "object",
                "properties": {name: {"type": "number"} for name in OPPONENT_KEYS},
                "required": list(OPPONENT_KEYS),
                "additionalProperties": False,
            },
            "mutation_kind": {"type": "string", "enum": sorted(PROPOSABLE_MUTATION_KINDS)},
            "primary_parent_id": {"type": ["string", "null"]},
            "secondary_parent_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "inheritance_mode": {"type": "string", "enum": sorted(INHERITANCE_MODES)},
            "mutation_manifest": expression_object(
                {
                    "changed_paths": {"type": "array", "items": {"type": "string"}, "maxItems": 256},
                    "summary": {"type": "string"},
                }
            ),
            "parameter_constraint_coefficient": {"type": "number", "const": 0.0},
            "rationale": {"type": "string"},
        },
        "required": [
            "reward_program",
            "ppo_config",
            "opponent_mix",
            "mutation_kind",
            "primary_parent_id",
            "secondary_parent_ids",
            "inheritance_mode",
            "mutation_manifest",
            "parameter_constraint_coefficient",
            "rationale",
        ],
        "additionalProperties": False,
    }


def build_codex_prompt(
    parents: list[EvolutionCandidate],
    results: list[CandidateResult],
    *,
    island: int,
    generation: int,
    rules_context: Mapping[str, str] | None = None,
) -> str:
    parent_payload = [parent.to_dict() for parent in parents]
    selected_results = select_codex_feedback_results(parents, results)
    result_payload = [_codex_result_feedback(result) for result in selected_results]
    island_role = {
        0: "parameter: change only one or two numeric leaves; keep AST topology unchanged; inherit policy_value",
        1: "structural: make one phase-aware local AST edit using turn/night gates; inherit policy",
        2: (
            "feature: add or delete exactly one direct normalized fuel-delivery, City-risk, cargo, or resource-access "
            "metric component; inherit policy"
        ),
        3: "diversity: freely redesign the bounded reward structure, recombine parents, or restart",
    }.get(island, "parameter")
    edit_guidance = (
        "- On islands 0-2, prefer the single targeted change required by the island contract."
        if island != 3
        else """- On island 3, coordinated edits across multiple components and subtrees are encouraged when supported
  by diagnostics. Use only structural, crossover, or restart; feature_generated is deprecated.
- AST distance 0.20 to 0.65 is a diversity target, not a validity boundary. Safe smaller or larger edits are accepted
  and the server reclassifies mutation_kind, inheritance, constraint, and changed_paths from the actual diff.
- Island-3 structural proposals may additionally change either active PPO settings or opponent_mix.
- Island-3 crossover may recombine multiple reward components/derived metrics, must include a distinct contribution
  from a secondary parent, and must keep the primary parent's active PPO settings and opponent mix.
- Declare restart only when intentionally starting from the distilled base without a parent checkpoint."""
    )
    context = {
        "context_schema_version": 1,
        "rules": dict(rules_context or lux_s1_rules_context()),
        "anchor_hierarchy": {
            "strongest_anchor": "teacher",
            "teacher_identity": "Lux AI 2021 official first-place agent",
            "selection_policy": (
                "Teacher performance is a non-regression guard. A gain against self_base must not compensate for "
                "a regression against teacher."
            ),
        },
        "metric_semantics": {
            "terminal_outcome": "authoritative win=1, draw=0, loss=-1 objective",
            "city_tiles": "relative proxy for the primary end-game tiebreak; expansion also adds fuel liability",
            "city_survival": "aggregate relative City fuel safety",
            "min_city_survival": "minimum individual-City survival; use for local collapse",
            "night_fuel_deficit": "relative signal oriented so larger is better",
            "own_night_fuel_deficit": "absolute own-team deficit; lower is better",
            "stranded_fuel": (
                "relative avoidable fuel concentration across disconnected Cities; larger is better"
            ),
            "own_stranded_fuel": (
                "absolute own-team fuel simultaneously surplus in one City and needed by another; lower is better"
            ),
            "fuel_delivery_coverage": "relative ability to deliver carried fuel to at-risk Cities",
            "own_fuel_delivery_coverage": "absolute own-team delivery coverage; larger is better",
            "turn_phase_metrics": "turn/night/cycle values are gates, not standalone objectives",
        },
        "generation": generation,
        "island": island,
        "island_role": island_role,
        "parents": parent_payload,
        "recent_results": result_payload,
    }
    prompt = f"""You are evolving a safe reward program and PPO configuration for Lux AI Challenge 2021.
Return exactly one proposal matching the supplied JSON schema.

Hard constraints:
- Keep the observation schema, first_place_flat_v1 actions, UNet, and ResAttn8 architectures unchanged.
- Reward expressions may use only schema operations and normalized metrics.
- Preserve terminal win/loss reward; design bounded potential shaping for city survival and match strength.
- The official first-place Teacher is stronger than the distilled bases. Teacher non-regression is a hard objective;
  a self_base gain cannot compensate for a Teacher regression.
- PPO uses clipping for update stability and the first-place Teacher distillation anchor; it trains on one GPU.
- Parent-policy KL loss and L2-SP are disabled. Always set
  parameter_constraint_coefficient to 0; do not propose mutations of either field.
- Opponent weights must sum to exactly 1.0.
{edit_guidance}
- Illegal-action events identify hard action-mask defects. Do not trade them against reward and never weaken an
  existing illegal-action mask; runtime masks may only become stricter.
- Use the reported city-loss/night-fuel turns and parent deltas to explain the proposed change.
- Emit reward_program version 3 with terminal_potential_zero=true.
  Derived metrics must use only safe Reward IR expressions in the schema.
- turn/night/cycle/turns_until_night/night_turns_remaining describe phase and should be used as gate conditions,
  not standalone objectives. Relative city-risk/loss/deficit metrics are oriented so larger is better. Absolute
  own_city_tiles_at_risk, own_night_fuel_deficit, own_stranded_fuel, own_city_tiles_lost, and
  own_night_fuel_shortage are lower-is-better.
- min_city_survival is the minimum over individual cities, while city_survival uses aggregate team fuel. Prefer the
  minimum/risk/deficit signals when feedback reports a local night-fuel collapse despite adequate total fuel.
- own_stranded_fuel is nonzero only when one connected City has surplus fuel while another connected City has a
  simultaneous deficit. Its relative counterpart stranded_fuel is oriented so larger is better.
- On island 2, every generation adds or deletes exactly one direct whitelist metric; do not create derived metrics.
- A restart sets primary_parent_id to null, inheritance_mode to base, and parameter_constraint_coefficient to 0.
- mutation_kind, parent provenance, inheritance_mode, and changed_paths must truthfully describe the proposal.

Evolution context JSON (authoritative rules, metric semantics, parents, and feedback):
{json.dumps(context, ensure_ascii=False, separators=(",", ":"), sort_keys=True)}
"""  # noqa: S608
    return prompt  # noqa: RET504


def _stage_priority(stage: str) -> int:
    if stage.startswith("final"):
        return 3
    if stage.startswith("medium"):
        return 2
    return 1


def select_codex_feedback_results(
    parents: list[EvolutionCandidate], results: list[CandidateResult]
) -> list[CandidateResult]:
    """Select parent evidence plus two distinct high-fitness non-parent candidates."""
    selected: list[CandidateResult] = []
    parent_ids = {parent.candidate_id for parent in parents}
    for parent in parents:
        parent_results = [result for result in results if result.candidate_id == parent.candidate_id]
        completed = [result for result in parent_results if result.status == "completed"]
        best_completed = max(
            completed,
            key=lambda result: (_stage_priority(result.stage), result.fitness),
            default=None,
        )
        if best_completed is not None:
            selected.append(best_completed)
        failures = [result for result in parent_results if result.status == "failed"]
        if failures:
            highest_failure = max(failures, key=lambda result: _stage_priority(result.stage))
            if best_completed is None or _stage_priority(highest_failure.stage) > _stage_priority(best_completed.stage):
                selected.append(highest_failure)

    best_non_parent: dict[str, CandidateResult] = {}
    for result in results:
        if result.candidate_id in parent_ids or result.status != "completed":
            continue
        current = best_non_parent.get(result.candidate_id)
        if current is None or (_stage_priority(result.stage), result.fitness) > (
            _stage_priority(current.stage),
            current.fitness,
        ):
            best_non_parent[result.candidate_id] = result
    selected.extend(sorted(best_non_parent.values(), key=lambda result: result.fitness, reverse=True)[:2])
    deduplicated: list[CandidateResult] = []
    seen: set[tuple[str, str, str]] = set()
    for result in selected:
        key = result.candidate_id, result.stage, result.status
        if key not in seen:
            seen.add(key)
            deduplicated.append(result)
    return deduplicated[:8]


def compress_turn_ranges(turns: object) -> list[list[int]]:
    normalized = sorted({int(turn) for turn in turns}) if isinstance(turns, (list, tuple, set)) else []
    ranges: list[list[int]] = []
    for turn in normalized:
        if ranges and turn == ranges[-1][1] + 1:
            ranges[-1][1] = turn
        else:
            ranges.append([turn, turn])
    return ranges


def _event_deficit(event: Mapping[str, Any]) -> float:
    values = [
        abs(float(value))
        for key, value in event.items()
        if "deficit" in str(key).lower() and isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return max(values, default=0.0)


def _representative_city_events(events: object) -> list[Mapping[str, Any]]:
    values = [event for event in events if isinstance(event, Mapping)] if isinstance(events, (list, tuple)) else []
    if len(values) <= 8:
        return values
    indexes = {0, len(values) - 1, max(range(len(values)), key=lambda index: _event_deficit(values[index]))}
    remaining = 8 - len(indexes)
    if remaining > 0:
        indexes.update(round(index * (len(values) - 1) / (remaining + 1)) for index in range(1, remaining + 1))
    return [values[index] for index in sorted(indexes)[:8]]


def _codex_result_feedback(result: CandidateResult) -> dict[str, Any]:
    metrics = result.metrics
    training = metrics.get("training", {}) if isinstance(metrics, Mapping) else {}
    history = training.get("history", ()) if isinstance(training, Mapping) else ()
    evaluation = metrics.get("evaluation", {}) if isinstance(metrics, Mapping) else {}
    last_update = history[-1] if history and isinstance(history[-1], Mapping) else {}
    training_keys = (
        "loss",
        "policy_loss",
        "value_loss",
        "bc_loss",
        "entropy",
        "illegal_action_loss",
        "illegal_action_mass_mean",
        "illegal_action_mass_p95",
        "illegal_action_mass_max",
        "effective_reward_scale",
        "reward_scale",
        "decisions_per_second",
        "throughput",
    )
    reflection = metrics.get("reflection", {}) if isinstance(metrics, Mapping) else {}
    diagnostics = reflection.get("diagnostics", {}) if isinstance(reflection, Mapping) else {}
    city_events = diagnostics.get("city_loss_events", ()) if isinstance(diagnostics, Mapping) else ()
    illegal_events = diagnostics.get("illegal_action_events", ()) if isinstance(diagnostics, Mapping) else ()
    illegal_classes = Counter(
        str(event.get("action_class", event.get("action", "unknown")))
        for event in illegal_events
        if isinstance(event, Mapping)
    )
    parent_comparisons = []
    for comparison in reflection.get("parent_comparisons", ()) if isinstance(reflection, Mapping) else ():
        if not isinstance(comparison, Mapping):
            continue
        changes = comparison.get("changes", ())
        parent_comparisons.append(
            {
                "parent_id": comparison.get("parent_id"),
                "comparison_stage": comparison.get("comparison_stage"),
                "changed_paths": [
                    {
                        "path": change.get("path"),
                        "before": change.get("before"),
                        "after": change.get("after"),
                    }
                    for change in changes
                    if isinstance(change, Mapping)
                ],
                "improvement": comparison.get("improvement"),
            }
        )
    stored_illegal_classes = diagnostics.get("illegal_action_class_counts", {})
    if isinstance(stored_illegal_classes, Mapping):
        illegal_classes = Counter({str(key): int(value) for key, value in stored_illegal_classes.items()})
    failure_metric = metrics.get("failure", {}) if isinstance(metrics, Mapping) else {}
    failure_error = result.error
    failure_type = result.error.split(":", 1)[0] if result.error else None
    if not failure_error and isinstance(failure_metric, Mapping):
        failure_error = str(failure_metric.get("message", "")) or None
        failure_type = str(failure_metric.get("type", "")) or None
    return {
        "candidate_id": result.candidate_id,
        "stage": result.stage,
        "status": result.status,
        "score_rate": result.score_rate,
        "teacher_score_rate": result.teacher_score_rate,
        "duration_seconds": result.duration_seconds,
        "training_final": {key: last_update[key] for key in training_keys if key in last_update},
        "league_totals": evaluation.get("totals") if isinstance(evaluation, Mapping) else None,
        "diagnostics": {
            "city_tile_loss_turn_ranges": compress_turn_ranges(diagnostics.get("city_tile_loss_turns", ())),
            "night_fuel_shortage_turn_ranges": compress_turn_ranges(
                diagnostics.get("night_fuel_shortage_turns", ())
            ),
            "city_event_count": diagnostics.get("city_event_count", len(city_events)),
            "city_tiles_lost": diagnostics.get("city_tiles_lost", 0),
            "max_fuel_deficit": diagnostics.get(
                "max_fuel_deficit",
                max((_event_deficit(event) for event in city_events if isinstance(event, Mapping)), default=0.0),
            ),
            "city_events": _representative_city_events(city_events),
            "illegal_turn_ranges": compress_turn_ranges(diagnostics.get("illegal_action_turns", ())),
            "illegal_action_count": diagnostics.get("illegal_action_count", len(illegal_events)),
            "illegal_action_classes": dict(illegal_classes),
        },
        "parent_comparisons": parent_comparisons,
        "failure": None if not failure_error else {"type": failure_type, "error": failure_error[:1000]},
    }


def _flatten_settings(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened = {}
        for name in sorted(value):
            path = f"{prefix}.{name}" if prefix else str(name)
            flattened.update(_flatten_settings(value[name], path))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_settings(item, f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def candidate_setting_changes(
    parent: EvolutionCandidate,
    candidate: EvolutionCandidate,
) -> list[dict[str, object]]:
    parent_reward = parent.reward_program.to_dict()
    candidate_reward = candidate.reward_program.to_dict()
    if parent_reward.get("version") == 1 and candidate_reward.get("version") == 2:
        parent_reward = dict(parent_reward)
        parent_reward["version"] = 2
        parent_reward["derived_metrics"] = []
    parent_ppo = asdict(parent.ppo_config)
    candidate_ppo = asdict(candidate.ppo_config)
    parent_settings = {
        "reward_program": parent_reward,
        "ppo_config": parent_ppo,
        "opponent_mix": asdict(parent.opponent_mix),
    }
    candidate_settings = {
        "reward_program": candidate_reward,
        "ppo_config": candidate_ppo,
        "opponent_mix": asdict(candidate.opponent_mix),
    }
    before = _flatten_settings(parent_settings)
    after = _flatten_settings(candidate_settings)
    return [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    ]


_AST_DESCRIPTOR_CACHE: dict[str, tuple[Counter[str], Counter[str], tuple[float, ...], int]] = {}


def candidate_ast_descriptor(
    candidate: EvolutionCandidate,
) -> tuple[Counter[str], Counter[str], tuple[float, ...], int]:
    """Return an O(N) structural descriptor; numeric values never alter shape hashes."""
    cached = _AST_DESCRIPTOR_CACHE.get(candidate.candidate_id)
    if cached is not None:
        return cached
    subtrees: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    numbers: list[float] = []
    node_count = 0

    def visit(value: object) -> str:
        nonlocal node_count
        if isinstance(value, Mapping):
            node_count += 1
            op = str(value.get("op", "object"))
            tokens[f"op:{op}"] += 1
            if op in {"metric", "derived", "sum"}:
                tokens[f"ref:{value.get('name')}"] += 1
            children = []
            for name in sorted(value):
                item = value[name]
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    numbers.append(float(item))
                    children.append(f"{name}:#")
                else:
                    children.append(f"{name}:{visit(item)}")
            shape = hashlib.sha256((op + "|" + "|".join(children)).encode()).hexdigest()[:16]
            subtrees[shape] += 1
            return shape
        if isinstance(value, list):
            return "[" + ",".join(visit(item) for item in value) + "]"
        return str(value)

    visit(candidate.reward_program.to_dict())
    descriptor = subtrees, tokens, tuple(numbers), node_count
    _AST_DESCRIPTOR_CACHE[candidate.candidate_id] = descriptor
    return descriptor


def approximate_ast_distance(left: EvolutionCandidate, right: EvolutionCandidate) -> float:
    left_subtrees, left_tokens, left_numbers, _ = candidate_ast_descriptor(left)
    right_subtrees, right_tokens, right_numbers, _ = candidate_ast_descriptor(right)

    def multiset_jaccard(first: Counter[str], second: Counter[str]) -> float:
        union = sum((first | second).values())
        return 0.0 if union == 0 else 1.0 - sum((first & second).values()) / union

    names = left_tokens.keys() | right_tokens.keys()
    dot = sum(left_tokens[name] * right_tokens[name] for name in names)
    left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
    right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
    cosine_distance = 0.0 if left_norm == right_norm == 0 else 1.0 - dot / max(left_norm * right_norm, 1e-12)
    count = max(len(left_numbers), len(right_numbers), 1)
    padded_left = (*left_numbers, *((0.0,) * (count - len(left_numbers))))
    padded_right = (*right_numbers, *((0.0,) * (count - len(right_numbers))))
    numeric_distance = min(
        1.0,
        sum(abs(a - b) / max(abs(a), abs(b), 1.0) for a, b in zip(padded_left, padded_right)) / count,
    )
    return 0.5 * multiset_jaccard(left_subtrees, right_subtrees) + 0.3 * cosine_distance + 0.2 * numeric_distance


def validate_candidate_safety(parents: list[EvolutionCandidate], candidate: EvolutionCandidate) -> None:
    """Reject only candidates that cannot be executed safely or have broken lineage."""
    parent_by_id = {parent.candidate_id: parent for parent in parents}
    if len(parent_by_id) != len(parents):
        raise ValueError("Duplicate parent ids are not allowed")
    if candidate.mutation_kind == "initial":
        if parents or candidate.primary_parent_id is not None or candidate.secondary_parent_ids:
            raise ValueError("Initial candidates cannot reference parents")
        return
    if candidate.mutation_kind == "restart" and candidate.primary_parent_id is None:
        if candidate.secondary_parent_ids:
            raise ValueError("A base restart cannot reference secondary parents")
        return
    if not parents or candidate.primary_parent_id not in parent_by_id:
        raise ValueError("A non-restart candidate requires an existing primary parent")
    if len(set(candidate.secondary_parent_ids)) != len(candidate.secondary_parent_ids):
        raise ValueError("secondary_parent_ids must be unique")
    if any(parent_id not in parent_by_id for parent_id in candidate.secondary_parent_ids):
        raise ValueError("secondary_parent_ids contains an unknown parent")


def _serialized_counter(items: object) -> Counter[str]:
    return Counter(json.dumps(asdict(item), sort_keys=True, separators=(",", ":")) for item in items)


def _active_ppo_settings(config: PPOConfig) -> dict[str, object]:
    return asdict(config)


def _is_parameter_mutation(primary: EvolutionCandidate, candidate: EvolutionCandidate) -> bool:
    changes = candidate_setting_changes(primary, candidate)
    parent_subtrees, parent_tokens, _, _ = candidate_ast_descriptor(primary)
    child_subtrees, child_tokens, _, _ = candidate_ast_descriptor(candidate)
    return (
        1 <= len(changes) <= 2
        and parent_subtrees == child_subtrees
        and parent_tokens == child_tokens
        and all(
            isinstance(change["before"], (int, float))
            and not isinstance(change["before"], bool)
            and isinstance(change["after"], (int, float))
            and not isinstance(change["after"], bool)
            for change in changes
        )
    )


def _is_feature_mutation(primary: EvolutionCandidate, candidate: EvolutionCandidate) -> bool:
    parent_components = _serialized_counter(primary.reward_program.components)
    child_components = _serialized_counter(candidate.reward_program.components)
    added = list((child_components - parent_components).elements())
    removed = list((parent_components - child_components).elements())
    direct_addition = not added or json.loads(added[0]).get("expression", {}).get("op") == "metric"
    return (
        len(added) + len(removed) == 1
        and direct_addition
        and primary.reward_program.derived_metrics == candidate.reward_program.derived_metrics
        and primary.reward_program.reward_scale == candidate.reward_program.reward_scale
        and primary.reward_program.gamma == candidate.reward_program.gamma
        and _active_ppo_settings(primary.ppo_config) == _active_ppo_settings(candidate.ppo_config)
        and primary.opponent_mix == candidate.opponent_mix
    )


def _is_local_structural_mutation(primary: EvolutionCandidate, candidate: EvolutionCandidate) -> bool:
    if len(primary.reward_program.components) != len(candidate.reward_program.components):
        return False
    changed_components = [
        (before, after)
        for before, after in zip(primary.reward_program.components, candidate.reward_program.components)
        if before != after
    ]
    _, _, _, parent_nodes = candidate_ast_descriptor(primary)
    _, _, _, child_nodes = candidate_ast_descriptor(candidate)
    return (
        len(changed_components) == 1
        and abs(child_nodes - parent_nodes) <= 8
        and all(before.name == after.name and before.weight == after.weight for before, after in changed_components)
        and primary.reward_program.derived_metrics == candidate.reward_program.derived_metrics
        and primary.reward_program.reward_scale == candidate.reward_program.reward_scale
        and primary.reward_program.gamma == candidate.reward_program.gamma
        and _active_ppo_settings(primary.ppo_config) == _active_ppo_settings(candidate.ppo_config)
        and primary.opponent_mix == candidate.opponent_mix
    )


def _is_pure_crossover(
    primary: EvolutionCandidate,
    secondary: list[EvolutionCandidate],
    candidate: EvolutionCandidate,
) -> bool:
    if not secondary:
        return False
    primary_components = set(_serialized_counter(primary.reward_program.components))
    primary_derived = set(_serialized_counter(primary.reward_program.derived_metrics))
    secondary_components = {
        item for parent in secondary for item in _serialized_counter(parent.reward_program.components)
    }
    secondary_derived = {
        item for parent in secondary for item in _serialized_counter(parent.reward_program.derived_metrics)
    }
    candidate_components = set(_serialized_counter(candidate.reward_program.components))
    candidate_derived = set(_serialized_counter(candidate.reward_program.derived_metrics))
    secondary_contribution = bool(
        candidate_components & (secondary_components - primary_components)
        or candidate_derived & (secondary_derived - primary_derived)
    )
    return (
        secondary_contribution
        and candidate_components <= primary_components | secondary_components
        and candidate_derived <= primary_derived | secondary_derived
        and primary.reward_program != candidate.reward_program
        and primary.reward_program.reward_scale == candidate.reward_program.reward_scale
        and primary.reward_program.gamma == candidate.reward_program.gamma
        and _active_ppo_settings(primary.ppo_config) == _active_ppo_settings(candidate.ppo_config)
        and primary.opponent_mix == candidate.opponent_mix
    )


def canonicalize_candidate_proposal(
    proposal: Mapping[str, Any],
    parents: list[EvolutionCandidate],
    *,
    generation: int,
    island: int,
) -> tuple[dict[str, Any], EvolutionCandidate, dict[str, Any]]:
    """Normalize safe Codex output from its actual diff, retaining island lineage."""
    parent_ids = tuple(parent.candidate_id for parent in parents)
    raw = EvolutionCandidate.from_proposal(
        proposal,
        generation=generation,
        island=island,
        parent_ids=parent_ids,
    )
    validate_candidate_safety(parents, raw)
    parent_by_id = {parent.candidate_id: parent for parent in parents}
    primary = parent_by_id.get(raw.primary_parent_id)
    secondary = [parent_by_id[parent_id] for parent_id in raw.secondary_parent_ids if parent_id in parent_by_id]
    explicit_restart = raw.mutation_kind == "restart" and raw.primary_parent_id is None
    ast_distance = None if primary is None else approximate_ast_distance(primary, raw)
    if explicit_restart:
        effective_kind, scale = "restart", "restart"
    elif primary is not None and _is_pure_crossover(primary, secondary, raw):
        effective_kind, scale = "crossover", "recombined"
    elif primary is not None and _is_parameter_mutation(primary, raw):
        effective_kind, scale = "parameter", "numeric"
    elif primary is not None and _is_feature_mutation(primary, raw):
        effective_kind, scale = "feature_existing", "feature"
    elif primary is not None and _is_local_structural_mutation(primary, raw):
        effective_kind, scale = "structural", "local"
    else:
        effective_kind, scale = "structural", "large"

    canonical = copy.deepcopy(dict(proposal))
    canonical_ppo = dict(canonical["ppo_config"])
    canonical["ppo_config"] = canonical_ppo
    canonical["parameter_constraint_coefficient"] = 0.0
    canonical["mutation_kind"] = effective_kind
    if effective_kind == "restart":
        canonical.update(
            primary_parent_id=None,
            secondary_parent_ids=[],
            inheritance_mode="base",
            parameter_constraint_coefficient=0.0,
        )
    else:
        if primary is None:
            raise ValueError("A non-restart candidate requires an existing primary parent")
        canonical["primary_parent_id"] = primary.candidate_id
        canonical["inheritance_mode"] = "policy_value" if effective_kind == "parameter" else "policy"
        if effective_kind != "crossover":
            canonical["secondary_parent_ids"] = []

    manifest = dict(canonical.get("mutation_manifest", {}))
    manifest["changed_paths"] = []
    canonical["mutation_manifest"] = manifest
    provisional = EvolutionCandidate.from_proposal(
        canonical,
        generation=generation,
        island=island,
        parent_ids=parent_ids,
    )
    actual_changes = [] if effective_kind == "restart" else candidate_setting_changes(primary, provisional)
    manifest["changed_paths"] = [str(change["path"]) for change in actual_changes]
    canonical["mutation_manifest"] = manifest
    candidate = EvolutionCandidate.from_proposal(
        canonical,
        generation=generation,
        island=island,
        parent_ids=parent_ids,
    )
    validate_candidate_safety(parents, candidate)

    corrected_fields = [
        field
        for field in (
            "mutation_kind",
            "ppo_config",
            "primary_parent_id",
            "secondary_parent_ids",
            "inheritance_mode",
            "parameter_constraint_coefficient",
            "mutation_manifest",
        )
        if proposal.get(field) != canonical.get(field)
    ]
    deviations = []
    if raw.mutation_kind != effective_kind:
        deviations.append(f"declared {raw.mutation_kind} reclassified as {effective_kind}")
    if ast_distance is not None and not _I03_STRUCTURAL_DISTANCE_MIN <= ast_distance <= _I03_STRUCTURAL_DISTANCE_MAX:
        deviations.append(f"AST distance {ast_distance:.6f} is outside the advisory [0.20, 0.65]")
    deviations.extend(f"server corrected field: {field}" for field in corrected_fields)
    report = {
        "declared_mutation_kind": raw.mutation_kind,
        "effective_mutation_kind": effective_kind,
        "mutation_scale": scale,
        "ast_distance": ast_distance,
        "contract_deviations": deviations,
        "corrected_fields": corrected_fields,
        "changed_paths": manifest["changed_paths"],
    }
    return canonical, candidate, report


def validate_candidate_mutation(parents: list[EvolutionCandidate], candidate: EvolutionCandidate) -> None:
    """Backward-compatible public validator; semantic island contracts are advisory."""
    validate_candidate_safety(parents, candidate)


def _max_night_start_stranded_snapshot(
    fuel_snapshots: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    night_starts = [event for event in fuel_snapshots if bool(event.get("night_start", False))]
    return (
        max(night_starts, key=lambda event: float(event.get("stranded_fuel_fraction", 0.0)))
        if night_starts
        else None
    )


def summarize_diagnostic_events(metrics: Mapping[str, Any]) -> dict[str, object]:
    events: list[Mapping[str, Any]] = []
    training = metrics.get("training", {})
    if isinstance(training, Mapping):
        events.extend(event for event in training.get("diagnostic_events", ()) if isinstance(event, Mapping))
    evaluation = metrics.get("evaluation", {})
    if isinstance(evaluation, Mapping):
        for game in evaluation.get("games", ()):
            if isinstance(game, Mapping):
                events.extend(event for event in game.get("diagnostic_events", ()) if isinstance(event, Mapping))
    city_events = [dict(event) for event in events if event.get("event") == "city_destroyed_night_fuel"]
    fuel_snapshots = [dict(event) for event in events if event.get("event") == "night_fuel_snapshot"]
    max_stranded = _max_night_start_stranded_snapshot(fuel_snapshots)
    illegal_events = [dict(event) for event in events if event.get("event") == "illegal_action"]
    illegal_classes = Counter(
        str(event.get("action_class", event.get("action", "unknown"))) for event in illegal_events
    )
    return {
        "city_tile_loss_turns": sorted({int(event["turn"]) for event in city_events}),
        "night_fuel_shortage_turns": sorted({int(event["turn"]) for event in city_events}),
        "city_tiles_lost": sum(int(event.get("city_tiles_lost", 0)) for event in city_events),
        "city_event_count": len(city_events),
        "max_fuel_deficit": max((_event_deficit(event) for event in city_events), default=0.0),
        "min_night_fuel_margin": min(
            (float(event.get("min_city_fuel_margin", -1.0)) for event in fuel_snapshots),
            default=None,
        ),
        "max_night_start_stranded_fuel_fraction": (
            float(max_stranded.get("stranded_fuel_fraction", 0.0)) if max_stranded is not None else None
        ),
        "max_night_start_stranded_fuel_turn": (
            int(max_stranded.get("turn", 0)) if max_stranded is not None else None
        ),
        "last_night_city_zero_count": sum(
            int(event.get("turn", -1)) >= 350 and int(event.get("city_tiles", 0)) == 0
            for event in fuel_snapshots
        ),
        "city_loss_events": _representative_city_events(city_events),
        "illegal_action_turns": sorted({int(event["turn"]) for event in illegal_events}),
        "illegal_action_count": len(illegal_events),
        "illegal_action_class_counts": dict(illegal_classes),
        "illegal_action_events": illegal_events[:8],
    }


def compact_candidate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Remove high-volume raw traces after their Codex-facing summary is built."""
    compacted = dict(metrics)
    training = compacted.get("training")
    if isinstance(training, Mapping):
        compact_training = dict(training)
        training_events = compact_training.pop("diagnostic_events", ())
        compact_training["diagnostic_event_count"] = len(training_events)
        history = list(compact_training.get("history", ()))
        if history:
            compact_training["history_count"] = len(history)
            compact_training["history"] = [history[-1]]
        compacted["training"] = compact_training
    evaluation = compacted.get("evaluation")
    if isinstance(evaluation, Mapping):
        compact_evaluation = dict(evaluation)
        compact_games = []
        for game in compact_evaluation.get("games", ()):
            if not isinstance(game, Mapping):
                continue
            compact_game = dict(game)
            inference_seconds = [float(value) for value in compact_game.pop("candidate_inference_seconds", ())]
            if inference_seconds:
                ordered = sorted(inference_seconds)
                index = min(len(ordered) - 1, int(len(ordered) * 0.95))
                compact_game["candidate_inference_count"] = len(ordered)
                compact_game["candidate_inference_total_seconds"] = sum(ordered)
                compact_game["candidate_inference_p95_seconds"] = ordered[index]
            events = [event for event in compact_game.pop("diagnostic_events", ()) if isinstance(event, Mapping)]
            if events:
                city_events = [event for event in events if event.get("event") == "city_destroyed_night_fuel"]
                fuel_snapshots = [event for event in events if event.get("event") == "night_fuel_snapshot"]
                max_stranded = _max_night_start_stranded_snapshot(fuel_snapshots)
                illegal_events = [event for event in events if event.get("event") == "illegal_action"]
                compact_game["diagnostics"] = {
                    "city_tile_loss_turns": sorted({int(event["turn"]) for event in city_events}),
                    "city_tiles_lost": sum(int(event.get("city_tiles_lost", 0)) for event in city_events),
                    "city_loss_event_count": len(city_events),
                    "min_night_fuel_margin": min(
                        (float(event.get("min_city_fuel_margin", -1.0)) for event in fuel_snapshots),
                        default=None,
                    ),
                    "max_night_start_stranded_fuel_fraction": (
                        float(max_stranded.get("stranded_fuel_fraction", 0.0))
                        if max_stranded is not None
                        else None
                    ),
                    "max_night_start_stranded_fuel_turn": (
                        int(max_stranded.get("turn", 0)) if max_stranded is not None else None
                    ),
                    "illegal_action_turns": sorted({int(event["turn"]) for event in illegal_events}),
                    "illegal_action_count": len(illegal_events),
                }
            compact_games.append(compact_game)
        compact_evaluation["games"] = compact_games
        compacted["evaluation"] = compact_evaluation
    return compacted


def add_candidate_reflection(
    result: CandidateResult,
    candidate: EvolutionCandidate,
    candidates: Mapping[str, EvolutionCandidate],
    prior_results: list[CandidateResult],
) -> CandidateResult:
    parent_comparisons = []
    for parent_id in candidate.parent_ids:
        parent = candidates.get(parent_id)
        if parent is None:
            continue
        parent_result = next(
            (
                previous
                for previous in reversed(prior_results)
                if previous.candidate_id == parent_id
                and previous.stage == result.stage
                and previous.status == "completed"
            ),
            None,
        )
        improvement = None
        if parent_result is not None and result.status == "completed":
            improvement = {
                "score_rate_delta": result.score_rate - parent_result.score_rate,
                "teacher_score_rate_delta": result.teacher_score_rate - parent_result.teacher_score_rate,
                "duration_seconds_delta": result.duration_seconds - parent_result.duration_seconds,
            }
        parent_comparisons.append(
            {
                "parent_id": parent_id,
                "comparison_stage": result.stage if parent_result is not None else None,
                "changes": candidate_setting_changes(parent, candidate),
                "improvement": improvement,
            }
        )
    metrics = dict(result.metrics)
    metrics["reflection"] = {
        "diagnostics": summarize_diagnostic_events(metrics),
        "parent_comparisons": parent_comparisons,
    }
    return replace(result, metrics=compact_candidate_metrics(metrics))


class CodexCandidateGenerator:
    def __init__(
        self,
        *,
        repository: Path,
        run_dir: Path,
        executable: str = "codex",
        model: str | None = None,
        timeout_seconds: int = 900,
        validation_retries: int = 2,
    ) -> None:
        self.repository = repository
        self.run_dir = run_dir
        self.executable = executable
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.validation_retries = max(0, int(validation_retries))
        self.rules_context = lux_s1_rules_context()
        self.schema_path = run_dir / "codex-proposal-schema.json"
        self.schema_path.write_text(json.dumps(proposal_schema(), indent=2, sort_keys=True), encoding="utf-8")

    def output_path(self, generation: int, island: int) -> Path:
        return self.run_dir / f"codex-g{generation:02d}-i{island:02d}.json"

    def raw_output_path(self, generation: int, island: int) -> Path:
        return self.run_dir / f"codex-g{generation:02d}-i{island:02d}.raw.json"

    def prompt_path(self, generation: int, island: int, attempt: int) -> Path:
        return self.run_dir / f"codex-g{generation:02d}-i{island:02d}.attempt-{attempt:02d}.prompt.txt.gz"

    def metadata_path(self, generation: int, island: int) -> Path:
        return self.run_dir / f"codex-g{generation:02d}-i{island:02d}.meta.json"

    def error_path(self, generation: int, island: int) -> Path:
        return self.run_dir / f"codex-g{generation:02d}-i{island:02d}.error.json"

    @staticmethod
    def _validate_completed_process(completed: subprocess.CompletedProcess[str]) -> None:
        if completed.returncode == 0:
            return
        detail = completed.stderr[-4000:] or completed.stdout[-4000:]
        message = f"Codex candidate generation failed: {detail}"
        raise RuntimeError(message)

    def _archive_prior_failure(self, generation: int, island: int) -> None:
        error_path = self.error_path(generation, island)
        if not error_path.exists():
            return
        archive_dir = self.run_dir / "codex-rejections"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        paths = [error_path, self.output_path(generation, island), self.raw_output_path(generation, island)]
        paths.extend(self.run_dir.glob(f"codex-g{generation:02d}-i{island:02d}.attempt-*"))
        for path in paths:
            if path.exists():
                path.replace(archive_dir / f"{stamp}-{path.name}")

    @staticmethod
    def _repair_prompt(base_prompt: str, proposal: str, error: Exception) -> str:
        return (
            f"{base_prompt}\n\n"
            "VALIDATOR REPAIR REQUIRED. The previous proposal passed the JSON schema but was rejected by the "
            "deterministic server-side mutation validator. The validator is authoritative. Return a corrected full "
            "proposal JSON, not a patch. Do not merely change mutation_manifest.summary. AST-distance and island "
            "contract deviations are accepted and normalized; this repair request means the proposal is malformed, "
            "unsafe, or has invalid parent provenance.\n"
            f"Validator error: {type(error).__name__}: {error}\n"
            f"Rejected proposal:\n{proposal}"
        )

    def _failure_payload(
        self,
        *,
        parents: list[EvolutionCandidate],
        generation: int,
        island: int,
        attempt: int,
        started_at: float,
        completed: subprocess.CompletedProcess[str] | None,
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "status": "failed",
            "generation": generation,
            "island": island,
            "attempt": attempt,
            "parent_ids": [parent.candidate_id for parent in parents],
            "model": self.model,
            "executable": self.executable,
            "started_at": started_at,
            "failed_at": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "returncode": completed.returncode if completed is not None else None,
            "stdout": completed.stdout[-8000:] if completed is not None else "",
            "stderr": completed.stderr[-8000:] if completed is not None else "",
        }

    def generate(
        self,
        parents: list[EvolutionCandidate],
        results: list[CandidateResult],
        *,
        generation: int,
        island: int,
    ) -> EvolutionCandidate:
        base_prompt = build_codex_prompt(
            parents,
            results,
            island=island,
            generation=generation,
            rules_context=self.rules_context,
        )
        prompt = base_prompt
        output_path = self.output_path(generation, island)
        raw_output_path = self.raw_output_path(generation, island)
        self._archive_prior_failure(generation, island)
        rejected_attempts = []
        first_started_at = time.time()
        for attempt_index in range(self.validation_retries + 1):
            attempt = attempt_index + 1
            raw_output_path.unlink(missing_ok=True)
            prompt_bytes = prompt.encode("utf-8")
            prompt_path = self.prompt_path(generation, island, attempt)
            prompt_path.write_bytes(gzip.compress(prompt_bytes, mtime=0))
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(self.schema_path),
                "-o",
                str(raw_output_path),
            ]
            if self.model:
                command.extend(("--model", self.model))
            command.append("-")
            completed: subprocess.CompletedProcess[str] | None = None
            started_at = time.time()
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.repository,
                    check=False,
                    capture_output=True,
                    input=prompt,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                self._validate_completed_process(completed)
                proposal_text = raw_output_path.read_text(encoding="utf-8")
                proposal = json.loads(proposal_text)
                canonical_proposal, candidate, normalization = canonicalize_candidate_proposal(
                    proposal,
                    parents,
                    generation=generation,
                    island=island,
                )
                validate_candidate_mutation(parents, candidate)
                EvolutionStore.write_json(output_path, canonical_proposal)
            except Exception as error:
                payload = self._failure_payload(
                    parents=parents,
                    generation=generation,
                    island=island,
                    attempt=attempt,
                    started_at=started_at,
                    completed=completed,
                    error=error,
                )
                repairable = (
                    completed is not None
                    and completed.returncode == 0
                    and raw_output_path.exists()
                    and isinstance(error, (KeyError, TypeError, ValueError))
                )
                if repairable:
                    rejected_path = self.run_dir / (
                        f"codex-g{generation:02d}-i{island:02d}.attempt-{attempt:02d}.rejected.json"
                    )
                    rejected_path.write_bytes(raw_output_path.read_bytes())
                    attempt_error_path = self.run_dir / (
                        f"codex-g{generation:02d}-i{island:02d}.attempt-{attempt:02d}.error.json"
                    )
                    EvolutionStore.write_json(attempt_error_path, payload)
                    rejected_attempts.append(
                        {"attempt": attempt, "proposal_path": rejected_path.name, "error_path": attempt_error_path.name}
                    )
                    if attempt_index < self.validation_retries:
                        prompt = self._repair_prompt(base_prompt, raw_output_path.read_text(encoding="utf-8"), error)
                        continue
                EvolutionStore.write_json(self.error_path(generation, island), payload)
                raise
            break
        canonical_bytes = output_path.read_bytes()
        raw_bytes = raw_output_path.read_bytes()
        proposal_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        selected_result_ids = [
            f"{result.candidate_id}--{result.stage}--{result.status}"
            for result in select_codex_feedback_results(parents, results)
        ]
        EvolutionStore.write_json(
            self.metadata_path(generation, island),
            {
                "status": "accepted",
                "generation": generation,
                "island": island,
                "candidate_id": candidate.candidate_id,
                "parent_ids": [parent.candidate_id for parent in parents],
                "model": self.model,
                "executable": self.executable,
                "started_at": first_started_at,
                "completed_at": time.time(),
                "attempts": attempt,
                "rejected_attempts": rejected_attempts,
                "proposal_path": output_path.name,
                "proposal_sha256": proposal_sha256,
                "raw_proposal_path": raw_output_path.name,
                "raw_proposal_sha256": raw_sha256,
                "canonical_proposal_sha256": proposal_sha256,
                "normalization": normalization,
                "prompt_path": prompt_path.name,
                "prompt_bytes": len(prompt_bytes),
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
                "selected_result_ids": selected_result_ids,
                "context_schema_version": 1,
                "rules_source_url": self.rules_context["source_url"],
                "rules_summary_sha256": self.rules_context["summary_sha256"],
            },
        )
        return candidate


def initial_candidate(*, island: int, seed: int) -> EvolutionCandidate:
    rng = random.Random(seed + island)
    base_ppo = PPOConfig()
    reward = default_reward_program().to_dict()
    for component in reward["components"]:
        component["weight"] *= rng.uniform(0.85, 1.15)
    proposal = {
        "reward_program": reward,
        "ppo_config": asdict(base_ppo),
        "opponent_mix": asdict(OpponentMix()),
        "mutation_kind": "initial",
        "primary_parent_id": None,
        "secondary_parent_ids": [],
        "inheritance_mode": "base",
        "mutation_manifest": {"changed_paths": [], "summary": "Seeded bounded initialization"},
        "parameter_constraint_coefficient": 0.0,
        "rationale": "Bounded human initialization for the first island population.",
    }
    return EvolutionCandidate.from_proposal(proposal, generation=0, island=island, parent_ids=())


def _replace_random_expression_leaf(expression: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    result = copy.deepcopy(dict(expression))
    leaves: list[dict[str, Any]] = []

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        children = [value for value in node.values() if isinstance(value, dict) and "op" in value]
        if not children:
            leaves.append(node)
            return
        for child in children:
            visit(child)

    visit(result)
    target = rng.choice(leaves or [result])
    target.clear()
    target.update({"op": "metric", "name": rng.choice(sorted(DIRECT_REWARD_METRIC_NAMES))})
    return result


def _mutate_large_structural_reward(parent: EvolutionCandidate, rng: random.Random) -> dict[str, Any] | None:
    for attempt in range(_I03_STRUCTURAL_ATTEMPTS):
        reward = copy.deepcopy(parent.reward_program.to_dict())
        reward.update({"version": 3, "terminal_potential_zero": True})
        reward.setdefault("derived_metrics", [])
        for _ in range(rng.randint(2, 4)):
            index = rng.randrange(len(reward["components"]))
            expression = reward["components"][index]["expression"]
            operation = rng.choice(("wrap", "binary", "gate", "leaf"))
            if operation == "wrap":
                if expression.get("op") in _STRUCTURAL_WRAPPERS:
                    reward["components"][index]["expression"] = expression["value"]
                else:
                    reward["components"][index]["expression"] = {
                        "op": rng.choice(sorted(_STRUCTURAL_WRAPPERS)),
                        "value": expression,
                    }
            elif operation == "binary":
                reward["components"][index]["expression"] = {
                    "op": rng.choice(sorted(_STRUCTURAL_BINARY_OPS)),
                    "left": expression,
                    "right": {"op": "metric", "name": rng.choice(sorted(DIRECT_REWARD_METRIC_NAMES))},
                }
            elif operation == "gate":
                alternate = {"op": "metric", "name": rng.choice(sorted(DIRECT_REWARD_METRIC_NAMES))}
                original_is_true = bool(rng.getrandbits(1))
                reward["components"][index]["expression"] = {
                    "op": "gate",
                    "condition": {
                        "op": "metric",
                        "name": rng.choice(("night", "own_city_tiles_at_risk", "own_night_fuel_deficit")),
                    },
                    "when_true": expression if original_is_true else alternate,
                    "when_false": alternate if original_is_true else expression,
                }
            else:
                reward["components"][index]["expression"] = _replace_random_expression_leaf(expression, rng)
        try:
            program = RewardProgram.from_dict(reward)
        except (KeyError, TypeError, ValueError):
            continue
        digest = hashlib.sha256(json.dumps(reward, sort_keys=True).encode()).hexdigest()[:16]
        temporary = replace(parent, candidate_id=f"structural-{attempt}-{digest}", reward_program=program)
        distance = approximate_ast_distance(parent, temporary)
        if _I03_STRUCTURAL_DISTANCE_MIN <= distance <= _I03_STRUCTURAL_DISTANCE_MAX:
            return reward
    return None


def _crossover_reward(
    parent: EvolutionCandidate,
    secondary_parents: tuple[EvolutionCandidate, ...],
    rng: random.Random,
) -> tuple[dict[str, Any], str] | None:
    reward = copy.deepcopy(parent.reward_program.to_dict())
    reward.update({"version": 3, "terminal_potential_zero": True})
    reward.setdefault("derived_metrics", [])
    primary_components = {
        json.dumps(component, sort_keys=True, separators=(",", ":")) for component in reward["components"]
    }
    options = [
        (donor, copy.deepcopy(component))
        for donor in secondary_parents
        for component in donor.reward_program.to_dict()["components"]
        if json.dumps(component, sort_keys=True, separators=(",", ":")) not in primary_components
    ]
    rng.shuffle(options)
    for donor, component in options:
        candidate_reward = copy.deepcopy(reward)
        donor_reward = donor.reward_program.to_dict()
        derived_name = component["expression"].get("name") if component["expression"].get("op") == "derived" else None
        if derived_name is not None:
            donor_metric = next(
                (metric for metric in donor_reward.get("derived_metrics", ()) if metric["name"] == derived_name),
                None,
            )
            existing_metric = next(
                (metric for metric in candidate_reward["derived_metrics"] if metric["name"] == derived_name),
                None,
            )
            if donor_metric is None or (existing_metric is not None and existing_metric != donor_metric):
                continue
            if existing_metric is None:
                if len(candidate_reward["derived_metrics"]) >= 16:
                    continue
                candidate_reward["derived_metrics"].append(copy.deepcopy(donor_metric))
        replacement_indices = [
            index
            for index, existing in enumerate(candidate_reward["components"])
            if all(
                other_index == index or other["name"] != component["name"]
                for other_index, other in enumerate(candidate_reward["components"])
            )
        ]
        if not replacement_indices:
            continue
        candidate_reward["components"][rng.choice(replacement_indices)] = component
        try:
            RewardProgram.from_dict(candidate_reward)
        except (KeyError, TypeError, ValueError):
            continue
        return candidate_reward, donor.candidate_id
    return None


def _restart_reward(rng: random.Random) -> dict[str, Any]:
    reward = default_reward_program().to_dict()
    reward.update({"version": 3, "derived_metrics": [], "terminal_potential_zero": True})
    for component in reward["components"]:
        component["weight"] *= rng.uniform(0.5, 2.0)
    return reward


def mutate_candidate(
    parent: EvolutionCandidate,
    *,
    generation: int,
    island: int,
    seed: int,
    secondary_parents: tuple[EvolutionCandidate, ...] = (),
    stagnated: bool = False,
) -> EvolutionCandidate:
    rng = random.Random(seed)
    reward = parent.reward_program.to_dict()
    reward["version"] = 3
    reward["terminal_potential_zero"] = True
    reward.setdefault("derived_metrics", [])
    ppo = asdict(parent.ppo_config)

    opponent = asdict(parent.opponent_mix)
    changed_paths: list[str] = []
    selected_secondary_ids: list[str] = []
    kind = "parameter"
    inheritance = "policy_value"
    constraint = 0.0

    if island == 0:
        if rng.random() < 0.6:
            index = rng.randrange(len(reward["components"]))
            component = reward["components"][index]
            component["weight"] = max(
                -5.0,
                min(5.0, float(component["weight"]) * math.exp(rng.gauss(0.0, 0.08))),
            )
            changed_paths.append(f"reward_program.components[{index}].weight")
        else:
            selected = rng.choice(("learning_rate", "entropy_coefficient", "bc_coefficient"))
            ppo[selected] = float(ppo[selected]) * math.exp(rng.gauss(0.0, 0.12))
            changed_paths.append(f"ppo_config.{selected}")
    elif island == 1:
        kind = "structural"
        inheritance = "policy"
        index = rng.randrange(len(reward["components"]))
        expression = reward["components"][index]["expression"]
        if expression.get("op") in {"abs", "neg", "tanh", "exp_decay", "log1p_abs", "square"}:
            reward["components"][index]["expression"] = expression["value"]
        else:
            reward["components"][index]["expression"] = {"op": rng.choice(("tanh", "clip")), "value": expression}
            if reward["components"][index]["expression"]["op"] == "clip":
                reward["components"][index]["expression"].update({"low": -1.0, "high": 1.0})
        changed_paths.append(f"reward_program.components[{index}].expression")
    elif island == 2:
        inheritance = "policy"
        constraint = 0.0
        used = {
            component["expression"].get("name")
            for component in reward["components"]
            if component["expression"].get("op") == "metric"
        }
        existing_metric_indices = [
            index
            for index, component in enumerate(reward["components"])
            if component["expression"].get("op") == "metric"
        ]
        component_names = {str(component["name"]) for component in reward["components"]}
        available = sorted(
            metric for metric in DIRECT_REWARD_METRIC_NAMES - used if f"feature_{metric}" not in component_names
        )
        can_delete = bool(existing_metric_indices) and len(reward["components"]) > 1
        can_add = bool(available) and len(reward["components"]) < 16
        if can_delete and (not can_add or rng.random() < 0.35):
            kind = "feature_existing"
            reward["components"].pop(rng.choice(existing_metric_indices))
            changed_paths.append("reward_program.components")
        elif can_add:
            kind = "feature_existing"
            metric = rng.choice(available)
            magnitude = rng.uniform(0.1, 0.5)
            reward["components"].append(
                {
                    "name": f"feature_{metric}",
                    "expression": {"op": "metric", "name": metric},
                    "weight": -magnitude if metric in LOWER_IS_BETTER_METRIC_NAMES else magnitude,
                }
            )
            changed_paths.append("reward_program.components")
        else:
            raise ValueError("Island-2 fallback cannot add or delete a direct metric component")
    else:
        inheritance = "policy"
        draw = rng.random()
        restart_threshold = 0.4 if stagnated else 0.2
        crossover_threshold = restart_threshold + (0.2 if stagnated else 0.3)
        if draw < restart_threshold:
            kind = "restart"
            inheritance = "base"
            constraint = 0.0
            reward = _restart_reward(rng)
            changed_paths.append("reward_program")
        elif draw < crossover_threshold and secondary_parents:
            kind = "crossover"
            constraint = 0.0
            crossed = _crossover_reward(parent, secondary_parents, rng)
            if crossed is not None:
                reward, donor_id = crossed
                selected_secondary_ids = [donor_id]
                changed_paths.extend(("reward_program.derived_metrics", "reward_program.components"))
            else:
                kind = "structural"
                constraint = 0.0
                reward = _mutate_large_structural_reward(parent, rng)
        else:
            kind = "structural"
            reward = _mutate_large_structural_reward(parent, rng)
        if kind == "structural":
            if reward is None:
                crossed = _crossover_reward(parent, secondary_parents, rng)
                if crossed is not None:
                    kind = "crossover"
                    constraint = 0.0
                    reward, donor_id = crossed
                    selected_secondary_ids = [donor_id]
                    changed_paths.extend(("reward_program.derived_metrics", "reward_program.components"))
                else:
                    kind = "restart"
                    inheritance = "base"
                    constraint = 0.0
                    reward = _restart_reward(rng)
                    changed_paths.append("reward_program")
            else:
                changed_paths.append("reward_program.components")
    parent_ids = (parent.candidate_id, *(item.candidate_id for item in secondary_parents))
    proposal = {
        "reward_program": reward,
        "ppo_config": ppo,
        "opponent_mix": opponent,
        "mutation_kind": kind,
        "primary_parent_id": parent.candidate_id if kind != "restart" else None,
        "secondary_parent_ids": selected_secondary_ids,
        "inheritance_mode": inheritance,
        "mutation_manifest": {
            "changed_paths": changed_paths,
            "summary": f"Deterministic island-{island} {kind} fallback",
        },
        "parameter_constraint_coefficient": constraint,
        "rationale": f"Deterministic island-{island} fallback mutation ({kind}).",
    }
    return EvolutionCandidate.from_proposal(
        proposal,
        generation=generation,
        island=island,
        parent_ids=parent_ids,
    )


def select_elites(
    candidates: Mapping[str, EvolutionCandidate],
    results: list[CandidateResult],
    *,
    count: int,
) -> list[EvolutionCandidate]:
    def stage_priority(stage: str) -> int:
        if stage.startswith("final-"):
            return 3
        if stage.startswith("medium-"):
            return 2
        return 1

    latest: dict[str, CandidateResult] = {}
    for result in results:
        previous = latest.get(result.candidate_id)
        if previous is None or stage_priority(result.stage) >= stage_priority(previous.stage):
            latest[result.candidate_id] = result
    ranked = sorted(
        (result for result in latest.values() if result.candidate_id in candidates),
        key=lambda result: result.fitness,
        reverse=True,
    )
    return [candidates[result.candidate_id] for result in ranked[:count]]
