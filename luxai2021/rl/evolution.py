from __future__ import annotations

# ruff: noqa: C901, EM102, PLR0912, PLR0913, PLR0915, PLR2004, S311, S603, TC003
import hashlib
import json
import math
import random
import subprocess
import time
from collections import Counter
from collections.abc import Mapping
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
    RewardProgram,
    default_reward_program,
)

EVOLUTION_SCHEMA_VERSION = 2
LEGACY_EVOLUTION_SCHEMA_VERSION = 1
OPPONENT_KEYS = ("self_base", "other_base", "teacher", "snapshot")
MUTATION_KINDS = frozenset(
    {"initial", "legacy", "parameter", "structural", "feature_existing", "feature_generated", "crossover", "restart"}
)
INHERITANCE_MODES = frozenset({"base", "policy", "policy_value"})
_STRUCTURAL_WRAPPERS = frozenset({"abs", "neg", "tanh", "exp_decay", "log1p_abs", "square"})


@dataclass(frozen=True)
class OpponentMix:
    self_base: float = 0.45
    other_base: float = 0.25
    teacher: float = 0.20
    snapshot: float = 0.10

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
    parameter_constraint_coefficient: float = 0.05

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionCandidate:
        schema_version = int(value.get("schema_version", 0))
        if schema_version not in {LEGACY_EVOLUTION_SCHEMA_VERSION, EVOLUTION_SCHEMA_VERSION}:
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
        ppo = PPOConfig(**proposal["ppo_config"])
        opponent = OpponentMix(**proposal["opponent_mix"])
        rationale = str(proposal.get("rationale", ""))[:4000]
        mutation_kind = str(proposal.get("mutation_kind", "legacy" if schema_version == 1 else "parameter"))
        primary = proposal.get("primary_parent_id")
        primary_parent_id = str(primary) if primary is not None else None
        secondary_parent_ids = tuple(str(item) for item in proposal.get("secondary_parent_ids", ()))
        inheritance_mode = str(proposal.get("inheritance_mode", "base" if not parent_ids else "policy"))
        mutation_manifest = dict(proposal.get("mutation_manifest", {}))
        parameter_constraint_coefficient = float(proposal.get("parameter_constraint_coefficient", 0.05))
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
        canonical_value: dict[str, Any] = {
            "generation": generation,
            "island": island,
            "parents": parent_ids,
            "reward": reward.to_dict(),
            "ppo": asdict(ppo),
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
        )

    def to_dict(self) -> dict[str, Any]:
        schema_version = LEGACY_EVOLUTION_SCHEMA_VERSION if self.mutation_kind == "legacy" else EVOLUTION_SCHEMA_VERSION
        value = {
            "schema_version": schema_version,
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "island": self.island,
            "parent_ids": list(self.parent_ids),
            "reward_program": self.reward_program.to_dict(),
            "ppo_config": asdict(self.ppo_config),
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
        return valid, self.score_rate, self.teacher_score_rate - min(self.kl, 1.0) * 0.01


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
        if any((directory / filename).exists() for directory in (self.pending_dir, self.completed_dir)):
            return
        if any(self.running_dir.glob(f"*--{filename}")):
            return
        EvolutionStore.write_json(self.pending_dir / filename, job.to_dict())

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
        field: {"type": "integer" if field in {"update_epochs", "minibatch_turns"} else "number"}
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
                    "version": {"type": "integer", "const": 2},
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
                },
                "required": ["version", "derived_metrics", "components", "reward_scale", "gamma"],
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
            "mutation_kind": {"type": "string", "enum": sorted(MUTATION_KINDS - {"legacy", "initial"})},
            "primary_parent_id": {"type": ["string", "null"]},
            "secondary_parent_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "inheritance_mode": {"type": "string", "enum": sorted(INHERITANCE_MODES)},
            "mutation_manifest": expression_object(
                {
                    "changed_paths": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
                    "summary": {"type": "string"},
                }
            ),
            "parameter_constraint_coefficient": {"type": "number", "minimum": 0.0, "maximum": 1.0},
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
) -> str:
    parent_payload = [parent.to_dict() for parent in parents]
    result_payload = [_codex_result_feedback(result) for result in results[-12:]]
    island_role = {
        0: "parameter: change only one or two numeric leaves; keep AST topology unchanged; inherit policy_value",
        1: "structural: make one local AST operator/subtree edit; inherit policy",
        2: "feature: add/delete one existing or generated Metric DSL feature; inherit policy",
        3: "diversity: large structural mutation, crossover, generated feature, or restart",
    }.get(island, "parameter")
    return f"""You are evolving a safe reward program and PPO configuration for Lux AI Challenge 2021.
Return exactly one proposal matching the supplied JSON schema.

Hard constraints:
- Keep the observation schema, first_place_flat_v1 actions, UNet, and ResAttn8 architectures unchanged.
- Reward expressions may use only schema operations and normalized metrics.
- Preserve terminal win/loss reward; design bounded potential shaping for city survival and match strength.
- PPO must remain close to the distilled reference policy and train on one GPU.
- Opponent weights must sum to exactly 1.0.
- Prefer one targeted change justified by evaluation feedback instead of unrelated novelty.
- Illegal-action events identify hard action-mask defects. Do not trade them against reward and never weaken an
  existing illegal-action mask; runtime masks may only become stricter.
- Use the reported city-loss/night-fuel turns and parent deltas to explain the proposed change.
- Emit reward_program version 2. Derived metrics must use only the safe Metric DSL in the schema.
- turn/night/cycle/turns_until_night/night_turns_remaining describe phase and should be used as gate conditions,
  not standalone objectives. Relative city-risk/loss/deficit metrics are oriented so larger is better. Absolute
  own_city_tiles_at_risk, own_night_fuel_deficit, own_city_tiles_lost, and own_night_fuel_shortage are lower-is-better.
- min_city_survival is the minimum over individual cities, while city_survival uses aggregate team fuel. Prefer the
  minimum/risk/deficit signals when feedback reports a local night-fuel collapse despite adequate total fuel.
- On island 2, odd generations use existing whitelist metrics and even generations use generated Metric DSL.
- A restart sets primary_parent_id to null, inheritance_mode to base, and parameter_constraint_coefficient to 0.
- mutation_kind, parent provenance, inheritance_mode, and changed_paths must truthfully describe the proposal.

Generation: {generation}
Island: {island}
Island role: {island_role}
Parents:
{json.dumps(parent_payload, indent=2, sort_keys=True)}

Recent evaluation and reward-reflection feedback:
{json.dumps(result_payload, indent=2, sort_keys=True)}
"""


def _codex_result_feedback(result: CandidateResult) -> dict[str, Any]:
    metrics = result.metrics
    training = metrics.get("training", {}) if isinstance(metrics, Mapping) else {}
    history = training.get("history", ()) if isinstance(training, Mapping) else ()
    evaluation = metrics.get("evaluation", {}) if isinstance(metrics, Mapping) else {}
    return {
        "candidate_id": result.candidate_id,
        "stage": result.stage,
        "status": result.status,
        "score_rate": result.score_rate,
        "teacher_score_rate": result.teacher_score_rate,
        "kl": result.kl,
        "duration_seconds": result.duration_seconds,
        "last_training_update": history[-1] if history else None,
        "league_totals": evaluation.get("totals") if isinstance(evaluation, Mapping) else None,
        "reflection": metrics.get("reflection") if isinstance(metrics, Mapping) else None,
        "error": result.error,
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
    parent_settings = {
        "reward_program": parent_reward,
        "ppo_config": asdict(parent.ppo_config),
        "opponent_mix": asdict(parent.opponent_mix),
        "parameter_constraint_coefficient": parent.parameter_constraint_coefficient,
    }
    candidate_settings = {
        "reward_program": candidate_reward,
        "ppo_config": asdict(candidate.ppo_config),
        "opponent_mix": asdict(candidate.opponent_mix),
        "parameter_constraint_coefficient": candidate.parameter_constraint_coefficient,
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


def validate_candidate_mutation(parents: list[EvolutionCandidate], candidate: EvolutionCandidate) -> None:
    """Reject Codex proposals that do not obey the declared island operator."""
    if not parents:
        if candidate.mutation_kind not in {"initial", "restart"}:
            raise ValueError("A parentless candidate must be initial or restart")
        return
    parent_by_id = {parent.candidate_id: parent for parent in parents}
    if candidate.mutation_kind != "restart" and candidate.primary_parent_id not in parent_by_id:
        raise ValueError("Non-restart mutation requires a valid primary parent")
    primary = parent_by_id.get(candidate.primary_parent_id, parents[0])
    changes = candidate_setting_changes(primary, candidate)
    reported = tuple(str(path) for path in candidate.mutation_manifest.get("changed_paths", ()))
    if changes and not reported:
        raise ValueError("Mutation manifest omitted changed_paths")
    for change in changes:
        path = str(change["path"])
        if not any(path.startswith(item) or item.startswith(path) for item in reported):
            raise ValueError(f"Mutation manifest does not cover changed path: {path}")
    _, _, _, parent_nodes = candidate_ast_descriptor(primary)
    _, _, _, child_nodes = candidate_ast_descriptor(candidate)
    expected = {0: {"parameter"}, 1: {"structural"}, 2: {"feature_existing", "feature_generated"}}
    if candidate.island in expected and candidate.mutation_kind not in expected[candidate.island]:
        raise ValueError(f"Island {candidate.island} does not allow {candidate.mutation_kind}")
    if candidate.island == 0:
        if child_nodes != parent_nodes or len(changes) > 2:
            raise ValueError("Parameter island may change at most two numeric leaves without changing AST topology")
        if any(
            not isinstance(change["before"], (int, float)) or not isinstance(change["after"], (int, float))
            for change in changes
        ):
            raise ValueError("Parameter mutation changed a non-numeric setting")
        if candidate.inheritance_mode != "policy_value":
            raise ValueError("Parameter mutations must inherit policy and value")
    elif candidate.island == 1:
        changed_components = [
            (before, after)
            for before, after in zip(primary.reward_program.components, candidate.reward_program.components)
            if before != after
        ]
        same_non_ast_settings = (
            len(primary.reward_program.components) == len(candidate.reward_program.components)
            and primary.reward_program.derived_metrics == candidate.reward_program.derived_metrics
            and primary.reward_program.reward_scale == candidate.reward_program.reward_scale
            and primary.reward_program.gamma == candidate.reward_program.gamma
            and primary.ppo_config == candidate.ppo_config
            and primary.opponent_mix == candidate.opponent_mix
            and all(before.name == after.name and before.weight == after.weight for before, after in changed_components)
        )
        if (
            abs(child_nodes - parent_nodes) > 8
            or candidate.inheritance_mode != "policy"
            or len(changed_components) != 1
            or not same_non_ast_settings
        ):
            raise ValueError("Structural island exceeded its local-edit or inheritance bound")
    elif candidate.island == 2:

        def serialize(item: object) -> str:
            return json.dumps(asdict(item), sort_keys=True, separators=(",", ":"))

        parent_components = Counter(serialize(item) for item in primary.reward_program.components)
        child_components = Counter(serialize(item) for item in candidate.reward_program.components)
        parent_derived = Counter(serialize(item) for item in primary.reward_program.derived_metrics)
        child_derived = Counter(serialize(item) for item in candidate.reward_program.derived_metrics)
        component_added = sum((child_components - parent_components).values())
        component_removed = sum((parent_components - child_components).values())
        derived_added = sum((child_derived - parent_derived).values())
        derived_removed = sum((parent_derived - child_derived).values())
        parity_valid = (
            candidate.generation % 2 == 1
            and candidate.mutation_kind == "feature_existing"
            and derived_added == derived_removed == 0
        ) or (
            candidate.generation % 2 == 0
            and candidate.mutation_kind == "feature_generated"
            and max(derived_added, derived_removed) == 1
        )
        if (
            max(component_added, component_removed) > 1
            or max(derived_added, derived_removed) > 1
            or component_added + component_removed == 0
            or candidate.inheritance_mode != "policy"
            or not parity_valid
        ):
            raise ValueError("Feature island may add/delete only one feature and one component")
    elif candidate.mutation_kind == "crossover":
        allowed_components = {
            json.dumps(asdict(component), sort_keys=True)
            for parent in parents
            for component in parent.reward_program.components
        }
        if any(
            json.dumps(asdict(component), sort_keys=True) not in allowed_components
            for component in candidate.reward_program.components
        ):
            raise ValueError("Crossover introduced a component not present in its parents")
        allowed_derived = {
            json.dumps(asdict(metric), sort_keys=True)
            for parent in parents
            for metric in parent.reward_program.derived_metrics
        }
        if any(
            json.dumps(asdict(metric), sort_keys=True) not in allowed_derived
            for metric in candidate.reward_program.derived_metrics
        ):
            raise ValueError("Crossover introduced a derived metric not present in its parents")
    if candidate.mutation_kind == "restart" and (
        candidate.inheritance_mode != "base" or candidate.parameter_constraint_coefficient != 0
    ):
        raise ValueError("Restart must use base inheritance with no parent parameter constraint")


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
    illegal_events = [dict(event) for event in events if event.get("event") == "illegal_action"]
    return {
        "city_tile_loss_turns": sorted({int(event["turn"]) for event in city_events}),
        "night_fuel_shortage_turns": sorted({int(event["turn"]) for event in city_events}),
        "city_tiles_lost": sum(int(event.get("city_tiles_lost", 0)) for event in city_events),
        "city_loss_events": city_events[:32],
        "illegal_action_turns": sorted({int(event["turn"]) for event in illegal_events}),
        "illegal_action_count": len(illegal_events),
        "illegal_action_events": illegal_events[:32],
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
                illegal_events = [event for event in events if event.get("event") == "illegal_action"]
                compact_game["diagnostics"] = {
                    "city_tile_loss_turns": sorted({int(event["turn"]) for event in city_events}),
                    "city_tiles_lost": sum(int(event.get("city_tiles_lost", 0)) for event in city_events),
                    "city_loss_event_count": len(city_events),
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
                "kl_delta": result.kl - parent_result.kl,
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
    ) -> None:
        self.repository = repository
        self.run_dir = run_dir
        self.executable = executable
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.schema_path = run_dir / "codex-proposal-schema.json"
        self.schema_path.write_text(json.dumps(proposal_schema(), indent=2, sort_keys=True), encoding="utf-8")

    def generate(
        self,
        parents: list[EvolutionCandidate],
        results: list[CandidateResult],
        *,
        generation: int,
        island: int,
    ) -> EvolutionCandidate:
        prompt = build_codex_prompt(parents, results, island=island, generation=generation)
        output_path = self.run_dir / f"codex-g{generation:02d}-i{island:02d}.json"
        command = [
            self.executable,
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(self.schema_path),
            "-o",
            str(output_path),
        ]
        if self.model:
            command.extend(("--model", self.model))
        command.append(prompt)
        completed = subprocess.run(
            command,
            cwd=self.repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            EvolutionStore.write_json(
                self.run_dir / f"codex-g{generation:02d}-i{island:02d}.error.json",
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-8000:],
                    "stderr": completed.stderr[-8000:],
                },
            )
            message = completed.stderr[-4000:] or completed.stdout[-4000:]
            raise RuntimeError(f"Codex candidate generation failed: {message}")
        proposal = json.loads(output_path.read_text(encoding="utf-8"))
        candidate = EvolutionCandidate.from_proposal(
            proposal,
            generation=generation,
            island=island,
            parent_ids=tuple(parent.candidate_id for parent in parents),
        )
        validate_candidate_mutation(parents, candidate)
        return candidate


def initial_candidate(*, island: int, seed: int) -> EvolutionCandidate:
    rng = random.Random(seed + island)
    base_ppo = PPOConfig()
    reward = default_reward_program().to_dict()
    reward["version"] = 2
    reward["derived_metrics"] = []
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
    reward["version"] = 2
    reward.setdefault("derived_metrics", [])
    ppo = asdict(parent.ppo_config)
    opponent = asdict(parent.opponent_mix)
    changed_paths: list[str] = []
    kind = "parameter"
    inheritance = "policy_value"
    constraint = parent.parameter_constraint_coefficient if parent.parameter_constraint_coefficient > 0.0 else 0.05

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
            selected = rng.choice(("learning_rate", "entropy_coefficient", "kl_coefficient", "bc_coefficient"))
            ppo[selected] = float(ppo[selected]) * math.exp(rng.gauss(0.0, 0.12))
            changed_paths.append(f"ppo_config.{selected}")
        if rng.random() < 0.25:
            constraint = max(0.0, min(1.0, constraint * math.exp(rng.gauss(0.0, 0.2))))
            changed_paths.append("parameter_constraint_coefficient")
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
        available = sorted(DIRECT_REWARD_METRIC_NAMES - used)
        if (
            generation % 2 == 1
            and existing_metric_indices
            and len(reward["components"]) > 1
            and (rng.random() < 0.35 or not available)
        ):
            kind = "feature_existing"
            reward["components"].pop(rng.choice(existing_metric_indices))
            changed_paths.append("reward_program.components")
        elif generation % 2 == 1 and available:
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
            kind = "feature_generated"
            generated_names = {item["name"] for item in reward["derived_metrics"]}
            removable = [
                (index, component["expression"]["name"])
                for index, component in enumerate(reward["components"])
                if component["expression"].get("op") == "derived"
                and component["expression"].get("name") in generated_names
            ]
            delete_generated = (
                removable
                and len(reward["components"]) > 1
                and (rng.random() < 0.35 or len(reward["derived_metrics"]) >= 16)
            )
            if delete_generated:
                component_index, derived_name = rng.choice(removable)
                reward["components"].pop(component_index)
                reward["derived_metrics"] = [item for item in reward["derived_metrics"] if item["name"] != derived_name]
            else:
                if len(reward["components"]) >= 16:
                    replacement_index = rng.randrange(len(reward["components"]))
                    reward["components"].pop(replacement_index)
                selector = rng.choice(sorted(METRIC_SELECTORS))
                name = f"generated_{generation}_{selector}_{seed & 0xFFFF:04x}"
                reward["derived_metrics"].append({"name": name, "expression": {"op": "count", "selector": selector}})
                reward["components"].append(
                    {
                        "name": name,
                        "expression": {"op": "derived", "name": name},
                        "weight": rng.uniform(0.1, 0.5),
                    }
                )
            changed_paths.extend(("reward_program.derived_metrics", "reward_program.components"))
    else:
        inheritance = "policy"
        draw = rng.random()
        restart_threshold = 0.4 if stagnated else 0.2
        crossover_threshold = restart_threshold + (0.2 if stagnated else 0.3)
        feature_threshold = crossover_threshold + 0.2
        if draw < restart_threshold:
            kind = "restart"
            inheritance = "base"
            constraint = 0.0
            reward = default_reward_program().to_dict()
            reward.update({"version": 2, "derived_metrics": []})
            for component in reward["components"]:
                component["weight"] *= rng.uniform(0.5, 2.0)
            changed_paths.append("reward_program")
        elif draw < crossover_threshold and secondary_parents:
            kind = "crossover"
            donor = rng.choice(secondary_parents)
            donor_reward = donor.reward_program.to_dict()
            donor_options = [
                component
                for component in donor_reward["components"]
                if component["expression"].get("op") != "derived"
                or any(metric["name"] == component["expression"].get("name") for metric in reward["derived_metrics"])
            ]
            donor_component = rng.choice(donor_options or donor_reward["components"])
            if donor_component["expression"].get("op") == "derived":
                derived_name = donor_component["expression"]["name"]
                if not any(metric["name"] == derived_name for metric in reward["derived_metrics"]):
                    donor_metric = next(
                        metric for metric in donor_reward["derived_metrics"] if metric["name"] == derived_name
                    )
                    if len(reward["derived_metrics"]) >= 16:
                        reward["derived_metrics"].pop()
                    reward["derived_metrics"].append(donor_metric)
                    changed_paths.append("reward_program.derived_metrics")
            reward["components"][rng.randrange(len(reward["components"]))] = donor_component
            changed_paths.append("reward_program.components")
        elif draw < feature_threshold:
            kind = "feature_generated"
            selector = rng.choice(sorted(METRIC_SELECTORS))
            target = rng.choice(sorted(METRIC_SELECTORS))
            name = f"diverse_{generation}_{selector}_{target}_{seed & 0xFFFF:04x}"
            reward["derived_metrics"].append(
                {"name": name, "expression": {"op": "distance", "source": selector, "target": target, "reduce": "min"}}
            )
            reward["components"].append(
                {"name": name, "expression": {"op": "derived", "name": name}, "weight": rng.uniform(-1.0, 1.0)}
            )
            changed_paths.extend(("reward_program.derived_metrics", "reward_program.components"))
        else:
            kind = "structural"
            for index in rng.sample(range(len(reward["components"])), k=min(2, len(reward["components"]))):
                expression = reward["components"][index]["expression"]
                if expression.get("op") in _STRUCTURAL_WRAPPERS:
                    reward["components"][index]["expression"] = expression["value"]
                else:
                    reward["components"][index]["expression"] = {
                        "op": rng.choice(sorted(_STRUCTURAL_WRAPPERS)),
                        "value": expression,
                    }
                changed_paths.append(f"reward_program.components[{index}].expression")
    if (
        constraint != parent.parameter_constraint_coefficient
        and "parameter_constraint_coefficient" not in changed_paths
    ):
        changed_paths.append("parameter_constraint_coefficient")
    parent_ids = (parent.candidate_id, *(item.candidate_id for item in secondary_parents))
    proposal = {
        "reward_program": reward,
        "ppo_config": ppo,
        "opponent_mix": opponent,
        "mutation_kind": kind,
        "primary_parent_id": parent.candidate_id if kind != "restart" else None,
        "secondary_parent_ids": [item.candidate_id for item in secondary_parents],
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
