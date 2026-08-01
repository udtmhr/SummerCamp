from __future__ import annotations

# ruff: noqa: EM102, S311, S603, TC003
import hashlib
import json
import math
import random
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from luxai2021.rl.ppo import PPOConfig
from luxai2021.rl.reward import METRIC_NAMES, RewardProgram, default_reward_program

EVOLUTION_SCHEMA_VERSION = 1
OPPONENT_KEYS = ("self_base", "other_base", "teacher", "snapshot")


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvolutionCandidate:
        if int(value.get("schema_version", 0)) != EVOLUTION_SCHEMA_VERSION:
            raise ValueError("Unsupported evolution candidate schema")
        candidate = cls.from_proposal(
            value,
            generation=int(value["generation"]),
            island=int(value["island"]),
            parent_ids=tuple(value.get("parent_ids", ())),
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
    ) -> EvolutionCandidate:
        reward = RewardProgram.from_dict(proposal["reward_program"])
        ppo = PPOConfig(**proposal["ppo_config"])
        opponent = OpponentMix(**proposal["opponent_mix"])
        rationale = str(proposal.get("rationale", ""))[:4000]
        canonical = json.dumps(
            {
                "generation": generation,
                "island": island,
                "parents": parent_ids,
                "reward": reward.to_dict(),
                "ppo": asdict(ppo),
                "opponent": asdict(opponent),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        candidate_id = f"g{generation:02d}-i{island:02d}-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"
        return cls(candidate_id, generation, island, parent_ids, reward, ppo, opponent, rationale)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVOLUTION_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "island": self.island,
            "parent_ids": list(self.parent_ids),
            "reward_program": self.reward_program.to_dict(),
            "ppo_config": asdict(self.ppo_config),
            "opponent_mix": asdict(self.opponent_mix),
            "rationale": self.rationale,
        }


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
        valid = 1.0 if self.status == "completed" else 0.0
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
        )
        if job.base_name not in {"unet", "resattn8"} or job.seconds < 0 or job.eval_seeds < 1:
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
        payload = json.loads(claimed_path.read_text(encoding="utf-8"))
        payload.update({"result_status": result.status, "completed_at": time.time()})
        job = EvolutionJob.from_dict(payload)
        completed = self.completed_dir / f"{job.job_id}.json"
        EvolutionStore.write_json(completed, payload)
        claimed_path.unlink(missing_ok=True)

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
    expression: dict[str, Any] = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": [
                    "constant",
                    "metric",
                    "abs",
                    "neg",
                    "tanh",
                    "add",
                    "sub",
                    "mul",
                    "safe_div",
                    "min",
                    "max",
                    "clip",
                    "gate",
                ],
            },
            "name": {"type": "string", "enum": sorted(METRIC_NAMES)},
            "value": {"oneOf": [{"type": "number"}, {"$ref": "#/$defs/expression"}]},
            "left": {"$ref": "#/$defs/expression"},
            "right": {"$ref": "#/$defs/expression"},
            "low": {"type": "number"},
            "high": {"type": "number"},
            "condition": {"$ref": "#/$defs/expression"},
            "when_true": {"$ref": "#/$defs/expression"},
            "when_false": {"$ref": "#/$defs/expression"},
        },
        "required": ["op"],
        "additionalProperties": False,
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
                    "version": {"const": 1},
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
                "required": ["version", "components", "reward_scale", "gamma"],
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
            "rationale": {"type": "string"},
        },
        "required": ["reward_program", "ppo_config", "opponent_mix", "rationale"],
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

Generation: {generation}
Island: {island}
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
    parent_settings = {
        "reward_program": parent.reward_program.to_dict(),
        "ppo_config": asdict(parent.ppo_config),
        "opponent_mix": asdict(parent.opponent_mix),
    }
    candidate_settings = {
        "reward_program": candidate.reward_program.to_dict(),
        "ppo_config": asdict(candidate.ppo_config),
        "opponent_mix": asdict(candidate.opponent_mix),
    }
    before = _flatten_settings(parent_settings)
    after = _flatten_settings(candidate_settings)
    return [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    ]


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
    return replace(result, metrics=metrics)


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
            message = completed.stderr[-4000:] or completed.stdout[-4000:]
            raise RuntimeError(f"Codex candidate generation failed: {message}")
        proposal = json.loads(output_path.read_text(encoding="utf-8"))
        return EvolutionCandidate.from_proposal(
            proposal,
            generation=generation,
            island=island,
            parent_ids=tuple(parent.candidate_id for parent in parents),
        )


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
        "rationale": "Bounded human initialization for the first island population.",
    }
    return EvolutionCandidate.from_proposal(proposal, generation=0, island=island, parent_ids=())


def mutate_candidate(
    parent: EvolutionCandidate,
    *,
    generation: int,
    island: int,
    seed: int,
) -> EvolutionCandidate:
    rng = random.Random(seed)
    reward = parent.reward_program.to_dict()
    component = rng.choice(reward["components"])
    component["weight"] = max(-5.0, min(5.0, float(component["weight"]) * math.exp(rng.gauss(0.0, 0.15))))
    reward["reward_scale"] = max(0.0, min(0.5, reward["reward_scale"] * math.exp(rng.gauss(0.0, 0.1))))
    ppo = asdict(parent.ppo_config)
    numeric_choices = ("learning_rate", "entropy_coefficient", "kl_coefficient", "bc_coefficient")
    selected = rng.choice(numeric_choices)
    ppo[selected] = float(ppo[selected]) * math.exp(rng.gauss(0.0, 0.2))
    proposal = {
        "reward_program": reward,
        "ppo_config": ppo,
        "opponent_mix": asdict(parent.opponent_mix),
        "rationale": f"Deterministic fallback mutation of {component['name']} and {selected}.",
    }
    return EvolutionCandidate.from_proposal(
        proposal,
        generation=generation,
        island=island,
        parent_ids=(parent.candidate_id,),
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
