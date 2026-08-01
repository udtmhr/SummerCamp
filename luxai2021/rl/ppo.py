from __future__ import annotations

# ruff: noqa: EM102, PLC0415, PLR0913, PLR0915, TC001, TC003
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.distributions import Categorical, kl_divergence

from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.imitation.masking import apply_legal_action_mask
from luxai2021.rl.metrics import metrics_from_game
from luxai2021.rl.policy import (
    EpisodeTrajectory,
    FullTurnActorCritic,
    RolloutAgent,
    TurnRecord,
    deterministic_outcome,
)
from luxai2021.rl.reward import RewardProgram

TRAINING_CHECKPOINT_SCHEMA_VERSION = 2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from luxai2021.env.agent import Agent


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coefficient: float = 0.005
    value_coefficient: float = 0.5
    kl_coefficient: float = 0.05
    bc_coefficient: float = 0.05
    gradient_clip: float = 1.0
    update_epochs: int = 2
    minibatch_turns: int = 32

    def __post_init__(self) -> None:
        bounds = {
            "learning_rate": (1e-7, 1e-3),
            "weight_decay": (0.0, 0.1),
            "gamma": (0.9, 1.0),
            "gae_lambda": (0.8, 1.0),
            "clip_range": (0.05, 0.4),
            "value_clip_range": (0.05, 0.5),
            "entropy_coefficient": (0.0, 0.1),
            "value_coefficient": (0.0, 2.0),
            "kl_coefficient": (0.0, 1.0),
            "bc_coefficient": (0.0, 1.0),
            "gradient_clip": (0.1, 10.0),
        }
        for name, (low, high) in bounds.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.update_epochs < 1 or self.minibatch_turns < 1:
            raise ValueError("PPO update sizes must be positive")


@dataclass(frozen=True)
class TrainingResumeState:
    next_update: int
    cumulative_decisions: int = 0
    cumulative_turns: int = 0
    cumulative_episodes: int = 0
    elapsed_seconds: float = 0.0
    metrics: dict[str, Any] | None = None
    python_random_state: object | None = None


def finish_episode(
    agent: RolloutAgent,
    game: object,
    reward_program: RewardProgram,
    *,
    seed: int,
    opponent: str,
) -> EpisodeTrajectory:
    final_metrics = metrics_from_game(game, agent.team)
    outcome = deterministic_outcome(game, agent.team)
    records = agent.records
    for index, record in enumerate(records):
        following = records[index + 1].metrics if index + 1 < len(records) else final_metrics
        terminal = outcome if index + 1 == len(records) else 0.0
        breakdown = reward_program.reward(record.metrics, following, terminal_outcome=terminal)
        record.reward = breakdown.total
        record.reward_components = dict(breakdown.components)
    diagnostic_events = [
        dict(event)
        for event in getattr(game, "diagnostic_events", ())
        if int(event.get("team", agent.team)) == agent.team
    ]
    return EpisodeTrajectory(agent.team, records, final_metrics, outcome, seed, opponent, diagnostic_events)


def calculate_gae(episodes: Iterable[EpisodeTrajectory], config: PPOConfig) -> list[TurnRecord]:
    records = []
    for episode in episodes:
        next_value = 0.0
        next_advantage = 0.0
        for record in reversed(episode.records):
            delta = record.reward + config.gamma * next_value - record.value
            record.advantage = delta + config.gamma * config.gae_lambda * next_advantage
            record.return_value = record.advantage + record.value
            next_value = record.value
            next_advantage = record.advantage
        records.extend(episode.records)
    return records


def collect_episode(
    actor_critic: FullTurnActorCritic,
    opponent_factory: Callable[[], Agent],
    reward_program: RewardProgram,
    *,
    device: torch.device,
    seed: int,
    opponent_name: str,
    max_turns: int | None = None,
    inference_backend: Callable[[Tensor], tuple[dict[str, Tensor], Tensor]] | None = None,
) -> EpisodeTrajectory:
    rollout_agent = RolloutAgent(
        actor_critic,
        device=device if inference_backend is None else "cpu",
        inference_backend=inference_backend,
    )
    opponent = opponent_factory()
    config = dict(LuxMatchConfigs_Default)
    config["parameters"] = dict(LuxMatchConfigs_Default["parameters"])
    if max_turns is not None:
        config["parameters"]["MAX_DAYS"] = max_turns
    config["seed"] = seed
    environment = LuxEnvironment(config, rollout_agent, opponent)
    with suppress(StopIteration):
        environment.reset(seed=seed)
    return finish_episode(rollout_agent, environment.game, reward_program, seed=seed, opponent=opponent_name)


def collect_episodes_batched(
    actor_critic: FullTurnActorCritic,
    episode_specs: list[tuple[Callable[[], Agent], int, str]],
    reward_program: RewardProgram,
    *,
    device: torch.device,
    inference_backend: Callable[[Tensor], tuple[dict[str, Tensor], Tensor]],
    max_turns: int | None = None,
) -> list[EpisodeTrajectory]:
    """Run one lockstep wave while a shared backend batches policy inference."""
    if not episode_specs:
        return []

    def collect(spec: tuple[Callable[[], Agent], int, str]) -> EpisodeTrajectory:
        opponent_factory, seed, opponent_name = spec
        return collect_episode(
            actor_critic,
            opponent_factory,
            reward_program,
            device=device,
            seed=seed,
            opponent_name=opponent_name,
            max_turns=max_turns,
            inference_backend=inference_backend,
        )

    with ThreadPoolExecutor(max_workers=len(episode_specs), thread_name_prefix="lux-rollout") as executor:
        return list(executor.map(collect, episode_specs))


class PPOTrainer:
    def __init__(
        self,
        actor_critic: FullTurnActorCritic,
        reference_policy: torch.nn.Module,
        config: PPOConfig,
        device: torch.device,
        *,
        bc_batch_provider: Callable[[], Mapping[str, Tensor]] | None = None,
    ) -> None:
        self.actor_critic = actor_critic
        self.reference_policy = reference_policy.eval().requires_grad_(requires_grad=False)
        self.config = config
        self.device = device
        self.bc_batch_provider = bc_batch_provider
        self.optimizer = torch.optim.AdamW(
            actor_critic.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )

    @staticmethod
    def _masked_distribution(logits: Tensor, mask: Tensor) -> Categorical:
        masked = apply_legal_action_mask(logits[None], mask.to(logits.device)[None])[0]
        return Categorical(logits=masked.float())

    def update(self, episodes: list[EpisodeTrajectory]) -> dict[str, float]:
        records = calculate_gae(episodes, self.config)
        if not records:
            raise ValueError("Cannot update PPO without rollout turns")
        advantages = torch.tensor([record.advantage for record in records], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        generator = torch.Generator().manual_seed(sum(episode.seed for episode in episodes))
        totals: dict[str, float] = {}
        update_count = 0
        for _ in range(self.config.update_epochs):
            order = torch.randperm(len(records), generator=generator).tolist()
            for start in range(0, len(order), self.config.minibatch_turns):
                indices = order[start : start + self.config.minibatch_turns]
                batch_records = [records[index] for index in indices]
                observations = torch.stack([record.observation for record in batch_records]).to(
                    self.device,
                    non_blocking=True,
                )
                output, values = self.actor_critic(observations)
                with torch.no_grad():
                    reference_output = self.reference_policy(observations)
                policy_losses = []
                entropies = []
                kls = []
                batch_advantages = advantages[indices].to(self.device)
                for entity in output:
                    entity_decisions = [
                        (local_index, decision)
                        for local_index, record in enumerate(batch_records)
                        for decision in record.decisions
                        if decision.entity == entity
                    ]
                    if not entity_decisions:
                        continue
                    local_indices = torch.tensor(
                        [local_index for local_index, _ in entity_decisions], device=self.device
                    )
                    ys = torch.tensor([decision.position[0] for _, decision in entity_decisions], device=self.device)
                    xs = torch.tensor([decision.position[1] for _, decision in entity_decisions], device=self.device)
                    logits = output[entity][local_indices, :, ys, xs]
                    reference_logits = reference_output[entity][local_indices, :, ys, xs]
                    masks = torch.stack([decision.legal_mask for _, decision in entity_decisions]).to(self.device)
                    distribution = Categorical(logits=apply_legal_action_mask(logits, masks).float())
                    reference_distribution = Categorical(
                        logits=apply_legal_action_mask(reference_logits, masks).float()
                    )
                    actions = torch.tensor(
                        [decision.action for _, decision in entity_decisions], device=self.device
                    )
                    old_log_probs = torch.tensor(
                        [decision.old_log_prob for _, decision in entity_decisions], device=self.device
                    )
                    selected_advantages = batch_advantages[local_indices]
                    ratios = torch.exp(distribution.log_prob(actions) - old_log_probs)
                    unclipped = ratios * selected_advantages
                    clipped = ratios.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * selected_advantages
                    policy_losses.append(-torch.minimum(unclipped, clipped))
                    entropies.append(distribution.entropy())
                    kls.append(kl_divergence(reference_distribution, distribution))
                if not policy_losses:
                    continue
                old_values = torch.tensor([record.value for record in batch_records], device=self.device)
                returns = torch.tensor([record.return_value for record in batch_records], device=self.device)
                clipped_values = old_values + (values - old_values).clamp(
                    -self.config.value_clip_range,
                    self.config.value_clip_range,
                )
                value_loss = (
                    0.5 * torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
                )
                policy_loss = torch.cat(policy_losses).mean()
                entropy = torch.cat(entropies).mean()
                kl = torch.cat(kls).mean()
                bc_loss = self._distillation_anchor_loss(values)
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.kl_coefficient * kl
                    + self.config.bc_coefficient * bc_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO produced a non-finite loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config.gradient_clip,
                )
                self.optimizer.step()
                batch_metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "kl": kl,
                    "bc_loss": bc_loss,
                    "gradient_norm": gradient_norm,
                }
                for name, value in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                update_count += 1
        if update_count == 0:
            raise ValueError("PPO rollout contained no actionable entities")
        result = {name: value / update_count for name, value in totals.items()}
        result.update(
            {
                "episodes": float(len(episodes)),
                "turns": float(len(records)),
                "decisions": float(sum(len(record.decisions) for record in records)),
                "score_rate": sum((episode.outcome + 1.0) * 0.5 for episode in episodes) / len(episodes),
            }
        )
        return result

    def _distillation_anchor_loss(self, zero_source: Tensor) -> Tensor:
        if self.bc_batch_provider is None or self.config.bc_coefficient == 0:
            return zero_source.sum() * 0.0
        from luxai2021.imitation.distillation import augment_distillation_batch, distillation_loss

        batch = {name: value.to(self.device, non_blocking=True) for name, value in self.bc_batch_provider().items()}
        batch = augment_distillation_batch(batch)
        output = self.actor_critic.policy(batch["observation"])
        return distillation_loss(
            output,
            batch,
            temperature=2.0,
            distill_weight=1.0,
            hard_label_weight=0.0,
        )["loss"]

    def save_training_checkpoint(
        self,
        path: Path,
        *,
        source_checkpoint: str,
        reward_program: RewardProgram,
        update: int,
        metrics: dict[str, Any],
        training_state: Mapping[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
                "source_checkpoint": source_checkpoint,
                "policy": self.actor_critic.policy.state_dict(),
                "value_head": self.actor_critic.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ppo_config": asdict(self.config),
                "reward_program": reward_program.to_dict(),
                "update": update,
                "metrics": metrics,
                "training_state": dict(training_state or {}),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )

    def load_training_state(
        self,
        path: Path,
        *,
        source_checkpoint: str,
        reward_program: RewardProgram,
        legacy_target_decisions: int | None = None,
        legacy_stage_seconds: int | None = None,
    ) -> TrainingResumeState:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        schema_version = int(checkpoint.get("schema_version", 1))
        if schema_version not in {1, TRAINING_CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError("Unsupported RL training checkpoint schema")
        if checkpoint.get("source_checkpoint") != source_checkpoint:
            raise ValueError("RL resume source checkpoint does not match")
        if checkpoint.get("reward_program") != reward_program.to_dict():
            raise ValueError("RL resume reward program does not match")
        if checkpoint.get("ppo_config") != asdict(self.config):
            raise ValueError("RL resume PPO configuration does not match")
        self.actor_critic.policy.load_state_dict(checkpoint["policy"])
        self.actor_critic.value_head.load_state_dict(checkpoint["value_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].to(device="cpu", dtype=torch.uint8))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            cuda_rng_states = [
                state.to(device="cpu", dtype=torch.uint8) for state in checkpoint["cuda_rng_state_all"]
            ]
            torch.cuda.set_rng_state_all(cuda_rng_states)

        metrics = dict(checkpoint.get("metrics", {}))
        state = (
            dict(checkpoint.get("training_state", {}))
            if schema_version >= TRAINING_CHECKPOINT_SCHEMA_VERSION
            else {}
        )
        elapsed_seconds = float(state.get("elapsed_seconds", metrics.get("elapsed_seconds", 0.0)))
        decisions = int(state.get("cumulative_decisions", metrics.get("cumulative_decisions", 0)))
        if schema_version == 1 and decisions == 0 and legacy_target_decisions and legacy_stage_seconds:
            fraction = min(1.0, max(0.0, elapsed_seconds / max(float(legacy_stage_seconds), 1.0)))
            decisions = round(legacy_target_decisions * fraction)
        return TrainingResumeState(
            next_update=int(checkpoint["update"]) + 1,
            cumulative_decisions=decisions,
            cumulative_turns=int(state.get("cumulative_turns", metrics.get("cumulative_turns", 0))),
            cumulative_episodes=int(state.get("cumulative_episodes", metrics.get("cumulative_episodes", 0))),
            elapsed_seconds=elapsed_seconds,
            metrics=metrics,
            python_random_state=state.get("python_random_state"),
        )

    def load_training_checkpoint(
        self,
        path: Path,
        *,
        source_checkpoint: str,
        reward_program: RewardProgram,
    ) -> int:
        return self.load_training_state(
            path,
            source_checkpoint=source_checkpoint,
            reward_program=reward_program,
        ).next_update
