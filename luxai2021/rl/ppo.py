from __future__ import annotations

# ruff: noqa: EM102, PLC0415, PLR0913, PLR0915, TC001, TC003
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
) -> EpisodeTrajectory:
    rollout_agent = RolloutAgent(actor_critic, device=device)
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
                for local_index, (global_index, record) in enumerate(zip(indices, batch_records)):
                    advantage = advantages[global_index].to(self.device)
                    for decision in record.decisions:
                        y, x = decision.position
                        distribution = self._masked_distribution(
                            output[decision.entity][local_index, :, y, x],
                            decision.legal_mask,
                        )
                        reference_distribution = self._masked_distribution(
                            reference_output[decision.entity][local_index, :, y, x],
                            decision.legal_mask,
                        )
                        action = torch.tensor(decision.action, device=self.device)
                        log_probability = distribution.log_prob(action)
                        ratio = torch.exp(log_probability - decision.old_log_prob)
                        unclipped = ratio * advantage
                        clipped = ratio.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * advantage
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
                policy_loss = torch.stack(policy_losses).mean()
                entropy = torch.stack(entropies).mean()
                kl = torch.stack(kls).mean()
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
        metrics: dict[str, float],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "source_checkpoint": source_checkpoint,
                "policy": self.actor_critic.policy.state_dict(),
                "value_head": self.actor_critic.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ppo_config": asdict(self.config),
                "reward_program": reward_program.to_dict(),
                "update": update,
                "metrics": metrics,
            },
            path,
        )

    def load_training_checkpoint(
        self,
        path: Path,
        *,
        source_checkpoint: str,
        reward_program: RewardProgram,
    ) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("schema_version") != 1:
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
        return int(checkpoint["update"]) + 1
