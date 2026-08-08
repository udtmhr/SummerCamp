from __future__ import annotations

# ruff: noqa: C901, EM102, FBT003, PLC0415, PLR0912, PLR0913, PLR0915, PLR2004, TC001, TC003
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.distributions import Categorical

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

TRAINING_CHECKPOINT_SCHEMA_VERSION = 3
AUTO_ROLLOUT_BACKEND = "threaded"


def resolve_rollout_backend(requested: str) -> str:
    if requested not in {"auto", "lockstep", "threaded"}:
        raise ValueError(f"Unsupported rollout backend: {requested}")
    return AUTO_ROLLOUT_BACKEND if requested == "auto" else requested

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
    kl_coefficient: float = 0.0
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
    curriculum_progress_decisions: int = 0
    joint_update: int = 0
    source_checkpoint: str | None = None
    source_checkpoint_mismatch: bool = False
    source_checkpoint_sha256: str | None = None
    source_checkpoint_sha256_mismatch: bool = False


def _checkpoint_cuda_rng_state(checkpoint: Mapping[str, Any], device_index: int) -> torch.Tensor | None:
    state = checkpoint.get("cuda_rng_state")
    if isinstance(state, torch.Tensor):
        return state
    legacy_states = checkpoint.get("cuda_rng_state_all")
    if not isinstance(legacy_states, (list, tuple)) or not legacy_states:
        return None
    if 0 <= device_index < len(legacy_states):
        return legacy_states[device_index]
    return legacy_states[0]


def finish_episode(
    agent: RolloutAgent,
    game: object,
    reward_program: RewardProgram,
    *,
    seed: int,
    opponent: str,
) -> EpisodeTrajectory:
    reward_started = perf_counter()
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
    timings = dict(agent.timing_seconds)
    timings["reward_finalize"] = perf_counter() - reward_started
    timings.update(getattr(game, "performance_seconds", {}))
    return EpisodeTrajectory(agent.team, records, final_metrics, outcome, seed, opponent, diagnostic_events, timings)


def aggregate_episode_timings(episodes: Iterable[EpisodeTrajectory]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for episode in episodes:
        for name, seconds in episode.timings.items():
            totals[name] = totals.get(name, 0.0) + float(seconds)
    return totals


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


def apply_reward_program(episode: EpisodeTrajectory, reward_program: RewardProgram) -> None:
    """Recompute an already collected trajectory under a calibrated reward."""
    for index, record in enumerate(episode.records):
        following = episode.records[index + 1].metrics if index + 1 < len(episode.records) else episode.final_metrics
        terminal = episode.outcome if index + 1 == len(episode.records) else 0.0
        breakdown = reward_program.reward(record.metrics, following, terminal_outcome=terminal)
        record.reward = breakdown.total
        record.reward_components = dict(breakdown.components)


def _critic_examples(episodes: list[EpisodeTrajectory], gamma: float) -> tuple[Tensor, Tensor]:
    records: list[TurnRecord] = []
    returns: list[float] = []
    for episode in episodes:
        following_return = 0.0
        episode_returns = []
        for record in reversed(episode.records):
            following_return = record.reward + gamma * following_return
            episode_returns.append(following_return)
        records.extend(episode.records)
        returns.extend(reversed(episode_returns))
    if not records:
        raise ValueError("Critic calibration requires rollout turns")
    return torch.stack([record.observation for record in records]), torch.tensor(returns, dtype=torch.float32)


def value_head_calibration_loss(
    actor_critic: FullTurnActorCritic,
    episodes: list[EpisodeTrajectory],
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    """Measure inherited critic quality without modifying policy or value state."""
    observations, targets = _critic_examples(episodes, config.gamma)
    batch_size = max(1, config.minibatch_turns)
    total_loss = 0.0
    zero_loss = 0.0
    policy_was_training = actor_critic.policy.training
    value_was_training = actor_critic.value_head.training
    actor_critic.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(targets), batch_size):
                batch_targets = targets[start : start + batch_size].to(device)
                _, values = actor_critic(observations[start : start + batch_size].to(device))
                total_loss += float(torch.nn.functional.huber_loss(values.float(), batch_targets, reduction="sum"))
                zero_loss += float(
                    torch.nn.functional.huber_loss(torch.zeros_like(batch_targets), batch_targets, reduction="sum")
                )
    finally:
        actor_critic.policy.train(policy_was_training)
        actor_critic.value_head.train(value_was_training)
    loss = total_loss / len(targets)
    baseline = zero_loss / len(targets)
    threshold = max(0.5, 4.0 * baseline)
    return {
        "turns": float(len(targets)),
        "loss": loss,
        "zero_baseline_loss": baseline,
        "degradation_threshold": threshold,
        "requires_warmup": bool(not torch.isfinite(torch.tensor(loss)) or loss > threshold),
    }


def warmup_value_head(
    actor_critic: FullTurnActorCritic,
    episodes: list[EpisodeTrajectory],
    config: PPOConfig,
    device: torch.device,
    *,
    max_epochs: int = 5,
    minimum_epochs: int = 2,
) -> dict[str, float]:
    """Fit a reset critic to Monte-Carlo returns while leaving policy parameters untouched."""
    observations, targets = _critic_examples(episodes, config.gamma)
    split = max(1, min(len(targets) - 1, int(len(targets) * 0.8))) if len(targets) > 1 else 1
    train_observations, validation_observations = observations[:split], observations[split:]
    train_targets, validation_targets = targets[:split], targets[split:]
    if not len(validation_targets):
        validation_observations, validation_targets = train_observations, train_targets
    policy_flags = [parameter.requires_grad for parameter in actor_critic.policy.parameters()]
    actor_critic.policy.requires_grad_(False)
    actor_critic.value_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        actor_critic.value_head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batch_size = max(1, config.minibatch_turns)
    generator = torch.Generator().manual_seed(sum(episode.seed for episode in episodes))

    def validation_loss(batch_observations: Tensor, batch_targets: Tensor) -> float:
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(batch_targets), batch_size):
                targets_on_device = batch_targets[start : start + batch_size].to(device)
                _, values = actor_critic(batch_observations[start : start + batch_size].to(device))
                total += float(torch.nn.functional.huber_loss(values.float(), targets_on_device, reduction="sum"))
        return total / len(batch_targets)

    policy_was_training = actor_critic.policy.training
    value_was_training = actor_critic.value_head.training
    actor_critic.policy.eval()
    actor_critic.value_head.eval()
    initial_validation = validation_loss(validation_observations, validation_targets)
    history = []
    best = initial_validation
    best_state = {name: value.detach().clone() for name, value in actor_critic.value_head.state_dict().items()}
    stale = 0
    try:
        for epoch in range(max_epochs):
            actor_critic.policy.eval()
            actor_critic.value_head.train()
            order = torch.randperm(len(train_targets), generator=generator)
            train_total = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch_targets = train_targets[indices].to(device)
                _, values = actor_critic(train_observations[indices].to(device))
                loss = torch.nn.functional.huber_loss(values.float(), batch_targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Critic warm-up produced a non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.value_head.parameters(), config.gradient_clip)
                optimizer.step()
                train_total += float(loss.detach()) * len(indices)
            actor_critic.value_head.eval()
            validation = validation_loss(validation_observations, validation_targets)
            history.append((train_total / len(train_targets), validation))
            if validation < best - 1e-8:
                best = validation
                best_state = {
                    name: value.detach().clone() for name, value in actor_critic.value_head.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if epoch + 1 >= minimum_epochs and stale >= 2:
                break
    finally:
        if best_state is not None:
            actor_critic.value_head.load_state_dict(best_state)
        for parameter, requires_grad in zip(actor_critic.policy.parameters(), policy_flags):
            parameter.requires_grad_(requires_grad)
        actor_critic.policy.train(policy_was_training)
        actor_critic.value_head.train(value_was_training)
    return {
        "epochs": float(len(history)),
        "turns": float(len(targets)),
        "initial_train_loss": history[0][0],
        "final_train_loss": history[-1][0],
        "initial_validation_loss": initial_validation,
        "best_validation_loss": best,
        "final_validation_loss": best,
    }


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
    runner = _prepare_episode(
        actor_critic,
        opponent_factory,
        reward_program,
        device=device,
        seed=seed,
        opponent_name=opponent_name,
        max_turns=max_turns,
        inference_backend=inference_backend,
    )
    return runner.run()


@dataclass
class _PreparedEpisode:
    rollout_agent: RolloutAgent
    opponent_agent: Agent
    environment: LuxEnvironment
    reward_program: RewardProgram
    seed: int
    opponent_name: str

    def run(self) -> EpisodeTrajectory:
        with suppress(StopIteration):
            self.environment.reset(seed=self.seed)
        return finish_episode(
            self.rollout_agent,
            self.environment.game,
            self.reward_program,
            seed=self.seed,
            opponent=self.opponent_name,
        )

    def rollout_batchers(self, candidate_backend: object | None) -> set[object]:
        result = set()
        if candidate_backend is not None:
            result.add(candidate_backend)
        opponent_backend = getattr(self.opponent_agent, "rollout_batcher", None)
        if opponent_backend is not None:
            result.add(opponent_backend)
        return result


def _prepare_episode(
    actor_critic: FullTurnActorCritic,
    opponent_factory: Callable[[], Agent],
    reward_program: RewardProgram,
    *,
    device: torch.device,
    seed: int,
    opponent_name: str,
    max_turns: int | None,
    inference_backend: Callable[[Tensor], tuple[dict[str, Tensor], Tensor]] | None,
) -> _PreparedEpisode:
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
    return _PreparedEpisode(rollout_agent, opponent, environment, reward_program, seed, opponent_name)


def collect_episodes_batched(
    actor_critic: FullTurnActorCritic,
    episode_specs: list[tuple[Callable[[], Agent], int, str]],
    reward_program: RewardProgram,
    *,
    device: torch.device,
    inference_backend: Callable[[Tensor], tuple[dict[str, Tensor], Tensor]],
    max_turns: int | None = None,
    rollout_backend: str = "threaded",
) -> list[EpisodeTrajectory]:
    """Run one lockstep wave while a shared backend batches policy inference."""
    if not episode_specs:
        return []
    effective_backend = resolve_rollout_backend(rollout_backend)

    runners = [
        _prepare_episode(
            actor_critic,
            opponent_factory,
            reward_program,
            device=device,
            seed=seed,
            opponent_name=opponent_name,
            max_turns=max_turns,
            inference_backend=inference_backend,
        )
        for opponent_factory, seed, opponent_name in episode_specs
    ]
    candidate_backend = getattr(inference_backend, "__self__", None)
    if not hasattr(candidate_backend, "batch_scope"):
        candidate_backend = None
    participant_counts: dict[object, int] = {}
    runner_backends = []
    for runner in runners:
        backends = runner.rollout_batchers(candidate_backend)
        runner_backends.append(backends)
        for backend in backends:
            participant_counts[backend] = participant_counts.get(backend, 0) + 1

    def collect(item: tuple[_PreparedEpisode, set[object]]) -> EpisodeTrajectory:
        runner, backends = item
        try:
            return runner.run()
        finally:
            if effective_backend == "lockstep":
                for backend in backends:
                    backend.participant_done()

    with ExitStack() as scopes:
        if effective_backend == "lockstep":
            for backend, participants in participant_counts.items():
                scopes.enter_context(backend.batch_scope(participants))
        with ThreadPoolExecutor(max_workers=len(runners), thread_name_prefix="lux-rollout") as executor:
            return list(executor.map(collect, zip(runners, runner_backends)))


class PPOTrainer:
    def __init__(
        self,
        actor_critic: FullTurnActorCritic,
        config: PPOConfig,
        device: torch.device,
        *,
        bc_batch_provider: Callable[[], Mapping[str, Tensor]] | None = None,
        illegal_action_coefficient: float = 0.01,
    ) -> None:
        self.actor_critic = actor_critic
        self.config = config
        self.device = device
        self.bc_batch_provider = bc_batch_provider
        self.illegal_action_coefficient = float(illegal_action_coefficient)
        self.actor_lr_multiplier = 1.0
        self.bc_coefficient_multiplier = 1.0
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
                policy_losses = []
                entropies = []
                illegal_losses = []
                illegal_masses = []
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
                    masks = torch.stack([decision.legal_mask for _, decision in entity_decisions]).to(self.device)
                    masks = masks.to(dtype=torch.bool)
                    no_legal = ~masks.any(dim=-1)
                    if no_legal.any():
                        masks = masks.clone()
                        masks[no_legal, 0] = True
                    distribution = Categorical(logits=apply_legal_action_mask(logits, masks).float())
                    actions = torch.tensor([decision.action for _, decision in entity_decisions], device=self.device)
                    old_log_probs = torch.tensor(
                        [decision.old_log_prob for _, decision in entity_decisions], device=self.device
                    )
                    selected_advantages = batch_advantages[local_indices]
                    ratios = torch.exp(distribution.log_prob(actions) - old_log_probs)
                    unclipped = ratios * selected_advantages
                    clipped = ratios.clamp(1 - self.config.clip_range, 1 + self.config.clip_range) * selected_advantages
                    policy_losses.append(-torch.minimum(unclipped, clipped))
                    entropies.append(distribution.entropy())
                    legal_log_mass = torch.logsumexp(logits.float().masked_fill(~masks, -torch.inf), dim=-1)
                    all_log_mass = torch.logsumexp(logits.float(), dim=-1)
                    illegal_loss = all_log_mass - legal_log_mass
                    illegal_losses.append(illegal_loss)
                    illegal_masses.append((1.0 - torch.exp(-illegal_loss)).clamp(0.0, 1.0))
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
                bc_loss = self._distillation_anchor_loss(values)
                illegal_action_loss = torch.cat(illegal_losses).mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    + self.illegal_action_coefficient * illegal_action_loss
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO produced a non-finite loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.actor_lr_multiplier != 1.0:
                    for parameter in self.actor_critic.policy.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(self.actor_lr_multiplier)
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
                    "bc_loss": bc_loss,
                    "illegal_action_loss": illegal_action_loss,
                    "illegal_action_mass_mean": torch.cat(illegal_masses).mean(),
                    "illegal_action_mass_p95": torch.quantile(torch.cat(illegal_masses), 0.95),
                    "illegal_action_mass_max": torch.cat(illegal_masses).max(),
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
        if self.bc_batch_provider is None or self.config.bc_coefficient == 0 or self.bc_coefficient_multiplier == 0:
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

    def set_schedule_state(self, *, joint_update: int, bc_coefficient_multiplier: float = 1.0) -> None:
        self.actor_lr_multiplier = (0.25, 0.5, 1.0)[min(max(int(joint_update), 0), 2)]
        self.bc_coefficient_multiplier = float(bc_coefficient_multiplier)

    def save_training_checkpoint(
        self,
        path: Path,
        *,
        source_checkpoint: str,
        source_checkpoint_sha256: str | None = None,
        reward_program: RewardProgram,
        update: int,
        metrics: dict[str, Any],
        training_state: Mapping[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device).to(device="cpu", dtype=torch.uint8)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        torch.save(
            {
                "schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
                "source_checkpoint": source_checkpoint,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "policy": self.actor_critic.policy.state_dict(),
                "value_head": self.actor_critic.value_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ppo_config": asdict(self.config),
                "reward_program": reward_program.to_dict(),
                "update": update,
                "metrics": metrics,
                "training_state": dict(training_state or {}),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": cuda_rng_state,
                # Keep the old field readable by schema-v3 consumers without coupling it to visible GPU count.
                "cuda_rng_state_all": [cuda_rng_state] if cuda_rng_state is not None else None,
            },
            path,
        )

    def load_training_state(
        self,
        path: Path,
        *,
        source_checkpoint: str,
        source_checkpoint_sha256: str | None = None,
        reward_program: RewardProgram,
        legacy_target_decisions: int | None = None,
        legacy_stage_seconds: int | None = None,
        allow_compatible_source_checkpoint: bool = False,
    ) -> TrainingResumeState:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        schema_version = int(checkpoint.get("schema_version", 1))
        if schema_version not in {1, 2, TRAINING_CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError("Unsupported RL training checkpoint schema")
        stored_source_checkpoint = checkpoint.get("source_checkpoint")
        source_checkpoint_mismatch = stored_source_checkpoint != source_checkpoint
        if source_checkpoint_mismatch and not allow_compatible_source_checkpoint:
            raise ValueError("RL resume source checkpoint does not match")
        stored_source_sha256 = checkpoint.get("source_checkpoint_sha256")
        source_sha256_mismatch = bool(
            source_checkpoint_sha256 is not None
            and stored_source_sha256 is not None
            and stored_source_sha256 != source_checkpoint_sha256
        )
        if source_sha256_mismatch:
            raise ValueError("RL resume source checkpoint SHA-256 does not match")
        if checkpoint.get("reward_program") != reward_program.to_dict():
            raise ValueError("RL resume reward program does not match")
        if checkpoint.get("ppo_config") != asdict(self.config):
            raise ValueError("RL resume PPO configuration does not match")
        self.actor_critic.policy.load_state_dict(checkpoint["policy"])
        self.actor_critic.value_head.load_state_dict(checkpoint["value_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].to(device="cpu", dtype=torch.uint8))
        if self.device.type == "cuda" and torch.cuda.is_available():
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cuda_rng_state = _checkpoint_cuda_rng_state(checkpoint, device_index)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(
                    cuda_rng_state.to(device="cpu", dtype=torch.uint8),
                    self.device,
                )

        metrics = dict(checkpoint.get("metrics", {}))
        state = (
            dict(checkpoint.get("training_state", {})) if schema_version >= TRAINING_CHECKPOINT_SCHEMA_VERSION else {}
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
            curriculum_progress_decisions=int(
                state.get("curriculum_progress_decisions", state.get("constraint_progress", decisions))
            ),
            joint_update=int(state.get("joint_update", max(0, int(checkpoint["update"]) + 1))),
            source_checkpoint=str(stored_source_checkpoint) if stored_source_checkpoint is not None else None,
            source_checkpoint_mismatch=source_checkpoint_mismatch,
            source_checkpoint_sha256=str(stored_source_sha256) if stored_source_sha256 is not None else None,
            source_checkpoint_sha256_mismatch=source_sha256_mismatch,
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
