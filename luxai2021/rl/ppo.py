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

TRAINING_CHECKPOINT_SCHEMA_VERSION = 4
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
    learning_rate: float = 5e-6
    weight_decay: float = 1e-5
    gamma: float = 0.999
    gae_lambda: float = 0.995
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coefficient: float = 0.001
    value_coefficient: float = 0.5
    kl_coefficient: float = 0.0
    target_kl: float | None = 0.01
    bc_coefficient: float = 0.05
    gradient_clip: float = 1.0
    update_epochs: int = 2
    minibatch_turns: int = 64
    illegal_action_coefficient: float = 0.01
    joint_action_policy: bool = True
    joint_loss_reference_actions: int = 32
    online_teacher_kl: bool = True
    teacher_kl_target_grad_ratio: float = 0.33
    teacher_kl_coefficient_min: float = 0.001
    teacher_kl_coefficient_max: float = 0.05

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
            "illegal_action_coefficient": (0.0, 1.0),
        }
        for name, (low, high) in bounds.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.update_epochs < 1 or self.minibatch_turns < 1:
            raise ValueError("PPO update sizes must be positive")
        if self.joint_loss_reference_actions < 1:
            raise ValueError("joint_loss_reference_actions must be positive")
        if not 0.0 < self.teacher_kl_target_grad_ratio <= 2.0:
            raise ValueError("teacher_kl_target_grad_ratio must be in (0, 2]")
        if not 0.0 <= self.teacher_kl_coefficient_min <= self.teacher_kl_coefficient_max <= 1.0:
            raise ValueError("invalid teacher KL coefficient bounds")


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
    curriculum_progress_games: int = 0
    budget_unit: str = "decisions"
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
        is_terminal = index + 1 == len(records)
        terminal = outcome if is_terminal else 0.0
        breakdown = reward_program.reward(
            record.metrics,
            following,
            terminal_outcome=terminal,
            terminal=is_terminal,
        )
        record.reward = breakdown.total
        record.reward_components = dict(breakdown.components)
        record.reward_component_shaping = dict(breakdown.component_shaping)
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
        is_terminal = index + 1 == len(episode.records)
        terminal = episode.outcome if is_terminal else 0.0
        breakdown = reward_program.reward(
            record.metrics,
            following,
            terminal_outcome=terminal,
            terminal=is_terminal,
        )
        record.reward = breakdown.total
        record.reward_components = dict(breakdown.components)
        record.reward_component_shaping = dict(breakdown.component_shaping)


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
    teacher_inference_backend: Callable[[object, int], dict[str, Tensor]] | None = None,
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
        teacher_inference_backend=teacher_inference_backend,
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

    def rollout_batchers(self, candidate_backend: object | None, teacher_backend: object | None) -> set[object]:
        result = set()
        if candidate_backend is not None:
            result.add(candidate_backend)
        if teacher_backend is not None:
            result.add(teacher_backend)
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
    teacher_inference_backend: Callable[[object, int], dict[str, Tensor]] | None = None,
) -> _PreparedEpisode:
    rollout_agent = RolloutAgent(
        actor_critic,
        device=device if inference_backend is None else "cpu",
        inference_backend=inference_backend,
        teacher_inference_backend=teacher_inference_backend,
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
    teacher_inference_backend: Callable[[object, int], dict[str, Tensor]] | None = None,
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
            teacher_inference_backend=teacher_inference_backend,
        )
        for opponent_factory, seed, opponent_name in episode_specs
    ]
    candidate_backend = getattr(inference_backend, "__self__", None)
    if not hasattr(candidate_backend, "batch_scope"):
        candidate_backend = None
    teacher_backend = getattr(teacher_inference_backend, "__self__", None)
    if not hasattr(teacher_backend, "batch_scope"):
        teacher_backend = None
    participant_counts: dict[object, int] = {}
    runner_backends = []
    for runner in runners:
        backends = runner.rollout_batchers(candidate_backend, teacher_backend)
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
        reference_policy: torch.nn.Module | None = None,
        bc_batch_provider: Callable[[], Mapping[str, Tensor]] | None = None,
        illegal_action_coefficient: float | None = None,
    ) -> None:
        self.actor_critic = actor_critic
        self.reference_policy = reference_policy
        self.config = config
        self.device = device
        self.bc_batch_provider = bc_batch_provider
        self.illegal_action_coefficient = float(
            config.illegal_action_coefficient
            if illegal_action_coefficient is None
            else illegal_action_coefficient
        )
        self.actor_lr_multiplier = 1.0
        self.bc_coefficient_multiplier = 1.0
        self.effective_teacher_kl_coefficient: float | None = None
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

    @staticmethod
    def _forward_reference(reference_policy: torch.nn.Module, observations: Tensor) -> dict[str, Tensor]:
        if isinstance(reference_policy, FullTurnActorCritic):
            return reference_policy.forward_tta(observations)[0]
        return reference_policy(observations)

    @staticmethod
    def _categorical_kl(teacher_logits: Tensor, learner: Categorical, mask: Tensor) -> Tensor:
        teacher = Categorical(logits=apply_legal_action_mask(teacher_logits[None], mask[None])[0].float())
        safe_teacher = teacher.logits.masked_fill(~mask, 0.0)
        safe_learner = learner.logits.masked_fill(~mask, 0.0)
        return (teacher.probs * (safe_teacher - safe_learner)).sum()

    @staticmethod
    def _batched_categorical_kl(teacher_logits: Tensor, learner: Categorical, masks: Tensor) -> Tensor:
        teacher = Categorical(logits=apply_legal_action_mask(teacher_logits, masks).float())
        safe_teacher = teacher.logits.masked_fill(~masks, 0.0)
        safe_learner = learner.logits.masked_fill(~masks, 0.0)
        return (teacher.probs * (safe_teacher - safe_learner)).sum(dim=-1)

    @staticmethod
    def _gradient_norm(loss: Tensor, parameters: Iterable[Tensor]) -> float:
        if not loss.requires_grad:
            return 0.0
        params = [parameter for parameter in parameters if parameter.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        finite = [gradient for gradient in grads if gradient is not None]
        if not finite:
            return 0.0
        return float(torch.sqrt(sum(gradient.float().square().sum() for gradient in finite)).detach())

    def _calibrate_teacher_kl(self, policy_loss: Tensor, teacher_kl: Tensor) -> None:
        if self.effective_teacher_kl_coefficient is not None or not teacher_kl.requires_grad:
            return
        parameters = list(self.actor_critic.policy.encoder.parameters())
        policy_norm = self._gradient_norm(policy_loss, parameters)
        teacher_norm = self._gradient_norm(teacher_kl, parameters)
        if not torch.isfinite(torch.tensor([policy_norm, teacher_norm])).all() or teacher_norm <= 0.0:
            raise FloatingPointError("Online teacher KL calibration produced an invalid gradient norm")
        coefficient = self.config.teacher_kl_target_grad_ratio * policy_norm / teacher_norm
        self.effective_teacher_kl_coefficient = min(
            self.config.teacher_kl_coefficient_max,
            max(self.config.teacher_kl_coefficient_min, coefficient),
        )

    def _priority_log_prob_and_entropy(
        self,
        output: dict[str, Tensor],
        batch_index: int,
        decisions: list[object],
    ) -> tuple[Tensor, Tensor]:
        zero = next(iter(output.values())).sum() * 0.0
        total_log_prob = zero
        total_entropy = zero
        groups = {decision.priority_group for decision in decisions if decision.priority_group}
        for group in groups:
            ordered = sorted(
                (decision for decision in decisions if decision.priority_group == group),
                key=lambda decision: decision.priority_index,
            )
            for selected_index in range(len(ordered)):
                margins = []
                for decision in ordered[selected_index:]:
                    y, x = decision.position
                    logits = output[decision.entity][batch_index, :, y, x]
                    source_mask = (
                        decision.priority_legal_mask
                        if decision.priority_legal_mask is not None
                        else decision.legal_mask
                    )
                    mask = source_mask.to(self.device, dtype=torch.bool)
                    distribution = self._masked_distribution(logits, mask)
                    if int(mask.sum()) <= 1:
                        margins.append(logits.sum() * 0.0 + 20.0)
                    else:
                        top = torch.topk(distribution.probs, 2).values
                        margins.append(top[0] - top[1])
                priority = Categorical(logits=torch.stack(margins).float())
                total_log_prob = total_log_prob + priority.log_prob(torch.tensor(0, device=self.device))
                total_entropy = total_entropy + priority.entropy()
        return total_log_prob, total_entropy

    def _vectorized_turn_statistics(
        self,
        output: dict[str, Tensor],
        values: Tensor,
        batch_records: list[TurnRecord],
        reference_output: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        """Evaluate all conditional actions with one tensor operation per entity head."""
        num_turns = len(batch_records)
        zero = values.new_zeros(num_turns)
        turn_action_log_prob = zero
        turn_action_entropy = zero
        turn_teacher_kl = zero
        turn_reference_kl = zero
        turn_illegal_loss = zero
        action_counts = values.new_zeros(num_turns)
        illegal_masses: list[Tensor] = []
        priority_margins: dict[int, Tensor] = {}

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
                [local_index for local_index, _ in entity_decisions],
                device=self.device,
            )
            ys = torch.tensor([decision.position[0] for _, decision in entity_decisions], device=self.device)
            xs = torch.tensor([decision.position[1] for _, decision in entity_decisions], device=self.device)
            logits = output[entity][local_indices, :, ys, xs]
            masks = torch.stack([decision.legal_mask for _, decision in entity_decisions]).to(
                self.device,
                dtype=torch.bool,
            )
            distribution = Categorical(logits=apply_legal_action_mask(logits, masks).float())
            actions = torch.tensor([decision.action for _, decision in entity_decisions], device=self.device)
            selected_log_probs = distribution.log_prob(actions)
            turn_action_log_prob = turn_action_log_prob.scatter_add(0, local_indices, selected_log_probs)
            turn_action_entropy = turn_action_entropy.scatter_add(0, local_indices, distribution.entropy())
            action_counts = action_counts.scatter_add(0, local_indices, torch.ones_like(selected_log_probs))

            legal_log_mass = torch.logsumexp(logits.float().masked_fill(~masks, -torch.inf), dim=-1)
            entity_illegal = torch.logsumexp(logits.float(), dim=-1) - legal_log_mass
            turn_illegal_loss = turn_illegal_loss.scatter_add(0, local_indices, entity_illegal)
            illegal_masses.append((1.0 - torch.exp(-entity_illegal)).clamp(0.0, 1.0))

            if reference_output is not None:
                reference_logits = reference_output[entity][local_indices, :, ys, xs]
                reference_kls = self._batched_categorical_kl(reference_logits, distribution, masks)
                turn_reference_kl = turn_reference_kl.scatter_add(0, local_indices, reference_kls)

            teacher_positions = [
                index for index, (_, decision) in enumerate(entity_decisions) if decision.teacher_logits is not None
            ]
            if teacher_positions:
                teacher_selection = torch.tensor(teacher_positions, device=self.device)
                teacher_logits = torch.stack(
                    [entity_decisions[index][1].teacher_logits for index in teacher_positions]
                ).to(self.device)
                teacher_distribution = Categorical(logits=distribution.logits[teacher_selection])
                teacher_kls = self._batched_categorical_kl(
                    teacher_logits,
                    teacher_distribution,
                    masks[teacher_selection],
                )
                turn_teacher_kl = turn_teacher_kl.scatter_add(
                    0,
                    local_indices[teacher_selection],
                    teacher_kls,
                )

            priority_masks = torch.stack(
                [
                    decision.priority_legal_mask
                    if decision.priority_legal_mask is not None
                    else decision.legal_mask
                    for _, decision in entity_decisions
                ]
            ).to(self.device, dtype=torch.bool)
            priority_distribution = Categorical(
                logits=apply_legal_action_mask(logits, priority_masks).float()
            )
            top = torch.topk(priority_distribution.probs, min(2, logits.shape[-1]), dim=-1).values
            margins = top[:, 0] - top[:, 1] if top.shape[-1] > 1 else top[:, 0]
            margins = torch.where(priority_masks.sum(dim=-1) <= 1, margins.new_full((), 20.0), margins)
            priority_margins.update(
                {id(decision): margins[index] for index, (_, decision) in enumerate(entity_decisions)}
            )

        group_rows: list[tuple[int, list[Tensor]]] = []
        for local_index, record in enumerate(batch_records):
            groups = {decision.priority_group for decision in record.decisions if decision.priority_group}
            for group in groups:
                ordered = sorted(
                    (decision for decision in record.decisions if decision.priority_group == group),
                    key=lambda decision: decision.priority_index,
                )
                group_rows.append((local_index, [priority_margins[id(decision)] for decision in ordered]))

        turn_priority_log_prob = values.new_zeros(num_turns)
        turn_priority_entropy = values.new_zeros(num_turns)
        if group_rows:
            maximum = max(len(scores) for _, scores in group_rows)
            padded = torch.stack(
                [
                    torch.nn.functional.pad(
                        torch.stack(scores),
                        (0, maximum - len(scores)),
                        value=-1e9,
                    )
                    for _, scores in group_rows
                ]
            )
            lengths = torch.tensor([len(scores) for _, scores in group_rows], device=self.device)
            group_turns = torch.tensor([turn for turn, _ in group_rows], device=self.device)
            positions = torch.arange(maximum, device=self.device)[None]
            valid = positions < lengths[:, None]
            suffix_logsumexp = torch.logcumsumexp(padded.flip(dims=(1,)), dim=1).flip(dims=(1,))
            group_log_prob = ((padded - suffix_logsumexp) * valid).sum(dim=1)
            turn_priority_log_prob = turn_priority_log_prob.scatter_add(0, group_turns, group_log_prob)
            for position in range(maximum):
                active = lengths > position
                if not bool(active.any()):
                    break
                stage_distribution = Categorical(logits=padded[active, position:])
                turn_priority_entropy = turn_priority_entropy.scatter_add(
                    0,
                    group_turns[active],
                    stage_distribution.entropy(),
                )

        return {
            "joint_log_prob": turn_action_log_prob + turn_priority_log_prob,
            "entropy": turn_action_entropy + turn_priority_entropy,
            "teacher_kl": turn_teacher_kl,
            "reference_kl": turn_reference_kl,
            "illegal_loss": turn_illegal_loss,
            "illegal_masses": torch.cat(illegal_masses),
            "action_counts": action_counts,
        }

    def update(self, episodes: list[EpisodeTrajectory], *, record_grad_norms: bool = False) -> dict[str, float]:
        records = calculate_gae(episodes, self.config)
        if not records:
            raise ValueError("Cannot update PPO without rollout turns")
        advantages = torch.tensor([record.advantage for record in records], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        generator = torch.Generator().manual_seed(sum(episode.seed for episode in episodes))
        totals: dict[str, float] = {}
        update_count = 0
        gn_pols, gn_vals, gn_bcs, gn_teachers = [], [], [], []
        gn_illegals = []
        sampled_reference_kls: list[float] = []
        early_stop = False
        epoch_count = 0
        for _ in range(self.config.update_epochs):
            if early_stop:
                break
            epoch_count += 1
            order = torch.randperm(len(records), generator=generator).tolist()
            for start in range(0, len(order), self.config.minibatch_turns):
                indices = order[start : start + self.config.minibatch_turns]
                batch_records = [records[index] for index in indices]
                observations = torch.stack([record.observation for record in batch_records]).to(
                    self.device,
                    non_blocking=True,
                )
                output, values = self.actor_critic.forward_tta(observations)
                reference_output = None
                measure_reference_kl = self.reference_policy is not None and (
                    self.config.kl_coefficient > 0 or not sampled_reference_kls
                )
                if measure_reference_kl:
                    with torch.no_grad():
                        reference_output = self._forward_reference(self.reference_policy, observations)
                batch_advantages = advantages[indices].to(self.device)
                statistics = self._vectorized_turn_statistics(
                    output,
                    values,
                    batch_records,
                    reference_output,
                )
                active = statistics["action_counts"] > 0
                if not bool(active.any()):
                    continue
                old_joint_log_probs = torch.tensor(
                    [
                        record.old_joint_log_prob
                        if record.old_joint_log_prob is not None
                        else sum(
                            decision.old_log_prob + decision.priority_log_prob
                            for decision in record.decisions
                        )
                        for record in batch_records
                    ],
                    device=self.device,
                )
                log_ratios = statistics["joint_log_prob"][active] - old_joint_log_probs[active]
                ratios_tensor = torch.exp(log_ratios)
                active_advantages = batch_advantages[active]
                unclipped = ratios_tensor * active_advantages
                clipped = ratios_tensor.clamp(
                    1 - self.config.clip_range,
                    1 + self.config.clip_range,
                ) * active_advantages
                scale = float(self.config.joint_loss_reference_actions)
                policy_loss = (-torch.minimum(unclipped, clipped) / scale).mean()
                entropy = (statistics["entropy"][active] / scale).mean()
                teacher_kl = (statistics["teacher_kl"][active] / scale).mean()
                kl = (statistics["reference_kl"][active] / scale).mean()
                illegal_action_loss = (statistics["illegal_loss"][active] / scale).mean()
                per_action_approx_kls = -log_ratios / statistics["action_counts"][active].clamp_min(1)
                joint_approx_kls = -log_ratios
                approx_kl = per_action_approx_kls.mean()
                joint_kl = joint_approx_kls.mean()
                illegal_masses_tensor = statistics["illegal_masses"]
                old_values = torch.tensor([record.value for record in batch_records], device=self.device)
                returns = torch.tensor([record.return_value for record in batch_records], device=self.device)
                clipped_values = old_values + (values - old_values).clamp(
                    -self.config.value_clip_range,
                    self.config.value_clip_range,
                )
                value_loss = (
                    0.5 * torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
                )
                bc_loss = self._distillation_anchor_loss(values)
                if reference_output is not None:
                    sampled_reference_kls.append(float(kl.detach()))
                if self.config.online_teacher_kl and any(
                    decision.teacher_logits is not None
                    for record in batch_records
                    for decision in record.decisions
                ):
                    self._calibrate_teacher_kl(policy_loss, teacher_kl)
                teacher_coefficient = self.effective_teacher_kl_coefficient or 0.0
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.kl_coefficient * kl
                    + self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    + self.illegal_action_coefficient * illegal_action_loss
                    + teacher_coefficient * teacher_kl
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO produced a non-finite loss")

                # Gradient-ratio diagnostics require five additional backward passes.
                # One representative minibatch is sufficient; repeating this for every
                # minibatch made the first/last update several times slower than training.
                if record_grad_norms and not gn_pols:
                    def _gn() -> float:
                        grads = [p.grad for p in self.actor_critic.policy.encoder.parameters() if p.grad is not None]
                        if not grads:
                            return 0.0
                        return torch.norm(torch.stack([torch.norm(g) for g in grads])).item()

                    self.optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coefficient * value_loss).backward(retain_graph=True)
                    gn_vals.append(_gn())

                    self.optimizer.zero_grad(set_to_none=True)
                    policy_loss.backward(retain_graph=True)
                    gn_pols.append(_gn())

                    self.optimizer.zero_grad(set_to_none=True)
                    (teacher_coefficient * teacher_kl).backward(retain_graph=True)
                    gn_teachers.append(_gn())

                    self.optimizer.zero_grad(set_to_none=True)
                    bc_loss_weighted = self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    if isinstance(bc_loss_weighted, torch.Tensor) and bc_loss_weighted.requires_grad:
                        bc_loss_weighted.backward(retain_graph=True)
                        gn_bcs.append(_gn())
                    else:
                        gn_bcs.append(0.0)

                    self.optimizer.zero_grad(set_to_none=True)
                    illegal_weighted = self.illegal_action_coefficient * illegal_action_loss
                    illegal_weighted.backward(retain_graph=True)
                    gn_illegals.append(_gn())

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
                    "kl": kl,
                    "approx_kl": approx_kl,
                    "joint_kl": joint_kl,
                    "joint_clip_fraction": ((ratios_tensor - 1.0).abs() > self.config.clip_range).float().mean(),
                    "joint_log_ratio_p95": torch.quantile(joint_approx_kls.abs(), 0.95),
                    "joint_log_ratio_max": joint_approx_kls.abs().max(),
                    "actions_per_turn": statistics["action_counts"][active].mean(),
                    "bc_loss": bc_loss,
                    "online_teacher_kl": teacher_kl,
                    "online_teacher_kl_coefficient": torch.tensor(teacher_coefficient, device=self.device),
                    "illegal_action_loss": illegal_action_loss,
                    "illegal_action_mass_mean": illegal_masses_tensor.mean(),
                    "illegal_action_mass_p95": torch.quantile(illegal_masses_tensor, 0.95),
                    "illegal_action_mass_max": illegal_masses_tensor.max(),
                    "gradient_norm": gradient_norm,
                }
                for name, value in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                update_count += 1
                if (
                    self.config.target_kl is not None
                    and approx_kl.item() > 1.5 * self.config.target_kl
                ):
                    early_stop = True
                    break
        if update_count == 0:
            raise ValueError("PPO rollout contained no actionable entities")
        result = {name: value / update_count for name, value in totals.items() if update_count > 0}
        reference_kl = sum(sampled_reference_kls) / max(len(sampled_reference_kls), 1)
        result["reference_kl"] = reference_kl
        # Preserve the legacy metric name while separating it from PPO's old/current approximate KL.
        result["kl"] = reference_kl
        result["early_stopped"] = float(early_stop)
        result["epochs_completed"] = float(epoch_count)

        updates = {
            "episodes": float(len(episodes)),
            "turns": float(len(records)),
            "decisions": float(sum(len(record.decisions) for record in records)),
            "score_rate": sum((episode.outcome + 1.0) * 0.5 for episode in episodes) / len(episodes),
        }

        if record_grad_norms and gn_pols:
            import numpy as np
            updates.update({
                "grad_norm_samples": float(len(gn_pols)),
                "grad_norm_policy_mean": float(np.mean(gn_pols)),
                "grad_norm_policy_max": float(np.max(gn_pols)),
                "grad_norm_policy_p95": float(np.percentile(gn_pols, 95)),
                "grad_norm_value_mean": float(np.mean(gn_vals)),
                "grad_norm_value_max": float(np.max(gn_vals)),
                "grad_norm_value_p95": float(np.percentile(gn_vals, 95)),
                "grad_norm_bc_mean": float(np.mean(gn_bcs)),
                "grad_norm_bc_max": float(np.max(gn_bcs)),
                "grad_norm_bc_p95": float(np.percentile(gn_bcs, 95)),
                "grad_norm_teacher_mean": float(np.mean(gn_teachers)) if gn_teachers else 0.0,
                "grad_norm_teacher_to_policy_ratio": (
                    float(np.mean(gn_teachers)) / max(float(np.mean(gn_pols)), 1e-12) if gn_teachers else 0.0
                ),
                "grad_norm_illegal_mean": float(np.mean(gn_illegals)),
                "grad_norm_illegal_max": float(np.max(gn_illegals)),
                "grad_norm_illegal_p95": float(np.percentile(gn_illegals, 95)),
                "grad_norm_policy_to_bc_ratio": float(np.mean(gn_pols)) / max(float(np.mean(gn_bcs)), 1e-12),
                "grad_norm_policy_to_illegal_ratio": float(np.mean(gn_pols))
                / max(float(np.mean(gn_illegals)), 1e-12),
            })

        result.update(updates)
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
                "decoder_schema": "joint_sequential_v2",
                "inference_augmentation": "rot180",
                "effective_teacher_kl_coefficient": self.effective_teacher_kl_coefficient,
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
        budget_unit: str = "decisions",
    ) -> TrainingResumeState:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        schema_version = int(checkpoint.get("schema_version", 1))
        if schema_version not in {1, 2, 3, TRAINING_CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError("Unsupported RL training checkpoint schema")
        if schema_version < TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "RL checkpoint schema v3 or earlier cannot resume optimizer state; use policy/value stage inheritance"
            )
        if checkpoint.get("decoder_schema") != "joint_sequential_v2":
            raise ValueError("RL resume decoder schema does not match")
        if checkpoint.get("inference_augmentation") != "rot180":
            raise ValueError("RL resume inference augmentation does not match")
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
        stored_ppo_config = dict(checkpoint.get("ppo_config", {}))
        # Schema-v3 checkpoints predate the explicit auxiliary-loss field.
        stored_ppo_config.setdefault("illegal_action_coefficient", 0.01)
        if stored_ppo_config != asdict(self.config):
            raise ValueError("RL resume PPO configuration does not match")
        self.actor_critic.policy.load_state_dict(checkpoint["policy"])
        self.actor_critic.value_head.load_state_dict(checkpoint["value_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        coefficient = checkpoint.get("effective_teacher_kl_coefficient")
        self.effective_teacher_kl_coefficient = float(coefficient) if coefficient is not None else None
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
        stored_budget_unit = str(state.get("budget_unit", "decisions"))
        if stored_budget_unit != budget_unit:
            raise ValueError("Cannot mix game and decision budgets within one run")
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
            curriculum_progress_games=int(state.get("curriculum_progress_games", 0)),
            budget_unit=stored_budget_unit,
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
