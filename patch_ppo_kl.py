import re

with open("luxai2021/rl/ppo.py", "r") as f:
    code = f.read()

old_init = """    def __init__(
        self,
        actor_critic: FullTurnActorCritic,
        config: PPOConfig,
        device: torch.device,
        *,
        bc_batch_provider: Callable[[], Mapping[str, Tensor]] | None = None,
        illegal_action_coefficient: float = 0.01,
    ) -> None:
        self.actor_critic = actor_critic"""

new_init = """    def __init__(
        self,
        actor_critic: FullTurnActorCritic,
        config: PPOConfig,
        device: torch.device,
        *,
        reference_policy: nn.Module | None = None,
        bc_batch_provider: Callable[[], Mapping[str, Tensor]] | None = None,
        illegal_action_coefficient: float = 0.01,
    ) -> None:
        self.actor_critic = actor_critic
        self.reference_policy = reference_policy"""
code = code.replace(old_init, new_init)

old_update_loop = """                output, values = self.actor_critic(observations)
                policy_losses = []
                entropies = []
                illegal_losses = []"""

new_update_loop = """                output, values = self.actor_critic(observations)
                reference_output = None
                if self.config.kl_coefficient > 0 and self.reference_policy is not None:
                    with torch.no_grad():
                        reference_output = self.reference_policy(observations)
                policy_losses = []
                entropies = []
                kls = []
                illegal_losses = []"""
code = code.replace(old_update_loop, new_update_loop)

old_inner_loop = """                    distribution = Categorical(logits=apply_legal_action_mask(logits, masks).float())
                    actions = torch.tensor([decision.action for _, decision in entity_decisions], device=self.device)"""

new_inner_loop = """                    distribution = Categorical(logits=apply_legal_action_mask(logits, masks).float())
                    if reference_output is not None:
                        reference_logits = reference_output[entity][local_indices, :, ys, xs]
                        reference_distribution = Categorical(
                            logits=apply_legal_action_mask(reference_logits, masks).float()
                        )
                        from torch.distributions import kl_divergence
                        kls.append(kl_divergence(reference_distribution, distribution))
                    actions = torch.tensor([decision.action for _, decision in entity_decisions], device=self.device)"""
code = code.replace(old_inner_loop, new_inner_loop)

old_loss_calc = """                turn_count = max(1, len(batch_records))
                policy_loss = torch.cat(policy_losses).sum() / turn_count
                entropy = torch.cat(entropies).sum() / turn_count
                bc_loss = self._distillation_anchor_loss(values)
                illegal_action_loss = torch.cat(illegal_losses).sum() / turn_count
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    + self.illegal_action_coefficient * illegal_action_loss
                )"""

new_loss_calc = """                turn_count = max(1, len(batch_records))
                policy_loss = torch.cat(policy_losses).sum() / turn_count
                entropy = torch.cat(entropies).sum() / turn_count
                kl = torch.cat(kls).sum() / turn_count if kls else torch.tensor(0.0, device=self.device)
                bc_loss = self._distillation_anchor_loss(values)
                illegal_action_loss = torch.cat(illegal_losses).sum() / turn_count
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.kl_coefficient * kl
                    + self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    + self.illegal_action_coefficient * illegal_action_loss
                )"""
code = code.replace(old_loss_calc, new_loss_calc)

old_batch_metrics = """                batch_metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "bc_loss": bc_loss,
                    "illegal_action_loss": illegal_action_loss,
                    "illegal_action_mass_mean": torch.cat(illegal_masses).mean(),"""

new_batch_metrics = """                batch_metrics = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "kl": kl,
                    "bc_loss": bc_loss,
                    "illegal_action_loss": illegal_action_loss,
                    "illegal_action_mass_mean": torch.cat(illegal_masses).mean(),"""
code = code.replace(old_batch_metrics, new_batch_metrics)

with open("luxai2021/rl/ppo.py", "w") as f:
    f.write(code)
print("Patched ppo.py successfully")
