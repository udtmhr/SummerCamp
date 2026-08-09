import re

with open("luxai2021/rl/ppo.py", "r") as f:
    code = f.read()

# 1. Add target_kl to PPOConfig
old_config = """    kl_coefficient: float = 0.0
    bc_coefficient: float = 0.05
    gradient_clip: float = 1.0
    update_epochs: int = 2
    minibatch_turns: int = 32

    def __post_init__(self) -> None:"""

new_config = """    kl_coefficient: float = 0.0
    target_kl: float | None = 0.01
    bc_coefficient: float = 0.05
    gradient_clip: float = 1.0
    update_epochs: int = 2
    minibatch_turns: int = 32

    def __post_init__(self) -> None:"""
code = code.replace(old_config, new_config)

# 2. Add approx_kls and check inside update_epochs loop
old_epoch_loop = """        gn_pols, gn_vals, gn_bcs = [], [], []
        for _ in range(self.config.update_epochs):
            order = torch.randperm(len(records), generator=generator).tolist()"""

new_epoch_loop = """        gn_pols, gn_vals, gn_bcs = [], [], []
        early_stop = False
        epoch_count = 0
        for _ in range(self.config.update_epochs):
            if early_stop:
                break
            epoch_count += 1
            order = torch.randperm(len(records), generator=generator).tolist()"""
code = code.replace(old_epoch_loop, new_epoch_loop)

# 3. Add approx_kl computation and early stop check
old_inner_loop = """                policy_losses = []
                entropies = []
                kls = []
                illegal_losses = []
                illegal_masses = []
                turn_indices = []
                batch_advantages = advantages[indices].to(self.device)"""

new_inner_loop = """                policy_losses = []
                entropies = []
                kls = []
                approx_kls = []
                illegal_losses = []
                illegal_masses = []
                turn_indices = []
                batch_advantages = advantages[indices].to(self.device)"""
code = code.replace(old_inner_loop, new_inner_loop)

old_approx_kl = """                    ratios = torch.exp(distribution.log_prob(actions) - old_log_probs)
                    unclipped = ratios * selected_advantages"""

new_approx_kl = """                    ratios = torch.exp(distribution.log_prob(actions) - old_log_probs)
                    approx_kls.append((old_log_probs - distribution.log_prob(actions)).mean())
                    unclipped = ratios * selected_advantages"""
code = code.replace(old_approx_kl, new_approx_kl)

old_early_stop_check = """                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()"""

new_early_stop_check = """                if self.config.target_kl is not None and approx_kls:
                    if torch.stack(approx_kls).mean().item() > 1.5 * self.config.target_kl:
                        early_stop = True
                        break

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()"""
code = code.replace(old_early_stop_check, new_early_stop_check)

# 4. Also record early stop
old_result = """        result = {name: value / update_count for name, value in totals.items()}
        if record_grad_norms and gn_pols:"""

new_result = """        result = {name: value / update_count for name, value in totals.items() if update_count > 0}
        result["early_stopped"] = float(early_stop)
        result["epochs_completed"] = float(epoch_count)
        if record_grad_norms and gn_pols:"""
code = code.replace(old_result, new_result)

with open("luxai2021/rl/ppo.py", "w") as f:
    f.write(code)
print("Patched PPO target-KL successfully")
