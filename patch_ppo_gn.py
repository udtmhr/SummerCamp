import re

with open("luxai2021/rl/ppo.py", "r") as f:
    code = f.read()

# Modify update signature
old_update = """    def update(self, episodes: list[EpisodeTrajectory]) -> dict[str, float]:"""
new_update = """    def update(self, episodes: list[EpisodeTrajectory], record_grad_norms: bool = False) -> dict[str, float]:"""
code = code.replace(old_update, new_update)

# Modify initialization of gn lists
old_gn_init = """        update_count = 0
        gn_pol, gn_val, gn_bc = 0.0, 0.0, 0.0
        for _ in range(self.config.update_epochs):"""
new_gn_init = """        update_count = 0
        gn_pols, gn_vals, gn_bcs = [], [], []
        for _ in range(self.config.update_epochs):"""
code = code.replace(old_gn_init, new_gn_init)

# Modify the logic inside the loop
old_gn_loop = """                if update_count == 0:
                    def _gn() -> float:
                        grads = [p.grad for p in self.actor_critic.policy.encoder.parameters() if p.grad is not None]
                        if not grads: return 0.0
                        return torch.norm(torch.stack([torch.norm(g) for g in grads])).item()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coefficient * value_loss).backward(retain_graph=True)
                    gn_val = _gn()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    policy_loss.backward(retain_graph=True)
                    gn_pol = _gn()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    bc_loss_weighted = self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    if isinstance(bc_loss_weighted, torch.Tensor) and bc_loss_weighted.requires_grad:
                        bc_loss_weighted.backward(retain_graph=True)
                        gn_bc = _gn()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()"""

new_gn_loop = """                if record_grad_norms:
                    def _gn() -> float:
                        grads = [p.grad for p in self.actor_critic.policy.encoder.parameters() if p.grad is not None]
                        if not grads: return 0.0
                        return torch.norm(torch.stack([torch.norm(g) for g in grads])).item()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    (self.config.value_coefficient * value_loss).backward(retain_graph=True)
                    gn_vals.append(_gn())
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    policy_loss.backward(retain_graph=True)
                    gn_pols.append(_gn())
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    bc_loss_weighted = self.config.bc_coefficient * self.bc_coefficient_multiplier * bc_loss
                    if isinstance(bc_loss_weighted, torch.Tensor) and bc_loss_weighted.requires_grad:
                        bc_loss_weighted.backward(retain_graph=True)
                        gn_bcs.append(_gn())
                    else:
                        gn_bcs.append(0.0)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()"""
code = code.replace(old_gn_loop, new_gn_loop)

# Modify result compilation
old_result = """        result = {name: value / update_count for name, value in totals.items()}
        result.update(
            {
                "grad_norm_policy_encoder": gn_pol,
                "grad_norm_value_encoder": gn_val,
                "grad_norm_bc_encoder": gn_bc,
            }
        )"""

new_result = """        result = {name: value / update_count for name, value in totals.items()}
        if record_grad_norms and gn_pols:
            import numpy as np
            result.update(
                {
                    "grad_norm_policy_mean": float(np.mean(gn_pols)),
                    "grad_norm_policy_max": float(np.max(gn_pols)),
                    "grad_norm_policy_p95": float(np.percentile(gn_pols, 95)),
                    "grad_norm_value_mean": float(np.mean(gn_vals)),
                    "grad_norm_value_max": float(np.max(gn_vals)),
                    "grad_norm_value_p95": float(np.percentile(gn_vals, 95)),
                    "grad_norm_bc_mean": float(np.mean(gn_bcs)),
                    "grad_norm_bc_max": float(np.max(gn_bcs)),
                    "grad_norm_bc_p95": float(np.percentile(gn_bcs, 95)),
                }
            )"""
code = code.replace(old_result, new_result)

with open("luxai2021/rl/ppo.py", "w") as f:
    f.write(code)
print("Patched ppo.py successfully for gradient norms")
