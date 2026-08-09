import re

with open("luxai2021/rl/ppo.py", "r") as f:
    code = f.read()

old_loop_start = """        totals: dict[str, float] = {}
        update_count = 0
        for _ in range(self.config.update_epochs):"""

new_loop_start = """        totals: dict[str, float] = {}
        update_count = 0
        gn_pol, gn_val, gn_bc = 0.0, 0.0, 0.0
        for _ in range(self.config.update_epochs):"""
code = code.replace(old_loop_start, new_loop_start)

old_backward = """                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO produced a non-finite loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()"""

new_backward = """                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO produced a non-finite loss")
                
                if update_count == 0:
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
code = code.replace(old_backward, new_backward)

old_result = """        result = {name: value / update_count for name, value in totals.items()}
        result.update(
            {"""

new_result = """        result = {name: value / update_count for name, value in totals.items()}
        result.update(
            {
                "grad_norm_policy_encoder": gn_pol,
                "grad_norm_value_encoder": gn_val,
                "grad_norm_bc_encoder": gn_bc,"""
code = code.replace(old_result, new_result)

with open("luxai2021/rl/ppo.py", "w") as f:
    f.write(code)
print("Patched ppo.py successfully for gradient norms")
