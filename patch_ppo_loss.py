import re

with open("luxai2021/rl/ppo.py", "r") as f:
    code = f.read()

# Replace initialization
old_init = """                policy_losses = []
                entropies = []
                kls = []
                illegal_losses = []
                illegal_masses = []
                batch_advantages = advantages[indices].to(self.device)"""

new_init = """                policy_losses = []
                entropies = []
                kls = []
                illegal_losses = []
                illegal_masses = []
                turn_indices = []
                batch_advantages = advantages[indices].to(self.device)"""
code = code.replace(old_init, new_init)

# Append to turn_indices
old_append = """                    illegal_losses.append(illegal_loss)
                    illegal_masses.append((1.0 - torch.exp(-illegal_loss)).clamp(0.0, 1.0))
                if not policy_losses:"""

new_append = """                    illegal_losses.append(illegal_loss)
                    illegal_masses.append((1.0 - torch.exp(-illegal_loss)).clamp(0.0, 1.0))
                    turn_indices.append(local_indices)
                if not policy_losses:"""
code = code.replace(old_append, new_append)

# Replace the aggregation logic
old_agg = """                turn_count = max(1, len(batch_records))
                policy_loss = torch.cat(policy_losses).sum() / turn_count
                entropy = torch.cat(entropies).sum() / turn_count
                kl = torch.cat(kls).sum() / turn_count if kls else torch.tensor(0.0, device=self.device)
                bc_loss = self._distillation_anchor_loss(values)
                illegal_action_loss = torch.cat(illegal_losses).sum() / turn_count"""

new_agg = """                num_turns = len(batch_records)
                turn_count = max(1, num_turns)
                flat_indices = torch.cat(turn_indices)
                turn_action_counts = torch.bincount(flat_indices, minlength=num_turns).clamp_min(1)
                
                flat_policy = torch.cat(policy_losses)
                turn_policy_loss = torch.zeros(num_turns, device=self.device).scatter_add_(0, flat_indices, flat_policy) / turn_action_counts
                policy_loss = turn_policy_loss.sum() / turn_count
                
                flat_entropy = torch.cat(entropies)
                turn_entropy = torch.zeros(num_turns, device=self.device).scatter_add_(0, flat_indices, flat_entropy) / turn_action_counts
                entropy = turn_entropy.sum() / turn_count
                
                if kls:
                    flat_kls = torch.cat(kls)
                    turn_kl = torch.zeros(num_turns, device=self.device).scatter_add_(0, flat_indices, flat_kls) / turn_action_counts
                    kl = turn_kl.sum() / turn_count
                else:
                    kl = torch.tensor(0.0, device=self.device)
                    
                bc_loss = self._distillation_anchor_loss(values)
                
                flat_illegal = torch.cat(illegal_losses)
                turn_illegal = torch.zeros(num_turns, device=self.device).scatter_add_(0, flat_indices, flat_illegal) / turn_action_counts
                illegal_action_loss = turn_illegal.sum() / turn_count"""
code = code.replace(old_agg, new_agg)

with open("luxai2021/rl/ppo.py", "w") as f:
    f.write(code)
print("Patched policy_loss turn-mean calculation successfully")
