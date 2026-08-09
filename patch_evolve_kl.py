import re

with open("examples/evolve_rl.py", "r") as f:
    code = f.read()

old_actor_critic = """    base_checkpoint_sha256 = _sha256_file(base_checkpoint)
    _, base_source = load_bc_checkpoint(str(base_checkpoint), "cpu")
    actor_critic = FullTurnActorCritic.from_checkpoint(base_checkpoint, device)
    inherited_modules: list[str] = []"""

new_actor_critic = """    base_checkpoint_sha256 = _sha256_file(base_checkpoint)
    _, base_source = load_bc_checkpoint(str(base_checkpoint), "cpu")
    actor_critic = FullTurnActorCritic.from_checkpoint(base_checkpoint, device)
    
    reference_policy = None
    if candidate.ppo_config.kl_coefficient > 0:
        reference_policy = FullTurnActorCritic.from_checkpoint(base_checkpoint, device).policy
        reference_policy.eval()
        for param in reference_policy.parameters():
            param.requires_grad = False
            
    inherited_modules: list[str] = []"""
code = code.replace(old_actor_critic, new_actor_critic)

old_make_trainer = """    def make_trainer() -> PPOTrainer:
        return PPOTrainer(
            actor_critic,
            candidate.ppo_config,
            device,
            bc_batch_provider=bc_provider,
        )"""

new_make_trainer = """    def make_trainer() -> PPOTrainer:
        return PPOTrainer(
            actor_critic,
            candidate.ppo_config,
            device,
            reference_policy=reference_policy,
            bc_batch_provider=bc_provider,
        )"""
code = code.replace(old_make_trainer, new_make_trainer)

with open("examples/evolve_rl.py", "w") as f:
    f.write(code)
print("Patched evolve_rl.py successfully")
