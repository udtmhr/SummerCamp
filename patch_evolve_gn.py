import re

with open("examples/evolve_rl.py", "r") as f:
    code = f.read()

old_metrics_update = """            metrics = trainer.update(episodes)"""
new_metrics_update = """            is_first_update = (update == 0)
            is_last_update = (decision_budget is not None and target_update_decisions is not None and cumulative_decisions + target_update_decisions >= decision_budget)
            record_grad_norms = is_first_update or is_last_update
            metrics = trainer.update(episodes, record_grad_norms=record_grad_norms)"""
code = code.replace(old_metrics_update, new_metrics_update)

with open("examples/evolve_rl.py", "w") as f:
    f.write(code)
print("Patched evolve_rl.py successfully for record_grad_norms")
