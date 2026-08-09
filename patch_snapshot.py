import re

with open("examples/evolve_rl.py", "r") as f:
    code = f.read()

old_logic = """            if checkpoint_callback is not None:
                checkpoint_callback(output_dir, metrics)
            update += 1
            if decision_budget is None and seconds <= 0:"""

new_logic = """            if checkpoint_callback is not None:
                checkpoint_callback(output_dir, metrics)
            
            if update > 0 and update % 10 == 0:
                snapshot.load_state_dict(actor_critic.state_dict())
                
            update += 1
            if decision_budget is None and seconds <= 0:"""
code = code.replace(old_logic, new_logic)

with open("examples/evolve_rl.py", "w") as f:
    f.write(code)
print("Patched snapshot successfully")
