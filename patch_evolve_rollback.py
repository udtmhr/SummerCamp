import re

with open("examples/evolve_rl.py", "r") as f:
    code = f.read()

# 1. Early Stop flag
old_loop_init = """    saved_milestones = set()
    epochs_without_improvement = 0
    best_teacher_score_rate = -1.0"""
new_loop_init = """    saved_milestones = set()
    epochs_without_improvement = 0
    best_teacher_score_rate = -1.0
    should_early_stop = False"""
code = code.replace(old_loop_init, new_loop_init)

old_loop_cond = """        while (
            cumulative_decisions < decision_budget
            if decision_budget is not None
            else time.monotonic() < deadline or update == 0
        ):"""
new_loop_cond = """        while not should_early_stop and (
            cumulative_decisions < decision_budget
            if decision_budget is not None
            else time.monotonic() < deadline or update == 0
        ):"""
code = code.replace(old_loop_cond, new_loop_cond)

old_early_stop = """                        if epochs_without_improvement >= 2:
                            print(f"Early stopping at milestone {milestone}. Best: {best_teacher_score_rate:.3f}, Current: {score_rate:.3f}")
                            decision_budget = 0 # trigger early stop
                            break"""
new_early_stop = """                        if epochs_without_improvement >= 2:
                            print(f"Early stopping at milestone {milestone}. Best: {best_teacher_score_rate:.3f}, Current: {score_rate:.3f}")
                            should_early_stop = True
                            break"""
code = code.replace(old_early_stop, new_early_stop)

# 2. Change latest_rl.pt to best.pt for prior checkpoints
# Lines to replace:
# prior_short_checkpoint = ( ... / "latest_rl.pt" )
# Path(...) / "medium-resattn8" / "resattn8" / "latest_rl.pt"
# Path(...) / "probe-unet" / "unet" / "latest_rl.pt"

code = code.replace(
    'candidate.candidate_id / "short-resattn8" / "resattn8" / "latest_rl.pt"',
    'candidate.candidate_id / "short-resattn8" / "resattn8" / "best.pt"'
)
code = code.replace(
    'candidate.candidate_id / "medium-resattn8" / "resattn8" / "latest_rl.pt"',
    'candidate.candidate_id / "medium-resattn8" / "resattn8" / "best.pt"'
)
code = code.replace(
    'candidate.candidate_id / "probe-unet" / "unet" / "latest_rl.pt"',
    'candidate.candidate_id / "probe-unet" / "unet" / "best.pt"'
)

with open("examples/evolve_rl.py", "w") as f:
    f.write(code)
print("Patched evolve_rl.py early stop and rollback successfully")
