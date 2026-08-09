import re

with open("examples/evolve_rl.py", "r") as f:
    code = f.read()

# 1. Add args to train_candidate
old_def = """def train_candidate(
    candidate: EvolutionCandidate,
    *,
    base_name: str,
    base_checkpoint: Path,
    other_checkpoint: Path,
    teacher_checkpoint: Path,
    teacher_cache_dir: Path,"""

new_def = """def train_candidate(
    candidate: EvolutionCandidate,
    *,
    base_name: str,
    base_checkpoint: Path,
    other_checkpoint: Path,
    teacher_checkpoint: Path,
    teacher_cache_dir: Path,
    eval_seeds: int = 0,
    eval_seed_start: int = 0,"""
code = code.replace(old_def, new_def)

# 2. Modify milestone_points and add best_teacher_score_rate
old_milestones = """    milestone_points = (0.30, 0.60, 0.80, 1.00)
    saved_milestones = {"""

new_milestones = """    milestone_points = (0.20, 0.40, 0.60, 0.80, 1.00)
    best_teacher_score_rate = -1.0
    epochs_without_improvement = 0
    saved_milestones = {"""
code = code.replace(old_milestones, new_milestones)

# 3. Add evaluation logic inside the milestone loop
old_milestone_loop = """                    saved_milestones.add(milestone)
            if checkpoint_callback is not None:"""

new_milestone_loop = """                    saved_milestones.add(milestone)
                    
                    if eval_seeds > 0:
                        from luxai2021.rl.evaluation import evaluate_against_league, LeagueMember
                        import shutil
                        milestone_path = milestone_dir / f"p{round(milestone * 100):03d}.pt"
                        teacher = LeagueMember("first-place", teacher_checkpoint, model_type="first-place")
                        evaluation = evaluate_against_league(
                            LeagueMember("milestone", milestone_path),
                            [teacher],
                            seed_start=eval_seed_start,
                            seed_count=eval_seeds,
                            device=str(device),
                            max_turns=max_turns,
                        )
                        score_rate = float(evaluation["totals"]["score_rate"])
                        if score_rate > best_teacher_score_rate:
                            best_teacher_score_rate = score_rate
                            shutil.copyfile(milestone_path, output_dir / "best.pt")
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1
                        
                        if epochs_without_improvement >= 2:
                            print(f"Early stopping at milestone {milestone}. Best: {best_teacher_score_rate:.3f}, Current: {score_rate:.3f}")
                            decision_budget = 0 # trigger early stop
                            break
            if checkpoint_callback is not None:"""
code = code.replace(old_milestone_loop, new_milestone_loop)

# 4. Modify evaluate_stage to pass eval_seeds
old_call = """        checkpoint, training = train_candidate(
            candidate,
            base_name=base_name,
            base_checkpoint=base_checkpoint,
            other_checkpoint=other_checkpoint,
            teacher_checkpoint=Path(args.teacher_checkpoint),"""

new_call = """        checkpoint, training = train_candidate(
            candidate,
            base_name=base_name,
            base_checkpoint=base_checkpoint,
            other_checkpoint=other_checkpoint,
            teacher_checkpoint=Path(args.teacher_checkpoint),
            eval_seeds=min(eval_seeds, args.screening_seeds),
            eval_seed_start=eval_seed_start,"""
code = code.replace(old_call, new_call)

# 5. Prevent overwriting best.pt at the end if it was already saved
old_export = """    actor_critic.export_policy(
        output_dir / "best.pt",
        epoch=max(0, update - 1),
        metrics={"validation": final_metrics, "ppo": final_metrics},
        split=base_source["split"],
        metadata=summary,
    )"""

new_export = """    if not (output_dir / "best.pt").exists():
        actor_critic.export_policy(
            output_dir / "best.pt",
            epoch=max(0, update - 1),
            metrics={"validation": final_metrics, "ppo": final_metrics},
            split=base_source["split"],
            metadata=summary,
        )"""
code = code.replace(old_export, new_export)

with open("examples/evolve_rl.py", "w") as f:
    f.write(code)
print("Patched successfully")
