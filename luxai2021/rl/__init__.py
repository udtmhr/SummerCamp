"""Full-turn reinforcement learning and Codex-guided reward evolution."""

import os

# Inductor may initialize its compiler pool after CUDA and rollout threads are
# already active.  Never allow a fork-based pool to inherit those locks.
if os.environ.get("TORCHINDUCTOR_WORKER_START") in {None, "fork"}:
    os.environ["TORCHINDUCTOR_WORKER_START"] = "subprocess"

from luxai2021.rl.metrics import GameMetrics, metrics_from_game
from luxai2021.rl.reward import RewardBreakdown, RewardProgram, default_reward_program

__all__ = (
    "GameMetrics",
    "RewardBreakdown",
    "RewardProgram",
    "default_reward_program",
    "metrics_from_game",
)
