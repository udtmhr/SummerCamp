"""Full-turn reinforcement learning and Codex-guided reward evolution."""

import os

# ``torch.compile``/Inductor can lazily create compile workers from the rollout
# inference thread after CUDA and other Python threads are already active.  A
# fork-based worker is unsafe in that state and can inherit locked runtime/CUDA
# state, leaving ``torch._inductor.compile_worker --kind=fork`` processes stuck.
#
# ``spawn`` only changes how Inductor's compiler workers are created; generated
# kernels and the steady-state performance of compiled models are unchanged.
# Respect an explicit user setting for debugging/benchmarking.
os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")

from luxai2021.rl.metrics import GameMetrics, metrics_from_game
from luxai2021.rl.reward import RewardBreakdown, RewardProgram, default_reward_program

__all__ = (
    "GameMetrics",
    "RewardBreakdown",
    "RewardProgram",
    "default_reward_program",
    "metrics_from_game",
)
