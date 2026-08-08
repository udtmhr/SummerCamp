"""Lux AI 2021 package initialization."""

from __future__ import annotations

import os

# Avoid fork-based TorchInductor compile workers. Forking a multithreaded
# CUDA process can deadlock on inherited runtime locks. ``spawn`` changes only
# compiler worker startup; generated/compiled kernels and runtime performance
# are unchanged. Respect an explicit user override when one is provided.
os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
