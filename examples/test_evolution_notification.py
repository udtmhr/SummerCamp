from __future__ import annotations

# ruff: noqa: INP001
import argparse
import socket
import time
from pathlib import Path

from luxai2021.rl.notifications import EvolutionNotifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a Lux evolution notification smoke test.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    notifier = EvolutionNotifier.from_environment(Path(args.run_dir))
    if notifier.configuration_warnings:
        raise ValueError("; ".join(notifier.configuration_warnings))
    if not notifier.enabled:
        raise RuntimeError("No notification transport is configured")
    event_id = f"manual-test:{time.time_ns()}"
    delivered = notifier.notify_once(
        event_id,
        title="Lux evolution notification test",
        message=f"Notification delivery works.\nhost: {socket.gethostname()}\nrun_dir: {args.run_dir}",
        priority=3,
        tags=("white_check_mark", "test_tube"),
    )
    if not delivered:
        raise RuntimeError("Every configured notification transport failed")
    print("notification sent")


if __name__ == "__main__":
    main()
