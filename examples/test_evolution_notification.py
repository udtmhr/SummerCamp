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
    failures = []
    for transport in notifier.transports:
        transport_notifier = EvolutionNotifier(Path(args.run_dir), (transport,))
        delivered = transport_notifier.notify_once(
            f"manual-test:{transport.name}:{time.time_ns()}",
            title="Lux evolution notification test",
            message=f"Notification delivery works.\nhost: {socket.gethostname()}\nrun_dir: {args.run_dir}",
            priority=3,
            tags=("white_check_mark", "test_tube"),
        )
        status = "sent" if delivered else "failed"
        print(f"{transport.name}: {status}")
        if not delivered:
            failures.append(transport.name)
    if failures:
        message = f"Notification transports failed: {', '.join(failures)}"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
