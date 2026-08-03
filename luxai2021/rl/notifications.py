from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path


class NotificationTransport(Protocol):
    name: str

    def send(self, *, title: str, message: str, priority: int, tags: tuple[str, ...]) -> None: ...


@dataclass(frozen=True)
class NtfyTransport:
    server: str
    topic: str
    token: str | None = None
    timeout_seconds: float = 15.0
    name: str = "ntfy"

    def send(self, *, title: str, message: str, priority: int, tags: tuple[str, ...]) -> None:
        parsed_server = urllib.parse.urlsplit(self.server)
        if parsed_server.scheme not in {"http", "https"} or not parsed_server.netloc:
            raise ValueError("Ntfy server must be an HTTP(S) URL")
        if parsed_server.scheme != "https" and parsed_server.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Remote ntfy servers must use HTTPS")
        payload = json.dumps(
            {
                "topic": self.topic,
                "title": title,
                "message": message,
                "priority": max(1, min(int(priority), 5)),
                "tags": list(tags),
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(  # noqa: S310 - scheme and host are validated above.
            self.server.rstrip("/") + "/",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            response.read()


class EvolutionNotifier:
    def __init__(
        self,
        run_dir: Path,
        transports: tuple[NotificationTransport, ...] = (),
        *,
        configuration_warnings: tuple[str, ...] = (),
    ) -> None:
        self.run_dir = run_dir
        self.transports = transports
        self.configuration_warnings = configuration_warnings
        self.state_path = run_dir / "notifications" / "sent.json"
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, run_dir: Path) -> EvolutionNotifier:
        transports: list[NotificationTransport] = []
        ntfy_topic = os.environ.get("LUX_EVOLUTION_NTFY_TOPIC", "").strip()
        ntfy_server = os.environ.get("LUX_EVOLUTION_NTFY_SERVER", "https://ntfy.sh").strip()
        if ntfy_topic:
            transports.append(
                NtfyTransport(
                    ntfy_server,
                    ntfy_topic,
                    token=os.environ.get("LUX_EVOLUTION_NTFY_TOKEN") or None,
                )
            )
        return cls(run_dir, tuple(transports))

    @property
    def enabled(self) -> bool:
        return bool(self.transports)

    def _load_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {"schema_version": 1, "events": {}}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"schema_version": 1, "events": {}}
        if not isinstance(value, dict) or not isinstance(value.get("events"), dict):
            return {"schema_version": 1, "events": {}}
        return value

    def _save_state(self, state: dict[str, object]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)

    def notify_once(
        self,
        event_id: str,
        *,
        title: str,
        message: str,
        priority: int = 3,
        tags: tuple[str, ...] = (),
    ) -> bool:
        if not self.transports:
            return False
        delivered = False
        with self._lock:
            state = self._load_state()
            events = state["events"]
            if not isinstance(events, dict):
                raise TypeError("Notification state events must be a mapping")
            event = events.get(event_id, {})
            sent = set(event.get("transports", ())) if isinstance(event, dict) else set()
            for transport in self.transports:
                if transport.name in sent:
                    continue
                try:
                    transport.send(title=title, message=message, priority=priority, tags=tags)
                except Exception as error:  # noqa: BLE001 - notifications must never mask training failures.
                    detail = {
                        "notification": "failed",
                        "transport": transport.name,
                        "error_type": type(error).__name__,
                    }
                    print(json.dumps(detail, sort_keys=True), file=sys.stderr)
                    continue
                delivered = True
                sent.add(transport.name)
                events[event_id] = {
                    "title": title,
                    "transports": sorted(sent),
                    "sent_at": time.time(),
                }
                self._save_state(state)
        return delivered
