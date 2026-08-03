# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, ANN204, S101, S106
from __future__ import annotations

import json

from luxai2021.rl.notifications import EvolutionNotifier, GmailTransport, NtfyTransport


class _RecordingTransport:
    name = "recording"

    def __init__(self):
        self.messages = []

    def send(self, **message):
        self.messages.append(message)


class _FailingTransport:
    name = "failing"

    def send(self, **_):
        raise OSError("offline")


def test_notifier_deduplicates_success_and_does_not_persist_secrets(tmp_path):
    transport = _RecordingTransport()
    notifier = EvolutionNotifier(tmp_path, (_FailingTransport(), transport))

    first = notifier.notify_once("generation-01", title="done", message="details")
    second = notifier.notify_once("generation-01", title="done", message="details")

    assert first is True
    assert second is False
    assert len(transport.messages) == 1
    state_text = notifier.state_path.read_text()
    state = json.loads(state_text)
    assert state["events"]["generation-01"]["transports"] == ["recording"]
    assert "password" not in state_text.lower()


def test_gmail_transport_uses_app_password_without_spaces(monkeypatch):
    calls = {}

    class FakeSmtp:
        def __init__(self, host, port, *, timeout):
            calls["connection"] = (host, port, timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def login(self, username, password):
            calls["login"] = (username, password)

        def send_message(self, message):
            calls["message"] = message

    monkeypatch.setattr("luxai2021.rl.notifications.smtplib.SMTP_SSL", FakeSmtp)
    transport = GmailTransport("sender@gmail.com", "abcd efgh ijkl mnop", ("phone@gmail.com",))

    transport.send(title="Lux failed", message="traceback", priority=5, tags=("x",))

    assert calls["connection"][:2] == ("smtp.gmail.com", 465)
    assert calls["login"] == ("sender@gmail.com", "abcdefghijklmnop")
    assert calls["message"]["To"] == "phone@gmail.com"
    assert calls["message"].get_content().strip() == "traceback"


def test_ntfy_transport_posts_json_with_bearer_token(monkeypatch):
    calls = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self):
            return b"ok"

    def fake_urlopen(request, *, timeout):
        calls["request"] = request
        calls["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("luxai2021.rl.notifications.urllib.request.urlopen", fake_urlopen)
    transport = NtfyTransport("https://ntfy.example", "private-topic", token="secret-token")

    transport.send(title="Generation done", message="score=0.6", priority=4, tags=("tada",))

    request = calls["request"]
    payload = json.loads(request.data)
    assert request.full_url == "https://ntfy.example/"
    assert request.get_header("Authorization") == "Bearer secret-token"
    assert payload == {
        "topic": "private-topic",
        "title": "Generation done",
        "message": "score=0.6",
        "priority": 4,
        "tags": ["tada"],
    }


def test_notifier_environment_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("LUX_EVOLUTION_GMAIL_USER", "sender@gmail.com")
    monkeypatch.setenv("LUX_EVOLUTION_GMAIL_APP_PASSWORD", "app-password")
    monkeypatch.setenv("LUX_EVOLUTION_NOTIFY_EMAIL_TO", "one@example.com,two@example.com")
    monkeypatch.setenv("LUX_EVOLUTION_NTFY_TOPIC", "phone-topic")

    notifier = EvolutionNotifier.from_environment(tmp_path)

    assert [transport.name for transport in notifier.transports] == ["gmail", "ntfy"]
    assert notifier.configuration_warnings == ()
