"""Keep real notification credentials out of all test runs."""

import pytest


@pytest.fixture(autouse=True)
def isolate_telegram_settings(monkeypatch, tmp_path):
    # An ignored developer .env may contain the real bot. Individual tests
    # opt into mocked senders and fixture .env files, never that local file.
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS", "off")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT_ID", "")
    monkeypatch.setenv("TELEGRAM_ENV_FILE", str(tmp_path / "no-real-telegram.env"))
    monkeypatch.delenv("TELEGRAM_TERMINAL_MARKER", raising=False)
