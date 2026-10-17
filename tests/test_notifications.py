import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import notifications as notify


REAL_STDLIB_POST = notify._stdlib_post


@pytest.fixture(autouse=True)
def notification_isolation(monkeypatch):
    # Tests must never contact Telegram, even when developer credentials exist.
    monkeypatch.setattr(notify, "_ENV_WARNED", False)
    def unexpected_post(*args, **kwargs):
        pytest.fail("A test attempted an unmocked Telegram request")
    monkeypatch.setattr(notify, "_stdlib_post", unexpected_post)


def clear_known_env(monkeypatch):
    for key in notify.NOTIFICATION_KEYS:
        monkeypatch.delenv(key, raising=False)


def response(status=200, body=None):
    return SimpleNamespace(status_code=status, json=lambda: {"ok": True} if body is None else body)


def test_dotenv_allowlist_quotes_comments_export_empty_and_literal_shell(tmp_path):
    path = tmp_path / ".env"
    path.write_text("""# private test-only settings
export TELEGRAM_BOT_TOKEN = '123:test#literal' # comment
TELEGRAM_BOT_CHAT_ID="${NO_INTERPOLATION} $(no_execution)" # comment
TELEGRAM_NOTIFICATIONS= # empty
UNRELATED_API_KEY=must-not-be-loaded
""")
    assert notify.read_telegram_env(path) == {
        "TELEGRAM_BOT_TOKEN": "123:test#literal",
        "TELEGRAM_BOT_CHAT_ID": "${NO_INTERPOLATION} $(no_execution)",
        "TELEGRAM_NOTIFICATIONS": "",
    }


def test_dotenv_malformed_warnings_never_echo_values_and_warn_once(tmp_path, caplog):
    path = tmp_path / ".env"
    path.write_text("TELEGRAM_BOT_TOKEN='never-print-this-secret\ninvalid never-print-this-secret\n"
                    "TELEGRAM_BOT_CHAT_ID=-12345 # safe comment\n")
    assert notify.read_telegram_env(path) == {"TELEGRAM_BOT_CHAT_ID": "-12345"}
    notify.read_telegram_env(path)
    assert len(caplog.records) == 1
    assert "never-print-this-secret" not in caplog.text
    assert str(path) not in caplog.text


def test_dotenv_process_environment_including_empty_values_wins(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("TELEGRAM_BOT_TOKEN=file-value\nTELEGRAM_BOT_CHAT_ID=file-chat\nTELEGRAM_NOTIFICATIONS=on\n")
    monkeypatch.setenv("TELEGRAM_ENV_FILE", str(path))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "process-value")
    monkeypatch.setenv("TELEGRAM_BOT_CHAT_ID", "")
    monkeypatch.delenv("TELEGRAM_NOTIFICATIONS", raising=False)
    notify.load_notification_env()
    assert notify.os.environ["TELEGRAM_BOT_TOKEN"] == "process-value"
    assert notify.os.environ["TELEGRAM_BOT_CHAT_ID"] == ""
    assert notify.os.environ["TELEGRAM_NOTIFICATIONS"] == "on"


def test_default_dotenv_is_repository_root_not_working_directory(tmp_path, monkeypatch):
    repository, working = tmp_path / "repo", tmp_path / "elsewhere"
    repository.mkdir()
    working.mkdir()
    (repository / ".env").write_text("TELEGRAM_BOT_TOKEN=repo-test-value\n")
    (working / ".env").write_text("TELEGRAM_BOT_TOKEN=wrong-working-value\n")
    monkeypatch.setattr(notify, "REPO_ROOT", repository)
    monkeypatch.chdir(working)
    clear_known_env(monkeypatch)
    monkeypatch.delenv("TELEGRAM_ENV_FILE", raising=False)
    notify.load_notification_env()
    assert notify.os.environ["TELEGRAM_BOT_TOKEN"] == "repo-test-value"


def test_constructors_do_not_implicitly_load_dotenv(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("TELEGRAM_BOT_TOKEN=file-value\nTELEGRAM_BOT_CHAT_ID=file-chat\n")
    monkeypatch.setenv("TELEGRAM_ENV_FILE", str(path))
    assert notify.TelegramSender().configured is False
    notifier = notify.ProgressNotifier("test", 1)
    assert notifier.sender.configured is False


def test_missing_default_dotenv_is_silent(tmp_path, caplog):
    assert notify.read_telegram_env(tmp_path / "absent") == {}
    assert caplog.records == []


def test_invalid_environment_path_does_not_abort_work(monkeypatch, caplog):
    class InvalidPath:
        def expanduser(self):
            raise RuntimeError("secret-path-detail")
    monkeypatch.setattr(notify, "Path", lambda value: InvalidPath())
    notify.load_notification_env()
    assert len(caplog.records) == 1
    assert "secret-path-detail" not in caplog.text


def test_sender_posts_bounded_no_redirect_request_and_truncates_message():
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response()
    sender = notify.TelegramSender("123:test-token", "-42", post=post)
    assert sender.configured
    assert sender("x" * 5000) is True
    url, kwargs = calls[0]
    assert url == "https://api.telegram.org/bot123:test-token/sendMessage"
    assert kwargs == {"json": {"chat_id": "-42", "text": "x" * 4000}, "timeout": 5.0, "follow_redirects": False}


def test_sender_missing_credentials_warns_once_and_never_posts(caplog):
    sender = notify.TelegramSender(token="", chat_id="")
    assert sender("test") is False
    assert sender("test again") is False
    assert len(caplog.records) == 1


def test_sender_errors_are_best_effort_and_never_log_url_body_or_token(caplog):
    def post(*args, **kwargs):
        raise RuntimeError("https://api.telegram.org/botLEAK-TOKEN body=LEAK-BODY")
    sender = notify.TelegramSender("LEAK-TOKEN", "LEAK-CHAT", post=post)
    assert sender("test") is False
    assert sender("test again") is False
    assert len(caplog.records) == 1
    assert not any(value in caplog.text for value in ("LEAK", "https://", "body="))


@pytest.mark.parametrize("reply", [response(302), response(429), response(200, {"ok": False}), response(200, "bad body")])
def test_sender_rejected_responses_are_best_effort(reply):
    assert notify.TelegramSender("test-token", "test-chat", post=lambda *args, **kwargs: reply)("test") is False


def test_stdlib_transport_uses_no_redirect_opener_and_timeout(monkeypatch):
    captured = {}
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def getcode(self): return 200
        def read(self, limit):
            captured["limit"] = limit
            return b'{"ok":true}'
    class FakeOpener:
        def open(self, request, timeout):
            captured.update(request=request, timeout=timeout)
            return FakeResponse()
    def opener(handler):
        captured["handler"] = handler
        return FakeOpener()
    monkeypatch.setattr(notify.urllib.request, "build_opener", opener)
    reply = REAL_STDLIB_POST("https://api.telegram.org/bottest/sendMessage", json={"text": "test"}, timeout=5)
    assert reply.status_code == 200 and reply.json() == {"ok": True}
    assert isinstance(captured["handler"], notify._NoRedirect)
    assert captured["handler"].redirect_request(None, None, 302, "", {}, "https://other.example") is None
    assert captured["timeout"] == 5 and captured["limit"] == 16_384
    assert json.loads(captured["request"].data) == {"text": "test"}
    assert captured["request"].get_method() == "POST"


def test_progress_has_start_eighths_and_exactly_one_terminal(monkeypatch):
    messages = []
    with notify.ProgressNotifier("RQ3 shard", 8, enabled=True, sender=messages.append, clock=lambda: 0) as progress:
        for count in range(1, 9):
            progress.update(count)
        assert not any("\nCompleted\n" in message for message in messages)
    progress.finish()
    progress.__exit__(None, None, None)
    assert len(messages) == 9
    assert sum("Milestone:" in message for message in messages) == 7
    assert sum("\nCompleted\n" in message for message in messages) == 1
    assert "\nStarted\n" in messages[0]


def test_resume_eta_uses_only_new_work():
    now = [100.0]
    messages = []
    with notify.ProgressNotifier("resume", 8, completed=4, enabled=True,
                                sender=messages.append, clock=lambda: now[0]) as progress:
        now[0] += 20
        progress.update(5)
        assert "ETA: ~1m" in messages[-1]  # 20s / 1 newly done * 3 remaining.
        assert "Already finished when attached: 4" in messages[-1]
        progress.update(8)
    assert "Resumed / monitoring" in messages[0]


def test_auto_threshold_and_explicit_flags(monkeypatch):
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS", "auto")
    assert not notify.ProgressNotifier("short", 19).enabled
    assert notify.ProgressNotifier("one hour", 20).enabled
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS", "off")
    assert not notify.ProgressNotifier("long", 100).enabled
    assert notify.ProgressNotifier("forced", 1, enabled=True).enabled
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS", "on")
    assert notify.ProgressNotifier("short", 1).enabled
    assert not notify.ProgressNotifier("explicit off", 100, enabled=False).enabled


def test_progress_error_is_not_masked_or_included_in_messages():
    messages = []
    with pytest.raises(RuntimeError, match="secret prompt"):
        with notify.ProgressNotifier("task", 2, enabled=True, sender=messages.append) as progress:
            progress.update(1, failed=1)
            raise RuntimeError("secret prompt and credential")
    assert "Failed; completed outputs retained" in messages[-1]
    assert "secret" not in "\n".join(messages)
    assert "Errors: 1" in messages[-1]


def test_progress_interrupt_is_not_masked():
    messages = []
    with pytest.raises(KeyboardInterrupt):
        with notify.ProgressNotifier("task", 2, enabled=True, sender=messages.append):
            raise KeyboardInterrupt()
    assert "Interrupted; completed outputs retained" in messages[-1]


def test_custom_sender_failure_does_not_interrupt_work_or_leak(caplog):
    def sender(message):
        raise RuntimeError("LEAK-TOKEN")
    with notify.ProgressNotifier("task", 8, enabled=True, sender=sender) as progress:
        progress.update(4)
        progress.update(8)
    assert len(caplog.records) == 1
    assert "LEAK-TOKEN" not in caplog.text


def test_zero_work_is_supported_and_counts_remain_valid():
    messages = []
    with notify.ProgressNotifier("nothing to do", 0, enabled=True, sender=messages.append) as progress:
        progress.update(0)
    assert len(messages) == 2 and "Progress: 0/0 (100.0%)" in messages[-1]
    with pytest.raises(ValueError, match="monotonic"):
        progress.update(0)
    with pytest.raises(ValueError, match="counts"):
        notify.ProgressNotifier("invalid", 0, completed=1)


def test_cli_flags_are_tri_state():
    parser = argparse.ArgumentParser()
    notify.add_notification_options(parser)
    assert parser.parse_args([]).notify is None
    assert parser.parse_args(["--notify"]).notify is True
    assert parser.parse_args(["--no-notify"]).notify is False


def test_check_config_only_prints_booleans_and_never_sends(tmp_path, monkeypatch, capsys):
    path = tmp_path / ".env"
    path.write_text("TELEGRAM_BOT_TOKEN=private-test-token\nTELEGRAM_BOT_CHAT_ID=private-test-chat\nTELEGRAM_NOTIFICATIONS=on\n")
    clear_known_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ENV_FILE", str(path))
    assert notify.main(["--check-config"]) == 0
    output = capsys.readouterr().out
    configuration = json.loads(output)
    assert configuration["configured"] is True
    assert all(type(value) is bool for value in configuration.values())
    assert "private-test" not in output


def test_shell_event_contains_scheduler_identity_but_no_terminal_marker(tmp_path, monkeypatch):
    messages = []
    marker = tmp_path / "terminal.json"
    monkeypatch.setenv("TELEGRAM_TERMINAL_MARKER", str(marker))
    monkeypatch.setenv("PBS_JOBID", "123[2].pbs")
    monkeypatch.setenv("PBS_ARRAY_INDEX", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    monkeypatch.setattr(notify.socket, "gethostname", lambda: "compute-test")
    assert notify.event("bootstrap", "failed", exit_code=17, output="/scratch/test", enabled=True,
                        sender=lambda message: messages.append(message) or True)
    assert all(value in messages[0] for value in ("Failed", "123[2].pbs", "PBS array index: 2", "SLURM job: 456", "compute-test", "Exit code: 17"))
    assert not marker.exists()


def test_event_cli_respects_auto_threshold_and_returns_zero_on_failed_send(monkeypatch):
    monkeypatch.setenv("TELEGRAM_NOTIFICATIONS", "auto")
    calls = []
    monkeypatch.setattr(notify, "load_notification_env", lambda: None)
    monkeypatch.setattr(notify, "TelegramSender", lambda: lambda message: calls.append(message) or False)
    assert notify.main(["event", "--label", "bootstrap", "--status", "started", "--estimated-seconds", "3599"]) == 0
    assert calls == []
    assert notify.main(["event", "--label", "bootstrap", "--status", "failed", "--exit-code", "1", "--estimated-seconds", "3600"]) == 0
    assert len(calls) == 1


def test_terminal_marker_is_written_even_when_notifications_are_disabled(tmp_path, monkeypatch):
    marker = tmp_path / "attempt/terminal.json"
    monkeypatch.setenv("TELEGRAM_TERMINAL_MARKER", str(marker))
    with notify.ProgressNotifier("quiet", 1, enabled=False) as progress:
        progress.update(1)
    assert json.loads(marker.read_text()) == {"status": "completed", "completed": 1, "total": 1, "failed": 0}
    original = marker.read_bytes()
    progress.finish("failed")
    assert marker.read_bytes() == original
    assert list(marker.parent.glob(".telegram-terminal-*")) == []


def test_failed_progress_writes_nonsecret_terminal_marker(tmp_path, monkeypatch):
    marker = tmp_path / "terminal.json"
    monkeypatch.setenv("TELEGRAM_TERMINAL_MARKER", str(marker))
    with pytest.raises(RuntimeError):
        with notify.ProgressNotifier("sensitive label", 1, enabled=False):
            raise RuntimeError("sensitive error text")
    data = json.loads(marker.read_text())
    assert data["status"] == "failed"
    assert "sensitive" not in marker.read_text()


def test_marker_io_failure_is_best_effort_and_sanitized(tmp_path, monkeypatch, caplog):
    parent = tmp_path / "not-a-directory"
    parent.write_text("preserve me")
    monkeypatch.setenv("TELEGRAM_TERMINAL_MARKER", str(parent / "secret-filename"))
    with notify.ProgressNotifier("task", 0, enabled=False):
        pass
    assert parent.read_text() == "preserve me"
    assert "secret-filename" not in caplog.text
