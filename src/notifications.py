"""Best-effort Telegram job notifications, independent of scientific protocols.

Loading credentials is explicit: CLI entry points call load_notification_env().
Constructing a sender/notifier never reads a dotenv file or contacts a service.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import time
import urllib.parse
import urllib.request


LOGGER = logging.getLogger(__name__)
NOTIFICATION_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_CHAT_ID", "TELEGRAM_NOTIFICATIONS")
REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_WARNED = False


def _warn_environment():
    global _ENV_WARNED
    if not _ENV_WARNED:
        LOGGER.warning("Telegram settings could not be fully read; check the notification environment file")
        _ENV_WARNED = True


def _dotenv_value(value):
    value = value.strip()
    if not value or value.startswith("#"):
        return ""
    if value[0] not in ("'", '"'):
        # Comments begin after whitespace; embedded hashes remain literal.
        result = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if any(character.isspace() for character in result):
            raise ValueError("Malformed notification setting")
        return result
    quote, result, index = value[0], [], 1
    while index < len(value):
        character = value[index]
        if character == quote:
            rest = value[index + 1:].strip()
            if rest and not rest.startswith("#"):
                raise ValueError("Malformed notification setting")
            return "".join(result)
        if quote == '"' and character == "\\" and index + 1 < len(value) and value[index + 1] in ('"', "\\"):
            index += 1
            character = value[index]
        result.append(character)
        index += 1
    raise ValueError("Malformed notification setting")


def read_telegram_env(path) -> dict[str, str]:
    """Read only three allowlisted keys; never execute or interpolate values.

    Supports optional export, single/double quotes, comments and empty values.
    Unsupported/malformed entries are ignored with one value-free warning.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        _warn_environment()
        return {}
    result = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", stripped)
        if match is None:
            _warn_environment()
            continue
        key, value = match.groups()
        if key not in NOTIFICATION_KEYS:
            continue
        try:
            result[key] = _dotenv_value(value)
        except ValueError:
            _warn_environment()
    return result


def load_notification_env() -> None:
    """Load repo-root .env (or TELEGRAM_ENV_FILE); process values always win."""
    try:
        configured = os.environ.get("TELEGRAM_ENV_FILE")
        path = Path(configured).expanduser() if configured else REPO_ROOT / ".env"
        if configured and not path.is_file():
            _warn_environment()
            return
        values = read_telegram_env(path)
    except (OSError, ValueError, RuntimeError):
        _warn_environment()
        return
    for key, value in values.items():
        try:
            os.environ.setdefault(key, value)
        except ValueError:
            _warn_environment()


def add_notification_options(parser):
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=None,
                        help="force Telegram alerts (or --no-notify); default: jobs estimated at >=1 hour")


def duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h {minutes % 60:02d}m"


def notifications_enabled(enabled=None, *, estimated_seconds=0):
    if enabled is not None:
        return bool(enabled)
    mode = os.environ.get("TELEGRAM_NOTIFICATIONS", "auto").strip().lower()
    if mode in ("on", "true", "1"):
        return True
    if mode in ("off", "false", "0"):
        return False
    if mode == "auto":
        return estimated_seconds >= 3600
    _warn_environment()
    return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


@dataclass
class _Response:
    status_code: int
    body: object

    def json(self):
        return self.body


def _stdlib_post(url, *, json: dict, timeout=5.0, follow_redirects=False):
    if follow_redirects:
        raise ValueError("Telegram redirects are disabled")
    # Avoid a third-party HTTP dependency: this also runs before Conda setup.
    import json as json_module
    request = urllib.request.Request(url, data=json_module.dumps(json).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=min(float(timeout), 5.0)) as response:
        body = response.read(16_384)
        return _Response(response.getcode(), json_module.loads(body))


class TelegramSender:
    """Five-second HTTPS requests, no redirects, no credential/error logging."""

    def __init__(self, token=None, chat_id=None, *, post=None):
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN") if token is None else str(token)
        self._chat_id = os.environ.get("TELEGRAM_BOT_CHAT_ID") if chat_id is None else str(chat_id)
        self._post = _stdlib_post if post is None else post
        self._warned = False

    @property
    def configured(self):
        return bool(self._token and self._token.strip() and self._chat_id and self._chat_id.strip())

    def __call__(self, message):
        if not self.configured:
            if not self._warned:
                LOGGER.warning("Telegram disabled: set TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_CHAT_ID")
                self._warned = True
            return False
        try:
            token = urllib.parse.quote(self._token, safe=":")
            response = self._post(f"https://api.telegram.org/bot{token}/sendMessage",
                                  json={"chat_id": self._chat_id, "text": str(message)[:4000]},
                                  timeout=5.0, follow_redirects=False)
            if response.status_code != 200 or response.json().get("ok") is not True:
                raise ValueError("Telegram rejected the message")
            return True
        except Exception:
            # urllib/HTTP exceptions can embed token-bearing URLs and bodies.
            if not self._warned:
                LOGGER.warning("Telegram notification failed; processing will continue")
                self._warned = True
            return False


def _job_context():
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = "unknown"
    lines = [f"Host: {hostname}"]
    for label, key in (("PBS job", "PBS_JOBID"), ("PBS array index", "PBS_ARRAY_INDEX"),
                       ("SLURM job", "SLURM_JOB_ID"), ("SLURM array index", "SLURM_ARRAY_TASK_ID")):
        value = os.environ.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


class ProgressNotifier:
    """Start/eighth milestones/one terminal; ETA counts only newly done work."""

    def __init__(self, label, total, *, estimated_seconds_per_item=180, completed=0,
                 failed=0, enabled=None, output=None, sender=None, clock=time.monotonic):
        if (any(type(value) is not int for value in (total, completed, failed))
                or total < 0 or not 0 <= completed <= total or not 0 <= failed <= completed):
            raise ValueError("invalid progress counts")
        if (not isinstance(estimated_seconds_per_item, (int, float))
                or not math.isfinite(estimated_seconds_per_item) or estimated_seconds_per_item < 0):
            raise ValueError("estimated_seconds_per_item must be finite and nonnegative")
        self.label, self.total, self.completed, self.failed = label, total, completed, failed
        self.initial = completed
        self.estimate = estimated_seconds_per_item
        self.output = str(output) if output is not None else None
        self.enabled = notifications_enabled(enabled, estimated_seconds=total * self.estimate)
        self.sender = sender if sender is not None else TelegramSender()
        self.clock = clock
        self.started_at = None
        self.bucket = completed * 8 // total if total else 8
        self.closed = False
        self._warned = False
        self._terminal_marker = os.environ.get("TELEGRAM_TERMINAL_MARKER")

    def _emit(self, status, *, terminal=False):
        if not self.enabled:
            return
        elapsed = max(0, self.clock() - self.started_at) if self.started_at is not None else 0
        done_now = self.completed - self.initial
        eta = ((elapsed / done_now) if done_now else self.estimate) * (self.total - self.completed)
        percent = 100 * self.completed / self.total if self.total else 100.0
        message = (f"[ai6132] {self.label}\n{status}\n"
                   f"Progress: {self.completed}/{self.total} ({percent:.1f}%)\n"
                   f"Errors: {self.failed} | Elapsed: {duration(elapsed)}")
        if not terminal:
            message += f" | ETA: ~{duration(eta)}"
        if self.initial:
            message += f"\nAlready finished when attached: {self.initial}"
        if self.output:
            message += f"\nOutput: {self.output}"
        message += "\n" + _job_context()
        try:
            self.sender(message)
        except Exception:
            if not self._warned:
                LOGGER.warning("Telegram notification failed; processing will continue")
                self._warned = True

    def _write_terminal_marker(self, status):
        if not self._terminal_marker:
            return
        temporary = None
        try:
            path = Path(self._terminal_marker)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".telegram-terminal-", delete=False) as handle:
                temporary = handle.name
                json.dump({"status": status, "completed": self.completed, "total": self.total, "failed": self.failed}, handle)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            # Notification bookkeeping must never fail a scientific workload.
            LOGGER.warning("Telegram terminal status could not be recorded; processing will continue")
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def __enter__(self):
        if self.started_at is not None or self.closed:
            raise ValueError("A progress notifier can be entered only once")
        self.started_at = self.clock()
        self._emit("Resumed / monitoring" if self.initial else "Started")
        return self

    def update(self, completed, *, failed=0):
        if (self.closed or type(completed) is not int or type(failed) is not int
                or not self.completed <= completed <= self.total or not self.failed <= failed <= completed):
            raise ValueError("progress must be monotonic and within the workload")
        self.completed, self.failed = completed, failed
        bucket = completed * 8 // self.total if self.total else 8
        if bucket > self.bucket:
            self.bucket = bucket
            if bucket < 8:
                self._emit(f"Milestone: {bucket * 12.5:g}%")

    def finish(self, status=None):
        if self.closed:
            return
        if status == "complete":
            status = "completed"
        if status is None:
            status = "completed" if self.completed == self.total and not self.failed else "failed"
        if status not in {"completed", "failed", "interrupted"}:
            raise ValueError("Unsupported terminal notification status")
        # Mark closed first so reentrant/custom senders cannot duplicate alerts.
        self.closed = True
        text = {"completed": "Completed", "failed": "Failed; completed outputs retained",
                "interrupted": "Interrupted; completed outputs retained"}[status]
        try:
            self._emit(text, terminal=True)
        finally:
            self._write_terminal_marker(status)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish()
        elif issubclass(exc_type, KeyboardInterrupt) or isinstance(exc, SystemExit) and exc.code in (130, 143):
            self.finish("interrupted")
        else:
            self.finish("failed")
        return False


def event(label, status, *, exit_code=None, output=None, estimated_seconds=0, enabled=None, sender=None):
    """One shell lifecycle alert; does not create a progress terminal marker."""
    if status not in {"started", "completed", "failed", "interrupted"}:
        raise ValueError("Unsupported event status")
    if not notifications_enabled(enabled, estimated_seconds=estimated_seconds):
        return False
    message = f"[ai6132] {label}\n{status.capitalize()}\n{_job_context()}"
    if exit_code is not None:
        message += f"\nExit code: {exit_code}"
    if output is not None:
        message += f"\nOutput: {output}"
    try:
        return bool((sender if sender is not None else TelegramSender())(message))
    except Exception:
        LOGGER.warning("Telegram notification failed; processing will continue")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true", help="Print configured-key booleans only; never send a request")
    commands = parser.add_subparsers(dest="command")
    command = commands.add_parser("event", help="Best-effort scheduler lifecycle notification")
    command.add_argument("--label")
    command.add_argument("--status", choices=["started", "completed", "failed", "interrupted"])
    command.add_argument("--exit-code", type=int)
    command.add_argument("--output")
    command.add_argument("--estimated-seconds", type=int, default=0)
    command.add_argument("--check-config", action="store_true", default=argparse.SUPPRESS)
    add_notification_options(command)
    args = parser.parse_args(argv)
    load_notification_env()
    if args.check_config:
        print(json.dumps({"configured": TelegramSender().configured,
                          **{key: bool(os.environ.get(key, "").strip()) for key in NOTIFICATION_KEYS}}, sort_keys=True))
        return 0
    if args.command != "event" or not args.label or args.status is None:
        parser.error("Choose --check-config or event with --label and --status")
    if args.estimated_seconds < 0:
        parser.error("--estimated-seconds must be nonnegative")
    event(args.label, args.status, exit_code=args.exit_code, output=args.output,
          estimated_seconds=args.estimated_seconds, enabled=args.notify)
    # Telegram failures never replace the scheduler command's actual exit code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
