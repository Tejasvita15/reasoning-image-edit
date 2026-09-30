"""Scoring progress integration with fake judges/senders; never sends HTTP."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.notifications import ProgressNotifier
from src.rq3 import evaluate


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@contextmanager
def scoring_fixture(total=8, *, execute=True, cached=0, extra_cached=False, notify=True):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        image = root / "output.png"
        image.write_bytes(b"fake image; annotation decoding is mocked")
        digest = evaluate.file_sha256(image)
        rows = [{"example_uid": f"biology__{index}", "condition": "A", "base_seed": 42,
                 "protocol_hash": "fixture", "repeat": 0, "status": "ok", "output_path": str(image),
                 "source_path": str(image), "output_sha256": digest, "instruction": "fixture",
                 "knowledge_type": "conceptual"} for index in range(total)]
        config = {"protocol_hash": "fixture", "_splits": {"k_dev": [row["example_uid"] for row in rows],
                                                            "k_small": [row["example_uid"] for row in rows]},
                  "_judge": {"official_revision": evaluate.OFFICIAL_REVISION, "official_checkout": "unused",
                             "aggregate": "mean_applicable_normalized", "aggregate_confirmed": True,
                             "temperature": 0, "model": "gpt-4o-2024-08-06", "max_tokens": 1000},
                  "statistics": {"equivalence_margin": 5, "minimum_meaningful_difference": 1}}
        result = {"Q": 75.0, "consistency_score": 4, "instruction_score": 4, "quality_score": 4,
                  "knowledge_score": 4, "prompt_sha256": "prompt-fixture", "official_family": "knowledge",
                  "raw_responses": [{"response": {"fixture": True}} for _ in range(3)]}
        def cached_row(row):
            return {**{key: row[key] for key in ("example_uid", "condition", "base_seed", "protocol_hash")},
                    "output_sha256": digest, "judge_model": config["_judge"]["model"],
                    "aggregate": config["_judge"]["aggregate"], **result}
        output = root / "scores.jsonl"
        existing = [cached_row(row) for row in rows[:cached]]
        if extra_cached:
            existing.append(cached_row({**rows[0], "example_uid": "biology__outside"}))
        if existing:
            write_rows(output, existing)
        generations = root / "generations.jsonl"
        # Requested quality rows are the denominator. Timing repetitions,
        # another seed and an out-of-split UID are never scoring work.
        extras = [{**rows[0], "repeat": 1}, {**rows[0], "base_seed": 7},
                  {**rows[0], "example_uid": "biology__outside"}]
        write_rows(generations, rows + extras)
        args = SimpleNamespace(config="fixture.yaml", generations=str(generations), output=str(output), seeds=[42],
                               calibration=False, limit=None, execute=execute, notify=notify)
        messages, persisted_at_notification, progress_calls = [], [], []
        now = [0.0]
        def capture(message):
            messages.append(message)
            persisted_at_notification.append(len(evaluate.read_jsonl(output)) if output.exists() else 0)
        def progress_factory(*arguments, **kwargs):
            progress_calls.append((arguments, dict(kwargs)))
            return ProgressNotifier(*arguments, **kwargs, sender=capture, clock=lambda: now[0])
        def judge_one(row):
            now[0] += 10
            return dict(result)
        judge = SimpleNamespace(prompt_sha256="prompt-fixture", annotation=Mock(),
                                family=Mock(return_value="knowledge"), evaluate=Mock(side_effect=judge_one))
        create_client = Mock(return_value=object())
        progress_factory_mock = Mock(side_effect=progress_factory)
        stack.enter_context(patch("src.rq3.protocol.load_protocol", return_value=config))
        stack.enter_context(patch("src.rq3.protocol.build_jobs", return_value=rows))
        stack.enter_context(patch.object(evaluate, "OfficialEvaluator", return_value=judge))
        stack.enter_context(patch.object(evaluate, "ProgressNotifier", progress_factory_mock))
        stack.enter_context(patch.dict("sys.modules", {"openai": SimpleNamespace(OpenAI=create_client)}))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(args=args, config=config, rows=rows, judge=judge, result=result, messages=messages,
                              persisted=persisted_at_notification, progress_calls=progress_calls,
                              progress_factory=progress_factory_mock, create_client=create_client,
                              output=output, root=root, now=now)


class EvaluateNotificationTests(unittest.TestCase):
    def test_live_scoring_reports_eighths_after_persisting_requested_quality_rows(self):
        with scoring_fixture() as fixture:
            evaluate.score(fixture.args)
            self.assertEqual(fixture.judge.evaluate.call_count, 8)
            self.assertEqual(len(fixture.messages), 9)
            self.assertIn("0/8", fixture.messages[0])
            self.assertIn("Completed", fixture.messages[-1])
            self.assertEqual(fixture.persisted, list(range(9)))
            self.assertEqual([message.splitlines()[1] for message in fixture.messages[1:-1]],
                             [f"Milestone: {index * 12.5:g}%" for index in range(1, 8)])
            self.assertEqual(fixture.progress_calls[0][1]["estimated_seconds_per_item"], 60)

    def test_resume_counts_only_selected_valid_cache_and_excludes_it_from_eta(self):
        with scoring_fixture(cached=2, extra_cached=True) as fixture:
            evaluate.score(fixture.args)
            self.assertEqual(fixture.judge.evaluate.call_count, 6)
            self.assertEqual(fixture.progress_calls[0][1]["completed"], 2)
            self.assertIn("2/8", fixture.messages[0])
            self.assertIn("Milestone: 37.5%", fixture.messages[1])
            # One new output took 10 seconds, five remain: cached work must not
            # turn this into an erroneously short 10/3 * 5 second estimate.
            self.assertIn("ETA: ~50s", fixture.messages[1])
            self.assertEqual(fixture.persisted[-1], 9)  # 8 selected + 1 unrelated cache row.
            self.assertNotIn("Milestone: 12.5%", "\n".join(fixture.messages))

    def test_preview_stays_silent_even_when_notify_is_forced(self):
        with scoring_fixture(execute=False, notify=True) as fixture:
            evaluate.score(fixture.args)
            fixture.progress_factory.assert_not_called()
            fixture.judge.evaluate.assert_not_called()
            self.assertFalse(fixture.output.exists())

    def test_fully_cached_batch_completes_without_an_api_client(self):
        with scoring_fixture(cached=8) as fixture:
            evaluate.score(fixture.args)
            fixture.create_client.assert_not_called()
            fixture.judge.evaluate.assert_not_called()
            self.assertEqual(len(fixture.messages), 2)
            self.assertIn("8/8", fixture.messages[0])
            self.assertIn("Completed", fixture.messages[-1])

    def test_api_client_setup_failure_gets_a_terminal_status(self):
        with scoring_fixture() as fixture:
            fixture.create_client.side_effect = RuntimeError("private credential setup details")
            with self.assertRaises(RuntimeError):
                evaluate.score(fixture.args)
            self.assertEqual(len(fixture.messages), 2)
            self.assertIn("Failed", fixture.messages[-1])
            self.assertNotIn("private credential", "\n".join(fixture.messages))

    def test_judge_failure_sends_terminal_status_without_private_exception_text(self):
        with scoring_fixture() as fixture:
            fixture.judge.evaluate.side_effect = [fixture.result, RuntimeError("private prompt must not be sent")]
            with self.assertRaisesRegex(RuntimeError, "private prompt"):
                evaluate.score(fixture.args)
            self.assertIn("Failed", fixture.messages[-1])
            self.assertIn("2/8", fixture.messages[-1])
            self.assertIn("Errors: 1", fixture.messages[-1])
            self.assertEqual(fixture.persisted[-1], 1)
            self.assertNotIn("private prompt", "\n".join(fixture.messages))
            self.assertNotIn("Completed", fixture.messages[-1])

    def test_write_failure_counts_one_failed_attempt_without_saved_completion(self):
        with scoring_fixture() as fixture:
            with patch.object(evaluate, "append_jsonl", side_effect=OSError("private path out of space")):
                with self.assertRaises(OSError):
                    evaluate.score(fixture.args)
            self.assertIn("Failed", fixture.messages[-1])
            self.assertIn("1/8", fixture.messages[-1])
            self.assertIn("Errors: 1", fixture.messages[-1])
            self.assertEqual(fixture.persisted[-1], 0)
            self.assertNotIn("private path", "\n".join(fixture.messages))

    def test_failed_attempt_is_not_counted_as_cached_on_resume(self):
        with scoring_fixture(cached=2) as fixture:
            fixture.judge.evaluate.side_effect = RuntimeError("judge failed")
            with self.assertRaises(RuntimeError):
                evaluate.score(fixture.args)
            self.assertIn("3/8", fixture.messages[-1])
            self.assertIn("Errors: 1", fixture.messages[-1])
            self.assertEqual(len(evaluate.read_jsonl(fixture.output)), 2)
            fixture.judge.evaluate.side_effect = lambda row: dict(fixture.result)
            fixture.messages.clear()
            evaluate.score(fixture.args)
            self.assertIn("2/8", fixture.messages[0])
            self.assertIn("Errors: 0", fixture.messages[-1])
            self.assertEqual(fixture.progress_calls[-1][1]["completed"], 2)

    def test_transport_failure_is_not_a_scoring_error(self):
        with scoring_fixture() as fixture:
            messages = []
            def broken_sender(message):
                messages.append(message)
                raise OSError("transport failed")
            def notifier(*args, **kwargs):
                return ProgressNotifier(*args, **kwargs, sender=broken_sender)
            with patch.object(evaluate, "ProgressNotifier", side_effect=notifier), self.assertLogs("src.notifications", level="WARNING"):
                evaluate.score(fixture.args)
            self.assertEqual(len(evaluate.read_jsonl(fixture.output)), 8)
            self.assertIn("Completed", messages[-1])
            self.assertIn("Errors: 0", messages[-1])

    def test_invalid_cached_score_does_not_count_as_completed(self):
        with scoring_fixture(cached=1) as fixture:
            existing = evaluate.read_jsonl(fixture.output)
            existing[0]["Q"] = 1
            write_rows(fixture.output, existing)
            with self.assertRaisesRegex(ValueError, "cached score is incomplete"):
                evaluate.score(fixture.args)
            fixture.progress_factory.assert_not_called()
            fixture.judge.evaluate.assert_not_called()

    def test_auto_uses_documented_rough_estimate_and_no_notify_overrides(self):
        for total, notify, mode, expected in ((59, None, "auto", False), (60, None, "auto", True),
                                               (60, False, "on", False)):
            with self.subTest(total=total, notify=notify, mode=mode), \
                    patch.dict(os.environ, {"TELEGRAM_NOTIFICATIONS": mode}), \
                    scoring_fixture(total=total, notify=notify) as fixture:
                evaluate.score(fixture.args)
                self.assertEqual(bool(fixture.messages), expected)

    def test_repeatability_has_no_notification_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"example_uid": f"biology__{index}", "condition": "A", "base_seed": 42,
                     "protocol_hash": "fixture", "output_sha256": "image", "judge_model": "judge",
                     "prompt_sha256": "prompt", "aggregate": "aggregate", "Q": 75} for index in range(20)]
            first, second = root / "first.jsonl", root / "second.jsonl"
            write_rows(first, rows)
            write_rows(second, rows)
            with patch.object(evaluate, "ProgressNotifier") as notifier, \
                    patch.object(evaluate, "load_notification_env") as load, redirect_stdout(io.StringIO()):
                evaluate.repeatability(SimpleNamespace(first=first, second=second, output=root / "report.json"))
            notifier.assert_not_called()
            load.assert_not_called()

    def test_cli_loads_env_only_for_execution_and_exposes_score_notify_options(self):
        for arguments, expected_notify, load_count in ((["--execute", "--notify"], True, 1),
                                                        (["--execute", "--no-notify"], False, 1),
                                                        (["--notify"], True, 0)):
            with self.subTest(arguments=arguments), \
                    patch("sys.argv", ["evaluate", "score", "--generations", "unused", "--output", "unused", *arguments]), \
                    patch.object(evaluate, "score") as score, \
                    patch.object(evaluate, "load_notification_env") as load:
                evaluate.main()
            self.assertEqual(score.call_args.args[0].notify, expected_notify)
            self.assertEqual(load.call_count, load_count)


if __name__ == "__main__":
    unittest.main()
