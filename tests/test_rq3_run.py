from io import BytesIO
import errno
import json
import os
from pathlib import Path

from PIL import Image, PngImagePlugin
import pytest

from src.rq3.run import (atomic_save_image, execute, main, make_shard_plan,
                         read_completed, seed_for, selected_run_keys)


class FakeBackend:
    def __init__(self):
        self.thoughts = []
        self.generations = []

    def think(self, image, instruction, seed):
        self.thoughts.append(seed)
        return {"rewrite": "frozen rewrite", "raw_text": "frozen rewrite", "token_count": 2,
                "seconds": 2.0, "offload_seconds": 0.25, "peak_vram_bytes": 200}

    def generate(self, image, prompt, seed, condition):
        self.generations.append((prompt, seed, condition))
        return {"image": image.copy(), "timings": {"measured_replay_seconds": 1.0, "diffusion_seconds": 0.8},
                "peak_vram_bytes": 100, "regione": {}, "initial_noise_sha256": str(seed)}


def fixture_data():
    raw = BytesIO()
    Image.new("RGB", (8, 8), "red").save(raw, format="PNG")
    examples = {"mathematics__1": {"instruction": "correct the equation", "knowledge_type": "conceptual",
                                  "category": "mathematics", "image": {"bytes": raw.getvalue()}}}
    config = {"protocol_hash": "frozen-test", "model": {"model_id": "fake"}}
    jobs = [{"example_uid": "mathematics__1", "base_seed": 42, "condition": c, "phase": "pilot"}
            for c in "ABCDEF"]
    return config, jobs, examples


def test_all_arms_replay_one_thought_and_identical_noise(tmp_path):
    config, jobs, examples = fixture_data()
    backend = FakeBackend()
    execute(config, jobs, tmp_path, backend, examples)
    assert len(backend.thoughts) == 1
    assert len({g[1] for g in backend.generations}) == 1
    assert [g[0] for g in backend.generations] == ["correct the equation", "frozen rewrite",
                                                "correct the equation", "frozen rewrite",
                                                "frozen rewrite", "frozen rewrite"]
    rows = read_completed(tmp_path / "generations.jsonl", config["protocol_hash"])
    b = rows[("mathematics__1", "B", 42, 0)]
    assert b["timings"]["charged_total_seconds"] == pytest.approx(b["timings"]["measured_replay_seconds"] + 2.25)
    assert b["peak_vram_bytes"] == 200
    execute(config, jobs, tmp_path, backend, examples, resume=True)
    assert len(backend.generations) == 6
    assert len(backend.thoughts) == 1


def test_resume_rejects_mutated_output_and_protocol(tmp_path):
    config, jobs, examples = fixture_data()
    execute(config, jobs[:1], tmp_path, FakeBackend(), examples)
    with pytest.raises(ValueError, match="another protocol"):
        execute({**config, "protocol_hash": "changed"}, jobs, tmp_path, FakeBackend(), examples, resume=True)
    row = json.loads((tmp_path / "generations.jsonl").read_text())
    Path(row["output_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        read_completed(tmp_path / "generations.jsonl", config["protocol_hash"])


def test_failed_generation_is_logged_and_can_resume(tmp_path):
    config, jobs, examples = fixture_data()
    class Failed(FakeBackend):
        def generate(self, *args):
            raise RuntimeError("GPU failure")
    with pytest.raises(RuntimeError, match="GPU failure"):
        execute(config, jobs[:1], tmp_path, Failed(), examples)
    execute(config, jobs[:1], tmp_path, FakeBackend(), examples, resume=True)
    rows = [json.loads(line) for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["error", "ok"]


def test_seed_has_no_condition_or_repeat_dependency():
    assert seed_for(42, "mathematics__1", "diffusion_0") == seed_for(42, "mathematics__1", "diffusion_0")
    assert seed_for(42, "mathematics__1", "thinking") != seed_for(42, "mathematics__1", "diffusion_0")


def test_timing_thought_repeats_are_shared_between_arms(tmp_path):
    config, jobs, examples = fixture_data()
    backend = FakeBackend()
    execute(config, [jobs[1], jobs[3]], tmp_path, backend, examples, timing_repeats=3)
    assert len(backend.thoughts) == 4  # One quality rewrite + three shared timing measurements.
    assert len(backend.generations) == 8
    rows = read_completed(tmp_path / "generations.jsonl", config["protocol_hash"])
    assert sorted(k[-1] for k in rows) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_identical_timing_png_shares_inode_but_atomic_replacement_preserves_original(tmp_path):
    quality, repeat = tmp_path / "quality.png", tmp_path / "repeat.png"
    image = Image.new("RGB", (8, 8), "red")
    atomic_save_image(image, quality)
    original = quality.read_bytes()
    atomic_save_image(image.copy(), repeat, reuse_path=quality)
    assert os.path.samefile(quality, repeat)
    assert quality.stat().st_nlink == 2

    atomic_save_image(Image.new("RGB", (8, 8), "blue"), repeat, reuse_path=quality)
    assert not os.path.samefile(quality, repeat)
    assert quality.read_bytes() == original
    assert repeat.read_bytes() != original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["quality.png", "repeat.png"]


def test_equal_pixels_with_different_png_metadata_do_not_share_storage(tmp_path):
    quality, repeat = tmp_path / "quality.png", tmp_path / "repeat.png"
    image = Image.new("RGB", (8, 8), "red")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("record", "different encoded bytes")
    image.save(quality, pnginfo=metadata)
    atomic_save_image(image, repeat, reuse_path=quality)
    assert not os.path.samefile(quality, repeat)
    assert quality.read_bytes() != repeat.read_bytes()
    with Image.open(quality) as first, Image.open(repeat) as second:
        assert first.tobytes() == second.tobytes()


def test_interrupted_link_install_keeps_previous_files_and_removes_staging(tmp_path, monkeypatch):
    quality, repeat = tmp_path / "quality.png", tmp_path / "repeat.png"
    image = Image.new("RGB", (8, 8), "red")
    atomic_save_image(image, quality)
    atomic_save_image(Image.new("RGB", (8, 8), "blue"), repeat)
    original_quality, original_repeat = quality.read_bytes(), repeat.read_bytes()
    replace = Path.replace

    def interrupted_replace(path, target):
        if path.suffix == ".link":
            raise RuntimeError("interrupted before atomic install")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_save_image(image, repeat, reuse_path=quality)
    assert quality.read_bytes() == original_quality
    assert repeat.read_bytes() == original_repeat
    assert quality.stat().st_nlink == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["quality.png", "repeat.png"]


@pytest.mark.parametrize("error_number", [errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP])
def test_timing_png_link_failure_falls_back_to_complete_regular_file(tmp_path, monkeypatch, error_number):
    quality, repeat = tmp_path / "quality.png", tmp_path / "repeat.png"
    image = Image.new("RGB", (8, 8), "red")
    atomic_save_image(image, quality)

    def unavailable_link(*args, **kwargs):
        raise OSError(error_number, "hardlink unavailable")

    monkeypatch.setattr(os, "link", unavailable_link)
    atomic_save_image(image, repeat, reuse_path=quality)
    assert repeat.read_bytes() == quality.read_bytes()
    assert not os.path.samefile(quality, repeat)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["quality.png", "repeat.png"]


def test_timing_storage_sharing_preserves_generation_records_and_resume(tmp_path):
    config, jobs, examples = fixture_data()
    backend = FakeBackend()
    execute(config, jobs[:1], tmp_path, backend, examples, timing_repeats=3)
    rows = read_completed(tmp_path / "generations.jsonl", config["protocol_hash"])
    assert len(rows) == len(backend.generations) == 4
    paths = [Path(row["output_path"]) for row in rows.values()]
    assert len(set(paths)) == 4
    assert all(os.path.samefile(paths[0], path) for path in paths[1:])
    assert len({row["output_sha256"] for row in rows.values()}) == 1
    execute(config, jobs[:1], tmp_path, backend, examples, timing_repeats=3, resume=True)
    assert len(backend.generations) == 4


def test_nondeterministic_timing_result_keeps_its_distinct_image(tmp_path):
    config, jobs, examples = fixture_data()

    class DifferentRepeat(FakeBackend):
        def generate(self, *args):
            result = super().generate(*args)
            if len(self.generations) == 2:
                result["image"] = Image.new("RGB", (8, 8), "blue")
            return result

    backend = DifferentRepeat()
    execute(config, jobs[:1], tmp_path, backend, examples, timing_repeats=3)
    rows = read_completed(tmp_path / "generations.jsonl", config["protocol_hash"])
    quality, changed, identical = [rows[("mathematics__1", "A", 42, repeat)] for repeat in (0, 1, 2)]
    assert len(backend.generations) == 4
    assert changed["output_sha256"] != quality["output_sha256"]
    assert not os.path.samefile(changed["output_path"], quality["output_path"])
    assert os.path.samefile(identical["output_path"], quality["output_path"])
    with Image.open(changed["output_path"]) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_notification_milestones_follow_persisted_rows_and_terminal_follows_marker(tmp_path):
    from src.notifications import ProgressNotifier

    config, jobs, examples = fixture_data()
    jobs = jobs[:2]
    plan = make_shard_plan(config, jobs, 0, 2, timing_repeats=3)
    messages = []

    def capture(message):
        log = tmp_path / "generations.jsonl"
        count = len(log.read_text().splitlines()) if log.exists() else 0
        messages.append((message, count, (tmp_path / "shard_complete.json").exists()))

    with ProgressNotifier("test", 8, enabled=True, sender=capture) as progress:
        execute(config, jobs, tmp_path, FakeBackend(), examples, timing_repeats=3,
                shard_plan=plan, notify=progress)
    milestones = [(message, count) for message, count, _ in messages if "Milestone:" in message]
    assert len(milestones) == 7
    for expected, (message, count) in enumerate(milestones, 1):
        assert count == expected
        assert f"Progress: {expected}/8" in message
    assert "Started" in messages[0][0]
    assert messages[0][1:] == (0, False)
    assert "Completed" in messages[-1][0]
    assert messages[-1][1:] == (8, True)


@pytest.mark.parametrize("error_type,terminal", [(RuntimeError, "Failed"), (KeyboardInterrupt, "Interrupted")])
def test_notification_reports_generation_failure_and_reraises(tmp_path, error_type, terminal):
    from src.notifications import ProgressNotifier

    config, jobs, examples = fixture_data()
    messages = []

    class Failed(FakeBackend):
        def generate(self, *args):
            raise error_type("inference stopped")

    with pytest.raises(error_type):
        with ProgressNotifier("test", 1, enabled=True, sender=messages.append) as progress:
            execute(config, jobs[:1], tmp_path, Failed(), examples, notify=progress)
    assert sum(terminal in message for message in messages) == 1
    assert terminal in messages[-1]
    if error_type is RuntimeError:
        assert json.loads((tmp_path / "generations.jsonl").read_text())["status"] == "error"
        assert "Errors: 1" in messages[-1]


def _mock_cli_generation(monkeypatch, messages, backend_class=FakeBackend):
    """The real notifier always receives a local sender, never credentials."""
    import src.notifications as notifications
    import src.rq3.backend as backend_module
    import src.rq3.protocol as protocol
    import src.rq3.run as run

    real_notifier = notifications.ProgressNotifier
    monkeypatch.setattr(notifications, "ProgressNotifier",
                        lambda *args, **kwargs: real_notifier(*args, **kwargs, sender=messages.append))
    monkeypatch.setattr(notifications, "load_notification_env", lambda: None)
    monkeypatch.setattr(protocol, "validate_for_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_module, "Step1XBackend", lambda config: backend_class())
    monkeypatch.delenv("RQ3_PREPARED_DATA", raising=False)
    example = next(iter(fixture_data()[2].values()))
    monkeypatch.setattr(run, "load_examples", lambda config, uids: {uid: example for uid in uids})


def test_cli_resume_notifications_count_only_selected_conditions_and_timing(tmp_path, monkeypatch):
    messages = []
    _mock_cli_generation(monkeypatch, messages)
    base_args = ["run", "--phase", "pilot", "--limit", "1", "--output", str(tmp_path)]
    monkeypatch.setattr("sys.argv", base_args + ["--no-notify"])
    main()  # Six successful conditions are cached, but only B/D are selected below.
    assert messages == []
    monkeypatch.setattr("sys.argv", base_args + ["--resume", "--conditions", "B", "D",
                                                 "--timing-repeats", "3", "--notify"])
    main()
    assert "Progress: 2/8" in messages[0]
    assert "Resumed" in messages[0]
    assert all("Milestone: 12.5%" not in message and "Milestone: 25%" not in message
               for message in messages)
    assert "Milestone: 37.5%" in messages[1]
    assert "Progress: 8/8" in messages[-1]
    assert "Completed" in messages[-1]
    rows = [json.loads(line) for line in (tmp_path / "generations.jsonl").read_text().splitlines()]
    selected = [row for row in rows if row["condition"] in "BD"]
    assert len(rows) == 12
    assert len(selected_run_keys(selected, 3)) == len(selected) == 8


@pytest.mark.parametrize("stage", ["loading", "warmup"])
def test_cli_notifications_cover_model_loading_and_warmup_failures(tmp_path, monkeypatch, stage):
    messages = []

    class FailedStartup(FakeBackend):
        def __init__(self):
            if stage == "loading":
                raise RuntimeError("model loading failed")
            super().__init__()

        def think(self, *args):
            raise RuntimeError("warmup failed")

    _mock_cli_generation(monkeypatch, messages, FailedStartup)
    monkeypatch.setattr("sys.argv", ["run", "--phase", "pilot", "--limit", "1", "--conditions", "A",
                                     "--output", str(tmp_path), "--notify"])
    with pytest.raises(RuntimeError, match="failed"):
        main()
    assert len(messages) == 2
    assert "Started" in messages[0]
    assert "Failed" in messages[-1]
    assert not (tmp_path / "generations.jsonl").exists()


def test_dry_run_never_loads_notification_settings_or_backend(monkeypatch, capsys):
    import src.notifications as notifications
    import src.rq3.backend as backend_module

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run touched notifications or model loading")

    monkeypatch.setattr(notifications, "ProgressNotifier", forbidden)
    monkeypatch.setattr(notifications, "load_notification_env", forbidden)
    monkeypatch.setattr(backend_module, "Step1XBackend", forbidden)
    monkeypatch.setattr("sys.argv", ["run", "--dry-run", "--notify"])
    main()
    assert json.loads(capsys.readouterr().out)["generations"] == 200
