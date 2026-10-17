"""Exercise PBS wrapper dispatch without a scheduler, CUDA or network."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from src.rq3.budget import estimate
from src.rq3.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_job_tree(tmp_path):
    jobs = tmp_path / "jobs/nscc"
    jobs.mkdir(parents=True)
    for name in ("gpu.pbs", "finish.pbs", "bootstrap.pbs"):
        shutil.copy(ROOT / "jobs/nscc" / name, jobs / name)
    shutil.copy(ROOT / "jobs/notifications.sh", jobs.parent / "notifications.sh")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/notifications.py").write_text("# Replaced by fake Python; no network calls.\n")
    (jobs / "bootstrap.sh").write_text('exit "${RQ3_TEST_CHILD_EXIT:-0}"\n')
    # Only the cluster environment and Python process are replaced. The actual
    # PBS wrapper performs argument construction, paths and shell traps.
    (jobs / "common.sh").write_text('''
export RQ3_WORKDIR=$PBS_O_WORKDIR
export RQ3_STORAGE_ROOT=$RQ3_WORKDIR/scratch/ai6132
export RQ3_STATE_DIR=$RQ3_STORAGE_ROOT/state
export RQ3_OUTPUT_ROOT=$RQ3_STORAGE_ROOT/outputs/rq3/nscc
export RQ3_LOG_DIR=$RQ3_STORAGE_ROOT/logs/nscc
export RQ3_CONFIG=$RQ3_WORKDIR/configs/rq3.yaml
export RQ3_PREPARED_DATA=$RQ3_STATE_DIR/data/rq3
export TMPDIR=$RQ3_STATE_DIR/tmp
rq3_paths() { mkdir -p "$RQ3_STATE_DIR" "$RQ3_OUTPUT_ROOT" "$RQ3_LOG_DIR" "$TMPDIR"; }
rq3_assert_scratch_path() { :; }
rq3_quota_report() { :; }
rq3_load_modules() { :; }
rq3_activate() { :; }
rq3_check_driver() { return "${RQ3_TEST_DRIVER_EXIT:-0}"; }
rq3_check_prepared() { :; }
module() { :; }
''')
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "python").write_text('''#!/bin/bash
if [[ $1 == -c ]]; then exit 0; fi
if [[ $1 == */src/notifications.py ]]; then
  printf '%s\\t' "$@" >> "$RQ3_TEST_NOTIFY_ARGS"
  printf '\\n' >> "$RQ3_TEST_NOTIFY_ARGS"
  exit "${RQ3_TEST_NOTIFY_EXIT:-0}"
fi
if [[ $1 == - ]]; then cat >/dev/null; exit 0; fi
printf '%s\\n' "$@" >> "$RQ3_TEST_ARGS"
if [[ ${RQ3_TEST_CHILD_MARKS_TERMINAL:-0} == 1 ]]; then
  printf '{}\\n' > "$TELEGRAM_TERMINAL_MARKER"
fi
exit "${RQ3_TEST_CHILD_EXIT:-0}"
''')
    (fakebin / "python3").symlink_to(fakebin / "python")
    (fakebin / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    for command in fakebin.iterdir():
        command.chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}", "PBS_O_WORKDIR": str(tmp_path),
           "PBS_JOBID": "123[2].pbs101", "RQ3_TEST_ARGS": str(tmp_path / "args.txt"),
           "RQ3_TEST_NOTIFY_ARGS": str(tmp_path / "notifications.txt")}
    for name in list(env):
        if name.startswith("RQ3_") and name not in ("RQ3_TEST_ARGS", "RQ3_TEST_NOTIFY_ARGS"):
            env.pop(name)
    return tmp_path, env


def invoke(fake_job_tree, script="gpu.pbs", **variables):
    root, env = fake_job_tree
    result = subprocess.run(["bash", str(root / "jobs/nscc" / script)],
                            env={**env, **variables}, text=True, capture_output=True, timeout=15)
    arguments = (root / "args.txt").read_text().splitlines() if (root / "args.txt").exists() else []
    return result, arguments


def test_gpu_array_keeps_paired_shard_and_resume_arguments(fake_job_tree):
    result, args = invoke(fake_job_tree, PBS_ARRAY_INDEX="2", RQ3_SEEDS="42:1234:7", RQ3_TIMING_REPEATS="3")
    assert result.returncode == 0, result.stderr + result.stdout
    assert args[:3] == ["-m", "src.rq3.run", "--config"]
    assert "--resume" in args and "--prepared-data" in args
    assert args[args.index("--shard-index") + 1] == "2"
    assert args[args.index("--num-shards") + 1] == "4"
    assert args[args.index("--seeds") + 1:args.index("--seeds") + 4] == ["42", "1234", "7"]
    assert args[args.index("--output") + 1].endswith("/main/shard_2")


def test_pilot_is_not_sharded_and_child_failure_is_propagated(fake_job_tree):
    result, args = invoke(fake_job_tree, RQ3_MODE="smoke", RQ3_TEST_CHILD_EXIT="17")
    assert result.returncode == 17
    assert "--shard-index" not in args
    assert args[args.index("--limit") + 1] == "3"
    assert args[args.index("--phase") + 1] == "pilot"
    exit_file = fake_job_tree[0] / "scratch/ai6132/outputs/rq3/nscc/smoke/attempts/123[2].pbs101.exitcode"
    assert exit_file.read_text().strip() == "17"
    assert not list((fake_job_tree[0] / "scratch/ai6132/state/tmp").glob("rq3-compile.*"))


def test_nonarray_main_requires_explicit_shard(fake_job_tree):
    result, args = invoke(fake_job_tree)
    assert result.returncode == 2
    assert not args
    result, args = invoke(fake_job_tree, RQ3_SHARD_INDEX="2")
    assert result.returncode == 0
    assert args[args.index("--shard-index") + 1] == "2"


def test_pilot_array_rejected(fake_job_tree):
    result, args = invoke(fake_job_tree, RQ3_MODE="pilot_first", PBS_ARRAY_INDEX="0")
    assert result.returncode == 2
    assert not args


def test_finish_merge_constructs_all_shards_and_repeat_count(fake_job_tree):
    result, args = invoke(fake_job_tree, script="finish.pbs", RQ3_ACTION="merge", RQ3_TIMING_REPEATS="3")
    assert result.returncode == 0, result.stderr + result.stdout
    assert args[1] == "src.rq3.merge"
    i = args.index("--shards")
    assert [Path(v).name for v in args[i + 1:i + 5]] == [f"shard_{k}" for k in range(4)]
    assert args[args.index("--timing-repeats") + 1] == "3"


def notification_events(root):
    path = root / "notifications.txt"
    return [line.rstrip("\t").split("\t") for line in path.read_text().splitlines()] if path.exists() else []


def test_gpu_startup_failure_notified_and_notification_failure_preserves_exit(fake_job_tree):
    result, args = invoke(fake_job_tree, RQ3_MODE="smoke", RQ3_TEST_DRIVER_EXIT="23", RQ3_TEST_NOTIFY_EXIT="99")
    assert result.returncode == 23
    assert not args
    events = notification_events(fake_job_tree[0])
    assert len(events) == 1
    assert events[0][events[0].index("--status") + 1] == "failed"
    assert events[0][events[0].index("--exit-code") + 1] == "23"
    assert "job status is unchanged" in result.stdout + result.stderr


def test_child_terminal_marker_suppresses_duplicate_pbs_alert(fake_job_tree):
    result, _ = invoke(fake_job_tree, RQ3_MODE="smoke", RQ3_TEST_CHILD_MARKS_TERMINAL="1")
    assert result.returncode == 0, result.stderr
    assert notification_events(fake_job_tree[0]) == []


@pytest.mark.parametrize("script", ["bootstrap.pbs", "finish.pbs"])
@pytest.mark.parametrize("child_exit,expected", [("0", "completed"), ("19", "failed")])
def test_cpu_job_lifecycle_and_exit_status(fake_job_tree, script, child_exit, expected):
    result, _ = invoke(fake_job_tree, script=script, RQ3_TEST_CHILD_EXIT=child_exit)
    assert result.returncode == int(child_exit), result.stderr
    events = notification_events(fake_job_tree[0])
    assert [event[event.index("--status") + 1] for event in events] == ["started", expected]
    # Credentials are never command-line arguments, even if exported by users.
    assert "TELEGRAM_BOT_TOKEN" not in str(events)
    assert "TELEGRAM_BOT_CHAT_ID" not in str(events)


def test_budget_counts_complete_work_and_requires_pilot_protocol():
    config = load_protocol()
    rows = [{"status": "ok", "phase": "pilot", "example_uid": config["_splits"]["k_dev"][0],
             "protocol_hash": config["protocol_hash"], "condition": condition,
             "timings": {"charged_total_seconds": 60}} for condition in "ABCDEF"]
    report = estimate(config, rows, num_shards=4, timing_repeats=3)
    assert sum(shard["quality_generations"] for shard in report["shards"]) == 200
    assert sum(shard["timing_generations"] for shard in report["shards"]) == 600
    assert all(shard["examples"] == 10 for shard in report["shards"])
    assert report["array_walltime"] == "06:00:00"
    with pytest.raises(ValueError, match="six conditions"):
        estimate(config, rows[:-1])
    rows[0]["phase"] = "all"
    with pytest.raises(ValueError, match="development pilot"):
        estimate(config, rows)


@pytest.mark.parametrize("path", sorted((ROOT / "jobs/nscc").glob("*")) + [ROOT / "jobs/notifications.sh", ROOT / "jobs/runrq3.sh"])
def test_shell_syntax(path):
    if path.suffix in (".sh", ".pbs"):
        subprocess.run(["bash", "-n", str(path)], check=True)
