"""Run the actual bootstrap flow with isolated, entirely mocked installers."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bootstrap_tree(tmp_path):
    jobs = tmp_path / "jobs/nscc"
    jobs.mkdir(parents=True)
    shutil.copy(ROOT / "jobs/nscc/bootstrap.sh", jobs / "bootstrap.sh")
    (jobs / "common.sh").write_text('''
export RQ3_WORKDIR=$RQ3_TEST_ROOT
export RQ3_STATE_DIR=$RQ3_TEST_ROOT/state
export RQ3_ENV_PREFIX=$RQ3_STATE_DIR/envs/ai6132
export RQ3_ENV_STAMP=$RQ3_ENV_PREFIX/.rq3-ready.json
export RQ3_CONFIG=$RQ3_TEST_ROOT/configs/rq3.yaml
export RQ3_PREPARED_DATA=$RQ3_STATE_DIR/data/rq3
export RQ3_JOB_TAG=mock-bootstrap
export RQ3_TORCH_VERSION=2.5.1
export RQ3_TORCHVISION_VERSION=0.20.1
export RQ3_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
export RQ3_EXPECTED_CUDA=12.1
rq3_paths() { mkdir -p "$RQ3_STATE_DIR/logs"; }
rq3_load_modules() { :; }
rq3_quota_report() { :; }
rq3_link_sources() { :; }
rq3_error() { printf '%s\\n' "$*" >&2; }
rq3_activate_env() { export PATH="$RQ3_ENV_PREFIX/bin:$PATH"; }
rq3_activate() { rq3_activate_env; }
# NSCC runs modern Bash. Keep this control-flow test runnable on macOS Bash 3.
if ! type mapfile >/dev/null 2>&1; then
    mapfile() {
        RQ3_SOURCE_PATHS=()
        local source_path index=0
        while IFS= read -r source_path; do
            RQ3_SOURCE_PATHS[$index]=$source_path
            index=$((index + 1))
        done
    }
fi
''')
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "uname").write_text("#!/bin/bash\nprintf 'Linux\\n'\n")
    (fakebin / "conda").write_text('''#!/bin/bash
printf 'conda' >> "$RQ3_TEST_EVENTS"
printf '\\t%s' "$@" >> "$RQ3_TEST_EVENTS"
printf '\\n' >> "$RQ3_TEST_EVENTS"
case $1 in
    create)
        mkdir -p "$RQ3_ENV_PREFIX/bin"
        cp "$RQ3_TEST_ROOT/python-template" "$RQ3_ENV_PREFIX/bin/python"
        for argument in "$@"; do
            if [[ $argument == git ]]; then
                cp "$RQ3_TEST_ROOT/git-template" "$RQ3_ENV_PREFIX/bin/git"
            fi
        done
        ;;
    install)
        if [[ ${RQ3_TEST_REPAIR_EXIT:-0} != 0 ]]; then
            exit "$RQ3_TEST_REPAIR_EXIT"
        fi
        cp "$RQ3_TEST_ROOT/git-template" "$RQ3_ENV_PREFIX/bin/git"
        ;;
    list) printf '# mocked explicit package list\\n' ;;
    *) exit 90 ;;
esac
''')
    (tmp_path / "python-template").write_text('''#!/bin/bash
printf 'python' >> "$RQ3_TEST_EVENTS"
printf '\\t%s' "$@" >> "$RQ3_TEST_EVENTS"
printf '\\n' >> "$RQ3_TEST_EVENTS"
if [[ $1 == - ]]; then
    cat >/dev/null
    printf '%s\\n' "$RQ3_TEST_ROOT/external/diffusers" "$RQ3_TEST_ROOT/external/RegionE"
elif [[ " $* " == *" --stage-sources "* ]]; then
    # A host Git must never hide a missing executable in the activated prefix.
    [[ $(command -v git) == "$RQ3_ENV_PREFIX/bin/git" ]] || exit 91
elif [[ " $* " == *" --environment-stamp write "* ]]; then
    printf '{}\\n' > "$RQ3_ENV_STAMP"
elif [[ " $* " == *" pip freeze "* ]]; then
    printf '# mocked pip freeze\\n'
fi
''')
    (tmp_path / "git-template").write_text('''#!/bin/bash
printf 'git' >> "$RQ3_TEST_EVENTS"
printf '\\t%s' "$@" >> "$RQ3_TEST_EVENTS"
printf '\\n' >> "$RQ3_TEST_EVENTS"
printf 'git version mock\\n'
''')
    for executable in [*fakebin.iterdir(), tmp_path / "python-template", tmp_path / "git-template"]:
        executable.chmod(0o755)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("RQ3_", "PBS_"))}
    environment.update(PATH=f"{fakebin}:{environment['PATH']}", RQ3_TEST_ROOT=str(tmp_path),
                       RQ3_TEST_EVENTS=str(tmp_path / "events.tsv"))
    return tmp_path, environment


def seed_prefix(tree, *, git=False, ready=False):
    root, _ = tree
    prefix = root / "state/envs/ai6132"
    (prefix / "bin").mkdir(parents=True)
    shutil.copy(root / "python-template", prefix / "bin/python")
    if git:
        shutil.copy(root / "git-template", prefix / "bin/git")
    marker = ".rq3-ready.json" if ready else ".rq3-bootstrap-in-progress"
    (prefix / marker).write_text("{}\n")
    return prefix


def invoke(tree, **overrides):
    root, environment = tree
    result = subprocess.run(["bash", str(root / "jobs/nscc/bootstrap.sh"), "--prepare"],
                            env={**environment, **overrides}, capture_output=True, text=True, timeout=15)
    log = root / "events.tsv"
    events = [line.split("\t") for line in log.read_text().splitlines()] if log.exists() else []
    return result, events


def installations(events):
    return [event for event in events if event[:2] in (["conda", "create"], ["conda", "install"])]


def test_fresh_environment_provisions_git_before_python_staging(bootstrap_tree):
    result, events = invoke(bootstrap_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    installs = installations(events)
    assert len(installs) == 1 and installs[0][:2] == ["conda", "create"]
    assert {"git", "python=3.10", "pip"}.issubset(installs[0])
    assert events.index(["git", "--version"]) < next(i for i, event in enumerate(events) if event[0] == "python")
    assert any("--stage-sources" in event for event in events)


def test_interrupted_environment_repairs_git_before_pip_and_source_staging(bootstrap_tree):
    prefix = seed_prefix(bootstrap_tree)
    result, events = invoke(bootstrap_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    installs = installations(events)
    assert len(installs) == 1 and installs[0][:2] == ["conda", "install"]
    repair = installs[0]
    assert repair[repair.index("--prefix") + 1] == str(prefix)
    assert "--freeze-installed" in repair and "--override-channels" in repair
    assert repair[repair.index("--channel") + 1] == "conda-forge" and repair[-1] == "git"
    checked = events.index(["git", "--version"])
    assert events.index(repair) < checked < next(i for i, event in enumerate(events) if event[0] == "python")
    assert any("--stage-sources" in event for event in events)
    assert not (prefix / ".rq3-bootstrap-in-progress").exists()


def test_interrupted_environment_with_git_skips_extra_conda_install(bootstrap_tree):
    seed_prefix(bootstrap_tree, git=True)
    result, events = invoke(bootstrap_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert installations(events) == []
    assert ["git", "--version"] in events
    assert any("--stage-sources" in event for event in events)


def test_git_repair_failure_stops_before_pip_and_source_staging(bootstrap_tree):
    prefix = seed_prefix(bootstrap_tree)
    result, events = invoke(bootstrap_tree, RQ3_TEST_REPAIR_EXIT="37")
    assert result.returncode == 37, result.stdout + result.stderr
    assert len(installations(events)) == 1
    assert not any(event[0] == "python" for event in events)
    assert (prefix / ".rq3-bootstrap-in-progress").exists()
    assert not (prefix / ".rq3-ready.json").exists()


def test_matching_ready_environment_keeps_installation_skip_path(bootstrap_tree):
    seed_prefix(bootstrap_tree, git=True, ready=True)
    result, events = invoke(bootstrap_tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert installations(events) == []
    assert not any("--stage-sources" in event or "install" in event for event in events)
    assert any("--output-dir" in event for event in events)
