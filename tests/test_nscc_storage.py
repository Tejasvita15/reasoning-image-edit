"""Exercise the real storage shell guards without NSCC, Conda or network."""

import os
from pathlib import Path
import subprocess

import pytest


COMMON = Path(__file__).resolve().parents[1] / "jobs/nscc/common.sh"
CACHE_VARIABLES = (
    "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
    "TORCH_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "CONDA_ENVS_PATH", "XDG_CACHE_HOME", "MPLCONFIGDIR",
    "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TMPDIR", "TMP", "TEMP",
)


@pytest.fixture
def storage_tree(tmp_path):
    fake_home = tmp_path / "home"
    repository = fake_home / "repo"
    (repository / "src/rq3").mkdir(parents=True)
    (repository / "src/rq3/run.py").write_text("# repository sentinel\n")
    scratch = tmp_path / "scratch_fs/user"
    scratch.mkdir(parents=True)
    (fake_home / "scratch").symlink_to(scratch, target_is_directory=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("RQ3_", "PBS_")) and key not in CACHE_VARIABLES}
    # HOME is isolated inside this child-process test, never changed in the
    # runner's process or a user's shell/environment configuration.
    environment.update(HOME=str(fake_home), RQ3_WORKDIR=str(repository), RQ3_TEST_COMMON=str(COMMON))
    return {"home": fake_home, "repo": repository, "scratch": scratch, "env": environment}


def invoke(tree, script="rq3_paths", **overrides):
    return subprocess.run(["bash", "-c", 'set -euo pipefail\nsource "$RQ3_TEST_COMMON"\n' + script],
                          env={**tree["env"], **overrides}, text=True, capture_output=True, timeout=15)


def assert_no_bulk_directories(tree):
    assert not (tree["scratch"] / "ai6132").exists()
    assert not (tree["repo"] / ".nscc").exists()
    assert not (tree["repo"] / "outputs").exists()
    assert not (tree["repo"] / "logs").exists()


def test_defaults_resolve_site_scratch_link_and_keep_bulk_state_outside_home(storage_tree):
    names = ["RQ3_SCRATCH_ROOT", "RQ3_STORAGE_ROOT", "RQ3_STATE_DIR", "RQ3_OUTPUT_ROOT", "RQ3_LOG_DIR",
             "RQ3_EXTERNAL_DIR", "RQ3_ENV_PREFIX", "RQ3_PREPARED_DATA", "HF_HOME", "HF_HUB_CACHE",
             "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME", "CONDA_PKGS_DIRS",
             "CONDA_ENVS_PATH", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TMPDIR"]
    script = 'rq3_paths\nprintf "%s\\n" ' + " ".join(f'"${name}"' for name in names)
    result = invoke(storage_tree, script)
    assert result.returncode == 0, result.stderr
    paths = [Path(line) for line in result.stdout.splitlines()]
    assert paths[0] == storage_tree["scratch"].resolve()
    assert len(paths) == len(names)
    assert all(path.is_relative_to(storage_tree["scratch"].resolve()) for path in paths)
    assert not (storage_tree["repo"] / ".nscc").exists()
    assert not (storage_tree["home"] / ".cache").exists()
    assert (storage_tree["scratch"] / "ai6132/outputs/rq3/nscc").is_dir()
    assert (storage_tree["scratch"] / "ai6132/logs/nscc").is_dir()


def test_missing_scratch_fails_without_creating_home_scratch(storage_tree):
    (storage_tree["home"] / "scratch").unlink()
    result = invoke(storage_tree)
    assert result.returncode != 0 and "Scratch is unavailable" in result.stderr
    assert not (storage_tree["home"] / "scratch").exists()
    assert_no_bulk_directories(storage_tree)


def test_plain_home_scratch_directory_is_rejected(storage_tree):
    link = storage_tree["home"] / "scratch"
    link.unlink()
    link.mkdir()
    result = invoke(storage_tree)
    assert result.returncode != 0 and "inside HOME" in result.stderr
    assert list(link.iterdir()) == []
    assert_no_bulk_directories(storage_tree)


def test_broken_scratch_link_is_rejected_without_recreating_target(storage_tree):
    link = storage_tree["home"] / "scratch"
    link.unlink()
    target = storage_tree["scratch"].parent / "missing"
    link.symlink_to(target, target_is_directory=True)
    result = invoke(storage_tree)
    assert result.returncode != 0 and "Scratch is unavailable" in result.stderr
    assert link.is_symlink() and not target.exists()
    assert_no_bulk_directories(storage_tree)


def test_explicit_existing_scratch_root_works_without_home_link(storage_tree):
    (storage_tree["home"] / "scratch").unlink()
    result = invoke(storage_tree, RQ3_SCRATCH_ROOT=str(storage_tree["scratch"]))
    assert result.returncode == 0, result.stderr
    assert not (storage_tree["home"] / "scratch").exists()


@pytest.mark.parametrize("variable", ["HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
                                     "CONDA_PKGS_DIRS", "CONDA_ENVS_PATH", "XDG_CACHE_HOME",
                                     "MPLCONFIGDIR", "RQ3_ENV_PREFIX", "RQ3_OUTPUT_ROOT", "RQ3_LOG_DIR"])
def test_inherited_home_bulk_path_is_rejected_before_mkdir(storage_tree, variable):
    bad_path = storage_tree["home"] / "unwanted" / variable.lower()
    result = invoke(storage_tree, **{variable: str(bad_path)})
    assert result.returncode != 0 and "outside RQ3_SCRATCH_ROOT" in result.stderr
    assert not bad_path.parent.exists()
    assert_no_bulk_directories(storage_tree)


def test_existing_ancestor_symlink_cannot_escape_scratch(storage_tree):
    home_cache = storage_tree["home"] / ".cache"
    home_cache.mkdir()
    (storage_tree["scratch"] / "escape").symlink_to(home_cache, target_is_directory=True)
    result = invoke(storage_tree, HF_HUB_CACHE=str(storage_tree["scratch"] / "escape/huggingface/hub"))
    assert result.returncode != 0 and "outside RQ3_SCRATCH_ROOT" in result.stderr
    assert list(home_cache.iterdir()) == []
    assert_no_bulk_directories(storage_tree)


def test_home_tmpdir_is_rejected_before_bulk_directories_are_created(storage_tree):
    temporary = storage_tree["home"] / "tmp"
    result = invoke(storage_tree, TMPDIR=str(temporary))
    assert result.returncode != 0 and "TMPDIR resolves inside HOME" in result.stderr
    assert not temporary.exists()
    assert_no_bulk_directories(storage_tree)


def test_scheduler_local_tmpdir_is_allowed(storage_tree, tmp_path):
    local = tmp_path / "node_local"
    local.mkdir()
    result = invoke(storage_tree, 'rq3_paths\nprintf "%s\\n" "$TMPDIR"', PBS_JOBFS=str(local), PBS_JOBID="123.server")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == local / "rq3-123.server"
    assert Path(result.stdout.strip()).is_dir()


def test_legacy_hf_cache_aliases_are_normalized_and_pip_cache_is_disabled(storage_tree):
    script = 'rq3_paths\nprintf "%s\\n" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$PIP_NO_CACHE_DIR"'
    result = invoke(storage_tree, script, TRANSFORMERS_CACHE=str(storage_tree["home"] / ".cache/old1"),
                    HUGGINGFACE_HUB_CACHE=str(storage_tree["home"] / ".cache/old2"), PIP_NO_CACHE_DIR="0")
    assert result.returncode == 0, result.stderr
    hub, transformers, legacy_hub, no_cache = result.stdout.splitlines()
    assert hub == transformers == legacy_hub
    assert Path(hub).is_relative_to(storage_tree["scratch"])
    assert no_cache == "1"
    assert not (storage_tree["home"] / ".cache").exists()


@pytest.mark.parametrize("suffix", ["/../escape", "/./child", ":/other/cache", "\nother"])
def test_ambiguous_override_paths_are_rejected_before_mkdir(storage_tree, suffix):
    value = str(storage_tree["scratch"] / "cache") + suffix
    result = invoke(storage_tree, CONDA_PKGS_DIRS=value)
    assert result.returncode != 0
    assert_no_bulk_directories(storage_tree)


def test_external_link_is_created_only_for_fresh_layout_and_can_be_reused(storage_tree):
    result = invoke(storage_tree, "rq3_paths\nrq3_link_sources\nrq3_link_sources")
    assert result.returncode == 0, result.stderr
    link = storage_tree["repo"] / "external"
    assert link.is_symlink()
    assert link.resolve() == storage_tree["scratch"] / "ai6132/external"


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_home_external_directory_is_never_replaced(storage_tree, nonempty):
    external = storage_tree["repo"] / "external"
    external.mkdir()
    if nonempty:
        (external / "precious.py").write_text("preserve local source\n")
    result = invoke(storage_tree, "rq3_paths\nrq3_link_sources")
    assert result.returncode != 0 and "Existing external/ is outside scratch" in result.stderr
    assert external.is_dir() and not external.is_symlink()
    if nonempty:
        assert (external / "precious.py").read_text() == "preserve local source\n"


def test_external_link_back_into_home_is_rejected(storage_tree):
    elsewhere = storage_tree["home"] / "old_external"
    elsewhere.mkdir()
    (storage_tree["repo"] / "external").symlink_to(elsewhere, target_is_directory=True)
    result = invoke(storage_tree, "rq3_paths\nrq3_link_sources")
    assert result.returncode != 0 and "outside scratch" in result.stderr
    assert (storage_tree["repo"] / "external").resolve() == elsewhere
