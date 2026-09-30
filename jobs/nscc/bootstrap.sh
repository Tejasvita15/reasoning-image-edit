#!/usr/bin/env bash
# Run once on an NSCC CPU/transfer host with outbound HTTPS, or via bootstrap.pbs.
set -euo pipefail
# shellcheck source=common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RQ3_DO_PREPARE=0
case ${1:-} in
    '') ;;
    --prepare) RQ3_DO_PREPARE=1; shift ;;
    *) printf 'Usage: bash jobs/nscc/bootstrap.sh [--prepare]\n' >&2; exit 2 ;;
esac
[[ $# -eq 0 ]] || { rq3_error 'Unexpected bootstrap arguments.'; exit 2; }
[[ $(uname -s) == Linux ]] || { rq3_error 'Bootstrap targets NSCC Linux; it must not install this CUDA stack on a Mac.'; exit 1; }
rq3_paths
RQ3_BOOTSTRAP_LOG=$RQ3_STATE_DIR/logs/bootstrap-$(date -u +%Y%m%dT%H%M%SZ)-$RQ3_JOB_TAG.log
exec > >(tee -a "$RQ3_BOOTSTRAP_LOG") 2>&1
printf 'RQ3 bootstrap: workdir=%s env=%s log=%s\n' "$RQ3_WORKDIR" "$RQ3_ENV_PREFIX" "$RQ3_BOOTSTRAP_LOG"

# The kernel releases flock after a crashed job; the portable mkdir fallback
# leaves a visible lock requiring manual inspection after an uncatchable kill.
RQ3_MKDIR_LOCK=''
rq3_bootstrap_exit() {
    local rq3_exit_code=$?
    if [[ -n $RQ3_MKDIR_LOCK ]]; then
        rm -f -- "$RQ3_MKDIR_LOCK/owner"
        rmdir -- "$RQ3_MKDIR_LOCK" || true
    fi
    printf 'RQ3 bootstrap finished with exit code %s at %s\n' "$rq3_exit_code" "$(date -u +%FT%TZ)"
    exit "$rq3_exit_code"
}
trap rq3_bootstrap_exit EXIT
if command -v flock >/dev/null 2>&1; then
    exec 9>"$RQ3_STATE_DIR/bootstrap.lock"
    flock -n 9 || { rq3_error 'Another bootstrap is installing this shared environment; wait for its job to finish.'; exit 1; }
else
    RQ3_LOCK_CANDIDATE=$RQ3_STATE_DIR/bootstrap.lock.d
    mkdir -- "$RQ3_LOCK_CANDIDATE" || { rq3_error "Bootstrap lock exists: $RQ3_LOCK_CANDIDATE; inspect its owner before removing a stale lock."; exit 1; }
    RQ3_MKDIR_LOCK=$RQ3_LOCK_CANDIDATE
    printf 'host=%s pid=%s job=%s\n' "$(hostname)" "$$" "$RQ3_JOB_TAG" > "$RQ3_MKDIR_LOCK/owner"
fi
rq3_load_modules
rq3_paths
rq3_quota_report
rq3_link_sources

if [[ $RQ3_TORCH_VERSION != 2.5.1 || $RQ3_TORCHVISION_VERSION != 0.20.1 ||
      $RQ3_TORCH_INDEX_URL != https://download.pytorch.org/whl/cu121 || $RQ3_EXPECTED_CUDA != 12.1 ]]; then
    [[ ${RQ3_TORCH_STACK_VALIDATED:-0} == 1 ]] || {
        rq3_error 'A changed PyTorch stack needs an explicitly validated configuration (RQ3_TORCH_STACK_VALIDATED=1), a fresh prefix and new development pilots.'
        exit 1
    }
fi

if [[ -f $RQ3_ENV_STAMP ]]; then
    rq3_activate
    printf 'Matching ready environment found; skipping every conda/pip installation.\n'
else
    if [[ ! -x $RQ3_ENV_PREFIX/bin/python ]]; then
        conda create --yes --prefix "$RQ3_ENV_PREFIX" --override-channels --channel conda-forge python=3.10 pip
    elif [[ ! -f $RQ3_ENV_PREFIX/.rq3-bootstrap-in-progress ]]; then
        rq3_error "Unstamped pre-existing environment at $RQ3_ENV_PREFIX. Choose a fresh RQ3_ENV_PREFIX; bootstrap will not modify an unrelated environment."
        exit 1
    fi
    touch "$RQ3_ENV_PREFIX/.rq3-bootstrap-in-progress"
    rq3_activate_env
    python -m pip install --no-cache-dir "torch==$RQ3_TORCH_VERSION" "torchvision==$RQ3_TORCHVISION_VERSION" --index-url "$RQ3_TORCH_INDEX_URL"
    python -m pip install --no-cache-dir -r "$RQ3_WORKDIR/requirements-rq3.txt"
    python -m src.rq3.prepare --config "$RQ3_CONFIG" --stage-sources
    # Paths come from the pinned protocol, not a moving PyPI diffusers release.
    mapfile -t RQ3_SOURCE_PATHS < <(python - <<'PY'
import os
from src.rq3.protocol import load_protocol
config = load_protocol(os.environ['RQ3_CONFIG'])
print(config['_paths']['diffusers'])
print(config['_paths']['regione'])
PY
)
    [[ ${#RQ3_SOURCE_PATHS[@]} -eq 2 ]] || { rq3_error 'Cannot resolve the two pinned source paths.'; exit 1; }
    python -m pip install --no-cache-dir --no-deps --editable "${RQ3_SOURCE_PATHS[0]}" --editable "${RQ3_SOURCE_PATHS[1]}"
    # RegionE's upstream install_requires pins unrelated/older packages. The
    # audited adapter uses the exact project pins; validate that contract below
    # instead of letting upstream metadata replace the frozen environment.
    python -m src.rq3.prepare --config "$RQ3_CONFIG" --environment-stamp write --stamp "$RQ3_ENV_STAMP"
    rm -f -- "$RQ3_ENV_PREFIX/.rq3-bootstrap-in-progress"
fi

# Keep portable records alongside the readiness stamp. The lock serializes
# updates, and these commands do not modify the installed environment.
python -m pip freeze > "$RQ3_STATE_DIR/logs/pip-freeze.txt.tmp"
mv -- "$RQ3_STATE_DIR/logs/pip-freeze.txt.tmp" "$RQ3_STATE_DIR/logs/pip-freeze.txt"
conda list --explicit > "$RQ3_STATE_DIR/logs/conda-explicit.txt.tmp"
mv -- "$RQ3_STATE_DIR/logs/conda-explicit.txt.tmp" "$RQ3_STATE_DIR/logs/conda-explicit.txt"

if [[ $RQ3_DO_PREPARE -eq 1 ]]; then
    # Preparation is online; compute jobs switch to HF_HUB_OFFLINE/HF_DATASETS_OFFLINE.
    export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
    python -m src.rq3.prepare --config "$RQ3_CONFIG" --output-dir "$RQ3_PREPARED_DATA"
fi
