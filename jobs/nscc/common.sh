#!/usr/bin/env bash
# Shared NSCC ASPIRE2A paths and read-only job startup checks.
# Source this file; GPU jobs call rq3_activate, never bootstrap/install packages.

RQ3_SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export RQ3_WORKDIR=${RQ3_WORKDIR:-${PBS_O_WORKDIR:-$RQ3_SCRIPT_ROOT}}
# HOME/scratch is a site-provided link, never a directory we create in HOME.
export RQ3_SCRATCH_ROOT=${RQ3_SCRATCH_ROOT:-$HOME/scratch}
if [[ ! -d $RQ3_SCRATCH_ROOT ]]; then
    printf 'RQ3: Scratch is unavailable: %s. Set RQ3_SCRATCH_ROOT to your existing /scratch/users/<org>/<uid> directory.\n' "$RQ3_SCRATCH_ROOT" >&2
    return 1
fi
RQ3_SCRATCH_ROOT=$(cd -- "$RQ3_SCRATCH_ROOT" && pwd -P) || return 1
RQ3_HOME_PHYSICAL=$(cd -- "$HOME" && pwd -P) || return 1
case "$RQ3_SCRATCH_ROOT/" in
    "$RQ3_HOME_PHYSICAL/"*) printf 'RQ3: Scratch resolves inside HOME; refusing bulk writes to the 50GB home allocation.\n' >&2; return 1 ;;
esac
export RQ3_STORAGE_ROOT=${RQ3_STORAGE_ROOT:-$RQ3_SCRATCH_ROOT/ai6132}
export RQ3_STATE_DIR=${RQ3_STATE_DIR:-$RQ3_STORAGE_ROOT/state}
export RQ3_OUTPUT_ROOT=${RQ3_OUTPUT_ROOT:-$RQ3_STORAGE_ROOT/outputs/rq3/nscc}
export RQ3_LOG_DIR=${RQ3_LOG_DIR:-$RQ3_STORAGE_ROOT/logs/nscc}
export RQ3_EXTERNAL_DIR=${RQ3_EXTERNAL_DIR:-$RQ3_STORAGE_ROOT/external}
export RQ3_ENV_PREFIX=${RQ3_ENV_PREFIX:-$RQ3_STATE_DIR/envs/ai6132}
export RQ3_CONFIG=${RQ3_CONFIG:-$RQ3_WORKDIR/configs/rq3.yaml}
export RQ3_PREPARED_DATA=${RQ3_PREPARED_DATA:-$RQ3_STATE_DIR/data/rq3}
export RQ3_ENV_STAMP=$RQ3_ENV_PREFIX/.rq3-ready.json
export HF_HOME=${HF_HOME:-$RQ3_STATE_DIR/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_XET_CACHE=${HF_XET_CACHE:-$HF_HOME/xet}
export HF_ASSETS_CACHE=${HF_ASSETS_CACHE:-$HF_HOME/assets}
# Legacy aliases must not silently redirect libraries back to ~/.cache.
[[ -z ${TRANSFORMERS_CACHE+x} ]] || export TRANSFORMERS_CACHE=$HF_HUB_CACHE
[[ -z ${HUGGINGFACE_HUB_CACHE+x} ]] || export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TORCH_HOME=${TORCH_HOME:-$RQ3_STATE_DIR/cache/torch}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$RQ3_STATE_DIR/cache/pip}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$RQ3_STATE_DIR/cache/conda}
export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-$RQ3_STATE_DIR/envs}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$RQ3_STATE_DIR/cache/xdg}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}
# Retaining large CUDA wheels adds no value after the environment is ready.
export PIP_NO_CACHE_DIR=1
export RQ3_CONDA_MODULE=${RQ3_CONDA_MODULE:-miniforge3}
export RQ3_CUDA_MODULE=${RQ3_CUDA_MODULE:-cuda/12.2.2}
export RQ3_TORCH_VERSION=${RQ3_TORCH_VERSION:-2.5.1}
export RQ3_TORCHVISION_VERSION=${RQ3_TORCHVISION_VERSION:-0.20.1}
export RQ3_TORCH_INDEX_URL=${RQ3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
export RQ3_EXPECTED_CUDA=${RQ3_EXPECTED_CUDA:-12.1}
export RQ3_MIN_DRIVER_VERSION=${RQ3_MIN_DRIVER_VERSION:-535.54.03}
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# Temporary writes use scheduler-local storage when provided; results and model
# caches stay on shared scratch for resume. Preserve an existing site TMPDIR.
RQ3_JOB_TAG=${PBS_JOBID:-interactive}
RQ3_JOB_TAG=${RQ3_JOB_TAG//\//_}
if [[ -n ${PBS_JOBFS:-} ]]; then
    export TMPDIR=$PBS_JOBFS/rq3-$RQ3_JOB_TAG
elif [[ -z ${TMPDIR:-} ]]; then
    export TMPDIR=$RQ3_STATE_DIR/tmp/rq3-$RQ3_JOB_TAG
fi

rq3_error() {
    printf 'RQ3: %s\n' "$*" >&2
    return 1
}

# Resolve existing ancestors, including symlinks, without requiring Python or
# GNU realpath before Miniforge is loaded. Reject ambiguous override paths.
rq3_physical_path() {
    local path=$1 suffix='' part
    [[ $path == /* && $path != *:* && $path != *$'\n'* ]] || { rq3_error "Expected one absolute path: $path"; return 1; }
    case "/$path/" in
        */../*|*/./*) rq3_error "Use a normalized path without . or ..: $path"; return 1 ;;
    esac
    while [[ ! -d $path ]]; do
        [[ ! -e $path && ! -L $path ]] || { rq3_error "Not a directory: $path"; return 1; }
        part=${path##*/}
        suffix=/$part$suffix
        path=${path%/*}
        [[ -n $path ]] || path=/
    done
    printf '%s%s\n' "$(cd -- "$path" && pwd -P)" "$suffix"
}

rq3_assert_scratch_path() {
    local path
    path=$(rq3_physical_path "$1") || return 1
    case "$path/" in
        "$RQ3_SCRATCH_ROOT/"*) ;;
        *) rq3_error "Bulk path is outside RQ3_SCRATCH_ROOT: $1 -> $path. Unset the inherited override or point it to scratch."; return 1 ;;
    esac
}

rq3_paths() {
    [[ -f $RQ3_WORKDIR/src/rq3/run.py ]] || { rq3_error "Set RQ3_WORKDIR to the ai6132_group repository: $RQ3_WORKDIR"; return 1; }
    local variable temp_path
    for variable in RQ3_STORAGE_ROOT RQ3_STATE_DIR RQ3_OUTPUT_ROOT RQ3_LOG_DIR RQ3_EXTERNAL_DIR RQ3_ENV_PREFIX RQ3_PREPARED_DATA \
                    HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_XET_CACHE HF_ASSETS_CACHE TORCH_HOME PIP_CACHE_DIR CONDA_PKGS_DIRS CONDA_ENVS_PATH XDG_CACHE_HOME MPLCONFIGDIR; do
        rq3_assert_scratch_path "${!variable}" || return 1
    done
    temp_path=$(rq3_physical_path "$TMPDIR") || return 1
    case "$temp_path/" in
        "$RQ3_HOME_PHYSICAL/"*) rq3_error "TMPDIR resolves inside HOME: $TMPDIR"; return 1 ;;
    esac
    mkdir -p -- "$RQ3_STATE_DIR/logs" "$RQ3_OUTPUT_ROOT" "$RQ3_LOG_DIR" "$RQ3_EXTERNAL_DIR" \
        "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$TMPDIR"
    cd -- "$RQ3_WORKDIR" || return 1
}

# Keep protocol paths unchanged while editable sources live on scratch. Never
# move or replace an existing checkout: installed prefixes embed absolute paths.
rq3_link_sources() {
    local link=$RQ3_WORKDIR/external
    if [[ -e $link || -L $link ]]; then
        [[ -d $link ]] || { rq3_error "Broken or invalid source directory: $link"; return 1; }
        rq3_assert_scratch_path "$link" || {
            rq3_error 'Existing external/ is outside scratch. Use a fresh code-only checkout for this layout; see the migration notes.'
            return 1
        }
    else
        ln -s -- "$RQ3_EXTERNAL_DIR" "$link" || return 1
    fi
}

rq3_quota_report() {
    printf 'Bulk storage: %s\nEnvironment: %s\nOutputs: %s\n' "$RQ3_STORAGE_ROOT" "$RQ3_ENV_PREFIX" "$RQ3_OUTPUT_ROOT"
    if command -v myquota >/dev/null 2>&1; then
        myquota || printf 'RQ3: myquota failed; account quota is unverified.\n' >&2
    else
        printf 'RQ3: myquota unavailable here; check it in your NSCC login shell before submission.\n' >&2
    fi
}

rq3_load_modules() {
    if ! type module >/dev/null 2>&1; then
        local rq3_module_init
        for rq3_module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
            if [[ -r $rq3_module_init ]]; then
                # shellcheck disable=SC1090
                source "$rq3_module_init"
                break
            fi
        done
    fi
    type module >/dev/null 2>&1 || { rq3_error 'Environment Modules is unavailable; run in the NSCC login/PBS shell.'; return 1; }
    module load "$RQ3_CONDA_MODULE" "$RQ3_CUDA_MODULE" || return 1
    command -v conda >/dev/null 2>&1 || { rq3_error 'miniforge3 did not expose conda.'; return 1; }
}

rq3_activate_env() {
    rq3_paths || return 1
    rq3_load_modules || return 1
    rq3_paths || return 1  # Recheck any variables changed by site modules.
    [[ -x $RQ3_ENV_PREFIX/bin/python ]] || { rq3_error 'Run the bootstrap CPU job before submitting GPU jobs.'; return 1; }
    local rq3_conda_hook
    rq3_conda_hook=$(conda shell.bash hook) || return 1
    eval "$rq3_conda_hook"
    conda activate "$RQ3_ENV_PREFIX" || return 1
}

rq3_check_environment() {
    [[ -f $RQ3_ENV_STAMP ]] || { rq3_error "Environment is not ready: $RQ3_ENV_STAMP; run bootstrap first."; return 1; }
    "$RQ3_ENV_PREFIX/bin/python" -m src.rq3.prepare --config "$RQ3_CONFIG" --environment-stamp check --stamp "$RQ3_ENV_STAMP"
}

rq3_activate() {
    rq3_activate_env || return 1
    rq3_check_environment
}

rq3_check_prepared() {
    "$RQ3_ENV_PREFIX/bin/python" -m src.rq3.prepare --config "$RQ3_CONFIG" --output-dir "$RQ3_PREPARED_DATA" --check-only
}

rq3_check_driver() {
    command -v nvidia-smi >/dev/null 2>&1 || { rq3_error 'nvidia-smi is unavailable; request a GPU PBS allocation.'; return 1; }
    "$RQ3_ENV_PREFIX/bin/python" - <<'PY'
import json
import os
import subprocess
import torch

def version(value):
    return tuple(int(part) for part in value.strip().split('.'))

drivers = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()
minimum = os.environ['RQ3_MIN_DRIVER_VERSION']
if not drivers or any(version(driver) < version(minimum) for driver in drivers):
    raise SystemExit(f'RQ3 needs NVIDIA driver >= {minimum}; found {drivers}. No automatic wheel fallback is permitted.')
if torch.version.cuda != os.environ['RQ3_EXPECTED_CUDA']:
    raise SystemExit(f'Unexpected PyTorch CUDA runtime: {torch.version.cuda}')
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot access the allocated GPU.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('The frozen bfloat16 protocol requires a GPU with bfloat16 support.')
print(json.dumps({'driver_versions': drivers, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                  'visible_gpus': torch.cuda.device_count(), 'gpu': torch.cuda.get_device_name(0)}, indent=2))
PY
}
