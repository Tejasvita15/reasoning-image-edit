#!/usr/bin/env bash
# Read-only size audit, apart from creating the normal scratch/log directories.
# Recursive scans belong on compute nodes, not NSCC login nodes.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[[ -n ${PBS_JOBID:-} ]] || { rq3_error 'Run this audit via storage.pbs or inside an interactive CPU allocation.'; exit 1; }
rq3_paths
exec > >(tee -a "$RQ3_LOG_DIR/storage_${PBS_JOBID}.log") 2>&1
rq3_quota_report
printf '\nFilesystem free space (NOT your account quota):\n'
df -h "$HOME" "$RQ3_SCRATCH_ROOT"
printf '\nAllocated bytes by RQ3 directory (hardlinks counted once):\n'
rq3_audit_paths=()
for rq3_path in "$RQ3_STATE_DIR/envs" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TORCH_HOME" \
                "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$RQ3_PREPARED_DATA" \
                "$RQ3_EXTERNAL_DIR" "$RQ3_OUTPUT_ROOT" "$RQ3_LOG_DIR" "$RQ3_STATE_DIR/tmp"; do
    [[ ! -d $rq3_path ]] || rq3_audit_paths+=("$rq3_path")
done
du -sch -- "${rq3_audit_paths[@]}"
printf '\nHOME directories (symlinks to scratch are not followed):\n'
du -h --max-depth=1 -- "$HOME"
printf '\nNo files were deleted. Account quotas also include files outside this experiment.\n'
