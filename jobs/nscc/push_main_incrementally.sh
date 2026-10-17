#!/usr/bin/env bash
# Resume main-bundle transfer with one confirmed Git push per artifact.
# May be pasted into an NSCC Bash shell; no Conda activation is needed.
(
set -euo pipefail
cd "$HOME/ai6132_group"
source jobs/nscc/common.sh
rq3_paths
RQ3_SOURCE="$RQ3_STORAGE_ROOT/transfers/chatgpt_main_repair"
RQ3_XFER="$RQ3_STORAGE_ROOT/transfers/chatgpt_main_incremental"
RQ3_REF=refs/heads/rq3_transfer_main
test -d "$RQ3_SOURCE/.git"
git -C "$RQ3_SOURCE" ls-files --error-unmatch PARTS.sha256 BUNDLE.sha256 > /dev/null
test -z "$(git -C "$RQ3_SOURCE" status --porcelain --untracked-files=no)"
(
cd "$RQ3_SOURCE"
sha256sum -c PARTS.sha256
)
(
cd "$RQ3_OUTPUT_ROOT"
sha256sum -c "$RQ3_SOURCE/BUNDLE.sha256"
)
RQ3_LOCAL_HEAD=$(git -C "$RQ3_SOURCE" rev-parse HEAD)
RQ3_REMOTE_HEAD=$(git -C "$RQ3_SOURCE" ls-remote --exit-code origin "$RQ3_REF" | awk '{print $1}')
if [ "$RQ3_LOCAL_HEAD" = "$RQ3_REMOTE_HEAD" ]; then
    echo 'The complete repair commit is already on GitHub. Continue with the Mac pull/checksum commands.'
    exit 0
fi
if [ ! -d "$RQ3_XFER/.git" ]; then
    RQ3_REMOTE=$(git -C "$RQ3_SOURCE" remote get-url origin)
    git clone --depth 1 --single-branch --branch rq3_transfer_main "$RQ3_REMOTE" "$RQ3_XFER"
fi
cd "$RQ3_XFER"
test "$(git branch --show-current)" = rq3_transfer_main
test -z "$(git status --porcelain --untracked-files=no)"
git pull --ff-only origin rq3_transfer_main

rq3_push_confirmed() {
    local wanted seen
    wanted=$(git rev-parse HEAD)
    if ! git push origin "HEAD:$RQ3_REF"; then
        echo 'Push reported an error; checking whether GitHub accepted the commit.'
    fi
    seen=$(git ls-remote --exit-code origin "$RQ3_REF" | awk '{print $1}')
    if [ "$seen" != "$wanted" ]; then
        echo 'Remote commit does not match. Stopped; rerun this block to retry before adding more chunks.' >&2
        return 1
    fi
    printf 'Remote confirmed: %s\n' "$wanted"
}

# Retry any previously committed but unaccepted chunk before adding another.
rq3_push_confirmed
for RQ3_FILE in "$RQ3_SOURCE"/chatgpt_main.zip.part-* "$RQ3_SOURCE/BUNDLE.sha256" "$RQ3_SOURCE/PARTS.sha256"; do
    RQ3_NAME=${RQ3_FILE##*/}
    if [ -e "$RQ3_NAME" ]; then
        cmp "$RQ3_FILE" "$RQ3_NAME"
    else
        cp "$RQ3_FILE" "$RQ3_NAME"
    fi
    git add -- "$RQ3_NAME"
    if ! git diff --cached --quiet; then
        git -c user.name='RQ3 transfer' -c user.email='rq3-transfer@localhost' commit -m "Transfer $RQ3_NAME"
    fi
    rq3_push_confirmed
done
echo 'All chunks and checksum files are confirmed on GitHub.'
)
