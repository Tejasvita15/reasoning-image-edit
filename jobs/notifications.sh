#!/usr/bin/env bash
# Source from job wrappers. Only a .env PATH crosses the scheduler boundary;
# Python reads credentials without shell evaluation or command-line secrets.
AI6132_NOTIFY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

ai6132_notifications_begin() {
    AI6132_NOTIFY_LABEL=$1
    AI6132_NOTIFY_ESTIMATE=$2
    AI6132_NOTIFY_OUTPUT=${3:-}
    AI6132_NOTIFY_DIRECTORY=''
    export TELEGRAM_ENV_FILE=${TELEGRAM_ENV_FILE:-$AI6132_NOTIFY_ROOT/.env}
    # A child ProgressNotifier marks terminal handling to suppress a duplicate
    # shell completion/failure alert. Each allocation gets a fresh marker.
    if AI6132_NOTIFY_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/ai6132-notify.XXXXXXXX"); then
        export TELEGRAM_TERMINAL_MARKER=$AI6132_NOTIFY_DIRECTORY/terminal.json
    else
        unset TELEGRAM_TERMINAL_MARKER
        printf 'Telegram: cannot create completion marker; duplicate terminal alerts are possible.\n' >&2
    fi
    return 0
}

ai6132_notify_event() {
    local status=$1 code=${2:-0} interpreter='' candidate
    local candidates=()
    if [[ -n ${RQ3_ENV_PREFIX:-} && -x $RQ3_ENV_PREFIX/bin/python ]]; then
        candidates+=("$RQ3_ENV_PREFIX/bin/python")
    fi
    if command -v python3 >/dev/null 2>&1; then candidates+=("$(command -v python3)"); fi
    if command -v python >/dev/null 2>&1; then candidates+=("$(command -v python)"); fi
    for candidate in "${candidates[@]}"; do
        if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
            interpreter=$candidate
            break
        fi
    done
    if [[ -z $interpreter ]]; then
        printf 'Telegram: Python >=3.9 is unavailable; notification skipped.\n' >&2
        return 0
    fi
    # The helper uses only the standard library, including before bootstrap.
    "$interpreter" "$AI6132_NOTIFY_ROOT/src/notifications.py" event \
        --label "$AI6132_NOTIFY_LABEL" --status "$status" --exit-code "$code" \
        --estimated-seconds "$AI6132_NOTIFY_ESTIMATE" --output "$AI6132_NOTIFY_OUTPUT" \
        </dev/null || printf 'Telegram: notification helper failed; job status is unchanged.\n' >&2
    return 0
}

ai6132_notifications_finish() {
    local code=$1 status
    case $code in
        0) status=completed ;;
        130|137|143) status=interrupted ;;
        *) status=failed ;;
    esac
    if [[ -z ${TELEGRAM_TERMINAL_MARKER:-} || ! -f $TELEGRAM_TERMINAL_MARKER ]]; then
        ai6132_notify_event "$status" "$code"
    fi
    if [[ -n ${AI6132_NOTIFY_DIRECTORY:-} ]]; then
        rm -f -- "$AI6132_NOTIFY_DIRECTORY/terminal.json" || true
        rmdir -- "$AI6132_NOTIFY_DIRECTORY" 2>/dev/null || true
    fi
    return 0
}
