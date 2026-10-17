#!/usr/bin/env bash
# Paste into NSCC Bash after setting RQ3_ANALYSIS_DIR to the completed run.
# Packages analysis artifacts only; frozen experiment code and records are unchanged.
(
set -euo pipefail
cd "${RQ3_WORKDIR:-$HOME/ai6132_group}"
source jobs/nscc/common.sh
rq3_paths
: "${RQ3_ANALYSIS_DIR:?Set RQ3_ANALYSIS_DIR to the completed analysis directory}"
test -d "$RQ3_ANALYSIS_DIR"
test -x "$RQ3_ENV_PREFIX/bin/python"
RQ3_NAME=$(basename -- "$RQ3_ANALYSIS_DIR")
case "$RQ3_NAME" in
    analysis_manual|analysis_manual.[A-Za-z0-9]*) ;;
    *) echo 'Expected an analysis_manual directory.' >&2; exit 1 ;;
esac
RQ3_PACKAGE="$RQ3_STORAGE_ROOT/transfers/analysis_export/$RQ3_NAME"
rq3_assert_scratch_path "$RQ3_PACKAGE"
mkdir -p "$RQ3_PACKAGE"
export RQ3_ANALYSIS_DIR RQ3_PACKAGE

"$RQ3_ENV_PREFIX/bin/python" -I - <<'PY'
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

analysis = Path(os.environ['RQ3_ANALYSIS_DIR']).resolve()
root = Path(os.environ['RQ3_OUTPUT_ROOT'])
merged = root / 'main' / 'merged'
bundle = root / 'chatgpt_main'
package = Path(os.environ['RQ3_PACKAGE'])

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read(path):
    return json.loads(path.read_text())

report = read(analysis / 'report.json')
merge = read(merged / 'merge.json')
manifest = read(bundle / 'private' / 'manifest.json')
generation_hash = manifest['generation_protocol_hash']
evaluation_hash = manifest['evaluation_protocol_hash']
assert report['generation_protocol_hash'] == merge['protocol_hash'] == generation_hash, 'Generation protocol mismatch'
assert report['evaluation_protocol_hash'] == evaluation_hash, 'Evaluation protocol mismatch'
assert report['expected_quality_outputs'] == 200, 'Expected 200 scored outputs'
snapshot = manifest['generation_snapshot']
assert read(merged / 'protocol_snapshot.json') == snapshot, 'Archived protocol snapshot differs'
assert read(merged / 'protocol.json') == {**snapshot['config'], 'protocol_hash': generation_hash}, 'Archived configuration differs'
generations = merged / 'generations.jsonl'
assert sha(generations) == merge['generations_sha256'] == manifest['generations_sha256'], 'Generation log changed'
rows = [json.loads(line) for line in generations.read_text().splitlines() if line.strip()]
assert len(rows) == merge['completed_generations'] == 800, 'Expected all 800 generation records'
keys = {(r['example_uid'], r['condition'], r['base_seed'], r['repeat']) for r in rows}
assert len(keys) == 800 and {r['base_seed'] for r in rows} == {42}, 'Duplicate rows or unexpected seed'
quality = [r for r in rows if r['repeat'] == 0]
assert Counter(r['condition'] for r in quality) == Counter(A=40, B=40, C=40, D=40, E=20, F=20), 'Quality coverage differs'
assert len({r['example_uid'] for r in quality}) == 40, 'Expected 40 examples'
assert keys == {(r['example_uid'], r['condition'], 42, repeat) for r in quality for repeat in range(4)}, 'Missing timing repetitions'
assert all(r['status'] == 'ok' and r['protocol_hash'] == generation_hash for r in rows), 'Invalid generation records'
scores = [json.loads(line) for line in (bundle / 'scores.jsonl').read_text().splitlines() if line.strip()]
assert len(scores) == 200, 'Expected 200 scores'
assert all(r['generation_protocol_hash'] == generation_hash and r['evaluation_protocol_hash'] == evaluation_hash for r in scores), 'Score protocol mismatch'
assert {(r['example_uid'], r['condition'], r['base_seed']) for r in scores} == {(r['example_uid'], r['condition'], r['base_seed']) for r in quality}, 'Score coverage differs'
assert all(r['bundle_manifest_hash'] == manifest['manifest_hash'] for r in scores), 'Scores refer to another judging bundle'

files = {}
for path in sorted(analysis.rglob('*')):
    if path.is_symlink():
        raise ValueError(f'Unexpected symlink in analysis: {path}')
    if path.is_file():
        files['analysis/' + path.relative_to(analysis).as_posix()] = path
for name in ('report.json', 'paired_effects.csv', 'quality_latency.png', 'cache_activity.json', 'run.log'):
    assert 'analysis/' + name in files, f'Missing analysis artifact: {name}'
assert 'Analysis exit code: 0' in (analysis / 'run.log').read_text(), 'Analysis log does not record success'
for name in ('generations.jsonl', 'merge.json', 'protocol.json', 'protocol_snapshot.json', 'runtime.json'):
    files['records/' + name] = merged / name
files['records/scores.jsonl'] = bundle / 'scores.jsonl'
for name in ('rq3_determinism.json', 'rq3_components.json', 'rq3_nscc_budget.json'):
    path = Path('reports') / name
    if path.is_file():
        assert read(path)['protocol_hash'] == generation_hash, f'Stale pilot report: {name}'
        files['provenance/' + name] = path
files['provenance/calibration_repeatability.json'] = Path('reports/rq3_chatgpt_calibration/repeatability.json')
assert read(files['provenance/calibration_repeatability.json']) == manifest['calibration_report'], 'Calibration report differs from the frozen main bundle'
files['provenance/judging_manifest.json'] = bundle / 'private' / 'manifest.json'
inventory = {name: {'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in sorted(files.items())}
metadata = {
    'schema_version': 1, 'kind': 'rq3_analysis_transfer',
    'analysis_source': str(analysis), 'generation_protocol_hash': generation_hash,
    'evaluation_protocol_hash': evaluation_hash, 'bundle_manifest_hash': manifest['manifest_hash'],
    'generations': 800, 'quality_outputs': 200, 'examples': 40, 'files': inventory,
    'note': 'Original NSCC paths and records are preserved. Use the existing chatgpt_main bundle for local images.',
}
archive = package / 'rq3_analysis.zip'
if archive.exists():
    with zipfile.ZipFile(archive) as zipped:
        assert json.loads(zipped.read('rq3_analysis/TRANSFER.json')) == metadata, 'Sources changed since packaging; retain this package and use a new completed analysis directory'
        expected_names = {'rq3_analysis/' + name for name in files} | {'rq3_analysis/TRANSFER.json'}
        assert len(zipped.namelist()) == len(expected_names) and set(zipped.namelist()) == expected_names, 'Archive entries changed'
        for name, info in inventory.items():
            assert hashlib.sha256(zipped.read('rq3_analysis/' + name)).hexdigest() == info['sha256'], f'Archived bytes changed: {name}'
else:
    with tempfile.NamedTemporaryFile(dir=package, prefix='.analysis-', suffix='.zip', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for name, path in sorted(files.items()):
                data = path.read_bytes()
                assert hashlib.sha256(data).hexdigest() == inventory[name]['sha256'], f'File changed during packaging: {name}'
                zipped.writestr('rq3_analysis/' + name, data)
            zipped.writestr('rq3_analysis/TRANSFER.json', json.dumps(metadata, indent=2) + '\n')
        os.link(temporary_path, archive)  # Publish without replacing another package.
    finally:
        temporary_path.unlink(missing_ok=True)

def publish(path, data):
    if path.exists():
        assert path.read_bytes() == data, f'Existing transfer file conflicts: {path}'
        return
    with tempfile.NamedTemporaryFile(dir=package, prefix='.part-', delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

parts = []
with archive.open('rb') as handle:
    while chunk := handle.read(45 * 1024 * 1024):
        name = f'rq3_analysis.zip.part-{len(parts):04d}'
        publish(package / name, chunk)
        parts.append(name)
assert {p.name for p in package.glob('rq3_analysis.zip.part-*')} == set(parts), 'Unexpected leftover chunks'
publish(package / 'ARCHIVE.sha256', f'{sha(archive)}  rq3_analysis.zip\n'.encode())
publish(package / 'PARTS.sha256', ''.join(f'{sha(package / name)}  {name}\n' for name in parts).encode())
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'chunks': len(parts), 'files': len(files)}))
PY

RQ3_XFER="$RQ3_STORAGE_ROOT/transfers/chatgpt_main_incremental"
test -d "$RQ3_XFER/.git"
cd "$RQ3_XFER"
test "$(git branch --show-current)" = rq3_transfer_main
test -z "$(git status --porcelain --untracked-files=no)"
git pull --ff-only origin rq3_transfer_main
RQ3_REF=refs/heads/rq3_transfer_main
rq3_analysis_push_confirmed() {
    local wanted seen
    wanted=$(git rev-parse HEAD)
    if ! git push origin "HEAD:$RQ3_REF"; then
        echo 'Push reported an error; checking whether GitHub accepted the commit.'
    fi
    seen=$(git ls-remote --exit-code origin "$RQ3_REF" | awk '{print $1}')
    if [ "$seen" != "$wanted" ]; then
        echo 'Remote commit differs. Rerun this block to resume the pending push.' >&2
        return 1
    fi
    printf 'Remote confirmed: %s\n' "$wanted"
}

rq3_analysis_push_confirmed
RQ3_REMOTE_DIR="analysis_return/$RQ3_NAME"
mkdir -p "$RQ3_REMOTE_DIR"
for RQ3_FILE in "$RQ3_PACKAGE"/rq3_analysis.zip.part-* "$RQ3_PACKAGE/ARCHIVE.sha256" "$RQ3_PACKAGE/PARTS.sha256"; do
    RQ3_DEST="$RQ3_REMOTE_DIR/${RQ3_FILE##*/}"
    if [ -e "$RQ3_DEST" ]; then
        cmp "$RQ3_FILE" "$RQ3_DEST"
    else
        cp -- "$RQ3_FILE" "$RQ3_DEST"
    fi
    git add -- "$RQ3_DEST"
    if ! git diff --cached --quiet; then
        git -c user.name='RQ3 transfer' -c user.email='rq3-transfer@localhost' commit -m "Transfer $RQ3_DEST"
    fi
    rq3_analysis_push_confirmed
done
printf 'Analysis transfer confirmed: %s\n' "$RQ3_REMOTE_DIR"
)
