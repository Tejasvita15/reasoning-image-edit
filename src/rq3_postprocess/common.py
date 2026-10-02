"""Validate archived evidence and resolve local images without editing records."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.rq3.analyze import (
    ABLATION_CONTRASTS, CONTRASTS, example_contrasts, make_score_table,
    paired_summary, summarize_timing, validate_pair_metadata,
)
from src.rq3.evaluate import read_jsonl, row_key
from src.rq3_manual.common import load_bundle
from src.rq3_manual.results import validate_score_row


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values, *, resamples=10000, confidence=.95, seed=42):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires finite, nonempty example-level values')
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    bounds = np.quantile(values[draws].mean(axis=1), [(1-confidence)/2, 1-(1-confidence)/2])
    return {'n_examples': len(values), 'mean': float(values.mean()),
            'ci_low': float(bounds[0]), 'ci_high': float(bounds[1])}


def checked_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Inventory paths must be relative')
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Inventory path escapes its directory')
    return path


@dataclass
class AnalysisData:
    root: Path
    bundle_root: Path
    manifest: dict
    transfer: dict
    report: dict
    generation_rows: list
    quality: dict
    scores: dict
    timing: dict
    paths: dict
    uids: list
    subset: list
    statistics: dict


def load_data(analysis, bundle):
    root, bundle_root = Path(analysis).resolve(), Path(bundle).resolve()
    transfer = json.loads((root / 'TRANSFER.json').read_text())
    if transfer.get('schema_version') != 1 or transfer.get('kind') != 'rq3_analysis_transfer':
        raise ValueError('Not a supported RQ3 analysis transfer')
    for name, expected in transfer['files'].items():
        path = checked_path(root, name)
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Archived artifact changed: {name}')
    manifest = load_bundle(bundle_root)
    if manifest['kind'] != 'main' or manifest['seeds'] != [42]:
        raise ValueError('This supplementary analysis supports the completed seed-42 main experiment')
    for field in ('generation_protocol_hash', 'evaluation_protocol_hash'):
        if manifest[field] != transfer[field]:
            raise ValueError(f'Transfer identity differs: {field}')
    if manifest['manifest_hash'] != transfer['bundle_manifest_hash']:
        raise ValueError('Transfer uses a different judging bundle')
    snapshot = manifest['generation_snapshot']
    statistics = snapshot['config']['statistics']
    uids, subset = snapshot['splits']['k_small'], snapshot['subset']['example_uids']
    expected = {row_key(item['row']) for item in manifest['items']}
    generations = read_jsonl(root / 'records/generations.jsonl')
    if sha256(root / 'records/generations.jsonl') != manifest['generations_sha256']:
        raise ValueError('Generation log differs from the judged experiment')
    if len(generations) != 4 * len(expected):
        raise ValueError('Expected quality records and exactly three timing repetitions')
    medians, failures, successful = summarize_timing(generations, expected, manifest['generation_protocol_hash'], 3)
    if failures or any({key[3] for key in successful if key[:3] == cell} != {0, 1, 2, 3} for cell in expected):
        raise ValueError('Generation coverage is incomplete')
    validate_pair_metadata(successful, expected)
    public = {**snapshot['config'], 'protocol_hash': manifest['generation_protocol_hash']}
    if any(row['config'] != public for row in generations):
        raise ValueError('Generation configuration differs from frozen settings')
    scores = make_score_table(read_jsonl(root / 'records/scores.jsonl'), expected, manifest['generation_protocol_hash'])
    items = {row_key(item['row']): item for item in manifest['items']}
    for key, score in scores.items():
        validate_score_row(score)
        item = items[key]
        if (score['evaluation_protocol_hash'] != manifest['evaluation_protocol_hash']
                or score['bundle_manifest_hash'] != manifest['manifest_hash']
                or score['generation_snapshot'] != snapshot
                or score['selection_hash'] != manifest['selection_hash']
                or score['source_sha256'] != item['row']['source_sha256']
                or score['output_sha256'] != item['row']['output_sha256']
                or score['pass_number'] != 1):
            raise ValueError(f'Score identity changed: {key}')
        if successful[(*key, 0)] != item['row']:
            raise ValueError(f'Quality record differs from original manifest: {key}')
    indexed = {item['item_key']: item for item in manifest['items']}
    paths = {}
    for batch in manifest['batches']:
        for entry in batch['items']:
            row = indexed[entry['item_key']]['row']
            key = (row['example_uid'], row['condition'])
            paths[key] = {name: checked_path(bundle_root, entry[field]) for name, field in
                          (('source', 'source_file'), ('edited', 'edited_file'), ('ground_truth', 'ground_truth_file'))
                          if field in entry}
    report = json.loads((root / 'analysis/report.json').read_text())
    for field in ('generation_protocol_hash', 'evaluation_protocol_hash'):
        if report[field] != manifest[field]:
            raise ValueError(f'Canonical report identity mismatch: {field}')
    kwargs = {'resamples': statistics['bootstrap_resamples'], 'confidence': statistics['confidence'],
              'seed': statistics['bootstrap_seed']}
    for ids, definitions, archived in ((uids, CONTRASTS, report['contrasts']),
                                      (subset, ABLATION_CONTRASTS, report['ablations']['contrasts'])):
        for name, coefficients in definitions.items():
            recalculated = paired_summary(example_contrasts(scores, ids, [42], coefficients), **kwargs)
            for field in ('n_examples', 'mean', 'ci_low', 'ci_high', 'wilcoxon_p'):
                if not math.isclose(recalculated[field], archived[name][field], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f'Canonical effect does not reproduce locally: {name}/{field}')
    quality = {(uid, condition): successful[(uid, condition, seed, 0)] for uid, condition, seed in expected}
    return AnalysisData(root, bundle_root, manifest, transfer, report, generations, quality,
                        {(uid, condition): value for (uid, condition, _), value in scores.items()},
                        {(uid, condition): value for (uid, condition, _), value in medians.items()},
                        paths, uids, subset, statistics)
