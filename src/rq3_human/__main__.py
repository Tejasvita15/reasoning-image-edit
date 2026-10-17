"""Validate paired human submissions and analyse the same frozen 30 examples."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil

import numpy as np

from src.rq3.analyze import CONTRASTS
from src.rq3_postprocess.common import load_data, sha256, write_csv
from .metrics import icc_a1, quadratic_kappa, spearman, summary

ROOT = Path(__file__).resolve().parents[2]
SCORE_FIELDS = ('consistency_score', 'instruction_score', 'quality_score', 'knowledge_score')
HEADERS = ['reviewer_id', 'presentation_order', 'blind_id', 'knowledge_required', 'status',
           *SCORE_FIELDS, 'rationale', 'confidence', 'ambiguity_note']
REVIEWERS = ('reviewer_1', 'reviewer_2')
CONDITIONS = 'ABCD'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def parse_csv(path):
    """Keep invalid bytes visible only in optional notes, never repair scores."""
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig', errors='surrogateescape'), newline=''))
    if reader.fieldnames != HEADERS:
        raise ValueError(f'{path}: CSV headers differ from the original template')
    rows, encoding_warnings = [], []
    for line, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'{path}:{line}: malformed CSV')
        for field, value in row.items():
            invalid = [ord(c) - 0xDC00 for c in value if 0xDC80 <= ord(c) <= 0xDCFF]
            if invalid:
                if field != 'ambiguity_note':
                    raise ValueError(f'{path}:{line}: invalid UTF-8 outside optional ambiguity note')
                encoding_warnings.append({'line': line, 'blind_id': row['blind_id'],
                                          'field': field, 'invalid_bytes_hex': [f'{b:02x}' for b in invalid]})
                row[field] = value.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='backslashreplace')
        rows.append(row)
    return rows, encoding_warnings


def validate_submission(root, reviewer, mapping, export):
    folder = root / reviewer
    warnings, summaries, ratings = [], {}, {}
    for phase, filename in (('main', 'ratings.csv'), ('practice', 'practice_ratings.csv')):
        expected = {row['blind_id']: row for row in mapping
                    if row['reviewer_id'] == reviewer and row['phase'] == phase}
        rows, encoding = parse_csv(folder / filename)
        warnings.extend({'file': filename, 'kind': 'escaped_invalid_UTF8_note', **w} for w in encoding)
        if len(rows) != len(expected) or Counter(row['blind_id'] for row in rows) != Counter(expected.keys()):
            raise ValueError(f'{reviewer}: {phase} coverage/duplicate-ID error')
        normalized_booleans = 0
        for row in rows:
            ref = expected[row['blind_id']]
            if row['reviewer_id'] != reviewer or row['presentation_order'] != str(ref['presentation_order']):
                raise ValueError(f'{reviewer}: edited identity or ordering')
            boolean = row['knowledge_required'].lower()
            if boolean != str(ref['knowledge_required']).lower():
                raise ValueError(f'{reviewer}: knowledge applicability changed')
            normalized_booleans += boolean != row['knowledge_required']
            if row['status'] != 'ok':
                raise ValueError(f'{reviewer}: incomplete/unreadable outputs require explicit missing-data review')
            applicable = SCORE_FIELDS if ref['knowledge_required'] else SCORE_FIELDS[:3]
            for field in applicable:
                if re.fullmatch('[1-5]', row[field]) is None:
                    raise ValueError(f'{reviewer}: invalid {field} for {row["blind_id"]}')
            if not ref['knowledge_required'] and row['knowledge_score'] != '':
                raise ValueError(f'{reviewer}: non-applicable knowledge must be empty')
            if ref['knowledge_required'] and int(row['instruction_score']) <= 2 and row['knowledge_score'] != '1':
                raise ValueError(f'{reviewer}: violated instruction/knowledge scoring rule')
            if not row['rationale'].strip() or row['confidence'] not in ('', '1', '2', '3'):
                raise ValueError(f'{reviewer}: missing rationale or invalid confidence')
            if phase == 'main':
                numeric = {field: int(row[field]) if field in applicable else None for field in SCORE_FIELDS}
                numeric['Q'] = float(np.mean([25 * (numeric[field] - 1) for field in applicable]))
                ratings[(ref['example_uid'], ref['condition'])] = {
                    **numeric, 'blind_id': row['blind_id'], 'rationale': row['rationale'],
                    'ambiguity_note': row['ambiguity_note'], 'mapping': ref}
        summaries[phase] = {'rows': len(rows), 'knowledge_applicable': sum(r['knowledge_required'] for r in expected.values()),
                            'boolean_case_normalizations': normalized_booleans,
                            'encoding_warning_rows': len(encoding), 'status': 'passed'}
    metadata = json.loads((folder / 'reviewer_metadata.json').read_text())
    if (metadata['reviewer_id'] != reviewer or metadata['independent_rating_confirmed'] is not True
            or not metadata['prior_exposure'].strip() or not metadata['background'].strip()):
        raise ValueError(f'{reviewer}: metadata declaration incomplete')
    fields = (['practice_started_at'] if 'practice_started_at' in metadata else []) + [
        'practice_completed_at', 'started_at', 'completed_at']
    times = [datetime.fromisoformat(metadata[key].replace('Z', '+00:00')) for key in fields]
    if any(t.tzinfo is None for t in times) or any(a > b for a, b in zip(times, times[1:])):
        raise ValueError(f'{reviewer}: invalid metadata timestamp ordering/timezone')
    mutable = {f'{reviewer}/{name}' for name in ('ratings.csv', 'practice_ratings.csv', 'reviewer_metadata.json')}
    checked = 0
    for relative, expected in export['files'].items():
        if relative.startswith(reviewer + '/') and relative not in mutable:
            if sha256(root / relative) != expected:
                raise ValueError(f'Reviewer reference packet changed: {relative}')
            checked += 1
    hashes = {name: sha256(folder / name) for name in ('ratings.csv', 'practice_ratings.csv', 'reviewer_metadata.json')}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    archive = root / 'submissions' / reviewer / fingerprint[:16]
    archive.mkdir(parents=True, exist_ok=True)
    for name, expected in hashes.items():
        if not (archive / name).exists():
            shutil.copyfile(folder / name, archive / name)
        if sha256(archive / name) != expected:
            raise ValueError('Submission snapshot collision')
    return ratings, {'summary': summaries, 'warnings': warnings, 'metadata': metadata,
                     'immutable_packet_files_checked': checked, 'file_sha256': hashes,
                     'snapshot': str(archive), 'independence': 'operator/reviewer attested, not independently verified'}


def build_analysis(data, subset, submissions, draws):
    uids = subset['example_uids']
    vectors = {}
    joined = []
    for rater in (*REVIEWERS, 'chatgpt'):
        vectors[rater] = {}
        for field in (*SCORE_FIELDS, 'Q'):
            values = []
            for uid in uids:
                row = []
                for condition in CONDITIONS:
                    record = (data.scores[(uid, condition)] if rater == 'chatgpt'
                              else submissions[rater][(uid, condition)])
                    value = record[field]
                    row.append(float(value) if value is not None else np.nan)
                values.append(row)
            vectors[rater][field] = np.asarray(values)
    vectors['human_mean'] = {'Q': (vectors['reviewer_1']['Q'] + vectors['reviewer_2']['Q']) / 2}
    for uid in uids:
        for condition in CONDITIONS:
            original = data.scores[(uid, condition)]
            line = {'example_uid': uid, 'condition': condition, 'knowledge_required': original['knowledge_required'],
                    'instruction': data.quality[(uid, condition)]['instruction']}
            for rater in REVIEWERS:
                record = submissions[rater][(uid, condition)]
                if (record['mapping']['output_sha256'] != original['output_sha256']
                        or record['mapping']['source_sha256'] != original['source_sha256']
                        or record['mapping']['source_bundle_manifest_hash'] != data.manifest['manifest_hash']):
                    raise ValueError('Human and original judge rated different images')
                for field in (*SCORE_FIELDS, 'Q', 'blind_id', 'rationale', 'ambiguity_note'):
                    line[f'{rater}_{field}'] = record[field]
            for field in (*SCORE_FIELDS, 'Q'):
                line[f'chatgpt_{field}'] = original[field]
            line['chatgpt_rationale'] = original['rationale']
            line['human_mean_Q'] = (line['reviewer_1_Q'] + line['reviewer_2_Q']) / 2
            line['chatgpt_minus_human_mean_Q'] = line['chatgpt_Q'] - line['human_mean_Q']
            joined.append(line)
    agreements = []
    pairs = (('reviewer_1', 'reviewer_2'), ('reviewer_1', 'chatgpt'), ('reviewer_2', 'chatgpt'))
    for first, second in pairs:
        metrics = [('Q', 'spearman_rho', spearman), ('Q', 'icc_a1', icc_a1),
                   ('Q', 'mean_difference', lambda a, b: np.nanmean(b-a, axis=-1)),
                   ('Q', 'mean_absolute_difference', lambda a, b: np.nanmean(np.abs(b-a), axis=-1))]
        if second == 'reviewer_2':
            metrics += [(field, 'quadratic_kappa', quadratic_kappa) for field in SCORE_FIELDS]
        for dimension, name, function in metrics:
            x, y = vectors[first][dimension], vectors[second][dimension]
            mask = np.isfinite(x) & np.isfinite(y)
            bx, by = x[draws].reshape(len(draws), -1), y[draws].reshape(len(draws), -1)
            point, boot = function(x.ravel(), y.ravel()), function(bx, by)
            agreements.append({'pair': f'{first}_vs_{second}', 'first': first, 'second': second,
                               'dimension': dimension, 'metric': name, 'n_outputs': int(mask.sum()),
                               'n_examples': int(mask.any(axis=1).sum()), **summary(point, boot)})
    means, effects, differences, by_example = [], [], [], []
    effect_vectors = {}
    for rater in (*REVIEWERS, 'chatgpt', 'human_mean'):
        q = vectors[rater]['Q']
        for column, condition in enumerate(CONDITIONS):
            means.append({'rater': rater, 'condition': condition, 'n_examples': len(uids),
                          **summary(q[:, column].mean(), q[draws, column].mean(axis=1))})
        effect_vectors[rater] = {}
        for name, coefficients in CONTRASTS.items():
            values = sum(coef * q[:, CONDITIONS.index(c)] for c, coef in coefficients.items())
            effect_vectors[rater][name] = values
            effects.append({'rater': rater, 'contrast': name, 'n_examples': len(uids),
                            **summary(values.mean(), values[draws].mean(axis=1))})
            by_example.extend({'rater': rater, 'example_uid': uid, 'contrast': name, 'difference': float(v)}
                              for uid, v in zip(uids, values))
    for rater in (*REVIEWERS, 'human_mean'):
        for name in CONTRASTS:
            values = effect_vectors[rater][name] - effect_vectors['chatgpt'][name]
            differences.append({'rater': rater, 'contrast': name, 'n_examples': len(uids),
                                'definition': 'human effect minus judge effect, same 30 examples',
                                **summary(values.mean(), values[draws].mean(axis=1))})
    decisions = {}
    for rater in (*REVIEWERS, 'chatgpt', 'human_mean'):
        table = {row['contrast']: row for row in effects if row['rater'] == rater}
        thinking = table['G0_B_minus_A']
        acceleration = table['thinking_on_acceleration_D_minus_B']
        interaction = table['interaction_D_minus_C_minus_B_plus_A']
        decisions[rater] = {'unaccelerated_thinking_lower_bound_above_zero': thinking['ci_low'] > 0,
                            'D_minus_B_lower_bound_above_minus_5': acceleration['ci_low'] > -5,
                            'interaction_interval_within_plus_minus_5': interaction['ci_low'] > -5 and interaction['ci_high'] < 5}
    return {'agreements': agreements, 'conditions': means, 'effects': effects,
            'effect_differences': differences, 'same_margin_sensitivity': decisions}, joined, by_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, default=ROOT / 'outputs/rq3/human_validation')
    parser.add_argument('--analysis', type=Path, default=ROOT / 'outputs/rq3/evidence/analysis')
    parser.add_argument('--bundle', type=Path, default=ROOT / 'outputs/rq3/evidence/chatgpt_main')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/rq3/human_validation/analysis')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Analysis destination exists; preserve it and use a fresh --output-dir')
    for protected in (args.review_root / 'reviewer_1', args.review_root / 'reviewer_2', args.analysis, args.bundle):
        if args.output_dir.resolve() == protected.resolve() or args.output_dir.resolve().is_relative_to(protected.resolve()):
            raise ValueError('Analysis output must be separate from source evidence')
    plan_path = ROOT / 'reports/rq3_human_analysis_plan.json'
    plan = json.loads(plan_path.read_text())
    root = args.review_root
    export = json.loads((root / 'private/export_manifest.json').read_text())
    for relative in ('private/mapping.json', 'private/subset.json'):
        if sha256(root / relative) != export['files'][relative]:
            raise ValueError('Human mapping/subset changed from export')
    mapping = json.loads((root / 'private/mapping.json').read_text())
    subset = json.loads((root / 'private/subset.json').read_text())
    if (canonical(subset) != export['subset_hash']
            or json.loads((ROOT / 'data/rq3_human_validation_subset.json').read_text()) != subset):
        raise ValueError('Human subset differs from frozen export')
    uids = subset['example_uids']
    if len(uids) != 30 or len(set(uids)) != 30 or subset['conditions'] != list(CONDITIONS):
        raise ValueError('Expected 30 examples across A-D')
    submissions, checks = {}, {}
    for reviewer in REVIEWERS:
        submissions[reviewer], checks[reviewer] = validate_submission(root, reviewer, mapping, export)
        if set(submissions[reviewer]) != {(uid, c) for uid in uids for c in CONDITIONS}:
            raise ValueError('Human submissions do not match the 30-example cohort')
    data = load_data(args.analysis, args.bundle)
    if data.manifest['manifest_hash'] != subset['source_bundle_manifest_hash']:
        raise ValueError('Human subset and original judge use different bundles')
    draws = np.random.default_rng(plan['bootstrap_seed']).integers(0, len(uids), size=(plan['bootstrap_resamples'], len(uids)))
    result, joined, by_example = build_analysis(data, subset, submissions, draws)
    result.update({'schema_version': 1, 'plan': plan, 'submission_checks': checks,
                   'counts': {'examples': 30, 'outputs': 120, 'human_main_ratings': 240,
                              'practice_excluded': 8, 'knowledge_outputs': 64, 'knowledge_examples': 16},
                   'provenance': {'created_at': datetime.now(timezone.utc).isoformat(),
                      'generation_protocol_hash': data.manifest['generation_protocol_hash'],
                      'evaluation_protocol_hash': data.manifest['evaluation_protocol_hash'],
                      'judge_scores_sha256': sha256(args.analysis / 'records/scores.jsonl'),
                      'subset_hash': export['subset_hash'], 'mapping_sha256': sha256(root / 'private/mapping.json'),
                      'analysis_plan_sha256': sha256(plan_path),
                      'implementation_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                      'python': platform.python_version(),
                      'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib')}}})
    args.output_dir.mkdir(parents=True)
    tables = args.output_dir / 'tables'
    save_json(args.output_dir / 'report.json', result)
    save_json(args.output_dir / 'analysis_plan.json', plan)
    for name, rows in (('agreement', result['agreements']), ('condition_means', result['conditions']),
                       ('paired_effects', result['effects']), ('effect_differences_vs_judge', result['effect_differences']),
                       ('joined_scores', joined), ('per_example_effects', by_example)):
        write_csv(tables / f'{name}.csv', rows)
    disagreements = sorted(joined, key=lambda r: (-abs(r['chatgpt_minus_human_mean_Q']), r['example_uid'], r['condition']))
    write_csv(tables / 'disagreements.csv', disagreements)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/rq3/cache/matplotlib'))
    from .presentation import render
    render(args.output_dir, result, joined)
    # Guard against a reviewer saving changes during this run.
    for reviewer, check in checks.items():
        for name, expected in check['file_sha256'].items():
            if sha256(root / reviewer / name) != expected:
                raise ValueError('Submission changed during analysis; discard this derived run and revalidate')
    save_json(args.output_dir / 'artifact_inventory.json', {
        str(p.relative_to(args.output_dir)): sha256(p) for p in sorted(args.output_dir.rglob('*'))
        if p.is_file() and p.name != 'artifact_inventory.json'})
    print(json.dumps({'status': 'complete', 'output_dir': str(args.output_dir), 'human_ratings': 240,
                      'examples': 30, 'original_ratings_modified': False}, indent=2))


if __name__ == '__main__':
    main()
