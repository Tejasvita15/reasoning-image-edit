#!/usr/bin/env python3
"""Merge staged KRIS-Bench judge metrics with successful RQ1 K-dev records.

Read-only inputs; writes new JSONL files. Standard library only.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

VARIANTS = ('base', 'thinking', 'correction')
REQUIRED_SCORES = ('instruction_score', 'consistency_score', 'quality_score')
EXTRA_SCORES = ('knowledge_score',)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-root', type=Path, default=Path('analysis/rq1/kris_judge/full/results'))
    parser.add_argument('--inputs-dir', type=Path, default=Path('analysis/rq1/k_dev/inputs'))
    parser.add_argument('--output-dir', type=Path, default=Path('analysis/rq1/k_dev/judged_inputs'))
    parser.add_argument('--expected-count', type=int, default=90)
    return parser.parse_args()


def valid_score(value):
    return type(value) in (int, float) and 1 <= value <= 5


def load_metrics(variant, root):
    folder = root / variant
    if not folder.is_dir():
        raise RuntimeError(f'Missing judge results folder: {folder}')
    paths = sorted(folder.glob('*/metrics.json'))
    if not paths:
        raise RuntimeError(f'No metrics files found in {folder}')
    by_uid = {}
    for path in paths:
        category = path.parent.name
        metrics = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(metrics, dict):
            raise RuntimeError(f'Expected JSON object in {path}')
        for image_id, result in metrics.items():
            uid = f'{category}__{image_id}'
            if uid in by_uid:
                raise RuntimeError(f'Duplicate score for {variant}/{uid}')
            if not isinstance(result, dict):
                raise RuntimeError(f'Invalid score object for {variant}/{uid}')
            for key in REQUIRED_SCORES:
                if not valid_score(result.get(key)):
                    raise RuntimeError(f'Missing/invalid {key} for {variant}/{uid}: {result.get(key)!r}')
            for key in EXTRA_SCORES:
                if result.get(key) is not None and not valid_score(result[key]):
                    raise RuntimeError(f'Invalid {key} for {variant}/{uid}: {result[key]!r}')
            by_uid[uid] = result
    return by_uid, len(paths)


def load_successful_rows(path):
    rows = {}
    for number, raw_line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if row.get('status') != 'success':
            continue
        uid = row['example_uid']
        if uid in rows:
            raise RuntimeError(f'Duplicate successful row in {path} at line {number}: {uid}')
        rows[uid] = row
    return rows


def main():
    args = get_args()
    staged = {}
    uid_sets = []
    for variant in VARIANTS:
        source = args.inputs_dir / f'rq1_{variant}_k_dev_seed42.jsonl'
        if not source.is_file():
            raise RuntimeError(f'Missing normalized input: {source}')
        records = load_successful_rows(source)
        metrics, n_categories = load_metrics(variant, args.scores_root)
        if len(records) != args.expected_count:
            raise RuntimeError(f'{variant}: expected {args.expected_count} successful inputs; found {len(records)}')
        if len(metrics) != args.expected_count:
            raise RuntimeError(f'{variant}: expected {args.expected_count} scored examples; found {len(metrics)}')
        missing = set(records) - set(metrics)
        extra = set(metrics) - set(records)
        if missing or extra:
            raise RuntimeError(f'{variant}: score-ID mismatch! Missing={sorted(missing)[:10]}; extra={sorted(extra)[:10]}')
        output_rows = []
        for uid in sorted(records):
            row = dict(records[uid])
            result = dict(metrics[uid])
            if result.get('instruction') and ' '.join(str(result['instruction']).split()) != ' '.join(str(row['instruction']).split()):
                raise RuntimeError(f'{variant}/{uid}: judge instruction differs from generation instruction')
            row['judge_result'] = result
            row['judge_result']['judge_model'] = 'gpt-4o'
            output_rows.append(row)
        staged[variant] = (output_rows, n_categories)
        uid_sets.append(set(records))
    if not all(s == uid_sets[0] for s in uid_sets[1:]):
        raise RuntimeError('Variants contain different successful example IDs; cannot pair reliably')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        rows, n_categories = staged[variant]
        destination = args.output_dir / f'rq1_{variant}_k_dev_seed42.jsonl'
        with destination.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        avg = sum(r['judge_result']['instruction_score'] for r in rows) / len(rows)
        with_knowledge = sum(r['judge_result'].get('knowledge_score') is not None for r in rows)
        print(f'{variant:10s}: {len(rows):3d} judged; {n_categories:2d} categories; avg instruction={avg:.3f}/5; knowledge scored={with_knowledge}; wrote {destination}')
    print('PASS: 270 matched scores; all 90 example IDs paired across variants; original files unchanged')


if __name__ == '__main__':
    main()
