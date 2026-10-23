#!/usr/bin/env python3
"""Build a blinded, one-handoff RQ1 Astra UI judging package.

Run from the reasoning-image-edit repository root:

    python -m scripts.export_rq1_combined_astra_judging

The existing RQ1 exporter supplies the official rubric and response schema.
No model inference, API judging, or changes to existing image files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.dataset import get_decoded_source_images, load_kris_bench, prepare_example
from scripts.export_rq1_full_astra_judging import (
    extract_rubrics, get_within_root, load_json, make_prompt,
    response_template, sha256_file, verify_png,
)

ROOT = Path.cwd()
DEFAULT_UNIREASON = Path('/home/users/ntu/tejasvit/scratch/UniReason')
DEFAULT_OUTPUT = Path('outputs/rq1/judge/rq1_astra_combined_full')
OLD_INPUT = Path('outputs/rq1/judge/ksmall_multiseed')
CALIBRATION = Path('outputs/rq1/judge/chatgpt_calibration/private/manifest.json')
SEEDS = (42, 1234, 7)
MODELS = ('step1x', 'unireason')
CONDITIONS = ('base', 'thinking', 'correction')


def json_write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unireason-root', type=Path, default=DEFAULT_UNIREASON)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--order-seed', type=int, default=20261023)
    args = ap.parse_args()

    splits = load_json(Path('data/splits.json'))
    ktest, ksmall = splits['k_test'], splits['k_small']
    assert len(ktest) == len(set(ktest)) == 300
    assert len(ksmall) == len(set(ksmall)) == 40
    assert set(ksmall) <= set(ktest), 'K-small must be a subset of K-test'
    small_set = set(ksmall)

    old_manifest = load_json(OLD_INPUT / 'manifest.json')
    small_info = {e['example_uid']: e for e in old_manifest['examples']}
    assert set(small_info) == small_set, 'Existing K-small package does not match frozen splits'

    calib = load_json(CALIBRATION)
    knowledge_categories = set(calib['knowledge_categories_from_official_evaluator'])
    assert len(knowledge_categories) == 10
    repeat_anchors = sorted({c['example_uid'] for c in calib['cases']})
    assert len(repeat_anchors) == 10 and set(repeat_anchors) <= small_set

    uni_root = args.unireason_root.resolve()
    uni_code = uni_root / 'UniReason'
    uni_metadata_path = uni_code / 'eval/gen/kris/rq1_k_test.json'
    uni_metadata = load_json(uni_metadata_path)
    assert len(uni_metadata) == 300
    uni_by_uid = {x['id']: x for x in uni_metadata}
    assert set(uni_by_uid) == set(ktest), 'UniReason and Step1X K-test identities differ'

    # Resolve every final output with exact UID filename, never _candidate0.
    image_index = {}
    for uid in ktest:
        category = uni_by_uid[uid]['type']
        assert uid.startswith(category + '__'), f'Unexpected category in {uid}'
        for model in MODELS:
            for condition in ('base', 'thinking'):
                if model == 'step1x':
                    p = Path('outputs/rq1') / condition / 'k_test/seed_42' / f'{uid}.png'
                else:
                    p = uni_root / 'outputs/rq1' / condition / 'k_test/seed_42' / category / f'{uid}.png'
                image_index[(uid, model, condition, 42)] = p

        if uid not in small_set:
            continue

        for seed in (1234, 7):
            for model in MODELS:
                for condition in ('base', 'thinking'):
                    if model == 'step1x':
                        if condition == 'thinking':
                            # Use the same vetted K-small finals as the previous Step1X package.
                            rel = small_info[uid]['outputs']['thinking'][str(seed)]
                            p = get_within_root(OLD_INPUT, rel)
                        else:
                            p = Path('outputs/rq1/base/k_small') / f'seed_{seed}' / f'{uid}.png'
                    else:
                        p = uni_root / 'outputs/rq1' / condition / 'k_small' / f'seed_{seed}' / category / f'{uid}.png'
                    image_index[(uid, model, condition, seed)] = p

        for seed in SEEDS:
            rel = small_info[uid]['outputs']['correction'][str(seed)]
            image_index[(uid, 'step1x', 'correction', seed)] = get_within_root(OLD_INPUT, rel)
            image_index[(uid, 'unireason', 'correction', seed)] = (
                uni_root / 'outputs/rq1/correction/k_small' / f'seed_{seed}_refine'
                / category / f'{uid}.png'
            )

    assert len(image_index) == 1760, f'Unexpected edited case count: {len(image_index)}'
    print('Checking all 1,760 exact final output paths...', flush=True)
    for p in image_index.values():
        verify_png(p)
    print('PASS: All 1,760 unique-condition PNGs are valid', flush=True)

    # Check that the category and instruction agree with the frozen Step1X data.
    data_by_uid = {}
    for row in load_kris_bench():
        e = prepare_example(row)
        uid = e['example_uid']
        if uid in uni_by_uid:
            assert uid not in data_by_uid, f'Duplicate KRIS ID: {uid}'
            uni = uni_by_uid[uid]
            assert e['category'] == uni['type'], f'Category mismatch: {uid}'
            assert e['instruction'] == uni['ins_en'], f'Instruction mismatch: {uid}'
            data_by_uid[uid] = e
        if len(data_by_uid) == 300:
            break
    assert set(data_by_uid) == set(ktest), 'KRIS dataset missing frozen K-test examples'
    print('PASS: All 300 original instructions agree with UniReason metadata', flush=True)

    root = args.out
    build = root.with_name(root.name + '.building')
    if root.exists() or build.exists():
        raise FileExistsError(f'Output already exists: {root} or {build}; refusing to overwrite')

    rubrics = extract_rubrics()
    rng = random.Random(args.order_seed)
    private_salt = secrets.token_hex(24)
    used_blinds = set()
    cases = []
    case_lookup = {}
    upload_sha = {}
    all_batches = []

    def make_blind(uid, model, condition, seed, phase):
        digest = hashlib.sha256(
            f'{private_salt}|{uid}|{model}|{condition}|{seed}|{phase}'.encode()
        ).hexdigest()[:20]
        blind = 'case_' + digest
        assert blind not in used_blinds
        used_blinds.add(blind)
        return blind

    def copy_record(src: Path, dst: Path):
        shutil.copyfile(src, dst)
        upload_sha[str(dst.relative_to(build))] = sha256_file(dst)

    def write_batch(batch_id, phase, uid, specs, sources):
        example = data_by_uid[uid]
        dest = build / 'upload' / phase / batch_id
        dest.mkdir(parents=True)
        copy_record(sources[uid], dest / 'source.png')
        items = []
        for model, condition, seed, repeat_of in specs:
            blind = make_blind(uid, model, condition, seed, phase)
            image = image_index[(uid, model, condition, seed)]
            copy_record(image, dest / f'{blind}_edited.png')
            items.append({'blind_id': blind})
            case = {
                'blind_id': blind,
                'example_uid': uid,
                'model': model,
                'condition': condition,
                'seed': seed,
                'phase': phase,
                'batch_id': batch_id,
                'repeat_of': repeat_of,
                'category': example['category'],
                'knowledge_type': example['knowledge_type'],
                'knowledge_applicable': example['category'] in knowledge_categories,
                'edited_original': str(image),
                'edited_sha256': upload_sha[str((dest / f'{blind}_edited.png').relative_to(build))],
            }
            cases.append(case)
            if phase == 'main':
                case_lookup[(uid, model, condition, seed)] = blind
        prompt = make_prompt(batch_id, example, items, rubrics, knowledge_categories)
        prompt = prompt.replace(
            '# RQ1 blinded Step1X image-editing evaluation',
            '# RQ1 blinded image-editing evaluation',
            1,
        )
        (dest / 'prompt.md').write_text(prompt, encoding='utf-8')
        json_write(dest / 'response_template.json', response_template(batch_id, items))
        for suffix in ('prompt.md', 'response_template.json'):
            upload_sha[str((dest / suffix).relative_to(build))] = sha256_file(dest / suffix)
        all_batches.append({'batch_id': batch_id, 'phase': phase, 'example_uid': uid, 'items': len(items)})

    # On error, retain the .building directory for inspection; do not touch inputs.
    build.mkdir(parents=True)
    (build / 'private').mkdir()
    (build / 'responses').mkdir()
    source_root = build / '_temporary_sources'
    source_root.mkdir()
    sources = {}
    for uid in ktest:
        e = data_by_uid[uid]
        images = get_decoded_source_images(e)
        assert len(images) == 1, f'Expected one source image for {uid}'
        dst = source_root / f'{uid}.png'
        images[0].convert('RGB').save(dst)
        verify_png(dst)
        sources[uid] = dst
    print('PASS: All 300 original source images exported', flush=True)

    shuffled = list(ktest)
    rng.shuffle(shuffled)
    main_number = 0
    for uid in shuffled:
        specs = [(m, c, 42, None) for m in MODELS for c in ('base', 'thinking')]
        if uid in small_set:
            specs += [
                (m, c, seed, None)
                for m in MODELS for c in ('base', 'thinking')
                for seed in (1234, 7)
            ]
            specs += [
                (m, 'correction', seed, None)
                for m in MODELS for seed in SEEDS
            ]
            assert len(specs) == 18
        else:
            assert len(specs) == 4
        rng.shuffle(specs)
        for offset in range(0, len(specs), 6):
            main_number += 1
            write_batch(f'main_b{main_number:03d}', 'main', uid, specs[offset:offset+6], sources)

    assert main_number == 380
    assert len(cases) == 1760
    repeat_number = 0
    rng.shuffle(repeat_anchors)
    for uid in repeat_anchors:
        repeat_number += 1
        choices = [('step1x', 'thinking', 42), ('unireason', 'correction', 42)]
        rng.shuffle(choices)
        specs = [(*key, case_lookup[(uid, *key)]) for key in choices]
        write_batch(f'repeat_b{repeat_number:03d}', 'repeat', uid, specs, sources)

    assert repeat_number == 10 and len(cases) == 1780
    counts = Counter(c['model'] for c in cases if c['phase'] == 'main')
    assert dict(counts) == {'step1x': 880, 'unireason': 880}
    assert sum(x['items'] for x in all_batches if x['phase'] == 'main') == 1760
    assert sum(x['items'] for x in all_batches if x['phase'] == 'repeat') == 20
    assert len({x['blind_id'] for x in cases}) == 1780
    shutil.rmtree(source_root)

    manifest = {
        'research_question': 'RQ1 Step1X versus UniReason',
        'protocol': 'Frozen K-test/K-small, KRIS rubrics, blinded Astra UI judging',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'order_seed': args.order_seed,
        'knowledge_categories_from_official_evaluator': sorted(knowledge_categories),
        'counts': {
            'unique_examples': 300, 'k_small_examples': 40,
            'step1x_unique': 880, 'unireason_unique': 880,
            'main_ratings': 1760, 'repeat_ratings': 20,
            'main_batches': 380, 'repeat_batches': 10,
        },
        'cases': cases,
        'batches': all_batches,
        'upload_sha256': upload_sha,
    }
    json_write(build / 'private/manifest.json', manifest)
    json_write(build / 'session_log_template.json', {
        'judge_model_as_displayed': '', 'UI_mode': '',
        'judged_on_date': '', 'fresh_chats_used': '', 'notes': '',
    })
    (build / 'README_FOR_JUDGE.md').write_text('''# RQ1 blinded image-editing evaluation — manual Astra UI handoff

**380 main judging batches and 10 repeat batches; 1,780 individual image ratings.**
This is a manual ChatGPT UI adaptation of the KRIS-Bench rubric, not the original GPT-4o API scorer.

For EACH directory under `upload/main/`, open a NEW chat using the recorded Astra Pro mode. Attach `source.png` and every `*_edited.png` in that directory (5 or 7 images total). Paste the full `prompt.md`, including the JSON output template. Save the verbatim JSON as `responses/<batch_id>.json`.

For EACH directory under `upload/repeat/`, also use a NEW chat, independently of any earlier ratings. Attach the three PNG images and paste `prompt.md`. Save the JSON using its `repeat_bNNN` batch ID.

Judge every edited image independently using absolute 1–5 rubric scales. Do not rank edits. Mark unreadable items as unreadable with null scores rather than guessing. If a knowledge score is inapplicable, use null.

Record the exact displayed model/mode and date in `session_log_template.json`. Return the entire `responses/` folder and that session log together, in one handoff.

**Blinding:** Share only this README, `upload/` and the session log template with the judge. NEVER provide `private/manifest.json`, original labeled output directories, earlier scores or previous responses to a judging chat. In each chat, upload only the files of ONE batch; do not upload the full archive into the judge's chat.

Grouping multiple edits by source reduces uploads but can introduce context effects. Ratings are a UI adaptation and should be reported as such.
''', encoding='utf-8')
    root.parent.mkdir(parents=True, exist_ok=True)
    build.rename(root)
    print('PASS: Combined Astra package built', flush=True)
    print('Location:', root.resolve(), flush=True)
    print('Main ratings: 1760 | Repeat ratings: 20 | Total: 1780', flush=True)
    print('Main batches: 380 | Repeat batches: 10 | Total: 390', flush=True)
    print('PRIVATE mapping:', (root / 'private/manifest.json').resolve(), flush=True)
    print('JUDGE upload tree:', (root / 'upload').resolve(), flush=True)


if __name__ == '__main__':
    main()
