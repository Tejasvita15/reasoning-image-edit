#!/usr/bin/env python3
"""Fail if an official KRIS judge script silently produced incomplete pilot results."""
import json
from pathlib import Path

workspace = Path('analysis/rq1/kris_judge/pilot')
manifest = json.loads((workspace / 'staging_manifest.json').read_text())
assert manifest['subset'] == 'pilot' and manifest['n_examples'] == 3, manifest
families = {
    'knowledge': {'abstract_reasoning', 'mathematics', 'practical_knowledge', 'medicine',
                  'rule-based_reasoning', 'biology', 'geography', 'chemistry', 'humanities', 'physics'},
}
failures = []
for entry in manifest['entries']:
    cat, imgid = entry['category'], entry['id']
    for variant in ('base', 'thinking', 'correction'):
        path = workspace / 'results' / variant / cat / 'metrics.json'
        if not path.exists():
            failures.append(f'Missing judge metrics: {variant}/{cat}/{imgid}')
            continue
        d = json.loads(path.read_text())
        rec = d.get(imgid)
        if not rec:
            failures.append(f'Missing judge record: {variant}/{cat}/{imgid}')
            continue
        required = ['consistency_score', 'instruction_score', 'quality_score']
        if cat in families['knowledge']:
            required.append('knowledge_score')
        for k in required:
            val = rec.get(k)
            if type(val) is not int or not 1 <= val <= 5:
                failures.append(f'Invalid {k}={val!r}: {variant}/{cat}/{imgid}')
        print(variant, entry['example_uid'],
              'instruction:', rec.get('instruction_score'),
              'knowledge:', rec.get('knowledge_score'),
              'consistency:', rec.get('consistency_score'),
              'quality:', rec.get('quality_score'))
if failures:
    print('\nERRORS:', *failures, sep='\n  ')
    raise SystemExit(1)
print('\nPASS: 9 pilot images have complete official-style scores.')
