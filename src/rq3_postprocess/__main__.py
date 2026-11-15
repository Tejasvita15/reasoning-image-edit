"""Create supplementary Mac charts and discussion artifacts from frozen evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from .common import load_data, sha256, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True, help='Verified extracted rq3_analysis directory')
    parser.add_argument('--bundle', required=True, help='Original verified chatgpt_main bundle')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--lpips', action='store_true', help='Also compute CPU AlexNet LPIPS on saved outputs')
    parser.add_argument('--dependency-dir', help='Optional repository-local Python packages for LPIPS')
    args = parser.parse_args()
    data = load_data(args.analysis, args.bundle)
    output = Path(args.output_dir).resolve()
    for source in (data.root, data.bundle_root):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Keep derived outputs separate from the original evidence directories')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    from .quantitative import run as quantitative
    from .qualitative import run as qualitative
    result = {'quantitative': quantitative(data, output), 'qualitative': qualitative(data, output)}
    if args.lpips:
        if args.dependency_dir:
            sys.path.insert(0, str(Path(args.dependency_dir).resolve()))
        os.environ.setdefault('TORCH_HOME', str(Path('outputs/rq3/cache/torch').resolve()))
        from .perceptual import run as perceptual
        result['lpips'] = perceptual(data, output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    shutil.copy2(data.root / 'analysis/report.json', output / 'canonical_report.json')
    mapping = []
    for (uid, condition), paths in sorted(data.paths.items()):
        row = data.quality[(uid, condition)]
        mapping.append({'example_uid': uid, 'condition': condition,
                        'original_source_path': row['source_path'], 'original_output_path': row['output_path'],
                        'local_source_path': str(paths['source']), 'local_output_path': str(paths['edited']),
                        'source_sha256': row['source_sha256'], 'output_sha256': row['output_sha256']})
    write_csv(output / 'tables/image_path_mapping.csv', mapping)
    package = Path(__file__).parent
    provenance = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'generation_protocol_hash': data.manifest['generation_protocol_hash'],
        'evaluation_protocol_hash': data.manifest['evaluation_protocol_hash'],
        'bundle_manifest_hash': data.manifest['manifest_hash'],
        'analysis_transfer_sha256': sha256(data.root / 'TRANSFER.json'),
        'generation_records_sha256': sha256(data.root / 'records/generations.jsonl'),
        'scores_sha256': sha256(data.root / 'records/scores.jsonl'),
        'postprocessor_sha256': {p.name: sha256(p) for p in sorted(package.glob('*.py'))},
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib', 'pillow')},
        'record_policy': 'Original records/manifests/NSCC paths unchanged; separate local image-path lookup.',
        'resampling_policy': 'Paired example-level bootstrap; 40 main/20 ablation; seed42; 10000 draws. Timing repeats are not independent examples.',
        'image_metric_policy': 'Supplementary saved-image diagnostics, not substitutes for editing quality scores.',
    }
    (output / 'run_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    from .discussion import run as discussion
    discussion(data, output, result)
    print(json.dumps({'status': 'complete', 'output': str(output),
                      'main_examples': len(data.uids), 'ablation_examples': len(data.subset),
                      'quality_outputs': len(data.scores), 'generation_records': len(data.generation_rows)}, indent=2))


if __name__ == '__main__':
    main()
