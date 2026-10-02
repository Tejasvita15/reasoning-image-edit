"""Render a concise research report and an offline figure index."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

from .common import sha256


def run(data, output, results):
    report = data.report
    contrast = report['contrasts']
    eff = report['efficiency_thinking_on']['conditions']
    b, d = eff['B'], eff['D']
    loss = contrast['thinking_on_acceleration_D_minus_B']
    thinking = contrast['G0_B_minus_A']
    interaction = contrast['interaction_D_minus_C_minus_B_plus_A']
    retention = report['retention']
    speed = d['charged_end_to_end_speedup_mean_paired_ratio']
    main_ties = sum(abs(data.scores[(uid, 'D')]['Q']-data.scores[(uid, 'B')]['Q']) < 1e-9 for uid in data.uids)
    ablation = report['ablations']
    lines = [
        '# RQ3 results and discussion', '',
        'This report uses the frozen 40-example Step1X experiment at 28 denoising steps, CFG 6 and seed 42. '
        'The component-removal comparisons use the predetermined 20-example subset. There are 200 judged quality outputs '
        'and 600 additional timing generations; the sample sizes are 40 and 20 examples, respectively.', '',
        f"Under the recorded manual-judge scores, RegionE with thinking enabled changes mean Q by {loss['mean']:.2f} points "
        f"(95% CI [{loss['ci_low']:.2f}, {loss['ci_high']:.2f}]) and provides {speed:.2f}× mean paired charged-total speedup. "
        f"The frozen 5-point acceptable-loss rule is {'met' if retention['quality_retained_one_sided_rule'] else 'not met'}. "
        f"The unaccelerated thinking contrast is {thinking['mean']:.2f} points "
        f"(95% CI [{thinking['ci_low']:.2f}, {thinking['ci_high']:.2f}]); "
        'this run does not establish a positive thinking benefit. The appropriate conclusion concerns accelerated '
        'thinking-enabled editing under this configuration, not preservation of an independently established reasoning gain.', '',
        '## Main conditions', '',
        '| Condition | Mean Q | Generator seconds | Charged-total seconds | Mean peak allocated VRAM (GiB) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    all_eff = {**report['efficiency_thinking_off']['conditions'], **eff}
    for c in 'ABCD':
        e = all_eff[c]
        lines.append(f"| {c} | {report['conditions'][c]['Q']['mean']:.2f} | {e['mean_diffusion_seconds']:.2f} | "
                     f"{e['mean_charged_total_seconds']:.2f} | {e['mean_peak_vram_bytes']/2**30:.2f} |")
    lines += ['', 'A/B: RegionE off, thinking off/on. C/D: RegionE on, thinking off/on.', '',
              'Each timing value averages within-example medians of three NSCC repetitions. Charged total reconstructs '
              'replay plus one cached thinking/offload pass; it excludes model loading, warmup and artifact writes. '
              'Peak memory is the per-example maximum over timing repeats, then averaged over examples. '
              'It is PyTorch allocated memory, not total board memory.', '',
              '## Paired quality effects', '', '| Contrast | Mean Q difference | 95% example-bootstrap CI |',
              '| --- | ---: | --- |']
    names = [('G0_B_minus_A', 'Thinking without RegionE: B−A'), ('G1_D_minus_C', 'Thinking with RegionE: D−C'),
             ('interaction_D_minus_C_minus_B_plus_A', 'Interaction: (D−C)−(B−A)'),
             ('thinking_on_acceleration_D_minus_B', 'Acceleration with thinking: D−B'),
             ('thinking_off_acceleration_C_minus_A', 'Acceleration without thinking: C−A')]
    for key, label in names:
        value = contrast[key]
        lines.append(f"| {label} | {value['mean']:.2f} | [{value['ci_low']:.2f}, {value['ci_high']:.2f}] |")
    lines += ['', f"The interaction CI [{interaction['ci_low']:.2f}, {interaction['ci_high']:.2f}] lies within ±5 under the frozen rule. "
              'This is a small average interaction in the recorded scores; it does not demonstrate that thinking improves editing. '
              'The retention decision uses the lower endpoint of the existing two-sided 95% interval, as predeclared. '
              'It is not a newly calculated one-sided 95% interval.', '',
              f'{main_ties}/40 D/B pairs have tied aggregate ratings. Inspect individual outcomes: average retention does not '
              'mean every example stays within five points. Confidence intervals resample examples 10,000 times with seed 42; '
              'they do not include generation-seed or evaluator uncertainty. No multiplicity correction was added after observing results.', '',
              '## Component analysis', '', '| Contrast, same 20 examples | Mean Q difference | 95% CI |',
              '| --- | ---: | --- |']
    for key, label in [('full_D_minus_vanilla_B', 'D−B'), ('without_RIKV_E_minus_full_D', 'E−D: remove RIKV'),
                       ('without_AVD_F_minus_full_D', 'F−D: remove AVD')]:
        value = ablation['contrasts'][key]
        lines.append(f"| {label} | {value['mean']:.2f} | [{value['ci_low']:.2f}, {value['ci_high']:.2f}] |")
    e_ab = report['efficiency_ablations']['conditions']
    lines += ['', 'B, D and F have identical component-score vectors across these 20 examples. Their images are not identical. '
              'The zero-width empirical score intervals arise from these ties; they do not establish zero population variability '
              'or exact perceptual equivalence.', '',
              f"Removing AVD increases generator time from {e_ab['D']['mean_diffusion_seconds']:.2f}s (D) to "
              f"{e_ab['F']['mean_diffusion_seconds']:.2f}s (F), with no detected change in the coarse recorded ratings. "
              'All C/D/E examples log 15 AVD-reuse steps and 26 transformer calls versus 56 calls in A/B/F; '
              'the call counts include conditional and unconditional CFG branches and are not scheduler-step counts.', '',
              f"Removing RIKV reduces mean peak allocated memory from {e_ab['D']['mean_peak_vram_bytes']/2**30:.2f} to "
              f"{e_ab['E']['mean_peak_vram_bytes']/2**30:.2f} GiB and charged total from "
              f"{e_ab['D']['mean_charged_total_seconds']:.2f} to {e_ab['E']['mean_charged_total_seconds']:.2f}s. "
              'Its quality interval extends below −5, so the same retention rule would not establish retention for that removal. '
              'The interval also reaches zero; describe the estimated loss and uncertainty rather than claiming a conclusive nonzero effect.', '',
              '## Image differences', '',
              'These compare generated outputs against other generated outputs. They measure deviation, not correctness '
              'against a ground-truth edit. Images are compared as full-resolution RGB without metric resizing.', '',
              '| Comparison | Examples | Mean MAE (0–255) | Mean PSNR (dB) | Mean RGB SSIM |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, row in results['qualitative']['metrics_summary'].items():
        lines.append(f"| {name} | {row['n']} | {row['mae_rgb_0_255']['mean']:.3f} | "
                     f"{row['psnr_db']['mean']:.3f} | {row['ssim_rgb']['mean']:.4f} |")
    lines += ['', 'SSIM uses an 11×11 Gaussian window, sigma 1.5, population covariance and a mean across RGB channels. '
              'See [image-diagnostic methods](qualitative/README.md) and the saved known-answer checks.', '']
    if 'lpips' in results:
        lines += ['LPIPS was computed separately on the same 160 pairs with the calibrated AlexNet metric, version 0.1, '
                  'on the Mac CPU. Inputs are full-resolution RGB scaled to [−1, 1]. Package versions, pretrained-weight '
                  'hashes and per-pair results are saved in the `perceptual/` directory. '
                  '[Metric reference](https://github.com/richzhang/PerceptualSimilarity).', '']
        lines += ['| Comparison | Examples | Mean LPIPS (lower = less deviation) |',
                  '| --- | ---: | ---: |']
        for name, row in results['lpips']['comparisons'].items():
            lines.append(f"| {name} | {row['n']} | {row['mean']:.5f} |")
        lines.append('')
    else:
        lines += ['LPIPS was not computed in this invocation; use the documented `--lpips` option for the planned perceptual metric.', '']
    lines += ['## Case selection and visual observations', '',
              'Seven cases were selected by recorded post-hoc rules: typical score/speed, fast with a small score change, '
              'largest acceleration loss, largest thinking loss and gain, largest RIKV-removal loss, and greatest F/D pixel '
              'variation among tied ratings. All examples remain in the quantitative estimates. See '
              '[the case catalogue](qualitative/candidates.md) for exact rules, scores, generated rewrites and attributed judge rationales.', '']
    observations_path = Path(__file__).resolve().parents[2] / 'reports/rq3_case_observations.json'
    if observations_path.exists():
        observations = json.loads(observations_path.read_text())
        if (observations['generation_protocol_hash'] == data.manifest['generation_protocol_hash']
                and observations['evaluation_protocol_hash'] == data.manifest['evaluation_protocol_hash']):
            for note in observations['observations']:
                uid = note['example_uid']
                if any(data.quality[(uid, c)]['output_sha256'] != digest for c, digest in note['output_sha256'].items()):
                    raise ValueError('Visual observation refers to different image bytes')
                lines += [f"**{uid}:** {note['observation']}", '']
            lines += ['These notes are assistant visual inspection of the displayed source/output grids, not an independent '
                      'human review or replacement scoring pass. The original ratings are unchanged. The charger example '
                      'in particular makes the validity of the manual evaluator an important limitation, even where repeatability is high.', '']
            shutil.copy2(observations_path, output / 'visual_observations.json')
            provenance_path = output / 'run_manifest.json'
            provenance = json.loads(provenance_path.read_text())
            provenance['visual_observations_sha256'] = sha256(observations_path)
            provenance['visual_observations_source'] = 'reports/rq3_case_observations.json'
            provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    lines += ['## Limits and reporting scope', '',
              '- One generation seed and 40 examples; the component subset has 20 examples. Timing repeats do not test seed robustness.',
              '- One manual ChatGPT judging pass, an operator-recorded UI model label and coarse 1–5 component ratings. '
              'This adaptation is not the published KRIS aggregate/API evaluation.',
              '- Calibration agreement was exact on 20 paired re-scores, but its session metadata was corrected retrospectively '
              'under operator attestation, without independent transcript verification. Repeatability does not establish correctness.',
              '- No independent human–human/human–judge agreement study is included. Visual inconsistencies above warrant '
              'separate blinded review before broad claims about edit correctness.',
              '- Region overlays show saved token-selection snapshots at zero-based step 5, not attention maps or a complete spatial trajectory.',
              '- Category and case analyses are exploratory. The data do not support transferring these conclusions to RQ1, '
              'other step counts, checkpoints or deployment settings.', '',
              '## Reproducibility', '',
              f"Generation protocol: `{data.manifest['generation_protocol_hash']}`.\n\n"
              f"Evaluation protocol: `{data.manifest['evaluation_protocol_hash']}`.", '',
              'Original records are preserved under `outputs/rq3/evidence/`. The local image lookup is separate from their '
              'unchanged NSCC paths. `run_manifest.json` records input and postprocessor hashes and local package versions. '
              'The canonical main and component effect intervals reproduce on the Mac.', '',
              '[Browse all figures](index.html) · [Original canonical report](canonical_report.json) · '
              '[Derived tables](tables/) · [Qualitative cases](qualitative/candidates.md)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    figures = [(path, 'quantitative') for path in sorted((output / 'figures').glob('*.png'))]
    figures += [(path, 'qualitative') for path in sorted((output / 'qualitative').glob('*.png'))]
    cards = []
    for path, category in figures:
        relative = path.relative_to(output).as_posix()
        title = path.stem.replace('__', ' · ').replace('_', ' ')
        cards.append(f'<article class="asset" data-kind="{category}"><h2>{html.escape(title)}</h2>'
                     f'<a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}" alt="{html.escape(title)}"></a>'
                     f'<p>{category} · <a href="{html.escape(relative)}">Open full image</a></p></article>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RQ3 analysis</title><style>body{{font:16px system-ui,sans-serif;max-width:1240px;margin:36px auto;padding:0 24px;background:#f5f7fa;color:#172536}}h1{{font-size:32px}}h2{{font-size:18px}}a{{color:#145b9b}}.asset{{background:white;border:1px solid #dce3eb;border-radius:10px;padding:20px;margin:22px 0}}img{{width:100%;height:auto}}.summary{{padding:18px;background:#e7eff7;line-height:1.65}}select{{padding:8px;margin:18px 0}}[hidden]{{display:none}}</style>
<h1>RQ3 analysis</h1><p>40 main examples · 20 component examples · 200 judgments · 3 timing repetitions · seed 42</p>
<div class="summary">D−B: <strong>{loss['mean']:.2f} Q points</strong>, 95% CI [{loss['ci_low']:.2f}, {loss['ci_high']:.2f}].
Charged-total speedup: <strong>{speed:.2f}×</strong>. Thinking benefit was not established at 28 steps.
Mean peak allocated VRAM: B {b['mean_peak_vram_bytes']/2**30:.2f} GiB → D {d['mean_peak_vram_bytes']/2**30:.2f} GiB.</div>
<p><a href="RESULTS.md">Results and discussion</a> · <a href="tables/">CSV tables</a> · <a href="qualitative/candidates.md">Case catalogue</a> · <a href="run_manifest.json">Provenance</a></p>
<label>Show <select id="filter"><option value="all">All figures</option><option value="quantitative">Quantitative charts</option><option value="qualitative">Qualitative comparisons</option></select></label>
{''.join(cards)}<script>document.getElementById('filter').onchange=function(){{document.querySelectorAll('.asset').forEach(el=>el.hidden=this.value!=='all'&&el.dataset.kind!==this.value)}};</script></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
