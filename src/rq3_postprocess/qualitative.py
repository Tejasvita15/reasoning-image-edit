"""Post-hoc example selection and exact-geometry RGB image diagnostics.

This module never changes frozen generation records or judge scores. Figure
thumbnails are resized for display only; all metrics use original decoded pixels.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import convolve1d

from .common import write_csv


SSIM_REFERENCE = "https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity"
SSIM_POLICY = {
    "implementation": "local_numpy_scipy",
    "window": 11,
    "sigma": 1.5,
    "K1": 0.01,
    "K2": 0.03,
    "data_range": 255,
    "covariance": "population",
    "channels": "arithmetic mean of separate R, G, B SSIM",
    "boundary": "exclude five-pixel border; only complete 11x11 windows",
    "reference": SSIM_REFERENCE,
}


def _get(data, key):
    return data[key] if isinstance(data, dict) else getattr(data, key)


def ssim_rgb(first: np.ndarray, second: np.ndarray) -> float:
    """Gaussian-window SSIM; population covariance, complete windows, RGB mean.

    Per-window SSIM = ((2*mu_x*mu_y+C1)*(2*cov_xy+C2)) /
                      ((mu_x**2+mu_y**2+C1)*(var_x+var_y+C2)).
    C1=(0.01*255)**2 and C2=(0.03*255)**2. No metric resizing.
    """
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("SSIM requires identical H x W x 3 RGB geometry")
    if min(first.shape[:2]) < 11:
        raise ValueError("SSIM requires both image dimensions >= 11 pixels")
    x = first.astype(np.float64)
    y = second.astype(np.float64)
    coordinates = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-coordinates**2 / (2 * 1.5**2))
    kernel /= kernel.sum()

    def smooth(array):
        return convolve1d(convolve1d(array, kernel, axis=0, mode="reflect"),
                          kernel, axis=1, mode="reflect")

    ux, uy = smooth(x), smooth(y)
    vx, vy = smooth(x*x) - ux*ux, smooth(y*y) - uy*uy
    covariance = smooth(x*y) - ux*uy
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    similarity = ((2*ux*uy+c1) * (2*covariance+c2)
                  / ((ux*ux+uy*uy+c1) * (vx+vy+c2)))
    return float(np.mean(similarity[5:-5, 5:-5, :], dtype=np.float64))


def pixel_metrics(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape:
        raise ValueError(f"Metric geometry mismatch: {first.shape} != {second.shape}")
    difference = first.astype(np.float64) - second.astype(np.float64)
    mse = float(np.mean(difference**2))
    return {
        "mae_rgb_0_255": float(np.mean(np.abs(difference))),
        "rmse_rgb_0_255": math.sqrt(mse),
        "psnr_db": 10*math.log10(255**2/mse) if mse else None,
        "identical_pixels": mse == 0,
        "psnr_infinite": mse == 0,
        "ssim_rgb": ssim_rgb(first, second),
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
    }


def validate_metric_implementation() -> dict:
    """Known-answer, symmetry and independent direct-window checks."""
    random = np.random.default_rng(6132)
    first = random.integers(0, 256, size=(17, 19, 3), dtype=np.uint8)
    second = random.integers(0, 256, size=first.shape, dtype=np.uint8)
    identical = pixel_metrics(first, first)
    forward, reverse = ssim_rgb(first, second), ssim_rgb(second, first)
    constant_low = np.full(first.shape, 30, dtype=np.uint8)
    constant_high = np.full(first.shape, 90, dtype=np.uint8)
    expected_constant = (2*30*90+(0.01*255)**2)/(30**2+90**2+(0.01*255)**2)
    constant = ssim_rgb(constant_low, constant_high)
    axis = np.arange(-5, 6, dtype=float)
    kernel = np.exp(-(axis[:, None]**2+axis[None, :]**2)/(2*1.5**2))
    kernel /= kernel.sum()
    direct = []
    for row in range(5, first.shape[0]-5):
        for column in range(5, first.shape[1]-5):
            for channel in range(3):
                x = first[row-5:row+6, column-5:column+6, channel].astype(float)
                y = second[row-5:row+6, column-5:column+6, channel].astype(float)
                ux, uy = float(np.sum(kernel*x)), float(np.sum(kernel*y))
                vx, vy = float(np.sum(kernel*(x-ux)**2)), float(np.sum(kernel*(y-uy)**2))
                cov = float(np.sum(kernel*(x-ux)*(y-uy)))
                c1, c2 = (0.01*255)**2, (0.03*255)**2
                direct.append(((2*ux*uy+c1)*(2*cov+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2)))
    errors = {
        "identical_ssim_error": abs(identical["ssim_rgb"]-1),
        "symmetry_error": abs(forward-reverse),
        "constant_known_answer_error": abs(constant-expected_constant),
        "direct_window_error": abs(forward-float(np.mean(direct))),
    }
    passed = identical["mae_rgb_0_255"] == 0 and identical["psnr_infinite"] and max(errors.values()) < 1e-11
    if not passed:
        raise ValueError(f"Image metric implementation checks failed: {errors}")
    return {"passed": True, **errors, "ssim_policy": SSIM_POLICY}


def _rgb(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB")).copy()


def _font(size):
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(text, font, pixels):
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if line and font.getlength(trial) > pixels:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _strip(panels, title, captions, destination):
    """Shared-size display canvas; metric data is never passed through resize."""
    panel_width, gap, padding = 440, 12, 18
    font, heading = _font(20), _font(24)
    image_height = min(560, max(round(panel_width*im.height/im.width) for im, _ in panels))
    width = 2*padding + len(panels)*panel_width + (len(panels)-1)*gap
    title_lines = _wrap(title, heading, width-2*padding)
    top = padding+len(title_lines)*32+12
    label_lines = [_wrap(label, font, panel_width) for _, label in panels]
    labels_height = max(len(lines) for lines in label_lines)*26+10
    bottom_lines = []
    for caption in captions:
        bottom_lines.extend(_wrap(caption, font, width-2*padding))
    height = top+labels_height+image_height+22+len(bottom_lines)*27+padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(title_lines):
        draw.text((padding, padding+index*32), line, font=heading, fill="#152b45")
    for index, ((original, _), labels) in enumerate(zip(panels, label_lines)):
        left = padding+index*(panel_width+gap)
        for line_index, line in enumerate(labels):
            draw.text((left, top+line_index*26), line, font=font, fill="#1b1b1b")
        displayed = original.copy()
        displayed.thumbnail((panel_width, image_height), Image.Resampling.LANCZOS)
        x = left+(panel_width-displayed.width)//2
        y = top+labels_height+(image_height-displayed.height)//2
        canvas.paste(displayed, (x, y))
        draw.rectangle((left, top+labels_height, left+panel_width-1,
                        top+labels_height+image_height-1), outline="#dddddd")
    y = top+labels_height+image_height+22
    for line in bottom_lines:
        draw.text((padding, y), line, font=font, fill="#333333")
        y += 27
    canvas.save(destination, dpi=(180, 180))


def _selection_overlay(source, selection):
    height, width = int(selection["grid_height"]), int(selection["grid_width"])
    indices = selection["selected_indices"]
    if any(not isinstance(index, int) or index < 0 or index >= height*width for index in indices):
        raise ValueError("Invalid saved selected-region indices")
    mask = np.zeros(height*width, dtype=np.uint8)
    mask[indices] = 140
    alpha = Image.fromarray(mask.reshape(height, width)).resize(source.size, Image.Resampling.NEAREST)
    return Image.composite(Image.new("RGB", source.size, "red"), source, alpha)


def select_candidates(data, diagnostics):
    quality, scores, timing = (_get(data, key) for key in ("quality", "scores", "timing"))
    uids, subset = _get(data, "uids"), _get(data, "subset")
    q = lambda uid, condition: float(scores[(uid, condition)]["Q"])
    speed = lambda uid: timing[(uid, "B")]["charged_total_seconds"]/timing[(uid, "D")]["charged_total_seconds"]
    delta = lambda uid, first, second: q(uid, first)-q(uid, second)
    median_delta = float(np.median([delta(uid, "D", "B") for uid in uids]))
    median_speed = float(np.median([speed(uid) for uid in uids]))
    measures = {(row["example_uid"], row["comparison"]): row for row in diagnostics}
    requests = [
        ("typical", "Closest to median D-B, then closest to median charged speedup B/D", uids,
         lambda uid: (abs(delta(uid, "D", "B")-median_delta), abs(speed(uid)-median_speed), uid)),
        ("fast_with_small_score_change", "Highest charged speedup B/D among examples with |D-B| <= 5", [uid for uid in uids if abs(delta(uid, "D", "B")) <= 5],
         lambda uid: (-speed(uid), uid)),
        ("lowest_acceleration_effect", "Lowest D-B quality difference", uids,
         lambda uid: (delta(uid, "D", "B"), uid)),
        ("lowest_thinking_effect", "Lowest B-A quality difference", uids,
         lambda uid: (delta(uid, "B", "A"), uid)),
        ("highest_thinking_effect", "Highest B-A quality difference", uids,
         lambda uid: (-delta(uid, "B", "A"), uid)),
        ("lowest_no_rikv_effect", "Lowest E-D quality difference on the fixed 20-example subset", subset,
         lambda uid: (delta(uid, "E", "D"), uid)),
        ("no_avd_pixel_variation", "Largest F-vs-D RGB MAE on the fixed subset; descriptive image variation, not inferred quality loss", subset,
         lambda uid: (-measures[(uid, "F-D")]["mae_rgb_0_255"], uid)),
    ]
    used, candidates = set(), []
    for role, rule, pool, key in requests:
        ranked = sorted(pool, key=key)
        if not ranked:
            candidates.append({"role": role, "selection_rule": rule, "status": "no_eligible_example"})
            continue
        uid = next((candidate for candidate in ranked if candidate not in used), ranked[0])
        used.add(uid)
        conditions = [condition for condition in "ABCDEF" if (uid, condition) in quality]
        candidates.append({
            "role": role, "status": "selected", "example_uid": uid,
            "selection_rule": rule, "eligible_count": len(ranked), "selected_rank": ranked.index(uid)+1,
            "diversity_policy": "Select first unused UID in ranked pool; reuse rank 1 only if all eligible UIDs already used; UID breaks remaining ties",
            "rank_1_uid": ranked[0], "posthoc_diagnostic_selection": True,
            "category": quality[(uid, "A")]["category"],
            "knowledge_type": quality[(uid, "A")]["knowledge_type"],
            "instruction": quality[(uid, "A")]["instruction"],
            "saved_thought": quality[(uid, "B")].get("thought"),
            "Q": {condition: q(uid, condition) for condition in conditions},
            "differences": {"B-A": delta(uid, "B", "A"), "D-B": delta(uid, "D", "B"),
                            **({"E-D": delta(uid, "E", "D"), "F-D": delta(uid, "F", "D")} if uid in subset else {})},
            "charged_speedup_B_over_D": speed(uid),
            "charged_median_seconds": {condition: timing[(uid, condition)]["charged_total_seconds"] for condition in conditions},
            "judge_rationales_original": {condition: scores[(uid, condition)].get("rationale", "") for condition in conditions},
            "visual_interpretation": "Selection and captions use saved scores and metadata; no new visual judgement is asserted by this script.",
        })
    return candidates


def _render_candidate(data, candidate, directory):
    uid, role = candidate["example_uid"], candidate["role"]
    paths, quality = _get(data, "paths"), _get(data, "quality")
    source = Image.fromarray(_rgb(paths[(uid, "A")]["source"]))
    source_dimensions = source.size
    thought = candidate.get("saved_thought") or {}
    captions = [f"Instruction: {candidate['instruction']}",
                f"Saved thinking rewrite ({thought.get('token_count', 'unlogged')} tokens incl. EOS where recorded): {thought.get('rewrite', '')}",
                "Q: original saved manual-judge aggregate (0-100). Time: median of three NSCC charged-time repetitions. Display thumbnails only."]
    figures = {}
    for conditions in ("ABCD", "BDEF"):
        if not all((uid, condition) in paths for condition in conditions):
            continue
        panels = [(source, f"Source ({source_dimensions[0]} x {source_dimensions[1]})")]
        for condition in conditions:
            image = Image.fromarray(_rgb(paths[(uid, condition)]["edited"]))
            panels.append((image, f"{condition} | Q={candidate['Q'][condition]:.2f}\n{candidate['charged_median_seconds'][condition]:.2f} s"))
        filename = f"{role}__{uid}__{conditions}.png"
        _strip(panels, f"{role.replace('_', ' ')} | {uid}", captions, directory/filename)
        figures[conditions] = filename
    overlays = [(source, "Source")]
    for condition in "DEF":
        if (uid, condition) not in quality:
            continue
        selections = quality[(uid, condition)].get("regione", {}).get("selected_regions", [])
        for selection in selections:
            overlays.append((_selection_overlay(source, selection),
                             f"{condition}: selected tokens\nstep {selection['step']} (zero-based)"))
    if len(overlays) > 1:
        filename = f"{role}__{uid}__selected_regions.png"
        _strip(overlays, f"Saved region selection snapshots | {uid}",
               [f"Instruction: {candidate['instruction']}",
                "Red indicates saved selected image-token cells projected to source geometry; these are not attention weights or a per-step trajectory."], directory/filename)
        figures["selected_regions"] = filename
    pairs = [("D", "B")]
    if (uid, "E") in paths:
        pairs.extend((("E", "D"), ("F", "D")))
    differences = [(source, "Source")]
    for first, second in pairs:
        first_array = _rgb(paths[(uid, first)]["edited"])
        second_array = _rgb(paths[(uid, second)]["edited"])
        if first_array.shape != second_array.shape:
            raise ValueError(f"Difference-map image geometry mismatch: {uid}/{first}/{second}")
        heat = np.clip(np.abs(first_array.astype(float)-second_array.astype(float))*4, 0, 255).astype(np.uint8)
        differences.append((Image.fromarray(heat), f"|{first}-{second}| x4 (RGB)"))
    filename = f"{role}__{uid}__differences.png"
    _strip(differences, f"Pixel differences, fixed x4 display scale | {uid}",
           [f"Instruction: {candidate['instruction']}",
            "Each RGB absolute difference is multiplied by 4 and clipped at 255; identical scaling across all panels. Changes do not imply an editing error."], directory/filename)
    figures["differences"] = filename
    return figures


def run(data, output: Path) -> dict:
    directory = output/"qualitative"
    directory.mkdir(parents=True, exist_ok=True)
    validation = validate_metric_implementation()
    (directory/"metric_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    paths, quality = _get(data, "paths"), _get(data, "quality")
    uids, subset = _get(data, "uids"), _get(data, "subset")
    comparisons = [("C", "A", uids), ("D", "B", uids), ("E", "D", subset),
                   ("F", "D", subset), ("E", "B", subset), ("F", "B", subset)]
    diagnostics = []
    for first, second, members in comparisons:
        for uid in members:
            a, b = _rgb(paths[(uid, first)]["edited"]), _rgb(paths[(uid, second)]["edited"])
            diagnostics.append({"example_uid": uid, "comparison": f"{first}-{second}",
                                "category": quality[(uid, first)]["category"],
                                "knowledge_type": quality[(uid, first)]["knowledge_type"],
                                "first_output_sha256": quality[(uid, first)]["output_sha256"],
                                "second_output_sha256": quality[(uid, second)]["output_sha256"],
                                **pixel_metrics(a, b)})
    write_csv(directory/"image_diagnostics.csv", diagnostics)
    grouped = defaultdict(list)
    for row in diagnostics:
        grouped[row["comparison"]].append(row)
    summary = {}
    for comparison, rows in grouped.items():
        summary[comparison] = {"n": len(rows), "identical_images": sum(row["identical_pixels"] for row in rows)}
        for metric in ("mae_rgb_0_255", "rmse_rgb_0_255", "psnr_db", "ssim_rgb"):
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[comparison][metric] = {"n_finite": len(values), "mean": float(np.mean(values)) if values else None,
                                           "median": float(np.median(values)) if values else None}
    (directory/"image_diagnostics_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    candidates = select_candidates(data, diagnostics)
    markdown = ["# RQ3 examples for inspection", "",
                "These examples were selected after scoring using the explicit deterministic rules below. They are diagnostic illustrations, not a representative sample or an independent confirmatory analysis. All 40 examples remain in quantitative estimates.", "",
                "Quality scores and rationales below are the original saved ChatGPT judge outputs. They are not new human annotations. Inspect the images before writing any additional visual claim.", "",
                "Each timing label is the per-example median of the three NSCC timing repetitions. Differences and speedups below use these medians.", ""]
    for candidate in candidates:
        if candidate["status"] != "selected":
            markdown.extend([f"## {candidate['role']}", "", "No example satisfied the selection rule.", ""])
            continue
        candidate["figures"] = _render_candidate(data, candidate, directory)
        uid = candidate["example_uid"]
        markdown.extend([f"## {candidate['role']}: {uid}", "", f"Rule: {candidate['selection_rule']}.", "",
                         f"Selected rank {candidate['selected_rank']} of {candidate['eligible_count']}; rank-1 UID: `{candidate['rank_1_uid']}`. {candidate['diversity_policy']}.", "",
                         f"Instruction: {candidate['instruction']}", "",
                         f"Saved rewrite: {(candidate.get('saved_thought') or {}).get('rewrite', '')}", "",
                         f"Saved raw thinking output: {(candidate.get('saved_thought') or {}).get('raw_text', '')}", "",
                         f"Saved generated token count: {(candidate.get('saved_thought') or {}).get('token_count', 'unlogged')}; count includes EOS: {(candidate.get('saved_thought') or {}).get('token_count_includes_eos', 'unlogged')}.", "",
                         f"Q differences: `{json.dumps(candidate['differences'], sort_keys=True)}`; charged B/D speedup: {candidate['charged_speedup_B_over_D']:.3f}x.", ""])
        for name, path in candidate["figures"].items():
            markdown.extend([f"[{name} figure]({path})", ""])
        for condition, rationale in candidate["judge_rationales_original"].items():
            markdown.extend([f"**Original judge rationale, {condition} (Q={candidate['Q'][condition]:.2f}):**", "",
                             *["> "+line for line in rationale.splitlines()], ""])
    (directory/"candidates.json").write_text(json.dumps(candidates, indent=2, ensure_ascii=False)+"\n")
    (directory/"candidates.md").write_text("\n".join(markdown)+"\n")
    methods = """# Image diagnostics methods

Comparisons: C-A and D-B over all 40 examples; E-D, F-D, E-B and F-B over the fixed 20-example component subset. Each row uses quality repeat 0. All images are decoded as RGB and compared at their exact original geometry. A geometry mismatch stops the run; no resizing is allowed for metrics.

MAE and RMSE average all RGB pixel values on the 0-255 scale. PSNR = 10 log10(255^2 / MSE). Identical pairs have mathematical PSNR +infinity; the CSV/JSON leave numeric PSNR blank/null and explicitly set psnr_infinite=true. Summary PSNR means exclude those infinite values and report n_finite.

SSIM uses a normalized 11x11 Gaussian window, sigma 1.5, population covariance, K1=0.01, K2=0.03, data range 255, separate RGB channels followed by their arithmetic mean, and only complete windows (five-pixel border excluded). At each window:

SSIM = ((2 mu_x mu_y + C1)(2 cov_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(var_x + var_y + C2)), with C1=(0.01*255)^2 and C2=(0.03*255)^2.

This local NumPy/SciPy implementation uses parameters documented in [the official scikit-image SSIM reference](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity). metric_validation.json records identity, symmetry, constant-image closed-form and independent direct-window checks. scikit-image is not required.

These are differences between generated images, not accuracy against a target, semantic editing quality, or ground truth. Lower MAE/higher PSNR/SSIM cannot by themselves establish better editing. The frozen manual Q remains the quality endpoint. LPIPS is handled separately by the optional --lpips stage; when enabled, its results, versions and weight hashes are in ../perceptual/. The frozen evaluator dependencies are unchanged.

Figure thumbnails are resized only for display. Pixel-difference figures use the same fixed absolute RGB difference multiplied by 4 and clipped at 255. Region overlays show the logged selected image-token grid projected onto the source image using nearest-neighbor cells and red opacity 140/255. They are selection snapshots, not attention weights, object segmentation ground truth, or a reconstructed per-step region trajectory.

Candidate rules are post-hoc diagnostics and are recorded in candidates.json, including selected rank and any diversity fallback. No examples are removed from quantitative estimates. Saved judge rationales and generated rewrites are reproduced verbatim and attributed. The script does not make new visual judgements.
"""
    (directory/"README.md").write_text(methods)
    return {"metrics_summary": summary, "metric_validation": validation,
            "diagnostic_pair_count": len(diagnostics), "candidates": candidates,
            "lpips": "handled separately by the optional perceptual stage"}
