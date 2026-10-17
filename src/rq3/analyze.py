"""Paired, example-level RQ3 contrasts, efficiency and component diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from .evaluate import SCORE_FIELDS, file_sha256, quality_score, read_jsonl, row_key, validate_judge


CONTRASTS = {
    "G0_B_minus_A": {"B": 1, "A": -1},
    "G1_D_minus_C": {"D": 1, "C": -1},
    "interaction_D_minus_C_minus_B_plus_A": {"D": 1, "C": -1, "B": -1, "A": 1},
    "thinking_on_acceleration_D_minus_B": {"D": 1, "B": -1},
    "thinking_off_acceleration_C_minus_A": {"C": 1, "A": -1},
}
ABLATION_CONTRASTS = {
    "full_D_minus_vanilla_B": {"D": 1, "B": -1},
    "without_RIKV_E_minus_full_D": {"E": 1, "D": -1},
    "without_AVD_F_minus_full_D": {"F": 1, "D": -1},
    "without_RIKV_E_minus_vanilla_B": {"E": 1, "B": -1},
    "without_AVD_F_minus_vanilla_B": {"F": 1, "B": -1},
}
TIMING_FIELDS = ("preprocess_seconds", "thinking_seconds", "thinking_offload_seconds", "diffusion_seconds",
                 "postprocess_seconds", "offload_seconds", "measured_replay_seconds", "charged_total_seconds")


def paired_summary(differences, *, resamples=10000, confidence=.95, seed=42):
    import numpy as np
    from scipy.stats import wilcoxon
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Differences must be a nonempty, finite, one-dimensional vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, [(1-confidence)/2, 1-(1-confidence)/2])
    pvalue = 1.0 if np.all(values == 0) else float(wilcoxon(values, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)
    return {"n_examples": len(values), "mean": float(values.mean()), "ci_low": float(low),
            "ci_high": float(high), "ci_width": float(high-low), "wilcoxon_p": pvalue,
            "bootstrap_resamples": resamples, "confidence": confidence}


def make_score_table(rows, expected_keys, protocol_hash):
    table = {}
    for row in rows:
        if row["protocol_hash"] != protocol_hash:
            raise ValueError("Scores from different protocols cannot be pooled")
        key = row_key(row)
        if key in table:
            raise ValueError(f"Duplicate score: {key}")
        if not isinstance(row.get("Q"), (int, float)) or not math.isfinite(row["Q"]) or not 0 <= row["Q"] <= 100:
            raise ValueError(f"Invalid Q: {key}")
        table[key] = row
    if set(table) != set(expected_keys):
        missing, extra = set(expected_keys)-set(table), set(table)-set(expected_keys)
        raise ValueError(f"Unpaired score matrix: missing={len(missing)}, extra={len(extra)}; first missing={sorted(missing)[:3]}")
    return table


def example_contrasts(table, uids, seeds, coefficients, metric="Q"):
    """Average seeds WITHIN each example before inference across examples."""
    differences = []
    for uid in uids:
        seed_differences = []
        for seed in seeds:
            values = [table[(uid, condition, seed)].get(metric) for condition in coefficients]
            if any(value is None for value in values):
                raise ValueError(f"Missing metric {metric} in a required pair: {uid}")
            seed_differences.append(sum(coefficients[c] * table[(uid, c, seed)][metric] for c in coefficients))
        differences.append(statistics.mean(seed_differences))
    return differences


def condition_means(table, uids, seeds, conditions):
    result = {}
    for condition in conditions:
        metrics = {}
        for metric in ("Q",) + SCORE_FIELDS + ("knowledge_score",):
            example_values = []
            for uid in uids:
                values = [table[(uid, condition, seed)].get(metric) for seed in seeds]
                if all(v is not None for v in values):
                    example_values.append(statistics.mean(values))
                elif any(v is not None for v in values):
                    raise ValueError("A component is missing for only some seeds")
            metrics[metric] = {"mean": statistics.mean(example_values) if example_values else None,
                               "n_examples": len(example_values)}
        result[condition] = metrics
    return result


def summarize_timing(rows, expected_keys, protocol_hash, minimum_repeats=3):
    """Exclude quality/warmup runs; median repetitions then mean seeds/examples."""
    successful, failed = {}, []
    for row in rows:
        if row["protocol_hash"] != protocol_hash:
            raise ValueError("Generation protocols differ")
        if row.get("status") != "ok":
            failed.append(row)
            continue
        key = row_key(row)
        if key not in expected_keys:
            raise ValueError(f"Unexpected generation key: {key}")
        repeat = row.get("repeat", 0)
        full_key = (*key, repeat)
        if full_key in successful:
            raise ValueError(f"Duplicate successful timing/quality record: {full_key}")
        successful[full_key] = row
    medians = {}
    for key in sorted(expected_keys):
        quality = successful.get((*key, 0))
        if not quality:
            raise ValueError(f"Missing successful quality generation {key}")
        measurements = [row for full_key, row in successful.items() if full_key[:3] == key and full_key[3] > 0]
        if len(measurements) < minimum_repeats:
            raise ValueError(f"Need >= {minimum_repeats} timing repeats after warmup for {key}; got {len(measurements)}")
        for row in measurements:
            for field in ("diffusion_seconds", "charged_total_seconds", "measured_replay_seconds"):
                value = row["timings"].get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid measured timing {field}: {key}")
        medians[key] = {field: statistics.median(row["timings"][field] for row in measurements)
                        for field in ("diffusion_seconds", "charged_total_seconds", "measured_replay_seconds")}
        for field in TIMING_FIELDS:
            values = [row["timings"].get(field) for row in measurements]
            if all(value is not None for value in values):
                if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"Invalid stage timing {field}: {key}")
                medians[key][field] = statistics.median(values)
            elif any(value is not None for value in values):
                raise ValueError(f"Stage timing {field} missing in some repetitions: {key}")
            else:
                medians[key][field] = None
        medians[key]["peak_vram_bytes"] = max(row["peak_vram_bytes"] for row in measurements)
        medians[key]["timing_repeats"] = len(measurements)
    return medians, failed, successful


def efficiency_report(medians, uids, seeds, conditions, baseline):
    conditions_report = {}
    for condition in conditions:
        conditions_report[condition] = {}
        for field in TIMING_FIELDS + ("peak_vram_bytes",):
            available = all(medians[(uid, condition, seed)].get(field) is not None for uid in uids for seed in seeds)
            example_values = [statistics.mean(medians[(uid, condition, seed)][field] for seed in seeds) for uid in uids] if available else []
            conditions_report[condition]["mean_" + field] = statistics.mean(example_values) if example_values else None
        if condition != baseline:
            for label, field in (("generator", "diffusion_seconds"), ("charged_end_to_end", "charged_total_seconds")):
                ratios = [statistics.mean(medians[(uid, baseline, seed)][field] / medians[(uid, condition, seed)][field] for seed in seeds) for uid in uids]
                conditions_report[condition][label + "_speedup_mean_paired_ratio"] = statistics.mean(ratios)
                conditions_report[condition][label + "_speedup_ratio_of_means"] = (
                    statistics.mean(statistics.mean(medians[(uid, baseline, seed)][field] for seed in seeds) for uid in uids)
                    / conditions_report[condition]["mean_" + field])
    return {"baseline": baseline, "conditions": conditions_report}


def plot_quality_latency(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    efficiency = report["efficiency_thinking_on"]["conditions"] | report["efficiency_thinking_off"]["conditions"]
    for axis, field, title in zip(axes, ("mean_diffusion_seconds", "mean_charged_total_seconds"), ("Generator", "Charged end-to-end")):
        for condition in "ABCD":
            x = efficiency[condition][field]
            y = report["conditions"][condition]["Q"]["mean"]
            axis.scatter(x, y)
            axis.annotate(condition, (x, y), xytext=(5, 5), textcoords="offset points")
        axis.set(xlabel=title + " seconds", ylabel="Subset Q (0–100)", ylim=(0, 100))
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "quality_latency.png", dpi=180)
    plt.close(fig)


def similarity_diagnostics(successful, uids, seeds, output):
    """Same-size comparisons to vanilla B; no resizing that hides errors."""
    import numpy as np
    import torch
    import lpips
    from PIL import Image, ImageDraw
    import textwrap
    model = lpips.LPIPS(net="alex").eval()
    rows = []
    folder = output / "differences"
    folder.mkdir(exist_ok=True)
    for uid in uids:
        for seed in seeds:
            baseline = successful[(uid, "B", seed, 0)]
            with Image.open(baseline["output_path"]) as handle:
                vanilla = np.asarray(handle.convert("RGB")).copy()
            for condition in "DEF":
                row = successful[(uid, condition, seed, 0)]
                with Image.open(row["output_path"]) as handle:
                    edited = np.asarray(handle.convert("RGB")).copy()
                if vanilla.shape != edited.shape:
                    raise ValueError(f"Image geometry differs within pair {uid}: B/{condition}")
                diff = vanilla.astype(float) - edited.astype(float)
                mse = float(np.mean(diff ** 2))
                def tensor(array):
                    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
                with torch.inference_mode():
                    distance = float(model(tensor(vanilla), tensor(edited)).item())
                rows.append({"example_uid": uid, "base_seed": seed, "condition": condition,
                             "lpips_vs_B": distance, "psnr_vs_B_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
                             "identical_pixels": mse == 0, "mean_absolute_pixel_difference": float(np.mean(np.abs(diff)))})
                heat = np.clip(np.abs(diff) * 4, 0, 255).astype("uint8")
                height, width = vanilla.shape[:2]
                with Image.open(row["source_path"]) as handle:
                    source = handle.convert("RGB")
                if source.size != (width, height):
                    raise ValueError(f"Output/source dimensions differ for region diagnosis: {uid}")
                selected = source.copy()
                selections = row.get("regione", {}).get("selected_regions", [])
                if selections:
                    selection = selections[0]
                    mask = np.zeros(selection["grid_height"] * selection["grid_width"], dtype=np.uint8)
                    mask[selection["selected_indices"]] = 150
                    alpha = Image.fromarray(mask.reshape(selection["grid_height"], selection["grid_width"])).resize(source.size, Image.Resampling.NEAREST)
                    selected = Image.composite(Image.new("RGB", source.size, "red"), source, alpha)
                panels = ((source, "Source"), (selected, "Selected image tokens"),
                          (Image.fromarray(vanilla), "B vanilla"), (Image.fromarray(edited), condition),
                          (Image.fromarray(heat), "absolute difference x4"))
                caption = textwrap.fill(row["instruction"], width=max(60, width * 5 // 7))
                caption_height = 20 * (caption.count("\n") + 1)
                strip = Image.new("RGB", (width * 5, height + 24 + caption_height), "white")
                for index, (panel, label) in enumerate(panels):
                    strip.paste(panel, (width * index, 24))
                    ImageDraw.Draw(strip).text((width * index + 4, 4), label, fill="black")
                ImageDraw.Draw(strip).multiline_text((4, height + 28), caption, fill="black", spacing=5)
                strip.save(folder / f"{uid}_seed{seed}_{condition}.png")
    (output / "similarity.json").write_text(json.dumps(rows, indent=2) + "\n")


def region_diagnostics(successful, uids, seeds, output):
    """Render logged selections and summarize actual cache/transformer activity."""
    import numpy as np
    from PIL import Image, ImageDraw
    folder = output / "selected_regions"
    folder.mkdir(exist_ok=True)
    summaries = []
    for uid in uids:
        for seed in seeds:
            for condition in "BDEF":
                row = successful[(uid, condition, seed, 0)]
                telemetry = row.get("regione", {})
                steps = telemetry.get("steps", [])
                summaries.append({"example_uid": uid, "base_seed": seed, "condition": condition,
                                  "generator_calls": telemetry.get("generator_calls"),
                                  "full_image_generator_calls": telemetry.get("full_image_generator_calls"),
                                  "reduced_token_generator_calls": telemetry.get("reduced_token_generator_calls"),
                                  "avd_cache_reuse_steps": telemetry.get("avd_cache_reuse_steps"),
                                  "refresh_events": sum(bool(step.get("refresh_event")) for step in steps),
                                  "mean_rho_t": statistics.mean(step["rho_t"] for step in steps) if steps else None,
                                  "mean_selected_fraction": statistics.mean(step["selected_fraction"] for step in steps) if steps else None,
                                  "rikv_cache_writes": sum(step.get("rikv_cache_writes", 0) for step in steps),
                                  "rikv_context_reuses": sum(step.get("rikv_context_reuses", 0) for step in steps)})
                for selection in telemetry.get("selected_regions", []):
                    grid_height, grid_width = selection["grid_height"], selection["grid_width"]
                    indices = selection["selected_indices"]
                    if any(index < 0 or index >= grid_height * grid_width for index in indices):
                        raise ValueError("Logged region indices are outside the image-token grid")
                    mask = np.zeros(grid_height * grid_width, dtype=np.uint8)
                    mask[indices] = 150
                    with Image.open(row["source_path"]) as handle:
                        source = handle.convert("RGB")
                    alpha = Image.fromarray(mask.reshape(grid_height, grid_width)).resize(source.size, Image.Resampling.NEAREST)
                    selected = Image.composite(Image.new("RGB", source.size, "red"), source, alpha)
                    width, height = source.size
                    panel = Image.new("RGB", (2 * width, height + 24), "white")
                    panel.paste(source, (0, 24))
                    panel.paste(selected, (width, 24))
                    draw = ImageDraw.Draw(panel)
                    draw.text((4, 4), "Source", fill="black")
                    draw.text((width + 4, 4), f"{condition}: selected tokens, step {selection['step']}", fill="black")
                    panel.save(folder / f"{uid}_seed{seed}_{condition}_step{selection['step']}.png")
    (output / "cache_activity.json").write_text(json.dumps(summaries, indent=2) + "\n")


def validate_pair_metadata(successful, expected):
    """Confirm logged source, initial noise and frozen rewrite pairing."""
    for uid, condition, seed in sorted(expected):
        row = successful[(uid, condition, seed, 0)]
        base = successful[(uid, "A", seed, 0)]
        for field in ("source_sha256", "initial_noise_sha256", "stage_seeds"):
            if not row.get(field) or row[field] != base.get(field):
                raise ValueError(f"Paired generation metadata differs in {field}: {uid}/{condition}/{seed}")
        if condition in "BDEF":
            thought = successful[(uid, "B", seed, 0)]
            if not row.get("prompt") or row["prompt"] != thought.get("prompt"):
                raise ValueError(f"Frozen thinking rewrite differs: {uid}/{condition}/{seed}")
        elif row.get("prompt") != base.get("instruction"):
            raise ValueError(f"Thinking-off prompt differs from instruction: {uid}/{condition}/{seed}")
    for full_key, row in successful.items():
        quality = successful[(*full_key[:3], 0)]
        for field in ("prompt", "source_sha256", "initial_noise_sha256", "stage_seeds"):
            if row.get(field) != quality.get(field):
                raise ValueError(f"Timing repetition changed {field}: {full_key}")


def analyze(config, score_rows, generation_rows, seeds):
    from .protocol import build_jobs
    validate_judge(config)
    expected = {row_key(job) for job in build_jobs(config, phase="all", seeds=seeds)}
    scores = [row for row in score_rows if row["base_seed"] in seeds]
    generations = [row for row in generation_rows if row["base_seed"] in seeds]
    table = make_score_table(scores, expected, config["protocol_hash"])
    prompt_hashes = {row.get("prompt_sha256") for row in table.values()}
    if len(prompt_hashes) != 1 or None in prompt_hashes or "" in prompt_hashes:
        raise ValueError("All scores must identify the same pinned prompt file")
    for key, row in table.items():
        if row.get("judge_model") != config["_judge"]["model"] or row.get("aggregate") != config["_judge"]["aggregate"]:
            raise ValueError(f"Score judge or aggregate differs from the frozen protocol: {key}")
        category = key[0].rsplit("__", 1)[0]
        from src.taxonomy import get_reasoning_dimension
        knowledge = get_reasoning_dimension(category) in {"social_science", "natural_science", "logical_reasoning"}
        if not math.isclose(row["Q"], quality_score(row, knowledge), abs_tol=1e-10):
            raise ValueError(f"Q differs from the declared component aggregation: {key}")
    medians, failures, successful = summarize_timing(generations, expected, config["protocol_hash"], max(3, config.get("timing", {}).get("repeats", 3)))
    validate_pair_metadata(successful, expected)
    for key, score in table.items():
        generated = successful[(*key, 0)]
        if score.get("output_sha256") != generated.get("output_sha256") or file_sha256(generated["output_path"]) != score.get("output_sha256"):
            raise ValueError(f"Score/image hash mismatch: {key}")
    uids = config["_splits"]["k_small"]
    subset = config["_subset"]["example_uids"]
    stats = config["statistics"]
    kwargs = dict(resamples=stats["bootstrap_resamples"], confidence=stats["confidence"], seed=stats["bootstrap_seed"])
    def contrasts(ids, definitions):
        return {name: paired_summary(example_contrasts(table, ids, seeds, coefficients), **kwargs) for name, coefficients in definitions.items()}
    report = {"protocol_hash": config["protocol_hash"], "seeds": seeds,
              "aggregate": config["_judge"]["aggregate"], "scope": "RQ3 subset, example-weighted; not the published full-benchmark aggregate",
              "deployment": config.get("model"), "generation": config.get("generation"),
              "conditions": condition_means(table, uids, seeds, "ABCD"),
              "contrasts": contrasts(uids, CONTRASTS),
              "ablations": {"conditions": condition_means(table, subset, seeds, "BDEF"), "contrasts": contrasts(subset, ABLATION_CONTRASTS)},
              "failed_generation_attempts": len(failures), "expected_quality_outputs": len(expected),
              "efficiency_thinking_on": efficiency_report(medians, uids, seeds, "BD", "B"),
              "efficiency_thinking_off": efficiency_report(medians, uids, seeds, "AC", "A"),
              "efficiency_ablations": efficiency_report(medians, subset, seeds, "BDEF", "B")}
    loss = report["contrasts"]["thinking_on_acceleration_D_minus_B"]
    interaction = report["contrasts"]["interaction_D_minus_C_minus_B_plus_A"]
    g0 = report["contrasts"]["G0_B_minus_A"]
    delta, mmd = stats["equivalence_margin"], stats["minimum_meaningful_difference"]
    report["retention"] = {"equivalence_margin": delta, "minimum_meaningful_difference": mmd,
                           "quality_retained_one_sided_rule": loss["ci_low"] > -delta,
                           "interaction_within_equivalence_margin": interaction["ci_low"] > -delta and interaction["ci_high"] < delta,
                           "unaccelerated_thinking_benefit_established_above_MMD": g0["ci_low"] > mmd,
                           "note": "Quality retention uses lower95CI(D-B)>-delta. Nonsignificance is not equivalence. Do not import an RQ1 thinking benefit."}
    groups = sorted({row.get("knowledge_type") for row in table.values() if row.get("knowledge_type")})
    report["knowledge_types_descriptive_only"] = {}
    for group in groups:
        group_ids = [uid for uid in uids if table[(uid, "A", seeds[0])].get("knowledge_type") == group]
        report["knowledge_types_descriptive_only"][group] = condition_means(table, group_ids, seeds, "ABCD")
    records = [{"example_uid": uid, **{name: example_contrasts(table, [uid], seeds, coefficients)[0] for name, coefficients in CONTRASTS.items()}} for uid in uids]
    return report, records, successful


def main():
    from .protocol import load_protocol
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rq3.yaml")
    parser.add_argument("--scores", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--similarity", action="store_true", help="Compute LPIPS/PSNR and save difference strips against B")
    args = parser.parse_args()
    config = load_protocol(args.config)
    report, records, successful = analyze(config, read_jsonl(args.scores), read_jsonl(args.generations), args.seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "paired_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    plot_quality_latency(report, output)
    region_diagnostics(successful, config["_subset"]["example_uids"], args.seeds, output)
    if args.similarity:
        similarity_diagnostics(successful, config["_subset"]["example_uids"], args.seeds, output)
    print(json.dumps({"report": str(output / "report.json"), "retention": report["retention"]}, indent=2))


if __name__ == "__main__":
    main()
