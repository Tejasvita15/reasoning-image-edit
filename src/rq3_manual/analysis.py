"""Use RQ3 statistics with an explicitly separate manual evaluation protocol."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from src.rq3.analyze import (
    ABLATION_CONTRASTS, CONTRASTS, condition_means, efficiency_report,
    example_contrasts, make_score_table, paired_summary, plot_quality_latency,
    region_diagnostics, similarity_diagnostics, summarize_timing, validate_pair_metadata,
)
from src.rq3.evaluate import file_sha256, read_jsonl, row_key
from src.rq3.protocol import build_jobs
from .common import load_bundle


def analyze_manual(args):
    from .results import validate_score_row
    bundle = load_bundle(args.bundle)
    if bundle["kind"] != "main":
        raise ValueError("Use a complete main bundle for test analysis")
    snapshot = bundle["generation_snapshot"]
    config = {**snapshot["config"], "protocol_hash": bundle["generation_protocol_hash"],
              "_splits": snapshot["splits"], "_subset": snapshot["subset"]}
    seeds = bundle["seeds"]
    expected = {row_key(row) for row in build_jobs(config, phase="all", seeds=seeds)}
    items = {row_key(item["row"]): item for item in bundle["items"]}
    if set(items) != expected:
        raise ValueError("Bundle is not a complete main experiment")
    scores = read_jsonl(args.scores)
    table = make_score_table(scores, expected, config["protocol_hash"])
    assignments = {entry["item_key"]: (batch["batch_id"], entry["blind_id"])
                   for batch in bundle["batches"] for entry in batch["items"]}
    positions = {item["item_key"]: index for index, item in enumerate(bundle["items"])}
    batch_records, conversations = {}, {}
    for key, score in table.items():
        validate_score_row(score)
        item = items[key]
        if (score.get("evaluation_protocol_hash") != bundle["evaluation_protocol_hash"]
                or score.get("bundle_manifest_hash") != bundle["manifest_hash"]
                or score.get("selection_hash") != bundle["selection_hash"]
                or score.get("pass_number") != 1
                or score.get("bundle_kind") != "main"
                or score.get("generation_snapshot") != snapshot
                or score.get("item_key") != item["item_key"]
                or score.get("selection_size") != len(items)
                or score.get("selection_index") != positions[item["item_key"]]
                or score.get("official_family") != item["official_family"]
                or score.get("knowledge_required") != item["knowledge_required"]
                or (score.get("batch_id"), score.get("blind_id")) != assignments[item["item_key"]]
                or score.get("source_sha256") != item["row"]["source_sha256"]
                or score.get("output_sha256") != item["row"]["output_sha256"]):
            raise ValueError(f"Manual score does not match the frozen main bundle: {key}")
        batch_identity = (score["raw_response_sha256"], score["session_metadata_sha256"])
        if batch_records.setdefault(score["batch_id"], batch_identity) != batch_identity:
            raise ValueError("Scores within a batch have different responses/session metadata")
        session = score["session_metadata"]["conversation_url_or_id"].strip().rstrip("/")
        if conversations.setdefault(session, score["batch_id"]) != score["batch_id"]:
            raise ValueError("Different batches must use separate conversations")
    generation_rows = read_jsonl(args.generations)
    public = {**snapshot["config"], "protocol_hash": config["protocol_hash"]}
    for row in generation_rows:
        if row.get("config") != public:
            raise ValueError("Generation configuration differs from the archived main bundle")
    generation_rows = [row for row in generation_rows if row["base_seed"] in seeds]
    medians, failures, successful = summarize_timing(
        generation_rows, expected, config["protocol_hash"], max(3, config.get("timing", {}).get("repeats", 3)))
    validate_pair_metadata(successful, expected)
    for key, score in table.items():
        generated = successful[(*key, 0)]
        if generated != items[key]["row"]:
            raise ValueError(f"Quality generation record changed since export: {key}")
        for field, hash_field in (("output_path", "output_sha256"), ("source_path", "source_sha256")):
            if file_sha256(generated[field]) != generated[hash_field]:
                raise ValueError(f"Generation image changed: {key}")
    stats = config["statistics"]
    mmd, delta = stats["minimum_meaningful_difference"], stats["equivalence_margin"]
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (mmd, delta))
            or not 0 <= mmd <= delta or delta <= 0):
        raise ValueError("The generation protocol must freeze positive delta >= MMD >= 0")
    calibration = bundle["calibration_report"]
    candidate = calibration.get("candidate_minimum_meaningful_difference")
    if (calibration["evaluation_protocol_hash"] != bundle["evaluation_protocol_hash"]
            or calibration.get("judge_interface") != "chatgpt_manual"
            or calibration.get("method") != "percentile95_absolute_rescore_Q_difference"
            or type(calibration.get("outputs")) is not int or calibration["outputs"] < 20
            or type(candidate) not in (int, float) or not math.isfinite(candidate)
            or not 0 <= candidate <= mmd):
        raise ValueError("Manual calibration does not match the frozen main evaluation")
    uids, subset = snapshot["splits"]["k_small"], snapshot["subset"]["example_uids"]
    kwargs = dict(resamples=stats["bootstrap_resamples"], confidence=stats["confidence"], seed=stats["bootstrap_seed"])
    def contrasts(ids, definitions):
        return {name: paired_summary(example_contrasts(table, ids, seeds, coefficients), **kwargs)
                for name, coefficients in definitions.items()}
    report = {
        "protocol_hash": config["protocol_hash"], "generation_protocol_hash": config["protocol_hash"],
        "evaluation_protocol_hash": bundle["evaluation_protocol_hash"],
        "evaluation_protocol": bundle["evaluation_protocol"], "judge_interface": "chatgpt_manual",
        "aggregate": "mean_applicable_normalized", "seeds": seeds,
        "scope": "RQ3 subset, example-weighted; manual ChatGPT adaptation, not the published KRIS aggregate/API protocol",
        "deployment": config["model"], "generation": config["generation"],
        "conditions": condition_means(table, uids, seeds, "ABCD"),
        "contrasts": contrasts(uids, CONTRASTS),
        "ablations": {"conditions": condition_means(table, subset, seeds, "BDEF"),
                      "contrasts": contrasts(subset, ABLATION_CONTRASTS)},
        "failed_generation_attempts": len(failures), "expected_quality_outputs": len(expected),
        "efficiency_thinking_on": efficiency_report(medians, uids, seeds, "BD", "B"),
        "efficiency_thinking_off": efficiency_report(medians, uids, seeds, "AC", "A"),
        "efficiency_ablations": efficiency_report(medians, subset, seeds, "BDEF", "B"),
    }
    loss = report["contrasts"]["thinking_on_acceleration_D_minus_B"]
    interaction = report["contrasts"]["interaction_D_minus_C_minus_B_plus_A"]
    g0 = report["contrasts"]["G0_B_minus_A"]
    report["retention"] = {
        "equivalence_margin": delta, "minimum_meaningful_difference": mmd,
        "quality_retained_one_sided_rule": loss["ci_low"] > -delta,
        "interaction_within_equivalence_margin": interaction["ci_low"] > -delta and interaction["ci_high"] < delta,
        "unaccelerated_thinking_benefit_established_above_MMD": g0["ci_low"] > mmd,
        "note": "Nonsignificance is not equivalence. Do not import an RQ1 thinking benefit.",
    }
    records = [{"example_uid": uid, **{name: example_contrasts(table, [uid], seeds, coefficients)[0]
                for name, coefficients in CONTRASTS.items()}} for uid in uids]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "paired_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    plot_quality_latency(report, output)
    region_diagnostics(successful, subset, seeds, output)
    if args.similarity:
        similarity_diagnostics(successful, subset, seeds, output)
    print(json.dumps({"report": str(output / "report.json"), "retention": report["retention"]}, indent=2))
