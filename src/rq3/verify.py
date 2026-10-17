"""Validate independent development reruns and the two acceleration ablations."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .run import atomic_json, read_completed


def _valid_noise(row):
    value = row.get("initial_noise_sha256")
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_thought(row):
    thought = row.get("thought")
    return (isinstance(thought, dict)
            and isinstance(thought.get("raw_text"), str) and bool(thought["raw_text"].strip())
            and isinstance(thought.get("rewrite"), str) and bool(thought["rewrite"].strip())
            and type(thought.get("token_count")) is int and thought["token_count"] > 0
            and row.get("prompt") == thought["rewrite"])


def _groups(rows):
    groups = defaultdict(dict)
    for row in rows.values():
        groups[(row["example_uid"], row["base_seed"])][row["condition"]] = row
    return groups


def _paired_controls(group):
    """Same actual noise for every arm, same cached thought for thinking arms."""
    values = list(group.values())
    noises = {row.get("initial_noise_sha256") for row in values}
    thoughts = [row for condition, row in group.items() if condition in "BDEF"]
    same_thought = (all(_valid_thought(row) for row in thoughts)
                    and len({(row["thought"]["raw_text"], row["thought"]["rewrite"])
                             for row in thoughts}) <= 1)
    direct_prompts = all(row.get("prompt") == row.get("instruction")
                         for condition, row in group.items() if condition in "AC")
    return all(_valid_noise(row) for row in values) and len(noises) == 1 and same_thought and direct_prompts


def _trace_events(row):
    steps = row.get("regione", {}).get("steps", [])
    if (len(steps) != 28 or [step.get("step") for step in steps] != list(range(28))
            or not all(type(step.get("avd_cache_reused")) is bool for step in steps)):
        return None
    return [step["avd_cache_reused"] for step in steps]


def _valid_regions(regions):
    return (isinstance(regions, list) and bool(regions)
            and all(isinstance(region, dict)
                    and type(region.get("grid_height")) is int and region["grid_height"] > 0
                    and type(region.get("grid_width")) is int and region["grid_width"] > 0
                    and isinstance(region.get("selected_indices"), list)
                    and all(type(index) is int and 0 <= index < region["grid_height"] * region["grid_width"]
                            for index in region["selected_indices"])
                    for region in regions))


def load_rows(root):
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    rows = read_completed(root / "generations.jsonl", protocol["protocol_hash"])
    return protocol, {key: row for key, row in rows.items() if row.get("repeat", 0) == 0}


def compare_reruns(first, second, tolerance=0.0):
    if not 0 <= tolerance <= 1:
        raise ValueError("Tolerance must be 0 (bitwise) or <=1 (strict MAE bound) on the 0..255 scale")
    if Path(first).resolve() == Path(second).resolve():
        raise ValueError("Determinism requires two independent run directories")
    config_a, rows_a = load_rows(first)
    config_b, rows_b = load_rows(second)
    if config_a["protocol_hash"] != config_b["protocol_hash"] or set(rows_a) != set(rows_b):
        raise ValueError("Determinism runs must have the same protocol and generation identities")
    if not rows_a:
        raise ValueError("No completed development outputs")
    if any(row.get("phase") != "pilot" for row in [*rows_a.values(), *rows_b.values()]):
        raise ValueError("Determinism checks use development pilots only")
    results = []
    for key, a in rows_a.items():
        b = rows_b[key]
        with Image.open(a["output_path"]) as image:
            left = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(b["output_path"]) as image:
            right = np.asarray(image.convert("RGB"), dtype=np.float32)
        same_size = left.shape == right.shape
        difference = float(np.abs(left - right).mean()) if same_size else None
        if a["condition"] in "BDEF":
            thought_equal = (_valid_thought(a) and _valid_thought(b)
                             and a["thought"]["raw_text"] == b["thought"]["raw_text"])
        else:
            thought_equal = a.get("thought") is None and b.get("thought") is None
        rewrite_equal = (isinstance(a.get("prompt"), str) and bool(a["prompt"].strip())
                         and a["prompt"] == b.get("prompt"))
        noise_equal = _valid_noise(a) and _valid_noise(b) and a["initial_noise_sha256"] == b["initial_noise_sha256"]
        source_equal = a.get("source_sha256") == b.get("source_sha256")
        pixels_pass = same_size and (difference == 0 if tolerance == 0 else difference < tolerance)
        passed = pixels_pass and thought_equal and rewrite_equal and noise_equal and source_equal
        results.append({"key": list(key), "mean_absolute_pixel_difference": difference,
                        "thought_equal": thought_equal, "rewrite_equal": rewrite_equal,
                        "initial_noise_equal": noise_equal, "source_equal": source_equal, "passed": passed})
    n_examples = len({row["example_uid"] for row in rows_a.values()})
    groups_a, groups_b = _groups(rows_a), _groups(rows_b)
    missing = [{"example_uid": uid, "base_seed": seed, "conditions": sorted(set("ABCD") - set(group))}
               for (uid, seed), group in groups_a.items() if not set("ABCD") <= set(group)]
    coverage = n_examples >= 10 and not missing
    paired = [{"example_uid": uid, "base_seed": seed,
               "passed": _paired_controls(group) and _paired_controls(groups_b[(uid, seed)])}
              for (uid, seed), group in groups_a.items()]
    return {"protocol_hash": config_a["protocol_hash"], "kind": "determinism", "tolerance": tolerance,
            "n_examples": n_examples, "coverage_passed": coverage,
            "missing_cells": missing, "paired_controls": paired,
            "passed": coverage and all(row["passed"] for row in results + paired), "comparisons": results}


def compare_components(root):
    config, rows = load_rows(root)
    if any(row.get("phase") != "pilot" for row in rows.values()):
        raise ValueError("Component validation must use K-dev pilot outputs")
    results, missing = [], []
    for (uid, seed), group in _groups(rows).items():
        if not set("DEF") <= set(group):
            missing.append({"example_uid": uid, "base_seed": seed, "conditions": sorted(set("DEF") - set(group))})
            continue
        d, e, f = (group[condition] for condition in "DEF")
        same_noise = all(_valid_noise(r) for r in (d, e, f)) and len({r["initial_noise_sha256"] for r in (d, e, f)}) == 1
        same_rewrite = all(_valid_thought(r) for r in (d, e, f)) and len({r["prompt"] for r in (d, e, f)}) == 1
        same_thought = (same_rewrite and len({r["thought"]["raw_text"] for r in (d, e, f)}) == 1)
        d_regions = d.get("regione", {}).get("selected_regions")
        e_regions = e.get("regione", {}).get("selected_regions")
        same_regions = _valid_regions(d_regions) and _valid_regions(e_regions) and d_regions == e_regions
        d_events, e_events, f_events = (_trace_events(r) for r in (d, e, f))
        events_present = all(events is not None for events in (d_events, e_events, f_events))
        same_events = events_present and d_events == e_events
        no_avd = events_present and not any(f_events)
        no_rikv = (e.get("regione", {}).get("rikv_enabled") is False and events_present
                   and all(step.get("rikv_cache_writes") == 0 and step.get("rikv_context_reuses") == 0
                           for step in e["regione"]["steps"]))
        flags_match = all(r.get("regione", {}).get("enabled") is True for r in (d, e, f))
        flags_match = (flags_match and d["regione"].get("rikv_enabled") is True
                       and f["regione"].get("rikv_enabled") is True
                       and d["regione"].get("avd_enabled") is True
                       and e["regione"].get("avd_enabled") is True
                       and f["regione"].get("avd_enabled") is False)
        passed = same_noise and same_thought and same_regions and same_events and no_avd and no_rikv and flags_match
        results.append({"example_uid": uid, "base_seed": seed, "same_noise": same_noise,
                        "same_rewrite": same_rewrite, "same_thought": same_thought,
                        "same_selected_regions_D_E": same_regions, "condition_flags_match": flags_match,
                        "same_AVD_events_D_E": same_events, "F_has_no_AVD_reuse": no_avd,
                        "E_has_no_RIKV_injection": no_rikv, "passed": passed})
    coverage = len({r["example_uid"] for r in results}) >= 3 and not missing
    return {"protocol_hash": config["protocol_hash"], "kind": "components",
            "passed": coverage and all(r["passed"] for r in results),
            "coverage_passed": coverage, "missing_cells": missing,
            "comparisons": results,
            "note": "D/E trajectories may diverge after context removal. Inspect any region/event mismatch before scaling; never silently force a matching result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    determinism = subs.add_parser("determinism")
    determinism.add_argument("--first", required=True)
    determinism.add_argument("--second", required=True)
    determinism.add_argument("--tolerance", type=float, default=0.0)
    determinism.add_argument("--output", required=True)
    components = subs.add_parser("components")
    components.add_argument("--run", required=True)
    components.add_argument("--output", required=True)
    args = parser.parse_args()
    report = (compare_reruns(args.first, args.second, args.tolerance) if args.command == "determinism"
              else compare_components(args.run))
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
