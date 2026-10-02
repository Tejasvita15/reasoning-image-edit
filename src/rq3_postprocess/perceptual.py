"""Optional CPU LPIPS diagnostics without changing frozen evaluation modules.

Dependencies and official AlexNet weights must already be installed/cached by
the caller. Results measure generated-image differences, not editing accuracy.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import sha256, write_csv


def _get(data, key):
    return data[key] if isinstance(data, dict) else getattr(data, key)


def _canonical_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _load_rgb(path):
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB")).copy()


def _pair_plan(data, output):
    """Match the already-computed pixel diagnostic table exactly."""
    uids, subset = _get(data, "uids"), _get(data, "subset")
    expected = [(uid, first, second) for first, second, members in
                (("C", "A", uids), ("D", "B", uids), ("E", "D", subset),
                 ("F", "D", subset), ("E", "B", subset), ("F", "B", subset))
                for uid in members]
    with (output/"qualitative/image_diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = [(row["example_uid"], *row["comparison"].split("-")) for row in rows]
    if observed != expected:
        raise ValueError("LPIPS pair plan differs from the complete pixel diagnostics table")
    return rows


def _resume(path, identity_hash, pairs):
    expected = {(row["example_uid"], row["comparison"]): row for row in pairs}
    completed = {}
    if not path.exists():
        return completed
    with path.open() as handle:
        for index, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid LPIPS resume record at line {index}; preserve and inspect it before resuming") from error
            key = (row.get("example_uid"), row.get("comparison"))
            if key not in expected or key in completed:
                raise ValueError(f"Unexpected or duplicate LPIPS resume record: {key}")
            baseline = expected[key]
            if row.get("metric_identity_sha256") != identity_hash:
                raise ValueError("LPIPS resume metric/weight/code identity differs from this run")
            for field in ("first_output_sha256", "second_output_sha256"):
                if row.get(field) != baseline[field]:
                    raise ValueError(f"LPIPS resume image identity differs: {key}/{field}")
            if row.get("width") != int(baseline["width"]) or row.get("height") != int(baseline["height"]):
                raise ValueError(f"LPIPS resume image geometry differs: {key}")
            value = row.get("lpips_alex_v0_1")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < -1e-7:
                raise ValueError(f"Invalid LPIPS resume score: {key}")
            completed[key] = row
    return completed


def run(data, output: Path) -> dict:
    """Compute or resume the same 160 exact-geometry pairs as pixel diagnostics."""
    import torch
    import torchvision
    import lpips

    output = Path(output)
    directory = output/"perceptual"
    directory.mkdir(parents=True, exist_ok=True)
    pairs = _pair_plan(data, output)
    paths, quality = _get(data, "paths"), _get(data, "quality")
    torch.set_num_threads(4)
    lpips_root = Path(lpips.__file__).resolve().parent
    alexnet_weight = Path(torch.hub.get_dir())/"checkpoints/alexnet-owt-7be5be79.pth"
    metric_weight = lpips_root/"weights/v0.1/alex.pth"
    for weight in (alexnet_weight, metric_weight):
        if not weight.is_file():
            raise FileNotFoundError(f"Required LPIPS weight is not cached: {weight}; prepare dependencies/weights before this CPU analysis")
    identity = {
        "schema_version": 1,
        "metric": "LPIPS",
        "network": "alex",
        "version": "0.1",
        "learned_linear_weights": True,
        "device": "cpu",
        "dtype": "float32",
        "threads": 4,
        "execution": "eval + torch.inference_mode",
        "input": "Pillow RGB decode; NCHW; uint8/127.5-1; original geometry; no resizing; normalize=False",
        "aggregation": "default LPIPS spatial average, one scalar per image pair",
        "weights": {
            "alexnet_imagenet": {"filename": alexnet_weight.name, "sha256": sha256(alexnet_weight)},
            "lpips_linear": {"filename": "lpips/weights/v0.1/alex.pth", "sha256": sha256(metric_weight)},
        },
        "code_sha256": {
            "perceptual.py": sha256(Path(__file__)),
            "lpips/lpips.py": sha256(lpips_root/"lpips.py"),
            "lpips/pretrained_networks.py": sha256(lpips_root/"pretrained_networks.py"),
        },
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "lpips", "numpy", "Pillow")},
        "references": ["https://github.com/richzhang/PerceptualSimilarity",
                       "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"],
    }
    identity_hash = _canonical_hash(identity)
    identity_path = directory/"metric_identity.json"
    identity_record = {"metric_identity_sha256": identity_hash, "identity": identity}
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity_record:
            raise ValueError("Existing LPIPS output uses a different metric identity; choose a fresh derived output directory")
    else:
        identity_path.write_text(json.dumps(identity_record, indent=2, allow_nan=False)+"\n")
    resume_path = directory/"lpips_pairs.jsonl"
    completed = _resume(resume_path, identity_hash, pairs)
    # A cached score is only reused after checking the current original images.
    checked = set()
    for row in pairs:
        uid = row["example_uid"]
        first, second = row["comparison"].split("-")
        for condition, field in ((first, "first_output_sha256"), (second, "second_output_sha256")):
            key = (uid, condition)
            if row[field] != quality[key]["output_sha256"]:
                raise ValueError(f"LPIPS plan differs from original generation identity: {key}")
            if key not in checked:
                if sha256(paths[key]["edited"]) != row[field]:
                    raise ValueError(f"Local image bytes changed before LPIPS: {key}")
                checked.add(key)
    # Avoid perturbing caller RNG while the constructor initializes then loads
    # frozen pretrained modules. This metric uses no stochastic forward layers.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = lpips.LPIPS(net="alex", version="0.1", lpips=True, pnet_rand=False,
                            pnet_tune=False, verbose=False).to(device="cpu", dtype=torch.float32).eval()

    def tensor(array):
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)/127.5-1

    # Check an actual unresized experiment image, not a synthetic favorable case.
    first_pair = pairs[0]
    first_uid = first_pair["example_uid"]
    first_condition = first_pair["comparison"].split("-")[0]
    test_array = _load_rgb(paths[(first_uid, first_condition)]["edited"])
    test_tensor = tensor(test_array)
    with torch.inference_mode():
        identical = float(model(test_tensor, test_tensor, normalize=False).item())
    del test_array, test_tensor
    if not math.isfinite(identical) or abs(identical) > 1e-6:
        raise ValueError(f"LPIPS identical-image check failed: {identical}")
    validation = {"passed": True, "identical_image_lpips": identical, "tolerance": 1e-6,
                  "example_uid": first_uid, "condition": first_condition,
                  "metric_identity_sha256": identity_hash}
    (directory/"metric_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({"stage": "lpips", "completed": len(completed), "total": len(pairs),
                      "status": "ready", "metric_identity_sha256": identity_hash}), flush=True)
    with resume_path.open("a") as handle:
        for pair in pairs:
            uid, comparison = pair["example_uid"], pair["comparison"]
            key = (uid, comparison)
            if key in completed:
                continue
            first, second = comparison.split("-")
            a = _load_rgb(paths[(uid, first)]["edited"])
            b = _load_rgb(paths[(uid, second)]["edited"])
            if a.shape != b.shape or a.shape[:2] != (int(pair["height"]), int(pair["width"])):
                raise ValueError(f"LPIPS exact image geometry differs: {key}")
            first_tensor, second_tensor = tensor(a), tensor(b)
            with torch.inference_mode():
                score = float(model(first_tensor, second_tensor, normalize=False).item())
            del a, b, first_tensor, second_tensor
            if not math.isfinite(score) or score < -1e-7:
                raise ValueError(f"LPIPS returned an invalid score: {key}/{score}")
            row = {"example_uid": uid, "comparison": comparison,
                   "first_output_sha256": pair["first_output_sha256"],
                   "second_output_sha256": pair["second_output_sha256"],
                   "width": int(pair["width"]), "height": int(pair["height"]),
                   "lpips_alex_v0_1": score, "metric_identity_sha256": identity_hash}
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[key] = row
            if len(completed) % 10 == 0 or len(completed) == len(pairs):
                print(json.dumps({"stage": "lpips", "completed": len(completed), "total": len(pairs)}), flush=True)
    rows = [completed[(pair["example_uid"], pair["comparison"])] for pair in pairs]
    write_csv(directory/"lpips.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["comparison"]].append(row["lpips_alex_v0_1"])
    summary = {comparison: {"n": len(values), "mean": float(np.mean(values)),
                            "median": float(np.median(values)), "minimum": min(values),
                            "maximum": max(values)} for comparison, values in grouped.items()}
    result = {"metric_identity_sha256": identity_hash, "pair_count": len(rows),
              "metric_validation": validation, "comparisons": summary,
              "interpretation": "Supplementary generated-image differences; lower LPIPS alone does not establish better editing or correctness."}
    (directory/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (directory/"README.md").write_text("""# LPIPS image diagnostics

LPIPS uses the official lpips package, AlexNet backbone, learned linear weights version 0.1, CPU float32, model.eval() and torch.inference_mode(). RGB uint8 images are mapped to [-1,1] in NCHW format. Images are compared at their original identical geometry without resizing. Each scalar is the package's default spatially averaged perceptual distance.

Comparisons match qualitative/image_diagnostics.csv exactly: C-A and D-B on 40 examples each; E-D, F-D, E-B and F-B on the fixed 20-example subset. All images are quality repeat 0. These are distances between generated images, not against semantic ground truth. Lower distances cannot independently establish that an edit is correct or better.

metric_identity.json records package versions, implementation hashes, official AlexNet checkpoint hash and bundled learned LPIPS weight hash. The official AlexNet checkpoint must already be cached; this analysis does not download weights. metric_validation.json records a zero-distance check on an actual identical image pair. lpips_pairs.jsonl is flushed after each completed pair and supports resume only if all image, metric, code and weight identities match. Invalid, duplicate or mismatched records cause an error rather than being skipped.

References: [official LPIPS repository](https://github.com/richzhang/PerceptualSimilarity) and [official AlexNet checkpoint](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth).
""")
    return result
