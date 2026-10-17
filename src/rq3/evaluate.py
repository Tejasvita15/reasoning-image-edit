"""Run pinned official KRIS metric functions with a recorded, pinned judge.

The upstream scripts parse CLI arguments and initialize a client at import time.
We load their function ASTs (unchanged), literal constants and prompt strings,
then supply paths and a recording client. No rubric or score parser is rewritten.
Only repeat=0 outputs are scored. The CLI defaults to a no-API validation pass.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from types import SimpleNamespace

from src.notifications import ProgressNotifier, add_notification_options, load_notification_env


SCORE_FIELDS = ("consistency_score", "instruction_score", "quality_score")
OFFICIAL_REVISION = "cf1475253a13f5e5666826c10d8967e38917468a"
# Initial notification estimate only: each output needs three sequential judge
# calls. This is not a measured benchmark; observed throughput replaces it.
SCORING_ESTIMATED_SECONDS_PER_OUTPUT = 60


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def row_key(row):
    return (str(row["example_uid"]), str(row["condition"]), int(row["base_seed"]))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_score(components, knowledge_required=False):
    """Explicit proposed subset aggregate, not an upstream aggregate function."""
    fields = SCORE_FIELDS + (("knowledge_score",) if knowledge_required else ())
    values = [components.get(field) for field in fields]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 1 <= v <= 5 for v in values):
        raise ValueError(f"Missing/invalid official metric(s): {dict(zip(fields, values))}")
    return statistics.mean((value - 1) * 25 for value in values)


def validate_judge(config, calibration=False):
    judge = config["_judge"]
    if judge.get("official_revision") != OFFICIAL_REVISION:
        raise ValueError("This adapter must be reviewed before changing official_revision")
    if judge.get("aggregate") != "mean_applicable_normalized" or not judge.get("aggregate_confirmed"):
        raise ValueError("Coordinate the subset Q definition, then set judge.aggregate_confirmed=true before scoring")
    if judge.get("temperature") != 0 or not re.fullmatch(r"gpt-4o-\d{4}-\d{2}-\d{2}", judge.get("model", "")):
        raise ValueError("The judge needs a pinned GPT-4o snapshot and temperature=0")
    if not calibration:
        stats = config["statistics"]
        margin, mmd = stats.get("equivalence_margin"), stats.get("minimum_meaningful_difference")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (margin, mmd)) or not 0 <= mmd <= margin:
            raise ValueError("Freeze numeric equivalence_margin >= minimum_meaningful_difference >= 0 before test scoring")


def _official_namespace(path, base=None):
    """Compile original functions and literal assignments without CLI side effects."""
    namespace = dict(base or {})
    namespace.update(os=os, json=json, base64=base64, time=time, re=re, logging=logging)
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = []
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            nodes.append(node)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class RecordingClient:
    """Keep every successful raw response and failed attempt before parsing."""
    def __init__(self, client, judge, raw_path):
        self.client, self.judge, self.raw_path = client, judge, raw_path
        self.context = {}
        self.responses = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        kwargs.update(model=self.judge["model"], temperature=0, max_tokens=self.judge["max_tokens"])
        # Upstream always labels bytes JPEG, including PNG files. Correct MIME
        # without re-encoding pixels or altering the official text/image order.
        for message in kwargs["messages"]:
            for item in message["content"]:
                if item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    data = base64.b64decode(url.split(",", 1)[1])
                    if data.startswith(b"\x89PNG\r\n\x1a\n"):
                        item["image_url"]["url"] = url.replace("image/jpeg", "image/png", 1)
        # Record text verbatim and image content hashes, not duplicated base64.
        request = json.loads(json.dumps(kwargs))
        for message in request["messages"]:
            for item in message["content"]:
                if item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    item["image_url"]["url"] = "sha256:" + hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
        record = {**self.context, "request": request, "timestamp": time.time()}
        try:
            response = self.client.chat.completions.create(**kwargs)
            record["response"] = response.model_dump(mode="json")
            self.responses.append(record)
            append_jsonl(self.raw_path, record)
            return response
        except Exception as error:
            record["error"] = repr(error)
            append_jsonl(self.raw_path, record)
            raise


class OfficialEvaluator:
    def __init__(self, judge, client=None, raw_path=None):
        self.root = Path(judge["official_checkout"]).resolve()
        revision = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(self.root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if revision != judge["official_revision"] or dirty:
            raise ValueError("Official evaluator checkout must be clean and match the pinned revision")
        prompts = _official_namespace(self.root / judge["prompt_file"])
        common = _official_namespace(self.root / "metrics_common.py", prompts)
        self.modules = {"common": common}
        for family, filename in (("knowledge", "metrics_knowledge.py"), ("view", "metrics_view_change.py"), ("temporal", "metrics_temporal_prediction.py")):
            self.modules[family] = _official_namespace(self.root / filename, common)
        self.prompt_sha256 = file_sha256(self.root / judge["prompt_file"])
        self.client = RecordingClient(client, judge, raw_path) if client else None

    def family(self, category):
        for family, module in self.modules.items():
            if category in module["CATEGORIES"]:
                return family
        raise ValueError(f"No official evaluator for eligible category {category}")

    def annotation(self, row):
        category, image_id = row["example_uid"].rsplit("__", 1)
        self.family(category)
        path = self.root / "KRIS_Bench" / category / "annotation.json"
        annotation = json.loads(path.read_text(encoding="utf-8"))[image_id]
        if annotation.get("ins_en") != row["instruction"]:
            raise ValueError(f"HF/official annotation instruction mismatch for {row['example_uid']}")
        original = annotation["ori_img"]
        if not isinstance(original, str):
            raise ValueError("Official annotation is not a single-reference example")
        from PIL import Image, ImageOps
        with Image.open(self.root / "KRIS_Bench" / category / original) as handle:
            official = ImageOps.exif_transpose(handle).convert("RGB")
        with Image.open(row["source_path"]) as handle:
            used = ImageOps.exif_transpose(handle).convert("RGB")
        if official.size != used.size or official.tobytes() != used.tobytes():
            raise ValueError(f"HF/generation reference differs from official evaluator reference: {row['example_uid']}")
        output = Path(row["output_path"])
        if not output.is_file():
            raise FileNotFoundError(output)
        return category, image_id, annotation

    def evaluate(self, row):
        category, image_id, annotation = self.annotation(row)
        family = self.family(category)
        module = self.modules[family]
        self.client.context = {key: row[key] for key in ("example_uid", "condition", "base_seed", "protocol_hash")}
        self.client.responses = []
        with tempfile.TemporaryDirectory(prefix="rq3-eval-") as temp:
            out = Path(temp) / "rq3" / category
            out.mkdir(parents=True)
            (out / f"{image_id}.jpg").symlink_to(Path(row["output_path"]).resolve())
            module.update(BENCH_DIR=str(self.root / "KRIS_Bench"), RESULTS_DIR=temp, client=self.client)
            function = module["evaluate_temporal_images" if family == "temporal" else "evaluate_images"]
            components = function("rq3", category, image_id)
        return {**components, "Q": quality_score(components, family == "knowledge"),
                "raw_responses": self.client.responses, "prompt_sha256": self.prompt_sha256,
                "official_family": family}


def score(args):
    from src.rq3.protocol import build_jobs, load_protocol
    config = load_protocol(args.config)
    validate_judge(config, args.calibration)
    rows = [row for row in read_jsonl(args.generations) if row.get("repeat", 0) == 0]
    rows = [row for row in rows if row["base_seed"] in args.seeds]
    allowed = set(config["_splits"]["k_dev" if args.calibration else "k_small"])
    rows = [row for row in rows if row["example_uid"] in allowed]
    if not rows:
        raise ValueError("No eligible quality generations to evaluate")
    if args.limit is not None:
        if not args.calibration:
            raise ValueError("--limit is only for development repeatability scoring")
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        # Select before discarding failed attempts: failures cannot silently
        # change the predetermined repeatability sample.
        keys = set(list(dict.fromkeys(row_key(row) for row in rows))[:args.limit])
        rows = [row for row in rows if row_key(row) in keys]
    failures = [row for row in rows if row.get("status") != "ok"]
    rows = [row for row in rows if row.get("status") == "ok"]
    seen = set()
    for row in rows:
        if row["protocol_hash"] != config["protocol_hash"]:
            raise ValueError("Generation protocol hash differs from the frozen scoring configuration")
        if row.get("status") != "ok" or row_key(row) in seen:
            raise ValueError("Failed or duplicate generation in scoring batch; resolve before scoring")
        seen.add(row_key(row))
        if row.get("output_sha256") != file_sha256(row["output_path"]):
            raise ValueError("Generation output changed after logging")
    unresolved = {row_key(row) for row in failures} - seen
    if unresolved:
        raise ValueError(f"Unresolved generation failures: {sorted(unresolved)[:3]}")
    if not args.calibration:
        expected = {row_key(job) for job in build_jobs(config, phase="all", seeds=args.seeds)}
        if seen != expected:
            raise ValueError(f"Score all RQ3 conditions together: missing={len(expected-seen)}, extra={len(seen-expected)}")
    from src.rq3.protocol import REPO_ROOT
    judge = dict(config["_judge"])
    judge["official_checkout"] = str(REPO_ROOT / judge["official_checkout"])
    evaluator = OfficialEvaluator(judge)
    for row in rows:
        evaluator.annotation(row)
    existing = read_jsonl(args.output) if Path(args.output).exists() else []
    completed = {}
    for row in existing:
        if row["protocol_hash"] != config["protocol_hash"] or row_key(row) in completed:
            raise ValueError("Existing scores contain duplicate keys or a different protocol")
        completed[row_key(row)] = row
    for row in rows:
        previous = completed.get(row_key(row))
        if previous:
            expected_metadata = {"output_sha256": row["output_sha256"],
                                 "judge_model": config["_judge"]["model"],
                                 "aggregate": config["_judge"]["aggregate"],
                                 "prompt_sha256": evaluator.prompt_sha256,
                                 "official_family": evaluator.family(row["example_uid"].rsplit("__", 1)[0])}
            if any(previous.get(key) != value for key, value in expected_metadata.items()):
                raise ValueError("Existing cached score does not match the generation image or frozen evaluator")
            expected_q = quality_score(previous, expected_metadata["official_family"] == "knowledge")
            value = previous.get("Q")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not math.isclose(value, expected_q, abs_tol=1e-10)
                    or not isinstance(previous.get("raw_responses"), list) or len(previous["raw_responses"]) != 3):
                raise ValueError("Existing cached score is incomplete or has an invalid aggregate")
    pending = [row for row in rows if row_key(row) not in completed]
    print(json.dumps({"eligible_outputs": len(rows), "pending_outputs": len(pending), "failed_generation_attempts": len(failures), "execute": args.execute,
                      "prompt_sha256": evaluator.prompt_sha256, "judge": config["_judge"]["model"]}, indent=2))
    if not args.execute:
        return
    done = len(rows) - len(pending)
    with ProgressNotifier("RQ3 judge scoring", len(rows), completed=done,
                          estimated_seconds_per_item=SCORING_ESTIMATED_SECONDS_PER_OUTPUT,
                          enabled=getattr(args, "notify", False), output=args.output) as progress:
        if pending:
            # Avoid requiring an API key when every selected output already has
            # a valid score. Client setup failures also get a terminal status.
            from openai import OpenAI
            evaluator.client = RecordingClient(OpenAI(), judge, str(args.output) + ".raw.jsonl")
        for row in pending:
            try:
                scores = evaluator.evaluate(row)
                append_jsonl(args.output, {**{key: row[key] for key in ("example_uid", "condition", "base_seed", "protocol_hash")},
                                          "output_sha256": file_sha256(row["output_path"]),
                                          "judge_model": config["_judge"]["model"], "aggregate": config["_judge"]["aggregate"],
                                          "knowledge_type": row.get("knowledge_type"), **scores})
            except Exception:
                # Progress counts finished attempts, including this failed one;
                # a future resume still counts only validated saved scores.
                try:
                    progress.update(done + 1, failed=1)
                except Exception:
                    # Notification bookkeeping must preserve the scoring error.
                    logging.warning("Scoring progress could not be updated; retaining the original error")
                raise
            # Advance only after the complete score/raw response record is
            # durably appended; terminal completion follows the final write.
            done += 1
            progress.update(done)


def repeatability(args):
    import numpy as np
    maps = []
    for path in (args.first, args.second):
        rows = read_jsonl(path)
        mapping = {row_key(row): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError("Repeatability files must have unique output keys")
        maps.append(mapping)
    first, second = maps
    if first.keys() != second.keys() or len(first) < 20:
        raise ValueError("Re-score the same predetermined set of at least 20 outputs")
    differences = []
    for key in sorted(first):
        a, b = first[key], second[key]
        for field in ("protocol_hash", "output_sha256", "judge_model", "prompt_sha256", "aggregate"):
            if not a.get(field) or a[field] != b.get(field):
                raise ValueError(f"Repeatability mismatch in {field}: {key}")
        differences.append(abs(a["Q"] - b["Q"]))
    report = {"outputs": len(first), "method": "percentile95_absolute_rescore_Q_difference",
              "candidate_minimum_meaningful_difference": float(np.quantile(differences, .95)),
              "mean_absolute_difference": float(np.mean(differences)),
              "note": "Choose and freeze delta >= this MMD before test scoring; do not tune on test scores."}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("score")
    command.add_argument("--config", default="configs/rq3.yaml")
    command.add_argument("--generations", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--seeds", type=int, nargs="+", default=[42])
    command.add_argument("--execute", action="store_true", help="Send paid judge API requests; default only validates")
    command.add_argument("--calibration", action="store_true", help="Score only K-dev; permit unfrozen delta/MMD")
    command.add_argument("--limit", type=int)
    add_notification_options(command)
    command.set_defaults(function=score)
    command = commands.add_parser("repeatability")
    command.add_argument("--first", required=True)
    command.add_argument("--second", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=repeatability)
    args = parser.parse_args()
    if args.command == "score" and args.execute:
        load_notification_env()
    args.function(args)


if __name__ == "__main__":
    main()
