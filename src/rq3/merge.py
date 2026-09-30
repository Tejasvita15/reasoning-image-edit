"""Validate and merge complete example shards without copying images or caches."""

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import tempfile

from .protocol import build_jobs, load_protocol, protocol_manifest
from .run import (atomic_json, digest_file, expected_shard_keys, make_shard_plan,
                  read_completed, read_jsonl_records, run_key, seed_for, select_shard_jobs)


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing required shard artifact: {path}")
    return json.loads(path.read_text())


def _check_pairing(rows):
    pairs, sources = {}, {}
    for row in rows:
        uid, seed = row["example_uid"], row["base_seed"]
        if row["stage_seeds"] != {stage: seed_for(seed, uid, stage) for stage in ("thinking", "diffusion_0")}:
            raise ValueError(f"Incorrect derived seeds for {run_key(row)}")
        noise = row.get("initial_noise_sha256")
        if not isinstance(noise, str) or re.fullmatch(r"[0-9a-f]{64}", noise) is None:
            raise ValueError(f"Missing actual initial-noise hash for {run_key(row)}")
        pair = pairs.setdefault((uid, seed), {"noise": noise, "instruction": row["instruction"], "thought": None})
        if pair["noise"] != noise or pair["instruction"] != row["instruction"]:
            raise ValueError(f"Noise or instruction differs between paired conditions/repeats for {uid}, seed {seed}")
        previous_source = sources.setdefault(uid, row["source_sha256"])
        if previous_source != row["source_sha256"]:
            raise ValueError(f"Source image differs between paired generations for {uid}")
        for name in ("source_path", "output_path"):
            if not Path(row[name]).is_absolute():
                raise ValueError(f"Merged artifact references must be absolute: {row[name]}")
        if row["condition"] in "BDEF":
            thought = row.get("thought")
            if (not isinstance(thought, dict) or not isinstance(thought.get("raw_text"), str)
                    or not thought["raw_text"].strip() or not isinstance(thought.get("rewrite"), str)
                    or not thought["rewrite"].strip() or row["prompt"] != thought["rewrite"]
                    or type(thought.get("token_count")) is not int or thought["token_count"] <= 0):
                raise ValueError(f"Missing valid cached thinking for {run_key(row)}")
            identity = {"protocol_hash": row["protocol_hash"], "example_uid": uid, "base_seed": seed,
                        "instruction": row["instruction"], "source_sha256": row["source_sha256"],
                        "seed": row["stage_seeds"]["thinking"]}
            if any(thought.get(key) != value for key, value in identity.items()):
                raise ValueError(f"Cached thought identity differs for {run_key(row)}")
            cached = _read(row["thought_path"])
            if cached != thought:
                raise ValueError(f"Cached thought file changed: {row['thought_path']}")
            thought_pair = (thought["raw_text"], thought["rewrite"])
            if pair["thought"] is not None and pair["thought"] != thought_pair:
                raise ValueError(f"Thinking differs between paired conditions/repeats for {uid}, seed {seed}")
            pair["thought"] = thought_pair
        elif row.get("thought") is not None or row["prompt"] != row["instruction"]:
            raise ValueError(f"Direct editing must use the original instruction for {run_key(row)}")


def merge_shards(config, shards, output, *, num_shards, seeds=None, timing_repeats=0):
    """Publish only a full, compatible all-phase experiment.

    Original artifacts stay in their shard directories. Keep these directories
    available for evaluation and analysis; the merged JSONL references them.
    """
    seeds = [42] if seeds is None else seeds
    all_jobs = build_jobs(config, phase="all", seeds=seeds)
    if type(num_shards) is not int or not 1 <= num_shards <= len(config["_splits"]["k_small"]):
        raise ValueError("num_shards must be between 1 and the K-small example count")
    if type(timing_repeats) is not int or timing_repeats != 0 and timing_repeats < 3:
        raise ValueError("timing_repeats must be zero or at least three")
    paths = [Path(path).resolve() for path in shards]
    if len(paths) != num_shards or len(set(paths)) != num_shards:
        raise ValueError(f"Provide exactly {num_shards} distinct shard directories")
    output = Path(output).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Merged output is not a directory: {output}")
    if any(output == path or path in output.parents or output in path.parents for path in paths):
        raise ValueError("Merged output must be separate from every source shard")
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    snapshot = protocol_manifest(config)
    merged, metadata, indices, uid_owner = {}, [], set(), {}
    runtime = None
    with ExitStack() as stack:
        for path in paths:
            lock_path = path / ".run.lock"
            if lock_path.exists():
                lock = stack.enter_context(lock_path.open("r"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError(f"Shard is still running: {path}") from exc
            plan = _read(path / "shard.json")
            completion = _read(path / "shard_complete.json")
            index = plan.get("shard_index")
            if type(index) is not int or not 0 <= index < num_shards or index in indices:
                raise ValueError(f"Duplicate or invalid shard index in {path}")
            indices.add(index)
            expected_jobs = select_shard_jobs(config, all_jobs, index, num_shards)
            expected_plan = make_shard_plan(config, expected_jobs, index, num_shards, timing_repeats)
            if plan != expected_plan:
                raise ValueError(f"Shard plan differs from complete requested experiment: {path}")
            expected_keys = expected_shard_keys(expected_plan)
            expected_complete = {
                "schema_version": 1, "protocol_hash": config["protocol_hash"],
                "shard_index": index, "num_shards": num_shards,
                "completed_generations": len(expected_keys),
                "shard_sha256": digest_file(path / "shard.json"),
                "generations_sha256": digest_file(path / "generations.jsonl"),
            }
            if completion != expected_complete:
                raise ValueError(f"Shard completion marker is stale or incomplete: {path}")
            if _read(path / "protocol.json") != public or _read(path / "protocol_snapshot.json") != snapshot:
                raise ValueError(f"Protocol/configuration differs in shard: {path}")
            shard_runtime = _read(path / "runtime.json")
            runtime_fields = {"torch", "cuda", "device_name", "offload_mode", "attention_backend", "deterministic_algorithms"}
            if not isinstance(shard_runtime, dict) or not runtime_fields <= set(shard_runtime):
                raise ValueError(f"Runtime metadata is incomplete in shard: {path}")
            if runtime is None:
                runtime = shard_runtime
            elif runtime != shard_runtime:
                raise ValueError(f"Runtime libraries, platform or GPU differ across shards: {path}")
            rows = read_jsonl_records(path / "generations.jsonl")
            latest = {}
            for row in rows:
                key = run_key(row)
                if (key not in expected_keys or row.get("phase") != "all"
                        or row.get("protocol_hash") != config["protocol_hash"] or row.get("config") != public
                        or row.get("status") not in {"ok", "error"}):
                    raise ValueError(f"Unexpected generation metadata in shard {path}: {key}")
                latest[key] = row["status"]
            if set(latest) != expected_keys or any(status != "ok" for status in latest.values()):
                raise ValueError(f"Shard has missing generations or unresolved failures: {path}")
            completed = read_completed(path / "generations.jsonl", config["protocol_hash"])
            if set(completed) != expected_keys:
                raise ValueError(f"Incomplete or extra successful generations in shard: {path}")
            for uid in plan["example_uids"]:
                if uid in uid_owner:
                    raise ValueError(f"Example appears in more than one shard: {uid}")
                uid_owner[uid] = index
            if set(merged) & set(completed):
                raise ValueError("Duplicate generation identities across shards")
            _check_pairing(completed.values())
            merged.update(completed)
            metadata.append({"shard_index": index, "directory": str(path),
                             "generations_sha256": completion["generations_sha256"],
                             "completed_generations": len(completed),
                             "resolved_failure_records": sum(row["status"] == "error" for row in rows)})
        expected_all = {(*run_key(job)[:3], repeat) for job in all_jobs for repeat in range(timing_repeats + 1)}
        if indices != set(range(num_shards)) or set(merged) != expected_all:
            raise ValueError("Merged shards do not cover the complete frozen RQ3 experiment")
        order = [(*run_key(job)[:3], repeat) for job in all_jobs for repeat in range(timing_repeats + 1)]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.merge-", dir=output.parent) as temporary:
            staged = Path(temporary)
            atomic_json(staged / "protocol.json", public)
            atomic_json(staged / "protocol_snapshot.json", snapshot)
            atomic_json(staged / "runtime.json", runtime)
            with (staged / "generations.jsonl").open("w") as handle:
                for key in order:
                    handle.write(json.dumps(merged[key], sort_keys=True) + "\n")
            report = {"schema_version": 1, "protocol_hash": config["protocol_hash"],
                      "num_shards": num_shards, "seeds": sorted(seeds), "timing_repeats": timing_repeats,
                      "completed_generations": len(merged), "example_count": len(uid_owner),
                      "shards": sorted(metadata, key=lambda value: value["shard_index"]),
                      "generations_sha256": digest_file(staged / "generations.jsonl"),
                      "artifact_policy": "References original absolute paths; retain every source shard directory"}
            atomic_json(staged / "merge.json", report)
            if output.exists():
                same = all((output / name).is_file() and _read(output / name) == _read(staged / name)
                           for name in ("protocol.json", "protocol_snapshot.json", "runtime.json", "merge.json"))
                same = (same and (output / "generations.jsonl").is_file()
                        and digest_file(output / "generations.jsonl") == report["generations_sha256"])
                if not same:
                    raise ValueError(f"Merged output differs; use a fresh directory to preserve existing scores: {output}")
                # Existing scores/analysis are preserved on an identical rerun.
                return report
            staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rq3.yaml")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = merge_shards(load_protocol(args.config), args.shards, args.output,
                          num_shards=args.num_shards, seeds=args.seeds, timing_repeats=args.timing_repeats)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
