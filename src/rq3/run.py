"""Run the paired Step1X/RegionE experiment, or inspect its CPU-only plan."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import warnings

from .protocol import build_jobs, load_protocol


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _same_file_bytes(first, second):
    """Compare encoded bytes, including PNG metadata; pixel equality is weaker."""
    try:
        with Path(first).open("rb") as left, Path(second).open("rb") as right:
            if os.fstat(left.fileno()).st_size != os.fstat(right.fileno()).st_size:
                return False
            while True:
                block = left.read(1024 * 1024)
                if block != right.read(1024 * 1024):
                    return False
                if not block:
                    return True
    except OSError:
        # An absent/unreadable candidate never prevents saving a new result.
        return False


def atomic_save_image(image, path, *, reuse_path=None):
    """Save a PNG atomically, sharing an immutable, byte-identical file if possible.

    Generation and PNG encoding always run. A timing repeat may share storage
    with repeat zero only after exact byte comparison; its own path is retained.
    A failed/unsupported hardlink falls back to the newly encoded regular file.
    Never overwrite an existing inode: other repeat paths may link to it.
    """
    path = Path(path)
    temporary, linked = None, None
    try:
        candidate = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        # Exclusive creation preserves the job's umask and cannot truncate an
        # inode retained by a previous attempt or another hardlinked path.
        handle = candidate.open("xb")
        temporary = candidate
        with handle:
            image.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        if reuse_path is not None and _same_file_bytes(temporary, reuse_path):
            link_candidate = temporary.with_suffix(".link")
            try:
                os.link(reuse_path, link_candidate)
            except OSError:
                # EXDEV, permission restrictions, link limits and filesystems
                # without hardlinks still preserve a complete independent PNG.
                pass
            else:
                linked = link_candidate
                linked.replace(path)
                return
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if linked is not None:
            linked.unlink(missing_ok=True)


def seed_for(base_seed, uid, stage):
    """Byte-for-byte equivalent to the shared src.seeding.derive_seed."""
    key = f"{base_seed}|{uid}|{stage}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 1)


def run_key(row):
    return (row["example_uid"], row["condition"], row["base_seed"], row.get("repeat", 0))


def selected_run_keys(jobs, timing_repeats=0):
    """Count only this invocation's selected quality and timing generations."""
    return {(*run_key(job)[:3], repeat) for job in jobs for repeat in range(timing_repeats + 1)}


def read_jsonl_records(path, *, repair_final=False):
    """Read a log; resume may salvage only an unterminated, invalid last line.

    The original bytes are retained beside the log before truncation. A malformed
    earlier line or newline-terminated record always remains a hard error.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records, offset = [], 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            partial_last = index == len(lines) - 1 and not line.endswith(b"\n")
            if not (repair_final and partial_last):
                raise ValueError(f"Malformed JSONL at {path}:{index + 1}; log was not changed") from exc
            backup = path.with_name(path.name + ".truncated-" + hashlib.sha256(raw).hexdigest()[:16] + ".bin")
            if not backup.exists():
                with backup.open("xb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
            elif backup.read_bytes() != raw:
                raise ValueError(f"Recovery backup conflict: {backup}")
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            warnings.warn(f"Removed incomplete final JSONL record; original log preserved at {backup}", RuntimeWarning)
            return records
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object at {path}:{index + 1}")
        records.append(record)
        offset += len(line)
    if repair_final and raw and not raw.endswith(b"\n"):
        # A complete final object is valid; ensure the next append starts a row.
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return records


def read_completed(path, protocol_hash, *, repair_final=False):
    completed = {}
    if not Path(path).exists():
        return completed
    for row in read_jsonl_records(path, repair_final=repair_final):
        if row["protocol_hash"] != protocol_hash:
            raise ValueError("Existing log has a different protocol; use a new output directory")
        if row["status"] == "ok":
            output = Path(row["output_path"])
            if not output.exists() or digest_file(output) != row["output_sha256"]:
                raise ValueError(f"Completed output is missing or changed: {output}")
            source = Path(row["source_path"])
            if not source.exists() or digest_file(source) != row["source_sha256"]:
                raise ValueError(f"Completed source is missing or changed: {source}")
            key = run_key(row)
            if key in completed:
                raise ValueError(f"Duplicate successful generation: {key}")
            completed[key] = row
    return completed


def shard_jobs(jobs, shard_index=0, num_shards=1, *, example_uids=None):
    """Assign sorted IDs round-robin, never splitting a paired example's jobs."""
    if type(num_shards) is not int or num_shards < 1:
        raise ValueError("--num-shards must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index is zero-based and must be smaller than --num-shards")
    universe = sorted(set(example_uids) if example_uids is not None else {job["example_uid"] for job in jobs})
    if num_shards > len(universe):
        raise ValueError("--num-shards cannot exceed the available example count")
    selected = set(universe[shard_index::num_shards])
    result = [job for job in jobs if job["example_uid"] in selected]
    if not result:
        raise ValueError("The requested shard has no jobs after phase/condition selection")
    return result


def select_shard_jobs(config, jobs, shard_index=0, num_shards=1):
    phases = {job["phase"] for job in jobs}
    if len(phases) != 1:
        raise ValueError("A shard must contain one nonempty experiment phase")
    # Using K-small even for rq3_2 keeps membership stable between phases.
    universe = None if phases == {"pilot"} else config["_splits"]["k_small"]
    return shard_jobs(jobs, shard_index, num_shards, example_uids=universe)


def make_shard_plan(config, jobs, shard_index=0, num_shards=1, timing_repeats=0):
    """Runtime layout metadata; deliberately excluded from protocol hashing."""
    if timing_repeats != 0 and timing_repeats < 3:
        raise ValueError("Timing needs zero or at least three repeats")
    return {"schema_version": 1, "protocol_hash": config["protocol_hash"],
            "assignment": "sorted_uid_round_robin_v1", "shard_index": shard_index,
            "num_shards": num_shards, "phase": next(iter({job["phase"] for job in jobs})),
            "seeds": sorted({job["base_seed"] for job in jobs}),
            "example_uids": sorted({job["example_uid"] for job in jobs}),
            "job_keys": sorted([[job["example_uid"], job["condition"], job["base_seed"]] for job in jobs]),
            "timing_repeats": timing_repeats}


def expected_shard_keys(plan):
    return {(*key, repeat) for key in plan["job_keys"] for repeat in range(plan["timing_repeats"] + 1)}


def prepare_shard(output_root, plan, *, resume=False):
    path = output_root / "shard.json"
    plan = dict(plan)
    if path.exists():
        previous = json.loads(path.read_text())
        stable = lambda value: {key: item for key, item in value.items() if key != "timing_repeats"}
        if stable(previous) != stable(plan) or not resume:
            raise ValueError("Shard assignment changed or already exists; use its original parameters with --resume")
        plan["timing_repeats"] = max(plan["timing_repeats"], previous["timing_repeats"])
    elif (output_root / "generations.jsonl").exists():
        raise ValueError("Cannot assign a shard to a pre-existing log without shard metadata")
    atomic_json(path, plan)
    (output_root / "shard_complete.json").unlink(missing_ok=True)
    return plan


def append_row(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def decode_source(example):
    from io import BytesIO
    from PIL import Image, ImageOps
    images = [example.get(k) for k in ("image", "image_1", "image_2", "image_3", "image_4")
              if example.get(k) is not None]
    if len(images) != 1:
        raise ValueError("RQ3 requires exactly one reference image")
    encoded = images[0]
    with Image.open(BytesIO(encoded["bytes"]) if encoded.get("bytes") is not None else encoded["path"]) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_examples(config, uids):
    from datasets import Image, load_dataset
    from src.dataset import ALL_IMAGE_COLUMNS, is_rq1_eligible_example, prepare_example

    data = config["data"]
    dataset = load_dataset(data["dataset_id"], revision=data["revision"], split=data.get("split", "train"))
    for column in ALL_IMAGE_COLUMNS:
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Image(decode=False))
    examples = {}
    for item in dataset:
        uid = f'{item["category"]}__{item["id"]}'
        if uid in uids:
            if not is_rq1_eligible_example(item):
                raise ValueError(f"Ineligible frozen example: {uid}")
            if uid in examples:
                raise ValueError(f"Duplicate dataset UID: {uid}")
            examples[uid] = prepare_example(item)
    if set(examples) != set(uids):
        raise ValueError(f"Dataset is missing IDs: {sorted(set(uids) - set(examples))}")
    return examples


def get_thought(backend, image, example, job, config, output_root, source_hash):
    path = output_root / "thoughts" / f'{job["base_seed"]}' / f'{job["example_uid"]}.json'
    expected = {
        "protocol_hash": config["protocol_hash"], "example_uid": job["example_uid"],
        "base_seed": job["base_seed"], "instruction": example["instruction"],
        "source_sha256": source_hash, "seed": seed_for(job["base_seed"], job["example_uid"], "thinking"),
    }
    if path.exists():
        thought = json.loads(path.read_text())
        if any(thought.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Thought cache identity changed: {path}")
    else:
        thought = {**backend.think(image, example["instruction"], expected["seed"]), **expected}
        if not isinstance(thought.get("rewrite"), str) or not thought["rewrite"].strip():
            raise ValueError("Thinker returned an empty rewrite")
        atomic_json(path, thought)
    return thought, str(path)


def get_timed_thought(backend, image, thought, output_root, repeat):
    """Measure the same seeded thinker once per repeat, shared by B/D/E/F.

    The quality rewrite remains fixed. Timing reruns are additional work, just
    like diffusion timing reruns; a differing rewrite fails the replay check.
    """
    path = output_root / "thought_timings" / str(thought["base_seed"]) / f'{thought["example_uid"]}_{repeat}.json'
    fields = ("protocol_hash", "example_uid", "base_seed", "instruction", "source_sha256", "seed")
    expected = {field: thought[field] for field in fields}
    if path.exists():
        measured = json.loads(path.read_text())
        if any(measured.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Thinking timing cache identity changed: {path}")
    else:
        measured = {**backend.think(image, thought["instruction"], thought["seed"]), **expected}
        if measured["rewrite"] != thought["rewrite"] or measured["raw_text"] != thought["raw_text"]:
            raise ValueError("Thinking changed during a same-seed timing repetition")
        atomic_json(path, measured)
    return measured


def execute(config, jobs, output_root, backend, examples, *, resume=False, timing_repeats=0, shard_plan=None,
            notify=False):
    """Execute jobs with an injectable backend and optional active notifier.

    ``notify`` is an already-entered ProgressNotifier, or False (the default).
    Its caller owns the terminal event so startup failures can share the same
    context, and success is reported only after the shard marker is persisted.
    """
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "merge.json").exists():
        raise ValueError("A merged analysis directory cannot be used for generation")
    manifest_path = output_root / "protocol.json"
    # Private loader metadata may contain Path objects; the public config is JSON-compatible.
    manifest = {k: v for k, v in config.items() if not k.startswith("_")}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["protocol_hash"] != config["protocol_hash"]:
            raise ValueError("Output directory is bound to another protocol")
        if not resume:
            raise ValueError("Output exists; use --resume or a fresh output directory")
    else:
        atomic_json(manifest_path, manifest)
    if hasattr(backend, "runtime_metadata"):
        runtime_path = output_root / "runtime.json"
        runtime = backend.runtime_metadata
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
            raise ValueError("Runtime libraries or GPU changed; use a new run directory and repeat pilots")
        if not runtime_path.exists():
            atomic_json(runtime_path, runtime)
    if "_splits" in config:
        from .protocol import protocol_manifest
        frozen = protocol_manifest(config)
        snapshot = output_root / "protocol_snapshot.json"
        if snapshot.exists() and json.loads(snapshot.read_text()) != frozen:
            raise ValueError("Frozen protocol snapshot changed")
        if not snapshot.exists():
            atomic_json(snapshot, frozen)
    log_path = output_root / "generations.jsonl"
    if shard_plan is not None:
        shard_plan = prepare_shard(output_root, shard_plan, resume=resume)
        timing_repeats = shard_plan["timing_repeats"]
    elif (output_root / "shard.json").exists():
        raise ValueError("Resume this output with its original --shard-index and --num-shards")
    completed = read_completed(log_path, config["protocol_hash"], repair_final=resume)
    if shard_plan is not None and not set(completed) <= expected_shard_keys(shard_plan):
        raise ValueError("Completed log contains jobs outside the declared shard")
    progress_completed = len(selected_run_keys(jobs, timing_repeats) & set(completed))
    failures = 0
    for job in jobs:
        uid, condition, base_seed = job["example_uid"], job["condition"], job["base_seed"]
        for repeat in range(timing_repeats + 1):
            identity = {"example_uid": uid, "condition": condition, "base_seed": base_seed, "repeat": repeat}
            if run_key(identity) in completed:
                continue
            row = {**identity, "protocol_hash": config["protocol_hash"], "phase": job.get("phase"),
                   "model": config["model"]["model_id"], "config": manifest,
                   "stage_seeds": {stage: seed_for(base_seed, uid, stage) for stage in ("thinking", "diffusion_0")}}
            try:
                start = time.perf_counter()
                example = examples[uid]
                image = decode_source(example)
                preprocess_seconds = time.perf_counter() - start
                source_path = output_root / "sources" / f"{uid}.png"
                source_path.parent.mkdir(exist_ok=True)
                if not source_path.exists():
                    atomic_save_image(image, source_path)
                else:
                    from PIL import Image
                    with Image.open(source_path) as saved:
                        if saved.size != image.size or saved.convert("RGB").tobytes() != image.tobytes():
                            raise ValueError(f"Source pixels changed for frozen ID: {uid}")
                source_hash = digest_file(source_path)
                thought, thought_path = None, None
                if condition in ("B", "D", "E", "F"):
                    thought, thought_path = get_thought(backend, image, example, job, config, output_root, source_hash)
                charged_thought = get_timed_thought(backend, image, thought, output_root, repeat) if thought and repeat else thought
                prompt = thought["rewrite"] if thought else example["instruction"]
                result = backend.generate(image, prompt, row["stage_seeds"]["diffusion_0"], condition)
                output_path = output_root / "images" / f"seed{base_seed}" / condition / f"{uid}_repeat{repeat}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                repeat_zero_path = output_path.with_name(f"{uid}_repeat0.png") if repeat else None
                atomic_save_image(result["image"], output_path, reuse_path=repeat_zero_path)
                timings = dict(result["timings"])
                timings["preprocess_seconds"] = timings.get("preprocess_seconds", 0.0) + preprocess_seconds
                timings["thinking_seconds"] = charged_thought["seconds"] if charged_thought else 0.0
                timings["thinking_offload_seconds"] = charged_thought.get("offload_seconds", 0.0) if charged_thought else 0.0
                # Backend measures the replay invocation directly; charge one cached thought separately.
                timings["measured_replay_seconds"] = result["timings"]["measured_replay_seconds"] + preprocess_seconds
                timings["charged_total_seconds"] = (timings["measured_replay_seconds"] + timings["thinking_seconds"]
                                                    + timings["thinking_offload_seconds"])
                row.update(status="ok", output_path=str(output_path), output_sha256=digest_file(output_path),
                           source_path=str(source_path), source_sha256=source_hash, instruction=example["instruction"],
                           knowledge_type=example["knowledge_type"], category=example["category"],
                           prompt=prompt, thought=thought, thought_path=thought_path, timings=timings,
                           peak_vram_bytes=max(result["peak_vram_bytes"], (charged_thought or {}).get("peak_vram_bytes", 0)),
                           initial_noise_sha256=result.get("initial_noise_sha256"),
                           preprocessing=result.get("preprocessing"),
                           regione=result["regione"], candidate_paths=[str(output_path)], selected_candidate_index=0,
                           attempt_count=1, retry_triggered=False, stop_reason="single_pass", reflection=None)
                completed[run_key(row)] = row
            except Exception as exc:
                failures += 1
                row.update(status="error", error_type=type(exc).__name__, error=str(exc))
                append_row(log_path, row)
                if notify:
                    # This attempt has terminated, but resume will count only
                    # successful artifacts and retry the failed generation.
                    notify.update(progress_completed + 1, failed=failures)
                raise
            append_row(log_path, row)
            progress_completed += 1
            if notify:
                notify.update(progress_completed, failed=failures)
            print(f"{condition} {uid} seed={base_seed} repeat={repeat}: saved", flush=True)
    if shard_plan is not None:
        if set(completed) != expected_shard_keys(shard_plan):
            raise ValueError("Shard execution ended without all declared jobs and repetitions")
        atomic_json(output_root / "shard_complete.json", {
            "schema_version": 1, "protocol_hash": config["protocol_hash"],
            "shard_index": shard_plan["shard_index"], "num_shards": shard_plan["num_shards"],
            "completed_generations": len(completed),
            "shard_sha256": digest_file(output_root / "shard.json"),
            "generations_sha256": digest_file(log_path),
        })
    return {"completed": len(completed), "failures": failures}


def main():
    from src.notifications import add_notification_options

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rq3.yaml")
    parser.add_argument("--phase", choices=["all", "rq3_1", "rq3_2", "pilot"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--conditions", nargs="+", choices=list("ABCDEF"))
    parser.add_argument("--limit", type=int, help="Limit pilot examples only")
    parser.add_argument("--output", default="outputs/rq3/main")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based index for example-level parallel generation")
    parser.add_argument("--num-shards", type=int, default=1, help="Total example shards; does not change protocol or seeds")
    parser.add_argument("--prepared-data", default=os.environ.get("RQ3_PREPARED_DATA"), help="Prepared offline data directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timing-repeats", type=int, default=0, help="Additional unscored runs; use >=3 for timing")
    add_notification_options(parser)
    args = parser.parse_args()
    if args.limit is not None and (args.phase != "pilot" or args.limit < 1):
        parser.error("--limit is a positive pilot-only option; test membership is frozen")
    if args.timing_repeats not in (0,) and args.timing_repeats < 3:
        parser.error("Timing needs at least three repeats")
    if args.num_shards > 1 and (args.phase != "all" or args.conditions or args.limit is not None):
        parser.error("Parallel shards require --phase all with every condition and the full frozen example set")
    config = load_protocol(args.config)
    jobs = build_jobs(config, phase=args.phase, seeds=args.seeds)
    if args.limit:
        selected = list(dict.fromkeys(j["example_uid"] for j in jobs))[:args.limit]
        jobs = [j for j in jobs if j["example_uid"] in selected]
    if args.conditions:
        jobs = [j for j in jobs if j["condition"] in args.conditions]
    try:
        jobs = select_shard_jobs(config, jobs, args.shard_index, args.num_shards)
    except ValueError as exc:
        parser.error(str(exc))
    shard_plan = make_shard_plan(config, jobs, args.shard_index, args.num_shards, args.timing_repeats)
    if args.dry_run:
        from collections import Counter
        print(json.dumps({"protocol_hash": config["protocol_hash"], "generations": len(jobs),
                          "conditions": dict(Counter(j["condition"] for j in jobs)),
                          "additional_timing_generations": len(jobs) * args.timing_repeats,
                          "shard": shard_plan,
                          "jobs": jobs}, indent=2))
        return
    from src.notifications import ProgressNotifier, load_notification_env
    import fcntl
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # A lock prevents two processes from corrupting one experiment's cache/log.
    with (output_root / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        effective_repeats = args.timing_repeats
        existing_shard = output_root / "shard.json"
        if args.resume and args.num_shards > 1 and existing_shard.exists():
            # prepare_shard preserves the larger timing plan on resume. The
            # progress denominator must match that effective workload too.
            effective_repeats = max(effective_repeats, json.loads(existing_shard.read_text())["timing_repeats"])
        selected_keys = selected_run_keys(jobs, effective_repeats)
        cached = read_completed(output_root / "generations.jsonl", config["protocol_hash"],
                                repair_final=True) if args.resume else {}
        initial = len(selected_keys & set(cached))
        load_notification_env()
        label = f"RQ3 {args.phase} generation"
        if args.num_shards > 1:
            label += f" (shard {args.shard_index + 1}/{args.num_shards})"
        with ProgressNotifier(label, len(selected_keys), completed=initial, enabled=args.notify,
                              output=output_root) as progress:
            from .protocol import validate_for_execution
            validate_for_execution(config, phase=args.phase)
            from .backend import Step1XBackend
            warmup_uid = config["_splits"]["k_dev"][0]
            requested_uids = {j["example_uid"] for j in jobs} | {warmup_uid}
            if args.prepared_data:
                from .prepare import load_prepared_examples
                examples = load_prepared_examples(config, args.prepared_data, requested_uids)
            else:
                examples = load_examples(config, requested_uids)
            backend = Step1XBackend(config)
            warmup = examples[warmup_uid]
            image = decode_source(warmup)
            thought = backend.think(image, warmup["instruction"], seed_for(42, warmup_uid, "thinking"))
            for condition in sorted({j["condition"] for j in jobs}):
                backend.generate(image, thought["rewrite"] if condition in "BDEF" else warmup["instruction"],
                                 seed_for(42, warmup_uid, "diffusion_0"), condition)
            print(json.dumps(execute(config, jobs, output_root, backend, examples,
                                     resume=args.resume, timing_repeats=args.timing_repeats,
                                     shard_plan=shard_plan if args.num_shards > 1 else None,
                                     notify=progress)))


if __name__ == "__main__":
    main()
