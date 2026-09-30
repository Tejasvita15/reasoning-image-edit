from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from src.rq3.merge import merge_shards
from src.rq3.protocol import build_jobs, load_protocol
from src.rq3.run import (atomic_json, atomic_save_image, digest_file, execute, main,
                         make_shard_plan, read_jsonl_records, select_shard_jobs, shard_jobs)


class ShardBackend:
    runtime_metadata = {"torch": "test", "cuda": "test", "device_name": "test GPU",
                        "offload_mode": "model", "attention_backend": "pytorch_sdpa",
                        "deterministic_algorithms": True, "platform": "same-kernel"}

    def think(self, image, instruction, seed):
        return {"rewrite": "frozen rewrite", "raw_text": "frozen native thought", "token_count": 3,
                "seconds": 1.0, "offload_seconds": 0.1, "peak_vram_bytes": 200}

    def generate(self, image, prompt, seed, condition):
        return {"image": image.copy(), "timings": {"measured_replay_seconds": 1.0, "diffusion_seconds": 0.7},
                "peak_vram_bytes": 100, "regione": {},
                "initial_noise_sha256": hashlib.sha256(str(seed).encode()).hexdigest()}


def complete_shards(tmp_path, timing_repeats=0):
    config = load_protocol()
    jobs = build_jobs(config)
    raw = BytesIO()
    Image.new("RGB", (2, 2), "red").save(raw, format="PNG")
    examples = {job["example_uid"]: {"instruction": "make the requested edit",
                "knowledge_type": job["knowledge_type"], "category": job["example_uid"].split("__")[0],
                "image": {"bytes": raw.getvalue()}} for job in jobs}
    shards = []
    for index in range(2):
        selected = select_shard_jobs(config, jobs, index, 2)
        plan = make_shard_plan(config, selected, index, 2, timing_repeats)
        path = tmp_path / f"shard_{index}"
        execute(config, selected, path, ShardBackend(), examples, timing_repeats=timing_repeats, shard_plan=plan)
        shards.append(path)
    return config, shards


def rewrite_shard_log(path, rows):
    log = path / "generations.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    marker = json.loads((path / "shard_complete.json").read_text())
    marker["generations_sha256"] = digest_file(log)
    atomic_json(path / "shard_complete.json", marker)


def test_sharding_preserves_every_condition_and_seed_for_an_example():
    config = load_protocol()
    jobs = build_jobs(config, seeds=[42, 1234, 7])
    shards = [shard_jobs(jobs, index, 4) for index in range(4)]
    assert sum(map(len, shards)) == 600
    assert all(len({job["example_uid"] for job in selected}) == 10 for selected in shards)
    owners = {}
    for index, selected in enumerate(shards):
        for uid in {job["example_uid"] for job in selected}:
            assert uid not in owners
            owners[uid] = index
            assert [job for job in selected if job["example_uid"] == uid] == [job for job in jobs if job["example_uid"] == uid]
    assert shards == [shard_jobs(jobs, index, 4) for index in range(4)]
    assert len(owners) == 40


@pytest.mark.parametrize("index,count", [(-1, 4), (4, 4), (0, 0), (0, 41)])
def test_invalid_shard_parameters_rejected(index, count):
    with pytest.raises(ValueError):
        shard_jobs(build_jobs(load_protocol()), index, count)


def test_dry_run_reports_shard_without_changing_protocol(monkeypatch, capsys):
    config = load_protocol()
    monkeypatch.setattr("sys.argv", ["run", "--dry-run", "--shard-index", "2", "--num-shards", "4"])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["protocol_hash"] == config["protocol_hash"]
    assert result["shard"]["num_shards"] == 4
    assert result["shard"]["shard_index"] == 2
    assert len(result["shard"]["example_uids"]) == 10


def test_merge_complete_shards_retains_original_artifact_paths(tmp_path):
    config, shards = complete_shards(tmp_path)
    output = tmp_path / "merged"
    report = merge_shards(config, list(reversed(shards)), output, num_shards=2)
    assert report["completed_generations"] == 200
    assert report["example_count"] == 40
    rows = read_jsonl_records(output / "generations.jsonl")
    assert len(rows) == 200
    assert all(Path(row["output_path"]).is_file() for row in rows)
    assert all(any(path in Path(row["output_path"]).parents for path in shards) for row in rows)
    assert not (output / "images").exists()
    (output / "scores.jsonl").write_text("preserve scores")
    assert merge_shards(config, shards, output, num_shards=2) == report
    assert (output / "scores.jsonl").read_text() == "preserve scores"
    (output / "generations.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Merged output differs"):
        merge_shards(config, shards, output, num_shards=2)


def test_merge_rejects_missing_shard_or_completion_marker(tmp_path):
    config, shards = complete_shards(tmp_path)
    with pytest.raises(ValueError, match="exactly 2"):
        merge_shards(config, shards[:1], tmp_path / "merged", num_shards=2)
    (shards[0] / "shard_complete.json").unlink()
    with pytest.raises(ValueError, match="Missing required shard artifact"):
        merge_shards(config, shards, tmp_path / "merged", num_shards=2)


def test_merge_rejects_missing_generation_even_if_log_checksum_updated(tmp_path):
    config, shards = complete_shards(tmp_path)
    rows = read_jsonl_records(shards[0] / "generations.jsonl")
    rewrite_shard_log(shards[0], rows[:-1])
    with pytest.raises(ValueError, match="missing generations"):
        merge_shards(config, shards, tmp_path / "merged", num_shards=2)
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_unresolved_failed_generation(tmp_path):
    config, shards = complete_shards(tmp_path)
    rows = read_jsonl_records(shards[0] / "generations.jsonl")
    rows[-1]["status"] = "error"
    rewrite_shard_log(shards[0], rows)
    with pytest.raises(ValueError, match="unresolved failures"):
        merge_shards(config, shards, tmp_path / "merged", num_shards=2)


def test_merge_rejects_different_runtime(tmp_path):
    config, shards = complete_shards(tmp_path)
    runtime = json.loads((shards[1] / "runtime.json").read_text())
    runtime["device_name"] = "different GPU"
    atomic_json(shards[1] / "runtime.json", runtime)
    with pytest.raises(ValueError, match="differ across shards"):
        merge_shards(config, shards, tmp_path / "merged", num_shards=2)


def test_merge_rejects_duplicate_shard_index(tmp_path):
    config, shards = complete_shards(tmp_path)
    plan = json.loads((shards[1] / "shard.json").read_text())
    plan["shard_index"] = 0
    atomic_json(shards[1] / "shard.json", plan)
    with pytest.raises(ValueError, match="Duplicate or invalid shard index"):
        merge_shards(config, shards, tmp_path / "merged", num_shards=2)


def test_merge_rejects_paired_noise_mismatch(tmp_path):
    config, shards = complete_shards(tmp_path)
    rows = read_jsonl_records(shards[0] / "generations.jsonl")
    next(row for row in rows if row["condition"] == "D")["initial_noise_sha256"] = "a" * 64
    rewrite_shard_log(shards[0], rows)
    with pytest.raises(ValueError, match="Noise or instruction differs"):
        merge_shards(config, shards, tmp_path / "merged", num_shards=2)


def test_merge_requires_exact_timing_repeats(tmp_path):
    config, shards = complete_shards(tmp_path, timing_repeats=3)
    with pytest.raises(ValueError, match="Shard plan differs"):
        merge_shards(config, shards, tmp_path / "wrong", num_shards=2, timing_repeats=0)
    report = merge_shards(config, shards, tmp_path / "merged", num_shards=2, timing_repeats=3)
    assert report["completed_generations"] == 800


def test_quality_shards_resume_to_timing_after_truncated_append(tmp_path):
    config, shards = complete_shards(tmp_path)
    jobs = build_jobs(config)
    for index, path in enumerate(shards):
        quality_rows = read_jsonl_records(path / "generations.jsonl")
        examples = {row["example_uid"]: {"instruction": row["instruction"],
                    "knowledge_type": row["knowledge_type"], "category": row["category"],
                    "image": {"path": row["source_path"]}} for row in quality_rows}
        selected = select_shard_jobs(config, jobs, index, 2)
        plan = make_shard_plan(config, selected, index, 2, timing_repeats=3)
        with (path / "generations.jsonl").open("ab") as handle:
            handle.write(b'{"status": "unfinished')
        with pytest.warns(RuntimeWarning, match="original log preserved"):
            execute(config, selected, path, ShardBackend(), examples, resume=True,
                    timing_repeats=3, shard_plan=plan)
        rows = read_jsonl_records(path / "generations.jsonl")
        assert [row for row in rows if row["repeat"] == 0] == quality_rows
        assert len(rows) == 4 * len(quality_rows)
        assert json.loads((path / "shard.json").read_text())["timing_repeats"] == 3
    report = merge_shards(config, shards, tmp_path / "merged", num_shards=2, timing_repeats=3)
    assert report["completed_generations"] == 800


def test_resume_only_repairs_unterminated_invalid_final_record(tmp_path):
    log = tmp_path / "generations.jsonl"
    original = b'{"status":"ok"}\n{"status": "err'
    log.write_bytes(original)
    with pytest.raises(ValueError, match="Malformed JSONL"):
        read_jsonl_records(log)
    with pytest.warns(RuntimeWarning, match="original log preserved"):
        assert read_jsonl_records(log, repair_final=True) == [{"status": "ok"}]
    assert log.read_bytes() == b'{"status":"ok"}\n'
    backups = list(tmp_path.glob("*.truncated-*.bin"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


@pytest.mark.parametrize("raw", [b'{bad}\n{"valid":true}\n', b'{"valid":true}\n{bad}\n'])
def test_resume_never_swallows_earlier_or_terminated_corruption(tmp_path, raw):
    path = tmp_path / "log.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="log was not changed"):
        read_jsonl_records(path, repair_final=True)
    assert path.read_bytes() == raw


def test_complete_final_json_without_newline_is_kept_before_append(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"valid":true}')
    assert read_jsonl_records(path, repair_final=True) == [{"valid": True}]
    assert path.read_bytes() == b'{"valid":true}\n'


def test_atomic_image_save_preserves_source_if_write_is_interrupted(tmp_path):
    path = tmp_path / "source.png"
    atomic_save_image(Image.new("RGB", (2, 2), "red"), path)
    original = path.read_bytes()

    class Interrupted:
        def save(self, handle, format):
            handle.write(b"partial")
            raise RuntimeError("walltime kill")

    with pytest.raises(RuntimeError, match="walltime kill"):
        atomic_save_image(Interrupted(), path)
    assert path.read_bytes() == original
    atomic_save_image(Image.new("RGB", (2, 2), "blue"), path)
    assert path.read_bytes() != original
