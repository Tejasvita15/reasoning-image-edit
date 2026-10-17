from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.rq3.prepare import (
    MODEL_COMPONENTS, MODEL_ID, MODEL_REVISION, MODEL_SNAPSHOT_FILES,
    _model_inventory, download_model_snapshot,
    atomic_json, canonical_hash, data_identity, expected_uids,
    load_prepared_examples, materialize_dataset, preparation_lock,
)


def fixture_config():
    return {"model": {"model_id": "test/model", "revision": "a" * 40},
            "data": {"dataset_id": "test/dataset", "revision": "b" * 40, "split": "train"},
            "preprocessing": {"mode": "step1x_native", "size_level": 1024, "exif_transpose": True, "color_mode": "RGB"},
            "_splits": {"k_dev": ["biology__1"], "k_test": ["count_change__2"], "k_small": ["count_change__2"]},
            "_subset": {"example_uids": ["count_change__2"]}}


def fixture_dataset():
    from PIL import Image
    buffer = BytesIO()
    Image.new("RGB", (5, 3), "green").save(buffer, format="PNG")
    encoded = {"bytes": buffer.getvalue(), "path": None}
    return [{"id": "1", "category": "biology", "instruction": "Make it flower", "image_1": encoded},
            {"id": "2", "category": "count_change", "instruction": "Add one object", "image": encoded,
             "gt_image": encoded}]


def write_fixture(config, output, dataset=None):
    records = materialize_dataset(config, output, fixture_dataset() if dataset is None else dataset)
    identity = data_identity(config)
    manifest = {"identity": identity, "data_identity": canonical_hash(identity),
                "examples_sha256": canonical_hash(records["examples"]), **records}
    atomic_json(Path(output) / "manifest.json", manifest)
    return manifest


def fixture_model_snapshot(directory):
    """Tiny files with the real component/shard layout; no weights/downloads."""
    root = Path(directory)
    for relative in MODEL_SNAPSHOT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (root / "model_index.json").write_text(json.dumps(MODEL_COMPONENTS))
    for relative in ("text_encoder/model.safetensors.index.json", "transformer/diffusion_pytorch_model.safetensors.index.json"):
        path = root / relative
        shards = sorted(name for name in MODEL_SNAPSHOT_FILES if name.startswith(path.parent.name + "/") and name.endswith(".safetensors"))
        path.write_text(json.dumps({"weight_map": {f"tensor_{index}": Path(shard).name for index, shard in enumerate(shards)}}))
    return {relative: (root / relative).stat().st_size for relative in MODEL_SNAPSHOT_FILES}


class PreparationTests(unittest.TestCase):
    def test_round_trip_all_split_union_without_hugging_face_or_gpu(self):
        config = fixture_config()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_fixture(config, directory)
            examples = load_prepared_examples(config, directory, expected_uids(config))
            self.assertEqual(set(examples), {"biology__1", "count_change__2"})
            self.assertEqual(manifest["dataset_rows"], 2)
            self.assertIsNone(examples["biology__1"]["image_1"])
            self.assertTrue(Path(examples["biology__1"]["image"]["path"]).is_file())
            self.assertTrue(Path(examples["count_change__2"]["gt_image"]["path"]).is_file())
            # Margin/judge freeze changes do not rematerialize unchanged data.
            changed = deepcopy(config)
            changed["statistics"] = {"equivalence_margin": 3, "minimum_meaningful_difference": 1}
            self.assertEqual(data_identity(config), data_identity(changed))

    def test_missing_duplicate_and_multiple_reference_ids_fail(self):
        config = fixture_config()
        for dataset, message in ((fixture_dataset()[:1], "missing frozen"),
                                 (fixture_dataset() + fixture_dataset()[:1], "Duplicate frozen")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    materialize_dataset(config, directory, dataset)
        dataset = fixture_dataset()
        dataset[0]["image"] = dataset[0]["image_1"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly one source"):
                materialize_dataset(config, directory, dataset)

    def test_image_and_manifest_corruption_rejected(self):
        config = fixture_config()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_fixture(config, directory)
            source = Path(directory) / manifest["examples"]["biology__1"]["source_path"]
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                load_prepared_examples(config, directory, ["biology__1"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_fixture(config, directory)
            manifest["examples"]["biology__1"]["instruction"] = "Changed instruction"
            atomic_json(Path(directory) / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                load_prepared_examples(config, directory, ["biology__1"])

    def test_data_revision_and_unsafe_paths_rejected(self):
        config = fixture_config()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_fixture(config, directory)
            changed = deepcopy(config)
            changed["data"]["revision"] = "c" * 40
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_prepared_examples(changed, directory, ["biology__1"])
            manifest["examples"]["biology__1"]["source_path"] = "../outside.png"
            manifest["examples_sha256"] = canonical_hash(manifest["examples"])
            atomic_json(Path(directory) / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "outside its data directory"):
                load_prepared_examples(config, directory, ["biology__1"])

    def test_preparation_lock_refuses_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with preparation_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    with preparation_lock(directory):
                        self.fail("Concurrent writer acquired the same lock")
            with preparation_lock(directory):
                pass

    def test_snapshot_filter_is_shared_for_online_and_offline_loading(self):
        model = {"model_id": MODEL_ID, "revision": MODEL_REVISION}
        with tempfile.TemporaryDirectory() as directory:
            inventory = fixture_model_snapshot(directory)
            extra = Path(directory) / "unused_monolithic.safetensors"
            extra.write_bytes(b"pre-existing file; do not delete")
            download = Mock(return_value=directory)
            with patch.dict("src.rq3.prepare.MODEL_SNAPSHOT_FILES", inventory, clear=True), \
                    patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
                self.assertEqual(download_model_snapshot(model), directory)
                self.assertEqual(download_model_snapshot(model, local_files_only=True), directory)
                for call, offline in zip(download.call_args_list, (False, True)):
                    self.assertEqual(call.args, (MODEL_ID,))
                    self.assertEqual(call.kwargs["revision"], MODEL_REVISION)
                    self.assertEqual(call.kwargs["allow_patterns"], sorted(inventory))
                    self.assertEqual(call.kwargs["local_files_only"], offline)
                self.assertEqual(_model_inventory(directory), inventory)
                self.assertTrue(extra.is_file())

    def test_offline_snapshot_must_include_every_component_and_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = fixture_model_snapshot(directory)
            with patch.dict("src.rq3.prepare.MODEL_SNAPSHOT_FILES", inventory, clear=True):
                (Path(directory) / "processor/tokenizer.json").unlink()
                with self.assertRaisesRegex(ValueError, "missing, truncated or changed"):
                    _model_inventory(directory)
        with tempfile.TemporaryDirectory() as directory:
            inventory = fixture_model_snapshot(directory)
            with patch.dict("src.rq3.prepare.MODEL_SNAPSHOT_FILES", inventory, clear=True):
                (Path(directory) / "text_encoder/model-00004-of-00004.safetensors").write_bytes(b"truncated")
                with self.assertRaisesRegex(ValueError, "missing, truncated or changed"):
                    _model_inventory(directory)

    def test_snapshot_indexes_cannot_add_an_unreviewed_component_or_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = fixture_model_snapshot(directory)
            relative = "transformer/diffusion_pytorch_model.safetensors.index.json"
            (Path(directory) / relative).write_text(json.dumps({"weight_map": {"x": "../monolithic.safetensors"}}))
            inventory[relative] = (Path(directory) / relative).stat().st_size
            with patch.dict("src.rq3.prepare.MODEL_SNAPSHOT_FILES", inventory, clear=True):
                with self.assertRaisesRegex(ValueError, "outside the audited model snapshot"):
                    _model_inventory(directory)
        with tempfile.TemporaryDirectory() as directory:
            inventory = fixture_model_snapshot(directory)
            (Path(directory) / "model_index.json").write_text(json.dumps({**MODEL_COMPONENTS, "extra": ["transformers", "ExtraModel"]}))
            inventory["model_index.json"] = (Path(directory) / "model_index.json").stat().st_size
            with patch.dict("src.rq3.prepare.MODEL_SNAPSHOT_FILES", inventory, clear=True):
                with self.assertRaisesRegex(ValueError, "components differ"):
                    _model_inventory(directory)

    def test_download_allow_list_is_bound_to_the_audited_model_revision(self):
        download = Mock()
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaisesRegex(ValueError, "review it before changing"):
                download_model_snapshot({"model_id": MODEL_ID, "revision": "f" * 40})
            download.assert_not_called()
        self.assertEqual(len(MODEL_SNAPSHOT_FILES), 34)
        self.assertEqual(sum(MODEL_SNAPSHOT_FILES.values()), 41_836_780_185)


if __name__ == "__main__":
    unittest.main()
