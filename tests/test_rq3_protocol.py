"""Protocol tests need only Python and PyYAML; no torch, downloads, or GPU."""

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import yaml

from src.rq3.protocol import (
    REPO_ROOT, build_jobs, derive_seed, freeze_protocol, load_protocol,
    pilot_uids, validate_config, validate_for_execution, validate_runtime_protocol,
    validate_splits,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = yaml.safe_load((REPO_ROOT / "configs/rq3.yaml").read_text())
        self.judge_path = self.root / "judge.yaml"
        self.judge_path.write_text(yaml.safe_dump({
            "official_revision": "cf1475253a13f5e5666826c10d8967e38917468a",
            "model": "gpt-4o-2024-08-06", "temperature": 0,
        }))
        self.raw["judge"]["config_path"] = str(self.judge_path)
        self.path = self.root / "rq3.yaml"
        self.write_config()
        self.config = load_protocol(self.path)

    def write_config(self):
        self.path.write_text(yaml.safe_dump(self.raw))

    def test_plan_has_200_generations_and_reuses_B_D(self):
        jobs = build_jobs(self.config)
        self.assertEqual(len(jobs), 200)
        self.assertEqual(Counter(j["condition"] for j in jobs),
                         {"A": 40, "B": 40, "C": 40, "D": 40, "E": 20, "F": 20})
        self.assertEqual(len({(j["example_uid"], j["condition"], j["base_seed"]) for j in jobs}), 200)
        self.assertEqual(len(build_jobs(self.config, "rq3_1")), 160)
        self.assertEqual(len(build_jobs(self.config, "rq3_2")), 40)
        self.assertEqual(len(build_jobs(self.config, seeds=[42, 1234, 7])), 600)

    def test_all_conditions_share_thought_and_initial_noise_seeds(self):
        jobs = build_jobs(self.config)
        for uid in self.config["_subset"]["example_uids"]:
            matched = [j for j in jobs if j["example_uid"] == uid]
            self.assertEqual(len(matched), 6)
            self.assertEqual(len({json.dumps(j["stage_seeds"], sort_keys=True) for j in matched}), 1)
        uid = jobs[0]["example_uid"]
        expected = int.from_bytes(hashlib.sha256(f"42|{uid}|diffusion_0".encode()).digest()[:8], "big") % (2**31 - 1)
        self.assertEqual(derive_seed(42, uid, "diffusion_0"), expected)
        self.assertNotEqual(derive_seed(42, uid, "thinking"), expected)

    def test_pilot_is_balanced_and_disjoint_from_test(self):
        jobs = build_jobs(self.config, "pilot")
        self.assertEqual(len(jobs), 60)
        selected = set(pilot_uids(self.config))
        self.assertEqual(len(selected), 10)
        self.assertFalse(selected & set(self.config["_splits"]["k_test"]))
        self.assertEqual(set(j["condition"] for j in jobs), set("ABCDEF"))
        self.assertEqual(len(build_jobs(self.config, "pilot", k_dev=3)), 18)

    def test_frozen_split_checksums_detect_edit(self):
        path = self.root / "splits.json"
        path.write_bytes(Path(self.config["_paths"]["splits"]).read_bytes() + b"\n")
        self.raw["data"]["splits_path"] = str(path)
        self.write_config()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_protocol(self.path)

    def test_split_membership_and_duplicates_are_rejected(self):
        splits = copy.deepcopy(self.config["_splits"])
        subset = copy.deepcopy(self.config["_subset"])
        splits["k_dev"][0] = splits["k_dev"][1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_splits(splits, subset)
        splits = copy.deepcopy(self.config["_splits"])
        replacement = next(uid for uid in splits["k_dev"] if uid.startswith("biology__"))
        index = next(i for i, uid in enumerate(subset["example_uids"]) if uid.startswith("biology__"))
        subset["example_uids"][index] = replacement
        with self.assertRaisesRegex(ValueError, "contained in K-small"):
            validate_splits(splits, subset)

    def test_hash_changes_with_judge_and_refuses_resume(self):
        directory = self.root / "outputs"
        freeze_protocol(self.config, directory)
        freeze_protocol(self.config, directory)
        judge = yaml.safe_load(self.judge_path.read_text())
        judge["model"] = "replacement-fixed-snapshot"
        self.judge_path.write_text(yaml.safe_dump(judge))
        changed = load_protocol(self.path)
        self.assertNotEqual(changed["protocol_hash"], self.config["protocol_hash"])
        with self.assertRaisesRegex(ValueError, "different frozen protocol"):
            freeze_protocol(changed, directory)

    def test_fixed_design_rejects_50_step_and_sequential_offload(self):
        changed = copy.deepcopy(self.config)
        changed["generation"]["num_inference_steps"] = 50
        with self.assertRaisesRegex(ValueError, "num_inference_steps"):
            validate_config(changed)
        changed = copy.deepcopy(self.config)
        changed["model"]["offload_mode"] = "sequential"
        with self.assertRaisesRegex(ValueError, "offload_mode"):
            validate_config(changed)

    def test_unresolved_pins_only_block_runtime(self):
        self.raw["model"]["revision"] = "main"
        self.write_config()
        config = load_protocol(self.path)
        self.assertEqual(len(build_jobs(config)), 200)
        with self.assertRaisesRegex(ValueError, "immutable.*model.revision"):
            validate_runtime_protocol(config)

    def test_calibration_and_seed_validation(self):
        with self.assertRaisesRegex(ValueError, "Freeze equivalence_margin"):
            validate_runtime_protocol(self.config, require_equivalence=True)
        changed = copy.deepcopy(self.config)
        changed["statistics"].update(equivalence_margin=1.0, minimum_meaningful_difference=2.0)
        with self.assertRaisesRegex(ValueError, "at least"):
            validate_config(changed)
        for seeds in ([], [42, 42], [999]):
            with self.assertRaisesRegex(ValueError, "Seeds"):
                build_jobs(self.config, seeds=seeds)

    def test_runtime_requires_matching_successful_development_reports(self):
        for kind in ("determinism", "components"):
            self.config["validation"][f"{kind}_report"] = str(self.root / f"{kind}.json")
        with patch("src.rq3.protocol.verify_checkout"):
            validate_for_execution(self.config, phase="pilot")
            with self.assertRaisesRegex(ValueError, "Missing determinism"):
                validate_for_execution(self.config)
            for kind in ("determinism", "components"):
                Path(self.config["validation"][f"{kind}_report"]).write_text(json.dumps({
                    "kind": kind, "protocol_hash": self.config["protocol_hash"], "passed": True,
                }))
            validate_for_execution(self.config)
            self.config["protocol_hash"] = "different-protocol"
            with self.assertRaisesRegex(ValueError, "exact protocol"):
                validate_for_execution(self.config)


if __name__ == "__main__":
    unittest.main()
