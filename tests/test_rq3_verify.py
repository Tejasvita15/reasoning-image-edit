import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.rq3.verify import compare_components, compare_reruns


class VerifyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def rows(self, count=10, conditions="ABCD"):
        rows = []
        for index in range(count):
            uid = f"biology__{index}"
            for condition in conditions:
                thought = {"raw_text": "a sampled thought", "rewrite": "a rewritten instruction", "token_count": 7}
                if condition in "AC":
                    thought = None
                enabled, rikv, avd = condition in "CDEF", condition in "CDF", condition in "CDE"
                steps = [{"step": step, "avd_cache_reused": avd and step == 8,
                          "rikv_cache_writes": int(rikv and step == 5),
                          "rikv_context_reuses": int(rikv and 6 <= step <= 25)}
                         for step in range(28)]
                regions = [{"step": 5, "grid_height": 2, "grid_width": 2, "selected_indices": [1]}]
                rows.append({"example_uid": uid, "condition": condition, "base_seed": 42,
                             "repeat": 0, "phase": "pilot", "status": "ok", "protocol_hash": "frozen",
                             "instruction": "original instruction", "thought": thought,
                             "prompt": thought["rewrite"] if thought else "original instruction",
                             "initial_noise_sha256": hashlib.sha256(uid.encode()).hexdigest(),
                             "regione": {"enabled": enabled, "rikv_enabled": rikv, "avd_enabled": avd,
                                          "steps": steps, "selected_regions": regions}})
        return rows

    def write_run(self, name, rows):
        root = self.root / name
        root.mkdir()
        (root / "protocol.json").write_text(json.dumps({"protocol_hash": "frozen"}))
        source = root / "source.png"
        Image.new("RGB", (3, 3), (64, 64, 64)).save(source)
        saved = copy.deepcopy(rows)
        for index, row in enumerate(saved):
            path = root / f"image{index}.png"
            Image.new("RGB", (3, 3), (128, 128, 128)).save(path)
            row["output_path"] = str(path)
            row["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["source_path"] = str(source)
            row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.save_rows(root, saved)
        return root

    @staticmethod
    def save_rows(root, rows):
        (root / "generations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_complete_independent_reruns_pass(self):
        rows = self.rows()
        report = compare_reruns(self.write_run("first", rows), self.write_run("second", rows))
        self.assertTrue(report["passed"])
        self.assertEqual(report["n_examples"], 10)
        self.assertEqual(len(report["comparisons"]), 40)

    def test_each_example_needs_all_four_conditions(self):
        rows = self.rows()
        rows = [row for row in rows if row["condition"] == "A" or row["example_uid"] == "biology__0"]
        report = compare_reruns(self.write_run("first", rows), self.write_run("second", rows))
        self.assertFalse(report["passed"])
        self.assertFalse(report["coverage_passed"])
        self.assertEqual(len(report["missing_cells"]), 9)

    def test_missing_thoughts_cannot_compare_equal_and_pass(self):
        rows = self.rows()
        for row in rows:
            row["thought"] = None
        report = compare_reruns(self.write_run("first", rows), self.write_run("second", rows))
        self.assertFalse(report["passed"])

    def test_same_wrong_noise_across_reruns_fails_pair_controls(self):
        rows = self.rows()
        for row in rows:
            if row["condition"] == "D":
                row["initial_noise_sha256"] = "a" * 64
        report = compare_reruns(self.write_run("first", rows), self.write_run("second", rows))
        self.assertTrue(all(row["passed"] for row in report["comparisons"]))
        self.assertFalse(report["passed"])
        self.assertTrue(any(not row["passed"] for row in report["paired_controls"]))

    def test_same_run_cannot_be_its_own_rerun(self):
        root = self.write_run("first", self.rows())
        with self.assertRaisesRegex(ValueError, "independent"):
            compare_reruns(root, root)

    def test_tolerance_one_requires_strictly_less_than_one(self):
        first = self.write_run("first", self.rows())
        second = self.write_run("second", self.rows())
        rows = [json.loads(line) for line in (second / "generations.jsonl").read_text().splitlines()]
        path = Path(rows[0]["output_path"])
        Image.new("RGB", (3, 3), (129, 129, 129)).save(path)
        rows[0]["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_rows(second, rows)
        report = compare_reruns(first, second, tolerance=1.0)
        self.assertFalse(report["passed"])
        self.assertEqual(report["comparisons"][0]["mean_absolute_pixel_difference"], 1.0)

    def test_three_complete_component_examples_pass(self):
        report = compare_components(self.write_run("components", self.rows(3, "DEF")))
        self.assertTrue(report["passed"])
        self.assertTrue(report["coverage_passed"])

    def test_missing_component_cell_reports_failed_coverage(self):
        rows = self.rows(3, "DEF")[:-1]
        report = compare_components(self.write_run("components", rows))
        self.assertFalse(report["passed"])
        self.assertFalse(report["coverage_passed"])
        self.assertEqual(report["missing_cells"][0]["conditions"], ["F"])

    def test_missing_F_events_cannot_vacuously_pass_no_reuse(self):
        rows = self.rows(3, "DEF")
        for row in rows:
            if row["condition"] == "F":
                row["regione"]["steps"] = []
        report = compare_components(self.write_run("components", rows))
        self.assertFalse(report["passed"])
        self.assertTrue(all(not row["F_has_no_AVD_reuse"] for row in report["comparisons"]))

    def test_missing_regions_cannot_compare_equal_and_pass(self):
        rows = self.rows(3, "DEF")
        for row in rows:
            row["regione"].pop("selected_regions")
        report = compare_components(self.write_run("components", rows))
        self.assertFalse(report["passed"])

    def test_E_context_injection_cannot_pass_disabled_flag_only(self):
        rows = self.rows(3, "DEF")
        for row in rows:
            if row["condition"] == "E":
                row["regione"]["steps"][7]["rikv_context_reuses"] = 1
        report = compare_components(self.write_run("components", rows))
        self.assertFalse(report["passed"])

    def test_component_pilot_needs_three_examples(self):
        report = compare_components(self.write_run("components", self.rows(2, "DEF")))
        self.assertFalse(report["passed"])
        self.assertFalse(report["coverage_passed"])


if __name__ == "__main__":
    unittest.main()
