import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.rq3.analyze import (
    CONTRASTS, TIMING_FIELDS, analyze, example_contrasts, make_score_table, paired_summary,
    summarize_timing,
)
from src.rq3.evaluate import file_sha256, quality_score, validate_judge, OfficialEvaluator, OFFICIAL_REVISION


class AnalysisTests(unittest.TestCase):
    def test_known_factorial_effects_average_seeds_before_bootstrap(self):
        table = {}
        # Seed-specific G0 differs, but the example is still the sampling unit.
        for uid, shift in (("biology__1", 0), ("biology__2", 10)):
            for seed, thinking in ((42, 10), (7, 20)):
                for condition, q in zip("ABCD", (30, 30 + thinking, 25, 25 + thinking - 2)):
                    table[(uid, condition, seed)] = {"Q": q + shift}
        uids = ["biology__1", "biology__2"]
        self.assertEqual(example_contrasts(table, uids, [42, 7], CONTRASTS["G0_B_minus_A"]), [15, 15])
        self.assertEqual(example_contrasts(table, uids, [42, 7], CONTRASTS["G1_D_minus_C"]), [13, 13])
        effect = example_contrasts(table, uids, [42, 7], CONTRASTS["interaction_D_minus_C_minus_B_plus_A"])
        self.assertEqual(effect, [-2, -2])
        result = paired_summary(effect)
        self.assertEqual(result["n_examples"], 2)
        self.assertEqual((result["mean"], result["ci_low"], result["ci_high"]), (-2, -2, -2))

    def test_zero_difference_has_valid_wilcoxon_and_bootstrap(self):
        result = paired_summary([0] * 40)
        self.assertEqual(result["wilcoxon_p"], 1)
        self.assertEqual(result["ci_width"], 0)

    def test_pairing_rejects_missing_duplicate_and_cross_protocol_scores(self):
        row = {"example_uid": "biology__1", "condition": "A", "base_seed": 42, "protocol_hash": "h", "Q": 50}
        expected = {("biology__1", "A", 42), ("biology__1", "B", 42)}
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            make_score_table([row], expected, "h")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            make_score_table([row, row], expected, "h")
        with self.assertRaisesRegex(ValueError, "different protocols"):
            make_score_table([row], expected, "other")

    def test_timing_uses_median_repetitions_and_excludes_quality_run(self):
        rows = []
        for repeat, seconds in enumerate((999, 4, 2, 30)):
            rows.append({"example_uid": "biology__1", "condition": "B", "base_seed": 42,
                         "protocol_hash": "h", "repeat": repeat, "status": "ok", "peak_vram_bytes": 123,
                         "timings": {"diffusion_seconds": seconds, "charged_total_seconds": seconds + 2,
                                     "measured_replay_seconds": seconds + 1}})
        key = ("biology__1", "B", 42)
        medians, failures, _ = summarize_timing(rows, {key}, "h")
        self.assertEqual(medians[key]["diffusion_seconds"], 4)
        self.assertEqual(medians[key]["charged_total_seconds"], 6)
        self.assertEqual(failures, [])
        with self.assertRaisesRegex(ValueError, "timing repeats"):
            summarize_timing(rows[:-1], {key}, "h")

    def test_missing_knowledge_is_not_silently_averaged_away(self):
        components = {"consistency_score": 5, "instruction_score": 1, "quality_score": 3}
        self.assertEqual(quality_score(components), 50)
        with self.assertRaisesRegex(ValueError, "Missing/invalid"):
            quality_score(components, knowledge_required=True)
        components["knowledge_score"] = 5
        self.assertEqual(quality_score(components, knowledge_required=True), 62.5)

    def test_predeclared_margin_must_cover_judge_noise(self):
        config = {"_judge": {"official_revision": OFFICIAL_REVISION, "model": "gpt-4o-2024-08-06", "temperature": 0,
                              "aggregate": "mean_applicable_normalized", "aggregate_confirmed": True},
                  "statistics": {"equivalence_margin": 2, "minimum_meaningful_difference": 3}}
        with self.assertRaisesRegex(ValueError, "Freeze numeric"):
            validate_judge(config)
        validate_judge(config, calibration=True)
        config["statistics"]["equivalence_margin"] = 3
        validate_judge(config)

    def test_complete_200_output_design_and_600_timing_repeats(self):
        from PIL import Image
        from src.rq3.protocol import build_jobs, load_protocol
        config = load_protocol("configs/rq3.yaml")
        config["_judge"]["aggregate_confirmed"] = True
        config["statistics"].update(equivalence_margin=5, minimum_meaningful_difference=1)
        scores, generations = [], []
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "fixture.png"
            Image.new("RGB", (8, 8), "blue").save(image_path)
            digest = file_sha256(image_path)
            for job in build_jobs(config, "all", [42]):
                uid, condition = job["example_uid"], job["condition"]
                q = 25 if condition in "AC" else 50
                identity = {key: job[key] for key in ("example_uid", "condition", "base_seed", "knowledge_type", "stage_seeds")}
                row = {**identity, "protocol_hash": config["protocol_hash"], "output_path": str(image_path),
                       "output_sha256": digest, "source_sha256": digest,
                       "initial_noise_sha256": "initial_noise_" + uid, "instruction": "edit " + uid,
                       "prompt": ("edit " if condition in "AC" else "rewrite ") + uid,
                       "status": "ok", "peak_vram_bytes": 123}
                for repeat in range(4):
                    duration = 4 if condition in "AB" else 2
                    timings = {field: 0.1 for field in TIMING_FIELDS}
                    timings.update(diffusion_seconds=duration, measured_replay_seconds=duration + .5,
                                   charged_total_seconds=duration + 1)
                    generations.append({**row, "repeat": repeat, "timings": timings})
                scores.append({**identity, "protocol_hash": config["protocol_hash"], "output_sha256": digest,
                               "Q": q, "consistency_score": q/25+1, "instruction_score": q/25+1,
                               "quality_score": q/25+1, "knowledge_score": q/25+1,
                               "prompt_sha256": "fixture", "judge_model": config["_judge"]["model"],
                               "aggregate": config["_judge"]["aggregate"]})
            report, effects, _ = analyze(config, scores, generations, [42])
            self.assertEqual(report["expected_quality_outputs"], 200)
            self.assertEqual(len(generations), 800)
            self.assertEqual(len(effects), 40)
            self.assertEqual(report["ablations"]["contrasts"]["full_D_minus_vanilla_B"]["n_examples"], 20)
            self.assertEqual(report["contrasts"]["G0_B_minus_A"]["mean"], 25)
            self.assertEqual(report["contrasts"]["interaction_D_minus_C_minus_B_plus_A"]["mean"], 0)
            self.assertTrue(report["retention"]["quality_retained_one_sided_rule"])
            self.assertEqual(report["efficiency_thinking_on"]["conditions"]["D"]["generator_speedup_mean_paired_ratio"], 2)
            generations[4]["initial_noise_sha256"] = "different_noise"
            with self.assertRaisesRegex(ValueError, "initial_noise_sha256"):
                analyze(config, scores, generations, [42])

    @unittest.skipUnless(os.environ.get("RQ3_OFFICIAL_CHECKOUT"), "Optional pinned upstream checkout not provided")
    def test_actual_official_functions_with_recorded_fake_judge(self):
        from PIL import Image, ImageOps
        checkout = Path(os.environ["RQ3_OFFICIAL_CHECKOUT"])
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            content = json.dumps({"consistency_score": 5, "instruction_score": 4,
                                  "quality_score": 3, "knowledge_score": 2, "reasoning": "fixture"})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                                   model_dump=lambda **unused: {"fixture": True, "content": content})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        judge = {"official_checkout": str(checkout), "official_revision": OFFICIAL_REVISION,
                 "prompt_file": "utils/prompts.py", "model": "gpt-4o-2024-08-06", "temperature": 0, "max_tokens": 1000}
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.jsonl"
            evaluator = OfficialEvaluator(judge, client, raw_path)
            for category in ("count_change", "biology", "viewpoint_change"):
                annotations = json.loads((checkout / "KRIS_Bench" / category / "annotation.json").read_text())
                image_id, annotation = next(iter(annotations.items()))
                source_path = Path(directory) / "source.png"
                with Image.open(checkout / "KRIS_Bench" / category / annotation["ori_img"]) as image:
                    ImageOps.exif_transpose(image).convert("RGB").save(source_path)
                row = {"example_uid": f"{category}__{image_id}", "instruction": annotation["ins_en"],
                       "source_path": str(source_path), "output_path": str(source_path),
                       "condition": "A", "base_seed": 42, "protocol_hash": "test"}
                result = evaluator.evaluate(row)
                self.assertEqual(result["Q"], 62.5 if category == "biology" else 75)
                self.assertEqual(len(result["raw_responses"]), 3)
            # Temporal examples have multiple sources and are intentionally
            # excluded by RQ3 eligibility. Still smoke the loaded upstream
            # function in isolation so its helpers remain intact.
            category = "temporal_prediction"
            annotations = json.loads((checkout / "KRIS_Bench" / category / "annotation.json").read_text())
            image_id, annotation = next(iter(annotations.items()))
            target = Path(directory) / "rq3" / category
            target.mkdir(parents=True)
            (target / f"{image_id}.jpg").symlink_to(source_path)
            module = evaluator.modules["temporal"]
            module.update(BENCH_DIR=str(checkout / "KRIS_Bench"), RESULTS_DIR=directory, client=evaluator.client)
            components = module["evaluate_temporal_images"]("rq3", category, image_id)
            self.assertEqual(quality_score(components), 75)
            self.assertEqual(len(raw_path.read_text().splitlines()), 12)
        self.assertTrue(all(request["temperature"] == 0 for request in requests))
        self.assertTrue(all(request["model"] == judge["model"] for request in requests))


if __name__ == "__main__":
    unittest.main()
