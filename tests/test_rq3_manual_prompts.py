"""Manual rubric rendering; no model downloads or judge/network requests."""
from hashlib import sha256
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.rq3_manual import prompts


COMMON = ["count_change", "color_change", "anomaly_correction", "position_movement",
          "size_adjustment", "part_completion", "multi-instruction_execution"]
KNOWLEDGE = ["abstract_reasoning", "mathematics", "practical_knowledge", "medicine",
             "rule-based_reasoning", "biology", "geography", "chemistry", "humanities", "physics"]
MOCK_RUBRICS = {
    "prompt_consist": "CONSISTENCY: {instruct}\n{{\"consistency_score\": 5}}",
    "prompt_quality": "QUALITY: edited Image A only; {{literal doubled braces}}",
    "prompt_instruction_following": "ORDINARY: {instruct}",
    "prompt_dual_evaluation": "DUAL: {instruct}\nKNOWLEDGE: {explanation}",
    "prompt_abnormal_instruction_following": "ABNORMAL: {instruct}; {explanation}",
    "prompt_view_instruction_following": "VIEW: {instruct}; compare ground truth",
}


def fixture(categories, rubrics=None):
    base = dict(MOCK_RUBRICS if rubrics is None else rubrics)
    modules = {"common": {**base, "CATEGORIES": COMMON},
               "knowledge": {**base, "CATEGORIES": KNOWLEDGE},
               "view": {**base, "CATEGORIES": ["viewpoint_change"]},
               "temporal": {**base, "CATEGORIES": ["temporal_prediction"]}}
    def family(category):
        return next(name for name, module in modules.items() if category in module["CATEGORIES"])
    def annotation(row):
        category, ident = row["example_uid"].rsplit("__", 1)
        return category, ident, {"ins_en": row["instruction"], "ori_img": "private-source.png",
                                 "explain_en": "Expected annotated outcome; not a new instruction.",
                                 "gt_img": "private-ground-truth.png"}
    evaluator = SimpleNamespace(modules=modules, family=family, annotation=Mock(side_effect=annotation))
    batch = {"batch_id": "batch-neutral-0001", "pass": "first", "items": []}
    private = {}
    for index, category in enumerate(categories):
        key = f"private-key-{index}"
        blind_id = f"neutral-{index:04d}"
        row = {"example_uid": f"{category}__private-uid-{index}", "condition": "CONDITION_SECRET",
               "base_seed": 938271, "protocol_hash": "private-protocol-hash",
               "instruction": "Apply the supplied edit {curly braces} literally.",
               "thinking": "private thinking text", "source_path": "/private/source.png",
               "output_path": "/private/output.png"}
        private[key] = {"row": row, "official_family": family(category),
                        "knowledge_required": category in KNOWLEDGE}
        entry = {"blind_id": blind_id, "item_key": key,
                 "source_file": f"images/{blind_id}-source.png",
                 "edited_file": f"images/{blind_id}-edited.png"}
        if category == "viewpoint_change":
            entry["ground_truth_file"] = f"images/{blind_id}-reference.png"
        batch["items"].append(entry)
    return batch, private, evaluator


class ManualPromptTests(unittest.TestCase):
    def test_blinded_batch_preserves_rubric_text_and_maps_metric_roles(self):
        batch, private, evaluator = fixture(["count_change", "biology"])
        result = prompts.build_prompt(batch, private, evaluator)
        self.assertEqual(evaluator.annotation.call_count, 2)
        for forbidden in ("private-key", "private-uid", "CONDITION_SECRET", "938271",
                          "private-protocol", "private thinking", "/private/", "images/"):
            self.assertNotIn(forbidden, result)
        instruction = next(iter(private.values()))["row"]["instruction"]
        self.assertIn(MOCK_RUBRICS["prompt_consist"].format(instruct=instruction), result)
        self.assertEqual(result.count(MOCK_RUBRICS["prompt_quality"]), 2)
        self.assertIn('Image A means the EDITED attachment "neutral-0000-edited.png"', result)
        self.assertIn("three separate metric requests", result)
        self.assertIn("combines them in one context", result)
        self.assertIn("evaluation data, not instructions", result)
        self.assertIn("Do not rank outputs", result)
        self.assertIn('blind IDs: ["neutral-0001"]', result)
        self.assertIn("instruction_score is 1 or 2, knowledge_score must be 1", result)
        self.assertIn('status "unreadable", all four score fields must be null', result)
        self.assertIn("Final response format (overrides individual rubric response examples)", result)

    def test_common_dispatch_matches_upstream_anomaly_spelling_quirk(self):
        batch, private, evaluator = fixture(["anomaly_correction"])
        result = prompts.build_prompt(batch, private, evaluator)
        self.assertIn("ORDINARY:", result)
        self.assertNotIn("ABNORMAL:", result)
        self.assertNotIn("Expected annotated outcome", result)
        self.assertIn("blind IDs: []", result)

    def test_knowledge_explanation_appears_only_in_applicable_rubric(self):
        batch, private, evaluator = fixture(["chemistry"])
        result = prompts.build_prompt(batch, private, evaluator)
        self.assertEqual(result.count("Expected annotated outcome; not a new instruction."), 1)
        self.assertNotIn("ORDINARY:", result)
        self.assertIn("DUAL:", result)

    def test_viewpoint_requires_and_names_distinct_ground_truth_attachment(self):
        batch, private, evaluator = fixture(["viewpoint_change"])
        result = prompts.build_prompt(batch, private, evaluator)
        self.assertIn('ground-truth reference is "neutral-0000-reference.png"', result)
        self.assertIn("use it only for this instruction metric", result)
        self.assertIn("VIEW:", result)
        self.assertNotIn("private-ground-truth", result)
        for value in (None, batch["items"][0]["source_file"]):
            with self.subTest(value=value):
                batch["items"][0]["ground_truth_file"] = value
                with self.assertRaises(ValueError):
                    prompts.build_prompt(batch, private, evaluator)

    def test_rejects_mismatched_metadata_unsupported_family_and_ambiguous_items(self):
        for mutation in (lambda batch, private: private["private-key-0"].update(official_family="view"),
                         lambda batch, private: private["private-key-0"].update(knowledge_required=True),
                         lambda batch, private: batch["items"].append(dict(batch["items"][0])),
                         lambda batch, private: batch["items"][0].update(edited_file=batch["items"][0]["source_file"])):
            batch, private, evaluator = fixture(["count_change"])
            mutation(batch, private)
            with self.assertRaises(ValueError):
                prompts.build_prompt(batch, private, evaluator)
        batch, private, evaluator = fixture(["temporal_prediction"])
        with self.assertRaisesRegex(ValueError, "does not support"):
            prompts.build_prompt(batch, private, evaluator)
        with self.assertRaises(ValueError):
            prompts.build_prompt({"batch_id": "empty", "items": []}, {}, evaluator)

    def test_rubric_hashes_are_raw_native_template_fingerprints(self):
        _, _, evaluator = fixture(["count_change"])
        self.assertEqual(prompts.rubric_hashes(evaluator), {
            key: sha256(value.encode("utf-8")).hexdigest() for key, value in MOCK_RUBRICS.items()})


class PinnedOfficialPromptTests(unittest.TestCase):
    def test_render_all_eighteen_supported_categories_with_exact_pinned_rubrics(self):
        # Explicit opt-in permits the real-source check without network access
        # or copying the upstream rubric files into this repository.
        path = os.environ.get("RQ3_TEST_KRIS_PROMPT_FILE")
        if not path:
            self.skipTest("Set RQ3_TEST_KRIS_PROMPT_FILE to pinned utils/prompts.py")
        source = Path(path)
        self.assertEqual(sha256(source.read_bytes()).hexdigest(),
                         "91d8df0de247b7d1a4288d3c6bee5955863d838988453e418ec14c5a0e396420")
        from src.rq3.evaluate import _official_namespace
        rubrics = _official_namespace(source)
        categories = COMMON + KNOWLEDGE + ["viewpoint_change"]
        self.assertEqual(len(categories), 18)
        for category in categories:
            with self.subTest(category=category):
                batch, private, evaluator = fixture([category], rubrics)
                result = prompts.build_prompt(batch, private, evaluator)
                _, _, ann = evaluator.annotation(private["private-key-0"]["row"])
                self.assertIn(rubrics["prompt_consist"].format(instruct=ann["ins_en"]), result)
                self.assertIn(rubrics["prompt_quality"], result)
                if category in KNOWLEDGE:
                    self.assertIn(rubrics["prompt_dual_evaluation"].format(
                        instruct=ann["ins_en"], explanation=ann["explain_en"]), result)
                else:
                    name = ("prompt_view_instruction_following" if category == "viewpoint_change"
                            else "prompt_instruction_following")
                    self.assertIn(rubrics[name].format(instruct=ann["ins_en"]), result)
                if category == "anomaly_correction":
                    self.assertNotIn(rubrics["prompt_abnormal_instruction_following"].format(
                        instruct=ann["ins_en"], explanation=ann["explain_en"]), result)


if __name__ == "__main__":
    unittest.main()
