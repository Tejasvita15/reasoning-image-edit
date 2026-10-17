# src/check_determinism.py

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.config import load_config, print_config_summary
from src.dataset import (
    get_decoded_source_images,
    load_kris_bench,
    prepare_example,
    select_split_examples,
)
from src.step1x_runner import Step1XRunner


def image_to_uint8_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compare_images(image_a: Image.Image, image_b: Image.Image) -> Dict[str, Any]:
    a = image_to_uint8_array(image_a)
    b = image_to_uint8_array(image_b)

    if a.shape != b.shape:
        return {
            "same_shape": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "bitwise_equal": False,
            "mean_abs_pixel_diff": None,
            "max_abs_pixel_diff": None,
            "mae_lt_1": False,
        }

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mae = float(diff.mean())

    return {
        "same_shape": True,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "bitwise_equal": bool(np.array_equal(a, b)),
        "mean_abs_pixel_diff": mae,
        "max_abs_pixel_diff": int(diff.max()),
        "mae_lt_1": mae < 1.0,
    }


def normalize_text(value: Optional[Any]) -> Optional[str]:
    return None if value is None else str(value)


def run_once(
    *,
    runner: Step1XRunner,
    raw_example,
    example: Dict[str, Any],
    base_seed: int,
) -> Dict[str, Any]:
    images = get_decoded_source_images(raw_example)

    if len(images) != 1:
        raise RuntimeError(
            f"{example['example_uid']}: expected exactly one source image, got {len(images)}."
        )

    return runner.run(
        images=images,
        instruction=example["instruction"],
        example_uid=example["example_uid"],
        base_seed=base_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Step1X RQ1 reproducibility on fixed K-dev examples."
    )
    parser.add_argument("--config", default="configs/rq1_thinking.yaml")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/determinism")
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    if args.num_examples <= 0:
        raise ValueError("--num-examples must be > 0.")

    torch.use_deterministic_algorithms(True, warn_only=args.warn_only)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = True

    config = load_config(args.config)
    print_config_summary(config, seed=args.seed)

    print("Deterministic algorithms:", torch.are_deterministic_algorithms_enabled())
    print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    print("Loading KRIS-Bench...")
    dataset = load_kris_bench()

    k_dev_examples = select_split_examples(
        dataset=dataset,
        splits_path=args.splits,
        split_name="k_dev",
    )

    if len(k_dev_examples) < args.num_examples:
        raise ValueError(
            f"K-dev contains only {len(k_dev_examples)} examples, "
            f"but {args.num_examples} were requested."
        )

    test_examples = k_dev_examples.select(range(args.num_examples))

    output_root = Path(args.output_dir)
    run_a_dir = output_root / "run_a"
    run_b_dir = output_root / "run_b"
    run_a_dir.mkdir(parents=True, exist_ok=True)
    run_b_dir.mkdir(parents=True, exist_ok=True)

    runner = Step1XRunner(config)

    records = []
    exact_image_count = 0
    rewrite_match_count = 0
    fallback_pass_count = 0
    failure_count = 0

    print("=" * 70)
    print(f"Determinism check: {len(test_examples)} K-dev examples")
    print("=" * 70)

    for index, raw_example in enumerate(test_examples, start=1):
        example = prepare_example(raw_example)
        example_uid = example["example_uid"]

        print(f"\n[{index}/{len(test_examples)}] {example_uid}")

        try:
            result_a = run_once(
                runner=runner,
                raw_example=raw_example,
                example=example,
                base_seed=args.seed,
            )
            result_b = run_once(
                runner=runner,
                raw_example=raw_example,
                example=example,
                base_seed=args.seed,
            )

            final_a = result_a["final_image"]
            final_b = result_b["final_image"]

            path_a = run_a_dir / f"{example_uid}.png"
            path_b = run_b_dir / f"{example_uid}.png"
            final_a.save(path_a)
            final_b.save(path_b)

            image_comparison = compare_images(final_a, final_b)

            rewrite_a = normalize_text(result_a.get("reformat_prompt"))
            rewrite_b = normalize_text(result_b.get("reformat_prompt"))
            rewrite_equal = rewrite_a == rewrite_b

            decision_keys = (
                "attempt_count",
                "retry_triggered",
                "selected_candidate_index",
                "selected_is_last_candidate",
                "stop_reason",
                "reflection_info",
            )

            decision_a = {k: result_a.get(k) for k in decision_keys}
            decision_b = {k: result_b.get(k) for k in decision_keys}
            decision_match = {
                k: decision_a[k] == decision_b[k]
                for k in decision_keys
            }

            candidates_a = result_a.get("candidate_images") or []
            candidates_b = result_b.get("candidate_images") or []

            candidate_images_equal = (
                len(candidates_a) == len(candidates_b)
                and len(candidates_a) > 0
                and all(
                    compare_images(a, b)["bitwise_equal"]
                    for a, b in zip(candidates_a, candidates_b)
                )
            )

            if rewrite_equal:
                rewrite_match_count += 1
            if image_comparison["bitwise_equal"]:
                exact_image_count += 1
            if image_comparison["mae_lt_1"]:
                fallback_pass_count += 1

            record = {
                "example_uid": example_uid,
                "category": example.get("category"),
                "knowledge_type": example.get("knowledge_type"),
                "base_seed": args.seed,
                "stage_seeds_run_a": result_a.get("stage_seeds"),
                "stage_seeds_run_b": result_b.get("stage_seeds"),
                "stage_seeds_equal": (
                    result_a.get("stage_seeds") == result_b.get("stage_seeds")
                ),
                "rewrite_run_a": rewrite_a,
                "rewrite_run_b": rewrite_b,
                "rewrite_equal": rewrite_equal,
                "decision_a": decision_a,
                "decision_b": decision_b,
                "decision_match": decision_match,
                "candidate_count_a": len(candidates_a),
                "candidate_count_b": len(candidates_b),
                "candidate_images_equal": candidate_images_equal,
                "image": image_comparison,
                "run_a_path": str(path_a),
                "run_b_path": str(path_b),
                "status": "success",
            }
            records.append(record)

            print("  rewrite equal:", rewrite_equal)
            print("  bitwise image equal:", image_comparison["bitwise_equal"])
            print("  mean abs pixel diff:", image_comparison["mean_abs_pixel_diff"])

        except Exception as exc:
            failure_count += 1
            records.append(
                {
                    "example_uid": example_uid,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            print("  FAILED:", repr(exc))

    successful_count = len(records) - failure_count

    all_rewrites_exact = (
        successful_count > 0 and rewrite_match_count == successful_count
    )
    all_images_exact = (
        successful_count > 0 and exact_image_count == successful_count
    )
    all_images_fallback = (
        successful_count > 0 and fallback_pass_count == successful_count
    )

    exact_pass = (
        failure_count == 0
        and all_rewrites_exact
        and all_images_exact
    )

    fallback_pass = (
        failure_count == 0
        and all_rewrites_exact
        and all_images_fallback
    )

    summary = {
        "config": args.config,
        "split": "k_dev",
        "base_seed": args.seed,
        "num_examples": len(test_examples),
        "warn_only": args.warn_only,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "successful_examples": successful_count,
        "failed_examples": failure_count,
        "rewrite_exact_matches": rewrite_match_count,
        "bitwise_image_matches": exact_image_count,
        "mae_lt_1_matches": fallback_pass_count,
        "exact_pass": exact_pass,
        "fallback_pass_mae_lt_1": fallback_pass,
        "records": records,
    }

    summary_path = output_root / "determinism_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DETERMINISM SUMMARY")
    print("=" * 70)
    print("Successful:", successful_count, "/", len(test_examples))
    print("Rewrite exact:", rewrite_match_count, "/", successful_count)
    print("Bitwise image exact:", exact_image_count, "/", successful_count)
    print("MAE < 1:", fallback_pass_count, "/", successful_count)
    print("Exact determinism PASS:", exact_pass)
    print("Fallback MAE<1 PASS:", fallback_pass)
    print("Summary:", summary_path)
    print("=" * 70)

    if exact_pass:
        print("PASS: rewrites and output images are exactly reproducible.")
        return

    if fallback_pass:
        print(
            "WARNING: exact pixel equality failed, "
            "but every pair has MAE < 1."
        )
        return

    raise SystemExit(
        "FAIL: determinism requirement was not met. "
        "Inspect determinism_summary.json before running K-test."
    )


if __name__ == "__main__":
    main()
