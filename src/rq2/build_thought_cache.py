# src/rq2/build_thought_cache.py

"""
Generate each example's Step1X thinking rewrite once and cache it
(RQ2 replay, S4).

Step1X-v1p2's "thought" is the rewritten instruction itself: the
thinker returns only the rewrite, with no separate reasoning trace.

For each example:
    - seed the global torch RNG with derive_seed(base, uid, "thinking"),
      the same seed Step1XRunner sets before a live thinking call
    - build Step1XEditThinker and call think(), exactly as the pipeline
      does when enable_thinking_mode=True
    - repeat timing_repeats times (reseeding each time) to get a median
      thinking latency and a within-process determinism check

The thinking latency stored here is the one thinking pass charged to
every replayed thinking-on generation. Under model CPU offload the
text encoder stays on the GPU between the timed calls, so the
text-encoder transfer is charged to the generate call instead (where
the pipeline also pays it for prompt encoding). It is not counted twice.

A finished cache is never regenerated. The file is written as
<out>.partial and renamed only when complete.
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.rq2.common import (
    collect_env,
    count_tokens,
    enable_determinism,
    fix_text_encoder_offload,
    load_rq2_examples,
    offload_mode,
)


def think_once(
    pipe,
    thinker_cls,
    image,
    instruction: str,
    thinking_seed: int,
):
    """
    One timed Step1X thinking pass.
    Returns (rewrite, seconds, peak VRAM GB).
    """

    import torch

    from src.seeding import set_global_seed

    # The thinker samples through the global torch RNG (S4).
    set_global_seed(thinking_seed)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    # Built per call, as in the pipeline (it loads its own processor).
    thinker = thinker_cls(
        pipe.text_encoder,
        pipe.processor,
    )

    rewrite = thinker.think(
        image,
        instruction,
    )

    torch.cuda.synchronize()

    latency_sec = time.perf_counter() - start

    peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    return rewrite, latency_sec, peak_vram_gb


def think_once_unireason(
    runner,
    images,
    instruction: str,
    thinking_seed: int,
):
    """
    One timed UniReason thinking pass.
    Returns (think_output, seconds, peak VRAM GB).

    runner._think() calls set_global_seed(thinking_seed) internally,
    so seeding is handled there.
    """

    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    think_output = runner._think(
        images=images,
        instruction=instruction,
        thinking_seed=thinking_seed,
    )

    torch.cuda.synchronize()

    latency_sec = time.perf_counter() - start

    peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    return think_output, latency_sec, peak_vram_gb


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the RQ2 Step1X thought cache."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--splits",
        default="data/splits.json",
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["k_dev", "k_small", "k_test"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Stratified subset (smoke / S7 only).",
    )

    parser.add_argument(
        "--timing_repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--backend",
        default="step1x",
        choices=["step1x", "unireason"],
        help="Model backend; selects the thinking implementation.",
    )

    args = parser.parse_args()

    out_path = Path(args.out)

    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists. A thought cache is built "
            "once and never regenerated."
        )

    if args.timing_repeats < 1:
        parser.error("--timing_repeats must be >= 1")

    determinism = enable_determinism()

    from src.config import load_config
    from src.dataset import (
        get_decoded_source_images,
        prepare_example,
    )
    from src.seeding import derive_seed

    config = load_config(args.config)

    examples = load_rq2_examples(
        args.splits,
        args.split,
        args.n_examples,
    )

    print(
        f"Backend: {args.backend} | split: {args.split} "
        f"| examples: {len(examples)} "
        f"| seed: {args.seed} | out: {out_path}"
    )

    if args.backend == "unireason":
        from src.unireason_runner import UniReasonRunner
        runner = UniReasonRunner(config)
        tokenizer = runner.tokenizer
    else:
        from diffusers.pipelines.step1x_edit.pipeline_step1x_edit_thinker import (
            Step1XEditThinker,
        )
        from src.step1x_runner import Step1XRunner
        runner = Step1XRunner(config)
        pipe = runner.pipe
        fix_text_encoder_offload(pipe)
        tokenizer = pipe.processor.tokenizer

    env = collect_env()

    # =========================================================
    # Warm-up (discarded)
    # =========================================================

    first = examples[0]
    first_images = [get_decoded_source_images(first)[0]]

    if args.backend == "unireason":
        think_once_unireason(
            runner,
            first_images,
            first["instruction"],
            thinking_seed=0,
        )
    else:
        think_once(
            pipe,
            Step1XEditThinker,
            first_images[0],
            first["instruction"],
            thinking_seed=0,
        )

    # =========================================================
    # Cache
    # =========================================================

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_path = out_path.with_name(
        out_path.name + ".partial"
    )

    n_nondeterministic = 0

    with open(partial_path, "w", encoding="utf-8") as f:

        for idx in range(len(examples)):

            raw_example = examples[idx]
            example = prepare_example(raw_example)
            uid = example["example_uid"]

            image = get_decoded_source_images(raw_example)[0]

            thinking_seed = derive_seed(
                args.seed,
                uid,
                "thinking",
            )

            if args.backend == "unireason":
                runs = [
                    think_once_unireason(
                        runner,
                        [image],
                        example["instruction"],
                        thinking_seed,
                    )
                    for _ in range(args.timing_repeats)
                ]
            else:
                runs = [
                    think_once(
                        pipe,
                        Step1XEditThinker,
                        image,
                        example["instruction"],
                        thinking_seed,
                    )
                    for _ in range(args.timing_repeats)
                ]

            rewrite = runs[0][0]
            latencies = [run[1] for run in runs]

            identical = all(
                run[0] == rewrite
                for run in runs
            )

            if not identical:
                n_nondeterministic += 1

            record = {
                "example_uid": uid,
                "split_name": args.split,
                "instruction": example["instruction"],

                # Replayed verbatim during generation.
                # Step1X: rewritten instruction (replaces prompt).
                # UniReason: full <think>...</think> chain
                #            (appended alongside the original instruction).
                "rewrite": rewrite,
                "thinking_token_count":
                    count_tokens(tokenizer, rewrite),

                "base_seed": args.seed,
                "thinking_seed": thinking_seed,
                "rewrite_identical_across_repeats": identical,

                "latency_thinking_repeats_sec": latencies,
                "latency_thinking_sec":
                    statistics.median(latencies),
                "peak_vram_thinking_gb":
                    max(run[2] for run in runs),

                "model_id": config["model"]["model_id"],
                "backend": args.backend,
                "offload_mode": offload_mode(config),
                "determinism": determinism,
                "env": env,
                "created_at_utc":
                    datetime.now(timezone.utc).isoformat(),
            }

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

            f.flush()

            print(
                f"[{idx + 1}/{len(examples)}] {uid} "
                f"| {record['thinking_token_count']} tokens "
                f"| {record['latency_thinking_sec']:.2f}s "
                f"| identical={identical}"
            )
            print(f"    rewrite: {rewrite}")

    os.replace(partial_path, out_path)

    print("=" * 60)
    print(f"Saved: {out_path}")
    print(
        f"Rewrites differing across repeats: "
        f"{n_nondeterministic}/{len(examples)}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
