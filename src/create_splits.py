# src/create_splits.py

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.dataset import (
    load_kris_bench,
    get_rq1_eligible_dataset,
    prepare_example,
)


SPLIT_SEED = 42

K_DEV_PER_TYPE = 30
K_TEST_PER_TYPE = 100

# 40 total:
# factual = 14
# conceptual = 13
# procedural = 13
K_SMALL_COUNTS = {
    "factual": 14,
    "conceptual": 13,
    "procedural": 13,
}

OUTPUT_PATH = Path("data/splits.json")


def main():

    rng = random.Random(SPLIT_SEED)

    print("Loading KRIS-Bench...")

    dataset = load_kris_bench()

    print(
        f"Original dataset size: "
        f"{len(dataset)}"
    )

    # ---------------------------------------------------------
    # Remove examples that RQ1 cannot use
    # ---------------------------------------------------------

    eligible_dataset = (
        get_rq1_eligible_dataset(
            dataset
        )
    )

    print(
        f"RQ1 eligible examples: "
        f"{len(eligible_dataset)}"
    )

    # ---------------------------------------------------------
    # Group eligible examples by knowledge type
    # ---------------------------------------------------------

    groups = defaultdict(list)

    category_counts = Counter()

    for raw_example in eligible_dataset:

        example = prepare_example(
            raw_example
        )

        knowledge_type = example[
            "knowledge_type"
        ]

        uid = example[
            "example_uid"
        ]

        groups[
            knowledge_type
        ].append(uid)

        category_counts[
            example["category"]
        ] += 1

    print()
    print("Eligible examples by knowledge type:")

    for knowledge_type in sorted(groups):
        print(
            f"  {knowledge_type}: "
            f"{len(groups[knowledge_type])}"
        )

    print()
    print("Eligible examples by category:")

    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"  {category}: {count}"
        )

    # ---------------------------------------------------------
    # Validate sufficient examples
    # ---------------------------------------------------------

    required_per_type = (
        K_DEV_PER_TYPE
        + K_TEST_PER_TYPE
    )

    for knowledge_type in [
        "factual",
        "conceptual",
        "procedural",
    ]:

        available = len(
            groups[knowledge_type]
        )

        if available < required_per_type:
            raise RuntimeError(
                f"Not enough eligible "
                f"{knowledge_type} examples. "
                f"Need at least "
                f"{required_per_type}, "
                f"found {available}."
            )

    # ---------------------------------------------------------
    # Shuffle independently within each knowledge type
    # ---------------------------------------------------------

    for knowledge_type in groups:
        rng.shuffle(
            groups[knowledge_type]
        )

    # ---------------------------------------------------------
    # K-dev
    #
    # 30 factual
    # 30 conceptual
    # 30 procedural
    #
    # K-dev is disjoint from K-test.
    # ---------------------------------------------------------

    k_dev = []

    remaining = {}

    for knowledge_type in [
        "factual",
        "conceptual",
        "procedural",
    ]:

        group = groups[
            knowledge_type
        ]

        k_dev.extend(
            group[:K_DEV_PER_TYPE]
        )

        remaining[
            knowledge_type
        ] = group[
            K_DEV_PER_TYPE:
        ]

    # ---------------------------------------------------------
    # K-test
    #
    # 100 factual
    # 100 conceptual
    # 100 procedural
    # ---------------------------------------------------------

    k_test_by_type = {}

    k_test = []

    for knowledge_type in [
        "factual",
        "conceptual",
        "procedural",
    ]:

        selected = remaining[
            knowledge_type
        ][:K_TEST_PER_TYPE]

        k_test_by_type[
            knowledge_type
        ] = selected

        k_test.extend(
            selected
        )

    # ---------------------------------------------------------
    # K-small
    #
    # Must be a PRESELECTED subset of K-test.
    #
    # 14 factual
    # 13 conceptual
    # 13 procedural
    # ---------------------------------------------------------

    k_small = []

    for knowledge_type in [
        "factual",
        "conceptual",
        "procedural",
    ]:

        n = K_SMALL_COUNTS[
            knowledge_type
        ]

        # Do not simply take the first n because those
        # positions came from the earlier shuffle.
        #
        # sample() makes the subset-selection step explicit.
        selected = rng.sample(
            k_test_by_type[
                knowledge_type
            ],
            n,
        )

        k_small.extend(
            selected
        )

    # ---------------------------------------------------------
    # Final deterministic ordering
    # ---------------------------------------------------------

    # The actual membership is random/frozen.
    # Sorting only makes the JSON easier to diff/read.
    k_dev = sorted(k_dev)
    k_test = sorted(k_test)
    k_small = sorted(k_small)

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------

    assert len(k_dev) == 90
    assert len(k_test) == 300
    assert len(k_small) == 40

    assert len(set(k_dev)) == 90
    assert len(set(k_test)) == 300
    assert len(set(k_small)) == 40

    # K-dev and K-test must not overlap.
    assert set(k_dev).isdisjoint(
        set(k_test)
    )

    # K-small must be entirely inside K-test.
    assert set(k_small).issubset(
        set(k_test)
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    splits = {
        "split_seed": SPLIT_SEED,

        "sampling": {
            "primary_grouping":
                "knowledge_type",

            "single_reference_only":
                True,

            "k_dev": {
                "total": 90,
                "per_knowledge_type": {
                    "factual": 30,
                    "conceptual": 30,
                    "procedural": 30,
                },
            },

            "k_test": {
                "total": 300,
                "per_knowledge_type": {
                    "factual": 100,
                    "conceptual": 100,
                    "procedural": 100,
                },
            },

            "k_small": {
                "total": 40,
                "per_knowledge_type": {
                    "factual": 14,
                    "conceptual": 13,
                    "procedural": 13,
                },
            },
        },

        "k_dev": k_dev,
        "k_test": k_test,
        "k_small": k_small,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            splits,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print(
        f"K-dev:   {len(k_dev)}"
    )
    print(
        f"K-test:  {len(k_test)}"
    )
    print(
        f"K-small: {len(k_small)}"
    )
    print("=" * 60)

    print()


if __name__ == "__main__":
    main()