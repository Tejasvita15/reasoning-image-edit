import json
import random
from collections import defaultdict

from dataset import load_kris_bench
from taxonomy import (
    CATEGORY_TO_DIMENSION,
    DIMENSION_TO_KNOWLEDGE,
)


SEED = 42
PER_DIMENSION = 50

OUTPUT_PATH = "data/evaluation_set.json"


def balanced_sample_from_categories(
    category_examples,
    total,
    rng,
):
    """
    Select approximately equal numbers from every task category
    within a reasoning dimension.
    """

    categories = sorted(category_examples.keys())

    base_n = total // len(categories)
    remainder = total % len(categories)

    selected = []

    for i, category in enumerate(categories):

        n = base_n

        if i < remainder:
            n += 1

        candidates = category_examples[category].copy()
        rng.shuffle(candidates)

        if len(candidates) < n:
            raise ValueError(
                f"Category {category} only contains "
                f"{len(candidates)} examples, but {n} requested."
            )

        selected.extend(candidates[:n])

    rng.shuffle(selected)

    return selected


def main():

    rng = random.Random(SEED)

    dataset = load_kris_bench()

    # dimension -> category -> examples
    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for example in dataset:

        category = example["category"]

        dimension = CATEGORY_TO_DIMENSION[category]

        grouped[dimension][category].append(
            {
                "id": example["id"],
                "category": category,
                "reasoning_dimension": dimension,
                "knowledge_type":
                    DIMENSION_TO_KNOWLEDGE[dimension],
            }
        )

    evaluation_set = []

    for dimension in sorted(grouped.keys()):

        selected = balanced_sample_from_categories(
            grouped[dimension],
            PER_DIMENSION,
            rng,
        )

        evaluation_set.extend(selected)

    # Give every example a globally unique identifier,
    # because KRIS IDs repeat across categories.
    for example in evaluation_set:
        example["example_uid"] = (
            f"{example['category']}__{example['id']}"
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation_set,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved {len(evaluation_set)} examples "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()