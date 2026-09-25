# src/dataset.py

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import Image, load_dataset
from PIL import Image as PILImage

from src.taxonomy import (
    get_knowledge_type,
    get_reasoning_dimension,
    is_rq1_category_allowed,
)


KRIS_DATASET_ID = "Liang0223/KRIS_Bench"

IMAGE_COLUMNS = [
    "image",
    "image_1",
    "image_2",
    "image_3",
    "image_4",
]

ALL_IMAGE_COLUMNS = IMAGE_COLUMNS + [
    "gt_image",
]


def make_example_uid(
    category: str,
    example_id: str,
) -> str:
    """
    Construct a stable unique identifier for one KRIS-Bench example.

    KRIS IDs are not necessarily globally unique across categories,
    so category + id is used.
    """

    return f"{category}__{example_id}"


def decode_image(image_data):
    """
    Decode one Hugging Face Image(decode=False) object
    into a PIL RGB image.

    Expected image_data format:
        {
            "bytes": ...,
            "path": ...
        }
    """

    if image_data is None:
        return None

    if not isinstance(image_data, dict):
        raise TypeError(
            "Expected encoded image dictionary from "
            "datasets.Image(decode=False), "
            f"got {type(image_data)}."
        )

    if image_data.get("bytes") is not None:
        image = PILImage.open(
            BytesIO(image_data["bytes"])
        )

    elif image_data.get("path") is not None:
        image = PILImage.open(
            image_data["path"]
        )

    else:
        raise ValueError(
            "Image contains neither bytes nor path."
        )

    return image.convert("RGB")


def count_source_images(
    example: Dict[str, Any],
) -> int:
    """
    Count the number of source/reference images in one
    KRIS-Bench example without decoding them.
    """

    count = 0

    for key in IMAGE_COLUMNS:
        if example.get(key) is not None:
            count += 1

    return count


def is_single_reference_example(
    example: Dict[str, Any],
) -> bool:
    """
    Return True only when the example contains exactly
    one source/reference image.
    """

    return count_source_images(example) == 1


def is_rq1_eligible_example(
    example: Dict[str, Any],
) -> bool:
    """
    Determine whether an example is eligible for the RQ1
    Step1X/UniReason comparison.

    RQ1 uses only single-reference image-editing examples.
    """

    category = str(example["category"])

    if not is_rq1_category_allowed(category):
        return False

    if not is_single_reference_example(example):
        return False

    return True


def get_decoded_source_images(
    example: Dict[str, Any],
) -> List[PILImage.Image]:
    """
    Decode the source/reference images for one example.

    RQ1 should normally return exactly one image because
    multi-reference examples are excluded during split creation.
    """

    images = []

    for key in IMAGE_COLUMNS:
        image_data = example.get(key)

        if image_data is None:
            continue

        images.append(
            decode_image(image_data)
        )

    return images


def get_decoded_gt_image(
    example: Dict[str, Any],
):
    """
    Decode the ground-truth image if KRIS-Bench provides one.
    """

    gt = example.get("gt_image")

    if gt is None:
        return None

    return decode_image(gt)


def load_kris_bench(
    split: str = "train",
):
    """
    Load KRIS-Bench from Hugging Face.

    All image columns are immediately changed to decode=False so
    the dataset does not automatically decode images while
    filtering/selecting examples.
    """

    dataset = load_dataset(
        KRIS_DATASET_ID,
        split=split,
    )

    for column in ALL_IMAGE_COLUMNS:
        if column in dataset.column_names:
            dataset = dataset.cast_column(
                column,
                Image(decode=False),
            )

    return dataset


def prepare_example(
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add the metadata required by the RQ1 experiment while
    preserving the encoded image fields.

    Images remain encoded and are decoded only immediately
    before inference.
    """

    category = str(example["category"])
    example_id = str(example["id"])

    prepared = {
        "example_uid": make_example_uid(
            category,
            example_id,
        ),
        "id": example_id,
        "category": category,
        "reasoning_dimension":
            get_reasoning_dimension(category),
        "knowledge_type":
            get_knowledge_type(category),
        "instruction":
            example["instruction"],
        "explanation":
            example.get("explanation"),
        "num_source_images":
            count_source_images(example),
    }

    for column in ALL_IMAGE_COLUMNS:
        prepared[column] = example.get(column)

    return prepared


def build_rq1_eligible_indices(
    dataset,
) -> List[int]:
    """
    Return indices of all KRIS-Bench examples eligible for RQ1.

    This performs metadata/image-presence checks only and does
    not decode any images.
    """

    eligible_indices = []

    for idx, example in enumerate(dataset):

        if not is_rq1_eligible_example(example):
            continue

        eligible_indices.append(idx)

    return eligible_indices


def get_rq1_eligible_dataset(
    dataset,
):
    """
    Return the full RQ1-eligible KRIS-Bench pool.

    This should normally be used when creating splits.json.
    """

    eligible_indices = build_rq1_eligible_indices(
        dataset
    )

    return dataset.select(
        eligible_indices
    )


def load_splits(
    path: str,
) -> Dict[str, Any]:
    """
    Load the frozen RQ1 splits.json file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        splits = json.load(f)

    required_splits = [
        "k_test",
        "k_small",
    ]

    for split_name in required_splits:
        if split_name not in splits:
            raise ValueError(
                f"Missing required split in "
                f"{path}: {split_name}"
            )

    return splits


def get_split_uids(
    splits: Dict[str, Any],
    split_name: str,
) -> Set[str]:
    """
    Return the example UIDs belonging to one frozen split.
    """

    if split_name not in splits:
        raise ValueError(
            f"Unknown split: {split_name}"
        )

    split_items = splits[split_name]

    uids = set()

    for item in split_items:

        # Preferred format:
        # "category__id"
        if isinstance(item, str):
            uids.add(item)
            continue

        # Also allow:
        # {"id": "...", "category": "..."}
        if isinstance(item, dict):
            if (
                "id" not in item
                or "category" not in item
            ):
                raise ValueError(
                    f"Invalid split item: {item}"
                )

            uids.add(
                make_example_uid(
                    str(item["category"]),
                    str(item["id"]),
                )
            )

            continue

        raise ValueError(
            f"Unsupported split entry: {item}"
        )

    return uids


def select_split_examples(
    dataset,
    splits_path: str,
    split_name: str,
):
    """
    Select one frozen RQ1 split from KRIS-Bench.

    The selection is based on example_uid and preserves
    lazy image decoding.

    Supported RQ1 splits:
        - k_test
        - k_small
    """

    splits = load_splits(
        splits_path
    )

    selected_uids = get_split_uids(
        splits,
        split_name,
    )

    selected_indices = []
    found_uids = set()

    # We only need ID/category/source-image presence here.
    # Images remain encoded.
    for idx, example in enumerate(dataset):

        category = str(
            example["category"]
        )

        example_id = str(
            example["id"]
        )

        uid = make_example_uid(
            category,
            example_id,
        )

        if uid not in selected_uids:
            continue

        # Frozen splits should contain only eligible items.
        # Still validate this to catch split-generation mistakes.
        if not is_rq1_eligible_example(example):
            raise RuntimeError(
                f"Split {split_name} contains "
                f"ineligible example: {uid}. "
                f"Source images="
                f"{count_source_images(example)}"
            )

        selected_indices.append(idx)
        found_uids.add(uid)

    missing = selected_uids - found_uids

    if missing:
        preview = sorted(missing)[:10]

        raise RuntimeError(
            f"Could not find {len(missing)} examples "
            f"from split '{split_name}'. "
            f"First missing IDs: {preview}"
        )

    selected_dataset = dataset.select(
        selected_indices
    )

    if len(selected_dataset) != len(selected_uids):
        raise RuntimeError(
            f"Expected {len(selected_uids)} examples "
            f"for split '{split_name}', "
            f"but selected {len(selected_dataset)}."
        )

    return selected_dataset


def print_example_info(
    example: Dict[str, Any],
) -> None:
    """
    Convenience function for inspecting one KRIS-Bench example.
    """

    prepared = prepare_example(
        example
    )

    print("=" * 60)

    print(
        "UID:",
        prepared["example_uid"],
    )

    print(
        "ID:",
        prepared["id"],
    )

    print(
        "Category:",
        prepared["category"],
    )

    print(
        "Reasoning dimension:",
        prepared["reasoning_dimension"],
    )

    print(
        "Knowledge type:",
        prepared["knowledge_type"],
    )

    print(
        "Instruction:",
        prepared["instruction"],
    )

    print(
        "Number of source images:",
        prepared["num_source_images"],
    )

    print(
        "RQ1 eligible:",
        is_rq1_eligible_example(example),
    )

    if prepared["explanation"] is not None:
        print(
            "Explanation:",
            prepared["explanation"],
        )

    print("=" * 60)