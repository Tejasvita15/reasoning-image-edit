# src/dataset.py

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_dataset, Image
from io import BytesIO
from PIL import Image as PILImage

from src.taxonomy import (get_reasoning_dimension,get_knowledge_type,)

KRIS_DATASET_ID = "Liang0223/KRIS_Bench"
IMAGE_COLUMNS = [
    "image",
    "image_1",
    "image_2",
    "image_3",
    "image_4",
]


def decode_image(image_data):
    """
    Decode one Hugging Face Image(decode=False) object.
    """

    if image_data is None:
        return None

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


def get_decoded_source_images(example):
    """
    Decode only the source images for a single KRIS-Bench example.
    """

    images = []

    for key in IMAGE_COLUMNS:
        image_data = example.get(key)

        if image_data is not None:
            images.append(
                decode_image(image_data)
            )

    return images


def get_decoded_gt_image(example):
    """
    Decode ground-truth image if available.
    """

    gt = example.get("gt_image")

    if gt is None:
        return None

    return decode_image(gt)

def load_kris_bench(split: str = "train"):
    """
    Load KRIS-Bench from Hugging Face.
    """
    dataset = load_dataset(
        KRIS_DATASET_ID,
        split=split,
    )

    return dataset


def load_evaluation_set(path: str) -> List[Dict[str, Any]]:
    """
    Load the frozen evaluation_set.json file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation set not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluation_lookup(
    evaluation_set: List[Dict[str, Any]]
):
    """
    Build a lookup of selected KRIS-Bench examples.

    Expected evaluation_set.json format:
    [
        {
            "id": "...",
            "category": "..."
        },
        ...
    ]
    """
    lookup = set()

    for item in evaluation_set:
        lookup.add(
            (
                str(item["id"]),
                str(item["category"]),
            )
        )

    return lookup


def select_evaluation_examples(
    dataset,
    evaluation_set_path: str,
):
    """
    Select the frozen evaluation set without decoding images.

    Returns a Hugging Face Dataset, not a Python list.
    Images remain stored as {bytes/path} and are decoded only
    when explicitly requested later.
    """

    evaluation_set = load_evaluation_set(
        evaluation_set_path
    )

    lookup = {
        (
            str(item["id"]),
            str(item["category"]),
        )
        for item in evaluation_set
    }

    # Metadata-only view for matching.
    metadata = dataset.select_columns(
        ["id", "category"]
    )

    selected_indices = []

    for idx, example in enumerate(metadata):
        key = (
            str(example["id"]),
            str(example["category"]),
        )

        if key in lookup:
            selected_indices.append(idx)

    if len(selected_indices) != len(evaluation_set):
        raise RuntimeError(
            f"Expected {len(evaluation_set)} examples, "
            f"but found {len(selected_indices)}."
        )

    selected_dataset = dataset.select(
        selected_indices
    )

    # VERY IMPORTANT:
    # Keep image fields encoded rather than converting them
    # automatically to PIL images.
    image_columns = [
        "image",
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "gt_image",
    ]

    for column in image_columns:
        if column in selected_dataset.column_names:
            selected_dataset = selected_dataset.cast_column(
                column,
                Image(decode=False),
            )

    return selected_dataset
def get_source_images(
    example: Dict[str, Any]
):
    """
    Collect all available input/reference images for an example.

    KRIS-Bench examples may contain multiple image fields.
    """

    image_keys = [
        "image",
        "image_1",
        "image_2",
        "image_3",
        "image_4",
    ]

    images = []

    for key in image_keys:

        if key not in example:
            continue

        image = example[key]

        if image is None:
            continue

        images.append(image.convert("RGB"))

    return images


def print_example_info(
    example: Dict[str, Any]
):
    """
    Convenience function for inspecting KRIS-Bench records.
    """

    print("=" * 60)

    print("ID:", example.get("id"))
    print("Category:", example.get("category"))
    print("Instruction:", example.get("instruction"))

    if "explanation" in example:
        print("Explanation:", example.get("explanation"))

    print(
        "Number of source images:",
        len(get_source_images(example)),
    )

    print("=" * 60)
    
def prepare_example(example):
    """
    Prepare one KRIS-Bench example without keeping unnecessary
    decoded image copies in memory.
    """

    category = str(example["category"])
    example_id = str(example["id"])

    return {
        "example_uid": f"{category}__{example_id}",
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

        "image":
            example.get("image"),

        "image_1":
            example.get("image_1"),

        "image_2":
            example.get("image_2"),

        "image_3":
            example.get("image_3"),

        "image_4":
            example.get("image_4"),

        "gt_image":
            example.get("gt_image"),
    }