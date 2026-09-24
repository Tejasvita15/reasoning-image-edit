from dataset import (
    load_kris_bench,
    select_evaluation_examples,
)

from taxonomy import (
    get_reasoning_dimension,
    get_knowledge_type,
)


dataset = load_kris_bench()

examples = select_evaluation_examples(
    dataset,
    "data/evaluation_set.json",
)

print("Loaded:", len(examples))

example = examples[0]

category = str(example["category"])
example_id = str(example["id"])

print("UID:", f"{category}__{example_id}")
print("Category:", category)
print(
    "Dimension:",
    get_reasoning_dimension(category),
)
print(
    "Knowledge:",
    get_knowledge_type(category),
)
print(
    "Instruction:",
    example["instruction"],
)

image_keys = [
    "image",
    "image_1",
    "image_2",
    "image_3",
    "image_4",
]

num_images = sum(
    example[k] is not None
    for k in image_keys
)

print("Source images:", num_images)
print(
    "Has GT:",
    example["gt_image"] is not None,
)

print(
    "Raw image representation:",
    type(example["image"]),
)