from dataset import (
    load_kris_bench,
    select_evaluation_examples,
    get_decoded_source_images,
    get_decoded_gt_image,
)


dataset = load_kris_bench()

examples = select_evaluation_examples(
    dataset,
    "data/evaluation_set.json",
)

example = examples[0]

images = get_decoded_source_images(example)
gt = get_decoded_gt_image(example)

print("Example:", example["category"], example["id"])
print("Source images:", len(images))

for i, image in enumerate(images):
    print(
        f"Image {i}:",
        image.size,
        image.mode,
    )

print(
    "GT:",
    None if gt is None else gt.size
)