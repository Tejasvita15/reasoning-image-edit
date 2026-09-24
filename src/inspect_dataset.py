from dataset import (
    load_kris_bench,
    get_source_images,
    print_example_info,
)


dataset = load_kris_bench()

print(dataset)

print("\nColumns:")
print(dataset.column_names)

print("\nFirst example:")
print_example_info(dataset[0])

print("\nRaw keys:")
print(dataset[0].keys())