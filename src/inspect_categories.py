from collections import Counter

from dataset import load_kris_bench


dataset = load_kris_bench()

counts = Counter(dataset["category"])

print(f"Total examples: {len(dataset)}")
print(f"Number of categories: {len(counts)}")
print()

for category, count in sorted(counts.items()):
    print(f"{category:35s} {count}")