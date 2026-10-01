# src/rq2/check_determinism.py

"""
S7 determinism check: compare two regenerations made with identical
config and seeds.

    python -m src.rq2.check_determinism \
        --images DIR_A DIR_B \
        --caches CACHE_A.jsonl CACHE_B.jsonl

Images pass if bitwise identical, or else if the mean absolute pixel
difference is < 1 on the 0-255 scale (the plan's stated tolerance).
Rewrites must be identical. Exits 1 on failure.
"""

import argparse
import json
import sys
from pathlib import Path

MAD_TOLERANCE = 1.0


def mean_abs_diff(path_a, path_b) -> float:
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.int16)

    if a.shape != b.shape:
        return float("inf")

    return float(np.abs(a - b).mean())


def check_images(dir_a: Path, dir_b: Path) -> bool:

    paths_a = sorted(dir_a.glob("*.png"))

    if not paths_a:
        print(f"FAIL: no images in {dir_a}")
        return False

    worst = 0.0
    n_bitwise = 0
    ok = True

    for path_a in paths_a:
        path_b = dir_b / path_a.name

        if not path_b.exists():
            print(f"FAIL: {path_b} missing")
            ok = False
            continue

        mad = mean_abs_diff(path_a, path_b)

        # Pixel-identical; PNG bytes may differ only in metadata.
        bitwise = mad == 0.0

        n_bitwise += bitwise
        worst = max(worst, mad)

        print(f"  {path_a.name}: bitwise={bitwise} MAD={mad:.4f}")

    ok = ok and worst < MAD_TOLERANCE

    print(
        f"Images: {'PASS' if ok else 'FAIL'} | "
        f"{n_bitwise}/{len(paths_a)} bitwise identical | "
        f"worst MAD {worst:.4f} (tolerance < {MAD_TOLERANCE})"
    )

    return ok


def load_rewrites(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return {
            record["example_uid"]: record["rewrite"]
            for record in map(json.loads, filter(str.strip, f))
        }


def check_caches(path_a: Path, path_b: Path) -> bool:

    a = load_rewrites(path_a)
    b = load_rewrites(path_b)

    ok = set(a) == set(b)

    if not ok:
        print("FAIL: caches cover different examples")

    for uid in sorted(set(a) & set(b)):
        same = a[uid] == b[uid]
        ok = ok and same

        if not same:
            print(f"  {uid}: rewrites differ")
            print(f"    A: {a[uid]}")
            print(f"    B: {b[uid]}")

    print(
        f"Rewrites: {'PASS' if ok else 'FAIL'} | "
        f"{sum(a[u] == b.get(u) for u in a)}/{len(a)} identical"
    )

    return ok


def main():

    parser = argparse.ArgumentParser(
        description="S7 determinism check."
    )

    parser.add_argument(
        "--images",
        nargs=2,
        type=Path,
        metavar=("DIR_A", "DIR_B"),
    )

    parser.add_argument(
        "--caches",
        nargs=2,
        type=Path,
        metavar=("CACHE_A", "CACHE_B"),
    )

    args = parser.parse_args()

    if not args.images and not args.caches:
        parser.error("pass --images and/or --caches")

    ok = True

    if args.caches:
        ok = check_caches(*args.caches) and ok

    if args.images:
        ok = check_images(*args.images) and ok

    print("S7:", "PASS" if ok else "FAIL")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
