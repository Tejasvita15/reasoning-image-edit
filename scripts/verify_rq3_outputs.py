#!/usr/bin/env python3
"""Verify the completed RQ3 branch artifacts without third-party packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "reports/rq3_shared_outputs.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported RQ3 sharing manifest")
    output_root = (root / "outputs/rq3").resolve()
    errors = []
    total = 0
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(output_root):
            raise ValueError(f"Manifest path escapes the RQ3 output directory: {relative}")
        if not path.is_file():
            errors.append({"path": relative, "error": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        size = path.stat().st_size
        if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            errors.append({"path": relative, "error": "size or SHA-256 differs"})
        total += size
    print(json.dumps({"status": "failed" if errors else "verified",
                      "files": len(manifest["files"]), "bytes": total,
                      "errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
