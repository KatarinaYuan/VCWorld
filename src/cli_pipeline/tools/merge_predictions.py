#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge per-shard VCWorld inference outputs back to one final file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_nonempty_chunks(text: str, separator: str) -> int:
    return len([b.strip() for b in text.split(separator) if b.strip()])


def merge_outputs(
    *,
    manifest_path: Path,
    output_path: Path,
    pred_dir: Path | None = None,
    pred_prefix: str | None = None,
    strict_count_check: bool = True,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    separator = manifest["separator"]
    shards = manifest["shards"]
    merged_parts = []

    for shard in shards:
        sid = int(shard["shard_id"])
        expected = int(shard["num_blocks"])
        default_pred_file = Path(shard["predictions_file"])
        if pred_prefix is None:
            pred_file = default_pred_file if pred_dir is None else (pred_dir / default_pred_file.name)
        else:
            base_dir = default_pred_file.parent if pred_dir is None else pred_dir
            pred_file = base_dir / f"{pred_prefix}_{sid:03d}.txt"

        if not pred_file.exists():
            raise FileNotFoundError(f"Missing shard output: {pred_file}")

        content = pred_file.read_text(encoding="utf-8")
        got = _count_nonempty_chunks(content, separator)
        if strict_count_check and got != expected:
            raise RuntimeError(
                f"Shard {sid} count mismatch: expected {expected}, got {got}. File: {pred_file}"
            )
        merged_parts.append(content.rstrip() + "\n\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(merged_parts), encoding="utf-8")
    print(f"Saved merged outputs: {output_path}")
    print(f"Merged shards: {len(shards)} | Total prompts: {manifest['total_blocks']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge sharded inference outputs.")
    parser.add_argument("--manifest", required=True, help="Path to shard_manifest.json")
    parser.add_argument("--out", required=True, help="Merged final output txt path")
    parser.add_argument(
        "--pred-dir",
        default=None,
        help="Optional directory overriding predictions path from manifest",
    )
    parser.add_argument(
        "--pred-prefix",
        default=None,
        help="Optional prediction filename prefix when --pred-dir is set",
    )
    parser.add_argument(
        "--no-strict-count-check",
        action="store_true",
        help="Disable per-shard output count checks",
    )
    args = parser.parse_args()

    merge_outputs(
        manifest_path=Path(args.manifest),
        output_path=Path(args.out),
        pred_dir=None if args.pred_dir is None else Path(args.pred_dir),
        pred_prefix=args.pred_prefix,
        strict_count_check=not args.no_strict_count_check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
