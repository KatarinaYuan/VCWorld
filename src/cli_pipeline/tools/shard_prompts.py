#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split a VCWorld prompts file into N shard files for parallel inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

PROMPT_SEPARATOR = "================================================================================"


def _load_blocks(prompts_path: Path) -> List[str]:
    content = prompts_path.read_text(encoding="utf-8")
    return [b.strip() for b in content.split(PROMPT_SEPARATOR) if b.strip()]


def _dump_blocks(blocks: List[str], out_path: Path) -> None:
    if not blocks:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", encoding="utf-8") as f:
        for block in blocks:
            f.write(block)
            f.write("\n\n" + PROMPT_SEPARATOR + "\n\n")


def split_prompts(prompts: Path, out_dir: Path, num_shards: int, shard_prefix: str) -> Path:
    blocks = _load_blocks(prompts)
    total = len(blocks)
    if total == 0:
        raise RuntimeError(f"No prompt block found in {prompts}")
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")

    out_dir.mkdir(parents=True, exist_ok=True)

    base = total // num_shards
    rem = total % num_shards
    start = 0
    shards = []
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        end = start + size
        shard_blocks = blocks[start:end]
        shard_file = out_dir / f"{shard_prefix}_{i:03d}.txt"
        _dump_blocks(shard_blocks, shard_file)
        shards.append(
            {
                "shard_id": i,
                "start_idx": start,
                "end_idx": end,
                "num_blocks": size,
                "prompts_file": str(shard_file),
                "predictions_file": str(out_dir / f"predictions_{i:03d}.txt"),
            }
        )
        start = end

    manifest = {
        "source_prompts": str(prompts),
        "separator": PROMPT_SEPARATOR,
        "total_blocks": total,
        "num_shards": num_shards,
        "shards": shards,
    }
    manifest_path = out_dir / "shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Human-readable range table for quick sanity checks.
    range_tsv = out_dir / "shard_ranges.tsv"
    with range_tsv.open("w", encoding="utf-8") as f:
        f.write("shard_id\tstart_idx_0based\tend_idx_exclusive_0based\tstart_prompt_1based\tend_prompt_1based\tnum_blocks\n")
        for s in shards:
            start_idx = int(s["start_idx"])
            end_idx = int(s["end_idx"])
            num_blocks = int(s["num_blocks"])
            if num_blocks == 0:
                start_prompt = "NA"
                end_prompt = "NA"
            else:
                start_prompt = str(start_idx + 1)
                end_prompt = str(end_idx)
            f.write(
                f"{s['shard_id']}\t{start_idx}\t{end_idx}\t{start_prompt}\t{end_prompt}\t{num_blocks}\n"
            )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Split prompts file for parallel inference.")
    parser.add_argument("--prompts", required=True, help="Input prompts txt")
    parser.add_argument("--out-dir", required=True, help="Output directory for shard prompts + manifest")
    parser.add_argument("--num-shards", type=int, default=32, help="Number of shards")
    parser.add_argument("--shard-prefix", default="prompts", help="Shard filename prefix")
    args = parser.parse_args()

    manifest_path = split_prompts(
        prompts=Path(args.prompts),
        out_dir=Path(args.out_dir),
        num_shards=args.num_shards,
        shard_prefix=args.shard_prefix,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Total prompts: {manifest['total_blocks']}")
    print(f"Num shards: {manifest['num_shards']}")
    print("Shard ranges:")
    for s in manifest["shards"]:
        start_idx = int(s["start_idx"])
        end_idx = int(s["end_idx"])
        num_blocks = int(s["num_blocks"])
        if num_blocks == 0:
            human_range = "prompt NA..NA"
        else:
            human_range = f"prompt {start_idx + 1}..{end_idx}"
        print(
            f"  shard {int(s['shard_id']):03d}: "
            f"idx[{start_idx}:{end_idx}) | {human_range} | n={num_blocks}"
        )
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved range table: {Path(args.out_dir) / 'shard_ranges.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
