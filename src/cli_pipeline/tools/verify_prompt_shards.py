#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that sharded prompt files reconstruct back to the original prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

PROMPT_SEPARATOR = "================================================================================"


def _split_blocks(text: str) -> List[str]:
    return [b.strip() for b in text.split(PROMPT_SEPARATOR) if b.strip()]


def _join_blocks(blocks: List[str]) -> str:
    if not blocks:
        return ""
    return "".join(f"{b}\n\n{PROMPT_SEPARATOR}\n\n" for b in blocks)


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_shards(
    *,
    original_prompts: Path,
    manifest_path: Path,
    reconstructed_out: Path,
    strict_bytes: bool,
) -> None:
    manifest = _load_manifest(manifest_path)
    shards = manifest["shards"]
    separator = manifest.get("separator", PROMPT_SEPARATOR)
    if separator != PROMPT_SEPARATOR:
        raise RuntimeError(f"Unexpected separator in manifest: {separator!r}")

    all_blocks: List[str] = []
    for shard in shards:
        shard_file = Path(shard["prompts_file"])
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_file}")
        shard_blocks = _split_blocks(shard_file.read_text(encoding="utf-8"))
        expected = int(shard["num_blocks"])
        got = len(shard_blocks)
        if got != expected:
            raise RuntimeError(
                f"Shard {int(shard['shard_id']):03d} mismatch: expected {expected} blocks, got {got}. "
                f"File: {shard_file}"
            )
        all_blocks.extend(shard_blocks)

    reconstructed_text = _join_blocks(all_blocks)
    reconstructed_out.parent.mkdir(parents=True, exist_ok=True)
    reconstructed_out.write_text(reconstructed_text, encoding="utf-8")

    original_text = original_prompts.read_text(encoding="utf-8")
    original_blocks = _split_blocks(original_text)

    if len(original_blocks) != len(all_blocks):
        raise RuntimeError(
            f"Total block count mismatch: original={len(original_blocks)}, reconstructed={len(all_blocks)}"
        )

    mismatch_idx = None
    for i, (a, b) in enumerate(zip(original_blocks, all_blocks)):
        if a != b:
            mismatch_idx = i
            break

    if mismatch_idx is not None:
        prompt_no = mismatch_idx + 1
        raise RuntimeError(
            f"Block content mismatch at prompt #{prompt_no} (0-based idx={mismatch_idx}). "
            f"Check reconstructed file: {reconstructed_out}"
        )

    if strict_bytes and original_text != reconstructed_text:
        raise RuntimeError(
            "Logical block comparison passed, but strict byte-level text differs "
            "(usually line-ending/extra-blank differences)."
        )

    print("Verification passed.")
    print(f"Original prompts: {original_prompts}")
    print(f"Manifest: {manifest_path}")
    print(f"Reconstructed prompts: {reconstructed_out}")
    print(f"Total prompts: {len(all_blocks)}")
    if not strict_bytes and original_text != reconstructed_text:
        print("Note: byte-level text differs, but block-level content is identical.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify prompt shards by reconstructing and comparing.")
    parser.add_argument("--original-prompts", required=True, help="Original prompts txt path")
    parser.add_argument("--manifest", required=True, help="shard_manifest.json path")
    parser.add_argument(
        "--reconstructed-out",
        required=True,
        help="Output path for reconstructed prompts file",
    )
    parser.add_argument(
        "--strict-bytes",
        action="store_true",
        help="Require exact byte-level equality with original prompts",
    )
    args = parser.parse_args()

    verify_shards(
        original_prompts=Path(args.original_prompts),
        manifest_path=Path(args.manifest),
        reconstructed_out=Path(args.reconstructed_out),
        strict_bytes=args.strict_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

