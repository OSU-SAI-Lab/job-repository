"""Run equal image chunks and merge them into one proposals directory."""

from __future__ import annotations

import shutil
from pathlib import Path

from merge_parts import merge_parts
from split_estimate import SplitPlan


def slice_items(items: list, n: int) -> list[list]:
    """Split items into n pieces whose lengths differ by at most one."""
    if n < 1 or n > len(items):
        raise ValueError(f"Cannot split {len(items)} items into {n} chunks")
    base, extra = divmod(len(items), n)
    chunks: list[list] = []
    start = 0
    for index in range(n):
        size = base + (1 if index < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def execute_chunks(items: list, plan: SplitPlan, output_dir: Path, timestamp: str, run_chunk) -> None:
    """Call run_chunk for each piece, merge into output_dir, then remove the parts.

    ``items`` is in the same order the plan was built from. ``run_chunk(chunk, part_dir)``
    writes a normal generate_proposals output into ``part_dir`` using ``timestamp``
    in the filenames. The merged files use that same timestamp.
    """
    if len(items) != sum(plan.chunk_image_counts):
        raise ValueError(
            f"Plan expects {sum(plan.chunk_image_counts)} images, got {len(items)}"
        )
    output_dir = Path(output_dir)
    parts_root = output_dir / "parts"
    if parts_root.exists():
        shutil.rmtree(parts_root)
    part_dirs: list[Path] = []
    for index, chunk in enumerate(slice_items(items, plan.n)):
        part_dir = parts_root / f"part{index:02d}"
        part_dir.mkdir(parents=True)
        run_chunk(chunk, part_dir)
        part_dirs.append(part_dir)
    merge_parts(part_dirs, output_dir, suffix=timestamp)
    shutil.rmtree(parts_root)
