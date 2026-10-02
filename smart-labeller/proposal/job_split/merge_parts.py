#!/usr/bin/env python3
"""Combine batch-size-1 part directories into one proposals folder.

Each part is a generate_proposals output. Files are named

    proposals/tensors/features_{embedder}_{proposer}_{timestamp}.npz
    proposals/generated_boxes/generated_boxes_{embedder}_{proposer}_{timestamp}.json

NPZ keys are ``{image}_features``, ``{image}_boxes``, and ``{image}_scores``.
Parts are concatenated in the order given. Every embedder file is included,
including the extra OWLv2 file an OWLv2 proposer writes.

The merged files are written to

    {output}/proposals/tensors/features_{embedder}_{proposer}_merged.npz
    {output}/proposals/generated_boxes/generated_boxes_{embedder}_{proposer}_merged.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

NPZ_RE = re.compile(
    r"^features_(?P<embedder>[a-z0-9]+)_(?P<proposer>[a-z0-9]+)_\d{8}_\d{6}\.npz$"
)
JSON_RE = re.compile(
    r"^generated_boxes_(?P<embedder>[a-z0-9]+)_(?P<proposer>[a-z0-9]+)_\d{8}_\d{6}\.json$"
)


def _files_by_kind(directory: Path, pattern: re.Pattern[str]) -> dict[tuple[str, str], list[Path]]:
    found: dict[tuple[str, str], list[Path]] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        match = pattern.match(path.name)
        if match is None:
            continue
        kind = (match.group("embedder"), match.group("proposer"))
        found.setdefault(kind, []).append(path)
    return found


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def _load_annotations(path: Path) -> list:
    with path.open() as handle:
        payload = json.load(handle)
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"{path} has no annotations list")
    return annotations


def merge_parts(
    part_dirs: list[Path],
    output_dir: Path,
    suffix: str = "merged",
) -> list[tuple[str, str]]:
    """Merge part directories into output_dir/proposals. Return the embedder pairs written.

    ``suffix`` is the filename tail. The default keeps the standalone merge names.
    A proposal run passes its timestamp so the merged files match a normal run.
    """
    if not part_dirs:
        raise ValueError("No part directories to merge")
    for part_dir in part_dirs:
        if not part_dir.is_dir():
            raise FileNotFoundError(f"Part directory does not exist: {part_dir}")

    json_maps = [
        _files_by_kind(part / "proposals" / "generated_boxes", JSON_RE) for part in part_dirs
    ]
    npz_maps = [
        _files_by_kind(part / "proposals" / "tensors", NPZ_RE) for part in part_dirs
    ]
    kinds: set[tuple[str, str]] = set()
    for mapping in json_maps:
        kinds.update(mapping)
    if not kinds:
        raise ValueError(f"No generated_boxes json files under {part_dirs[0]}")

    for index, mapping in enumerate(json_maps):
        for kind, paths in mapping.items():
            if len(paths) != 1:
                embedder, proposer = kind
                raise ValueError(
                    f"{part_dirs[index].name} has {len(paths)} json files for {embedder}_{proposer}"
                )
        absent = sorted(kinds - set(mapping))
        if absent:
            names = ", ".join(f"{embedder}_{proposer}" for embedder, proposer in absent)
            raise ValueError(f"{part_dirs[index].name} is missing json for {names}")

    for index, mapping in enumerate(npz_maps):
        for kind, paths in mapping.items():
            if len(paths) != 1:
                embedder, proposer = kind
                raise ValueError(
                    f"{part_dirs[index].name} has {len(paths)} npz files for {embedder}_{proposer}"
                )

    merged_arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {kind: {} for kind in kinds}
    merged_annotations: dict[tuple[str, str], list] = {kind: [] for kind in kinds}

    for index, (json_map, npz_map) in enumerate(zip(json_maps, npz_maps)):
        for kind in sorted(kinds):
            annotations = _load_annotations(json_map[kind][0])
            merged_annotations[kind].extend(annotations)
            npz_paths = npz_map.get(kind, [])
            if not npz_paths:
                if annotations:
                    embedder, proposer = kind
                    raise ValueError(
                        f"{part_dirs[index].name} has {len(annotations)} {embedder}_{proposer} "
                        "boxes but no npz"
                    )
                continue
            for key, value in _load_npz(npz_paths[0]).items():
                if key in merged_arrays[kind]:
                    raise ValueError(
                        f"Duplicate npz key {key} from {part_dirs[index].name}"
                    )
                merged_arrays[kind][key] = value

    tensors = output_dir / "proposals" / "tensors"
    boxes = output_dir / "proposals" / "generated_boxes"
    tensors.mkdir(parents=True, exist_ok=True)
    boxes.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, str]] = []
    for embedder, proposer in sorted(kinds):
        kind = (embedder, proposer)
        arrays = merged_arrays[kind]
        npz_path = tensors / f"features_{embedder}_{proposer}_{suffix}.npz"
        json_path = boxes / f"generated_boxes_{embedder}_{proposer}_{suffix}.json"
        if arrays:
            if npz_path.exists():
                npz_path.unlink()
            np.savez(npz_path, **arrays)
        elif npz_path.exists():
            npz_path.unlink()
        with json_path.open("w") as handle:
            json.dump({"annotations": merged_annotations[kind]}, handle, indent=2)
            handle.write("\n")
        image_count = sum(1 for key in arrays if key.endswith("_features"))
        print(
            f"merged {embedder}_{proposer}: {image_count} images, "
            f"{len(merged_annotations[kind])} boxes -> {output_dir}"
        )
        written.append(kind)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        merge_parts(args.parts, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
