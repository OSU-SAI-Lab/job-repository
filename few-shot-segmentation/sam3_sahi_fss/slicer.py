"""
sam3_sahi_fss/slicer.py — SAHI-style tiling helpers.

SAHI (Slicing Aided Hyper Inference) runs the detector on overlapping crops so that
small objects occupy a much larger fraction of the model's input resolution. Here the
detector is SAM 3 with a generic text concept; each slice is inferred independently
and the instances are mapped back into full-image coordinates, then de-duplicated
across the overlap regions with greedy box NMS.

No `sahi` dependency — the slicing/merging is a few dozen lines and keeps the package
self-contained (matching the rest of the repo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]   # x1, y1, x2, y2 (global image coords)


@dataclass
class Instance:
    """One SAM 3 instance in FULL-image coordinates.

    ``mask`` is cropped to ``box`` (shape (y2-y1, x2-x1)) rather than stored at full
    resolution — a "visual"/"everything" prompt can return hundreds of instances per
    image and a full H×W bool per instance blows up memory.
    """
    mask: np.ndarray
    box: Box
    score: float

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def generate_slices(width: int, height: int, slice_size: int,
                    overlap_ratio: float = 0.2) -> List[Box]:
    """Tile (width, height) with square windows of ``slice_size`` and given overlap."""
    if slice_size <= 0 or (slice_size >= width and slice_size >= height):
        return [(0, 0, width, height)]
    step = max(1, int(round(slice_size * (1.0 - overlap_ratio))))

    def starts(total: int) -> List[int]:
        if total <= slice_size:
            return [0]
        s = list(range(0, total - slice_size + 1, step))
        if s[-1] + slice_size < total:          # flush the last window to the edge
            s.append(total - slice_size)
        return s

    return [(x, y, min(x + slice_size, width), min(y + slice_size, height))
            for y in starts(height) for x in starts(width)]


def paint_union(instances: Sequence[Instance], height: int, width: int) -> np.ndarray:
    """OR every instance's cropped mask into one full-image bool mask."""
    out = np.zeros((height, width), dtype=bool)
    for ins in instances:
        x1, y1, x2, y2 = ins.box
        out[y1:y2, x1:x2] |= ins.mask
    return out


def _box_iou(a: Box, boxes: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = a
    ix1 = np.maximum(ax1, boxes[:, 0])
    iy1 = np.maximum(ay1, boxes[:, 1])
    ix2 = np.minimum(ax2, boxes[:, 2])
    iy2 = np.minimum(ay2, boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(1, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    return inter / (area_a + area_b - inter)


def nms(instances: Sequence[Instance], iou_threshold: float = 0.5) -> List[Instance]:
    """Greedy score-ordered box NMS — removes the duplicates from slice overlaps."""
    if len(instances) <= 1:
        return list(instances)
    order = sorted(range(len(instances)), key=lambda i: instances[i].score, reverse=True)
    boxes = np.array([instances[i].box for i in order], dtype=np.float64)
    alive = np.ones(len(order), dtype=bool)
    keep: List[Instance] = []
    for i in range(len(order)):
        if not alive[i]:
            continue
        alive[i] = False
        keep.append(instances[order[i]])
        rest = np.where(alive)[0]
        if rest.size == 0:
            continue
        ious = _box_iou(tuple(int(v) for v in boxes[i]), boxes[rest])
        alive[rest[ious >= iou_threshold]] = False
    return keep
