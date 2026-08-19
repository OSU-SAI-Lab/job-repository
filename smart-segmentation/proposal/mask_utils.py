"""
proposal/mask_utils.py  –  Shared mask utilities for the segmentation pipeline.

The whole pipeline is mask-based: SAM3 emits per-instance binary masks, DINOv3
embeds the object pixels, cosine matching classifies, and NMS / evaluation all
operate on mask-IoU.  Bounding boxes exist ONLY as an internal mask-extent crop
for embedding (see ``mask_extent_crop``); they are never stored, emitted, NMS'd,
or evaluated.

Serialization contract
----------------------
Masks are serialized as COCO RLE dicts::

    {"size": [H, W], "counts": <bytes|str>}

produced by ``pycocotools.mask``.  ``counts`` is stored as a UTF-8 ``str`` so the
dicts survive JSON round-trips; ``decode_rle`` accepts either ``bytes`` or ``str``.

Public API
----------
encode_rle(mask_bool)            -> rle dict
decode_rle(rle)                  -> np.ndarray (H, W) bool
mask_iou(a, b)                   -> float                (a, b are rle or bool arrays)
mask_nms(rles, scores, iou_thr)  -> list[int]            (kept indices, score-desc)
mask_extent_crop(image, mask)    -> PIL.Image            (bg zeroed, cropped to extent)
paste_mask(mask, offset, HW)     -> np.ndarray (H, W) bool  (tile mask -> full-image)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


RLE = dict
MaskLike = Union[RLE, np.ndarray]


# ──────────────────────────────────────────────────────────────────────────────
# RLE encode / decode
# ──────────────────────────────────────────────────────────────────────────────

def encode_rle(mask_bool: np.ndarray) -> RLE:
    """Encode a 2-D boolean mask as a COCO RLE dict with a str ``counts``."""
    m = np.asfortranarray(mask_bool.astype(np.uint8))
    rle = coco_mask.encode(m)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(mask_bool.shape[0]), int(mask_bool.shape[1])], "counts": counts}


def decode_rle(rle: RLE) -> np.ndarray:
    """Decode a COCO RLE dict back to a 2-D boolean mask."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    r = {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
    return coco_mask.decode(r).astype(bool)


def _as_rle(m: MaskLike) -> RLE:
    """Coerce an rle dict or a boolean array into a canonical bytes-counts rle."""
    if isinstance(m, dict):
        counts = m["counts"]
        if isinstance(counts, str):
            counts = counts.encode("ascii")
        return {"size": [int(m["size"][0]), int(m["size"][1])], "counts": counts}
    return {
        "size": [int(m.shape[0]), int(m.shape[1])],
        "counts": coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))["counts"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mask IoU + mask NMS
# ──────────────────────────────────────────────────────────────────────────────

def mask_iou(a: MaskLike, b: MaskLike) -> float:
    """IoU between two masks (each an rle dict or boolean array)."""
    ra, rb = _as_rle(a), _as_rle(b)
    return float(coco_mask.iou([ra], [rb], [0])[0, 0])


def mask_nms(
    masks: Sequence[MaskLike],
    scores: Sequence[float],
    iou_threshold: float = 0.5,
) -> List[int]:
    """Greedy, score-descending mask NMS.

    Returns the indices to keep.  A candidate is suppressed when its mask-IoU with
    an already-kept, higher-scoring mask exceeds ``iou_threshold`` — the same
    "these are the same object, keep the confident one" rule as box NMS, but on
    pixels.
    """
    n = len(masks)
    if n == 0:
        return []
    rles = [_as_rle(m) for m in masks]
    order = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)

    keep: List[int] = []
    suppressed = [False] * n
    for idx_pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        # Batch-IoU of i against all remaining lower-scoring candidates.
        rest = [order[j] for j in range(idx_pos + 1, n) if not suppressed[order[j]]]
        if not rest:
            continue
        gt = [rles[r] for r in rest]
        ious = coco_mask.iou([rles[i]], gt, [0] * len(gt))[0]  # (len(rest),)
        for r, iou in zip(rest, ious):
            if float(iou) > iou_threshold:
                suppressed[r] = True
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Embedding framing + tile→global placement
# ──────────────────────────────────────────────────────────────────────────────

def mask_extent_crop(image: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    """Zero the background, then crop to the mask's tight pixel extent.

    This is the single source of the embedding framing used by both class supports
    and proposals, so their DINOv3 vectors live in the same space.  The extent
    rectangle is an internal rendering detail — it is never stored or output.

    Returns a 1x1 black image for an empty mask (degenerate; caller filters these).
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return Image.new("RGB", (1, 1))

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1

    arr = np.asarray(image.convert("RGB"))
    # Guard against a mask larger than the image (shouldn't happen, but be safe).
    H, W = arr.shape[:2]
    m = mask_bool[:H, :W]
    masked = np.zeros_like(arr)
    masked[m] = arr[m]
    crop = masked[y1:y2, x1:x2]
    return Image.fromarray(crop)


def paste_mask(
    mask_bool: np.ndarray,
    offset: Tuple[int, int],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """Place a tile-local mask onto a full-image-sized boolean canvas.

    Args:
        mask_bool: (h, w) tile-local mask.
        offset:    (ox, oy) top-left of the tile in full-image pixel coords.
        full_hw:   (H, W) of the full image.
    """
    H, W = full_hw
    ox, oy = int(offset[0]), int(offset[1])
    canvas = np.zeros((H, W), dtype=bool)
    h, w = mask_bool.shape
    y1, x1 = oy, ox
    y2, x2 = min(oy + h, H), min(ox + w, W)
    canvas[y1:y2, x1:x2] = mask_bool[: y2 - y1, : x2 - x1]
    return canvas
