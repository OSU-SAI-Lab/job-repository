"""
proposal/sam_amg_proposal.py  –  SAM Automatic-Mask-Generation ("segment everything") proposer.

Why this exists
---------------
The SAM3 proposer (sam3_proposal.py) runs SAM3 in TEXT-CONCEPT mode and returns
only the few coarse instances matching a prompt — useless for a diffuse texture
like corn residue, where each image holds hundreds of small irregular pieces.

This proposer instead runs SAM in AUTOMATIC mode: a dense grid of point prompts
produces hundreds of CLASS-AGNOSTIC masks per image ("segment everything"). The
downstream DINOv3 + cosine classifier then decides which masks are residue. This
matches the per-piece granularity of the hand-drawn GT.

Interface
---------
Exposes generate_proposals_tiled(...) with the SAME signature/return as
sam3_proposal.generate_proposals_tiled, so generate_proposals.py can dispatch it
interchangeably. SAM provides boxes+masks only; embeddings come from the unified
embedder in embedding_utils.py (DINOv3 by default), keeping proposal and
class-support vectors in the same space.

Tunable via env vars (defaults in []):
  SAM_AMG_MODEL          HF model id for AMG                 [facebook/sam-vit-base]
  SAM_AMG_POINTS         points_per_side for the grid       [32]
  SAM_AMG_POINTS_BATCH   points_per_batch (memory/speed)    [64]
  SAM_AMG_PRED_IOU       pred_iou_thresh (mask quality)     [0.80]
  SAM_AMG_STABILITY      stability_score_thresh             [0.85]
  SAM_AMG_MIN_AREA       drop masks smaller than N px       [30]
  SAM_AMG_MAX_AREA_FRAC  drop masks bigger than frac*tile   [0.5]
"""

from __future__ import annotations

import logging
import os
import sys
import collections
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from torch.utils.data import DataLoader
from transformers import pipeline
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OverlappingTileDataset import OverlappingTileDataset
from proposal.embedding_utils import get_embedder, DEVICE

# ──────────────────────────────────────────────────────────────────────────────
# Config (env-overridable)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_ID        = os.environ.get("SAM_AMG_MODEL", "facebook/sam-vit-base")
POINTS_PER_SIDE  = int(os.environ.get("SAM_AMG_POINTS", "32"))
POINTS_PER_BATCH = int(os.environ.get("SAM_AMG_POINTS_BATCH", "64"))
PRED_IOU_THRESH  = float(os.environ.get("SAM_AMG_PRED_IOU", "0.80"))
STABILITY_THRESH = float(os.environ.get("SAM_AMG_STABILITY", "0.85"))
MIN_AREA         = int(os.environ.get("SAM_AMG_MIN_AREA", "30"))
MAX_AREA_FRAC    = float(os.environ.get("SAM_AMG_MAX_AREA_FRAC", "0.5"))

# ──────────────────────────────────────────────────────────────────────────────
# Model (loaded once) — transformers "mask-generation" pipeline = SAM AMG
# ──────────────────────────────────────────────────────────────────────────────

_device_arg = 0 if (isinstance(DEVICE, str) and DEVICE.startswith("cuda")) else -1
amg = pipeline("mask-generation", model=_MODEL_ID, device=_device_arg)
logger.info(f"Loaded SAM AMG '{_MODEL_ID}' on {DEVICE} "
            f"(points_per_side={POINTS_PER_SIDE}, pred_iou={PRED_IOU_THRESH}, "
            f"stability={STABILITY_THRESH})")


def _mask_to_box(mask: np.ndarray):
    """[x1,y1,x2,y2] from a boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@torch.no_grad()
def _infer_tile(tile_img: Image.Image):
    """Run automatic mask generation on a single tile.

    Returns (boxes [N,4] tensor, scores [N] tensor, masks list[bool HxW]) in
    tile-local coordinates, after area filtering.
    """
    out = amg(
        tile_img,
        points_per_side=POINTS_PER_SIDE,
        points_per_batch=POINTS_PER_BATCH,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
    )
    masks  = out.get("masks", []) or []
    scores = out.get("scores", None)
    if scores is None:
        scores = [1.0] * len(masks)
    elif hasattr(scores, "tolist"):
        scores = scores.tolist()

    W, H = tile_img.size
    tile_area = float(W * H)

    boxes, kept_scores, kept_masks = [], [], []
    for m, s in zip(masks, scores):
        m = np.asarray(m, dtype=bool)
        area = int(m.sum())
        if area < MIN_AREA or area > MAX_AREA_FRAC * tile_area:
            continue
        box = _mask_to_box(m)
        if box is None:
            continue
        boxes.append(box)
        kept_scores.append(float(s))
        kept_masks.append(m)

    if not boxes:
        return torch.zeros((0, 4)), torch.zeros((0,)), []
    return (torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(kept_scores, dtype=torch.float32),
            kept_masks)


def _global_nms(boxes, scores, feats, iou_threshold=0.5):
    if len(boxes) == 0:
        return None, None, None, []
    b = torch.from_numpy(boxes).float()
    s = torch.from_numpy(scores).float()
    f = torch.from_numpy(feats).float()
    keep = torchvision_nms(b, s, iou_threshold)
    return f[keep].numpy(), b[keep].numpy(), s[keep].numpy(), keep.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Public API (signature matches sam3_proposal.generate_proposals_tiled)
# ──────────────────────────────────────────────────────────────────────────────

def generate_proposals_tiled(
    image_paths,
    text_prompt="visual",        # ignored — AMG is class-agnostic
    confidence_threshold=0.5,    # ignored — use SAM_AMG_PRED_IOU/STABILITY instead
    is_sahi=False,
    tile_size=960,
    overlap_ratio=0.2,
    batch_size=8,                # tiles processed per loop chunk (AMG runs 1 tile at a time)
    nms_iou_threshold=0.5,
    embedding_backend="dinov3",
    model_id=None,               # unused; AMG model set via SAM_AMG_MODEL
    mask_background="zero",
):
    """Automatic "segment everything" proposals + masked embeddings.

    Returns dict: image_path -> {"features","boxes","scores","masks"} or None.
    """
    image_paths = [str(p) for p in image_paths]

    embedder = get_embedder(embedding_backend)
    logger.info(f"Embedding backend: {embedding_backend}")

    full_image_sizes = {}
    for idx, p in enumerate(image_paths):
        full_image_sizes[idx] = Image.open(p).size  # (W, H)

    if is_sahi:
        dataset = OverlappingTileDataset(image_paths, tile_size, overlap_ratio)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: x)
        logger.info(f"SAHI mode | tiles={len(dataset)}, batch_size={batch_size}")
    else:
        dataset = []
        for idx, p in enumerate(image_paths):
            dataset.append({
                "image":    Image.open(p).convert("RGB"),
                "metadata": {"img_idx": idx,
                              "coords": torch.tensor([0, 0], dtype=torch.float32)},
            })
        loader = [dataset[i: i + batch_size] for i in range(0, len(dataset), batch_size)]
        logger.info(f"Whole-image mode | images={len(dataset)}, batch_size={batch_size}")

    accum = collections.defaultdict(lambda: {"boxes": [], "scores": [], "feats": [], "masks": []})

    for batch_idx, batch in enumerate(loader):
        for item in batch:
            tile_img = item["image"].convert("RGB")
            coords   = item["metadata"]["coords"]
            img_idx  = int(item["metadata"]["img_idx"])

            tile_boxes, tile_scores, tile_masks = _infer_tile(tile_img)
            if tile_boxes.numel() == 0:
                continue

            ox, oy = float(coords[0]), float(coords[1])
            global_boxes = (tile_boxes + torch.tensor([ox, oy, ox, oy])).numpy()
            local_boxes  = tile_boxes.tolist()

            if mask_background != "none" and len(tile_masks) == len(local_boxes):
                feats_np = embedder.embed_masks(
                    tile_img, local_boxes, tile_masks, background=mask_background
                ).numpy()
            else:
                feats_np = embedder.embed_boxes(tile_img, local_boxes).numpy()

            accum[img_idx]["boxes"].append(global_boxes)
            accum[img_idx]["scores"].append(tile_scores.numpy())
            accum[img_idx]["feats"].append(feats_np)
            for mask in tile_masks:
                accum[img_idx]["masks"].append((mask, int(ox), int(oy)))

        logger.debug(f"Batch {batch_idx + 1}/{len(loader)} done")

    results = {p: None for p in image_paths}
    for img_idx, a in accum.items():
        boxes_raw  = np.vstack(a["boxes"])
        scores_raw = np.hstack(a["scores"])
        feats_raw  = np.vstack(a["feats"])
        masks_raw  = a["masks"]

        feats_f, boxes_f, scores_f, keep_idxs = _global_nms(
            boxes_raw, scores_raw, feats_raw, nms_iou_threshold
        )
        if boxes_f is not None:
            full_w, full_h = full_image_sizes[img_idx]
            final_masks = []
            for ki in keep_idxs:
                local_mask, ox, oy = masks_raw[ki]
                full_mask = np.zeros((full_h, full_w), dtype=bool)
                mh, mw = local_mask.shape
                y2 = min(oy + mh, full_h)
                x2 = min(ox + mw, full_w)
                full_mask[oy:y2, ox:x2] = local_mask[:y2 - oy, :x2 - ox]
                final_masks.append(full_mask)

            results[image_paths[img_idx]] = {
                "features": feats_f,
                "boxes":    boxes_f,
                "scores":   scores_f,
                "masks":    final_masks,
            }
            logger.info(f"{Path(image_paths[img_idx]).name}: "
                        f"{len(boxes_raw)} raw → {len(boxes_f)} after NMS")

    return results
