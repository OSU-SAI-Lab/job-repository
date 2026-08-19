"""
proposal/sam3_proposal.py  –  SAM3-based mask proposal generation.

SAM3 (text-prompted, concept-based) produces per-instance segmentation masks.
This module keeps those masks (the old box pipeline discarded them), embeds each
instance with DINOv3, places every mask into full-image coordinates, and dedupes
overlapping instances from SAHI tile seams with mask-IoU NMS.

Output per image
----------------
    {"features": ndarray (K, D)  float32,   # DINOv3, L2-normed
     "masks":    list[RLE] length K,        # COCO RLE, full-image sized
     "scores":   ndarray (K,)    float32}   # SAM3 instance scores

Embedding space
---------------
Embeddings come from the unified DINOv3 embedder (embedding_utils.py) via
``embed_masks`` — zero background, crop to the mask extent, DINOv3 pooler_output.
This matches the class-support embeddings from class_supports_utils.py exactly.
"""

from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OverlappingTileDataset import OverlappingTileDataset
from proposal.embedding_utils import get_embedder, DEVICE
from proposal.mask_utils import encode_rle, mask_nms, paste_mask

# ──────────────────────────────────────────────────────────────────────────────
# Model (loaded once)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_ID       = "facebook/sam3"
sam3_processor  = Sam3Processor.from_pretrained(_MODEL_ID)
sam3_model      = Sam3Model.from_pretrained(_MODEL_ID).to(DEVICE).eval()
logger.info(f"Loaded {_MODEL_ID} on {DEVICE}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_batch(
    tile_images: list,
    text_prompt: str,
    confidence_threshold: float,
    mask_threshold: float = 0.5,
):
    """Run SAM3 on a batch; return (masks_list, scores_list).

    masks_list[i]  : list of (h, w) boolean np arrays (tile-local, per instance)
    scores_list[i] : (M,) CPU float tensor
    """
    inputs = sam3_processor(
        images=tile_images,
        text=[text_prompt] * len(tile_images),
        return_tensors="pt",
    ).to(DEVICE)

    outputs = sam3_model(**inputs)

    target_sizes = inputs.get("original_sizes").tolist()
    batch_results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence_threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )

    masks_list  = []
    scores_list = []
    for result in batch_results:
        masks = result.get("masks", [])
        scores = result.get("scores", None)
        if len(masks) == 0:
            masks_list.append([])
            scores_list.append(torch.zeros((0,)))
            continue
        tile_masks = []
        for m in masks:
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            tile_masks.append(m.astype(bool))
        masks_list.append(tile_masks)
        scores_list.append(
            scores.float().cpu() if scores is not None else torch.ones(len(tile_masks))
        )

    return masks_list, scores_list


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_proposals_tiled(
    image_paths,
    text_prompt="visual",
    confidence_threshold=0.5,
    is_sahi=True,
    tile_size=960,
    overlap_ratio=0.2,
    batch_size=8,
    nms_iou_threshold=0.5,
    embedding_backend="dinov3",
    mask_threshold=0.5,
    model_id=None,              # unused for SAM3; kept for API compat
):
    """
    Generate mask proposals using SAM3.

    Returns
    -------
    dict: image_path (str) -> {"features": ndarray(K,D),
                                "masks":    list[RLE] length K,
                                "scores":   ndarray(K,)}
          or None for images with no detections.
    """
    image_paths = [str(p) for p in image_paths]

    # Load embedding model once (DINOv3 only).
    embedder = get_embedder(embedding_backend)
    logger.info(f"Embedding backend: {embedding_backend}")

    # Full-image size per image index, for placing tile masks into global coords.
    full_hw = {}
    for idx, p in enumerate(image_paths):
        with Image.open(p) as im:
            full_hw[idx] = (im.height, im.width)

    # Build dataset / loader.
    if is_sahi:
        dataset = OverlappingTileDataset(image_paths, tile_size, overlap_ratio)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=lambda x: x)
        logger.info(f"SAHI mode | tiles={len(dataset)}, batch_size={batch_size}")
    else:
        dataset = []
        for idx, p in enumerate(image_paths):
            img = Image.open(p).convert("RGB")
            dataset.append({
                "image":    img,
                "metadata": {"img_idx": idx,
                              "coords": torch.tensor([0, 0], dtype=torch.float32)},
            })
        loader = [dataset[i: i + batch_size] for i in range(0, len(dataset), batch_size)]
        logger.info(f"Whole-image mode | images={len(dataset)}, batch_size={batch_size}")

    accum = collections.defaultdict(lambda: {"masks": [], "scores": [], "feats": []})

    for batch_idx, batch in enumerate(loader):
        tile_images = [item["image"]                     for item in batch]
        coords_list = [item["metadata"]["coords"]        for item in batch]
        img_idxs    = [int(item["metadata"]["img_idx"])  for item in batch]

        masks_list, scores_list = _infer_batch(
            tile_images, text_prompt, confidence_threshold, mask_threshold
        )

        for tile_img, tile_masks, tile_scores, coords, img_idx in zip(
            tile_images, masks_list, scores_list, coords_list, img_idxs
        ):
            if len(tile_masks) == 0:
                continue

            ox, oy = int(coords[0]), int(coords[1])

            # Embed each instance from tile-local masks (object pixels only).
            feats_np = embedder.embed_masks(tile_img, tile_masks).numpy()  # (M, D)

            # Place each tile mask onto a full-image canvas → COCO RLE.
            H, W = full_hw[img_idx]
            for m in tile_masks:
                global_mask = paste_mask(m, (ox, oy), (H, W))
                accum[img_idx]["masks"].append(encode_rle(global_mask))
            accum[img_idx]["scores"].append(tile_scores.numpy())
            accum[img_idx]["feats"].append(feats_np)

        logger.debug(f"Batch {batch_idx + 1}/{len(loader)} done")

    results = {p: None for p in image_paths}
    for img_idx, a in accum.items():
        rles       = a["masks"]                       # list[RLE]
        scores_raw = np.hstack(a["scores"])           # (N,)
        feats_raw  = np.vstack(a["feats"])            # (N, D)

        keep = mask_nms(rles, scores_raw.tolist(), nms_iou_threshold)
        if not keep:
            continue

        results[image_paths[img_idx]] = {
            "features": feats_raw[keep],
            "masks":    [rles[i] for i in keep],
            "scores":   scores_raw[keep],
        }
        logger.info(f"{Path(image_paths[img_idx]).name}: "
                    f"{len(rles)} raw → {len(keep)} after mask-NMS")

    return results
