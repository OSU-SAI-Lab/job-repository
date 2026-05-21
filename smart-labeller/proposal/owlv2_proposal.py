"""
proposal/owlv2_proposal.py  –  OWLv2-based proposal generation.

Embedding space
---------------
Native OWLv2 anchor features come from model.image_embedder (patch tokens,
dim=768 for large model's hidden size BEFORE the class head).  However, for
cosine-similarity classification we need embeddings from vision_model.pooler_output
(dim=1024).  Therefore:

  embedding_backend='owlv2'  → re-crops each box and embeds via OWLv2Embedder
                                (vision_model.pooler_output, dim=1024)
  embedding_backend='dinov3' → re-crops and embeds via DINOv3Embedder
  embedding_backend='bioclip'→ re-crops and embeds via BioCLIPEmbedder

The native patch-token features (768-dim) are used ONLY for objectness scoring
and box prediction; they are never saved as the final proposal embeddings.

Memory / speed optimisations
-----------------------------
- Model loaded once, reused across all images.
- Pixel values stay on GPU throughout the batch loop.
- A single .cpu() + .numpy() call per batch item after NMS.
- No unnecessary tensor round-trips between CPU and GPU.
"""

from __future__ import annotations

import logging
import sys
import collections
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from torch.utils.data import DataLoader
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OverlappingTileDataset import OverlappingTileDataset
from proposal.embedding_utils import get_embedder, DEVICE, TORCH_DTYPE

# ──────────────────────────────────────────────────────────────────────────────
# Model initialisation (lazy, cached)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "google/owlv2-large-patch14-ensemble"

_processor: Optional[Owlv2Processor]            = None
_model:     Optional[Owlv2ForObjectDetection]   = None
_loaded_id: Optional[str]                       = None


def _load_owlv2(model_id: str):
    global _processor, _model, _loaded_id
    if _loaded_id == model_id:
        return _processor, _model
    logger.info(f"Loading {model_id}...")
    _processor = Owlv2Processor.from_pretrained(model_id)
    _model = (
        Owlv2ForObjectDetection
        .from_pretrained(model_id, torch_dtype=TORCH_DTYPE,
                         device_map="auto" if DEVICE == "cuda" else "cpu")
        .eval()
    )
    _loaded_id = model_id
    return _processor, _model


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_batch(
    processor: Owlv2Processor,
    model: Owlv2ForObjectDetection,
    pil_tiles: List[Image.Image],
    objectness_threshold: float,
    nms_iou_threshold: float,
):
    """
    Run OWLv2 objectness inference on a batch of PIL tiles.

    Returns per-tile lists of:
        boxes_list  : list of (M, 4) CPU float32 tensors  [x1,y1,x2,y2] pixels
        scores_list : list of (M,)   CPU float32 tensors  objectness scores
        shapes      : list of (H, W) tuples
    """
    shapes = [(img.height, img.width) for img in pil_tiles]

    inputs = processor(images=pil_tiles, return_tensors="pt", padding=False)
    pv = inputs["pixel_values"].to(DEVICE, dtype=TORCH_DTYPE)

    fmap          = model.image_embedder(pv)[0]         # (B, Hf, Wf, C)
    B, Hf, Wf, C = fmap.shape
    query_feats   = fmap.view(B, Hf * Wf, C)            # (B, N, C)
    obj_scores    = (
        model.objectness_predictor(query_feats)
        .sigmoid()
        .squeeze(-1)                                     # (B, N)
    )
    raw_boxes = model.box_predictor(
        query_feats, fmap, interpolate_pos_encoding=False
    )                                                    # (B, N, 4) cxcywh normalised

    boxes_list  = []
    scores_list = []

    for i in range(B):
        orig_h, orig_w = shapes[i]
        padded = max(orig_h, orig_w)

        mask = obj_scores[i] > objectness_threshold
        if not mask.any():
            boxes_list.append(torch.zeros((0, 4)))
            scores_list.append(torch.zeros((0,)))
            continue

        k_scores = obj_scores[i][mask]      # (M,)
        k_boxes  = raw_boxes[i][mask]       # (M, 4) cxcywh norm

        # cxcywh normalised → xyxy pixels
        cx, cy, bw, bh = k_boxes.unbind(-1)
        cx = cx * padded; cy = cy * padded
        bw = bw * padded; bh = bh * padded
        x1 = (cx - bw / 2).clamp(0, orig_w)
        y1 = (cy - bh / 2).clamp(0, orig_h)
        x2 = (cx + bw / 2).clamp(0, orig_w)
        y2 = (cy + bh / 2).clamp(0, orig_h)
        boxes_px = torch.stack([x1, y1, x2, y2], dim=1)  # (M, 4) still on GPU

        # Per-tile NMS
        keep = torchvision_nms(boxes_px, k_scores, nms_iou_threshold)
        boxes_list.append(boxes_px[keep].cpu().float())
        scores_list.append(k_scores[keep].cpu().float())

    return boxes_list, scores_list, shapes


# ──────────────────────────────────────────────────────────────────────────────
# Global NMS after tile stitching
# ──────────────────────────────────────────────────────────────────────────────

def _global_nms(boxes, scores, feats, iou_threshold=0.5):
    if len(boxes) == 0:
        return None, None, None
    b = torch.from_numpy(boxes).float()
    s = torch.from_numpy(scores).float()
    f = torch.from_numpy(feats).float()
    keep = torchvision_nms(b, s, iou_threshold)
    return f[keep].numpy(), b[keep].numpy(), s[keep].numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_proposals_tiled(
    image_paths,
    text_prompt="visual",           # unused — kept for API compat
    confidence_threshold=0.1,
    is_sahi=False,
    tile_size=960,
    overlap_ratio=0.25,
    batch_size=8,
    nms_iou_threshold=0.2,
    embedding_backend="owlv2",
    model_id=DEFAULT_MODEL_ID,
):
    """
    Generate bounding-box proposals for a list of images using OWLv2 objectness.

    All embeddings are produced via the unified embedder from embedding_utils.py,
    ensuring they match the class-support vector space used in classification.

    Returns
    -------
    dict: image_path (str) -> {"features": ndarray(K,D),
                                "boxes":    ndarray(K,4),
                                "scores":   ndarray(K,)}
          or None for images with no detections.
    """
    image_paths = [str(p) for p in image_paths]
    processor, model = _load_owlv2(model_id)

    # Load embedding model once
    embedder = get_embedder(embedding_backend)
    logger.info(f"Embedding backend: {embedding_backend}")

    # Build dataset
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
                              "coords": torch.tensor([0, 0, img.width, img.height])},
            })
        loader = [dataset[i: i + batch_size] for i in range(0, len(dataset), batch_size)]
        logger.info(f"Whole-image mode | images={len(dataset)}, batch_size={batch_size}")

    # Accumulator: img_idx → {boxes, scores, feats}
    accum = collections.defaultdict(lambda: {"boxes": [], "scores": [], "feats": []})

    for batch_idx, batch in enumerate(loader):
        tile_images = [item["image"]                     for item in batch]
        coords_list = [item["metadata"]["coords"]        for item in batch]
        img_idxs    = [int(item["metadata"]["img_idx"])  for item in batch]

        boxes_list, scores_list, _ = _infer_batch(
            processor, model, tile_images, confidence_threshold, nms_iou_threshold
        )

        for tile_img, tile_boxes, tile_scores, coords, img_idx in zip(
            tile_images, boxes_list, scores_list, coords_list, img_idxs
        ):
            if tile_boxes.numel() == 0:
                continue

            # Shift to global image coords
            ox, oy = float(coords[0]), float(coords[1])
            offset      = torch.tensor([ox, oy, ox, oy])
            global_boxes = (tile_boxes + offset).numpy()   # (M, 4)

            # Embed using the unified embedder (same space as class supports)
            local_boxes = tile_boxes.tolist()
            feats_np    = embedder.embed_boxes(tile_img, local_boxes).numpy()  # (M, D)

            accum[img_idx]["boxes"].append(global_boxes)
            accum[img_idx]["scores"].append(tile_scores.numpy())
            accum[img_idx]["feats"].append(feats_np)

        logger.debug(f"Batch {batch_idx + 1}/{len(loader)} done")

    # Stitch + global NMS
    results = {p: None for p in image_paths}
    for img_idx, a in accum.items():
        boxes_raw  = np.vstack(a["boxes"])
        scores_raw = np.hstack(a["scores"])
        feats_raw  = np.vstack(a["feats"])

        feats_f, boxes_f, scores_f = _global_nms(
            boxes_raw, scores_raw, feats_raw, nms_iou_threshold
        )
        if boxes_f is not None:
            results[image_paths[img_idx]] = {
                "features": feats_f,
                "boxes":    boxes_f,
                "scores":   scores_f,
            }
            logger.info(f"{Path(image_paths[img_idx]).name}: "
                        f"{len(boxes_raw)} raw → {len(boxes_f)} after NMS")

    return results
