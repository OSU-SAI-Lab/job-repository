"""
proposal/sam3_proposal.py  –  SAM3-based proposal generation.

Memory / speed optimisations
-----------------------------
- SAM3 model loaded once at module level.
- GPU tensors stay on GPU until embedder.embed_boxes() finalises them.
- Single .cpu().numpy() per tile after per-tile processing.
- Embedder loaded once per generate_proposals_tiled call.

Embedding space
---------------
SAM3 provides boxes + scores only (no native feature embeddings).
All embeddings are produced by the unified embedder from embedding_utils.py:
  dinov3   → facebook/dinov2-large pooler_output  (dim=1024)
  bioclip  → BioCLIP ViT-B/16                     (dim=512)
  owlv2    → OWLv2 vision_model pooler_output     (dim=1024)

These match the class-support embeddings from class_supports_utils.py exactly.
"""

from __future__ import annotations

import logging
import sys
import collections
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from torch.utils.data import DataLoader
from transformers import Sam3Model, Sam3Processor
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OverlappingTileDataset import OverlappingTileDataset
from proposal.embedding_utils import get_embedder, DEVICE

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
):
    """Run SAM3 on a batch; return (boxes_list, scores_list) as CPU tensors."""
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
        mask_threshold=confidence_threshold,
        target_sizes=target_sizes,
    )

    boxes_list  = []
    scores_list = []
    for result in batch_results:
        if len(result.get("masks", [])) == 0:
            boxes_list.append(torch.zeros((0, 4)))
            scores_list.append(torch.zeros((0,)))
        else:
            boxes_list.append(result["boxes"].float().cpu())
            scores_list.append(result["scores"].float().cpu())

    return boxes_list, scores_list


# ──────────────────────────────────────────────────────────────────────────────
# Global NMS
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
    text_prompt="visual",
    confidence_threshold=0.5,
    is_sahi=False,
    tile_size=960,
    overlap_ratio=0.2,
    batch_size=8,
    nms_iou_threshold=0.5,
    embedding_backend="dinov3",
    model_id=None,              # unused for SAM3; kept for API compat
):
    """
    Generate bounding-box proposals using SAM3.

    Returns
    -------
    dict: image_path (str) -> {"features": ndarray(K,D),
                                "boxes":    ndarray(K,4),
                                "scores":   ndarray(K,)}
          or None for images with no detections.
    """
    image_paths = [str(p) for p in image_paths]

    # Load embedding model once
    embedder_kwargs = {}
    if model_id:
        embedder_kwargs["model_id"] = model_id
    embedder = get_embedder(embedding_backend, **embedder_kwargs)
    logger.info(f"Embedding backend: {embedding_backend}")

    # Build dataset / loader
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

    accum = collections.defaultdict(lambda: {"boxes": [], "scores": [], "feats": []})

    for batch_idx, batch in enumerate(loader):
        tile_images = [item["image"]                     for item in batch]
        coords_list = [item["metadata"]["coords"]        for item in batch]
        img_idxs    = [int(item["metadata"]["img_idx"])  for item in batch]

        boxes_list, scores_list = _infer_batch(tile_images, text_prompt, confidence_threshold)

        for tile_img, tile_boxes, tile_scores, coords, img_idx in zip(
            tile_images, boxes_list, scores_list, coords_list, img_idxs
        ):
            if tile_boxes.numel() == 0:
                continue

            ox, oy      = float(coords[0]), float(coords[1])
            offset      = torch.tensor([ox, oy, ox, oy])
            global_boxes = (tile_boxes + offset).numpy()

            local_boxes = tile_boxes.tolist()
            feats_np    = embedder.embed_boxes(tile_img, local_boxes).numpy()

            accum[img_idx]["boxes"].append(global_boxes)
            accum[img_idx]["scores"].append(tile_scores.numpy())
            accum[img_idx]["feats"].append(feats_np)

        logger.debug(f"Batch {batch_idx + 1}/{len(loader)} done")

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
