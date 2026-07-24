"""
class_support/class_supports_utils.py

Extracts class-support embeddings from ground-truth annotations.

Segmentation pipeline (SAM3 + DINOv3)
------------------------------------
Ground-truth annotations carry a ``bounding_box`` only.  Since the whole pipeline
is mask-based, each GT box is first turned into a MASK by prompting the SAM3
Tracker with that box; the box is used purely as a SAM3 input prompt, never as a
pipeline box.  The resulting mask is then embedded the SAME way proposals are:

  box → SAM3 Tracker (box prompt) → mask
      → mask_extent_crop (zero background, crop to the mask's tight extent)
      → DINOv3 pooler_output (dim=1024) → L2-normalise

This shares the exact embedding path with proposal/embedding_utils.py
(DINOv3Embedder.embed_masks), so class-support vectors and proposal vectors live
in the same space and cosine similarity at classification time is valid.

If an annotation already carries a ``segmentation`` (COCO RLE), that mask is used
directly and SAM3 is skipped for it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from proposal.mask_utils import mask_extent_crop, decode_rle, encode_rle
except ImportError:  # flat container layout (mask_utils.py alongside this file)
    from mask_utils import mask_extent_crop, decode_rle, encode_rle


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_SAM3_MODEL   = "facebook/sam3"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_dinov3_model(model_name: str = DEFAULT_DINOV3_MODEL):
    """Load DINOv3 AutoModel + processor."""
    from transformers import AutoImageProcessor, AutoModel  # type: ignore
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    print(f"[DINOv3] Loaded {model_name} | dim={model.config.hidden_size}")
    return processor, model


def load_sam3_tracker(model_name: str = DEFAULT_SAM3_MODEL):
    """Load the SAM3 Tracker model + processor (for box-prompted masks)."""
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # type: ignore
    processor = Sam3TrackerProcessor.from_pretrained(model_name)
    model = Sam3TrackerModel.from_pretrained(model_name).to(DEVICE).eval()
    print(f"[SAM3 Tracker] Loaded {model_name}")
    return processor, model


def load_embedding_model(backend: str = "dinov3", device=None, torch_dtype=None,
                         model_name: str = None):
    """Factory kept for API compatibility. Only 'dinov3' is supported."""
    if backend != "dinov3":
        raise ValueError(f"Unknown backend '{backend}'. Only 'dinov3' is supported.")
    return load_dinov3_model(model_name or DEFAULT_DINOV3_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
# SAM3 box-prompt → mask
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def segment_box(
    img: Image.Image,
    bbox: List[float],
    tracker_bundle,
) -> Optional[np.ndarray]:
    """Prompt the SAM3 Tracker with ``bbox`` and return the best (H, W) bool mask.

    Returns None if SAM3 produces no foreground pixels.
    """
    processor, model = tracker_bundle
    W, H = img.size

    image_inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    image_embeddings = model.get_image_embeddings(image_inputs["pixel_values"])

    x1, y1, x2, y2 = [float(v) for v in bbox]
    prompt_inputs = processor(
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
        original_sizes=[(H, W)],
    ).to(DEVICE)

    outputs = model(**prompt_inputs, image_embeddings=image_embeddings)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        prompt_inputs["original_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.cpu()[0, 0]
    best_idx = int(torch.argmax(scores).item())
    best_mask = masks[0][best_idx].numpy() > 0

    if best_mask.sum() == 0:
        return None
    return best_mask.astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# DINOv3 embedding of a masked region
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _embed_dinov3_masked(
    img: Image.Image,
    mask_bool: np.ndarray,
    processor,
    model,
) -> torch.Tensor:
    """Embed the masked object region with DINOv3 (pooler_output), L2-normalised.

    Uses mask_extent_crop so the framing matches DINOv3Embedder.embed_masks.
    """
    crop = mask_extent_crop(img, mask_bool)
    pv = processor(images=[crop], return_tensors="pt")["pixel_values"].to(DEVICE)
    out = model(pixel_values=pv)
    emb = out.pooler_output[0].float()          # (D,) on GPU
    return F.normalize(emb, dim=-1).cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_support_embeddings(
    support_examples: List[dict],
    embedding_backend: str,           # kept for API compat; must be 'dinov3'
    model_bundle,                     # (dinov3_processor, dinov3_model)
    device: str,                      # kept for API compat; module-level DEVICE is used
    src_path: str = None,
    tracker_bundle=None,              # (sam3_tracker_processor, sam3_tracker_model)
) -> Tuple[Dict[str, torch.Tensor], list]:
    """
    Extract class-support embeddings from GT annotations.

    For each annotation: obtain a mask (from an existing ``segmentation`` RLE, else
    by prompting SAM3 with ``bounding_box``), then DINOv3-embed the masked region.

    Returns
    -------
    class_supports  : dict[class_name -> Tensor (N, D)]  stacked, L2-normed supports
    support_masks   : list of {"image_path","class","segmentation"(RLE)} for traceability
    """
    if embedding_backend != "dinov3":
        raise ValueError(f"Unknown backend '{embedding_backend}'. Only 'dinov3' is supported.")

    processor, model = model_bundle
    class_supports: Dict[str, list] = {}
    support_masks: list = []

    print(f"  [DEBUG] extract_support_embeddings: n_examples={len(support_examples)}, "
          f"src_path={src_path}")

    if not support_examples:
        print("  [ERROR] support_examples is empty — check annotation JSON.")
        return {}, []

    for idx, support in enumerate(support_examples):
        img_path   = os.path.join(src_path or "", support["image_path"])
        class_name = support["class"]

        if not os.path.exists(img_path):
            print(f"  [ERROR] Image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        # Obtain the object mask.
        if support.get("segmentation") is not None:
            mask = decode_rle(support["segmentation"])
        else:
            if tracker_bundle is None:
                print(f"  [ERROR] [{idx}] no segmentation and no SAM3 tracker provided — skipping.")
                continue
            bbox = support.get("bounding_box")
            if bbox is None:
                print(f"  [WARN] [{idx}] annotation has neither segmentation nor bounding_box — skipping.")
                continue
            mask = segment_box(img, bbox, tracker_bundle)
            if mask is None:
                print(f"  [WARN] [{idx}] SAM3 produced empty mask for bbox={bbox} — skipping.")
                continue

        emb = _embed_dinov3_masked(img, mask, processor, model)   # (D,)

        class_supports.setdefault(class_name, []).append(emb)
        support_masks.append({
            "image_path":   support["image_path"],
            "class":        class_name,
            "segmentation": encode_rle(mask),
        })
        print(f"  [DEBUG] [{idx}] class={class_name} | mask_px={int(mask.sum())}")

    if not class_supports:
        print("  [ERROR] class_supports is empty — check src_path, image_path, masks.")
        return {}, support_masks

    stacked: Dict[str, torch.Tensor] = {
        c: torch.stack(embs, dim=0) for c, embs in class_supports.items()
    }
    print("  [DEBUG] Final stacked supports:")
    for c, t in stacked.items():
        print(f"  [DEBUG]   '{c}': {tuple(t.shape)}")

    return stacked, support_masks
