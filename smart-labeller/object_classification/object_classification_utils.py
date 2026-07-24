"""
object_classification/object_classification_utils.py

Few-shot classification of SAM3 mask proposals against class-support embeddings.

Detection path (DINOv3 cosine)
------------------------------
Pure embedding cosine similarity between each proposal's DINOv3 vector and the
per-class support vectors.  Supports and proposals share the same masked-crop
DINOv3 space (see proposal/embedding_utils.py and class_support/), so cosine is
valid.

Everything is mask-based: proposals carry per-instance masks (COCO RLE), the
final de-duplication is mask-IoU NMS, and each emitted detection carries a
``segmentation`` RLE — no bounding boxes anywhere.

Dimension safety
-----------------
Support tensors are validated to be (N, D).  A transposed (D, N) save (D > N) is
auto-corrected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from proposal.mask_utils import mask_nms
except ImportError:  # flat container layout (mask_utils.py alongside this file)
    from mask_utils import mask_nms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Core: cosine-similarity detection (DINOv3)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity_detection(
    class_supports: dict,
    object_features: dict,
    device: str = DEVICE,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
) -> list:
    """
    Classify mask proposals against class supports via cosine similarity.

    Args
    ----
    class_supports   : {class_name -> Tensor (N_s, D)}  L2-normed support embeddings
    object_features  : {"features": Tensor (N_p, D),
                        "masks":    list[RLE] length N_p,
                        "scores":   Tensor (N_p,)}
    objectness_threshold : minimum SAM3 score to keep a proposal
    similarity_threshold : minimum cosine sim to emit a detection
    nms_iou_threshold    : mask-IoU threshold for final NMS

    Returns
    -------
    list of {"segmentation"(RLE), "score", "class"} dicts after mask-NMS
    """
    feats      = object_features.get("features")
    masks      = object_features.get("masks")
    obj_scores = object_features.get("scores")

    if feats is None or feats.numel() == 0:
        return []

    feats      = feats.to(device).float()
    obj_scores = obj_scores.to(device).float()

    # Objectness pre-filter, then keep top-100 by objectness.
    obj_mask = obj_scores > objectness_threshold
    if obj_mask.sum() == 0:
        return []

    keep_idx = torch.nonzero(obj_mask, as_tuple=True)[0]
    if keep_idx.numel() > 100:
        _, top = obj_scores[keep_idx].topk(100)
        keep_idx = keep_idx[top]

    keep_list  = keep_idx.tolist()
    feats      = feats[keep_idx]
    masks      = [masks[i] for i in keep_list]
    obj_scores = obj_scores[keep_idx]

    # L2-normalise proposals once for all classes.
    feats_norm = F.normalize(feats, p=2, dim=-1)   # (M, D)
    feat_dim   = feats_norm.shape[-1]

    detections = []

    for class_name, support_embs in class_supports.items():
        support_embs = support_embs.to(device).float()
        if support_embs.dim() == 1:
            support_embs = support_embs.unsqueeze(0)

        # Auto-fix transposed (D, N_s) saves.
        if support_embs.shape[-1] != feat_dim:
            if support_embs.shape[0] == feat_dim:
                print(f"  [WARN] '{class_name}' support transposed ({tuple(support_embs.shape)}) — fixing.")
                support_embs = support_embs.T
            else:
                print(
                    f"  [ERROR] '{class_name}' support dim mismatch: "
                    f"proposal={feat_dim}, support last_dim={support_embs.shape[-1]}. Skipping."
                )
                continue

        support_norm = F.normalize(support_embs, p=2, dim=-1)  # (N_s, D)

        sim_matrix = feats_norm @ support_norm.T          # (M, N_s)
        sim_scores, _ = sim_matrix.max(dim=1)              # (M,)

        keep_mask = sim_scores > similarity_threshold
        for i in torch.nonzero(keep_mask, as_tuple=True)[0].tolist():
            detections.append({
                "segmentation": masks[i],
                "score":        float(sim_scores[i]),
                "class":        class_name,
            })

    if not detections:
        return []

    # Final mask-NMS across all classes.
    keep = mask_nms(
        [d["segmentation"] for d in detections],
        [d["score"] for d in detections],
        iou_threshold=nms_iou_threshold,
    )
    return [detections[i] for i in keep]


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def run_detection_for_backend(
    backend: str,
    class_supports: dict,
    object_features: dict,
    device: str = DEVICE,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
) -> list:
    """Dispatch to cosine_similarity_detection (DINOv3 only)."""
    if backend != "dinov3":
        raise ValueError(f"Unknown backend '{backend}'. Only 'dinov3' is supported.")

    return cosine_similarity_detection(
        class_supports=class_supports,
        object_features=object_features,
        device=device,
        objectness_threshold=objectness_threshold,
        similarity_threshold=similarity_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
