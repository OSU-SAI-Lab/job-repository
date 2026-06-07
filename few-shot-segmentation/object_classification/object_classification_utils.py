"""
object_classification/object_classification_utils.py

Classification of proposals against class-support embeddings.

Two detection paths
-------------------
cosine_similarity_detection (bioclip, dinov3, owlv2)
    Pure embedding cosine similarity.  Works for any backend whose supports
    and proposals live in the same vector space.

    OWLv2 now also goes through this path using vision_model.pooler_output
    embeddings (dim=1024).  The old image_guided_object_detection path that
    called model.class_predictor is removed — it required proposals to be
    OWLv2 patch tokens (not pooler vectors) and caused the (52,768)x(1024,S)
    dimension mismatch.

Dimension safety
-----------------
Support tensors are validated to be (N, D) on load.  If the npz was saved
transposed as (D, N) (D > N), they are corrected automatically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import nms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Core: cosine-similarity detection (all backends)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity_detection(
    class_supports: dict,
    object_features: dict,
    device: str = DEVICE,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
    max_proposals: int = 100,
) -> list:
    """
    Classify proposals against class supports via cosine similarity.

    Args
    ----
    class_supports   : {class_name -> Tensor (N_s, D)}  L2-normed support embeddings
    object_features  : {"features": Tensor (N_p, D),
                        "boxes":    Tensor (N_p, 4),
                        "scores":   Tensor (N_p,)}
    device           : torch device string
    objectness_threshold : minimum objectness score to keep a proposal
    similarity_threshold : minimum cosine sim to emit a detection
    nms_iou_threshold    : IoU threshold for final NMS

    Returns
    -------
    list of {"bounding_box", "score", "class"} dicts after NMS
    """
    feats      = object_features.get("features")
    boxes      = object_features.get("boxes")
    obj_scores = object_features.get("scores")
    masks      = object_features.get("masks")  # list of (H,W) bool numpy arrays, or None

    if feats is None or feats.numel() == 0:
        return []

    # Move to device once
    feats      = feats.to(device).float()
    boxes      = boxes.to(device).float()
    obj_scores = obj_scores.to(device).float()

    # Objectness pre-filter: keep top-`max_proposals`; apply same filter to masks list
    obj_mask = obj_scores > objectness_threshold
    if obj_mask.sum() == 0:
        return []
    if obj_mask.sum() > max_proposals:
        _, top_idxs = obj_scores.topk(max_proposals)
        top_idx_list = top_idxs.tolist()
        feats      = feats[top_idxs]
        boxes      = boxes[top_idxs]
        obj_scores = obj_scores[top_idxs]
        masks_filtered = [masks[i] for i in top_idx_list] if masks is not None else None
    else:
        obj_mask_list = obj_mask.cpu().tolist()
        feats      = feats[obj_mask]
        boxes      = boxes[obj_mask]
        obj_scores = obj_scores[obj_mask]
        masks_filtered = [m for m, keep in zip(masks, obj_mask_list) if keep] if masks is not None else None

    # L2-normalise proposals once for all classes
    feats_norm = F.normalize(feats, p=2, dim=-1)   # (M, D)
    feat_dim   = feats_norm.shape[-1]

    detections = []

    for class_name, support_embs in class_supports.items():
        # Ensure (N_s, D)
        support_embs = support_embs.to(device).float()
        if support_embs.dim() == 1:
            support_embs = support_embs.unsqueeze(0)

        # Auto-fix transposed (D, N_s) saves
        if support_embs.shape[-1] != feat_dim:
            if support_embs.shape[0] == feat_dim:
                print(f"  [WARN] '{class_name}' support transposed ({support_embs.shape}) — fixing.")
                support_embs = support_embs.T
            else:
                print(
                    f"  [ERROR] '{class_name}' support dim mismatch: "
                    f"proposal={feat_dim}, support last_dim={support_embs.shape[-1]}. Skipping."
                )
                continue

        support_norm = F.normalize(support_embs, p=2, dim=-1)  # (N_s, D)

        # (M, D) @ (D, N_s) → (M, N_s) → max per proposal
        sim_matrix = feats_norm @ support_norm.T          # (M, N_s)
        sim_scores, _ = sim_matrix.max(dim=1)              # (M,)

        keep_mask = sim_scores > similarity_threshold
        for i in torch.nonzero(keep_mask, as_tuple=True)[0].tolist():
            detections.append({
                "bounding_box": boxes[i].tolist(),
                "score":        float(sim_scores[i]),
                "class":        class_name,
                "_mask_idx":    i,   # internal index into masks_filtered; removed before returning
            })

    if not detections:
        return []

    # Final NMS across all classes
    boxes_t  = torch.tensor([d["bounding_box"] for d in detections],
                             dtype=torch.float32, device=device)
    scores_t = torch.tensor([d["score"] for d in detections],
                             dtype=torch.float32, device=device)
    keep_idxs = nms(boxes_t, scores_t, iou_threshold=nms_iou_threshold)

    result = []
    for i in keep_idxs.tolist():
        d = dict(detections[i])
        mask_idx = d.pop("_mask_idx")
        if masks_filtered is not None and mask_idx < len(masks_filtered) and masks_filtered[mask_idx] is not None:
            d["segmentation"] = masks_filtered[mask_idx]   # (H,W) bool numpy array; encoded to RLE at save time
        result.append(d)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def run_detection_for_backend(
    backend: str,
    class_supports: dict,
    object_features: dict,
    device: str = DEVICE,
    # legacy kwargs kept for API compat — no longer used
    owlv2_model_bundle=None,
    torch_data_type=None,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
    max_proposals: int = 100,
) -> list:
    """
    Dispatch to cosine_similarity_detection for all backends.

    All three backends (owlv2, bioclip, dinov3) now use pure cosine similarity
    because all their embedders produce vectors in the same vision-backbone space.
    The old image_guided_object_detection (class_predictor path) is removed.
    """
    if backend not in ("owlv2", "bioclip", "dinov3"):
        raise ValueError(f"Unknown backend '{backend}'. Choose from: owlv2, bioclip, dinov3")

    return cosine_similarity_detection(
        class_supports=class_supports,
        object_features=object_features,
        device=device,
        objectness_threshold=objectness_threshold,
        similarity_threshold=similarity_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_proposals=max_proposals,
    )
