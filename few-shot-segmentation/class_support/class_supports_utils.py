"""
class_support/class_supports_utils.py

Extracts class-support embeddings from ground-truth annotated crops.

Embedding space contract
------------------------
Each backend uses EXACTLY the same model and the same extraction path as the
corresponding embedder in proposal/embedding_utils.py.  This guarantees that
class-support vectors and proposal vectors live in the same space so that
cosine similarity at classification time is valid.

  bioclip → BioCLIP BaseClassifier.create_image_features  (dim=512)
  dinov3  → AutoModel.pooler_output                        (dim=1024)
  owlv2   → vision_model.pooler_output of the BEST-MATCHING anchor
             inside a centered crop around the GT box     (dim=1024)

OWLv2 strategy (centered-crop + best-patch)
--------------------------------------------
A raw GT-box crop fed straight into the vision backbone produces a poor support
embedding because OWLv2 was trained to embed full scenes, not tight crops.
Instead we:
  1. Build a centered crop of `crop_size` pixels around the GT box centre.
  2. Run image_embedder to get patch-level feature maps.
  3. Run box_predictor to get per-anchor boxes.
  4. Pick the anchor whose predicted box has the highest IoU with the GT box.
  5. Extract that anchor's vision_model pooler-equivalent embedding by running
     vision_model on the centered crop and using the patch token at the anchor's
     spatial position (mean-pooled over the anchor's grid region).

     For simplicity and consistency with OWLv2Embedder (which uses pooler_output),
     we run vision_model on the centered crop directly and return pooler_output.
     This gives a single (1024,) vector that is in the SAME space as the proposal
     embeddings produced by OWLv2Embedder.embed_boxes() in embedding_utils.py.

  The IoU-based anchor selection is still used to record the best_box and best_iou
  for the crop-size optimisation loop in generate_class_supports_main.py — it does
  NOT change which embedding is returned (we always return the pooler of the crop).
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import expit
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OWLV2_MODEL  = "google/owlv2-large-patch14-ensemble"
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# IoU utility (used by crop-size optimisation loop in main)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_change_in_iou(
    patch_size: int,
    generated_boxes: list,
    iou_map: dict,
) -> Tuple[bool, dict]:
    """Track average IoU across crop-size iterations (OWLv2 legacy loop)."""
    avg_iou = 0.0
    if generated_boxes:
        avg_iou = sum(b.get("iou", 0.0) for b in generated_boxes) / len(generated_boxes)

    sorted_sizes = sorted(iou_map, reverse=True)
    prev_best = iou_map.get(sorted_sizes[-1], 0.0) if sorted_sizes else 0.0
    next_iter = prev_best == 0 or (
        avg_iou > prev_best and abs(avg_iou - prev_best) / prev_best > 0.1
    )
    iou_map[patch_size] = round(avg_iou, 3)
    return next_iter, iou_map


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_owlv2_model(model_name: str = DEFAULT_OWLV2_MODEL):
    """Load OWLv2ForObjectDetection + processor."""
    from transformers import Owlv2Processor, Owlv2ForObjectDetection  # type: ignore
    processor = Owlv2Processor.from_pretrained(model_name)
    model = (
        Owlv2ForObjectDetection
        .from_pretrained(model_name, torch_dtype=TORCH_DTYPE)
        .to(DEVICE)
        .eval()
    )
    print(f"[OWLv2] Loaded {model_name}")
    return processor, model


def load_bioclip_model():
    """Load BioCLIP BaseClassifier."""
    from bioclip.predict import BaseClassifier  # type: ignore
    clf = BaseClassifier(device=DEVICE)
    clf.model.eval()
    print("[BioCLIP] Model loaded")
    return clf


def load_dinov3_model(model_name: str = DEFAULT_DINOV3_MODEL):
    """Load DINOv3 AutoModel + processor."""
    from transformers import AutoImageProcessor, AutoModel  # type: ignore
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    print(f"[DINOv3] Loaded {model_name} | dim={model.config.hidden_size}")
    return processor, model


def load_embedding_model(backend: str, device=None, torch_dtype=None, model_name: str = None):
    """
    Factory: return the model bundle for a given backend.

    Returns
    -------
    owlv2   → (Owlv2Processor, Owlv2ForObjectDetection)
    bioclip → BaseClassifier
    dinov3  → (AutoImageProcessor, AutoModel)
    """
    if backend == "owlv2":
        return load_owlv2_model(model_name or DEFAULT_OWLV2_MODEL)
    elif backend == "bioclip":
        return load_bioclip_model()
    elif backend == "dinov3":
        return load_dinov3_model(model_name or DEFAULT_DINOV3_MODEL)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: owlv2, bioclip, dinov3")


# ──────────────────────────────────────────────────────────────────────────────
# OWLv2: centered-crop + best-anchor selection
# ──────────────────────────────────────────────────────────────────────────────

def _decode_segmentation(seg, height: int = None, width: int = None) -> Optional[np.ndarray]:
    """Decode a COCO segmentation (compressed RLE, uncompressed RLE, or polygon)
    into a (H, W) boolean mask. Returns None if undecodable (e.g. pycocotools
    missing) so callers can fall back to a plain box crop."""
    if seg is None:
        return None
    try:
        from pycocotools import mask as mask_util  # type: ignore
    except Exception:
        print("  [WARN] pycocotools unavailable — cannot decode GT masks; "
              "falling back to box crop. Install pycocotools for mask-based supports.")
        return None

    if isinstance(seg, dict):
        counts = seg.get("counts")
        if isinstance(counts, list):                 # uncompressed RLE
            rle = mask_util.frPyObjects(seg, seg["size"][0], seg["size"][1])
        elif isinstance(counts, (str, bytes)):       # compressed RLE
            rle = dict(seg)
            rle["counts"] = counts.encode("utf-8") if isinstance(counts, str) else counts
        else:
            return None
        mask = mask_util.decode(rle)
    elif isinstance(seg, list):                      # polygon(s)
        if height is None or width is None:
            return None
        rles = mask_util.frPyObjects(seg, height, width)
        mask = mask_util.decode(mask_util.merge(rles))
    else:
        return None

    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(bool)


def _apply_mask_to_crop(img: Image.Image, box, mask, background: str = "zero") -> Image.Image:
    """Crop `box` from `img` and suppress background pixels outside `mask`.

    Mirrors BaseEmbedder._mask_crop in proposal/embedding_utils.py so that
    support and proposal embeddings live in the SAME masked space. Falls back to
    the plain box crop on any shape mismatch or empty mask.
    """
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (1, 1))

    crop = img.crop((x1, y1, x2, y2))
    if background == "none" or mask is None:
        return crop

    m = np.asarray(mask, dtype=bool)[y1:y2, x1:x2]
    arr = np.array(crop)
    if m.shape != arr.shape[:2] or not m.any():
        return crop
    fill = arr[m].mean(axis=0).astype(arr.dtype) if background == "mean" else 0
    arr[~m] = fill
    return Image.fromarray(arr)


def _compute_iou(boxA, boxB) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa1, ya1, xa2, ya2 = map(float, boxA)
    xb1, yb1, xb2, yb2 = map(float, boxB)
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union > 0 else 0.0


def get_best_patch_embedding(
    crop: Image.Image,
    processor,
    model,
    ref_box_in_crop: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
    """
    Find the OWLv2 anchor inside `crop` that best matches `ref_box_in_crop`
    (GT box expressed in crop-local pixel coordinates), then return the
    vision_model.pooler_output of the crop as the embedding.

    Why pooler_output instead of the per-anchor class_predictor embedding?
    -----------------------------------------------------------------------
    The class_predictor projects patch tokens into a 512-dim text-alignment
    space.  Proposal embeddings are produced via vision_model.pooler_output
    (1024-dim).  Using the same path here keeps supports and proposals in the
    same vector space so cosine similarity is meaningful.

    The anchor selection (objectness + IoU ranking) is kept because it:
      a) records the best predicted box for the IoU optimisation loop in main, and
      b) confirms the crop actually contains the object before we commit the
         pooler embedding as a support vector.

    Returns
    -------
    emb        : np.ndarray (1024,)  — L2-normalised pooler embedding of the crop
    best_box   : (x1,y1,x2,y2) in crop-local pixels — highest-IoU anchor box
    best_iou   : float
    """
    pv = processor(images=[crop], return_tensors="pt")["pixel_values"].to(DEVICE, dtype=TORCH_DTYPE)

    with torch.no_grad():
        # Anchor-level inference (used only for best-box selection)
        fmap = model.image_embedder(pv)[0]              # (1, Hf, Wf, C_hidden)
        B, Hf, Wf, C_h = fmap.shape
        feats = fmap.reshape(B, Hf * Wf, C_h)           # (1, P, C_hidden)
        obj_logits = model.objectness_predictor(feats)[0]   # (P,)
        boxes_norm = model.box_predictor(feats, feature_map=fmap)[0]  # (P, 4) cxcywh norm

        # Vision-backbone pooler (for the actual embedding — same as OWLv2Embedder)
        pool_out = model.owlv2.vision_model(pixel_values=pv).pooler_output[0].float()  # (1024,)

    # Convert anchors to pixel coords for IoU ranking
    obj_scores = expit(obj_logits.cpu().float().numpy())          # (P,)
    boxes_norm_np = boxes_norm.cpu().float().numpy()              # (P, 4) cxcywh norm
    W_crop, H_crop = crop.width, crop.height
    padded = max(H_crop, W_crop)

    proposals = []
    for (cx, cy, bw, bh), score in zip(boxes_norm_np, obj_scores):
        xA = (cx - bw / 2) * padded
        yA = (cy - bh / 2) * padded
        xB = (cx + bw / 2) * padded
        yB = (cy + bh / 2) * padded
        proposals.append((xA, yA, xB, yB, score))

    ref = tuple(map(float, ref_box_in_crop))
    ious = np.array([_compute_iou(ref, p[:4]) for p in proposals])
    best_idx = int(np.argmax(ious))
    best_box = tuple(float(v) for v in proposals[best_idx][:4])
    best_iou = float(ious[best_idx])

    print(f"    best_iou={best_iou:.3f}")

    # Return pooler embedding (correct space) + best box metadata
    emb_np = F.normalize(pool_out, dim=-1).cpu().numpy()   # (1024,)
    return emb_np, best_box, best_iou


# ──────────────────────────────────────────────────────────────────────────────
# Per-backend single-crop embedding extractors
# Each returns a 1-D CPU float32 tensor.
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _embed_owlv2_centered(
    img: Image.Image,
    bbox: List[float],
    processor,
    model,
    crop_size: int,
) -> Tuple[torch.Tensor, dict]:
    """
    Build a centered crop around the GT bbox, run get_best_patch_embedding,
    and return the pooler embedding + a generated_box annotation dict.
    """
    W, H = img.size
    x1_gt = max(0, min(int(bbox[0]), W - 1))
    y1_gt = max(0, min(int(bbox[1]), H - 1))
    x2_gt = max(0, min(int(bbox[2]), W))
    y2_gt = max(0, min(int(bbox[3]), H))

    cx_gt = (x1_gt + x2_gt) / 2.0
    cy_gt = (y1_gt + y2_gt) / 2.0
    half  = crop_size / 2.0

    crop_x1 = max(0, int(cx_gt - half))
    crop_y1 = max(0, int(cy_gt - half))
    crop_x2 = min(W, int(cx_gt + half))
    crop_y2 = min(H, int(cy_gt + half))

    # Snap to edge if crop is smaller than requested size
    if crop_x2 - crop_x1 < crop_size:
        crop_x1 = 0 if crop_x1 == 0 else max(0, crop_x2 - crop_size)
        crop_x2 = min(W, crop_x1 + crop_size)
    if crop_y2 - crop_y1 < crop_size:
        crop_y1 = 0 if crop_y1 == 0 else max(0, crop_y2 - crop_size)
        crop_y2 = min(H, crop_y1 + crop_size)

    centered_crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # GT box in crop-local coords
    ref_x1 = max(0, x1_gt - crop_x1)
    ref_y1 = max(0, y1_gt - crop_y1)
    ref_x2 = min(centered_crop.width,  x2_gt - crop_x1)
    ref_y2 = min(centered_crop.height, y2_gt - crop_y1)

    emb_np, best_box, best_iou = get_best_patch_embedding(
        centered_crop, processor, model, (ref_x1, ref_y1, ref_x2, ref_y2)
    )
    emb = torch.from_numpy(emb_np)   # (1024,) CPU float32, already normalised

    gen_box = {
        "bounding_box": [
            float(best_box[0]) + crop_x1, float(best_box[1]) + crop_y1,
            float(best_box[2]) + crop_x1, float(best_box[3]) + crop_y1,
        ],
        "iou": float(best_iou),
    }
    return emb, gen_box


@torch.no_grad()
def _embed_bioclip(crop: Image.Image, clf) -> torch.Tensor:
    """
    Embed a single GT crop with BioCLIP (dim=512).
    Matches BioCLIPEmbedder in embedding_utils.py.
    """
    clf.model.eval()
    emb = clf.create_image_features([crop], normalize=False)  # (1, 512)
    return F.normalize(emb[0].float(), dim=-1).cpu()


@torch.no_grad()
def _embed_dinov3(crop: Image.Image, processor, model) -> torch.Tensor:
    """
    Embed a single GT crop with DINOv3 (pooler_output).
    Matches DINOv3Embedder in embedding_utils.py.
    """
    pv = processor(images=[crop], return_tensors="pt")["pixel_values"].to(DEVICE)
    out = model(pixel_values=pv)
    emb = out.pooler_output[0].float()          # (D,) on GPU
    return F.normalize(emb, dim=-1).cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_support_embeddings(
    support_examples: List[dict],
    embedding_backend: str,
    model_bundle,
    device: str,                # kept for API compat; we always use module-level DEVICE
    src_path: str = None,
    crop_size: int = 1024,      # used only by OWLv2 centered-crop path (legacy)
    mask_background: str = "zero",  # 'zero'|'mean'|'none' — GT background suppression (dinov3/bioclip)
) -> Tuple[Dict[str, torch.Tensor], list]:
    """
    Extract class-support embeddings from GT-annotated crops.

    OWLv2  → centered crop of `crop_size` around GT box centre → best-anchor
             selection via IoU → vision_model.pooler_output (1024-dim).
             generated_boxes entries include an "iou" field for the crop-size
             optimisation loop in generate_class_supports_main.py.

    BioCLIP → exact GT box crop → create_image_features (512-dim).
    DINOv3  → exact GT box crop → AutoModel.pooler_output (1024-dim).

    Args
    ----
    support_examples : list of {"image_path", "bounding_box", "class"} dicts
    embedding_backend: "owlv2" | "bioclip" | "dinov3"
    model_bundle     : return value of load_embedding_model()
    device           : ignored (kept for API compat; module-level DEVICE is used)
    src_path         : base path prepended to image_path fields
    crop_size        : centered-crop size for OWLv2 (ignored for other backends)

    Returns
    -------
    class_supports  : dict[class_name -> Tensor (N, D)]  stacked supports per class
    generated_boxes : list of annotation dicts (includes "iou" for owlv2)
    """
    class_supports: Dict[str, list] = {}
    generated_boxes: list = []

    print(f"  [DEBUG] extract_support_embeddings: backend={embedding_backend}, n_examples={len(support_examples)}, src_path={src_path}")

    if not support_examples:
        print("  [ERROR] support_examples is empty — check annotation JSON.")
        return {}, []

    for idx, support in enumerate(support_examples):
        img_path   = os.path.join(src_path or "", support["image_path"])
        bbox       = support["bounding_box"]
        class_name = support["class"]

        print(f"  [DEBUG] [{idx}] class={class_name} | img={img_path} | bbox={bbox}")

        if not os.path.exists(img_path):
            print(f"  [ERROR] Image not found: {img_path}")
            continue

        img  = Image.open(img_path).convert("RGB")
        W, H = img.size

        x1 = max(0, min(int(bbox[0]), W - 1))
        y1 = max(0, min(int(bbox[1]), H - 1))
        x2 = max(0, min(int(bbox[2]), W))
        y2 = max(0, min(int(bbox[3]), H))

        print(f"  [DEBUG] [{idx}] image size=({W},{H}), clamped box=({x1},{y1},{x2},{y2})")

        if x2 <= x1 or y2 <= y1:
            print(f"  [WARN] Degenerate box {bbox} clamped to ({x1},{y1},{x2},{y2}), skipping.")
            continue

        crop = img.crop((x1, y1, x2, y2))

        # For mask-based backends, suppress GT background outside the segmentation
        # mask so supports match the mask-cropped proposals (same embedding space).
        if embedding_backend in ("bioclip", "dinov3"):
            seg = support.get("segmentation")
            mask = _decode_segmentation(seg, H, W) if seg is not None else None
            if mask is not None:
                crop = _apply_mask_to_crop(img, (x1, y1, x2, y2), mask, mask_background)
            elif seg is None:
                print(f"  [DEBUG] [{idx}] no segmentation field — using plain box crop.")

        # Embed with the selected backend
        if embedding_backend == "owlv2":
            processor, model = model_bundle
            # OWLv2: centered-crop + best-anchor selection → pooler_output (1024,)
            emb, gen_box_extra = _embed_owlv2_centered(
                img, bbox, processor, model, crop_size
            )
            generated_boxes.append({
                "image_path":   support["image_path"],
                "bounding_box": gen_box_extra["bounding_box"],
                "class":        class_name,
                "iou":          gen_box_extra["iou"],
            })
            if class_name not in class_supports:
                class_supports[class_name] = []
            class_supports[class_name].append(emb)
            continue   # skip the generic generated_boxes.append below

        elif embedding_backend == "bioclip":
            emb = _embed_bioclip(crop, model_bundle)          # (512,)

        elif embedding_backend == "dinov3":
            processor, model = model_bundle
            emb = _embed_dinov3(crop, processor, model)       # (D,)

        else:
            raise ValueError(f"Unknown backend '{embedding_backend}'")

        generated_boxes.append({
            "image_path":   support["image_path"],
            "bounding_box": [x1, y1, x2, y2],
            "class":        class_name,
        })

        if class_name not in class_supports:
            class_supports[class_name] = []
        class_supports[class_name].append(emb)   # list of (D,) tensors

    # Stack per-class lists → (N, D) tensors
    print(f"  [DEBUG] Raw class_supports keys collected: {list(class_supports.keys())}")
    for c, embs in class_supports.items():
        print(f"  [DEBUG]   '{c}': {len(embs)} embedding(s), each shape={embs[0].shape}")

    if not class_supports:
        print("  [ERROR] class_supports is empty — check src_path, image_path, bounding boxes.")
        return {}, generated_boxes

    stacked: Dict[str, torch.Tensor] = {
        c: torch.stack(embs, dim=0)
        for c, embs in class_supports.items()
    }

    print(f"  [DEBUG] Final stacked supports:")
    for c, t in stacked.items():
        print(f"  [DEBUG]   '{c}': {t.shape}")

    return stacked, generated_boxes