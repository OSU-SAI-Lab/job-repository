"""
OWLv2-based tiled proposal generation.

Slices images into overlapping tiles via OverlappingTileDataset,
runs OWLv2 objectness inference in batches (no text query needed),
then stitches predictions back to original image coordinates using NMS.

The OWLv2 image-embedder produces native backbone features for every
anchor, so features can come either from:
  - "owlv2"   – the native OWLv2 anchor features  (dim 768, fastest)
  - "dinov3"  – DINOv3 crops via embedding_utils   (dim 384 / 768)
  - "bioclip" – BioCLIP crops via embedding_utils  (dim 512)

Output contract (matches optimize_objectness_threshold/ot_main.py):
    results[image_path] = {
        "features": np.ndarray  (K, C)  – per-box embeddings
        "boxes":    np.ndarray  (K, 4)  – [x1, y1, x2, y2] pixel coords
        "scores":   np.ndarray  (K,)    – objectness scores
    }
    or None for images with no detections.
"""

import sys
import collections
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from torch.utils.data import DataLoader
from transformers import Owlv2Processor, Owlv2ForObjectDetection

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.OverlappingTileDataset import OverlappingTileDataset
from proposal.embedding_utils import get_embedder

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

is_cuda = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda}")
DEVICE     = "cuda" if is_cuda else "cpu"
TORCH_DTYPE = torch.float16 if is_cuda else torch.float32

DEFAULT_MODEL_ID = "google/owlv2-large-patch14-ensemble"

_owlv2_processor: Owlv2Processor | None = None
_owlv2_model:     Owlv2ForObjectDetection | None = None
_loaded_model_id: str | None = None


def _load_owlv2(model_id: str):
    """Lazy-load OWLv2 once; reload only if model_id changes."""
    global _owlv2_processor, _owlv2_model, _loaded_model_id
    if _loaded_model_id == model_id:
        return _owlv2_processor, _owlv2_model
    print(f"[OWLv2] Loading model: {model_id}")
    _owlv2_processor = Owlv2Processor.from_pretrained(model_id)
    _owlv2_model = (
        Owlv2ForObjectDetection
        .from_pretrained(model_id, dtype=TORCH_DTYPE,
                         device_map="auto" if DEVICE == "cuda" else "cpu")
        .eval()
    )
    _loaded_model_id = model_id
    return _owlv2_processor, _owlv2_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_owlv2_on_batch(
    processor,
    model,
    pil_tiles: list,
    objectness_threshold: float,
):
    """
    Run OWLv2 objectness inference on a batch of PIL tile images.

    Returns per-tile lists of:
        native_feats_list : list of (M, C) tensors  – OWLv2 anchor features
        boxes_list        : list of (M, 4) tensors  – pixel [x1,y1,x2,y2]
        scores_list       : list of (M,)  tensors   – objectness scores
        shapes            : list of (H, W) tuples   – original tile shapes
    """
    np_tiles = [np.array(img) for img in pil_tiles]
    pv = processor(images=pil_tiles, return_tensors="pt", padding=False).pixel_values
    pv = pv.to(DEVICE, dtype=TORCH_DTYPE)

    with torch.no_grad():
        fmap        = model.image_embedder(pv)[0]          # (B, Hf, Wf, C)
        B, Hf, Wf, C = fmap.shape
        query_feats = fmap.view(B, Hf * Wf, C)            # (B, N, C)
        obj_scores  = model.objectness_predictor(query_feats).sigmoid()  # (B, N)
        raw_boxes   = model.box_predictor(
            query_feats, fmap, interpolate_pos_encoding=False
        )                                                   # (B, N, 4) cxcywh normalised

    native_feats_list = []
    boxes_list        = []
    scores_list       = []

    for i in range(B):
        scores_i = obj_scores[i]          # (N,)
        feats_i  = query_feats[i]         # (N, C)
        boxes_i  = raw_boxes[i]           # (N, 4)

        mask = scores_i > objectness_threshold
        if not mask.any():
            native_feats_list.append(torch.zeros((0, C)))
            boxes_list.append(torch.zeros((0, 4)))
            scores_list.append(torch.zeros((0,)))
            continue

        k_feats  = feats_i[mask]
        k_scores = scores_i[mask]
        k_boxes  = boxes_i[mask]          # cxcywh normalised

        # Convert cxcywh (normalised) → xyxy (pixels)
        orig_h, orig_w = np_tiles[i].shape[:2]
        padded = max(orig_h, orig_w)
        cx, cy, w, h = k_boxes.unbind(-1)
        cx, cy, w, h = cx * padded, cy * padded, w * padded, h * padded
        boxes_px = torch.stack([
            (cx - w / 2).clamp(0, orig_w),
            (cy - h / 2).clamp(0, orig_h),
            (cx + w / 2).clamp(0, orig_w),
            (cy + h / 2).clamp(0, orig_h),
        ], dim=1)

        # Per-tile NMS to thin out dense anchors before stitching
        keep = torchvision_nms(boxes_px, k_scores, 0.5)
        native_feats_list.append(k_feats[keep].cpu().float())
        boxes_list.append(boxes_px[keep].cpu().float())
        scores_list.append(k_scores[keep].cpu().float())

    return native_feats_list, boxes_list, scores_list


def _global_nms(boxes, scores, feats, iou_threshold=0.5):
    """
    NMS across all stitched tile results for one image.

    Args:
        boxes  : np.ndarray (N, 4)
        scores : np.ndarray (N,)
        feats  : np.ndarray (N, C)

    Returns:
        (feats, boxes, scores) as np.ndarray, or (None, None, None).
    """
    if len(boxes) == 0:
        return None, None, None
    b = torch.from_numpy(boxes).float()
    s = torch.from_numpy(scores).float()
    f = torch.from_numpy(feats).float()
    keep = torchvision_nms(b, s, iou_threshold)
    return f[keep].numpy(), b[keep].numpy(), s[keep].numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_proposals_tiled(
    image_paths,
    text_prompt="visual",          # unused for OWLv2 objectness – kept for API compat
    confidence_threshold=0.1,
    tile_size=960,
    overlap_ratio=0.25,
    batch_size=8,
    nms_iou_threshold=0.5,
    embedding_backend="owlv2",
    model_id=DEFAULT_MODEL_ID,
):
    """
    Generate bounding-box proposals for a list of images using OWLv2
    objectness scoring with tiled inference, then extract per-box
    embeddings and stitch back to original image coordinates.

    Args:
        image_paths:          List of Path / str image file paths.
        text_prompt:          Ignored (OWLv2 uses class-agnostic objectness).
                              Kept for API compatibility with generate_proposals.py.
        confidence_threshold: Minimum objectness score to keep a box.
        tile_size:            Tile size in pixels (square).
        overlap_ratio:        Fractional overlap between adjacent tiles.
        batch_size:           Number of tiles per inference call.
        nms_iou_threshold:    IoU threshold for global stitching NMS.
        embedding_backend:    'owlv2' (native anchor feats) | 'dinov3' | 'bioclip'.
                              'owlv2' is fastest; others re-crop & re-embed each box.
        model_id:             HuggingFace model ID for OWLv2.

    Returns:
        Dict mapping image_path (str) -> {
            'features': np.ndarray (K, C),
            'boxes':    np.ndarray (K, 4),   # [x1, y1, x2, y2] pixels
            'scores':   np.ndarray (K,),
        }
        Images with no detections map to None.
    """
    image_paths = [str(p) for p in image_paths]

    processor, model = _load_owlv2(model_id)

    # Load external embedder only when not using native OWLv2 features
    external_embedder = None
    if embedding_backend != "owlv2":
        external_embedder = get_embedder(embedding_backend)
        print(f"[OWLv2] Using external embedding backend: {embedding_backend}")
    else:
        print(f"[OWLv2] Using native OWLv2 anchor features")

    dataset = OverlappingTileDataset(
        image_paths=image_paths,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
    )

    # img_idx -> accumulated numpy arrays before stitching
    img_accum = collections.defaultdict(lambda: {
        "boxes":  [],
        "scores": [],
        "feats":  [],
    })

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: x,   # keep as list of dicts
    )

    print(f"[OWLv2] Total tiles: {len(dataset)} "
          f"(batch_size={batch_size}, tile_size={tile_size}, overlap={overlap_ratio})")

    for batch_idx, batch in enumerate(loader):
        tile_images = [item["image"]               for item in batch]
        coords_list = [item["metadata"]["coords"]   for item in batch]
        img_idxs    = [item["metadata"]["img_idx"]  for item in batch]

        native_feats_list, boxes_list, scores_list = _run_owlv2_on_batch(
            processor, model, tile_images, confidence_threshold
        )

        for tile_img, native_feats, tile_boxes, tile_scores, coords, img_idx in zip(
            tile_images, native_feats_list, boxes_list, scores_list, coords_list, img_idxs
        ):
            if tile_boxes.numel() == 0:
                continue

            # Shift tile-local boxes → original image coords
            ox, oy = coords[0].item(), coords[1].item()
            offset = torch.tensor([ox, oy, ox, oy], dtype=tile_boxes.dtype)
            global_boxes = (tile_boxes + offset).cpu()

            # Choose feature source
            if external_embedder is not None:
                # Re-crop detected boxes from the tile and embed externally
                local_boxes = tile_boxes.tolist()
                feats_np = external_embedder.embed_boxes(tile_img, local_boxes).numpy()
            else:
                # Use native OWLv2 anchor features directly
                feats_np = native_feats.numpy()

            img_accum[img_idx]["boxes"].append(global_boxes.numpy())
            img_accum[img_idx]["scores"].append(tile_scores.numpy())
            img_accum[img_idx]["feats"].append(feats_np)

        print(f"  [OWLv2] Batch {batch_idx + 1}/{len(loader)} done")

    # Stitch: concatenate all tile results per image, then global NMS
    results = {path: None for path in image_paths}

    for img_idx, accum in img_accum.items():
        boxes_raw  = np.vstack(accum["boxes"])   # (N, 4)
        scores_raw = np.hstack(accum["scores"])  # (N,)
        feats_raw  = np.vstack(accum["feats"])   # (N, C)

        feats_final, boxes_final, scores_final = _global_nms(
            boxes_raw, scores_raw, feats_raw, nms_iou_threshold
        )

        if boxes_final is not None:
            results[image_paths[img_idx]] = {
                "features": feats_final,   # (K, C)
                "boxes":    boxes_final,   # (K, 4)
                "scores":   scores_final,  # (K,)
            }
            print(f"  [OWLv2] {Path(image_paths[img_idx]).name}: "
                  f"{len(boxes_raw)} raw → {len(boxes_final)} after NMS")

    return results
