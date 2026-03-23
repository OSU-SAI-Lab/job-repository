"""
SAM3-based tiled proposal generation.

Slices images into overlapping tiles via OverlappingTileDataset,
runs SAM3 inference in batches, then stitches predictions back to
original image coordinates using NMS to suppress overlap duplicates.

Output contract (matches optimize_objectness_threshold/ot_main.py):
    results[image_path] = {
        "features": np.ndarray  (K, C)  – per-box embeddings
        "boxes":    np.ndarray  (K, 4)  – [x1, y1, x2, y2] pixel coords
        "scores":   np.ndarray  (K,)    – confidence scores
    }
"""

import sys
import collections
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms as torchvision_nms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import Sam3Model, Sam3Processor

sys.path.append(str(Path(__file__).resolve().parents[1]))
from OverlappingTileDataset import OverlappingTileDataset
from proposal.embedding_utils import get_embedder

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

is_cuda = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda}")
DEVICE = "cuda" if is_cuda else "cpu"

MODEL_ID = "facebook/sam3"

sam3_model     = Sam3Model.from_pretrained(MODEL_ID).to(DEVICE)
sam3_processor = Sam3Processor.from_pretrained(MODEL_ID)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_sam3_on_batch(tile_images, text_prompt, confidence_threshold):
    inputs = sam3_processor(
        images=tile_images,
        text=[text_prompt] * len(tile_images),  # one prompt per image
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    target_sizes = inputs.get("original_sizes").tolist()
    batch_results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence_threshold,
        mask_threshold=confidence_threshold,
        target_sizes=target_sizes
    )

    boxes_list  = []
    scores_list = []

    for result in batch_results:
        if len(result["masks"]) == 0:
            boxes_list.append(torch.zeros((0, 4)))
            scores_list.append(torch.zeros((0,)))
            continue

        # post_process_instance_segmentation returns boxes as [x1, y1, x2, y2]
        boxes  = result["boxes"].float().cpu()   # (M, 4) tensor
        scores = result["scores"].float().cpu()  # (M,)   tensor

        boxes_list.append(boxes)
        scores_list.append(scores)

    return boxes_list, scores_list


def _global_nms(boxes, scores, feats, iou_threshold=0.5):
    """
    Apply NMS across all stitched tile boxes for one image.

    Args:
        boxes:  np.ndarray (N, 4)
        scores: np.ndarray (N,)
        feats:  np.ndarray (N, C)

    Returns:
        Tuple of (feats, boxes, scores) as np.ndarray after NMS,
        or (None, None, None) if nothing to keep.
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
    text_prompt="visual",
    confidence_threshold=0.5,
    is_sahi=False,
    tile_size=960,
    overlap_ratio=0.2,
    batch_size=8,
    nms_iou_threshold=0.5,
    embedding_backend="dinov3",
):
    """
    Generate bounding-box proposals for a list of images using SAM3,
    with optional tiled inference via SAHI.

    Args:
        image_paths:        List of Path / str image file paths.
        text_prompt:        Text prompt passed to SAM3 for every tile.
        confidence_threshold: Minimum SAM3 score to keep a box.
        is_sahi:            Enable SAHI (Sliced Aided Hyper Inference).
                            If False, processes whole image. If True, tiles and stitches.
        tile_size:          Tile size in pixels (only used when is_sahi=True).
        overlap_ratio:      Fractional overlap between tiles (only used when is_sahi=True).
        batch_size:         Number of tiles/images per inference call.
        nms_iou_threshold:  IoU threshold for stitching NMS.
        embedding_backend:  One of 'dinov3', 'bioclip', 'owlv2'.

    Returns:
        Dict mapping image_path (str) -> {
            'features': np.ndarray (K, C),
            'boxes':    np.ndarray (K, 4),   # [x1, y1, x2, y2] pixels
            'scores':   np.ndarray (K,),
        }
        Images with no detections map to None.
    """
    image_paths = [str(p) for p in image_paths]

    # Load the chosen embedding model once
    embedder = get_embedder(embedding_backend)

    # Use tiling only if SAHI is enabled
    if is_sahi:
        dataset = OverlappingTileDataset(
            image_paths=image_paths,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
        )
    else:
        # Whole-image mode: treat entire image as single "tile"
        from PIL import Image
        dataset = []
        for img_idx, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            dataset.append({
                "image": img,
                "metadata": {
                    "img_idx": img_idx,
                    "coords": torch.tensor([0, 0], dtype=torch.float32),
                }
            })

    # img_idx -> accumulated lists before stitching
    img_accum = collections.defaultdict(lambda: {
        "boxes":  [],
        "scores": [],
        "feats":  [],
    })

    if is_sahi:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: x,   # keep as list of dicts
        )
    else:
        # For whole-image mode, batch images directly
        loader = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]

    if is_sahi:
        print(f"[SAM3] SAHI enabled | Total tiles: {len(dataset)} "
              f"(batch_size={batch_size}, tile_size={tile_size}, overlap={overlap_ratio})")
    else:
        print(f"[SAM3] Processing {len(dataset)} whole images (SAHI disabled)")

    for batch_idx, batch in enumerate(loader):
        tile_images = [item["image"]               for item in batch]
        coords_list = [item["metadata"]["coords"]   for item in batch]
        img_idxs    = [item["metadata"]["img_idx"]  for item in batch]

        boxes_list, scores_list = _run_sam3_on_batch(
            tile_images, text_prompt, confidence_threshold
        )

        for tile_img, tile_boxes, tile_scores, coords, img_idx in zip(
            tile_images, boxes_list, scores_list, coords_list, img_idxs
        ):
            if tile_boxes.numel() == 0:
                continue

            # --- shift tile-local boxes → original image coords ---
            ox, oy = coords[0].item(), coords[1].item()
            offset = torch.tensor([ox, oy, ox, oy], dtype=tile_boxes.dtype, device=tile_boxes.device)
            global_boxes = (tile_boxes + offset).cpu()

            # --- extract per-box embeddings from the tile crop ---
            box_list = global_boxes.tolist()
            # Shift back to tile-local coords for cropping the tile image
            local_boxes = tile_boxes.tolist()
            tile_feats = embedder.embed_boxes(tile_img, local_boxes)   # (M, C) cpu normalised

            img_accum[img_idx]["boxes"].append(global_boxes.numpy())
            img_accum[img_idx]["scores"].append(tile_scores.cpu().numpy())
            img_accum[img_idx]["feats"].append(tile_feats.numpy())

        print(f"  [SAM3] Batch {batch_idx + 1}/{len(loader)} done")

    # --- Stitch: cat all tile results per image, then global NMS ---
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
            print(f"  [SAM3] {Path(image_paths[img_idx]).name}: "
                  f"{len(boxes_raw)} raw → {len(boxes_final)} after NMS")

    return results
