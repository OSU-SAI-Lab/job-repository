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
    """
    Run SAM3 on a batch of PIL tile images.

    Returns:
        boxes_list:  List of tensors (M, 4) [x1,y1,x2,y2], one per tile.
        scores_list: List of tensors (M,), one per tile.
        Both lists contain empty tensors for tiles with no detections.
    """
    try:
        inputs = sam3_processor(
            images=tile_images,
            text=[text_prompt] * len(tile_images),
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = sam3_model(**inputs)

        raw_boxes = sam3_processor.post_process_masks_for_box_prediction(
            outputs,
            original_sizes=inputs["original_sizes"],
            reshaped_input_sizes=inputs["reshaped_input_sizes"]
        )

        boxes_list  = []
        scores_list = []

        if hasattr(outputs, "scores"):
            for boxes, scores in zip(raw_boxes, outputs.scores):
                mask = scores >= confidence_threshold
                if mask.any():
                    boxes_list.append(boxes[mask])
                    scores_list.append(scores[mask])
                else:
                    boxes_list.append(torch.zeros((0, 4)))
                    scores_list.append(torch.zeros((0,)))
        else:
            for boxes in raw_boxes:
                b = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes)
                boxes_list.append(b)
                scores_list.append(torch.ones(b.shape[0]))  # no score → 1.0

        return boxes_list, scores_list

    except Exception as e:
        print(f"Error during SAM3 inference: {str(e)}")
        empty_b = [torch.zeros((0, 4)) for _ in tile_images]
        empty_s = [torch.zeros((0,))   for _ in tile_images]
        return empty_b, empty_s


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
    tile_size=960,
    overlap_ratio=0.2,
    batch_size=8,
    nms_iou_threshold=0.5,
    embedding_backend="dinov3",
):
    """
    Generate bounding-box proposals for a list of images using SAM3
    with tiled inference, then extract per-box embeddings and stitch
    everything back to original image coordinates.

    Args:
        image_paths:        List of Path / str image file paths.
        text_prompt:        Text prompt passed to SAM3 for every tile.
        confidence_threshold: Minimum SAM3 score to keep a box.
        tile_size:          Tile size in pixels (square).
        overlap_ratio:      Fractional overlap between adjacent tiles.
        batch_size:         Number of tiles per SAM3 inference call.
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

    dataset = OverlappingTileDataset(
        image_paths=image_paths,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
    )

    # img_idx -> accumulated lists before stitching
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

    print(f"[SAM3] Total tiles: {len(dataset)} "
          f"(batch_size={batch_size}, tile_size={tile_size}, overlap={overlap_ratio})")

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
            offset = torch.tensor([ox, oy, ox, oy], dtype=tile_boxes.dtype)
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
