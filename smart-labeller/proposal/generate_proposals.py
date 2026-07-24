"""
generate_proposals.py  –  CLI entry point for SAM3 mask-proposal generation.

Pipeline: SAM3 (masks) + DINOv3 (embeddings).  The old box pipeline and the
OWLv2 proposer/embedder are gone.

  proposer  : sam3   – SAM3 tiled instance segmentation (sam3_proposal.py)
  embedder  : dinov3 – DINOv3 embeddings of each masked instance

SAHI (sliced inference) is ON by default — pass --no_sahi to process whole images.

Usage:
    python generate_proposals.py --image_dir ./images --output_dir ./output
    python generate_proposals.py --image_dir ./images --output_dir ./output --no_sahi
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from sam3_proposal import generate_proposals_tiled


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate SAM3 mask proposals with DINOv3 embeddings (tiled/SAHI)"
    )
    parser.add_argument("--image_dir",     type=str,   required=True,
                        help="Directory containing images to process")
    parser.add_argument("--output_dir",    type=str,   default="output",
                        help="Directory to save results (default: output)")
    parser.add_argument("--confidence",    type=float, default=0.1,
                        help="SAM3 instance score threshold (default: 0.1)")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="SAM3 mask binarization threshold (default: 0.5)")
    parser.add_argument("--no_sahi",       action="store_true", default=False,
                        help="Disable SAHI tiling and process whole images "
                             "(SAHI is ON by default)")
    parser.add_argument("--tile_size",     type=int,   default=960,
                        help="Tile size in pixels (SAHI only) (default: 960)")
    parser.add_argument("--overlap_ratio", type=float, default=0.2,
                        help="Overlap ratio between tiles (SAHI only) (default: 0.2)")
    parser.add_argument("--batch_size",    type=int,   default=4,
                        help="Tiles per inference batch (default: 4)")
    parser.add_argument("--nms_iou",       type=float, default=0.5,
                        help="Mask-IoU threshold for stitching NMS (default: 0.5)")
    parser.add_argument("--text_prompt",   type=str,   default="visual",
                        help="Text prompt for SAM3 (default: 'visual')")
    parser.add_argument("--embedder",      type=str,   default="dinov3",
                        choices=["dinov3"],
                        help="Embedding model for per-instance features (only dinov3)")
    parser.add_argument("--log_level",     type=str,   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    is_sahi = not args.no_sahi

    os.makedirs(args.output_dir, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_dir   = Path(args.image_dir)
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix.lower() in image_extensions])

    logger.info(f"Found {len(image_files)} images in {image_dir}")
    logger.info(f"Proposer: sam3 | Embedder: {args.embedder}")
    logger.info(f"SAHI: {is_sahi}")

    tile_size = args.tile_size if is_sahi else None
    overlap_ratio = args.overlap_ratio if is_sahi else None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pipeline_start = time.time()

    results = generate_proposals_tiled(
        image_paths=image_files,
        text_prompt=args.text_prompt,
        confidence_threshold=args.confidence,
        is_sahi=is_sahi,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        batch_size=args.batch_size,
        nms_iou_threshold=args.nms_iou,
        embedding_backend=args.embedder,
        mask_threshold=args.mask_threshold,
    )

    # ── Serialize: features/scores in .npz (masks as object array), masks as RLE ──
    all_data        = {}   # merged dict fed to np.savez
    all_annotations = []   # list of per-instance annotation dicts (RLE segmentation)

    for image_path, det in results.items():
        name = Path(image_path).name
        if det is None:
            logger.debug(f"No detections for {name}")
            continue

        features = det["features"]
        masks    = det["masks"]     # list[RLE]
        scores   = det["scores"]

        all_data[f"{name}_features"] = features
        all_data[f"{name}_scores"]   = scores
        all_data[f"{name}_masks"]    = np.array(masks, dtype=object)

        for i in range(len(scores)):
            all_annotations.append({
                "image_path":   name,
                "segmentation": masks[i],          # COCO RLE
                "score":        float(scores[i]),
                "class":        str(float(scores[i])),  # placeholder class name
            })

    features_dir = Path(args.output_dir) / "proposals/tensors"
    masks_dir    = Path(args.output_dir) / "proposals/generated_masks"
    features_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = features_dir / f"features_{args.embedder}_sam3_{timestamp}.npz"
    json_path = masks_dir    / f"generated_masks_{args.embedder}_sam3_{timestamp}.json"

    if all_data:
        np.savez(npz_path, **all_data)
        n_images = len(all_data) // 3   # features + scores + masks per image
        logger.info(f"Saved features ({n_images} images) → {npz_path}")

    with open(json_path, "w") as f:
        json.dump({"annotations": all_annotations}, f)
    logger.info(f"Saved {len(all_annotations)} mask annotations → {json_path}")

    total_elapsed = time.time() - pipeline_start
    logger.info(f"Proposal generation finished in {total_elapsed:.1f}s")
