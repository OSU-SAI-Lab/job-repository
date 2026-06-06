"""
generate_proposals.py  –  CLI entry point for proposal generation.

Supported proposers (backends):
  sam3   – SAM3 tiled inference  (sam3_proposal.py)
  owlv2  – OWLv2 tiled inference (owlv2_proposal.py)

Supported embedders:
  owlv2  – OWLv2 embeddings (native anchor features or re-crop)
  dinov3 – DINOv3 embeddings
  bioclip – BioCLIP embeddings

Constraints:
  - If proposer is 'sam3', OWLv2 cannot be used as an embedder
  - If proposer is 'owlv2', OWLv2 can be used as embedder (native features)

Usage:
    python generate_proposals.py --proposers sam3 owlv2 --image_dir ./images --embedders owlv2 dinov3 bioclip --output_dir ./output
    python generate_proposals.py --proposers sam3 --image_dir ./images --embedders dinov3 bioclip --output_dir ./output
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
import time
import numpy as np
from pycocotools import mask as mask_util


PROPOSERS = ["sam3", "owlv2"]
EMBEDDERS = ["owlv2", "dinov3", "bioclip"]


def get_proposer(name):
    if name == "sam3":
        from sam3_proposal import generate_proposals_tiled
        return generate_proposals_tiled
    elif name == "owlv2":
        from owlv2_proposal import generate_proposals_tiled
        return generate_proposals_tiled
    else:
        raise ValueError(f"Unknown proposer '{name}'. Choose from: {PROPOSERS}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate object proposals with tiled inference for multiple proposers and embedders"
    )
    parser.add_argument("--proposers",     type=str,   nargs="+", default=["sam3"],
                        choices=PROPOSERS,
                        help="Proposal backends to use (default: sam3). Can specify multiple: --proposers sam3 owlv2")
    parser.add_argument("--image_dir",     type=str,   required=True,
                        help="Directory containing images to process")
    parser.add_argument("--output_dir",    type=str,   default="output",
                        help="Directory to save results (default: output)")
    parser.add_argument("--confidence",    type=float, default=0.1,
                        help="Confidence threshold (default: 0.1)")
    parser.add_argument("--is_sahi",       action="store_true", default=False,
                        help="Enable SAHI (Sliced Aided Hyper Inference) for tiling (default: False, process whole image)")
    parser.add_argument("--tile_size",     type=int,   default=960,
                        help="Tile size in pixels (only used with --is_sahi) (default: 960)")
    parser.add_argument("--overlap_ratio", type=float, default=0.2,
                        help="Overlap ratio between tiles (only used with --is_sahi) (default: 0.2)")
    parser.add_argument("--batch_size",    type=int,   default=4,
                        help="Tiles per inference batch (default: 8)")
    parser.add_argument("--nms_iou",       type=float, default=0.5,
                        help="IoU threshold for stitching NMS (default: 0.5)")
    parser.add_argument("--text_prompt",   type=str,   default="visual",
                        help="Text prompt for the model (default: 'visual')")
    parser.add_argument("--embedders",     type=str,   nargs="+", default=["owlv2"],
                        choices=EMBEDDERS,
                        help="Embedding models for per-box features (default: owlv2). "
                             "Can specify multiple: --embedders owlv2 dinov3 bioclip. "
                             "Note: If proposer is 'sam3', OWLv2 cannot be used as embedder.")
    parser.add_argument("--model_id",      type=str,   default=None,
                        help="HuggingFace model ID override for the chosen backend")
    parser.add_argument("--log_level",     type=str,   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO). Use DEBUG to see per-batch progress.")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Validation: SAM3 proposer cannot use OWLv2 embedder - remove it and continue
    if "sam3" in args.proposers and "owlv2" in args.embedders:
        logger.warning("OWLv2 embedder cannot be used with SAM3 proposer. Removing OWLv2 from embedders.")
        args.embedders = [e for e in args.embedders if e != "owlv2"]
        if not args.embedders:
            logger.error("No valid embedders left after removing OWLv2. Use --embedders dinov3 bioclip.")
            sys.exit(1)

    # Auto-enable: If OWLv2 is a proposer, ensure OWLv2 is an embedder for it
    if "owlv2" in args.proposers and "owlv2" not in args.embedders:
        logger.info("OWLv2 proposer detected — adding OWLv2 as embedder automatically.")
        args.embedders = list(args.embedders) + ["owlv2"]

    os.makedirs(args.output_dir, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_dir   = Path(args.image_dir)
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix.lower() in image_extensions])

    logger.info(f"Found {len(image_files)} images in {image_dir}")
    logger.info(f"Proposers: {', '.join(args.proposers)}")
    logger.info(f"Embedders: {', '.join(args.embedders)}")
    logger.info(f"SAHI: {args.is_sahi}")

    # Only use tiling parameters if SAHI is enabled
    tile_size = args.tile_size if args.is_sahi else None
    overlap_ratio = args.overlap_ratio if args.is_sahi else None

    extra_kwargs = {}
    if args.model_id:
        extra_kwargs["model_id"] = args.model_id

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    pipeline_start = time.time()

    # Process each proposer
    for proposer in args.proposers:
        logger.info(f"--- Proposer: {proposer} ---")
        proposer_start = time.time()

        generate_proposals_tiled = get_proposer(proposer)

        results = generate_proposals_tiled(
            image_paths=image_files,
            text_prompt=args.text_prompt,
            confidence_threshold=args.confidence,
            is_sahi=args.is_sahi,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            batch_size=args.batch_size,
            nms_iou_threshold=args.nms_iou,
            embedding_backend=args.embedders[0],  # Start with first embedder for initial proposals
            **extra_kwargs,
        )

        # For each embedder, generate embeddings
        for embedder in args.embedders:
            logger.info(f"Generating embeddings with {embedder}...")

            all_data        = {}   # merged dict fed to np.savez
            all_annotations = []   # list of per-box annotation dicts

            for image_path, det in results.items():
                name = Path(image_path).name
                if det is None:
                    logger.debug(f"No detections for {name}")
                    continue

                features = det["features"]
                boxes    = det["boxes"]
                scores   = det["scores"]
                masks    = det.get("masks", [])  # list of (H,W) bool arrays; empty for non-SAM3 proposers

                all_data[f"{name}_features"] = features
                all_data[f"{name}_boxes"]    = boxes
                all_data[f"{name}_scores"]   = scores

                for i in range(len(scores)):
                    ann = {
                        "image_path":    name,
                        "bounding_box":  boxes[i].tolist() if hasattr(boxes[i], "tolist") else list(boxes[i]),
                        "score":         float(scores[i]),
                        "class":         str(float(scores[i])),   # placeholder class name
                    }
                    if i < len(masks) and masks[i] is not None:
                        rle = mask_util.encode(np.asfortranarray(masks[i].astype(np.uint8)))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        ann["segmentation"] = rle
                    all_annotations.append(ann)

            # Write combined .npz and JSON for this proposer-embedder combination
            features_dir = Path(args.output_dir) / "proposals/tensors"
            boxes_dir    = Path(args.output_dir) / "proposals/generated_boxes"
            features_dir.mkdir(parents=True, exist_ok=True)
            boxes_dir.mkdir(parents=True, exist_ok=True)

            npz_path  = features_dir / f"features_{embedder}_{proposer}_{timestamp}.npz"
            json_path = boxes_dir    / f"generated_boxes_{embedder}_{proposer}_{timestamp}.json"

            if all_data:
                np.savez(npz_path, **all_data)
                logger.info(f"Saved features ({len(all_data) // 3} images) → {npz_path}")

            with open(json_path, "w") as f:
                json.dump({"annotations": all_annotations}, f, indent=2)
            logger.info(f"Saved {len(all_annotations)} box annotations → {json_path}")

        elapsed = time.time() - proposer_start
        logger.info(f"Proposer '{proposer}' finished in {elapsed:.1f}s")

    total_elapsed = time.time() - pipeline_start
    logger.info(f"All proposers and embedders processed successfully in {total_elapsed:.1f}s")