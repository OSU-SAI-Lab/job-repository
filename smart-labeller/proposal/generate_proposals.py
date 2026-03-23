"""
generate_proposals.py  –  CLI entry point for proposal generation.

Supported backends:
  sam3   – SAM3 tiled inference  (sam3_proposal.py)
  owlv2  – OWLv2 tiled inference (owlv2_proposal.py)  [coming soon]

Usage:
    python generate_proposals.py --backend sam3 --image_dir ./images --output_dir ./output
"""

import argparse
import os
from pathlib import Path


BACKENDS = ["sam3", "owlv2"]


def get_backend(name):
    if name == "sam3":
        from sam3_proposal import generate_proposals_tiled
        return generate_proposals_tiled
    elif name == "owlv2":
        from owlv2_proposal import generate_proposals_tiled
        return generate_proposals_tiled
    else:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {BACKENDS}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate object proposals with tiled inference"
    )
    parser.add_argument("--backend",       type=str,   default="sam3",
                        choices=BACKENDS,
                        help="Proposal backend to use (default: sam3)")
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
    parser.add_argument("--nms_iou",          type=float, default=0.5,
                        help="IoU threshold for stitching NMS (default: 0.5)")
    parser.add_argument("--text_prompt",      type=str,   default="visual",
                        help="Text prompt for the model (default: 'visual')")
    parser.add_argument("--embedding_backend", type=str,  default="owlv2",
                        choices=["dinov3", "bioclip", "owlv2"],
                        help="Embedding model for per-box features. "
                             "'owlv2' uses native OWLv2 anchor features (fastest when backend=owlv2). "
                             "'dinov3' or 'bioclip' re-crop and re-embed each box. (default: owlv2)")
    parser.add_argument("--model_id",          type=str,  default=None,
                        help="HuggingFace model ID override for the chosen backend ")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_dir   = Path(args.image_dir)
    image_files = sorted([f for f in image_dir.iterdir()
                          if f.suffix.lower() in image_extensions])

    print(f"Found {len(image_files)} images  |  backend: {args.backend}  |  embeddings: {args.embedding_backend}  |  SAHI: {args.is_sahi}")

    generate_proposals_tiled = get_backend(args.backend)

    extra_kwargs = {}
    if args.model_id:
        extra_kwargs["model_id"] = args.model_id

    # Only use tiling parameters if SAHI is enabled
    tile_size = args.tile_size if args.is_sahi else None
    overlap_ratio = args.overlap_ratio if args.is_sahi else None

    results = generate_proposals_tiled(
        image_paths=image_files,
        text_prompt=args.text_prompt,
        confidence_threshold=args.confidence,
        is_sahi=args.is_sahi,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        batch_size=args.batch_size,
        nms_iou_threshold=args.nms_iou,
        embedding_backend=args.embedding_backend,
        **extra_kwargs,
    )

    # Save results in the same format as optimize_objectness_threshold/ot_main.py:
    #   - one combined .npz  (all images, keyed by "{filename}_features/boxes/scores")
    #   - one JSON           (per-box annotations list)
    import json
    import time
    import numpy as np

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_data        = {}   # merged dict fed to np.savez
    all_annotations = []   # list of per-box annotation dicts

    for image_path, det in results.items():
        name = Path(image_path).name
        if det is None:
            print(f"No detections for {name}")
            continue

        features = det["features"]
        boxes    = det["boxes"]
        scores   = det["scores"]

        all_data[f"{name}_features"] = features
        all_data[f"{name}_boxes"]    = boxes
        all_data[f"{name}_scores"]   = scores

        for i in range(len(scores)):
            all_annotations.append({
                "image_path":    name,
                "bounding_box":  boxes[i].tolist() if hasattr(boxes[i], "tolist") else list(boxes[i]),
                "score":         float(scores[i]),
                "class":         str(float(scores[i])),   # placeholder class name
            })

    # Write combined .npz
    features_dir = Path(args.output_dir) / "proposals/tensors"
    boxes_dir    = Path(args.output_dir) / "proposals/generated_boxes"
    features_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = features_dir / f"features_{args.embedding_backend}_{args.backend}_{timestamp}.npz"
    json_path = boxes_dir    / f"generated_boxes_{args.embedding_backend}_{args.backend}_{timestamp}.json"

    if all_data:
        np.savez(npz_path, **all_data)
        print(f"Saved features → {npz_path}")

    with open(json_path, "w") as f:
        json.dump({"annotations": all_annotations}, f, indent=2)
    print(f"Saved {len(all_annotations)} box annotations → {json_path}")