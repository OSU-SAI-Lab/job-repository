"""
class_support/generate_class_supports_main.py

Stage 1 of the segmentation pipeline: build per-class DINOv3 support embeddings
from ground-truth annotations.

Each GT box is turned into a mask via the SAM3 Tracker (box prompt), the masked
region is embedded with DINOv3, and per-class vectors are stacked and saved as
``class_supports_dinov3_<timestamp>.npz``.  The support masks (COCO RLE) are also
saved for traceability.
"""

import os
import time
import json
import argparse

import numpy as np
import torch

from class_supports_utils import (
    extract_support_embeddings,
    load_embedding_model,
    load_sam3_tracker,
    DEFAULT_DINOV3_MODEL,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(ann_file_path, src_path, output_path, model_name=None):
    print("Current working directory:", os.getcwd())

    if ann_file_path is None:
        print("No annotation file provided.")
        return False

    with open(ann_file_path, "r") as f:
        support_examples = json.load(f).get("annotations", [])
    print(f"Loaded {len(support_examples)} support examples from {ann_file_path}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    masks_dir   = os.path.join(output_path, "class_supports/support_masks")
    tensors_dir = os.path.join(output_path, "class_supports/tensors")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    # Load models once.
    dinov3_bundle  = load_embedding_model("dinov3", model_name=model_name)
    tracker_bundle = load_sam3_tracker()

    class_supports, support_masks = extract_support_embeddings(
        support_examples,
        embedding_backend="dinov3",
        model_bundle=dinov3_bundle,
        device=DEVICE,
        src_path=src_path,
        tracker_bundle=tracker_bundle,
    )

    if not class_supports:
        print("Failed: no class supports extracted.")
        return False

    # Save per-class embeddings.
    tensor_path = os.path.join(tensors_dir, f"class_supports_dinov3_{timestamp}.npz")
    class_supports_np = {k: v.cpu().numpy() for k, v in class_supports.items()}
    np.savez(tensor_path, **class_supports_np)
    print(f"Saved class supports [dinov3] → {tensor_path}")

    # Save support masks (RLE) for traceability.
    masks_path = os.path.join(masks_dir, f"support_masks_{timestamp}.json")
    with open(masks_path, "w") as f:
        json.dump({"annotations": support_masks}, f)
    print(f"Saved {len(support_masks)} support masks → {masks_path}")

    print("\nDone.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-Support Embedding Generator (DINOv3, segmentation)")
    parser.add_argument("--ann_path",          type=str,   default=None,
                        help="Annotation file path (COCO-style, with bounding_box)")
    parser.add_argument("--src_path",          type=str,   default=None,
                        help="Source image directory")
    parser.add_argument("--output_path",       type=str,   default=None,
                        help="Path to save output")
    parser.add_argument("--embedding_backend", type=str,   default="dinov3",
                        choices=["dinov3"],
                        help="Embedding backend (only dinov3 is supported)")
    parser.add_argument("--model_name",        type=str,   default=None,
                        help=f"HuggingFace DINOv3 model ID override (default: {DEFAULT_DINOV3_MODEL})")
    parser.add_argument("--device",            type=str,   default=DEVICE,
                        help="Device to use (default: auto)")
    args = parser.parse_args()

    DEVICE = args.device

    main(
        ann_file_path=args.ann_path,
        src_path=args.src_path,
        output_path=args.output_path,
        model_name=args.model_name,
    )
