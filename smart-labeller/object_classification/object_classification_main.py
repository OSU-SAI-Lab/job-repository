"""
object_classification/object_classification_main.py

Stage 3: few-shot classification of SAM3 mask proposals (DINOv3 cosine).

Flow
----
1. Load the class-support .npz (DINOv3).
2. Load the proposal-feature .npz (features + scores + masks as RLE).
3. For each query image, run cosine-similarity detection.
4. Save a detection JSON with per-instance ``segmentation`` (RLE), score, class.

Dimension safety on load
------------------------
Support tensors are squeezed to 2-D (N, D).  A transposed (D, N) save is
auto-corrected.  An explicit float32 cast prevents half-precision surprises.
"""

from __future__ import annotations

import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from object_classification_utils import run_detection_for_backend


DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

OBJECTNESS_THRESHOLD = 0.1
SIMILARITY_THRESHOLD = 0.2
NMS_IOU_THRESHOLD    = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_load_supports(npz) -> dict:
    """Load class supports, enforce (N, D) shape and float32; fix transposed saves."""
    supports = {}
    for k in npz.files:
        t = torch.from_numpy(npz[k]).float()
        t = t.squeeze()
        if t.dim() == 1:
            t = t.unsqueeze(0)   # (1, D)
        if t.dim() == 2 and t.shape[0] > t.shape[1] * 4:
            print(f"  [dinov3] '{k}': detected transposed shape {tuple(t.shape)}, transposing.")
            t = t.T
        print(f"  [dinov3] '{k}': {tuple(t.shape)}")
        supports[k] = t
    return supports


def _build_object_features(npz, query_file_name: str) -> dict:
    """Extract feature/score/mask tensors for a single query image from the npz."""
    if npz is None:
        return {"features": None, "masks": None, "scores": None}
    try:
        features = torch.from_numpy(npz[f"{query_file_name}_features"]).float()
        scores   = torch.from_numpy(npz[f"{query_file_name}_scores"]).float()
        masks    = list(npz[f"{query_file_name}_masks"])   # object array → list[RLE]
        print(f"  [dinov3] {query_file_name}: proposals={tuple(features.shape)}")
        return {"features": features, "masks": masks, "scores": scores}
    except KeyError as e:
        print(f"  [dinov3] Key not found for '{query_file_name}': {e}")
        return {"features": None, "masks": None, "scores": None}


def _save_detections(annotations: list, timestamp: str, output_path: str):
    det_dir = Path(output_path) / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    fname = f"detections_dinov3_{SIMILARITY_THRESHOLD}_{OBJECTNESS_THRESHOLD}_{timestamp}.json"
    path  = det_dir / fname
    with open(path, "w") as f:
        json.dump({"annotations": annotations}, f)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def process_image(query_file_name, image_name, class_supports, features_npz) -> list:
    """Run detection for one image."""
    if not class_supports:
        print("  [dinov3] No class supports — skipping.")
        return []

    object_features = _build_object_features(features_npz, query_file_name)
    if object_features["features"] is None:
        return []

    print("  [dinov3] Running detection...")
    dets = run_detection_for_backend(
        backend="dinov3",
        class_supports=class_supports,
        object_features=object_features,
        device=DEVICE,
        objectness_threshold=OBJECTNESS_THRESHOLD,
        similarity_threshold=SIMILARITY_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
    )
    for d in dets:
        d["image_path"] = image_name
    print(f"  [dinov3] {len(dets)} detection(s).")
    return dets


def main(qry_path, output_path, is_query_dir, class_support_file, object_features_file):
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load class supports.
    if class_support_file and os.path.exists(class_support_file):
        print(f"[dinov3] Loading class supports: {class_support_file}")
        class_supports = _safe_load_supports(np.load(class_support_file))
    else:
        print(f"[dinov3] Class support file missing: {class_support_file}")
        class_supports = {}

    # Load proposal features (allow_pickle for the RLE object array).
    if object_features_file and os.path.exists(object_features_file):
        print(f"[dinov3] Loading proposal features: {object_features_file}")
        features_npz = np.load(object_features_file, allow_pickle=True)
    else:
        print(f"[dinov3] Proposal features file missing: {object_features_file}")
        features_npz = None

    # Query images.
    if is_query_dir:
        image_files = [
            p for p in Path(qry_path).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    else:
        image_files = [Path(qry_path)]

    global_annotations = []
    for qf in image_files:
        print(f"\n── {qf.name} ──")
        dets = process_image(qf.name, str(qf), class_supports, features_npz)
        global_annotations.extend(dets)

    _save_detections(global_annotations, timestamp, output_path)
    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot Mask Classifier (DINOv3 cosine)")
    parser.add_argument("--qry_path",     type=str, required=True,
                        help="Query image path or directory")
    parser.add_argument("--output_path",  type=str, required=True,
                        help="Directory to save detection outputs")
    parser.add_argument("--embedding_backend", type=str, default="dinov3",
                        choices=["dinov3"],
                        help="Embedding backend (only dinov3)")
    parser.add_argument("--class_support_file_path", type=str, default=None,
                        help="DINOv3 class-support .npz path")
    parser.add_argument("--object_features_file_path", type=str, default=None,
                        help="DINOv3 proposal-feature .npz path")
    parser.add_argument("--objectness_threshold", type=float, default=0.1)
    parser.add_argument("--similarity_threshold",  type=float, default=0.3)
    parser.add_argument("--nms_iou_threshold",     type=float, default=0.5)
    parser.add_argument("--is_query_dir", action="store_true",
                        help="Treat --qry_path as a directory of images")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    DEVICE               = args.device
    OBJECTNESS_THRESHOLD = args.objectness_threshold
    SIMILARITY_THRESHOLD = args.similarity_threshold
    NMS_IOU_THRESHOLD    = args.nms_iou_threshold

    main(
        qry_path=args.qry_path,
        output_path=args.output_path,
        is_query_dir=args.is_query_dir,
        class_support_file=args.class_support_file_path,
        object_features_file=args.object_features_file_path,
    )
