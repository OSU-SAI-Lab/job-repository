"""
object_classification/object_classification_main.py

Multi-backend few-shot object classification.

Flow
----
1. Load class-support .npz files (one per backend).
2. Load proposal-feature .npz files (one per backend).
3. For each query image, run cosine-similarity detection per backend.
4. Save per-backend detection JSONs.

Dimension safety on load
------------------------
Support tensors are squeezed to 2-D (N, D).  If the npz was saved transposed
(D, N) — first dim larger than second — it is corrected automatically.
An explicit dtype=float32 cast prevents half-precision surprises.
"""

from __future__ import annotations

import os
import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_util

from object_classification_utils import run_detection_for_backend


# ──────────────────────────────────────────────────────────────────────────────
# Globals (populated from CLI args)
# ──────────────────────────────────────────────────────────────────────────────

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
QRY_PATH   = None
OUTPUT_PATH = None
IS_QUERY_DIR = False

EMBEDDING_BACKENDS          = ["owlv2"]
CLASS_SUPPORT_FILE_PATHS    = {}   # backend -> path
OBJECT_FEATURES_FILE_PATHS  = {}   # backend -> path
PROPOSALS_JSON_FILE_PATHS   = {}   # backend -> path (contains segmentation masks)

OBJECTNESS_THRESHOLD = 0.1
SIMILARITY_THRESHOLD = 0.2
NMS_IOU_THRESHOLD    = 0.5
MAX_PROPOSALS        = 100


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_masks_from_proposals_json(json_path: str) -> dict:
    """
    Load segmentation masks from a proposals JSON, keyed by image filename.
    Returns dict: image_name -> list of (H,W) bool numpy arrays (one per proposal, in order).
    """
    if not json_path or not os.path.exists(json_path):
        return {}

    with open(json_path) as f:
        data = json.load(f)

    masks_by_image = {}
    for ann in data.get("annotations", []):
        img_name = ann["image_path"]
        seg = ann.get("segmentation")
        if seg is not None:
            rle = dict(seg)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("utf-8")
            mask = mask_util.decode(rle).astype(bool)
        else:
            mask = None
        masks_by_image.setdefault(img_name, []).append(mask)

    return masks_by_image


def _safe_load_supports(npz, backend: str) -> dict:
    """
    Load class supports from an npz, enforce (N, D) shape and float32.
    Auto-corrects transposed (D, N) saves.
    """
    supports = {}
    for k in npz.files:
        t = torch.from_numpy(npz[k]).float()

        # Collapse any extra singleton dims → 2-D
        t = t.squeeze()
        if t.dim() == 1:
            t = t.unsqueeze(0)   # (1, D)

        # Detect and fix transposed (D, N) — first dim is much larger
        if t.dim() == 2 and t.shape[0] > t.shape[1] * 4:
            print(f"  [{backend}] '{k}': detected transposed shape {tuple(t.shape)}, transposing.")
            t = t.T

        print(f"  [{backend}] '{k}': {tuple(t.shape)}")
        supports[k] = t
    return supports


def _build_object_features(npz, query_file_name: str, backend: str, masks_by_image: dict = None) -> dict:
    """Extract feature tensors (and optional masks) for a single query image from the npz."""
    if npz is None:
        return {"features": None, "boxes": None, "scores": None, "masks": None}
    try:
        features = torch.from_numpy(npz[f"{query_file_name}_features"]).float()
        boxes    = torch.from_numpy(npz[f"{query_file_name}_boxes"]).float()
        scores   = torch.from_numpy(npz[f"{query_file_name}_scores"]).float()
        print(f"  [{backend}] {query_file_name}: proposals={features.shape}")
        masks = masks_by_image.get(query_file_name) if masks_by_image else None
        return {"features": features, "boxes": boxes, "scores": scores, "masks": masks}
    except KeyError as e:
        print(f"  [{backend}] Key not found for '{query_file_name}': {e}")
        return {"features": None, "boxes": None, "scores": None, "masks": None}


def _save_detections(annotations: list, timestamp: str, backend: str):
    # Encode any raw numpy masks to COCO RLE before JSON serialisation
    serializable = []
    for ann in annotations:
        d = dict(ann)
        seg = d.get("segmentation")
        if seg is not None and isinstance(seg, np.ndarray):
            rle = mask_util.encode(np.asfortranarray(seg.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            d["segmentation"] = rle
        serializable.append(d)

    det_dir = Path(OUTPUT_PATH) / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    fname = f"detections_{backend}_{SIMILARITY_THRESHOLD}_{OBJECTNESS_THRESHOLD}_{timestamp}.json"
    path  = det_dir / fname
    with open(path, "w") as f:
        json.dump({"annotations": serializable}, f)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def process_image(
    query_file_name: str,
    image_name: str,
    all_class_supports: dict,
    all_features_npz: dict,
    all_masks_by_image: dict = None,
) -> dict:
    """Run detection for one image across all backends."""
    per_backend = {}

    for backend in EMBEDDING_BACKENDS:
        class_supports = all_class_supports.get(backend, {})
        if not class_supports:
            print(f"  [{backend}] No class supports — skipping.")
            per_backend[backend] = []
            continue

        masks_by_image = (all_masks_by_image or {}).get(backend)
        object_features = _build_object_features(
            all_features_npz.get(backend), query_file_name, backend,
            masks_by_image=masks_by_image,
        )
        if object_features["features"] is None:
            per_backend[backend] = []
            continue

        print(f"  [{backend}] Running detection...")
        dets = run_detection_for_backend(
            backend=backend,
            class_supports=class_supports,
            object_features=object_features,
            device=DEVICE,
            objectness_threshold=OBJECTNESS_THRESHOLD,
            similarity_threshold=SIMILARITY_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
            max_proposals=MAX_PROPOSALS,
        )
        for d in dets:
            d["image_path"] = image_name
        print(f"  [{backend}] {len(dets)} detection(s).")
        per_backend[backend] = dets

    return per_backend


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    global_annotations = {b: [] for b in EMBEDDING_BACKENDS}

    # ── Load class supports ──────────────────────────────────────────────────
    all_class_supports = {}
    for backend, path in CLASS_SUPPORT_FILE_PATHS.items():
        if path and os.path.exists(path):
            print(f"[{backend}] Loading class supports: {path}")
            npz = np.load(path)
            all_class_supports[backend] = _safe_load_supports(npz, backend)
        else:
            print(f"[{backend}] Class support file missing: {path}")
            all_class_supports[backend] = {}

    # ── Load proposal feature npz files (keep open) ─────────────────────────
    all_features_npz = {}
    for backend, path in OBJECT_FEATURES_FILE_PATHS.items():
        if path and os.path.exists(path):
            print(f"[{backend}] Loading proposal features: {path}")
            all_features_npz[backend] = np.load(path)
        else:
            print(f"[{backend}] Proposal features file missing: {path}")
            all_features_npz[backend] = None

    # ── Load segmentation masks from proposals JSON (optional) ───────────────
    all_masks_by_image = {}
    for backend, path in PROPOSALS_JSON_FILE_PATHS.items():
        if path and os.path.exists(path):
            print(f"[{backend}] Loading proposal masks: {path}")
            all_masks_by_image[backend] = _load_masks_from_proposals_json(path)
        else:
            all_masks_by_image[backend] = {}

    # ── Query images ─────────────────────────────────────────────────────────
    if IS_QUERY_DIR:
        image_files = [
            p for p in Path(QRY_PATH).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    else:
        image_files = [Path(QRY_PATH)]

    for qf in image_files:
        query_file_name = qf.name
        image_name      = str(qf)
        print(f"\n── {query_file_name} ──")

        per_backend = process_image(
            query_file_name=query_file_name,
            image_name=image_name,
            all_class_supports=all_class_supports,
            all_features_npz=all_features_npz,
            all_masks_by_image=all_masks_by_image,
        )
        for backend, dets in per_backend.items():
            global_annotations[backend].extend(dets)

    # ── Save ─────────────────────────────────────────────────────────────────
    for backend, annotations in global_annotations.items():
        _save_detections(annotations, timestamp, backend)

    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-backend Few-Shot Object Classifier")
    parser.add_argument("--qry_path",     type=str, required=True,
                        help="Query image path or directory")
    parser.add_argument("--output_path",  type=str, required=True,
                        help="Directory to save detection outputs")
    parser.add_argument("--embedding_backends", type=str, nargs="+",
                        default=["owlv2"], choices=["owlv2", "bioclip", "dinov3"],
                        help="Backends to run (must match the backends used when generating "
                             "supports and proposals)")
    parser.add_argument("--class_support_file_paths", type=str, nargs="+", default=[],
                        help="Per-backend class-support .npz paths (aligned with "
                             "--embedding_backends)")
    parser.add_argument("--object_features_file_paths", type=str, nargs="+", default=[],
                        help="Per-backend proposal-feature .npz paths (aligned with "
                             "--embedding_backends)")
    parser.add_argument("--proposals_json_paths", type=str, nargs="+", default=[],
                        help="Per-backend proposals JSON paths containing segmentation masks "
                             "(aligned with --embedding_backends; optional)")
    parser.add_argument("--objectness_threshold", type=float, default=0.1)
    parser.add_argument("--similarity_threshold",  type=float, default=0.2)
    parser.add_argument("--nms_iou_threshold",     type=float, default=0.5)
    parser.add_argument("--max_proposals", type=int, default=100,
                        help="Max proposals per image kept (by objectness) before "
                             "classification. Raise for dense scenes like residue "
                             "where one image has hundreds of true pieces (default: 100).")
    parser.add_argument("--is_query_dir", action="store_true",
                        help="Treat --qry_path as a directory of images")
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    if args.class_support_file_paths and \
            len(args.class_support_file_paths) != len(args.embedding_backends):
        parser.error("--class_support_file_paths must have the same count as "
                     "--embedding_backends")
    if args.object_features_file_paths and \
            len(args.object_features_file_paths) != len(args.embedding_backends):
        parser.error("--object_features_file_paths must have the same count as "
                     "--embedding_backends")

    # Populate globals
    QRY_PATH             = args.qry_path
    OUTPUT_PATH          = args.output_path
    DEVICE               = args.device
    IS_QUERY_DIR         = args.is_query_dir
    EMBEDDING_BACKENDS   = args.embedding_backends
    OBJECTNESS_THRESHOLD = args.objectness_threshold
    SIMILARITY_THRESHOLD = args.similarity_threshold
    NMS_IOU_THRESHOLD    = args.nms_iou_threshold
    MAX_PROPOSALS        = args.max_proposals

    cs = args.class_support_file_paths
    of = args.object_features_file_paths
    pj = args.proposals_json_paths
    CLASS_SUPPORT_FILE_PATHS   = dict(zip(EMBEDDING_BACKENDS, cs)) if cs else \
                                 {b: None for b in EMBEDDING_BACKENDS}
    OBJECT_FEATURES_FILE_PATHS = dict(zip(EMBEDDING_BACKENDS, of)) if of else \
                                 {b: None for b in EMBEDDING_BACKENDS}
    PROPOSALS_JSON_FILE_PATHS  = dict(zip(EMBEDDING_BACKENDS, pj)) if pj else \
                                 {b: None for b in EMBEDDING_BACKENDS}

    main()
