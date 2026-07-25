"""
evaluate_annotations.py - Evaluate instance-segmentation masks against ground truth.

Computes segmentation metrics by:
1. Loading ground-truth masks
2. Loading generated masks (COCO RLE)
3. Filtering generated masks by similarity/score threshold
4. Matching masks greedily (score-descending) using mask-IoU
5. Computing precision, recall, F1, and mIoU (mean IoU over matched pairs)

Ground-truth mask formats accepted per annotation (in priority order):
    - "segmentation": COCO RLE dict {"size":[H,W], "counts":...}
    - "segmentation": COCO polygon [[x1,y1,x2,y2,...], ...]  (needs "height"/"width"
      on the annotation, or --img_height/--img_width)
    - "bounding_box": [x1,y1,x2,y2]  → rasterized to a rectangle ONLY if
      --gt_from_box is passed (coarse fallback; not true mask quality)

Usage:
    python evaluate_annotations.py \
      --gt_file path/to/ground_truth.json \
      --generated_file path/to/detections.json \
      --similarity_threshold 0.2 \
      --iou_threshold 0.5 \
      --output_path ./results
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from pycocotools import mask as coco_mask


# ──────────────────────────────────────────────────────────────────────────────
# Loading / rasterizing masks
# ──────────────────────────────────────────────────────────────────────────────

def _rle_to_bool(rle: dict) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    r = {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}
    return coco_mask.decode(r).astype(bool)


def _ann_to_mask(ann: dict, gt_from_box: bool, img_hw=None) -> np.ndarray:
    """Convert one annotation into a boolean mask, or return None if not possible."""
    seg = ann.get("segmentation")
    if isinstance(seg, dict):                       # RLE
        return _rle_to_bool(seg)
    if isinstance(seg, list) and seg:               # polygon(s)
        h = ann.get("height") or (img_hw[0] if img_hw else None)
        w = ann.get("width") or (img_hw[1] if img_hw else None)
        if h is None or w is None:
            raise ValueError("Polygon GT needs image height/width "
                             "(annotation 'height'/'width' or --img_height/--img_width).")
        rles = coco_mask.frPyObjects(seg, int(h), int(w))
        rle = coco_mask.merge(rles)
        return coco_mask.decode(rle).astype(bool)
    if gt_from_box and ann.get("bounding_box") is not None and img_hw is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in ann["bounding_box"]]
        m = np.zeros(img_hw, dtype=bool)
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        return m
    return None


def load_annotations(file_path, gt_from_box=False, img_hw=None):
    """Load annotations grouped by image_path, converting each to a boolean mask.

    Returns dict[image_path -> list of {"mask", "score", "class"}].
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    by_image = defaultdict(list)
    skipped = 0
    for ann in data.get("annotations", []):
        mask = _ann_to_mask(ann, gt_from_box, img_hw)
        if mask is None:
            skipped += 1
            continue
        by_image[ann["image_path"]].append({
            "mask":  mask,
            "score": float(ann.get("score", 1.0)),
            "class": ann.get("class", "object"),
        })
    if skipped:
        print(f"  [WARN] {skipped} annotation(s) had no usable mask and were skipped.")
    return by_image


# ──────────────────────────────────────────────────────────────────────────────
# Matching
# ──────────────────────────────────────────────────────────────────────────────

def _bbox(mask: np.ndarray):
    """Tight (y0, y1, x0, x1) extent of a boolean mask, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _boxes_disjoint(a, b) -> bool:
    return a is None or b is None or a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def match_masks(gt_items, gen_items, iou_threshold=0.5):
    """Greedy, score-descending mask matching.

    Each generated mask (highest score first) claims the highest-IoU unclaimed GT
    mask above ``iou_threshold``.  Class-agnostic, mirroring the original box
    evaluator.

    A cheap bounding-box overlap test skips the expensive full-array IoU for the
    (vast majority of) mask pairs that cannot possibly intersect.

    Returns matched_pairs [(gt_idx, gen_idx, iou)], unmatched_gt, unmatched_gen.
    """
    order = sorted(range(len(gen_items)), key=lambda i: gen_items[i]["score"], reverse=True)

    gt_boxes  = [_bbox(g["mask"]) for g in gt_items]
    gen_boxes = [_bbox(g["mask"]) for g in gen_items]

    matched_pairs = []
    matched_gt = set()
    matched_gen = set()

    for gen_idx in order:
        best_iou, best_gt = 0.0, -1
        gbox = gen_boxes[gen_idx]
        for gt_idx, gt in enumerate(gt_items):
            if gt_idx in matched_gt:
                continue
            if _boxes_disjoint(gt_boxes[gt_idx], gbox):
                continue
            iou = _mask_iou(gt["mask"], gen_items[gen_idx]["mask"])
            if iou > best_iou:
                best_iou, best_gt = iou, gt_idx
        if best_iou >= iou_threshold and best_gt >= 0:
            matched_pairs.append((best_gt, gen_idx, best_iou))
            matched_gt.add(best_gt)
            matched_gen.add(gen_idx)

    unmatched_gt  = [i for i in range(len(gt_items)) if i not in matched_gt]
    unmatched_gen = [i for i in range(len(gen_items)) if i not in matched_gen]
    return matched_pairs, unmatched_gt, unmatched_gen


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(gt_annotations, gen_annotations, similarity_threshold=0.0, iou_threshold=0.5):
    total_tp = total_fp = total_fn = 0
    matched_ious = []
    per_class_iou = defaultdict(list)
    per_image_results = {}

    all_images = set(gt_annotations) | set(gen_annotations)

    for image_path in sorted(all_images):
        gt_items  = gt_annotations.get(image_path, [])
        gen_items = gen_annotations.get(image_path, [])

        if similarity_threshold > 0:
            gen_items = [g for g in gen_items if g["score"] >= similarity_threshold]

        matched, unmatched_gt, unmatched_gen = match_masks(
            gt_items, gen_items, iou_threshold=iou_threshold
        )

        num_tp, num_fp, num_fn = len(matched), len(unmatched_gen), len(unmatched_gt)
        total_tp += num_tp
        total_fp += num_fp
        total_fn += num_fn

        for gt_idx, gen_idx, iou in matched:
            matched_ious.append(iou)
            per_class_iou[gt_items[gt_idx]["class"]].append(iou)

        per_image_results[image_path] = {
            "num_gt":        len(gt_items),
            "num_generated": len(gen_items),
            "num_tp":        num_tp,
            "num_fp":        num_fp,
            "num_fn":        num_fn,
            "mean_iou":      float(np.mean([m[2] for m in matched])) if matched else 0.0,
            "precision":     num_tp / (num_tp + num_fp) if (num_tp + num_fp) else 0.0,
            "recall":        num_tp / (num_tp + num_fn) if (num_tp + num_fn) else 0.0,
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mIoU      = float(np.mean(matched_ious)) if matched_ious else 0.0
    per_class_mIoU = {c: float(np.mean(v)) for c, v in per_class_iou.items()}

    return {
        "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mIoU": mIoU, "per_class_mIoU": per_class_mIoU,
        "per_image": per_image_results,
    }


def print_results(results, verbose=False):
    print("\n" + "=" * 70)
    print("INSTANCE SEGMENTATION EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nOverall Metrics (class-agnostic matching):")
    print(f"  True Positives (TP):  {results['total_tp']}")
    print(f"  False Positives (FP): {results['total_fp']}")
    print(f"  False Negatives (FN): {results['total_fn']}")
    print(f"\n  Precision: {results['precision']:.4f} (TP / (TP + FP))")
    print(f"  Recall:    {results['recall']:.4f} (TP / (TP + FN))")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  mIoU:      {results['mIoU']:.4f} (mean mask-IoU over matched pairs)")

    if results["per_class_mIoU"]:
        print(f"\n  Per-class mIoU:")
        for c, v in sorted(results["per_class_mIoU"].items()):
            print(f"    {c:<20} {v:.4f}")

    if verbose:
        print(f"\nPer-Image Breakdown:")
        print("-" * 70)
        for image_path, r in results["per_image"].items():
            print(f"\n  {Path(image_path).name}:")
            print(f"    GT masks:       {r['num_gt']}")
            print(f"    Generated:      {r['num_generated']}")
            print(f"    Matched (TP):   {r['num_tp']}")
            print(f"    False Pos (FP): {r['num_fp']}")
            print(f"    False Neg (FN): {r['num_fn']}")
            print(f"    Mean IoU:       {r['mean_iou']:.4f}")
            print(f"    Precision:      {r['precision']:.4f}")
            print(f"    Recall:         {r['recall']:.4f}")
    print("\n" + "=" * 70)


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_tp": results["total_tp"], "total_fp": results["total_fp"],
        "total_fn": results["total_fn"], "precision": results["precision"],
        "recall": results["recall"], "f1": results["f1"],
        "mIoU": results["mIoU"], "per_class_mIoU": results["per_class_mIoU"],
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {output_path / 'summary.json'}")

    detailed = {"summary": summary, "per_image": results["per_image"]}
    with open(output_path / "detailed_results.json", "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"Detailed results saved to: {output_path / 'detailed_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate instance-segmentation masks against ground truth."
    )
    parser.add_argument("--gt_file", type=str, required=True,
                        help="Path to ground truth annotations JSON (with masks)")
    parser.add_argument("--generated_file", type=str, required=True,
                        help="Path to generated detections JSON (RLE masks)")
    parser.add_argument("--similarity_threshold", type=float, default=0.0,
                        help="Filter generated masks by score (default: 0.0)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Minimum mask-IoU to count a match (default: 0.5)")
    parser.add_argument("--gt_from_box", action="store_true",
                        help="If GT has only bounding_box, rasterize it to a rectangle "
                             "(coarse fallback; not true mask quality)")
    parser.add_argument("--img_height", type=int, default=None,
                        help="Image height for polygon/box GT lacking size fields")
    parser.add_argument("--img_width", type=int, default=None,
                        help="Image width for polygon/box GT lacking size fields")
    parser.add_argument("--output_path", type=str, default="./evaluation_results")
    parser.add_argument("--verbose", action="store_true", help="Print per-image breakdown")
    args = parser.parse_args()

    img_hw = None
    if args.img_height and args.img_width:
        img_hw = (args.img_height, args.img_width)

    print(f"Loading ground truth from: {args.gt_file}")
    gt_annotations = load_annotations(args.gt_file, gt_from_box=args.gt_from_box, img_hw=img_hw)
    print(f"  {len(gt_annotations)} images, {sum(len(v) for v in gt_annotations.values())} masks")

    print(f"Loading generated from: {args.generated_file}")
    gen_annotations = load_annotations(args.generated_file, img_hw=img_hw)
    print(f"  {len(gen_annotations)} images, {sum(len(v) for v in gen_annotations.values())} masks")

    print(f"\nEvaluating: similarity_threshold={args.similarity_threshold}, "
          f"iou_threshold={args.iou_threshold}")
    results = evaluate(
        gt_annotations, gen_annotations,
        similarity_threshold=args.similarity_threshold,
        iou_threshold=args.iou_threshold,
    )
    print_results(results, verbose=args.verbose)
    save_results(results, args.output_path)
