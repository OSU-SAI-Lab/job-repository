"""
Visualize segmentation results: overlay predicted + GT masks on each query image,
color-coded by the evaluation outcome so the picture matches the metrics.

  green   = predicted mask that matched a GT (True Positive)
  red     = predicted mask that matched nothing (False Positive)
  yellow  = GT mask that no prediction found, drawn as an outline (False Negative)

Reuses the exact loading + matching logic from smart-labeller/evaluate_annotations.py,
so what you see is what was scored.

Usage:
  python visualize_results.py \
    --gt_file   $OUT/data/gt.json \
    --generated_file $DET \
    --iou_threshold 0.5 \
    --score_threshold 0.0 \
    --output_dir $OUT/viz \
    --max_side 2000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Reuse the evaluator's loaders + matcher (single source of truth).
SL = Path(__file__).resolve().parents[0].parent / "smart-labeller"
sys.path.insert(0, str(SL))
import evaluate_annotations as ev

Image.MAX_IMAGE_PIXELS = None

GREEN  = np.array([0, 220, 0], dtype=np.float32)      # TP prediction
RED    = np.array([230, 0, 0], dtype=np.float32)      # FP prediction
YELLOW = np.array([255, 215, 0], dtype=np.float32)    # FN ground-truth outline


def _outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    er = ndimage.binary_erosion(mask, iterations=width)
    return mask & ~er


CYAN = np.array([0, 200, 255], dtype=np.float32)   # prediction (no-GT mode)


def _score_color(s, lo, hi):
    """Low score -> orange, high score -> green (confidence heatmap)."""
    t = 0.0 if hi <= lo else (s - lo) / (hi - lo)
    return (1 - t) * np.array([255, 140, 0], np.float32) + t * np.array([0, 220, 0], np.float32)


def render(image_path, gt_items, gen_items, iou_threshold, alpha=0.45, max_side=2000):
    img = Image.open(image_path).convert("RGB")
    base = np.asarray(img, dtype=np.float32)
    overlay = base.copy()

    if gt_items is None:
        # No-GT (unseen) mode: draw every prediction, shaded by confidence.
        scores = [g["score"] for g in gen_items] or [0.0]
        lo, hi = min(scores), max(scores)
        for g in gen_items:
            m = g["mask"]
            overlay[m] = (1 - alpha) * overlay[m] + alpha * _score_color(g["score"], lo, hi)
        stats = (len(gen_items), None, None)
    else:
        matched, unmatched_gt, unmatched_gen = ev.match_masks(
            gt_items, gen_items, iou_threshold=iou_threshold
        )
        matched_gen = {gen_idx for _, gen_idx, _ in matched}
        for i, g in enumerate(gen_items):
            color = GREEN if i in matched_gen else RED
            overlay[g["mask"]] = (1 - alpha) * overlay[g["mask"]] + alpha * color
        for gi in unmatched_gt:
            overlay[_outline(gt_items[gi]["mask"])] = YELLOW
        stats = (len(matched), len(unmatched_gen), len(unmatched_gt))

    out = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    w, h = out.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        out = out.resize((int(w * s), int(h * s)))
    return out, *stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_file", default=None,
                   help="GT masks JSON. Omit for unseen images (predictions-only mode).")
    p.add_argument("--generated_file", required=True)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--score_threshold", type=float, default=0.0,
                   help="Drop predictions below this score before drawing")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_side", type=int, default=2000)
    args = p.parse_args()

    gt  = ev.load_annotations(args.gt_file) if args.gt_file else None
    gen = ev.load_annotations(args.generated_file)
    os.makedirs(args.output_dir, exist_ok=True)

    images = sorted(set(gen) | (set(gt) if gt else set()))
    for image_path in images:
        gt_items  = gt.get(image_path, []) if gt else None
        gen_items = [g for g in gen.get(image_path, []) if g["score"] >= args.score_threshold]
        if not os.path.exists(image_path):
            print(f"  [skip] image not found: {image_path}")
            continue
        out, a, b, c = render(
            image_path, gt_items, gen_items, args.iou_threshold, max_side=args.max_side
        )
        name = Path(image_path).stem + "_overlay.png"
        dst = os.path.join(args.output_dir, name)
        out.save(dst)
        if gt is None:
            print(f"  {Path(image_path).name}: predictions={a} → {dst}")
        else:
            print(f"  {Path(image_path).name}: TP={a} FP={b} FN={c} → {dst}")

    print(f"\nOverlays written to: {args.output_dir}")
    if gt is None:
        print("Legend: mask fill shaded by confidence (orange=low → green=high).")
    else:
        print("Legend: green=TP prediction, red=FP prediction, yellow outline=missed GT (FN)")


if __name__ == "__main__":
    main()
