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


def render(image_path, gt_items, gen_items, iou_threshold, alpha=0.45, max_side=2000):
    img = Image.open(image_path).convert("RGB")
    base = np.asarray(img, dtype=np.float32)

    matched, unmatched_gt, unmatched_gen = ev.match_masks(
        gt_items, gen_items, iou_threshold=iou_threshold
    )
    matched_gen = {gen_idx for _, gen_idx, _ in matched}

    overlay = base.copy()
    # Predicted masks: green if TP, red if FP.
    for i, g in enumerate(gen_items):
        color = GREEN if i in matched_gen else RED
        m = g["mask"]
        overlay[m] = (1 - alpha) * overlay[m] + alpha * color
    # Missed GT (FN): yellow outline.
    for gi in unmatched_gt:
        ol = _outline(gt_items[gi]["mask"])
        overlay[ol] = YELLOW

    out = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))

    # Downscale for quick viewing.
    w, h = out.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        out = out.resize((int(w * s), int(h * s)))

    return out, len(matched), len(unmatched_gen), len(unmatched_gt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_file", required=True)
    p.add_argument("--generated_file", required=True)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--score_threshold", type=float, default=0.0,
                   help="Drop predictions below this score before drawing")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_side", type=int, default=2000)
    args = p.parse_args()

    gt  = ev.load_annotations(args.gt_file)
    gen = ev.load_annotations(args.generated_file)
    os.makedirs(args.output_dir, exist_ok=True)

    images = sorted(set(gt) | set(gen))
    for image_path in images:
        gt_items  = gt.get(image_path, [])
        gen_items = [g for g in gen.get(image_path, []) if g["score"] >= args.score_threshold]
        if not os.path.exists(image_path):
            print(f"  [skip] image not found: {image_path}")
            continue
        out, tp, fp, fn = render(
            image_path, gt_items, gen_items, args.iou_threshold, max_side=args.max_side
        )
        name = Path(image_path).stem + "_overlay.png"
        dst = os.path.join(args.output_dir, name)
        out.save(dst)
        print(f"  {Path(image_path).name}: TP={tp} FP={fp} FN={fn} → {dst}")

    print(f"\nOverlays written to: {args.output_dir}")
    print("Legend: green=TP prediction, red=FP prediction, yellow outline=missed GT (FN)")


if __name__ == "__main__":
    main()
