"""
fss/eval.py — optional evaluation harness (stretch feature).

Computes mIoU and FB-IoU over a folder of FSS "episodes". Each episode is a JSON
file describing a support set, a query image, and the query's ground-truth mask:

    {
      "support": [{"image": "s1.png", "mask": "s1_m.png"}, ...],
      "query":   "q.png",
      "gt":      "q_gt.png",
      "class":   "optional-name-for-reporting"
    }

Paths are resolved relative to each episode JSON's directory. This is a generic
harness; point it at episodes exported from PASCAL-5i / COCO-20i.

    python -m fss.eval --episodes /path/to/episodes --dinov3 base
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Dict, List

import numpy as np


VARIANTS = ("prior_only", "sam3", "granule_sam3")


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (foreground IoU, background IoU) for two boolean masks."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    fg_inter = np.logical_and(pred, gt).sum()
    fg_union = np.logical_or(pred, gt).sum()
    bg_inter = np.logical_and(~pred, ~gt).sum()
    bg_union = np.logical_or(~pred, ~gt).sum()
    fg = fg_inter / fg_union if fg_union > 0 else 1.0
    bg = bg_inter / bg_union if bg_union > 0 else 1.0
    return float(fg), float(bg)


def precision_recall_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Return (precision, recall, IoU) of pred vs gt for two boolean masks."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if gt.sum() == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    return float(prec), float(rec), float(iou)


def run(episodes_dir: str, seg, verbose: bool = True) -> Dict[str, float]:
    """Evaluate ``seg`` (a FewShotSegmenter) over every *.json in episodes_dir.

    If ``seg.config.eval_variants`` is set, also prints a per-image side-by-side
    of the prior_only / sam3 / granule_sam3 mask variants and an aggregate, so we
    can see which path wins for this class.
    """
    from PIL import Image

    paths = sorted(glob(os.path.join(episodes_dir, "*.json")))
    if not paths:
        raise SystemExit(f"No episode JSONs found in {episodes_dir}")

    compare = bool(getattr(seg.config, "eval_variants", False))

    fg_ious: List[float] = []
    bg_ious: List[float] = []
    per_class: Dict[str, List[float]] = {}
    # variant -> list of (precision, recall, iou) per episode
    variant_prf: Dict[str, List[tuple[float, float, float]]] = {v: [] for v in VARIANTS}

    if compare and verbose:
        header = "episode".ljust(28) + "".join(v.rjust(20) for v in VARIANTS)
        print(header)
        print("-" * len(header))

    for ep_path in paths:
        base = os.path.dirname(ep_path)
        with open(ep_path) as f:
            ep = json.load(f)

        s_imgs = [os.path.join(base, s["image"]) for s in ep["support"]]
        s_masks = [os.path.join(base, s["mask"]) for s in ep["support"]]
        query = os.path.join(base, ep["query"])
        gt = np.array(Image.open(os.path.join(base, ep["gt"])).convert("L")) > 127

        seg.set_support(images=s_imgs, masks=s_masks)
        result = seg.segment(query)

        fg, bg = binary_iou(result["mask"], gt)
        fg_ious.append(fg)
        bg_ious.append(bg)
        per_class.setdefault(ep.get("class", "all"), []).append(fg)

        if compare and "variant_masks" in result:
            cells = []
            for v in VARIANTS:
                prf = precision_recall_iou(result["variant_masks"][v], gt)
                variant_prf[v].append(prf)
                # P also reads as 1-bgFP (bg-false-positive rate inside the mask).
                cells.append(f"IoU{prf[2]:.3f} P{prf[0]:.2f}R{prf[1]:.2f}".rjust(20))
            if verbose:
                print(os.path.basename(ep_path).ljust(28) + "".join(cells))
        elif verbose:
            print(f"{os.path.basename(ep_path)}: fg-IoU={fg:.3f}")

    metrics = {
        "mIoU": float(np.mean([np.mean(v) for v in per_class.values()])),
        "FB-IoU": float((np.mean(fg_ious) + np.mean(bg_ious)) / 2),
        "n_episodes": len(paths),
    }

    if compare and any(variant_prf[v] for v in VARIANTS):
        _report_variants(variant_prf, metrics, verbose)

    if verbose:
        print(f"\nmIoU={metrics['mIoU']:.4f}  FB-IoU={metrics['FB-IoU']:.4f}  "
              f"(n={metrics['n_episodes']})")
    return metrics


def _report_variants(
    variant_prf: Dict[str, List[tuple[float, float, float]]],
    metrics: Dict[str, float],
    verbose: bool,
) -> None:
    """Aggregate per-variant metrics into ``metrics`` and flag regressions.

    Reports precision and recall separately (IoU alone hides blobbiness) plus the
    background-false-positive rate inside the mask (bgFP = 1 - precision).
    """
    for v in VARIANTS:
        arr = np.array(variant_prf[v]) if variant_prf[v] else np.zeros((0, 3))
        mean = arr.mean(axis=0) if len(arr) else np.zeros(3)
        metrics[f"{v}_precision"] = float(mean[0])
        metrics[f"{v}_recall"] = float(mean[1])
        metrics[f"{v}_mIoU"] = float(mean[2])
        metrics[f"{v}_bgFP"] = float(1.0 - mean[0])

    # Acceptance check: granule_sam3 precision should beat the blobby sam3, while
    # recall must not collapse.
    g_prec = metrics.get("granule_sam3_precision", 0.0)
    s_prec = metrics.get("sam3_precision", 0.0)
    g_rec = metrics.get("granule_sam3_recall", 0.0)
    metrics["granule_precision_gain_vs_sam3"] = float(g_prec - s_prec)

    if verbose:
        print("\n── variant aggregate (mean over episodes) ──")
        for v in VARIANTS:
            print(f"  {v:<14} IoU={metrics[f'{v}_mIoU']:.4f}  "
                  f"P={metrics[f'{v}_precision']:.4f}  R={metrics[f'{v}_recall']:.4f}  "
                  f"bgFP={metrics[f'{v}_bgFP']:.4f}")
        print(f"\n  granule_sam3 precision gain vs sam3: "
              f"{metrics['granule_precision_gain_vs_sam3']:+.4f} "
              f"(recall {g_rec:.4f})")
        if g_prec <= s_prec:
            print("  ⚠ granule_sam3 did NOT improve precision over sam3 — "
                  "check seed detection / size prior.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fss.eval", description="FSS evaluation harness.")
    p.add_argument("--episodes", required=True, help="Folder of episode JSON files.")
    p.add_argument("--dinov3", default="base")
    p.add_argument("--sam3", default="facebook/sam3")
    p.add_argument("--device", default=None)
    p.add_argument("--corr-4d", action="store_true")
    # Variant comparison — on by default; this is the harness' purpose.
    p.add_argument("--no-variants", action="store_true",
                   help="Disable the prior_only / sam3 / granule_sam3 comparison.")
    p.add_argument("--prior-only-threshold", type=float, default=None,
                   help="Override the prior-only mask threshold.")
    p.add_argument("--selection", choices=["sam_score", "granule"], default=None,
                   help="Multimask candidate selection criterion for the live mask.")
    p.add_argument("--per-granule-prompts", action="store_true",
                   help="Use per-granule seed prompting for the live mask.")
    p.add_argument("--size-gate", action="store_true",
                   help="Reject SAM masks much larger than the granule size prior.")
    p.add_argument("--min-prior-overlap", type=float, default=None,
                   help="Reject SAM instances below this prior-overlap fraction.")
    args = p.parse_args(argv)

    from .config import FSSConfig
    from .pipeline import FewShotSegmenter

    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.matcher.use_4d_correlation = args.corr_4d
    cfg.segmenter.sam3 = args.sam3
    cfg.device = args.device
    cfg.eval_variants = not args.no_variants
    cfg.prompts.per_granule_prompts = args.per_granule_prompts
    cfg.segmenter.size_gate = args.size_gate
    if args.prior_only_threshold is not None:
        cfg.prior_mask.prior_only_threshold = args.prior_only_threshold
    if args.selection is not None:
        cfg.segmenter.selection_criterion = args.selection
    if args.min_prior_overlap is not None:
        cfg.segmenter.min_prior_overlap = args.min_prior_overlap

    seg = FewShotSegmenter(config=cfg)
    run(args.episodes, seg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
