"""
fss/cli.py — command-line entry point.

Example:
    python -m fss \
        --support img1.png:mask1.png img2.png:mask2.png \
        --query q.png --out out.png

Concept (matcher-free) mode:
    python -m fss --query q.png --out out.png --concept-text "yellow school bus"
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

from .config import DINOV3_VARIANTS, FSSConfig


def _parse_support(pairs: List[str]) -> Tuple[List[str], List[str]]:
    """Split ``img:mask`` tokens (rsplit on ':' to tolerate paths with colons)."""
    images, masks = [], []
    for tok in pairs:
        if ":" not in tok:
            raise SystemExit(f"--support expects 'image:mask', got '{tok}'")
        img, mask = tok.rsplit(":", 1)
        images.append(img)
        masks.append(mask)
    return images, masks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fss",
        description="Training-free few-shot segmentation (DINOv3 match + SAM 3 refine).",
    )
    p.add_argument("--support", nargs="+", metavar="IMG:MASK",
                   help="Support (image, binary mask) pairs.")
    p.add_argument("--query", required=True, help="Query image path.")
    p.add_argument("--out", required=True, help="Output mask overlay image path.")

    # Models (swapping variants is config-only).
    p.add_argument("--dinov3", default="base",
                   help=f"DINOv3 variant {list(DINOV3_VARIANTS)} or a full HF repo id.")
    p.add_argument("--sam3", default="facebook/sam3", help="SAM 3 HF repo id.")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    p.add_argument("--dtype", default=None,
                   help="autocast precision on CUDA: auto (bf16) / float16 / "
                        "bfloat16 / float32. Default float32.")

    # Matcher.
    p.add_argument("--no-bg", action="store_true", help="Disable background prototype.")
    p.add_argument("--corr-4d", action="store_true",
                   help="Use dense 4D correlation instead of a single prototype.")

    # Prompts.
    p.add_argument("--threshold", type=float, default=0.5, help="Prior fg threshold.")
    p.add_argument("--pos-points", type=int, default=3, help="Positive points / instance.")
    p.add_argument("--neg-points", type=int, default=2, help="Negative points / instance.")
    p.add_argument("--multi-instance", action="store_true",
                   help="Emit one prompt set per separated blob (multi-instance).")
    p.add_argument("--mask-prompt", action="store_true",
                   help="Also pass the coarse prior as a SAM 3 mask prompt.")

    # Per-granule prompting (the regression fix; off by default).
    p.add_argument("--per-granule-prompts", action="store_true",
                   help="One prompt per detected granule seed (brightness/prior "
                        "maxima); union per-granule masks. Prior never shapes output.")
    p.add_argument("--seed-source", choices=["brightness", "prior", "both"],
                   default="both", help="Granule-seed detection source.")
    p.add_argument("--granule-min-distance", type=int, default=6,
                   help="Min pixel spacing between granule seeds (~granule radius).")
    p.add_argument("--no-granule-box", action="store_true",
                   help="Disable the tiny per-seed box (point-only granule prompts).")

    # Per-region dense-vs-scattered router (primary; off by default).
    p.add_argument("--router", action="store_true",
                   help="Route each prior component independently: dense → box "
                        "prompts, scattered → per-granule prompts; union results.")
    p.add_argument("--router-density-metric",
                   choices=["seed_coverage", "nn_distance", "both"],
                   default="seed_coverage",
                   help="Per-component clumping metric driving the dense/scattered call.")
    p.add_argument("--router-seed-coverage-threshold", type=float, default=0.7,
                   help="Component is dense when seed-coverage fraction ≥ this.")
    p.add_argument("--router-nn-distance-threshold", type=float, default=2.0,
                   help="Component is dense when median normalised inter-seed NN "
                        "distance ≤ this (granule-diameter units).")
    p.add_argument("--router-min-seeds-dense", type=int, default=2,
                   help="Components with fewer seeds default to the box path.")

    # Small-scattered-instance (component) prompts (off by default).
    p.add_argument("--component-prompts", action="store_true",
                   help="Drive one prompt set per 8-connected prior component.")
    p.add_argument("--points-per-component", type=int, default=3,
                   help="Positive points per component (component-prompts path).")
    p.add_argument("--no-box-prompts", action="store_true",
                   help="Disable bounding boxes on the component-prompts path.")
    p.add_argument("--box-erode-frac", type=float, default=0.0,
                   help="Shrink component boxes toward the core by this fraction.")
    p.add_argument("--use-mask-prompt", action="store_true",
                   help="Feed an eroded high-confidence prior core as SAM mask input.")

    # Selection + gating (off by default).
    p.add_argument("--selection", choices=["sam_score", "granule"],
                   default="sam_score",
                   help="Multimask selection: SAM confidence, or granule "
                        "(centroid-in-prior + compact, within the size prior).")
    p.add_argument("--prior-gating", action="store_true",
                   help="Reject SAM instances barely overlapping the un-dilated "
                        "prior (accept/reject only; never grows masks).")
    p.add_argument("--min-prior-overlap", type=float, default=0.0,
                   help="Reject SAM instances below this prior-overlap fraction.")
    p.add_argument("--size-gate", action="store_true",
                   help="Reject SAM masks whose area >> expected granule area.")
    p.add_argument("--granule-size-mult", type=float, default=4.0,
                   help="Size-gate multiplier on the expected granule area.")
    p.add_argument("--intersect-prior-safety", action="store_true",
                   help="Intersect the final mask with the un-dilated prior.")

    # Prior-only diagnostic mask (task 1).
    p.add_argument("--prior-only-threshold", type=float, default=0.5,
                   help="Threshold for the prior-only baseline mask.")
    p.add_argument("--edge-snap", action="store_true",
                   help="Guided-filter edge snap for the prior-only mask.")

    # Concept (matcher-free) mode.
    p.add_argument("--concept-text", default=None,
                   help="Use SAM 3 concept prompting with this text (bypasses matcher).")

    # Evaluation / diagnostics.
    p.add_argument("--eval-variants", action="store_true",
                   help="Compute the prior_only / sam3 / granule_sam3 variants.")
    p.add_argument("--alignment-check", action="store_true",
                   help="Reverse prototype-alignment consistency: re-segment the "
                        "support from the predicted query mask; report reverse-IoU "
                        "(a label-free confidence signal; never alters the mask).")
    p.add_argument("--gt", default=None,
                   help="Ground-truth mask PNG; with --eval-variants, prints the "
                        "per-variant IoU/precision/recall comparison for this query.")

    # Outputs.
    p.add_argument("--mask-out", default=None, help="Also save the raw binary mask PNG.")
    p.add_argument("--debug-out", default=None,
                   help="Save a 4-panel debug overlay (query|prior|prompts|mask).")
    return p


def _print_variants(variant_masks: dict, gt_path: str | None) -> None:
    """Print the prior_only / sam3 / granule_sam3 comparison for one query.

    With a GT mask, reports IoU/precision/recall per variant; without one, just
    the foreground pixel count of each variant mask.
    """
    import numpy as np

    if gt_path is not None:
        from PIL import Image
        from .eval import precision_recall_iou
        gt = np.array(Image.open(gt_path).convert("L")) > 127
        print("[fss] variant comparison (vs GT):")
        for name, m in variant_masks.items():
            prec, rec, iou = precision_recall_iou(m, gt)
            print(f"    {name:<18} IoU={iou:.3f}  P={prec:.3f}  R={rec:.3f}  "
                  f"bgFP={1.0 - prec:.3f}")
    else:
        print("[fss] variant masks (no GT — fg pixel counts):")
        for name, m in variant_masks.items():
            print(f"    {name:<18} fg_px={int(np.asarray(m).sum())}")


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import heavy deps only after arg parsing so --help stays fast.
    from .pipeline import FewShotSegmenter
    from .viz import save_debug_overlay, overlay_mask, save_mask
    from PIL import Image

    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.matcher.use_bg_prototype = not args.no_bg
    cfg.matcher.use_4d_correlation = args.corr_4d
    cfg.segmenter.sam3 = args.sam3
    cfg.prompts.threshold = args.threshold
    cfg.prompts.num_pos_points = args.pos_points
    cfg.prompts.num_neg_points = args.neg_points
    cfg.prompts.multi_instance = args.multi_instance
    cfg.prompts.emit_mask_prompt = args.mask_prompt
    # Per-granule prompts (the regression fix).
    cfg.prompts.per_granule_prompts = args.per_granule_prompts
    cfg.prompts.seed_source = args.seed_source
    cfg.prompts.granule_min_distance = args.granule_min_distance
    cfg.prompts.granule_box = not args.no_granule_box
    # Per-region router.
    cfg.prompts.router = args.router
    cfg.prompts.router_density_metric = args.router_density_metric
    cfg.prompts.router_seed_coverage_threshold = args.router_seed_coverage_threshold
    cfg.prompts.router_nn_distance_threshold = args.router_nn_distance_threshold
    cfg.prompts.router_min_seeds_dense = args.router_min_seeds_dense
    # Component prompts.
    cfg.prompts.component_prompts = args.component_prompts
    cfg.prompts.points_per_component = args.points_per_component
    cfg.prompts.use_box_prompts = not args.no_box_prompts
    cfg.prompts.box_erode_frac = args.box_erode_frac
    cfg.prompts.use_mask_prompt = args.use_mask_prompt
    # Selection + gating.
    cfg.segmenter.selection_criterion = args.selection
    cfg.segmenter.prior_gating = args.prior_gating
    cfg.segmenter.min_prior_overlap = args.min_prior_overlap
    cfg.segmenter.size_gate = args.size_gate
    cfg.segmenter.granule_size_mult = args.granule_size_mult
    cfg.segmenter.intersect_prior_safety = args.intersect_prior_safety
    # Prior-only diagnostic mask.
    cfg.prior_mask.prior_only_threshold = args.prior_only_threshold
    cfg.prior_mask.use_edge_snap = args.edge_snap
    cfg.eval_variants = args.eval_variants or (args.gt is not None)
    cfg.alignment_check = args.alignment_check
    cfg.device = args.device
    cfg.dtype = args.dtype

    seg = FewShotSegmenter(config=cfg)

    if args.concept_text:
        result = seg.segment_concept(args.query, text=args.concept_text)
    else:
        if not args.support:
            raise SystemExit("Provide --support pairs, or use --concept-text.")
        images, masks = _parse_support(args.support)
        seg.set_support(images=images, masks=masks)
        result = seg.segment(args.query)

    query = Image.open(args.query).convert("RGB")
    Image.fromarray(overlay_mask(query, result["mask"])).save(args.out)
    print(f"[fss] wrote overlay -> {args.out} "
          f"(instances={len(result['masks'])}, best_score={result['score']:.3f})")

    if "active_flags" in result:
        flags = result["active_flags"]
        print("[fss] active downstream flags: " + ", ".join(
            f"{k}={v}" for k, v in flags.items()))

    if "alignment" in result:
        a = result["alignment"]
        print(f"[fss] alignment reverse-IoU={a['reverse_iou']:.3f} "
              f"(per-support {[round(x, 3) for x in a['per_support']]})")

    if "variant_masks" in result:
        _print_variants(result["variant_masks"], args.gt)

    if args.mask_out:
        save_mask(result["mask"], args.mask_out)
        print(f"[fss] wrote mask -> {args.mask_out}")
    if args.debug_out:
        save_debug_overlay(query, result["prior"], result["prompts"], result, args.debug_out)
        print(f"[fss] wrote debug overlay -> {args.debug_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
