#!/usr/bin/env python3
"""
Play with SAM 3 prompting STYLES and find the best one per regime (clumped piles
vs scattered granules) on the fertilizer data.

Distinct from the per-granule knob sweep (which tunes ONE style) and the router
(which picks a style per component): this evaluates each whole-image prompting
style on the clumped subset and the scattered subset SEPARATELY, so we can see
which prompt construction wins for each kind of fertilizer layout.

Styles compared (all downstream of the same DINOv3 prior; threshold etc. from
config.py so only the prompt construction differs):
  boxes             — tight box around each prior blob + peak points (box prompt)
  per_granule_box   — one (point + tiny box) per detected granule seed  [current default]
  per_granule_point — one point per seed, NO box
  component_box     — one (points + box) per 8-connected prior component
  component_point   — points only per component, NO box
  router            — adaptive: dense components→boxes, scattered→per-granule

One model load; support prototypes built once; cfg.prompts mutated between styles
(PromptGenerator reads it live). Per-(style,query) IoU is cached so overlays for
every style are written without re-running SAM.

    python scripts/sweep_prompt_styles.py --out-dir .../fss_results/prompt_styles
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fertilizer_test as R   # reuse COCO/crop/metric helpers

STYLES = ["boxes", "per_granule_box", "per_granule_point",
          "component_box", "component_point", "router"]


def apply_style(p, name: str) -> None:
    """Set cfg.prompts to a given prompting style (resets the path switches)."""
    p.per_granule_prompts = False
    p.component_prompts = False
    p.router = False
    p.multi_instance = True
    p.emit_box = True
    p.use_box_prompts = True
    p.granule_box = True
    p.use_mask_prompt = False
    if name == "boxes":
        pass                                   # blob path: tight box + peak points
    elif name == "per_granule_box":
        p.per_granule_prompts = True; p.granule_box = True
    elif name == "per_granule_point":
        p.per_granule_prompts = True; p.granule_box = False
    elif name == "component_box":
        p.component_prompts = True; p.use_box_prompts = True
    elif name == "component_point":
        p.component_prompts = True; p.use_box_prompts = False
    elif name == "router":
        p.router = True
    else:
        raise ValueError(f"unknown style {name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=R.COCO_DEFAULT)
    ap.add_argument("--img-root", default=R.IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir",
                    default="/fs/ess/PAS2699/kamath/fertilizer_data/fss_results/prompt_styles")
    ap.add_argument("--support-ids", type=int, nargs="+", default=[0])
    # Regime labels (q60 is the one clean dense pile; the rest are scattered).
    ap.add_argument("--clumped-ids", type=int, nargs="+", default=[60])
    ap.add_argument("--scattered-ids", type=int, nargs="+",
                    default=[15, 30, 45, 75, 90, 105])
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--dinov3", default="base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--styles", nargs="+", default=STYLES)
    ap.add_argument("--save-overlays", action="store_true", default=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    from fss import FewShotSegmenter
    from fss.config import FSSConfig
    from fss.viz import overlay_mask

    coco = R.load_coco(args.coco)
    ann_map = R.anns_by_image(coco)

    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.dtype = args.dtype
    seg = FewShotSegmenter(config=cfg)
    p = cfg.prompts
    print(f"[styles] device={seg.device} amp={seg.amp_dtype} "
          f"prior_threshold={p.threshold} seed_source={p.seed_source}", flush=True)

    sup_imgs, sup_masks = [], []
    for sid in args.support_ids:
        im, mk, _ = R.prepare_case(coco, ann_map, args.img_root, sid, "crop", args.crop_size)
        sup_imgs.append(im); sup_masks.append(mk)
    seg.set_support(images=sup_imgs, masks=sup_masks)

    # queries: (qid, image, gt, regime)
    regime = {qid: "clumped" for qid in args.clumped_ids}
    regime.update({qid: "scattered" for qid in args.scattered_ids})
    queries = []
    for qid in args.clumped_ids + args.scattered_ids:
        im, gt, _ = R.prepare_case(coco, ann_map, args.img_root, qid, "crop", args.crop_size)
        queries.append((qid, im, gt, regime[qid]))

    # style -> {qid: iou}
    table = {}
    for style in args.styles:
        apply_style(p, style)
        per = {}
        for qid, im, gt, reg in queries:
            res = seg.segment(im)
            v = R.iou(res["mask"], gt)
            per[qid] = float(v)
            if args.save_overlays:
                Image.fromarray(overlay_mask(im, res["mask"])).save(
                    os.path.join(args.out_dir, f"{style}_q{qid}_{reg}_overlay.png"))
        table[style] = per
        cl = np.mean([per[q] for q in args.clumped_ids]) if args.clumped_ids else float("nan")
        sc = np.mean([per[q] for q in args.scattered_ids]) if args.scattered_ids else float("nan")
        al = np.mean(list(per.values()))
        print(f"[style] {style:<18} clumped={cl:.3f}  scattered={sc:.3f}  all={al:.3f}  "
              f"per={ {q: round(v,2) for q,v in per.items()} }", flush=True)

    # Aggregate + winners.
    def regime_mean(style, ids):
        return float(np.mean([table[style][q] for q in ids])) if ids else float("nan")

    summary = {"clumped": {}, "scattered": {}, "all": {}}
    for style in args.styles:
        summary["clumped"][style] = regime_mean(style, args.clumped_ids)
        summary["scattered"][style] = regime_mean(style, args.scattered_ids)
        summary["all"][style] = float(np.mean(list(table[style].values())))

    best_clumped = max(args.styles, key=lambda s: summary["clumped"][s])
    best_scattered = max(args.styles, key=lambda s: summary["scattered"][s])
    best_all = max(args.styles, key=lambda s: summary["all"][s])

    print("\n── prompting-style comparison (mean IoU) ──")
    print("style".ljust(18) + "clumped".rjust(10) + "scattered".rjust(12) + "all".rjust(10))
    print("-" * 50)
    for style in args.styles:
        print(style.ljust(18) + f"{summary['clumped'][style]:.3f}".rjust(10)
              + f"{summary['scattered'][style]:.3f}".rjust(12)
              + f"{summary['all'][style]:.3f}".rjust(10))
    print("\nBEST per regime:")
    print(f"  clumped   → {best_clumped}  (IoU {summary['clumped'][best_clumped]:.3f})")
    print(f"  scattered → {best_scattered}  (IoU {summary['scattered'][best_scattered]:.3f})")
    print(f"  overall   → {best_all}  (IoU {summary['all'][best_all]:.3f})")
    if best_clumped != best_scattered:
        print(f"\n  ⇒ regimes prefer DIFFERENT styles ({best_clumped} vs {best_scattered}) "
              f"→ a per-region router is theoretically justified.")
    else:
        print(f"\n  ⇒ same style ({best_clumped}) wins BOTH regimes → one fixed style suffices.")

    with open(os.path.join(args.out_dir, "prompt_styles_results.json"), "w") as f:
        json.dump({"table": table, "summary": summary,
                   "best": {"clumped": best_clumped, "scattered": best_scattered,
                            "overall": best_all},
                   "clumped_ids": args.clumped_ids, "scattered_ids": args.scattered_ids},
                  f, indent=2)
    print(f"\n[styles] wrote results + overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
