#!/usr/bin/env python3
"""
Render FSS visual results on UNSEEN images using the current ``fss/config.py``
defaults verbatim.

Unlike ``run_fertilizer_test.py`` (which force-overrides threshold / seed_source /
min_distance from argparse defaults and doesn't expose seed_z / box_scale /
neg_points at all), this driver builds a bare ``FSSConfig()`` — so every swept knob
comes straight from config.py — and only flips on the per-granule path. It prints
the resolved knob values so the log proves which config was used.

Support = one labelled COCO image (April1); queries = images from the non-COCO
March13 / March14 folders (different day/lighting → truly unseen, no GT). Saves an
overlay, a 4-panel debug image, and the raw mask per query.

    python scripts/run_unseen_visuals.py --out-dir fss/results
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fertilizer_test as R   # reuse COCO/crop helpers


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w <= size and h <= size:
        return img
    return img.crop(((w - size) // 2, (h - size) // 2,
                     (w + size) // 2, (h + size) // 2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=R.COCO_DEFAULT)
    ap.add_argument("--img-root", default=R.IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default="fss/results")
    ap.add_argument("--support-id", type=int, default=0)
    ap.add_argument("--folders", nargs="+", default=["March13", "March14"])
    ap.add_argument("--n-per-folder", type=int, default=3)
    ap.add_argument("--start-index", type=int, default=0,
                    help="Skip the first N images per folder (to render a fresh "
                         "batch that doesn't overlap a previous run).")
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--dinov3", default="base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--router", action="store_true",
                    help="Use the per-region router (dense→box, scattered→per-granule) "
                         "instead of per-granule-everywhere.")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    from fss import FewShotSegmenter
    from fss.config import FSSConfig
    from fss.viz import save_debug_overlay, save_mask, overlay_mask

    # Bare config → all swept knobs come from config.py untouched.
    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.dtype = args.dtype
    if args.router:
        cfg.prompts.router = True                   # adaptive dense/scattered routing
    else:
        cfg.prompts.per_granule_prompts = True      # per-granule everywhere
    seg = FewShotSegmenter(config=cfg)

    p = cfg.prompts
    path = "router" if args.router else "per_granule"
    print(f"[unseen] device={seg.device} amp={seg.amp_dtype} path={path}")
    print(f"[unseen] CONFIG FROM config.py → threshold={p.threshold} "
          f"seed_z={p.granule_seed_z} seed_source={p.seed_source} "
          f"min_distance={p.granule_min_distance} box_scale={p.granule_box_scale} "
          f"neg_points={p.granule_neg_points} neg_threshold={p.neg_threshold} "
          f"router_area_frac={p.router_min_dense_area_frac} "
          f"router_cov={p.router_seed_coverage_threshold}")

    # Support from one labelled COCO image.
    coco = R.load_coco(args.coco)
    ann_map = R.anns_by_image(coco)
    sup_img, sup_mask, sup_meta = R.prepare_case(
        coco, ann_map, args.img_root, args.support_id, "crop", args.crop_size)
    seg.set_support(images=[sup_img], masks=[sup_mask])
    print(f"[support] id={args.support_id} {sup_meta['file_name']} "
          f"anns={sup_meta['n_anns']}")

    coco_files = {i["file_name"] for i in coco["images"]}
    n_done = 0
    for folder in args.folders:
        subdir = os.path.join(args.img_root, folder)
        if not os.path.isdir(subdir):
            print(f"[warn] missing folder {subdir}")
            continue
        files = sorted(f for f in os.listdir(subdir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))
                       and os.path.join(folder, f) not in coco_files)
        for f in files[args.start_index:args.start_index + args.n_per_folder]:
            img = _center_crop(Image.open(os.path.join(subdir, f)).convert("RGB"),
                               args.crop_size)
            res = seg.segment(img)
            name = f"{path}_unseen_{folder}_{os.path.splitext(f)[0]}"
            Image.fromarray(overlay_mask(img, res["mask"])).save(
                os.path.join(args.out_dir, f"{name}_overlay.png"))
            save_mask(res["mask"], os.path.join(args.out_dir, f"{name}_mask.png"))
            save_debug_overlay(img, res["prior"], res["prompts"], res,
                               os.path.join(args.out_dir, f"{name}_debug.png"))
            fg = float(res["mask"].mean())
            print(f"[query] {name}: instances={len(res['masks'])} "
                  f"fg_fraction={fg:.4f} best_score={res['score']:.3f}")
            n_done += 1

    print(f"[unseen] wrote {n_done} unseen results (overlay+debug+mask) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
