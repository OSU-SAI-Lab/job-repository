#!/usr/bin/env python3
"""
Implement + evaluate the DINOv3 cosine-verification idea:

    DINOv3 class-support vector → DINOv3 prior map → SAM segments → keep a SAM mask
    only if its pooled DINOv3 feature has high cosine similarity to the class-support
    vector (fg − bg). Masks that drifted onto soil pool a non-class feature → dropped.

Runs the per-granule path once per GT query (so SAM runs once), records the per-mask
cosine score via matcher.mask_cosine_similarities, then sweeps the keep-threshold
OFFLINE (no SAM reruns) reporting mIoU / precision / recall and the fraction of masks
kept. Saves a results JSON + baseline-vs-verified overlays per query.

    python scripts/eval_cosine_verify.py --out-dir fss/results/Cosine_Verify_Eval
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
import run_fertilizer_test as R

THRESHOLDS = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]


def union(masks, shape):
    return np.logical_or.reduce(masks) if masks else np.zeros(shape, dtype=bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=R.COCO_DEFAULT)
    ap.add_argument("--img-root", default=R.IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir",
                    default="/users/PAS2699/naveenkamath/job-repository/"
                            "few-shot-segmentation/fss/results/Cosine_Verify_Eval")
    ap.add_argument("--support-ids", type=int, nargs="+", default=[0])
    ap.add_argument("--query-ids", type=int, nargs="+",
                    default=[15, 30, 45, 60, 75, 90, 105])
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--dinov3", default="base")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    from fss import FewShotSegmenter
    from fss.config import FSSConfig
    from fss.viz import overlay_mask, save_mask

    coco = R.load_coco(args.coco)
    ann_map = R.anns_by_image(coco)

    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.dtype = args.dtype
    cfg.prompts.per_granule_prompts = True          # tuned config from config.py
    # cosine_verify OFF here — we score masks ourselves so SAM runs once.
    seg = FewShotSegmenter(config=cfg)
    print(f"[cosverify] device={seg.device} amp={seg.amp_dtype}", flush=True)

    sup_imgs, sup_masks = [], []
    for sid in args.support_ids:
        im, mk, _ = R.prepare_case(coco, ann_map, args.img_root, sid, "crop", args.crop_size)
        sup_imgs.append(im); sup_masks.append(mk)
    seg.set_support(images=sup_imgs, masks=sup_masks)

    # Run SAM once per query, capture per-mask masks + cosine sims + GT.
    per_query = []
    for qid in args.query_ids:
        im, gt, _ = R.prepare_case(coco, ann_map, args.img_root, qid, "crop", args.crop_size)
        res = seg.segment(im)
        masks = res["masks"]
        sims = seg.matcher.mask_cosine_similarities(im, masks, seg._protos)
        per_query.append({"qid": qid, "img": im, "gt": gt, "masks": masks, "sims": sims})
        if sims:
            print(f"[q{qid}] n_masks={len(masks)} cosine fg-bg "
                  f"min={min(sims):.3f} med={float(np.median(sims)):.3f} "
                  f"max={max(sims):.3f}", flush=True)

    # Offline threshold sweep (incl. baseline = keep all).
    rows = []
    for thr in [None] + THRESHOLDS:
        ious, precs, recs, keep_fracs = [], [], [], []
        for q in per_query:
            masks, sims, gt = q["masks"], q["sims"], q["gt"]
            if thr is None:
                kept = masks
            else:
                kept = [m for m, s in zip(masks, sims) if s >= thr]
            u = union(kept, gt.shape)
            m = R.prf(u, gt)
            ious.append(m["iou"]); precs.append(m["precision"]); recs.append(m["recall"])
            keep_fracs.append(len(kept) / max(1, len(masks)))
        rows.append({
            "threshold": thr,
            "mIoU": float(np.mean(ious)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "keep_frac": float(np.mean(keep_fracs)),
        })

    baseline = rows[0]
    best = max(rows, key=lambda r: r["mIoU"])
    print("\n── cosine-verify threshold sweep (mean over 7 GT queries) ──")
    print("threshold".ljust(12) + "mIoU".rjust(8) + "prec".rjust(8)
          + "recall".rjust(8) + "keep%".rjust(8))
    print("-" * 44)
    for r in rows:
        t = "none" if r["threshold"] is None else f"{r['threshold']:+.2f}"
        mark = "  <- best" if r is best else ("  (baseline)" if r is baseline else "")
        print(t.ljust(12) + f"{r['mIoU']:.3f}".rjust(8) + f"{r['precision']:.3f}".rjust(8)
              + f"{r['recall']:.3f}".rjust(8) + f"{r['keep_frac']:.2f}".rjust(8) + mark)
    print(f"\nBASELINE (no verify): mIoU={baseline['mIoU']:.4f} prec={baseline['precision']:.4f}")
    bt = "none" if best["threshold"] is None else f"{best['threshold']:+.2f}"
    print(f"BEST: threshold={bt} mIoU={best['mIoU']:.4f} prec={best['precision']:.4f} "
          f"(Δ mIoU {best['mIoU']-baseline['mIoU']:+.4f}, "
          f"Δ prec {best['precision']-baseline['precision']:+.4f})")

    # Save baseline vs verified@best overlays per query + JSON.
    bthr = best["threshold"]
    for q in per_query:
        base_u = union(q["masks"], q["gt"].shape)
        if bthr is None:
            ver_u = base_u
        else:
            ver_u = union([m for m, s in zip(q["masks"], q["sims"]) if s >= bthr],
                          q["gt"].shape)
        Image.fromarray(overlay_mask(q["img"], base_u)).save(
            os.path.join(args.out_dir, f"q{q['qid']}_baseline_overlay.png"))
        Image.fromarray(overlay_mask(q["img"], ver_u)).save(
            os.path.join(args.out_dir, f"q{q['qid']}_verified_overlay.png"))
        save_mask(ver_u, os.path.join(args.out_dir, f"q{q['qid']}_verified_mask.png"))

    with open(os.path.join(args.out_dir, "cosine_verify_results.json"), "w") as f:
        json.dump({"rows": rows, "baseline": baseline, "best": best,
                   "per_query_sims": {q["qid"]: [round(s, 4) for s in q["sims"]]
                                      for q in per_query}}, f, indent=2)
    print(f"\n[cosverify] wrote results + overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
