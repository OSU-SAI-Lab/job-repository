#!/usr/bin/env python3
"""
Test sam3_sahi_fss (SAHI SAM 3 "visual" proposals + DINOv3 prototype match).

Stage 1  support image → SAHI SAM 3 proposals → boxes labelled by the GT annotation
Stage 2  DINOv3 crop-embeds the positive (and background) boxes → prototype set
Stage 3  every query image → same SAHI proposals → DINOv3 → cosine vs prototype set

The cosine keep-threshold is swept OFFLINE in a single pass (metrics are accumulated
per threshold as each query is processed, so running the whole dataset does not hold
every query's proposals in memory).

    python sam3_sahi_fss/run_fertilizer.py --out-dir sam3_sahi_fss/results          # all images
    python sam3_sahi_fss/run_fertilizer.py --query-ids 15 30 45 --slice-size 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
from PIL import Image

# Make the sibling packages importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam3_dino_fss.run_fertilizer import (   # noqa: E402  (path set above)
    COCO_DEFAULT, IMG_ROOT_DEFAULT, anns_by_image, load_coco, overlay, prepare_case, prf,
)

Image.MAX_IMAGE_PIXELS = None

THRESHOLDS = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=COCO_DEFAULT)
    ap.add_argument("--img-root", default=IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--support-id", type=int, default=0)
    ap.add_argument("--query-ids", type=int, nargs="+", default=None,
                    help="default: every annotated COCO image except the support")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = no cap")
    ap.add_argument("--crop-size", type=int, default=1024)
    # models
    ap.add_argument("--dinov3", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--sam3", default="facebook/sam3")
    ap.add_argument("--dtype", default="bfloat16")
    # SAHI proposal stage
    ap.add_argument("--proposal-text", default="visual")
    ap.add_argument("--slice-size", type=int, default=256)
    ap.add_argument("--overlap-ratio", type=float, default=0.2)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--max-area-frac", type=float, default=0.25)
    # prototype / match stage
    ap.add_argument("--pos-coverage", type=float, default=0.6)
    ap.add_argument("--neg-coverage", type=float, default=0.05)
    ap.add_argument("--match-mode", default="prototype", choices=["prototype", "knn"])
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--subtract-negatives", action="store_true",
                    help="score = cosine(positive) - max cosine(background support boxes)")
    ap.add_argument("--overlay-threshold", type=float, default=0.5)
    ap.add_argument("--n-overlays", type=int, default=5)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    from sam3_sahi_fss import Sam3SahiDinoFSS
    from sam3_sahi_fss.slicer import paint_union

    coco = load_coco(args.coco)
    ann_map = anns_by_image(coco)

    query_ids: List[int] = args.query_ids if args.query_ids is not None else [
        i["id"] for i in coco["images"] if ann_map.get(i["id"]) and i["id"] != args.support_id
    ]
    if args.max_queries:
        query_ids = query_ids[:args.max_queries]

    seg = Sam3SahiDinoFSS(
        dinov3=args.dinov3, sam3=args.sam3, dtype=args.dtype,
        proposal_text=args.proposal_text, slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio, nms_iou=args.nms_iou,
        score_threshold=args.score_threshold, min_area=args.min_area,
        max_area_frac=args.max_area_frac, cosine_threshold=args.overlay_threshold,
        match_mode=args.match_mode, knn_k=args.knn_k,
        subtract_negatives=args.subtract_negatives,
        pos_coverage=args.pos_coverage, neg_coverage=args.neg_coverage, verbose=True)
    print(f"[run] device={seg.device} text='{args.proposal_text}' "
          f"slice={args.slice_size} overlap={args.overlap_ratio} "
          f"match={args.match_mode} queries={len(query_ids)}", flush=True)

    # ── stage 1+2: prototype set from the support image ──
    sup_img, sup_mask, sup_name = prepare_case(coco, ann_map, args.img_root,
                                               args.support_id, args.crop_size)
    seg.set_support([sup_img], [sup_mask])
    print(f"[support] id={args.support_id} {sup_name} pos={seg.protos.n_pos} "
          f"neg={seg.protos.n_neg}", flush=True)

    # ── stage 3: sweep the whole query set, accumulating metrics per threshold ──
    acc = {t: {"iou": [], "precision": [], "recall": [], "keep": []} for t in THRESHOLDS}
    ov_t = min(THRESHOLDS, key=lambda t: abs(t - args.overlay_threshold))
    per_query = {}
    for n, qid in enumerate(query_ids):
        img, gt, fname = prepare_case(coco, ann_map, args.img_root, qid, args.crop_size)
        res = seg.segment(img)
        props, sims = res["proposals"], res["sims"]
        H, W = gt.shape
        for t in THRESHOLDS:
            kept = [p for p, s in zip(props, sims) if s >= t]
            m = prf(paint_union(kept, H, W), gt)
            acc[t]["iou"].append(m["iou"])
            acc[t]["precision"].append(m["precision"])
            acc[t]["recall"].append(m["recall"])
            acc[t]["keep"].append(len(kept) / max(1, len(props)))
        per_query[qid] = {"file": fname, "n_proposals": len(props),
                          "sim_min": round(min(sims), 4) if sims else None,
                          "sim_med": round(float(np.median(sims)), 4) if sims else None,
                          "sim_max": round(max(sims), 4) if sims else None,
                          "iou_at_overlay_thr": round(acc[ov_t]["iou"][-1], 4)}
        print(f"[q{qid}] ({n + 1}/{len(query_ids)}) proposals={len(props)} "
              f"sim med={per_query[qid]['sim_med']} "
              f"iou@{ov_t}={per_query[qid]['iou_at_overlay_thr']}", flush=True)

        if n < args.n_overlays:
            overlay(img, paint_union(props, H, W)).save(
                os.path.join(args.out_dir, f"q{qid}_all_proposals.png"))
            matched = [p for p, s in zip(props, sims) if s >= args.overlay_threshold]
            overlay(img, paint_union(matched, H, W)).save(
                os.path.join(args.out_dir, f"q{qid}_matched_overlay.png"))

    rows = [{"threshold": t,
             "mIoU": float(np.mean(acc[t]["iou"])),
             "precision": float(np.mean(acc[t]["precision"])),
             "recall": float(np.mean(acc[t]["recall"])),
             "keep_frac": float(np.mean(acc[t]["keep"]))} for t in THRESHOLDS]
    best = max(rows, key=lambda r: r["mIoU"])

    print("\n── SAHI SAM3 + DINOv3 prototype match: threshold sweep ──")
    print("cos_thr".ljust(10) + "mIoU".rjust(8) + "prec".rjust(8) + "recall".rjust(8)
          + "keep%".rjust(8))
    print("-" * 42)
    for r in rows:
        mark = "  <- best" if r is best else ""
        print(f"{r['threshold']:.2f}".ljust(10) + f"{r['mIoU']:.3f}".rjust(8)
              + f"{r['precision']:.3f}".rjust(8) + f"{r['recall']:.3f}".rjust(8)
              + f"{r['keep_frac']:.2f}".rjust(8) + mark)
    print(f"\nBEST: cos_thr={best['threshold']:.2f} mIoU={best['mIoU']:.4f} "
          f"prec={best['precision']:.4f} recall={best['recall']:.4f}")

    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump({"config": vars(args), "rows": rows, "best": best,
                   "support": {"id": args.support_id, "file": sup_name,
                               "n_pos": seg.protos.n_pos, "n_neg": seg.protos.n_neg},
                   "per_query": per_query}, fh, indent=2)
    print(f"\n[run] wrote results + overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
