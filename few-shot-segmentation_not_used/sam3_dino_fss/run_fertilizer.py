#!/usr/bin/env python3
"""
Test sam3_dino_fss (SAM3 proposals + DINOv3 cosine match) on the fertilizer data.

Support = one labelled COCO image; queries = held-out COCO images (GT IoU) plus a
couple of truly-unseen March images (qualitative). Runs SAM3 once per query, embeds
every proposal, then sweeps the cosine keep-threshold OFFLINE reporting
mIoU/precision/recall/keep%, and saves "all-proposals vs matched" overlays.

    python sam3_dino_fss/run_fertilizer.py --out-dir sam3_dino_fss/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Make the package importable when run as a script (script dir != package parent).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Image.MAX_IMAGE_PIXELS = None

COCO_DEFAULT = "/fs/ess/PAS2699/kamath/fertilizer_data/fertilizer_segmentation_coco.json"
IMG_ROOT_DEFAULT = "/fs/ess/PAS2699/agriculture/fertilizer_dataset"
THRESHOLDS = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


# ── minimal COCO helpers (self-contained) ───────────────────────────────────
def load_coco(p):
    with open(p) as f:
        return json.load(f)


def anns_by_image(coco) -> Dict[int, list]:
    out: Dict[int, list] = {}
    for a in coco["annotations"]:
        out.setdefault(a["image_id"], []).append(a)
    return out


def build_mask(w, h, anns) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        seg = a.get("segmentation") or []
        if isinstance(seg, dict):
            continue
        for poly in seg:
            if len(poly) >= 6:
                d.polygon(list(zip(poly[0::2], poly[1::2])), fill=255)
    return np.array(m) > 127


def best_crop_window(anns, w, h, size) -> Tuple[int, int, int, int]:
    centers = [(a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2) for a in anns]
    centers = np.array(centers) if centers else np.array([[w / 2, h / 2]])
    best, best_n, half = (0, 0), -1, size / 2
    for cx, cy in centers:
        x0 = int(min(max(cx - half, 0), max(w - size, 0)))
        y0 = int(min(max(cy - half, 0), max(h - size, 0)))
        n = np.sum((centers[:, 0] >= x0) & (centers[:, 0] < x0 + size) &
                   (centers[:, 1] >= y0) & (centers[:, 1] < y0 + size))
        if n > best_n:
            best_n, best = n, (x0, y0)
    x0, y0 = best
    return x0, y0, min(x0 + size, w), min(y0 + size, h)


def prepare_case(coco, ann_map, img_root, image_id, crop_size):
    info = next(i for i in coco["images"] if i["id"] == image_id)
    img = Image.open(os.path.join(img_root, info["file_name"])).convert("RGB")
    w, h = img.size
    anns = ann_map.get(image_id, [])
    mask = build_mask(w, h, anns)
    x0, y0, x1, y1 = best_crop_window(anns, w, h, crop_size)
    return img.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1], info["file_name"]


def prf(pred, gt) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if gt.sum() == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    union = tp + fp + fn
    return {"precision": float(prec), "recall": float(rec),
            "iou": float(tp / union) if union else 1.0}


def overlay(img, mask, color=(255, 0, 0), alpha=0.5):
    rgb = np.array(img.convert("RGB")).astype("float32")
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color, "float32")
    return Image.fromarray(np.clip(rgb, 0, 255).astype("uint8"))


def union(masks, shape):
    return np.logical_or.reduce(masks) if masks else np.zeros(shape, bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=COCO_DEFAULT)
    ap.add_argument("--img-root", default=IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--support-id", type=int, default=0)
    ap.add_argument("--query-ids", type=int, nargs="+", default=[15, 30, 45, 60, 75, 90, 105])
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--dinov3", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--sam3", default="facebook/sam3")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--proposal-text", default="fertilizer granules")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--unseen-subdir", default="March13")
    ap.add_argument("--n-unseen", type=int, default=2)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    from sam3_dino_fss import SamDinoFSS

    coco = load_coco(args.coco)
    ann_map = anns_by_image(coco)
    coco_files = {i["file_name"] for i in coco["images"]}

    seg = SamDinoFSS(dinov3=args.dinov3, sam3=args.sam3, dtype=args.dtype,
                     proposal_text=args.proposal_text, cosine_threshold=-1.0,
                     score_threshold=args.score_threshold)
    print(f"[run] device={seg.device} proposal_text='{args.proposal_text}'", flush=True)

    sup_img, sup_mask, sup_name = prepare_case(coco, ann_map, args.img_root,
                                               args.support_id, args.crop_size)
    seg.set_support([sup_img], [sup_mask])
    print(f"[support] id={args.support_id} {sup_name} "
          f"instances={seg._instances.shape[0]}", flush=True)

    per_query = []
    for qid in args.query_ids:
        img, gt, _ = prepare_case(coco, ann_map, args.img_root, qid, args.crop_size)
        res = seg.segment(img)
        masks = [p.mask for p in res["proposals"]]
        sims = res["sims"]
        per_query.append({"qid": qid, "img": img, "gt": gt, "masks": masks, "sims": sims})
        if sims:
            print(f"[q{qid}] proposals={len(masks)} cosine "
                  f"min={min(sims):.3f} med={float(np.median(sims)):.3f} "
                  f"max={max(sims):.3f}", flush=True)
        else:
            print(f"[q{qid}] proposals=0 (SAM found nothing for the concept)", flush=True)

    # Offline cosine-threshold sweep (mIoU/precision/recall).
    rows = []
    for thr in THRESHOLDS:
        ious, precs, recs, keep = [], [], [], []
        for q in per_query:
            kept = [m for m, s in zip(q["masks"], q["sims"]) if s >= thr]
            u = union(kept, q["gt"].shape)
            m = prf(u, q["gt"])
            ious.append(m["iou"]); precs.append(m["precision"]); recs.append(m["recall"])
            keep.append(len(kept) / max(1, len(q["masks"])))
        rows.append({"threshold": thr, "mIoU": float(np.mean(ious)),
                     "precision": float(np.mean(precs)), "recall": float(np.mean(recs)),
                     "keep_frac": float(np.mean(keep))})
    best = max(rows, key=lambda r: r["mIoU"])
    print("\n── SAM3+DINOv3 cosine-match: threshold sweep (mean over GT queries) ──")
    print("cos_thr".ljust(10) + "mIoU".rjust(8) + "prec".rjust(8) + "recall".rjust(8) + "keep%".rjust(8))
    print("-" * 42)
    for r in rows:
        mark = "  <- best" if r is best else ""
        print(f"{r['threshold']:.2f}".ljust(10) + f"{r['mIoU']:.3f}".rjust(8)
              + f"{r['precision']:.3f}".rjust(8) + f"{r['recall']:.3f}".rjust(8)
              + f"{r['keep_frac']:.2f}".rjust(8) + mark)
    print(f"\nBEST: cos_thr={best['threshold']:.2f} mIoU={best['mIoU']:.4f} "
          f"prec={best['precision']:.4f} recall={best['recall']:.4f}")

    # Save all-proposals vs matched overlays at the best threshold.
    bt = best["threshold"]
    for q in per_query:
        Image.fromarray(np.array(overlay(q["img"], union(q["masks"], q["gt"].shape))))\
            .save(os.path.join(args.out_dir, f"q{q['qid']}_all_proposals.png"))
        matched = union([m for m, s in zip(q["masks"], q["sims"]) if s >= bt], q["gt"].shape)
        overlay(q["img"], matched).save(
            os.path.join(args.out_dir, f"q{q['qid']}_matched_overlay.png"))

    # Qualitative unseen.
    subdir = os.path.join(args.img_root, args.unseen_subdir)
    if os.path.isdir(subdir) and args.n_unseen > 0:
        files = sorted(f for f in os.listdir(subdir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))
                       and os.path.join(args.unseen_subdir, f) not in coco_files)
        for f in files[:args.n_unseen]:
            im = Image.open(os.path.join(subdir, f)).convert("RGB")
            w, h = im.size; s = args.crop_size
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            seg.cosine_threshold = bt
            res = seg.segment(im)
            overlay(im, res["mask"]).save(
                os.path.join(args.out_dir, f"unseen_{os.path.splitext(f)[0]}_matched.png"))
            print(f"[unseen] {f}: proposals={res['n_proposals']} kept={len(res['kept'])}",
                  flush=True)

    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump({"rows": rows, "best": best, "proposal_text": args.proposal_text,
                   "per_query_sims": {q["qid"]: [round(s, 4) for s in q["sims"]]
                                      for q in per_query}}, fh, indent=2)
    print(f"\n[run] wrote results + overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
