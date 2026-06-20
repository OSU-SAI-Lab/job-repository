#!/usr/bin/env python3
"""
Test the FSS pipeline on the fertilizer dataset.

Builds binary foreground masks from the COCO polygon annotations, then runs the
training-free few-shot segmenter with a SUPPORT image and a disjoint QUERY image
(the model never sees the query as support). Because the granules are tiny
relative to the 4284x4284 frames, the harness can operate on CROPS (default) so
the matcher has a usable patch grid; a whole-image mode is also available.

Three query types are evaluated:
  * held-out COCO image  -> quantitative IoU vs GT (disjoint from support)
  * an extra COCO image  -> quantitative IoU vs GT
  * an "unseen" image from a non-COCO folder (March13/14) -> qualitative overlay

Outputs (overlays, masks, results.json) are written next to the COCO file by
default: /fs/ess/PAS2699/kamath/fertilizer_data/fss_results
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None  # 4284x4284 is fine; silence the DecompressionBomb guard

COCO_DEFAULT = "/fs/ess/PAS2699/kamath/fertilizer_data/fertilizer_segmentation_coco.json"
IMG_ROOT_DEFAULT = "/fs/ess/PAS2699/agriculture/fertilizer_dataset"
OUT_DEFAULT = "/fs/ess/PAS2699/kamath/fertilizer_data/fss_results"


# ──────────────────────────────────────────────────────────────────────────
# COCO helpers
# ──────────────────────────────────────────────────────────────────────────

def load_coco(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def anns_by_image(coco: dict) -> Dict[int, list]:
    out: Dict[int, list] = {}
    for a in coco["annotations"]:
        out.setdefault(a["image_id"], []).append(a)
    return out


def build_mask(w: int, h: int, anns: list) -> np.ndarray:
    """Rasterise COCO polygon annotations into a full-res binary mask."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        seg = a.get("segmentation") or []
        if isinstance(seg, dict):          # RLE — not expected here, skip
            continue
        for poly in seg:
            if len(poly) >= 6:
                pts = list(zip(poly[0::2], poly[1::2]))
                d.polygon(pts, fill=255)
    return np.array(m) > 127


def best_crop_window(anns: list, w: int, h: int, size: int) -> Tuple[int, int, int, int]:
    """Pick a `size`x`size` window containing the most annotation centroids."""
    centers = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        centers.append((x + bw / 2, y + bh / 2))
    centers = np.array(centers) if centers else np.array([[w / 2, h / 2]])

    best, best_n = (0, 0), -1
    half = size / 2
    for cx, cy in centers:
        x0 = int(min(max(cx - half, 0), max(w - size, 0)))
        y0 = int(min(max(cy - half, 0), max(h - size, 0)))
        n = np.sum((centers[:, 0] >= x0) & (centers[:, 0] < x0 + size) &
                   (centers[:, 1] >= y0) & (centers[:, 1] < y0 + size))
        if n > best_n:
            best_n, best = n, (x0, y0)
    x0, y0 = best
    return x0, y0, min(x0 + size, w), min(y0 + size, h)


def fg_fraction(mask: np.ndarray) -> float:
    return float(mask.mean())


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else (1.0 if pred.sum() == 0 else 0.0)


def prf(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Precision, recall, IoU, and bg-false-positive rate inside the mask.

    bgFP = fraction of predicted-foreground pixels that are actually background
    (= 1 - precision); the direct measure of blobbiness IoU hides.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if gt.sum() == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    union = tp + fp + fn
    return {
        "precision": float(prec),
        "recall": float(rec),
        "iou": float(tp / union) if union else 1.0,
        "bg_fp_rate": float(1.0 - prec),
    }


# ──────────────────────────────────────────────────────────────────────────

def prepare_case(coco, ann_map, img_root, image_id, mode, crop_size):
    """Return (PIL image, bool mask, meta) for a COCO image id (full or crop)."""
    info = next(i for i in coco["images"] if i["id"] == image_id)
    path = os.path.join(img_root, info["file_name"])
    img = Image.open(path).convert("RGB")
    w, h = img.size
    anns = ann_map.get(image_id, [])
    mask = build_mask(w, h, anns)

    meta = {"image_id": image_id, "file_name": info["file_name"], "n_anns": len(anns)}
    if mode == "crop":
        x0, y0, x1, y1 = best_crop_window(anns, w, h, crop_size)
        img = img.crop((x0, y0, x1, y1))
        mask = mask[y0:y1, x0:x1]
        meta["crop"] = [x0, y0, x1, y1]
    meta["fg_fraction"] = fg_fraction(mask)
    return img, mask, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=COCO_DEFAULT)
    ap.add_argument("--img-root", default=IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--mode", choices=["crop", "full"], default="crop")
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--support-ids", type=int, nargs="+", default=[0])
    ap.add_argument("--query-ids", type=int, nargs="+", default=[105, 60])
    ap.add_argument("--unseen-subdir", default="March13",
                    help="Folder of images NOT in the COCO file (truly unseen).")
    ap.add_argument("--n-unseen", type=int, default=1)
    ap.add_argument("--dinov3", default="base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--multi-instance", action="store_true")
    ap.add_argument("--corr-4d", action="store_true")

    # Downstream options (all OFF by default -> identical to last-good runs).
    ap.add_argument("--eval-variants", action="store_true",
                    help="Also save prior_only / sam3 / granule_sam3 masks + metrics.")
    # Per-granule prompting (the regression fix).
    ap.add_argument("--per-granule-prompts", action="store_true",
                    help="One prompt per detected granule seed; union per-granule masks.")
    ap.add_argument("--seed-source", choices=["brightness", "prior", "both"],
                    default="both")
    ap.add_argument("--granule-min-distance", type=int, default=6)
    ap.add_argument("--no-granule-box", action="store_true")
    # Per-region dense-vs-scattered router (primary deliverable).
    ap.add_argument("--router", action="store_true",
                    help="Route each prior component: dense → box, scattered → per-granule.")
    ap.add_argument("--router-density-metric",
                    choices=["seed_coverage", "nn_distance", "both"],
                    default="seed_coverage")
    ap.add_argument("--router-seed-coverage-threshold", type=float, default=0.7)
    ap.add_argument("--router-nn-distance-threshold", type=float, default=2.0)
    ap.add_argument("--router-min-seeds-dense", type=int, default=2)
    # Acceptance comparison: run boxes-only / per-granule-only / router on every
    # query and print the per-regime IoU/P/R/bgFP table in one job.
    ap.add_argument("--compare-paths", action="store_true",
                    help="Evaluate boxes / per_granule / router paths per query.")
    ap.add_argument("--router-thresholds", type=float, nargs="*", default=None,
                    help="If set with --compare-paths, sweep these seed-coverage "
                         "thresholds for the router (calibration).")
    # Component prompting.
    ap.add_argument("--component-prompts", action="store_true",
                    help="One prompt set per 8-connected prior component.")
    ap.add_argument("--no-box-prompts", action="store_true",
                    help="Drop boxes on the component path.")
    ap.add_argument("--box-erode-frac", type=float, default=0.0)
    ap.add_argument("--points-per-component", type=int, default=3)
    ap.add_argument("--use-mask-prompt", action="store_true",
                    help="Feed an eroded high-confidence prior core as SAM mask input.")
    # Selection + gating.
    ap.add_argument("--selection", choices=["sam_score", "granule"],
                    default="sam_score")
    ap.add_argument("--prior-gating", action="store_true")
    ap.add_argument("--min-prior-overlap", type=float, default=0.0)
    ap.add_argument("--size-gate", action="store_true")
    ap.add_argument("--granule-size-mult", type=float, default=4.0)
    ap.add_argument("--intersect-prior-safety", action="store_true")
    ap.add_argument("--prior-only-threshold", type=float, default=0.5)
    ap.add_argument("--edge-snap", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    coco = load_coco(args.coco)
    ann_map = anns_by_image(coco)
    coco_files = {i["file_name"] for i in coco["images"]}

    # Sanity: support and query must be disjoint image ids.
    overlap = set(args.support_ids) & set(args.query_ids)
    if overlap:
        raise SystemExit(f"Support/query image ids overlap (model would 'see' query): {overlap}")

    from fss import FewShotSegmenter
    from fss.config import FSSConfig
    from fss.viz import save_debug_overlay, save_mask, overlay_mask

    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.matcher.use_4d_correlation = args.corr_4d
    cfg.prompts.threshold = args.threshold
    cfg.prompts.multi_instance = args.multi_instance
    cfg.dtype = args.dtype
    # Downstream config.
    cfg.eval_variants = args.eval_variants
    cfg.prompts.per_granule_prompts = args.per_granule_prompts
    cfg.prompts.seed_source = args.seed_source
    cfg.prompts.granule_min_distance = args.granule_min_distance
    cfg.prompts.granule_box = not args.no_granule_box
    cfg.prompts.router = args.router
    cfg.prompts.router_density_metric = args.router_density_metric
    cfg.prompts.router_seed_coverage_threshold = args.router_seed_coverage_threshold
    cfg.prompts.router_nn_distance_threshold = args.router_nn_distance_threshold
    cfg.prompts.router_min_seeds_dense = args.router_min_seeds_dense
    cfg.prompts.component_prompts = args.component_prompts
    cfg.prompts.use_box_prompts = not args.no_box_prompts
    cfg.prompts.box_erode_frac = args.box_erode_frac
    cfg.prompts.points_per_component = args.points_per_component
    cfg.prompts.use_mask_prompt = args.use_mask_prompt
    cfg.segmenter.selection_criterion = args.selection
    cfg.segmenter.prior_gating = args.prior_gating
    cfg.segmenter.min_prior_overlap = args.min_prior_overlap
    cfg.segmenter.size_gate = args.size_gate
    cfg.segmenter.granule_size_mult = args.granule_size_mult
    cfg.segmenter.intersect_prior_safety = args.intersect_prior_safety
    cfg.prior_mask.prior_only_threshold = args.prior_only_threshold
    cfg.prior_mask.use_edge_snap = args.edge_snap
    seg = FewShotSegmenter(config=cfg)
    print(f"[test] device={seg.device} amp_dtype={seg.amp_dtype} mode={args.mode} "
          f"crop={args.crop_size} dinov3={args.dinov3}")

    # ---- support ----
    sup_imgs, sup_masks, sup_meta = [], [], []
    for sid in args.support_ids:
        im, mk, meta = prepare_case(coco, ann_map, args.img_root, sid, args.mode, args.crop_size)
        sup_imgs.append(im); sup_masks.append(mk); sup_meta.append(meta)
        print(f"[support] id={sid} {meta['file_name']} fg={meta['fg_fraction']:.4f} "
              f"anns={meta['n_anns']} crop={meta.get('crop')}")
    seg.set_support(images=sup_imgs, masks=sup_masks)

    results = {"config": vars(args), "support": sup_meta, "queries": []}

    def run_query(name, img, gt, meta):
        res = seg.segment(img)
        entry = dict(meta); entry["name"] = name
        entry["pred_fg_fraction"] = fg_fraction(res["mask"])
        entry["n_instances"] = len(res["masks"])
        entry["best_score"] = res["score"]
        entry["active_flags"] = res.get("active_flags")
        if gt is not None:
            entry["metrics"] = prf(res["mask"], gt)         # precision/recall/IoU/bgFP
            entry["iou"] = entry["metrics"]["iou"]
        # save artefacts
        ov = os.path.join(args.out_dir, f"{name}_overlay.png")
        Image.fromarray(overlay_mask(img, res["mask"])).save(ov)
        save_mask(res["mask"], os.path.join(args.out_dir, f"{name}_mask.png"))
        save_debug_overlay(img, res["prior"], res["prompts"], res,
                           os.path.join(args.out_dir, f"{name}_debug.png"))

        # Per-variant artefacts + metrics (prior_only / sam3 / granule_sam3).
        variant_line = ""
        if "variant_masks" in res:
            entry["variants"] = {}
            for vname, vmask in res["variant_masks"].items():
                Image.fromarray(overlay_mask(img, vmask)).save(
                    os.path.join(args.out_dir, f"{name}_{vname}_overlay.png"))
                save_mask(vmask, os.path.join(args.out_dir, f"{name}_{vname}_mask.png"))
                v = {"fg_fraction": fg_fraction(vmask)}
                if gt is not None:
                    v.update(prf(vmask, gt))
                entry["variants"][vname] = v
            if gt is not None:
                variant_line = "  " + "  ".join(
                    f"{vn}[IoU{vv['iou']:.2f} P{vv['precision']:.2f} "
                    f"R{vv['recall']:.2f}]" for vn, vv in entry["variants"].items())

        results["queries"].append(entry)
        metric_str = ""
        if gt is not None:
            m = entry["metrics"]
            metric_str = (f"IoU={m['iou']:.3f} P={m['precision']:.3f} "
                          f"R={m['recall']:.3f} bgFP={m['bg_fp_rate']:.3f}")
        else:
            metric_str = "(no GT)"
        print(f"[query] {name}: instances={entry['n_instances']} "
              f"score={entry['best_score']:.3f} " + metric_str + variant_line)

    # ── Acceptance comparison: boxes-only vs per-granule-only vs router ──────
    # Mutates cfg.prompts in place (the PromptGenerator holds the same object and
    # reads it live; the cached DINOv3 prototypes are unaffected) so all paths run
    # against one set_support. Selection/gating stay at defaults (sam_score, no
    # gate) per Task 3 — the rejected pieces are NOT wired into the router path.
    def apply_path(kind: str, overrides: dict) -> None:
        p = cfg.prompts
        p.router = p.per_granule_prompts = p.component_prompts = False
        if kind == "per_granule":
            p.per_granule_prompts = True
        elif kind == "router":
            p.router = True
        # 'boxes' leaves everything off → the original blob+box pipeline.
        for k, v in overrides.items():
            setattr(p, k, v)

    def build_specs() -> list:
        specs = [("boxes", "boxes", {}), ("per_granule", "per_granule", {})]
        if args.router_thresholds:
            for thr in args.router_thresholds:
                specs.append((f"router_cov{thr:g}", "router",
                              {"router_seed_coverage_threshold": float(thr)}))
        else:
            specs.append(("router", "router", {}))
        return specs

    def run_query_compare(name, img, gt, meta, specs):
        entry = dict(meta); entry["name"] = name; entry["paths"] = {}
        line = f"[compare] {name}:"
        for disp, kind, ov in specs:
            apply_path(kind, ov)
            res = seg.segment(img)
            pr = {
                "fg_fraction": fg_fraction(res["mask"]),
                "n_instances": len(res["masks"]),
                "best_score": res["score"],
                "route_summary": res.get("route_summary"),
            }
            if gt is not None:
                pr.update(prf(res["mask"], gt))
            entry["paths"][disp] = pr
            Image.fromarray(overlay_mask(img, res["mask"])).save(
                os.path.join(args.out_dir, f"{name}_{disp}_overlay.png"))
            save_debug_overlay(img, res["prior"], res["prompts"], res,
                               os.path.join(args.out_dir, f"{name}_{disp}_debug.png"))
            if gt is not None:
                line += (f"  {disp}[IoU{pr['iou']:.2f} P{pr['precision']:.2f} "
                         f"R{pr['recall']:.2f} bgFP{pr['bg_fp_rate']:.2f}]")
            else:
                rs = pr["route_summary"]
                rs_s = f" routes={rs}" if rs else ""
                line += f"  {disp}[fg{pr['fg_fraction']:.3f} n{pr['n_instances']}{rs_s}]"
        results["queries"].append(entry)
        print(line)

    # ---- held-out COCO queries (with GT) ----
    specs = build_specs() if args.compare_paths else None
    for qid in args.query_ids:
        im, gt, meta = prepare_case(coco, ann_map, args.img_root, qid, args.mode, args.crop_size)
        if args.compare_paths:
            run_query_compare(f"coco_q{qid}", im, gt, meta, specs)
        else:
            run_query(f"coco_q{qid}", im, gt, meta)

    # ---- truly unseen images (not in COCO; no GT) ----
    subdir = os.path.join(args.img_root, args.unseen_subdir)
    if os.path.isdir(subdir):
        unseen = sorted(f for f in os.listdir(subdir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        and os.path.join(args.unseen_subdir, f) not in coco_files)
        for f in unseen[:args.n_unseen]:
            img = Image.open(os.path.join(subdir, f)).convert("RGB")
            if args.mode == "crop":
                w, h = img.size
                s = args.crop_size
                img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            uname = f"unseen_{os.path.splitext(f)[0]}"
            umeta = {"file_name": os.path.join(args.unseen_subdir, f)}
            if args.compare_paths:
                run_query_compare(uname, img, None, umeta, specs)
            else:
                run_query(uname, img, None, umeta)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.compare_paths:
        _print_acceptance(results, args.query_ids)
    ious = [q["iou"] for q in results["queries"] if "iou" in q]
    if ious:
        print(f"\n[test] mean IoU over {len(ious)} GT queries = {np.mean(ious):.4f}")
    print(f"[test] artefacts written to {args.out_dir}")
    return 0


def _print_acceptance(results: dict, query_ids: list) -> None:
    """Print the per-regime acceptance table and the router accept/reject verdict.

    Acceptance (per the brief): the router must match-or-beat the better of
    {boxes, per_granule} on EVERY GT query (its 'home turf' regime) — recover the
    dense pile's IoU AND keep the scattered field's recall/precision.
    """
    gt_queries = [q for q in results["queries"] if q.get("paths")
                  and any("iou" in p for p in q["paths"].values())]
    if not gt_queries:
        return
    paths = list(gt_queries[0]["paths"].keys())
    print("\n── acceptance table (IoU / P / R / bgFP per path) ──")
    head = "query".ljust(12) + "".join(p.rjust(26) for p in paths)
    print(head)
    print("-" * len(head))
    for q in gt_queries:
        row = q["name"].ljust(12)
        for p in paths:
            m = q["paths"][p]
            row += (f"{m['iou']:.2f}/{m['precision']:.2f}/"
                    f"{m['recall']:.2f}/{m['bg_fp_rate']:.2f}").rjust(26)
        print(row)

    # Router verdict: compare each router_* column to max(boxes, per_granule) IoU.
    router_paths = [p for p in paths if p.startswith("router")]
    pure = [p for p in paths if p in ("boxes", "per_granule")]
    if router_paths and pure:
        print("\n── router verdict (IoU vs best pure path, per query) ──")
        for rp in router_paths:
            ok = True
            cells = []
            for q in gt_queries:
                r_iou = q["paths"][rp]["iou"]
                best_pure = max(q["paths"][p]["iou"] for p in pure)
                delta = r_iou - best_pure
                ok = ok and (delta >= -0.02)        # within 2 pts of the better pure
                cells.append(f"{q['name']}:{r_iou:.2f}vs{best_pure:.2f}({delta:+.2f})")
            verdict = "ACCEPT" if ok else "needs-tuning"
            print(f"  {rp:<16} {verdict}  " + "  ".join(cells))


if __name__ == "__main__":
    raise SystemExit(main())
