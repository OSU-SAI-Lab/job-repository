#!/usr/bin/env python3
"""
Coordinate-descent sweep of the per-granule config knobs to MAXIMISE mean IoU over
the held-out fertilizer COCO queries.

One model load; the support prototypes are built once and reused. Between evals we
mutate ``cfg.prompts`` in place (the PromptGenerator holds the same object and reads
it live), so every config is evaluated against the same cached prototypes. Each
eval prints its mIoU immediately (flush) so partial progress survives a timeout.

The search is greedy coordinate descent: start from the current recommended config,
and for each knob try its candidate values (others held at the running best), keep
the best value, then move to the next knob. Repeat for ``--passes`` passes or until
no knob improves.

    python scripts/sweep_fertilizer.py --out-dir .../fss_results/sweep
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


# Per-knob candidate values. Coordinate descent tries each in turn; the baseline
# (the current recommended per-granule config) is the starting point.
# Round 2: refine around the round-1 winner (0.641), expanding past the edges
# where the optimum landed (threshold↓, seed_z↑, box_scale↓).
GRID = {
    "threshold":            [0.30, 0.35, 0.40, 0.45],
    "granule_seed_z":       [1.5, 2.0, 2.5],
    "granule_min_distance": [5, 6, 7],
    "seed_source":          ["brightness", "both"],
    "granule_box_scale":    [0.75, 1.0, 1.25],
    "granule_neg_points":   [2, 4, 6],
}
BASELINE = dict(threshold=0.40, granule_seed_z=1.5, granule_min_distance=6,
                seed_source="brightness", granule_box_scale=1.0, granule_neg_points=4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", default=R.COCO_DEFAULT)
    ap.add_argument("--img-root", default=R.IMG_ROOT_DEFAULT)
    ap.add_argument("--out-dir",
                    default="/fs/ess/PAS2699/kamath/fertilizer_data/fss_results/sweep")
    ap.add_argument("--support-ids", type=int, nargs="+", default=[0])
    ap.add_argument("--query-ids", type=int, nargs="+",
                    default=[15, 30, 45, 60, 75, 90, 105])
    ap.add_argument("--crop-size", type=int, default=1024)
    ap.add_argument("--dinov3", default="base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    from fss import FewShotSegmenter
    from fss.config import FSSConfig

    coco = R.load_coco(args.coco)
    ann_map = R.anns_by_image(coco)

    cfg = FSSConfig()
    cfg.matcher.dinov3 = args.dinov3
    cfg.dtype = args.dtype
    cfg.prompts.per_granule_prompts = True       # the path we're tuning
    cfg.prompts.multi_instance = True
    seg = FewShotSegmenter(config=cfg)
    print(f"[sweep] device={seg.device} amp={seg.amp_dtype} queries={args.query_ids}",
          flush=True)

    sup_imgs, sup_masks = [], []
    for sid in args.support_ids:
        im, mk, _ = R.prepare_case(coco, ann_map, args.img_root, sid, "crop", args.crop_size)
        sup_imgs.append(im); sup_masks.append(mk)
    seg.set_support(images=sup_imgs, masks=sup_masks)

    queries = []
    for qid in args.query_ids:
        im, gt, _ = R.prepare_case(coco, ann_map, args.img_root, qid, "crop", args.crop_size)
        queries.append((qid, im, gt))

    p = cfg.prompts

    def apply(state: dict) -> None:
        p.threshold = state["threshold"]
        p.granule_seed_z = state["granule_seed_z"]
        p.granule_min_distance = state["granule_min_distance"]
        p.seed_source = state["seed_source"]
        p.granule_box_scale = state["granule_box_scale"]
        p.granule_neg_points = state["granule_neg_points"]

    def evaluate(label: str):
        ious, per = [], {}
        for qid, im, gt in queries:
            res = seg.segment(im)
            v = R.iou(res["mask"], gt)
            ious.append(v); per[qid] = round(v, 3)
        m = float(np.mean(ious))
        print(f"[eval] {label:<40} mIoU={m:.4f}  per={per}", flush=True)
        return m, per

    state = dict(BASELINE)
    history = []
    apply(state)
    best_m, best_per = evaluate("baseline " + _fmt(state))
    history.append(["baseline", dict(state), best_m, best_per])

    for pass_i in range(args.passes):
        improved = False
        for knob, values in GRID.items():
            cur = state[knob]
            best_val = cur
            for val in values:
                if val == cur:
                    continue
                trial = dict(state); trial[knob] = val
                apply(trial)
                m, per = evaluate(f"p{pass_i} {knob}={val}")
                history.append([f"{knob}={val}", dict(trial), m, per])
                if m > best_m + 1e-6:
                    best_val, best_m, best_per = val, m, per
            if best_val != cur:
                state[knob] = best_val
                improved = True
                print(f"[update] {knob} {cur} -> {best_val}   running best mIoU={best_m:.4f}",
                      flush=True)
            apply(state)            # lock current best before the next knob
        if not improved:
            print(f"[sweep] converged after pass {pass_i}", flush=True)
            break

    print(f"\n[sweep] BEST mIoU={best_m:.4f}")
    print(f"[sweep] BEST config={state}")
    print(f"[sweep] BEST per-query={best_per}")
    with open(os.path.join(args.out_dir, "sweep_results.json"), "w") as f:
        json.dump({"best_miou": best_m, "best_config": state, "best_per": best_per,
                   "baseline": BASELINE, "grid": GRID, "history": history}, f, indent=2)
    print(f"[sweep] wrote {os.path.join(args.out_dir, 'sweep_results.json')}")
    return 0


def _fmt(state: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in state.items())


if __name__ == "__main__":
    raise SystemExit(main())
