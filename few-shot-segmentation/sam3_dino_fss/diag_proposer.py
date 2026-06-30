#!/usr/bin/env python3
"""Diagnostic: which SAM3 concept text actually proposes the granules?

Runs Sam3Proposer with several candidate texts at a LOW score threshold on a couple
of fertilizer crops and prints proposal counts + score ranges, to tell whether the
zero-proposal result is a text/threshold issue or SAM3 simply not seeing granules.
"""
from __future__ import annotations
import os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sam3_dino_fss.run_fertilizer as RF
from sam3_dino_fss.proposer import Sam3Proposer
from sam3_dino_fss.pipeline import _resolve_amp

TEXTS = ["visual", "object", "objects", "granule", "granules", "fertilizer",
         "fertilizer granule", "pellet", "seed", "small stones", "stone", "dots",
         "rocks", "particles"]


def main():
    coco = RF.load_coco(RF.COCO_DEFAULT)
    ann_map = RF.anns_by_image(coco)
    device = "cuda"
    amp = _resolve_amp("bfloat16", device)
    prop = Sam3Proposer(device=device, amp_dtype=amp, score_threshold=0.0)
    for qid in (60, 105):
        img, gt, _ = RF.prepare_case(coco, ann_map, RF.IMG_ROOT_DEFAULT, qid, 1024)
        print(f"\n=== q{qid} (GT fg px={int(gt.sum())}) ===", flush=True)
        for t in TEXTS:
            props = prop.propose(img, t)
            if props:
                sc = [p.score for p in props]
                areas = [int(p.mask.sum()) for p in props]
                print(f"  {t:<20} n={len(props):>4}  score[{min(sc):.2f},{max(sc):.2f}]"
                      f"  area[{min(areas)},{max(areas)}]", flush=True)
            else:
                print(f"  {t:<20} n=0", flush=True)


if __name__ == "__main__":
    main()
