"""
Prepare the fertilizer COCO segmentation dataset for the smart-labeller
segmentation pipeline.

Produces, under OUT:
  data/support.json   – few-shot support set (Stage 1 input): image_path, class,
                        bounding_box [x1,y1,x2,y2]   (from a SUPPORT image)
  data/gt.json        – ground-truth masks (Stage 4): image_path, class,
                        segmentation (polygon), height, width, score=1.0
                        (for the QUERY images)
  data/support_dir/   – symlink(s) to the support image(s)
  data/query_dir/     – symlinks to the query images (Stage 2 --image_dir)

Support/query are disjoint images so evaluation is a fair few-shot test.
"""

import json
import os
from pathlib import Path

from PIL import Image

COCO = "/fs/ess/PAS2699/kamath/fertilizer_data/fertilizer_segmentation_coco.json"
IMG_ROOT = "/fs/ess/PAS2699/agriculture/fertilizer_dataset"
OUT = "/users/PAS2699/naveenkamath/job-repository/smart-labeller-results"

# Which COCO image_ids to use.
SUPPORT_IMAGE_ID = 105          # source of the few-shot support instances
SUPPORT_K        = 10           # number of support instances to take
QUERY_IMAGE_IDS  = [60, 15]     # images to segment + evaluate (21 + 60 instances)

Image.MAX_IMAGE_PIXELS = None   # these frames are ~4284x4284


def main():
    d = json.load(open(COCO))
    imgs = {im["id"]: im for im in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    anns_by_img = {}
    for a in d["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    data_dir    = Path(OUT) / "data"
    support_dir = data_dir / "support_dir"
    query_dir   = data_dir / "query_dir"
    for p in (support_dir, query_dir):
        p.mkdir(parents=True, exist_ok=True)

    def real_size(image_id):
        p = os.path.join(IMG_ROOT, imgs[image_id]["file_name"])
        with Image.open(p) as im:
            return im.width, im.height

    def link(image_id, dst_dir):
        src = os.path.join(IMG_ROOT, imgs[image_id]["file_name"])
        base = Path(imgs[image_id]["file_name"]).name
        dst = dst_dir / base
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return base, str(dst)

    # ── Support set (Stage 1) ──────────────────────────────────────────────
    sup_base, _ = link(SUPPORT_IMAGE_ID, support_dir)
    support = []
    for a in anns_by_img[SUPPORT_IMAGE_ID][:SUPPORT_K]:
        x, y, w, h = a["bbox"]                      # COCO xywh
        support.append({
            "image_path":   sup_base,               # relative to support_dir
            "class":        cats[a["category_id"]],
            "bounding_box": [x, y, x + w, y + h],    # -> xyxy for SAM3 prompt
        })
    json.dump({"annotations": support}, open(data_dir / "support.json", "w"), indent=2)

    # ── Ground-truth masks for the query images (Stage 4) ──────────────────
    gt = []
    for iid in QUERY_IMAGE_IDS:
        _, qpath = link(iid, query_dir)             # absolute symlink path
        W, H = real_size(iid)
        for a in anns_by_img[iid]:
            gt.append({
                "image_path":   qpath,               # MUST match Stage 3's str(query_file)
                "class":        cats[a["category_id"]],
                "segmentation": a["segmentation"],   # COCO polygon
                "height":       H,
                "width":        W,
                "score":        1.0,
            })
    json.dump({"annotations": gt}, open(data_dir / "gt.json", "w"), indent=2)

    print(f"support: {len(support)} instances from image {SUPPORT_IMAGE_ID} ({sup_base})")
    print(f"query:   {len(QUERY_IMAGE_IDS)} images, {len(gt)} GT instances")
    print(f"support_dir: {support_dir}")
    print(f"query_dir:   {query_dir}")
    print(f"support.json: {data_dir / 'support.json'}")
    print(f"gt.json:      {data_dir / 'gt.json'}")


if __name__ == "__main__":
    main()
