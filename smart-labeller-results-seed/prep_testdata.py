"""
Prepare the test-data 'seed' COCO segmentation set for the smart-labeller pipeline.

Source COCO : test-data/segmentation.coco (3).json  (4 images, 301 seed instances)
Images root : /fs/ess/PAS2699/agriculture/fertilizer_dataset

Produces under OUT:
  data/support.json  – few-shot support (Stage 1): image_path, class, bounding_box xyxy
  data/gt.json       – GT masks (Stage 4): image_path, class, segmentation(polygon), H, W
  data/support_dir/  – symlink to the support image
  data/query_dir/    – symlinks to the query images (Stage 2 --image_dir)

Support (image 0) and query (images 1-3) are disjoint → fair few-shot eval.
"""

import json
import os
from pathlib import Path

from PIL import Image

COCO = "/users/PAS2699/naveenkamath/job-repository/test-data/segmentation.coco (3).json"
IMG_ROOT = "/fs/ess/PAS2699/agriculture/fertilizer_dataset"
OUT = "/users/PAS2699/naveenkamath/job-repository/smart-labeller-results-seed"

SUPPORT_IMAGE_ID = 0
SUPPORT_K        = 10
QUERY_IMAGE_IDS  = [1, 2, 3]

Image.MAX_IMAGE_PIXELS = None


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
        with Image.open(os.path.join(IMG_ROOT, imgs[image_id]["file_name"])) as im:
            return im.width, im.height

    def link(image_id, dst_dir):
        src = os.path.join(IMG_ROOT, imgs[image_id]["file_name"])
        base = Path(imgs[image_id]["file_name"]).name
        dst = dst_dir / base
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return base, str(dst)

    # Support (Stage 1)
    sup_base, _ = link(SUPPORT_IMAGE_ID, support_dir)
    support = []
    for a in anns_by_img[SUPPORT_IMAGE_ID][:SUPPORT_K]:
        x, y, w, h = a["bbox"]
        support.append({
            "image_path":   sup_base,
            "class":        cats[a["category_id"]],
            "bounding_box": [x, y, x + w, y + h],
        })
    json.dump({"annotations": support}, open(data_dir / "support.json", "w"), indent=2)

    # GT masks for query images (Stage 4)
    gt = []
    for iid in QUERY_IMAGE_IDS:
        _, qpath = link(iid, query_dir)
        W, H = real_size(iid)
        for a in anns_by_img[iid]:
            gt.append({
                "image_path":   qpath,
                "class":        cats[a["category_id"]],
                "segmentation": a["segmentation"],
                "height":       H,
                "width":        W,
                "score":        1.0,
            })
    json.dump({"annotations": gt}, open(data_dir / "gt.json", "w"), indent=2)

    print(f"support: {len(support)} '{cats[1]}' instances from image {SUPPORT_IMAGE_ID} ({sup_base})")
    print(f"query:   {len(QUERY_IMAGE_IDS)} images, {len(gt)} GT instances")
    print(f"support_dir: {support_dir}")
    print(f"query_dir:   {query_dir}")


if __name__ == "__main__":
    main()
