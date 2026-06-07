---
name: corn-residue-eval-limitation
description: Why the corn_residue few-shot pipeline can't be evaluated meaningfully right now
metadata:
  type: project
---

Corn-residue few-shot-segmentation pipeline at `/fs/ess/PAS2699/kamath/corn_residue/v1/`.

**UPDATE 2026-06-07 — largely resolved by the mask-based path; see [[corn-residue-fewshot-setup]].**
The DINOv3+SAM3 mask pipeline (`mask_step*.sh`) now puts COCO RLE segmentation on BOTH
sides (SAM3 proposal masks → detections; per-piece GT in `gt/`), so `evaluate_annotations.py`
uses real mask IoU. `evaluate_annotations.py` was edited to key by basename. The notes below
describe the OLD OWLv2 box path (kept for context).

The step-3 classifier (`object_classification.sif`) detect output carries only
`bounding_box / score / class / image_path` — **no segmentation field**. So
`evaluate_annotations.py` cannot use mask IoU and falls back to box IoU. GT in
`support/annotations.json` is full-frame `[0,0,W,H]` boxes, so box IoU is near-meaningless
(measures frame coverage, not detection quality). Support-set eval gave P=0.235 R=0.667 F1=0.348
at IoU 0.5 — do not trust as a real metric.

**Why:** real COCO masks exist in `field_{1,2,3}/annotations_coco_field_*.json` but are unusable
because the generated side has no masks; neither container has pycocotools either.
**How to apply:** for a meaningful metric, get the classifier to emit segmentation masks, or use
per-object GT boxes instead of whole-frame.

Gotchas fixed this session: step3 classifier rejects `--proposals_json_file_paths`; pipeline writes
to `detections/` not `detect/`; generated `image_path` is `/query/<file>` and must be basename-normalized
to match GT. Truncated field_1 images (GX010081–86) handled by bind-mounting the
`ImageFile.LOAD_TRUNCATED_IMAGES`-patched `OverlappingTileDataset.py` into the prebuilt proposals SIF.

**SAM box->mask added (Route A).** `jobs/add_masks_sam.py` + `jobs/step4_sam_masks.sh` prompt HF
`facebook/sam-vit-base` (cached at `models/hf_cache`, run offline) with each detection box to produce
masks, encoded as compressed COCO RLE. pycocotools installed `--no-deps` into `models/../pylibs`
(`PYTHONPATH=$BASE/pylibs`) since no container has it. Containers' `transformers` 4.57.3 exposes
SamModel/SamProcessor. Use `apptainer exec` (not run) for custom scripts in the proposals SIF.
**Result is poor:** support-set mask IoU mean ~0.14, best-per-image max 0.33, 0/30 images reach
IoU>=0.5 -> precision/recall 0.0 at thr 0.5. Root cause: OWLv2 proposes coarse ~half-frame boxes,
SAM just fills them (~50% coverage) while true residue masks are ~22% irregular coverage. Corn residue
is a diffuse texture class, not a crisp object, so box->SAM instance seg is a poor fit; a trained
semantic-segmentation model is likely the better route despite only 30 labeled images.
