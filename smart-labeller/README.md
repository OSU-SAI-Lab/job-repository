# Smart Labeller — Few-Shot Instance Segmentation

A training-free few-shot **instance-segmentation** pipeline built on two foundation
models only:

- **SAM3** — produces class-agnostic segmentation **masks** (text-prompted).
- **DINOv3** — embeds each masked object; cosine similarity against a handful of
  labelled examples assigns the class.

There are no bounding boxes anywhere in the outputs: proposals, matching, NMS, and
evaluation are all mask-based. (A box is used transiently in two places only — as a
SAM3 *prompt* to turn a GT box into a support mask, and as the internal crop extent
when framing an object for DINOv3 — but no box is ever stored, emitted, NMS'd, or
evaluated.)

> Converted in place from the earlier OWLv2/BioCLIP box-detection pipeline. The box
> version remains recoverable via git history.

---

## 📁 Directory Structure

#### [class_support/](class_support/README.md) — Stage 1
Build per-class **DINOv3 support embeddings** from ground-truth annotations. Each GT
`bounding_box` is turned into a mask by prompting the **SAM3 Tracker**, the masked
region is embedded with DINOv3, and vectors are stacked per class.
- Key files: `generate_class_supports_main.py`, `class_supports_utils.py`
- Output: `class_supports_dinov3_<ts>.npz` (+ `support_masks_<ts>.json`, RLE)

#### [proposal/](proposal/README.md) — Stage 2
Generate class-agnostic **mask proposals** with tiled (SAHI) **SAM3** inference, then
embed each masked instance with **DINOv3**. Overlapping tile duplicates are removed
with **mask-IoU NMS**.
- Key files: `generate_proposals.py`, `sam3_proposal.py`, `embedding_utils.py`,
  `mask_utils.py`, `OverlappingTileDataset.py`
- Proposer: **SAM3** only · Embedder: **DINOv3** only · SAHI: **on by default**
- Output: `features_dinov3_sam3_<ts>.npz` (`_features`, `_scores`, `_masks` RLE) +
  `generated_masks_dinov3_sam3_<ts>.json`

#### [object_classification/](object_classification/README.md) — Stage 3
**Few-shot classification** of mask proposals against class supports via DINOv3 cosine
similarity, with a final **mask-IoU NMS**.
- Key files: `object_classification_main.py`, `object_classification_utils.py`
- Output: `detections_dinov3_<sim>_<obj>_<ts>.json` — per-instance
  `segmentation` (RLE) + `score` + `class`

#### [evaluate_annotations.py](evaluate_annotations.py) — Stage 4
Compare generated masks with GT masks: greedy score-ordered **mask-IoU** matching →
precision, recall, F1, and **mIoU** (overall + per-class).

#### [sam3_service/](sam3_service/README.md)
FastAPI SAM3 segmentation service (Redis cache, K8s manifests). Interactive serving
path — not part of the batch stages.

---

## 🔄 Pipeline Workflow

```
GT annotations ─▶ [class_support]  SAM3(box→mask) + DINOv3 ─▶ class_supports_dinov3.npz
                                                                        │
Query images  ─▶ [proposal]  SAM3 masks (SAHI) + DINOv3 ─▶ features_dinov3_sam3.npz
                                                                        │
                                            [object_classification]  cosine + mask-NMS
                                                                        │
                                                          detections_dinov3_*.json (RLE)
                                                                        │
                                              [evaluate_annotations]  mask-IoU / mIoU
```

**Embedding-space contract:** supports and proposals are embedded through the identical
path — *zero background → crop to the mask's tight extent → DINOv3 pooler_output →
L2-normalise* — so cosine similarity is valid.

---

## 🎯 Model backends

| Role      | Model                                          | Notes |
|-----------|------------------------------------------------|-------|
| Proposer  | `facebook/sam3` (concept + tracker)            | masks; `"visual"` default prompt |
| Embedder  | `facebook/dinov3-vitl16-pretrain-lvd1689m`     | pooler_output, dim=1024 |

---

## 📋 Quick Start

```bash
# 1. Class supports (needs GT boxes; SAM3 turns each into a mask)
cd class_support && bash run.sh

# 2. Mask proposals (SAHI on by default)
cd ../proposal && bash run.sh

# 3. Classify proposals (fill in the two .npz timestamps from steps 1–2)
cd ../object_classification && bash run.sh

# 4. Evaluate against GT masks (needs GT segmentation; see note below)
cd .. && bash run.sh
```

### Ground-truth masks for evaluation
Stage 4 computes true mask-IoU/mIoU and therefore needs **GT masks**. Accepted per
annotation: COCO RLE (`"segmentation": {...}`) or polygon (`"segmentation": [[...]]`
with image `height`/`width`). If your GT has only `bounding_box`, pass `--gt_from_box`
to rasterize boxes into rectangles (coarse; measures localization, not mask quality).

---

## 📦 Containerization

Each module ships an Apptainer/Singularity definition:
- `class_support/class_supports.def`, `proposal/generate_proposals.def`,
  `object_classification/object_classification.def`

Build with: `singularity build module.sif module.def`

---

## ⚠️ Consistency requirement

All stages use the **same DINOv3 model**. Supports and proposals must be produced with
identical DINOv3 weights and the identical masked-crop embedding path, or cosine
similarity is invalid.





RUNNING Smart-Labeller for segmentation:
1. Rend a OSC node
2. Set up conda environment:
  module load miniconda3/24.1.2-py310
  source "${MINICONDA3_HOME}/etc/profile.d/conda.sh"
  conda activate fss

3. Set up Hugging Face:
export HF_HOME=/fs/scratch/PAS2699/$USER/hf_cache
export $(grep -v '^#' /users/PAS2699/naveenkamath/job-repository/.env | xargs)



