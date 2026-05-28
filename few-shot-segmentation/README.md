# Smart Labeller

A comprehensive multi-backend object classification and detection pipeline featuring few-shot learning with embeddings from multiple foundation models (OWLv2, BioCLIP, DINOv3). The system generates class-support embeddings, creates object proposals using tiled inference, and performs few-shot classification via cosine similarity or logit-based matching.

---

## 📁 Directory Structure

### Core Pipeline Components

#### [class_support/](class_support/README.md)
**Generate class-support embeddings from ground-truth annotations**

Reads COCO-format annotations with bounding boxes, crops GT regions, and embeds them using configurable backends (OWLv2, BioCLIP, DINOv3). Outputs per-class embedding tensors for downstream classification.

- **Key files:** `generate_class_supports_main.py`, `class_supports_utils.py`
- **Outputs:** `.npz` files with per-class support embeddings
- **Supports multiple crop sizes** with IoU-based best-size selection

➜ [See detailed documentation →](class_support/README.md)

---

#### [proposal/](proposal/README.md)
**Generate class-agnostic object proposals with embeddings**

Uses tiled inference (SAM3 or OWLv2) to detect proposals across images, extracts per-box embeddings, applies NMS, and saves proposal features ready for classification.

- **Key files:** `generate_proposals.py`, `sam3_proposal.py`, `owlv2_proposal.py`, `embedding_utils.py`
- **Proposers:** SAM3 (text-prompted segmentation) or OWLv2 (objectness scoring)
- **Embedding backends:** OWLv2, DINOv3, BioCLIP
- **Features:** Tiled inference, overlapping tiles, Global NMS, caching

➜ [See detailed documentation →](proposal/README.md)

---

#### [object_classification/](object_classification/README.md)
**Perform few-shot classification of proposals against class supports**

Loads class-support and proposal embeddings, matches proposals to classes using per-backend similarity metrics (cosine for BioCLIP/DINOv3, logits for OWLv2), applies NMS, and outputs detections.

- **Key files:** `object_classification_main.py`, `object_classification_utils.py`
- **Matching methods:** Cosine similarity (BioCLIP, DINOv3) + OWLv2 class predictor
- **Outputs:** Per-backend detection JSONs with bboxes, scores, and class labels

➜ [See detailed documentation →](object_classification/README.md)

---

#### [sam3_service/](sam3_service/README.md)
**FastAPI service for SAM3 segmentation (Kubernetes-ready)**

REST API providing concept segmentation (text-prompted) and point-based segmentation with Redis caching. Deployable to NRP Nautilus Kubernetes cluster.

- **Key files:** `main.py`, `Dockerfile`, `sam3_nrp_deployment.yaml`, `deploy.sh`
- **Features:** FastAPI, Redis caching, GPU support, Kubernetes deployment
- **Endpoints:** `/predict` (text/point mode), `/predict_box` (legacy)

➜ [See detailed documentation →](sam3_service/README.md)

---

### Supporting Components

#### [results/](results/)
**Output directory for classification results**

Stores workflow outputs:
- `summary.json` — High-level detection statistics
- `detailed_results.json` — Per-image detailed detections

---

#### [evaluate_annotations.py](evaluate_annotations.py)
**Evaluate classification quality against ground truth**

Compares generated detections with ground-truth annotations, computes metrics (precision, recall, mAP), and generates evaluation reports.

---

#### [run.sh](run.sh)
**Top-level orchestration script**

Coordinates execution of the full pipeline:
1. Generate class supports
2. Generate proposals
3. Run object classification
4. (Optional) Evaluate results

---

## 🔄 Pipeline Workflow

```
Annotations (COCO format)  →  [class_support/]  →  Class-support embeddings (.npz)
                                                              ↓
Images  →  [proposal/]  →  Proposal embeddings + boxes (.npz) →  [object_classification/]  →  Detections (.json)
```

**Flow Summary:**
1. **Class Supports** extract reference embeddings from ground-truth boxes
2. **Proposals** find candidate objects in query images and compute their embeddings
3. **Classification** matches proposals to known classes using similarity/logit scores
4. Results are saved per-backend with configurable thresholds

---

## ⚠️ Critical: Backend Consistency

**All three stages must use the same embedding backend(s):**

```bash
# ✅ CORRECT: All use dinov3
python class_support/generate_class_supports_main.py --embedding_backend dinov3 ...
python proposal/generate_proposals.py --embedding_backend dinov3 ...
python object_classification/object_classification_main.py --embedding_backends dinov3 ...

# ❌ INCORRECT: Mismatched backends → RuntimeError
python class_support/generate_class_supports_main.py --embedding_backend dinov3 ...
python proposal/generate_proposals.py --embedding_backend owlv2 ...  # Different!
python object_classification/object_classification_main.py --embedding_backends dinov3 ...
# Error: mat1 and mat2 shapes cannot be multiplied (52x768 and 1024x17)
```

---

## 🎯 Supported Embedding Backends

| Backend | Model | Dimension | Recommended Use |
|---------|-------|-----------|---|
| **OWLv2** | `google/owlv2-large-patch14-ensemble` | 1024 | General object detection, text-prompted |
| **DINOv3** | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 | Vision-only features, strong spatial encoding |
| **BioCLIP** | BioCLIP ViT-B/16 | 512 | Biological/ecological domain, faster inference |

---

## 📋 Quick Start

### 1. Generate Class Supports

```bash
cd class_support
python generate_class_supports_main.py \
  --ann_path /path/to/annotations.json \
  --src_path /path/to/images \
  --output_path /path/to/output \
  --embedding_backend dinov3 \
  --device cuda
```

### 2. Generate Proposals

```bash
cd ../proposal
python generate_proposals.py \
  --backend sam3 \
  --image_dir /path/to/query/images \
  --output_dir /path/to/output \
  --embedding_backend dinov3
```

### 3. Run Classification

```bash
cd ../object_classification
python object_classification_main.py \
  --qry_path /path/to/query/images \
  --is_query_dir \
  --embedding_backends dinov3 \
  --class_support_file_paths /path/to/class_supports_dinov3.npz \
  --object_features_file_paths /path/to/features_dinov3.npz \
  --output_path /path/to/output \
  --similarity_threshold 0.2 \
  --objectness_threshold 0.1
```

---

## 📦 Containerization

Each module includes Apptainer/Singularity definitions for reproducible execution:
- `class_support/class_supports.def`
- `proposal/generate_proposals.def`
- `object_classification/object_classification.def`

Build with: `singularity build module.sif module.def`

---

## 🔗 Further Reading

- **Embedding Details:** See [embedding_utils.py](proposal/embedding_utils.py) in proposal/
- **NMS & Filtering:** See [object_classification_utils.py](object_classification/object_classification_utils.py)
- **Tiled Inference:** See [OverlappingTileDataset.py](proposal/OverlappingTileDataset.py) in proposal/
- **Kubernetes Deployment:** See [sam3_service/README.md](sam3_service/README.md)
