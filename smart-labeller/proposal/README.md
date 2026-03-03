# Proposal Generation

Generates class-agnostic bounding-box proposals from images using tiled inference, then extracts per-box embeddings ready for downstream classification.

## How It Works

1. Slices each image into overlapping tiles via `OverlappingTileDataset`
2. Runs a **proposer** (SAM3 or OWLv2) on each tile to detect candidate boxes
3. Shifts tile-local boxes back to original image coordinates
4. Extracts per-box embeddings using the chosen **embedding backend**
5. Applies global NMS across all tiles to remove duplicate boxes
6. Saves one combined `.npz` (features + boxes + scores) and one `.json` (box annotations)

## Files

| File | Description |
|---|---|
| `generate_proposals.py` | CLI entry point — dispatches to the chosen proposer backend |
| `sam3_proposal.py` | SAM3 tiled inference + embedding extraction |
| `owlv2_proposal.py` | OWLv2 objectness tiled inference + embedding extraction |
| `embedding_utils.py` | Unified embedding interface for BioCLIP, DINOv3, OWLv2 |
| `run.sh` | Example run script |

## Usage

```bash
cd proposal

python generate_proposals.py \
  --backend sam3 \
  --image_dir /path/to/images \
  --output_dir /path/to/output \
  --embedding_backend dinov3
```

To run all three embedding backends (one at a time):

```bash
for emb in owlv2 dinov3 bioclip; do
  python generate_proposals.py \
    --backend sam3 \
    --image_dir /path/to/images \
    --output_dir /path/to/output/$emb \
    --embedding_backend $emb
done
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--backend` | `sam3` | Proposer backend: `sam3` or `owlv2` |
| `--image_dir` | *(required)* | Directory of input images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`) |
| `--output_dir` | `output` | Root directory to save outputs |
| `--embedding_backend` | `owlv2` | Embedding model for per-box features: `owlv2`, `dinov3`, or `bioclip` |
| `--confidence` | `0.5` | Minimum proposer score to keep a box |
| `--tile_size` | `960` | Tile size in pixels (square) |
| `--overlap_ratio` | `0.2` | Fractional overlap between adjacent tiles |
| `--batch_size` | `4` | Tiles per inference batch |
| `--nms_iou` | `0.5` | IoU threshold for stitching NMS |
| `--text_prompt` | `visual` | Text prompt passed to the proposer model |
| `--model_id` | backend default | HuggingFace model ID override |

## Proposer Backends

### SAM3 (`sam3_proposal.py`)
- Uses `facebook/sam3` via HuggingFace Transformers
- Text-prompted, class-agnostic segmentation/detection
- Embeddings always come from an **external** embedder (`dinov3`, `bioclip`, or `owlv2`)

### OWLv2 (`owlv2_proposal.py`)
- Uses `google/owlv2-large-patch14-ensemble` by default
- Class-agnostic objectness scoring (no text query used)
- When `--embedding_backend owlv2`: uses **native anchor features** directly (fastest, no re-cropping)
- When `--embedding_backend dinov3/bioclip`: re-crops each detected box and re-embeds externally

## Embedding Backends (`embedding_utils.py`)

| Backend | Model | Output Dim |
|---|---|---|
| `owlv2` | `google/owlv2-large-patch14-ensemble` | 1024 |
| `dinov3` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 |
| `bioclip` | BioCLIP ViT-B/16 | 512 |

> ⚠️ The embedding backend used here **must match** the one used in `class_supports/generate_class_supports_main.py`. Mismatched models produce different feature dimensions and will cause a `RuntimeError` during detection.

## Outputs

Saved under `<output_dir>/`:

```
objects/features_<embedding_backend>_<proposer_backend>_<timestamp>.npz
boxes/generated_boxes_<embedding_backend>_<proposer_backend>_<timestamp>.json
```

The `.npz` contains per-image keys:
```
<image_filename>_features  →  (K, D)  float32
<image_filename>_boxes     →  (K, 4)  [x1, y1, x2, y2] pixels
<image_filename>_scores    →  (K,)    float32
```

The `.json` has the structure:
```json
{
  "annotations": [
    {
      "image_path": "image.jpg",
      "bounding_box": [x1, y1, x2, y2],
      "score": 0.91,
      "class": "0.91"
    }
  ]
}
```

## Pipeline Position

```
class_supports/generate_class_supports_main.py  →  class-support .npz
                                                                        ↘
proposal/generate_proposals.py  →  proposal .npz + boxes .json  →  object_classification/object_classification_main.py
```
