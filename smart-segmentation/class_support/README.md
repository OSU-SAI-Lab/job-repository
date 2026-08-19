# Class Supports

Generates class-support embeddings from ground-truth annotated bounding boxes. These embeddings act as **reference feature vectors** per class and are used downstream by `object_classification` to match detected proposals against known classes.

## How It Works

1. Reads a COCO-format annotation JSON containing GT bounding boxes and class labels
2. Crops each GT box from its source image
3. Embeds the crop using one or more backends (OWLv2, BioCLIP, DINOv3)
4. Stacks all crop embeddings per class into a tensor → `class_supports[class_name] = (N, D)`
5. Optionally iterates over multiple crop sizes, tracking IoU improvement to find the best scale
6. Saves one `.npz` per backend and one shared GT-box `.json`

## Files

| File | Description |
|---|---|
| `generate_class_supports_main.py` | CLI entry point |
| `class_supports_utils.py` | Model loaders, per-backend crop embedding functions, `extract_support_embeddings()` |
| `class_supports.def` | Apptainer/Singularity container definition |
| `requirements.txt` | Python dependencies |
| `run.sh` | Example run script |

## Usage

```bash
cd class_supports

python generate_class_supports_main.py \
  --ann_path /path/to/annotations.json \
  --src_path /path/to/source/images \
  --output_path /path/to/output \
  --embedding_backend owlv2 bioclip dinov3 \
  --crop_size 1024 \
  --device cuda
```

To search over multiple crop sizes (runs largest first, stops if IoU stops improving):

```bash
python generate_class_supports_main.py \
  --ann_path /path/to/annotations.json \
  --src_path /path/to/source/images \
  --output_path /path/to/output \
  --embedding_backend owlv2 bioclip dinov3 \
  --crop_size '[512,1024]'
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--ann_path` | *(required)* | Path to COCO-format annotation JSON with GT boxes |
| `--src_path` | `None` | Directory containing the source images |
| `--output_path` | `None` | Root directory for output files |
| `--embedding_backend` | `owlv2` | One or more of: `owlv2`, `bioclip`, `dinov3` |
| `--crop_size` | `1024` | Single int or JSON list e.g. `'[512,1024]'` (sorted largest→smallest) |
| `--method` | `image` | Detection method: `image`, `text`, or `RPN` |
| `--model_name` | backend default | HuggingFace model ID override |
| `--device` | auto | `cuda` or `cpu` |
| `--use_sahi` | `False` | Enable SAHI slicing |

## Annotation Format

The input JSON must follow COCO format with an `"annotations"` key:

```json
{
  "annotations": [
    {
      "image_path": "image.jpg",
      "bounding_box": [x1, y1, x2, y2],
      "class": "species_name"
    }
  ]
}
```

## Embedding Backends

| Backend | Model | Output Dim |
|---|---|---|
| `owlv2` | `google/owlv2-large-patch14-ensemble` | 1024 |
| `dinov3` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 |
| `bioclip` | BioCLIP ViT-B/16 | 512 |

> ⚠️ The backend used here **must match** the embedding backend used in `proposal/generate_proposals.py`. Mismatched models produce incompatible feature dimensions and will cause a `RuntimeError` during classification.

## Outputs

Saved under `<output_path>/`:

```
tensors/class_supports_<backend>_<crop_size>_<timestamp>.npz     (one per backend)
generated_boxes/generated_boxes_<crop_size>_<timestamp>.json     (shared, saved once)
```

The `.npz` contains one key per class:
```
<class_name>  →  (N, D)  float32   (N = number of GT examples for that class)
```

The `.json` records the exact cropped box coordinates used:
```json
{
  "annotations": [
    {
      "image_path": "image.jpg",
      "bounding_box": [x1, y1, x2, y2],
      "class": "species_name"
    }
  ]
}
```

## Pipeline Position

```
GT annotation JSON  →  class_supports/generate_class_supports_main.py  →  class-support .npz (per backend)
                                                                                                    ↓
                                                          object_classification/object_classification_main.py
```

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `torch`, `transformers`, `bioclip`, `open-clip-torch`, `Pillow`, `numpy`
