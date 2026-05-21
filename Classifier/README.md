# DINOv3 Object Classifier

Refine detector outputs by assigning **embedding-based class labels** to each bounding box. Reference crops (per class) define prototype embeddings with Hugging Face **DINOv3**; each detected patch is embedded and matched by cosine similarity. This matches the Patra “classifier” step that consumes a detector JSON and writes labeled JSON plus optional per-class image folders.

## Files

| File | Description |
|---|---|
| `dinov3_classifier.py` | CLI entry point for DINOv3 classification |
| `dinov3_classifier.def` | Apptainer/Singularity definition |
| `environment.yml` | Conda environment (PyTorch, transformers, etc.) |

## What It Does

1. Requires **transformers ≥ 4.56** (native DINOv3 support)
2. Resolves the HF checkpoint from **`MODEL_NAME` / `--model_name`** (title must mention `dinov3` or `dino`) or from explicit `--model_id` (defaults to `facebook/dinov3-vitl16-pretrain-lvd1689m`)
3. Decodes base64 `--class_configs`: a JSON list of classes with `class_name`, `source_image_path`, and optional `crop_coordinates` `[x1, y1, x2, y2]` for reference patches
4. Loads `facebook/dinov3-…` (gated) with **`HF_TOKEN`** or `--hf_token`
5. Embeds each reference crop and each detection crop from `--input_json` / `--input_images_dir`
6. Overwrites/sets each object’s `label` with the best-matching class and writes **`classifier_confidence`** (dot product score)
7. Writes **`--output_json`** and copies source images into **`--output_images_dir/<class_name>/`** when that image had at least one object labeled with that class

## Usage

```bash
export MODEL_NAME="DINOv3 Weed Classifier"
export HF_TOKEN="hf_..."

python dinov3_classifier.py \
  --input_json /path/to/detections.json \
  --input_images_dir /path/to/images \
  --output_json /path/to/labeled.json \
  --output_images_dir /path/to/by_class_images \
  --class_configs "$(printf '%s' '[...]' | base64)"
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--input_json` | Yes | Detector JSON (list of objects with `img`, `bounding_box`, etc.) |
| `--input_images_dir` | Yes | Root directory for images referenced in JSON |
| `--output_json` | Yes | Path for the updated JSON (parent dirs are created) |
| `--output_images_dir` | Yes | Base folder; subfolders per `class_name` receive copied source images when applicable |
| `--class_configs` | Yes | Base64-encoded JSON **list** of class configs (see below) |
| `--model_name` | No | Patra/Harvest title tokens; used with `MODEL_NAME` env to infer DINOv3 checkpoint |
| `--model_id` | No | Explicit Hugging Face model id (skips name-based resolution) |
| `--hf_token` | Yes* | Hugging Face token for gated DINOv3; *or set **`HF_TOKEN`** |

## `class_configs` format

After base64 decode, JSON must be a non-empty list of objects, for example:

```json
[
  {
    "class_name": "Waterhemp",
    "source_image_path": "reference/waterhemp.jpg",
    "crop_coordinates": [100, 80, 400, 360]
  },
  {
    "class_name": "Foxtail",
    "source_image_path": "reference/foxtail.png"
  }
]
```

Fields:

- **`class_name`** — label applied on match
- **`source_image_path`** — relative to `--input_images_dir` or absolute
- **`crop_coordinates`** — optional `[x1, y1, x2, y2]` pixel crop; if omitted, the full reference image is embedded

## Input JSON expectations

Items are grouped by image basename (`img`). Each object should include a valid `bounding_box` `[x1, y1, x2, y2]` in pixel coordinates within the image. Invalid boxes are skipped.

## Output layout

- **`output_json`** — list of classified detection records with `label` and `classifier_confidence`
- **`output_images_dir`** — one subfolder per configured class name; whole source images copied when that image received that label

## Build the Container

```bash
apptainer build dinov3-classifier.sif dinov3_classifier.def
```

## Harvest Integration

Typical wiring:

- **`MODEL_NAME`** — must indicate DINO so the correct checkpoint is chosen (or pass **`--model_id`** explicitly)
- **`HF_TOKEN`** — required for the gated DINOv3 model
- **`--input_json`** / **`--input_images_dir`** — prior detector outputs
- **`--class_configs`** — base64 bundle built from the UI (reference crops per class)
- **`--output_json`** / **`--output_images_dir`** — downstream steps consume the refined labels and optional class-sorted imagery
