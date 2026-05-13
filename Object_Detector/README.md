# Unified Object Detector (OWLv2 / Grounding DINO / YOLOE / SAM3)

Run open-vocabulary or specialist object detection on a folder of images using a **single CLI**. The backend is chosen from the Patra/Harvest card title (`--model_name` or `MODEL_NAME`) by keyword: **OWLv2**, **Grounding+DINO**, **YOLOE**, or **SAM3**. Outputs aggregate JSON plus annotated overlays for review.

## Files

| File | Description |
|---|---|
| `owlv2_classifier.py` | CLI entry point (all four detector backends) |
| `owlv2_classifier.def` | Apptainer/Singularity definition |
| `environment.yml` | Conda environment (PyTorch, transformers, OpenCV, etc.) |

## What It Does

1. Resolves the detector backend from `MODEL_NAME` / `--model_name` using keyword rules (see below)
2. Decodes base64 `--class_configs` JSON (list of class definitions with `search_terms` text prompts)
3. Scans `--input_folder` for common raster formats
4. Runs the selected model (OWLv2, Grounding DINO, YOLOE, or SAM3) with model-specific defaults and optional tuning flags
5. Writes `overall_results.json` under `--output_folder` with per-image detections
6. Saves `annotated_<filename>` images under `annotated/` when detections are drawn
7. Optionally preserves image metadata when saving (default on)

**SAM3** requires a Hugging Face token (`HF_TOKEN` or `--hf_token`) to download weights from the Hub.

## Backend selection (`MODEL_NAME` / `--model_name`)

The title string must contain one of:

- **SAM3** — Ultralytics SAM3 (Hub weights; token required)
- **OWLv2** or **Owl v2** — Hugging Face OWLv2
- **Grounding+DINO** — Grounding DINO (default HF id `IDEA-Research/grounding-dino-tiny` unless overridden)
- **YOLOE** — Ultralytics YOLOE (default weights `yoloe-26s-seg.pt` unless `--weights` is set)

If the CLI omits `--model_name`, the `MODEL_NAME` environment variable must be set (typical for Tapis/Harvest jobs).

## Supported formats

Images directly in `--input_folder` matching:

- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` (common case extensions)

## Usage

```bash
export MODEL_NAME="OWLv2 Field Weeds"

python owlv2_classifier.py \
  --input_folder /path/to/images \
  --output_folder /path/to/output \
  --class_configs "$(printf '%s' '[...]' | base64)"
```

For SAM3, set a token before running:

```bash
export HF_TOKEN="hf_..."
export MODEL_NAME="SAM3 Segmenter"

python owlv2_classifier.py \
  --input_folder /path/to/images \
  --output_folder /path/to/output \
  --class_configs "$(printf '%s' '...' | base64)"
```

## Arguments

| Argument | Required | Description |
|---|---|---|
| `--model_name` | One of cli/env | Patra card title; keywords pick the backend (see above). Multi-word args are joined. |
| `--input_folder` | Yes | Directory of input images |
| `--output_folder` | Yes | Where `overall_results.json` and `annotated/` are written |
| `--class_configs` | Yes | Base64-encoded JSON **list** of objects; each should include `search_terms` (list of prompt strings) for detection classes |
| `--confidence_threshold` | No | Detection confidence; default is backend-specific if omitted (~0.1 OWLv2/SAM3, ~0.12 Grounding DINO/YOLOE) |
| `--preserve_metadata` | No | Store flags (default: metadata preservation on) |
| `--num_workers` | No | Worker count (default `12`) |
| `--batch_size` | No | Batch size (default `1`) |
| `--text_threshold` | No | Grounding DINO phrase threshold (default `0.25`) |
| `--model_id` | No | Grounding DINO HF model id |
| `--io_workers` | No | Grounding DINO parallel load threads (default `8`) |
| `--save_workers` | No | Grounding DINO async save threads (default `4`) |
| `--amp` | No | Grounding DINO CUDA autocast FP16 |
| `--weights` | No | YOLOE weights path (SAM3 ignores) |
| `--imgsz` | No | Inference size (YOLOE default 640; SAM3 default 644; `0` means omit for YOLOE) |
| `--half` | No | YOLOE FP16 predict |
| `--iou` | No | SAM3 NMS IoU (default `0.7`) |
| `--device` | No | SAM3 device, e.g. `0` or `cpu` |
| `--hf_token` | SAM3 | Hugging Face token (prefer `HF_TOKEN` env) |
| `--hf_repo_id` | No | SAM3 Hub repo (default `facebook/sam3`) |
| `--hf_weights_filename` | No | SAM3 file inside repo (default `sam3.pt`) |
| `--sam3_max_area_frac` | No | Drop SAM3 boxes larger than this image area fraction |
| `--sam3_max_side_frac` | No | Drop SAM3 boxes whose longest side exceeds this fraction of image |

## `class_configs` shape

Decode the base64 payload to UTF-8 JSON. It must be a **list** of objects. Each object should carry **`search_terms`**: a list of strings used as class prompts (OWLv2/Grounding DINO paths aggregate these; other backends follow the same config list for consistency).

Minimal pattern:

```json
[
  {
    "search_terms": ["weed", "corn"]
  }
]
```

(Exact schema may include additional keys your Harvest step supplies; the detector merges all `search_terms` from all entries.)

## Output layout

```text
output_folder/
  overall_results.json
  annotated/
    annotated_<original_image_name>
```

## Build the Container

```bash
apptainer build owlv2-classifier.sif owlv2_classifier.def
```

## Harvest Integration

This job aligns with the unified detector interface:

- **`MODEL_NAME`** (or `--model_name`) — selects OWLv2 vs Grounding DINO vs YOLOE vs SAM3
- **`--input_folder`** / **`--output_folder`** — image I/O roots
- **`--class_configs`** — base64 JSON from the UI/launcher encoding search prompts per class

SAM3 jobs must also provide **`HF_TOKEN`** (or `--hf_token`) after accepting the model license on Hugging Face.
