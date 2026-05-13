#!/usr/bin/env python3

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoImageProcessor, __version__ as HF_VER

# Single supported embedding backend today (same behaviour as Patra-id 4).
DEFAULT_DINOV3_HF = "facebook/dinov3-vitl16-pretrain-lvd1689m"

DEFAULT_CACHE_DIR = os.getenv("HF_HOME") or "/tmp/hf_cache"


def _require_transformers_456():
    v = HF_VER.split("+")[0]
    parts = [int(x) for x in v.split(".")[:3]]
    if parts < [4, 56, 0]:
        raise RuntimeError(
            f"Transformers >= 4.56.0 is required for DINOv3; found {HF_VER}. "
            "Upgrade transformers in your container."
        )


def _normalize_title_for_match(name: str) -> str:
    """Lowercase + normalize Patra-style punctuation so matching is stable."""
    s = str(name or "").strip()
    s = s.replace("—", "-").replace("–", "-").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def resolve_hf_model_id(model_name: str, model_id_override: Optional[str]) -> str:
    """Pick HF checkpoint from explicit --model_id or from display title keywords."""
    if model_id_override and model_id_override.strip():
        # Explicit override (unchanged).
        return model_id_override.strip()

    raw = (model_name or "").strip()
    if not raw:
        raise ValueError(
            "model_name is required: set MODEL_NAME in the job environment (Harvest/Tapis) "
            "or pass --model_name (can be multiple shell tokens; they are joined)."
        )

    lowered = _normalize_title_for_match(raw)
    compact = "".join(lowered.split())
    needs_dino = "dinov3" in lowered or "dino" in compact
    if not needs_dino:
        raise ValueError(
            f"Unsupported model_name={raw!r}. "
            "This container build only loads DINOv3; "
            "include 'dinov3' or 'dino' in the title, "
            "or pass --model_id explicitly."
        )
    return DEFAULT_DINOV3_HF


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def crop_image(image_path: str, coords) -> Image.Image:
    img = load_image(image_path)
    x1, y1, x2, y2 = coords
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img.width, int(x2))
    y2 = min(img.height, int(y2))
    return img.crop((x1, y1, x2, y2))


def embed_patch(pil_img: Image.Image, model, processor, device) -> np.ndarray:
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    n_special = 1 + int(getattr(model.config, "num_register_tokens", 0))
    if n_special >= tokens.shape[0]:
        raise RuntimeError(f"Unexpected token layout: T={tokens.shape[0]}, n_special={n_special}")

    patch_tokens = tokens[n_special:]
    emb = patch_tokens.mean(axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb.astype(np.float32)


def load_class_embeddings(class_configs, images_dir, model, processor, device):
    embs = {}
    for cfg in class_configs:
        class_name = cfg["class_name"]
        source_path = cfg["source_image_path"]
        coords = cfg.get("crop_coordinates")

        if os.path.isabs(source_path):
            full_path = source_path
        else:
            full_path = os.path.join(images_dir, source_path.lstrip("/"))

        if not os.path.exists(full_path):
            raise ValueError(f"Source image not found: {full_path}")

        patch = crop_image(full_path, coords) if coords else load_image(full_path)
        embs[class_name] = embed_patch(patch, model, processor, device)
    return embs


def load_detector_json(json_path):
    if not os.path.exists(json_path):
        raise ValueError(f"Detector JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    grouped = {}
    for item in data:
        key = Path(item.get("img", "")).name
        grouped.setdefault(key, []).append(item)
    return grouped, data


def validate_bbox(bbox) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if not all(isinstance(v, (int, float)) for v in bbox):
        return False
    if x1 >= x2 or y1 >= y2:
        return False
    if x1 < 0 or y1 < 0:
        return False
    return True


def classify_objects(class_embs, grouped, images_dir, model, processor, device, output_images_dir):
    results = []
    image_classes = {}

    for img_name, objects in grouped.items():
        image_path = Path(images_dir) / img_name
        if not image_path.exists():
            continue

        image_classes[img_name] = set()
        has_classified_objects = False

        for obj in objects:
            bbox = obj.get("bounding_box")
            if not validate_bbox(bbox):
                continue

            patch = crop_image(str(image_path), bbox)
            obj_emb = embed_patch(patch, model, processor, device)
            best_label, best_score = None, -1.0

            for class_name, class_emb in class_embs.items():
                score = float(np.dot(obj_emb, class_emb))
                if score > best_score:
                    best_score, best_label = score, class_name

            if best_label:
                obj["label"] = best_label
                image_classes[img_name].add(best_label)
                has_classified_objects = True

            obj["classifier_confidence"] = best_score
            results.append(obj)

        if not has_classified_objects and img_name in image_classes:
            del image_classes[img_name]

    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)

        for class_name in class_embs.keys():
            class_dir = os.path.join(output_images_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        for img_name, classes in image_classes.items():
            if not classes:
                continue

            image_path = Path(images_dir) / img_name
            if not image_path.exists():
                continue

            for class_name in classes:
                class_dir = os.path.join(output_images_dir, class_name)
                dest_path = os.path.join(class_dir, img_name)
                shutil.copy2(str(image_path), dest_path)
                print(f"Copied {img_name} to {class_name}/")

    return results


def prepare_cache_dir():
    cache_dir = Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    return str(cache_dir)


def require_hf_token(cli_token: Optional[str]) -> str:
    token = (cli_token or os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError(
            "HF token is required for gated DINOv3 access. "
            "Set HF_TOKEN env var or pass --hf_token."
        )
    os.environ["HF_TOKEN"] = token
    return token


def _display_name_from_args(args: argparse.Namespace) -> str:
    """
    Harvest may pass the Patra title only via MODEL_NAME (shell-safe).
    Legacy launchers may pass unquoted multi-token --model_name; join those.
    Env wins when set so backend-only MODEL_NAME matches steps.py payloads.
    """
    env_name = (os.environ.get("MODEL_NAME") or "").strip()
    if env_name:
        return env_name

    mn = getattr(args, "model_name", None)
    if isinstance(mn, list):
        return " ".join(mn).strip()
    return (mn or "").strip()


def main():
    _require_transformers_456()

    parser = argparse.ArgumentParser(description="DINOv3 crop-based classifier")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--input_images_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_images_dir", required=True)
    parser.add_argument("--class_configs", required=True, help="base64 encoded JSON list")

    parser.add_argument(
        "--model_name",
        nargs="+",
        default=None,
        help=(
            "Optional Patra/Harvest card title (multi-token OK if shell did not quote). "
            "If omitted, MODEL_NAME env must be set (preferred for Tapis jobs)."
        ),
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional HF model id override (same as before; skips name resolution).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HF token (prefer env HF_TOKEN)",
    )

    args = parser.parse_args()

    try:
        class_cfg_json = base64.b64decode(args.class_configs).decode("utf-8")
        class_configs = json.loads(class_cfg_json)
    except Exception as exc:
        raise ValueError(f"Failed to decode class_configs: {exc}")

    if not class_configs:
        raise ValueError("class_configs must be a non-empty list")

    display_name = _display_name_from_args(args)
    model_id = resolve_hf_model_id(display_name, args.model_id)

    print("DINOv3 Object Classifier")
    print("=" * 50)
    print(f"Model name: {display_name!r}")
    print(f"HF model id: {model_id}")
    print(f"Input JSON: {args.input_json}")
    print(f"Input images directory: {args.input_images_dir}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output images directory: {args.output_images_dir}")

    for idx, cfg in enumerate(class_configs, 1):
        print(f"\nClass {idx}: {cfg.get('class_name', 'Unknown')}")
        print(f"  Source: {cfg.get('source_image_path', 'N/A')}")
        print(f"  Crop: {cfg.get('crop_coordinates')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    cache_dir = prepare_cache_dir()
    print(f"\nHF cache: {cache_dir}")

    hf_token = require_hf_token(args.hf_token)

    print(f"\nLoading model & processor: {model_id}")
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print("Model loaded successfully")

    class_embeddings = load_class_embeddings(class_configs, args.input_images_dir, model, processor, device)
    grouped, _ = load_detector_json(args.input_json)

    results = classify_objects(
        class_embeddings,
        grouped,
        args.input_images_dir,
        model,
        processor,
        device,
        args.output_images_dir,
    )

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output_json}")
    print(f"Total objects classified: {len(results)}")

    summary = {}
    for obj in results:
        label = obj.get("label", "Unknown")
        summary[label] = summary.get(label, 0) + 1

    print("\nClassification Summary:")
    for label, count in summary.items():
        print(f" {label}: {count}")

    if args.output_images_dir:
        print(f"\nImages saved to: {args.output_images_dir}")
        for class_name in class_embeddings.keys():
            class_dir = os.path.join(args.output_images_dir, class_name)
            if os.path.exists(class_dir):
                image_count = len(
                    [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))]
                )
                print(f"  {class_name}: {image_count} images")

    print("\nClassification complete!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
