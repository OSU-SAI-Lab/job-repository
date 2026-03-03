import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os


# ──────────────────────────────────────────────────────────────────────────────
# IoU-tracking utility (used by the crop-size search loop in main)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_change_in_iou(patch_size: int, generated_boxes: list, map: dict) -> tuple:
    """Track average IoU across crop-size iterations (OWLv2 legacy use)."""
    average_iou = 0.0
    if len(generated_boxes) > 0:
        iou_values = [box.get('iou', 0.0) for box in generated_boxes]
        average_iou = sum(iou_values) / len(iou_values)
    sorted_patch_sizes = sorted(map, reverse=True)
    previous_best_iou = map.get(sorted_patch_sizes[-1] if len(sorted_patch_sizes) > 0 else 0.0, 0.0)
    next_iter = previous_best_iou == 0 or (
        average_iou > previous_best_iou
        and abs(average_iou - previous_best_iou) / previous_best_iou > 0.1
    )
    map[patch_size] = round(average_iou, 3)
    return next_iter, map


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_OWLV2_MODEL   = "google/owlv2-large-patch14-ensemble"
DEFAULT_DINOV3_MODEL  = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def load_owlv2_model(model_name: str, device, torch_dtype):
    """Load OWLv2 processor + model."""
    processor = Owlv2Processor.from_pretrained(model_name)
    model = (
        Owlv2ForObjectDetection
        .from_pretrained(model_name, torch_dtype=torch_dtype)
        .to(device)
        .eval()
    )
    return processor, model


# Backward-compat alias
load_model_and_processor = load_owlv2_model


def load_bioclip_model(device):
    """Load BioCLIP BaseClassifier."""
    from bioclip.predict import BaseClassifier
    return BaseClassifier(device=device)


def load_dinov3_model(model_name: str, device):
    """Load DINOv3 processor + model."""
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


def load_embedding_model(backend: str, device, torch_dtype=None, model_name: str = None):
    """
    Factory: load the model/processor bundle for a given embedding backend.

    Returns:
        owlv2  → (Owlv2Processor, Owlv2ForObjectDetection)
        bioclip → BaseClassifier
        dinov3  → (AutoImageProcessor, AutoModel)
    """
    if backend == "owlv2":
        return load_owlv2_model(
            model_name or DEFAULT_OWLV2_MODEL,
            device,
            torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32),
        )
    elif backend == "bioclip":
        return load_bioclip_model(device)
    elif backend == "dinov3":
        return load_dinov3_model(model_name or DEFAULT_DINOV3_MODEL, device)
    else:
        raise ValueError(f"Unknown embedding backend '{backend}'. Choose from: owlv2, bioclip, dinov3")


# ──────────────────────────────────────────────────────────────────────────────
# Per-backend embedding extractors
# Each accepts a PIL crop of the exact GT bounding box and returns a 1-D
# torch.Tensor on the appropriate device.
# ──────────────────────────────────────────────────────────────────────────────

def get_owlv2_embedding_from_crop(crop: Image.Image, processor, model, device) -> torch.Tensor:
    """
    Embed a GT-box crop with OWLv2.
    Passes the crop through OWLv2's image embedder and mean-pools the patch
    class embeddings into a single vector.
    """
    pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        fmap = model.image_embedder(pv)[0]           # (1, H', W', C_hidden)
        B, Hf, Wf, C_h = fmap.shape
        feats = fmap.reshape(B, Hf * Wf, C_h)        # (1, P, C_hidden)
        _, cls_emb = model.class_predictor(feats)    # (1, P, C_class)
    return cls_emb[0].mean(dim=0)                    # (C_class,)


def get_bioclip_embedding_from_crop(crop: Image.Image, classifier) -> torch.Tensor:
    """
    Embed a GT-box crop with BioCLIP.
    Returns an L2-normalised feature vector.
    """
    classifier.model.eval()
    with torch.no_grad():
        emb = classifier.create_image_features([crop], normalize=True)  # (1, D)
    return emb[0]


def get_dinov3_embedding_from_crop(crop: Image.Image, processor, model) -> torch.Tensor:
    """
    Embed a GT-box crop with DINOv3.
    Returns an L2-normalised pooler output.
    """
    inputs = processor(images=crop, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        emb = F.normalize(outputs.pooler_output, p=2, dim=-1)  # (1, D)
    return emb[0]


# ──────────────────────────────────────────────────────────────────────────────
# Core: extract class-support embeddings from GT annotations
# ──────────────────────────────────────────────────────────────────────────────

def extract_support_embeddings(
    support_examples: list,
    embedding_backend: str,
    model_bundle,
    device: str,
    src_path: str = None,
) -> tuple:
    """
    For every support example, crop the exact GT bounding box from the image
    and embed it with the chosen backend.

    Args:
        support_examples:  list of {"image_path", "bounding_box", "class"} dicts
        embedding_backend: "owlv2" | "bioclip" | "dinov3"
        model_bundle:      return value of load_embedding_model()
        device:            torch device string

    Returns:
        class_supports:  dict[class_name → torch.Tensor (N, D)]
        generated_boxes: list of per-example annotation dicts
    """
    class_supports  = {}
    generated_boxes = []

    for support in support_examples:
        img_path   = os.path.join(src_path, support['image_path'])
        bbox       = support['bounding_box']   # [x_min, y_min, x_max, y_max]
        class_name = support['class']

        img  = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Crop exactly the GT bounding box (clamped to image bounds)
        x1 = max(0, min(int(bbox[0]), W - 1))
        y1 = max(0, min(int(bbox[1]), H - 1))
        x2 = max(0, min(int(bbox[2]), W))
        y2 = max(0, min(int(bbox[3]), H))
        crop = img.crop((x1, y1, x2, y2))

        # Embed with the chosen backend
        if embedding_backend == "owlv2":
            processor, model = model_bundle
            emb = get_owlv2_embedding_from_crop(crop, processor, model, device)
        elif embedding_backend == "bioclip":
            emb = get_bioclip_embedding_from_crop(crop, model_bundle)
        elif embedding_backend == "dinov3":
            processor, model = model_bundle
            emb = get_dinov3_embedding_from_crop(crop, processor, model)
        else:
            raise ValueError(f"Unknown embedding backend '{embedding_backend}'")

        emb = emb.to(device).float()

        generated_boxes.append({
            'image_path':    support['image_path'],
            'bounding_box':  [x1, y1, x2, y2],
            'class':         class_name,
        })

        if class_name not in class_supports:
            class_supports[class_name] = []
        class_supports[class_name].append(emb)

    for c in class_supports:
        class_supports[c] = torch.stack(class_supports[c], dim=0)  # (N, D)

    return class_supports, generated_boxes