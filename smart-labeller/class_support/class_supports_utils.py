import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os
from scipy.special import expit


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

def get_best_patch_embedding(
    crop: Image.Image,
    processor,
    model,
    device: str,
    ref_box_in_crop: tuple
):
    pv = processor(images=crop, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        fmap = model.image_embedder(pv)[0]                   # (1, H', W', C_hidden)
        B, Hf, Wf, C_h = fmap.shape
        feats = fmap.reshape(B, Hf*Wf, C_h)                  # (1, P, C_hidden)
        obj_logits = model.objectness_predictor(feats)[0]    # (P,)
        boxes_norm = model.box_predictor(feats, feature_map=fmap)[0]  # (P,4)
        _, cls_emb = model.class_predictor(feats)            # (1, P, C_class)

    scores = expit(obj_logits.cpu().numpy())                 # (P,)
    boxes_norm = boxes_norm.cpu().numpy()                    # (P,4)
    cls_emb = cls_emb[0].cpu().numpy()                       # (P, C_class)

    proposals = []
    for (cx, cy, bw, bh), score, emb in zip(boxes_norm, scores, cls_emb):
        xA = (cx - bw/2) * crop.width
        yA = (cy - bh/2) * crop.height
        xB = (cx + bw/2) * crop.width
        yB = (cy + bh/2) * crop.height
        proposals.append((xA, yA, xB, yB, score, emb))

    def compute_iou(boxA, boxB):
        xa1, ya1, xa2, ya2 = np.array(boxA, dtype=np.float64)
        xb1, yb1, xb2, yb2 = np.array(boxB, dtype=np.float64)

        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)

        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        inter = inter_w * inter_h

        areaA = (xa2 - xa1) * (ya2 - ya1)
        areaB = (xb2 - xb1) * (yb2 - yb1)

        union = areaA + areaB - inter
        if union <= 0 or not np.isfinite(union):
            return 0.0
        return inter / union

    ref = tuple(map(float, ref_box_in_crop))
    ious = np.array([compute_iou(ref, prop[:4]) for prop in proposals])
    best_idx = int(np.argmax(ious))
    best_xA, best_yA, best_xB, best_yB, _, best_emb = proposals[best_idx]

    best_box = (float(best_xA), float(best_yA), float(best_xB), float(best_yB))
    print("best_iou", ious[best_idx])
    return best_emb, best_box, ious[best_idx]


def load_bioclip_model(device):
    """Load BioCLIP BaseClassifier."""
    from bioclip.predict import BaseClassifier
    return BaseClassifier(device=device)


def load_dinov3_model(model_name: str, device):
    """Load DINOv3 processor + model."""
    from transformers import AutoImageProcessor, AutoModel, AutoConfig

    cfg = AutoConfig.from_pretrained(model_name)
    cfg_model_type = getattr(cfg, "model_type", "unknown")
    if not str(cfg_model_type).startswith("dinov3"):
        raise ValueError(
            f"Requested dinov3 backend, but model '{model_name}' resolves to model_type='{cfg_model_type}'. "
            "Use a DINOv3 model id (for example: facebook/dinov3-vitl16-pretrain-lvd1689m)."
        )

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    model_type = getattr(model.config, "model_type", "unknown")
    if not str(model_type).startswith("dinov3"):
        raise RuntimeError(
            f"Loaded model_type='{model_type}' for model '{model_name}', expected a DINOv3 model. "
            "This usually means an incompatible transformers version or an incorrect --model_name override."
        )
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
        return load_dinov3_model(DEFAULT_DINOV3_MODEL, device)
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
        outputs = model(pixel_values=inputs["pixel_values"])
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
    crop_size: int = 1024
) -> tuple:
    """
    For every support example, extract embeddings with the chosen backend.

    For OWLv2:
      - Creates a centered crop of size `crop_size` around GT bounding box center
      - Uses get_best_patch_embedding to find the best proposal in that crop
      - Extracts the embedding with highest IoU score to the GT box
      - Tracks IoU scores for the crop_size optimization loop

    For BioCLIP & DINOv3:
      - Extracts the exact GT bounding box crop
      - Generates embeddings directly from the exact crop

    Args:
        support_examples:  list of {"image_path", "bounding_box", "class"} dicts
        embedding_backend: "owlv2" | "bioclip" | "dinov3"
        model_bundle:      return value of load_embedding_model()
        device:            torch device string
        src_path:          base path to image files
        crop_size:         size of crop to extract (used for OWLv2 centered crop; other models use exact GT crop)
    Returns:
        class_supports:  dict[class_name → torch.Tensor (N, D)]
        generated_boxes: list of per-example annotation dicts with IoU scores
    """
    class_supports  = {}
    generated_boxes = []

    for support in support_examples:
        img_path   = os.path.join(src_path, support['image_path'])
        bbox       = support['bounding_box']   # [x_min, y_min, x_max, y_max]
        class_name = support['class']

        img  = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Embed with the chosen backend
        if embedding_backend == "owlv2":
            processor, model = model_bundle
            
            # Calculate GT bounding box center
            x1_gt = max(0, min(int(bbox[0]), W - 1))
            y1_gt = max(0, min(int(bbox[1]), H - 1))
            x2_gt = max(0, min(int(bbox[2]), W))
            y2_gt = max(0, min(int(bbox[3]), H))
            
            cx_gt = (x1_gt + x2_gt) / 2.0
            cy_gt = (y1_gt + y2_gt) / 2.0
            
            # Create centered crop of size `crop_size` around GT center
            crop_half = crop_size / 2.0
            crop_x1 = max(0, int(cx_gt - crop_half))
            crop_y1 = max(0, int(cy_gt - crop_half))
            crop_x2 = min(W, int(cx_gt + crop_half))
            crop_y2 = min(H, int(cy_gt + crop_half))
            
            # Adjust if crop goes out of bounds
            if crop_x2 - crop_x1 < crop_size:
                if crop_x1 == 0:
                    crop_x2 = min(W, crop_size)
                else:
                    crop_x1 = max(0, crop_x2 - crop_size)
            if crop_y2 - crop_y1 < crop_size:
                if crop_y1 == 0:
                    crop_y2 = min(H, crop_size)
                else:
                    crop_y1 = max(0, crop_y2 - crop_size)
            
            # Extract centered crop
            centered_crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            
            # Transform GT bounding box to centered crop coordinates
            ref_x1 = max(0, x1_gt - crop_x1)
            ref_y1 = max(0, y1_gt - crop_y1)
            ref_x2 = min(centered_crop.width, x2_gt - crop_x1)
            ref_y2 = min(centered_crop.height, y2_gt - crop_y1)
            ref_box_in_crop = (ref_x1, ref_y1, ref_x2, ref_y2)
            
            # Get best patch embedding using OWLv2
            # get_best_patch_embedding finds the proposal with highest IoU to ref_box
            best_emb, best_box, best_iou = get_best_patch_embedding(
                centered_crop, processor, model, device, ref_box_in_crop
            )
            emb = torch.from_numpy(best_emb).to(device).float()
            
            # Store GT box with IOU score for optimization loop
            generated_boxes.append({
                'image_path':    support['image_path'],
                'bounding_box':  [x1_gt, y1_gt, x2_gt, y2_gt],
                'class':         class_name,
                'iou':           float(best_iou),
            })
            
        elif embedding_backend == "bioclip":
            # Extract exact GT bounding box
            x1 = max(0, min(int(bbox[0]), W - 1))
            y1 = max(0, min(int(bbox[1]), H - 1))
            x2 = max(0, min(int(bbox[2]), W))
            y2 = max(0, min(int(bbox[3]), H))
            crop = img.crop((x1, y1, x2, y2))
            
            emb = get_bioclip_embedding_from_crop(crop, model_bundle)
            
            generated_boxes.append({
                'image_path':    support['image_path'],
                'bounding_box':  [x1, y1, x2, y2],
                'class':         class_name,
            })
            
        elif embedding_backend == "dinov3":
            # Extract exact GT bounding box
            x1 = max(0, min(int(bbox[0]), W - 1))
            y1 = max(0, min(int(bbox[1]), H - 1))
            x2 = max(0, min(int(bbox[2]), W))
            y2 = max(0, min(int(bbox[3]), H))
            crop = img.crop((x1, y1, x2, y2))
            
            processor, model = model_bundle
            emb = get_dinov3_embedding_from_crop(crop, processor, model)
            
            generated_boxes.append({
                'image_path':    support['image_path'],
                'bounding_box':  [x1, y1, x2, y2],
                'class':         class_name,
            })
            
        else:
            raise ValueError(f"Unknown embedding backend '{embedding_backend}'")

        emb = emb.to(device).float()

        if class_name not in class_supports:
            class_supports[class_name] = []
        class_supports[class_name].append(emb)

    for c in class_supports:
        class_supports[c] = torch.stack(class_supports[c], dim=0)  # (N, D)

    return class_supports, generated_boxes