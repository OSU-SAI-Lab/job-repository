import torch
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision.ops import nms
import warnings
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np

def load_model_and_processor(model_name: str, device: torch.device, torch_data_type: torch.dtype):
    processor = Owlv2Processor.from_pretrained(model_name)
    model = (
        Owlv2ForObjectDetection
        .from_pretrained(model_name, dtype = torch_data_type, device_map="auto" if device == "cuda" else "cpu")
    ).eval()
    return processor, model

def image_guided_object_detection(
    processor,
    model,
    device,
    torch_data_type,
    class_supports: dict,
    object_features: dict,
    batches=None,
    np_img=None,
    use_sahi: bool = False,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,   
    nms_iou_threshold: float = 0.5,
):
    # if not isinstance(object_features, (tuple, list)) or len(object_features) != 3:
    #     raise ValueError("object_features must be a tuple or list of (feats, boxes, obj_scores)")
    feats = object_features['features']
    boxes = object_features['boxes']
    obj_scores = object_features['scores']
    if feats is None:
        return []

    obj_keep = obj_scores.float() > objectness_threshold
    if obj_keep.sum() > 100:
        topk = 100
        _, idxs = obj_scores.topk(topk)
        feats = feats[idxs]
        boxes = boxes[idxs]
        obj_scores = obj_scores[idxs]
        
    print(f"Features: {feats}")
    print(f"Boxes: {boxes}")
    print(f"Objectness Scores: {obj_scores}")    

    detections = []
    M, D = feats.shape
    potential_feats = feats.unsqueeze(0)            # (1, M, D)

    for class_name, support_embs in class_supports.items():
        # ensure (S, D)
        if support_embs.dim() == 1:
            support_embs = support_embs.unsqueeze(0)
        query_embeds = support_embs.unsqueeze(0)  # (1, S, D)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred_logits, _ = model.class_predictor(
                potential_feats,
                query_embeds=query_embeds,
                query_mask=None
            )
        logits = pred_logits.squeeze(0)

        logits_fp32   = logits.float()

        max_logits, _ = logits_fp32.max(dim=1)    # (M,)
        sim_scores    = torch.sigmoid(max_logits) # (M,)
        # print(sim_scores)

        keep = (obj_scores.float() > objectness_threshold) & \
                (sim_scores          > similarity_threshold)
        idxs = torch.nonzero(keep, as_tuple=True)[0]

        for i in idxs.tolist():
            detections.append({
                "bounding_box":  boxes[i].tolist(),
                "score": float(sim_scores[i]),   # use sim_scores, not probs
                "class": class_name
            })

    if not detections:
        return []

    boxes_tensor  = torch.tensor([d["bounding_box"] for d in detections], dtype=torch.float32, device=device)
    scores_tensor = torch.tensor([d["score"] for d in detections], dtype=torch.float32, device=device)
    keep_inds = nms(boxes_tensor, scores_tensor, iou_threshold=nms_iou_threshold)

    return [detections[i] for i in keep_inds.tolist()]


def cosine_similarity_detection(
    class_supports: dict,
    object_features: dict,
    device,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
) -> list:
    """
    Embedding-only detection via cosine similarity.
    Used for BioCLIP and DINOv3 backends where there is no class_predictor.
    """
    feats      = object_features['features']
    boxes      = object_features['boxes']
    obj_scores = object_features['scores']
    
    for cn, ce in class_supports.items():
        print(f"    '{cn}': {ce.shape}")
    
    if feats is None:
        return []

    # Objectness filtering (keep top-100)
    obj_keep = obj_scores.float() > objectness_threshold
    if obj_keep.sum() > 100:
        _, idxs = obj_scores.topk(100)
        feats      = feats[idxs]
        boxes      = boxes[idxs]
        obj_scores = obj_scores[idxs]

    # L2-normalise proposal features  (M, D)
    feats_norm = F.normalize(feats.float(), p=2, dim=-1)
    feat_dim = feats_norm.shape[-1]
    print(f"Proposal features: {feats.shape} → {feats_norm.shape}")
    
    detections = []
    for class_name, support_embs in class_supports.items():
        
        if support_embs.dim() == 1:
            support_embs = support_embs.unsqueeze(0)
        
        support_embs = support_embs.float().to(device)
        
        # Validate and fix dimension mismatch
        support_dim = support_embs.shape[-1]
        
        if support_dim != feat_dim:
            
            # If support is (D, S) instead of (S, D), transpose it
            if support_embs.shape[0] == feat_dim:
                print(f"       Transposing (detected D,S format)...")
                support_embs = support_embs.T
                support_dim = support_embs.shape[-1]
            
        
        support_norm = F.normalize(support_embs, p=2, dim=-1)  # (S, D)

        # (M, D) @ (D, S) → (M, S) cosine similarities → take max over supports
        sim_matrix = feats_norm @ support_norm.T
        sim_scores, _ = sim_matrix.max(dim=1)   # (M,)

        keep = (obj_scores.float() > objectness_threshold) & (sim_scores > similarity_threshold)
        for i in torch.nonzero(keep, as_tuple=True)[0].tolist():
            detections.append({
                "bounding_box": boxes[i].tolist(),
                "score":        float(sim_scores[i]),
                "class":        class_name,
            })

    if not detections:
        return []

    boxes_t  = torch.tensor([d["bounding_box"] for d in detections], dtype=torch.float32, device=device)
    scores_t = torch.tensor([d["score"]        for d in detections], dtype=torch.float32, device=device)
    keep_inds = nms(boxes_t, scores_t, iou_threshold=nms_iou_threshold)
    return [detections[i] for i in keep_inds.tolist()]


def run_detection_for_backend(
    backend: str,
    class_supports: dict,
    object_features: dict,
    device,
    owlv2_model_bundle=None,
    torch_data_type=None,
    objectness_threshold: float = 0.1,
    similarity_threshold: float = 0.2,
    nms_iou_threshold: float = 0.5,
) -> list:
    """
    Dispatch to the correct similarity function for the given backend.

    owlv2   → image_guided_object_detection  (uses model.class_predictor)
    bioclip → cosine_similarity_detection
    dinov3  → cosine_similarity_detection
    """
    if backend == "owlv2":
        if owlv2_model_bundle is None:
            raise ValueError("owlv2_model_bundle (processor, model) required for owlv2 backend")
        processor, model = owlv2_model_bundle
        return image_guided_object_detection(
            processor=processor,
            model=model,
            device=device,
            torch_data_type=torch_data_type,
            class_supports=class_supports,
            object_features=object_features,
            objectness_threshold=objectness_threshold,
            similarity_threshold=similarity_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
    elif backend in ("bioclip", "dinov3"):
        return cosine_similarity_detection(
            class_supports=class_supports,
            object_features=object_features,
            device=device,
            objectness_threshold=objectness_threshold,
            similarity_threshold=similarity_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: owlv2, bioclip, dinov3")