import torch
import numpy as np
import os
import math
import uvicorn
import requests
import time
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union
from contextlib import asynccontextmanager
from PIL import Image
import redis
import pickle
import io
import cv2

# --- HUGGING FACE IMPORTS ---
from transformers import Sam3Model, Sam3Processor
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

# --- CONFIGURATION ---
TAPIS_BASE_URL = "https://icicleai.tapis.io"
MODEL_ID = "facebook/sam3"
is_cuda = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda}")
DEVICE = "cuda" if is_cuda else "cpu"

# --- GLOBAL STATE ---
sam3_model = None
sam3_processor = None
sam3_tracker_model = None
sam3_tracker_processor = None

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    decode_responses=False,
    socket_timeout=5,
    ssl=False
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sam3_model, sam3_processor, sam3_tracker_model, sam3_tracker_processor
    print(f"🚀 Initializing SAM 3 Models on {DEVICE}...")
    
    try:
        print("Loading Sam3Model (Concept Segmentation)...")
        sam3_model = Sam3Model.from_pretrained(MODEL_ID).to(DEVICE)
        sam3_processor = Sam3Processor.from_pretrained(MODEL_ID)
        
        print("Loading Sam3TrackerModel (Visual Segmentation)...")
        sam3_tracker_model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(DEVICE)
        sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
        
        print("✅ Both Models Loaded Successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e
    
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UPDATED DATA MODELS ---

class SegmentationRequest(BaseModel):
    image_id: str
    pipe_id: str
    system_id: str  
    image_path: str 
    text_prompts: list[str] | None = None
    x: int | None = None  # Pipe syntax for Optional
    y: int | None = None
    patch_size: int | None = None
    crop_size: int | None = None
    overlap_ratio: float = 0.2
    threshold: float = 0.1
    mask_threshold: float = 0.1

    @field_validator('text_prompts')
    @classmethod
    def validate_text_prompts(cls, v):
        if v is not None:
            # Filter out empty strings
            v = [prompt.strip() for prompt in v if prompt and prompt.strip()]
            if len(v) == 0:
                return None
        return v

    @field_validator('patch_size', 'crop_size')
    @classmethod
    def validate_patch_sizes(cls, v):
        if v is not None and v <= 0:
            raise ValueError("patch_size/crop_size must be > 0")
        return v

    @field_validator('overlap_ratio')
    @classmethod
    def validate_overlap_ratio(cls, v):
        if v < 0 or v >= 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        return v

    @field_validator('threshold', 'mask_threshold')
    @classmethod
    def validate_thresholds(cls, v):
        if v < 0 or v > 1:
            raise ValueError("threshold/mask_threshold must be in [0, 1]")
        return v

    @model_validator(mode='after')
    def validate_patch_aliases(self):
        if self.patch_size is not None and self.crop_size is not None and self.patch_size != self.crop_size:
            raise ValueError("If both patch_size and crop_size are provided, they must be equal")
        return self

    def get_effective_patch_size(self) -> int | None:
        return self.patch_size if self.patch_size is not None else self.crop_size
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_123",
                "pipe_id": "pipe_456",
                "system_id": "tapis_system",
                "image_path": "/path/to/image.jpg",
                "text_prompts": ["red bottle", "person", "chair"],
                "patch_size": 960,
                "overlap_ratio": 0.2
            }
        }

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str  # The specific text prompt that found this object
    prompt_index: int  # Index in the text_prompts array
    segmentation: List[List[int]] = Field(default_factory=list)  # [[x, y], ...] boundary points

class SegmentationResponse(BaseModel):
    bboxes: List[BoundingBox]
    prompt_type: str  # "text" or "point"
    prompt_count: int  # Number of prompts processed (for text mode)
    total_detections: int
    model: str
    total_time_seconds: float
    per_prompt_breakdown: Optional[dict] = None  # Stats per prompt

class LegacyPointRequest(BaseModel):
    """Legacy single-point request for backwards compatibility"""
    image_id: str
    pipe_id: str
    system_id: str  
    image_path: str 
    x: int
    y: int

# --- HELPER FUNCTIONS ---

def fetch_image_from_tapis(system_id: str, path: str, token: str) -> Image.Image:
    """Downloads image from Tapis directly into memory."""
    clean_path = path.lstrip("/")
    url = f"{TAPIS_BASE_URL}/v3/files/content/{system_id}/{clean_path}"
    headers = {"X-Tapis-Token": token}
    
    print(f"⬇️ Downloading: {clean_path}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Tapis Error: {resp.text}")
        
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

async def encode_and_store_tracker_embeddings(image_id: str, pipe_id: str, raw_image: Image):
    """Store embeddings for point-based segmentation."""
    inputs = sam3_tracker_processor(images=raw_image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        embeddings = sam3_tracker_model.get_image_embeddings(inputs["pixel_values"])
        if isinstance(embeddings, list):
            bf16_embeddings = [e.to(device="cpu", dtype=torch.bfloat16) for e in embeddings]
        else:
            bf16_embeddings = embeddings.to(device="cpu", dtype=torch.bfloat16)
    
    data = {
        "embeddings": bf16_embeddings,
        "original_sizes": (raw_image.height, raw_image.width)
    }
    embeddings_bytes = pickle.dumps(data)
    r.set(f"sam3:emb:{pipe_id}_{image_id}", embeddings_bytes, ex=3600)

def get_center_crop_box(img_w: int, img_h: int, center_x: int, center_y: int, patch_size: int):
    """Returns (x1, y1, x2, y2) for a center crop clamped to image bounds."""
    crop_w = min(patch_size, img_w)
    crop_h = min(patch_size, img_h)

    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(img_w, x1 + crop_w)
    y2 = min(img_h, y1 + crop_h)

    # snap back to keep exact crop size when possible
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)
    return int(x1), int(y1), int(x2), int(y2)

def build_overlapping_tiles(img_w: int, img_h: int, tile_size: int, overlap_ratio: float):
    """Generate overlapping (x1, y1, x2, y2) tile coords."""
    tile_size = max(1, int(tile_size))
    stride = max(1, int(tile_size * (1 - overlap_ratio)))

    cols = max(1, math.ceil((img_w - tile_size) / stride) + 1)
    rows = max(1, math.ceil((img_h - tile_size) / stride) + 1)

    tiles = []
    seen = set()
    for r_i in range(rows):
        for c_i in range(cols):
            x1 = c_i * stride
            y1 = r_i * stride
            x2 = min(x1 + tile_size, img_w)
            y2 = min(y1 + tile_size, img_h)

            if x2 == img_w:
                x1 = max(0, img_w - tile_size)
            if y2 == img_h:
                y1 = max(0, img_h - tile_size)

            box = (int(x1), int(y1), int(x2), int(y2))
            if box not in seen:
                seen.add(box)
                tiles.append(box)
    return tiles

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def mask_to_boundary_points(mask: np.ndarray, max_points: int = 256) -> list[list[int]]:
    """Extract ordered contour points from a binary mask."""
    if mask is None or not np.any(mask):
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    # Pick the largest contour (the main object boundary)
    contour = max(contours, key=cv2.contourArea)
    points = contour.squeeze(axis=1)  # (N,1,2) → (N,2)
    if len(points) > max_points:
        indices = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
        points = points[indices]
    return [[int(p[0]), int(p[1])] for p in points]

def nms_bbox_candidates(candidates: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Simple NMS over candidates with keys: x_min,y_min,x_max,y_max,confidence."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept = []

    for cand in sorted_cands:
        cand_box = (cand["x_min"], cand["y_min"], cand["x_max"], cand["y_max"])
        should_keep = True
        for k in kept:
            k_box = (k["x_min"], k["y_min"], k["x_max"], k["y_max"])
            if iou_xyxy(cand_box, k_box) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(cand)
    return kept

def run_text_inference_on_image(raw_image: Image.Image, prompt: str, threshold: float = 0.1, mask_threshold: float = 0.1):
    """Runs SAM3 concept model on one image and one prompt. Returns list[(x1,y1,x2,y2,score)]."""
    inputs = sam3_processor(
        images=raw_image,
        text=prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    target_sizes = inputs.get("original_sizes").tolist()
    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes
    )[0]

    out = []
    if len(results["masks"]) > 0:
        for i in range(len(results["masks"])):
            box = results["boxes"][i]
            score = float(results["scores"][i].item())
            x_min, y_min, x_max, y_max = box.tolist()
            mask = results["masks"][i]
            mask_np = mask.cpu().numpy() > 0 if hasattr(mask, "numpy") else np.array(mask) > 0
            seg_points = mask_to_boundary_points(mask_np)
            out.append((int(x_min), int(y_min), int(x_max), int(y_max), score, seg_points))

    del inputs, outputs
    torch.cuda.empty_cache()
    return out

def run_point_inference_on_image(raw_image: Image.Image, x: int, y: int):
    """Runs tracker model on one image and one point. Returns (x1,y1,x2,y2,score) or None."""
    image_inputs = sam3_tracker_processor(images=raw_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        image_embeddings = sam3_tracker_model.get_image_embeddings(image_inputs["pixel_values"])

    input_points = [[[[x, y]]]]
    input_labels = [[[1]]]

    prompt_inputs = sam3_tracker_processor(
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
        original_sizes=[(raw_image.height, raw_image.width)]
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sam3_tracker_model(**prompt_inputs, image_embeddings=image_embeddings)

    masks = sam3_tracker_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        prompt_inputs["original_sizes"].cpu()
    )[0]

    scores = outputs.iou_scores.cpu()[0, 0]
    best_idx = torch.argmax(scores).item()
    best_mask = masks[0][best_idx].numpy() > 0

    y_indices, x_indices = np.where(best_mask)
    if len(x_indices) == 0:
        return None

    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    seg_points = mask_to_boundary_points(best_mask)
    return x_min, y_min, x_max, y_max, float(scores[best_idx]), seg_points

# --- MAIN ENDPOINT ---

@app.post("/predict", response_model=SegmentationResponse)
async def predict(
    req: SegmentationRequest, 
    token: str = Header(None, alias="token") 
):
    start_time = time.time()
    
    # Validation: Must have either text prompts or point coordinates
    has_text = req.text_prompts is not None and len(req.text_prompts) > 0
    has_point = req.x is not None and req.y is not None
    
    if not has_text and not has_point:
        raise HTTPException(
            status_code=400, 
            detail="Either text_prompts (array) or point coordinates (x,y) must be provided"
        )
    
    if has_text and has_point:
        raise HTTPException(
            status_code=400,
            detail="Cannot provide both text_prompts and point coordinates in one request. Call separately."
        )
    
    try:
        if has_text:
            return await predict_with_text_batch(req, token, start_time)
        else:
            return await predict_with_point(req, token, start_time)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

async def predict_with_text_batch(req: SegmentationRequest, token: str, start_time: float):
    """
    Process multiple text prompts on the same image.
    Each prompt is processed separately and results are aggregated.
    """
    global sam3_model, sam3_processor
    
    if not token:
        raise HTTPException(status_code=400, detail="Token required to fetch image")
    
    # Load image once
    step_start = time.time()
    raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)
    load_time = time.time() - step_start
    print(f"Image Load: {load_time:.4f}s")

    patch_size = req.get_effective_patch_size()
    if patch_size is not None:
        tile_coords = build_overlapping_tiles(raw_image.width, raw_image.height, patch_size, req.overlap_ratio)
        print(f"Using tiled text inference: {len(tile_coords)} tiles, patch_size={patch_size}, overlap_ratio={req.overlap_ratio}")
    else:
        tile_coords = [(0, 0, raw_image.width, raw_image.height)]
    
    all_bboxes = []
    per_prompt_stats = {}
    
    # Process each text prompt
    for idx, prompt in enumerate(req.text_prompts):
        prompt_start = time.time()
        print(f"\n🔍 Processing prompt {idx+1}/{len(req.text_prompts)}: '{prompt}'")
        
        try:
            candidates = []
            for x1, y1, x2, y2 in tile_coords:
                tile_img = raw_image.crop((x1, y1, x2, y2))
                tile_dets = run_text_inference_on_image(tile_img, prompt, req.threshold, req.mask_threshold)
                for bx1, by1, bx2, by2, score, seg_pts in tile_dets:
                    candidates.append({
                        "x_min": int(bx1 + x1),
                        "y_min": int(by1 + y1),
                        "x_max": int(bx2 + x1),
                        "y_max": int(by2 + y1),
                        "confidence": round(score, 4),
                        "label": prompt,
                        "prompt_index": idx,
                        "segmentation": [[px + x1, py + y1] for px, py in seg_pts]
                    })

            merged = nms_bbox_candidates(candidates, iou_threshold=0.5)
            prompt_bboxes = [BoundingBox(**b) for b in merged]
            all_bboxes.extend(prompt_bboxes)
            
            prompt_time = time.time() - prompt_start
            per_prompt_stats[prompt] = {
                "detections": len(prompt_bboxes),
                "raw_detections": len(candidates),
                "time_seconds": round(prompt_time, 4)
            }
            print(f"   Found {len(prompt_bboxes)} objects in {prompt_time:.4f}s")
            torch.cuda.empty_cache()

        except Exception as e:
            torch.cuda.empty_cache()
            print(f"   ❌ Error processing prompt '{prompt}': {e}")
            per_prompt_stats[prompt] = {"error": str(e), "detections": 0}
    
    total_time = time.time() - start_time
    
    return SegmentationResponse(
        bboxes=all_bboxes,
        prompt_type="text",
        prompt_count=len(req.text_prompts),
        total_detections=len(all_bboxes),
        model="SAM3-Concept",
        total_time_seconds=round(total_time, 4),
        per_prompt_breakdown=per_prompt_stats
    )

async def predict_with_point(req: SegmentationRequest, token: str, start_time: float):
    """Handle point-based segmentation using Sam3TrackerModel."""
    global sam3_tracker_model, sam3_tracker_processor

    patch_size = req.get_effective_patch_size()
    if patch_size is not None:
        if not token:
            raise HTTPException(status_code=400, detail="Token required for crop-based point inference")

        raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)
        if req.x < 0 or req.y < 0 or req.x >= raw_image.width or req.y >= raw_image.height:
            raise HTTPException(status_code=400, detail="Point coordinates are out of image bounds")

        x1, y1, x2, y2 = get_center_crop_box(raw_image.width, raw_image.height, req.x, req.y, patch_size)
        cropped = raw_image.crop((x1, y1, x2, y2))

        local_x = req.x - x1
        local_y = req.y - y1
        point_result = run_point_inference_on_image(cropped, local_x, local_y)

        if point_result is None:
            return SegmentationResponse(
                bboxes=[],
                prompt_type="point",
                prompt_count=0,
                total_detections=0,
                model="SAM3-Tracker",
                total_time_seconds=round(time.time() - start_time, 4)
            )

        bx1, by1, bx2, by2, score, seg_pts = point_result
        bbox = BoundingBox(
            x_min=int(bx1 + x1),
            y_min=int(by1 + y1),
            x_max=int(bx2 + x1),
            y_max=int(by2 + y1),
            confidence=score,
            label=f"point({req.x},{req.y})",
            prompt_index=0,
            segmentation=[[px + x1, py + y1] for px, py in seg_pts]
        )

        return SegmentationResponse(
            bboxes=[bbox],
            prompt_type="point",
            prompt_count=1,
            total_detections=1,
            model="SAM3-Tracker",
            total_time_seconds=round(time.time() - start_time, 4)
        )
    
    # Try to load cached embeddings
    embedding_bytes = r.get(f"sam3:emb:{req.pipe_id}_{req.image_id}")
    
    if embedding_bytes:
        try:
            loaded_data = pickle.loads(embedding_bytes)
            if not isinstance(loaded_data, dict) or "original_sizes" not in loaded_data:
                embedding_bytes = None
        except:
            embedding_bytes = None
    
    if embedding_bytes is None:
        if not token:
            raise HTTPException(status_code=400, detail="Image missing from cache and no Token provided")
            
        step_start = time.time()
        raw_image = fetch_image_from_tapis(req.system_id, req.image_path, token)
        print(f"⏱️ Load Image: {time.time() - step_start:.4f}s")
        
        await encode_and_store_tracker_embeddings(req.image_id, req.pipe_id, raw_image)
        embedding_bytes = r.get(f"sam3:emb:{req.pipe_id}_{req.image_id}")
    
    loaded_data = pickle.loads(embedding_bytes)
    loaded_embeddings = loaded_data["embeddings"]
    original_sizes = loaded_data["original_sizes"]
    
    if isinstance(loaded_embeddings, list):
        image_embeddings = [e.to(DEVICE) for e in loaded_embeddings]
    else:
        image_embeddings = loaded_embeddings.to(DEVICE)
    
    # Prepare point inputs
    input_points = [[[[req.x, req.y]]]]
    input_labels = [[[1]]]
    
    inputs = sam3_tracker_processor(
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt",
        original_sizes=[original_sizes]
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = sam3_tracker_model(**inputs, image_embeddings=image_embeddings)
    
    # Post-process
    masks = sam3_tracker_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu()
    )[0]
    
    scores = outputs.iou_scores.cpu()[0, 0]
    best_idx = torch.argmax(scores).item()
    best_mask = masks[0][best_idx].numpy() > 0
    
    # Extract bbox
    y_indices, x_indices = np.where(best_mask)
    if len(x_indices) == 0:
        return SegmentationResponse(
            bboxes=[],
            prompt_type="point",
            prompt_count=0,
            total_detections=0,
            model="SAM3-Tracker",
            total_time_seconds=round(time.time() - start_time, 4)
        )

    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

    bbox = BoundingBox(
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        confidence=float(scores[best_idx]),
        label=f"point({req.x},{req.y})",
        prompt_index=0,
        segmentation=mask_to_boundary_points(best_mask)
    )
    
    total_time = time.time() - start_time
    
    return SegmentationResponse(
        bboxes=[bbox],
        prompt_type="point",
        prompt_count=1,
        total_detections=1,
        model="SAM3-Tracker",
        total_time_seconds=round(total_time, 4)
    )

# Legacy endpoint for backwards compatibility
@app.post("/predict_box")
async def predict_box_legacy(
    req: LegacyPointRequest, 
    token: str = Header(None, alias="token") 
):
    """Legacy endpoint for single point prompts."""
    new_req = SegmentationRequest(
        image_id=req.image_id,
        pipe_id=req.pipe_id,
        system_id=req.system_id,
        image_path=req.image_path,
        x=req.x,
        y=req.y,
        text_prompts=None
    )
    result = await predict(new_req, token)
    return {
        "bbox": result.bboxes[0].dict() if result.bboxes else None,
        "confidence": result.bboxes[0].confidence if result.bboxes else 0,
        "model": result.model,
        "total_time_seconds": result.total_time_seconds
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2128)