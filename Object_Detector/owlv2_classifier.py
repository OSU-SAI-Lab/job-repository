#!/usr/bin/env python3
"""
Self-contained multi-model detector (no imports from other project scripts).

Select backend via display name only:
  --model_name ...  or environment variable MODEL_NAME

The title is matched with simple keyword rules to the same four backends as before:
  SAM3           -> Ultralytics SAM3 (internal index 4)
  OWLv2 / Owl v2 -> Hugging Face OWLv2 (internal index 1)
  Grounding+DINO -> Hugging Face Grounding DINO (internal index 2)
  YOLOE          -> Ultralytics YOLOE (internal index 3)

Either --model_name or MODEL_NAME must be non-empty.

Shared I/O: --input_folder, --output_folder, base64 --class_configs, etc.
Outputs match the original per-model scripts (overall_results.json, annotated/).

Container: install in your image definition, for example:
  pip install torch torchvision pillow  pip install transformers # OWLv2 + Grounding DINO
  pip install ultralytics   # YOLOE + SAM3 (versions per upstream docs)
Optional: bitsandbytes / accelerate for OWLv2 int8 if you use that code path.

Runtime weights: HF checkpoints for OWLv2 + Grounding DINO; YOLOE --weights for YOLOE; SAM3 always from Hub (token).
SAM3: weights always come from Hugging Face via hf_hub_download. Set HF_TOKEN or pass
--hf_token (accept the model license first). Prefer HF_TOKEN so the token is not visible in `ps`.
"""

import argparse
import base64
import glob
import hashlib
import importlib
import inspect
import json
import os
import re
import site
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Semaphore
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import Owlv2ForObjectDetection, Owlv2Processor
try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None
    _BITSANDBYTES_AVAILABLE = False

class Profiler:
    """Profiler for tracking timing of different operations"""
    def __init__(self):
        self.timings = defaultdict(list)
        self.gpu_events = {}
        self.enabled = True
        
    def start_timer(self, name: str):
        """Start CPU timer"""
        if not self.enabled:
            return None
        return time.time()
    
    def end_timer(self, name: str, start_time):
        """End CPU timer and record"""
        if not self.enabled or start_time is None:
            return
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        self.timings[name].append(elapsed)
        return elapsed
    
    def create_gpu_event(self, name: str):
        """Create GPU event for timing"""
        if not self.enabled or not torch.cuda.is_available():
            return None
        if name not in self.gpu_events:
            self.gpu_events[name] = {'start': torch.cuda.Event(enable_timing=True),
                                     'end': torch.cuda.Event(enable_timing=True)}
        return self.gpu_events[name]
    
    def record_gpu_time(self, name: str):
        """Record GPU time from event"""
        if not self.enabled or not torch.cuda.is_available():
            return None
        if name in self.gpu_events:
            elapsed = self.gpu_events[name]['start'].elapsed_time(self.gpu_events[name]['end'])
            self.timings[f"GPU_{name}"].append(elapsed)
            return elapsed
        return None
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get profiling summary statistics"""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'sum_ms': sum(times)
                }
        return summary
    
    def print_summary(self):
        """Print profiling summary"""
        if not self.enabled:
            return
        
        summary = self.get_summary()
        if not summary:
            return
        
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        print(f"{'Operation':<30} {'Count':<8} {'Total(ms)':<12} {'Avg(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12}")
        print("-"*80)
        
        # Sort by total time
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        for name, stats in sorted_ops:
            print(f"{name:<30} {stats['count']:<8} {stats['total_ms']:<12.2f} "
                  f"{stats['avg_ms']:<12.2f} {stats['min_ms']:<12.2f} {stats['max_ms']:<12.2f}")
        
        print("="*80)
        
        # Calculate percentages
        total_time = sum(s['total_ms'] for s in summary.values())
        print(f"\nTotal Time: {total_time:.2f}ms")
        print("\nTime Distribution:")
        for name, stats in sorted_ops[:10]:  # Top 10
            pct = (stats['total_ms'] / total_time * 100) if total_time > 0 else 0
            print(f"  {name:<30}: {pct:>6.2f}% ({stats['total_ms']:.2f}ms)")
        print("="*80)

class OWLv2Classifier:

    def __init__(self, input_folder, output_folder, confidence_threshold=0.1,
                 class_configs=None, preserve_metadata=True, num_workers=12, batch_size=1):

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.confidence_threshold = confidence_threshold
        self.class_configs = class_configs or []
        self.preserve_metadata = preserve_metadata
        self.num_workers = min(num_workers, 24)
        self.batch_size = batch_size
        self.use_amp = True
        self.use_compile = False
        self.use_int8 = False  # Disabled: hurts accuracy and speed for this model

        # Keep background-suppressing limits ON (as earlier)
        self.max_bbox_size = 700              # max side length (px) allowed
        self.nms_iou_threshold = 0.20         # overlap to suppress duplicates
        self.nms_expand_ratio = 0.10          # small context around kept box
        self.nms_max_coverage = 0.95          # reject near-full-image boxes

        # Use Semaphore instead of Lock for better structure
        # Keep at 1 concurrent batch (2 was slower due to memory contention)
        self.max_concurrent_batches = 1
        self._gpu_semaphore = Semaphore(self.max_concurrent_batches)
        self._gpu_monitor_running = False
        self._gpu_monitor_thread = None
        self._gpu_monitor_process = None
        self.gpu_log_file = None

        # CUDA Streams for overlapping transfer and computation
        self.use_cuda_streams = False
        self.transfer_stream = None
        self.compute_stream = None
        if torch.cuda.is_available():
            self.use_cuda_streams = True
            self.transfer_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
        
        # GPU-based preprocessing flag
        self.use_gpu_preprocessing = torch.cuda.is_available()
        if self.use_gpu_preprocessing:
            # Store preprocessing parameters from processor
            self._preprocessing_config = None

        # Profiler for performance analysis
        self.profiler = Profiler()
        
        # Preprocessing optimization: Cache for tokenized text
        self._tokenized_text_cache = None
        self._last_search_terms_hash = None
        
        os.makedirs(output_folder, exist_ok=True)

        print(f"Initializing OWLv2 Classifier")
        print(f"GPU: 1, CPU cores: {self.num_workers}, Batch size: {self.batch_size}")
        optimizations = ["Pinned Memory", "CUDA Streams", "Async Prefetching", "Text Tokenization Cache"]
        if self.use_gpu_preprocessing:
            optimizations.append("GPU Preprocessing")
        print(f"Optimizations: {', '.join(optimizations)}")
        self._load_model()

    def _load_model(self):
        print("Loading OWLv2 model with optimizations...")
        start_time = time.time()
        try:
            self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", use_fast=True)
            if self.use_int8 and _BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = Owlv2ForObjectDetection.from_pretrained(
                    "google/owlv2-base-patch16-ensemble",
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                self.use_amp = False  # Quantized model; skip autocast
                print("Model loaded in 8-bit (int8) via bitsandbytes")
            else:
                if self.use_int8 and not _BITSANDBYTES_AVAILABLE:
                    print("use_int8=True but bitsandbytes not installed; run: pip install bitsandbytes accelerate")
                self.use_int8 = False
                self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
            
            # Initialize GPU preprocessing config
            if self.use_gpu_preprocessing and hasattr(self.processor, 'image_processor'):
                img_proc = self.processor.image_processor
                self._preprocessing_config = {
                    'target_size': (img_proc.size.get('height', 960), img_proc.size.get('width', 960)),
                    'mean': img_proc.image_mean if hasattr(img_proc, 'image_mean') else [0.48145466, 0.4578275, 0.40821073],
                    'std': img_proc.image_std if hasattr(img_proc, 'image_std') else [0.26862954, 0.26130258, 0.27577711],
                    'do_pad': img_proc.do_pad if hasattr(img_proc, 'do_pad') else True,
                }
                print(f"GPU preprocessing enabled: target_size={self._preprocessing_config['target_size']}")

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                if not (self.use_int8 and _BITSANDBYTES_AVAILABLE):
                    self.model = self.model.to(self.device)
                # Quantized model already placed via device_map="auto"
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if self.use_amp:
                    # Use new API for GradScaler (PyTorch 2.0+)
                    try:
                        self.scaler = torch.amp.GradScaler('cuda')
                    except AttributeError:
                        # Fallback for older PyTorch versions
                        self.scaler = torch.cuda.amp.GradScaler()
                else:
                    self.scaler = None
                print(f"Using CUDA GPU acceleration (device: {self.device})")
                print(f"Optimizations enabled: CUDNN benchmark, TF32, AMP: {self.use_amp}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.model = self.model.to(self.device)
                print("Using Apple Silicon GPU (MPS) acceleration")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (no GPU acceleration available)")

            self.model_compiled = False
            if hasattr(torch, "compile") and torch.cuda.is_available() and self.use_compile:
                try:
                    self.model = torch.compile(self.model, fullgraph=False, dynamic=True)
                    self.model_compiled = True
                    print("Model compiled with torch.compile for faster inference")
                except Exception as e:
                    print(f"Could not compile model (falling back to eager): {e}")

            model_load_time = time.time() - start_time
            print(f"Model loaded in {model_load_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def _area(box):
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _iou(boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a = (ax2 - ax1) * (ay2 - ay1)
        b = (bx2 - bx1) * (by2 - by1)
        return inter / (a + b - inter + 1e-9)

    def _apply_nms_and_merge(self, results: Dict[str, torch.Tensor], image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        # Greedy NMS (no merging) + containment suppression + size guards
        if results is None or len(results.get("boxes", [])) == 0:
            return results

        W, H = image_size
        img_area = float(W * H)
        boxes = results["boxes"].detach().cpu().numpy().astype(float)
        scores = results["scores"].detach().cpu().numpy().astype(float)
        labels = results["labels"].detach().cpu().numpy().astype(int)

        keep_boxes, keep_scores, keep_labels = [], [], []

        def clip(b):
            x1 = max(0.0, min(b[0], W - 1))
            y1 = max(0.0, min(b[1], H - 1))
            x2 = max(0.0, min(b[2], W - 1))
            y2 = max(0.0, min(b[3], H - 1))
            return [x1, y1, x2, y2]

        def is_inside(a, b):
            return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]

        for lab in set(labels.tolist()):
            idxs = [i for i, L in enumerate(labels) if L == lab]
            if not idxs:
                continue

            # sort by highest confidence first
            idxs.sort(key=lambda i: float(scores[i]), reverse=True)
            suppressed = set()

            for i in idxs:
                if i in suppressed:
                    continue

                b = clip(boxes[i])
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 0 or h <= 0:
                    continue

                # reject over-large boxes (background-y)
                side = max(w, h)
                if self.max_bbox_size is not None and side > self.max_bbox_size:
                    continue

                coverage = (w * h) / (img_area + 1e-9)
                if coverage > self.nms_max_coverage:
                    continue

                # small context expansion (kept small to avoid bloat)
                if self.nms_expand_ratio > 0:
                    cx, cy = b[0] + w / 2.0, b[1] + h / 2.0
                    dw, dh = w * self.nms_expand_ratio, h * self.nms_expand_ratio
                    b = clip([cx - (w / 2.0 + dw), cy - (h / 2.0 + dh),
                              cx + (w / 2.0 + dw), cy + (h / 2.0 + dh)])
                    w, h = b[2] - b[0], b[3] - b[1]
                    if w <= 0 or h <= 0:
                        continue
                    side = max(w, h)
                    if self.max_bbox_size is not None and side > self.max_bbox_size:
                        continue

                # accept this one
                keep_boxes.append([round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)])
                keep_scores.append(round(float(scores[i]), 6))
                keep_labels.append(int(lab))

                # suppress overlapping or contained boxes
                for j in idxs:
                    if j == i or j in suppressed:
                        continue
                    bj = clip(boxes[j])
                    if self._iou(b, bj) >= self.nms_iou_threshold or is_inside(bj, b) or is_inside(b, bj):
                        suppressed.add(j)

        if not keep_boxes:
            return results

        device = results["boxes"].device
        return {
            "boxes": torch.tensor(keep_boxes, device=device, dtype=torch.float32),
            "scores": torch.tensor(keep_scores, device=device, dtype=torch.float32),
            "labels": torch.tensor(keep_labels, device=device, dtype=torch.int64),
        }

    def prepare_search_terms(self):
        all_search_terms = []
        for config in self.class_configs:
            all_search_terms.extend(config['search_terms'])
        return all_search_terms
    
    def _get_tokenized_text(self, all_search_terms: List[str]):
        """Get tokenized text, using cache if search terms haven't changed"""
        # Create hash of search terms to check if they changed
        search_terms_str = str(sorted(all_search_terms))
        search_terms_hash = hashlib.md5(search_terms_str.encode()).hexdigest()
        
        # Check if we can reuse cached tokenization
        if self._tokenized_text_cache is not None and self._last_search_terms_hash == search_terms_hash:
            return self._tokenized_text_cache
        
        # Tokenize text (this is the expensive operation)
        tokenize_start = self.profiler.start_timer("text_tokenization")
        tokenized_text = self.processor.tokenizer(
            all_search_terms,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        self.profiler.end_timer("text_tokenization", tokenize_start)
        
        # Cache the result
        self._tokenized_text_cache = tokenized_text
        self._last_search_terms_hash = search_terms_hash
        
        return tokenized_text

    def load_and_preprocess_batch(self, image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
        load_start = self.profiler.start_timer("image_loading")
        images, valid_paths = [], []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                continue
        self.profiler.end_timer("image_loading", load_start)
        return images, valid_paths
    
    def _preprocess_single_image(self, image: Image.Image):
        """Preprocess a single image (for parallel processing)"""
        try:
            # Process single image using image processor
            image_input = self.processor.image_processor(
                [image],
                return_tensors="pt"
            )
            return image_input
        except Exception as e:
            print(f"Error in _preprocess_single_image: {e}")
            return None
    
    def _preprocess_images_parallel(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """Preprocess all images in parallel (regardless of batch size)"""
        if not images:
            return {}
        
        # If only one image, process sequentially (no benefit from parallel)
        if len(images) == 1:
            image_inputs = self.processor.image_processor(
                images,
                return_tensors="pt"
            )
            return image_inputs
        
        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.num_workers, len(images))) as executor:
            # Submit all images for parallel preprocessing
            future_to_image = {
                executor.submit(self._preprocess_single_image, img): idx 
                for idx, img in enumerate(images)
            }
            
            # Collect results in order
            processed_results = [None] * len(images)
            for future in as_completed(future_to_image):
                idx = future_to_image[future]
                try:
                    image_input = future.result()
                    processed_results[idx] = image_input
                except Exception as e:
                    print(f"Error preprocessing image {idx}: {e}")
                    processed_results[idx] = None
            
            # Filter out None results
            valid_processed = [img for img in processed_results if img is not None]
            
            if not valid_processed:
                # Fallback to sequential if all failed
                return self.processor.image_processor(images, return_tensors="pt")
            
            # Combine parallel-processed images into batch
            keys = valid_processed[0].keys()
            image_inputs = {}
            for key in keys:
                # Stack tensors from all images
                tensors = []
                for img in valid_processed:
                    if key in img:
                        t = img[key]
                        # Remove batch dimension if present (from [1, ...] to [...])
                        if t.dim() > 1 and t.size(0) == 1:
                            tensors.append(t.squeeze(0))
                        else:
                            tensors.append(t)
                
                if tensors:
                    # Stack all tensors along batch dimension
                    image_inputs[key] = torch.stack(tensors, dim=0)
            
            return image_inputs

    def _preprocess_images_on_gpu(self, images: List[Image.Image]) -> Tuple[Dict[str, torch.Tensor], Optional[List[Dict]]]:
        """
        Preprocess images on GPU instead of CPU.
        Returns (pixel_values_dict, preprocess_meta_list).
        preprocess_meta_list is used to correct bounding box coordinates: the model outputs
        coords in 960x960 (letterboxed) space; we need scale/padding info to map back to original.
        """
        if not images or not self.use_gpu_preprocessing or not self._preprocessing_config:
            # Fallback to CPU preprocessing - no metadata needed, processor handles coordinates
            return self.processor.image_processor(images, return_tensors="pt"), None
        
        try:
            target_h, target_w = self._preprocessing_config['target_size']
            mean = torch.tensor(self._preprocessing_config['mean'], 
                              device=self.device, dtype=torch.float32)
            std = torch.tensor(self._preprocessing_config['std'], 
                             device=self.device, dtype=torch.float32)
            
            processed_tensors = []
            preprocess_meta = []
            
            for img in images:
                orig_w, orig_h = img.size
                C, H, W = 3, orig_h, orig_w
                
                # Step 1: Convert PIL to tensor (CPU)
                tensor = transforms.ToTensor()(img).to(self.device)
                
                # Step 2: Resize while maintaining aspect ratio
                scale = min(target_h / H, target_w / W)
                new_h, new_w = int(H * scale), int(W * scale)
                
                tensor = tensor.unsqueeze(0)
                tensor = F.interpolate(
                    tensor, size=(new_h, new_w),
                    mode='bilinear', align_corners=False
                )
                tensor = tensor.squeeze(0)
                
                # Step 3: Pad to target size (center pad)
                pad_h = target_h - new_h
                pad_w = target_w - new_w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                
                # Step 4: Normalize
                mean_view = mean.view(1, 3, 1, 1)
                std_view = std.view(1, 3, 1, 1)
                tensor = (tensor.unsqueeze(0) - mean_view) / std_view
                tensor = tensor.squeeze(0)
                
                processed_tensors.append(tensor)
                # Store metadata for coordinate conversion: model outputs are in 960x960
                preprocess_meta.append({
                    'orig_w': orig_w, 'orig_h': orig_h,
                    'scale': scale, 'new_w': new_w, 'new_h': new_h,
                    'pad_left': pad_left, 'pad_top': pad_top,
                })
            
            batch_tensor = torch.stack(processed_tensors, dim=0)
            return {'pixel_values': batch_tensor}, preprocess_meta
            
        except Exception as e:
            print(f"GPU preprocessing failed, falling back to CPU: {e}")
            return self.processor.image_processor(images, return_tensors="pt"), None

    def _convert_boxes_from_letterbox(
        self,
        per_image_results: List[Dict],
        preprocess_meta: List[Dict]
    ) -> List[Dict]:
        """
        Convert bounding boxes from 960x960 letterboxed space to original image coordinates.
        Model outputs (x1,y1,x2,y2) in 960x960; we center-padded the resized image.
        """
        converted = []
        for res, meta in zip(per_image_results, preprocess_meta):
            if res is None or len(res.get("boxes", [])) == 0:
                converted.append(res)
                continue
            orig_w = meta['orig_w']
            orig_h = meta['orig_h']
            scale = meta['scale']
            new_w = meta['new_w']
            new_h = meta['new_h']
            pad_left = meta['pad_left']
            pad_top = meta['pad_top']
            boxes = res["boxes"].cpu().numpy()
            new_boxes = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x1_orig = (x1 - pad_left) * (orig_w / new_w)
                y1_orig = (y1 - pad_top) * (orig_h / new_h)
                x2_orig = (x2 - pad_left) * (orig_w / new_w)
                y2_orig = (y2 - pad_top) * (orig_h / new_h)
                x1_orig = max(0, min(x1_orig, orig_w))
                y1_orig = max(0, min(y1_orig, orig_h))
                x2_orig = max(0, min(x2_orig, orig_w))
                y2_orig = max(0, min(y2_orig, orig_h))
                new_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
            res = res.copy()
            res["boxes"] = torch.tensor(new_boxes, dtype=res["boxes"].dtype, device=res["boxes"].device)
            converted.append(res)
        return converted

    def _post_process_owlv2(
        self,
        outputs: Any,
        target_sizes: torch.Tensor,
        all_search_terms: List[str],
        n_images: int,
    ):
        grounded_fn = getattr(self.processor, "post_process_grounded_object_detection", None)
        if callable(grounded_fn):
            try:
                return grounded_fn(
                    outputs=outputs,
                    threshold=self.confidence_threshold,
                    target_sizes=target_sizes,
                    text_labels=[all_search_terms] * n_images,
                )
            except TypeError:
                return grounded_fn(
                    outputs=outputs,
                    threshold=self.confidence_threshold,
                    target_sizes=target_sizes,
                )

        obj_fn = getattr(self.processor, "post_process_object_detection", None)
        if callable(obj_fn):
            return obj_fn(
                outputs=outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )

        image_proc = getattr(self.processor, "image_processor", None)
        image_proc_fn = getattr(image_proc, "post_process_object_detection", None)
        if callable(image_proc_fn):
            return image_proc_fn(
                outputs=outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )

        raise AttributeError(
            "No compatible OWLv2 post_process_* function found on this transformers version."
        )

    def process_batch(self, images: List[Image.Image], all_search_terms: List[str]) -> List[Dict]:
        if not images:
            return []

        try:
            # OPTIMIZATION: Use cached tokenized text to avoid re-tokenization
            # Get tokenized text (cached if search terms are the same)
            tokenized_text = self._get_tokenized_text(all_search_terms)
            
            # OPTIMIZATION: GPU-based preprocessing (eliminates transfer time)
            image_preprocess_start = self.profiler.start_timer("image_preprocessing")
            image_inputs, gpu_preprocess_meta = self._preprocess_images_on_gpu(images)
            self.profiler.end_timer("image_preprocessing", image_preprocess_start)
            
            # OPTIMIZATION: Optimize input combination using efficient tensor operations
            combine_start = self.profiler.start_timer("input_combination")
            batch_size = len(images)
            
            # Expand tokenized text to match batch size (optimized version)
            text_inputs = {}
            for key, value in tokenized_text.items():
                if value.dim() == 2:
                    # 2D tensor: [1, seq_len] -> [batch_size, seq_len]
                    if value.size(0) == 1:
                        # Use expand instead of repeat for better memory efficiency
                        text_inputs[key] = value.expand(batch_size, -1)
                    else:
                        # Already has batch dimension, just use as is
                        text_inputs[key] = value
                elif value.dim() == 1:
                    # 1D tensor: [seq_len] -> [batch_size, seq_len]
                    # Use unsqueeze + expand for efficiency
                    text_inputs[key] = value.unsqueeze(0).expand(batch_size, -1)
                else:
                    # 0D or other, use repeat only if needed
                    text_inputs[key] = value.repeat(batch_size) if batch_size > 1 else value
            
            # Combine text and image inputs
            # If GPU preprocessing was used, image_inputs are already on GPU
            # Text inputs need to be transferred to GPU
            inputs = {**text_inputs, **image_inputs}
            self.profiler.end_timer("input_combination", combine_start)
            
            # Profiling: Data transfer to GPU (only for text inputs if GPU preprocessing was used)
            transfer_start = self.profiler.start_timer("data_transfer")
            if gpu_preprocess_meta is not None and image_inputs.get('pixel_values', None) is not None:
                # Image inputs are already on GPU, only transfer text inputs
                if self.use_cuda_streams and self.device.type == "cuda":
                    gpu_event = self.profiler.create_gpu_event("data_transfer")
                    if gpu_event:
                        gpu_event['start'].record(self.transfer_stream)
                    
                    with torch.cuda.stream(self.transfer_stream):
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                                v = v.pin_memory()
                                inputs[k] = v.to(self.device, non_blocking=True)
                    
                    if gpu_event:
                        gpu_event['end'].record(self.transfer_stream)
                    self.transfer_stream.synchronize()
                    if gpu_event:
                        self.profiler.record_gpu_time("data_transfer")
                else:
                    # Fallback: standard transfer
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                            v = v.pin_memory()
                            inputs[k] = v.to(self.device, non_blocking=True)
            else:
                # Fallback: CPU preprocessing - transfer all inputs
                if self.use_cuda_streams and self.device.type == "cuda":
                    gpu_event = self.profiler.create_gpu_event("data_transfer")
                    if gpu_event:
                        gpu_event['start'].record(self.transfer_stream)
                    
                    with torch.cuda.stream(self.transfer_stream):
                        pinned_inputs = {}
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                if v.device.type == "cpu":
                                    v = v.pin_memory()
                                pinned_inputs[k] = v.to(self.device, non_blocking=True)
                            else:
                                pinned_inputs[k] = v
                        inputs = pinned_inputs
                    
                    if gpu_event:
                        gpu_event['end'].record(self.transfer_stream)
                    self.transfer_stream.synchronize()
                    if gpu_event:
                        self.profiler.record_gpu_time("data_transfer")
                else:
                    # Fallback: standard transfer with pinned memory if possible
                    pinned_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                            v = v.pin_memory()
                        pinned_inputs[k] = v.to(self.device, non_blocking=True)
                    inputs = pinned_inputs
            self.profiler.end_timer("data_transfer", transfer_start)

            # Profiling: GPU Inference
            with torch.no_grad():
                with self._gpu_semaphore:
                    inference_start = self.profiler.start_timer("gpu_inference")
                    gpu_event = self.profiler.create_gpu_event("inference")
                    
                    if self.use_cuda_streams and self.device.type == "cuda" and gpu_event:
                        gpu_event['start'].record(self.compute_stream)
                        with torch.cuda.stream(self.compute_stream):
                            if self.use_amp and getattr(self, "scaler", None) is not None:
                                try:
                                    with torch.amp.autocast('cuda'):
                                        outputs = self.model(**inputs)
                                except AttributeError:
                                    with torch.cuda.amp.autocast():
                                        outputs = self.model(**inputs)
                            else:
                                outputs = self.model(**inputs)
                        gpu_event['end'].record(self.compute_stream)
                        self.compute_stream.synchronize()
                        self.profiler.record_gpu_time("inference")
                    else:
                        # Fallback: standard inference
                        if self.use_amp and getattr(self, "scaler", None) is not None and self.device.type == "cuda":
                            try:
                                with torch.amp.autocast('cuda'):
                                    outputs = self.model(**inputs)
                            except AttributeError:
                                with torch.cuda.amp.autocast():
                                    outputs = self.model(**inputs)
                        else:
                            outputs = self.model(**inputs)
                    
                    self.profiler.end_timer("gpu_inference", inference_start)
                    
                    # Clear GPU cache after inference to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Profiling: Post-processing
            postprocess_start = self.profiler.start_timer("postprocessing")
            if gpu_preprocess_meta is not None:
                # CRITICAL: When using GPU letterbox preprocessing, pass target_sizes=(960,960)
                # so post_process outputs coords in 960x960 space. Then we convert to original.
                # If we passed original size, post_process would scale to original but wrong
                # (assumes full 960x960 maps to image; it doesn't due to letterboxing).
                target_sizes = torch.tensor(
                    [[960, 960]] * len(images),
                    dtype=torch.float32, device=self.device
                )
            else:
                target_sizes = torch.tensor([img.size[::-1] for img in images], dtype=torch.float32)
                if target_sizes.device.type == "cpu" and torch.cuda.is_available():
                    target_sizes = target_sizes.pin_memory()
                target_sizes = target_sizes.to(self.device, non_blocking=True)
            
            per_image_results = self._post_process_owlv2(
                outputs=outputs,
                target_sizes=target_sizes,
                all_search_terms=all_search_terms,
                n_images=len(images),
            )

            # When using GPU preprocessing: convert from 960x960 to original image coordinates
            if gpu_preprocess_meta is not None:
                per_image_results = self._convert_boxes_from_letterbox(
                    per_image_results, gpu_preprocess_meta
                )

            batch_results = []
            for img, res in zip(images, per_image_results):
                res = self._apply_nms_and_merge(res, img.size)
                batch_results.append(res)
            
            self.profiler.end_timer("postprocessing", postprocess_start)

            # Clear GPU cache after processing batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return batch_results
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return [None] * len(images)

    def draw_detections(self, image, results, image_name, all_search_terms):
        draw_start = self.profiler.start_timer("drawing")
        # Copy image so we don't modify the original (important when using queues/threads)
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        try:
            font = ImageFont.load_default()
        except:
            font = None

        detections = []

        if results is None:
            self.profiler.end_timer("drawing", draw_start)
            return annotated_image, detections

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            box = [round(float(i), 2) for i in box.tolist()]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            w = x2 - x1
            h = y2 - y1
            side = max(w, h)
            if self.max_bbox_size is not None and side > self.max_bbox_size:
                continue
            # Ensure visible box: expand zero-area boxes to min 4px so they draw
            if w < 2 or h < 2:
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                box = [max(0, cx - 2), max(0, cy - 2), min(annotated_image.width, cx + 2), min(annotated_image.height, cy + 2)]

            draw.rectangle(box, outline="red", width=3)
            label_idx = label.item() if hasattr(label, 'item') else int(label)
            detected_term = all_search_terms[label_idx]
            text = f"{detected_term}: {round(score.item(), 2)}"
            if font:
                draw.text((box[0], box[1] - 10), text, fill="red", font=font)
            else:
                draw.text((box[0], box[1] - 10), text, fill="red")

            detections.append({
                "label": detected_term,
                "score": round(score.item(), 2),
                "box": box
            })

        self.profiler.end_timer("drawing", draw_start)
        return annotated_image, detections

    def process_batch_worker(self, batch_data):
        image_paths, all_search_terms = batch_data
        batch_start_time = time.time()

        images, valid_paths = self.load_and_preprocess_batch(image_paths)
        
        if not images:
            return []

        batch_results = self.process_batch(images, all_search_terms)
        all_results = []

        for image, results, image_path in zip(images, batch_results, valid_paths):
            if results is None:
                continue

            image_name = os.path.basename(image_path)
            annotated_image, detections = self.draw_detections(image, results, image_name, all_search_terms)

            save_start = self.profiler.start_timer("image_save_immediate")
            annotated_folder = os.path.join(self.output_folder, "annotated")
            os.makedirs(annotated_folder, exist_ok=True)
            output_image_path = os.path.join(annotated_folder, f"annotated_{image_name}")
            annotated_image.save(output_image_path)
            self.profiler.end_timer("image_save_immediate", save_start)

            for det in detections:
                if det["score"] >= self.confidence_threshold:
                    detected_class = None
                    prediction_value = 0
                    for config in self.class_configs:
                        if det["label"] in config['search_terms']:
                            detected_class = config['class_name']
                            prediction_value = config['prediction_value']
                            break

                    if detected_class:
                        image_result = {
                            "img": image_name,
                            "prediction": prediction_value,
                            "label": detected_class,
                            "latency": round(time.time() - batch_start_time, 2),
                            "confidence": round(det["score"], 2),
                            "bounding_box": det["box"]
                        }
                        all_results.append(image_result)

        batch_time = time.time() - batch_start_time
        print(f"  - Batch processed in {batch_time:.2f}s ({len(valid_paths)} images)")
        print(f"  - Detections found: {len([d for d in all_results if d.get('confidence', 0) >= self.confidence_threshold])}")

        return all_results

    def create_batches(self, image_files: List[str]) -> List[List[str]]:
        batches = []
        for i in range(0, len(image_files), self.batch_size):
            batch = image_files[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def _prefetch_batch_worker(self, batch_queue: Queue, prefetch_queue: Queue, all_search_terms: List[str], stop_event: threading.Event):
        """OPTIMIZATION 3: Async Prefetching - Load and preprocess batches in background"""
        while not stop_event.is_set():
            try:
                # Get next batch to prefetch
                batch_idx, image_paths = batch_queue.get(timeout=0.1)
                if batch_idx is None:  # Sentinel value to stop
                    break
                
                # Load and preprocess images (CPU work)
                images, valid_paths = self.load_and_preprocess_batch(image_paths)
                
                # Put preprocessed batch in prefetch queue (non-blocking with timeout)
                try:
                    prefetch_queue.put((batch_idx, images, valid_paths, all_search_terms), timeout=1.0)
                except:
                    # Queue full, skip this batch (will be processed normally)
                    continue
            except:
                continue

    def _process_prefetched_batch(self, images: List[Image.Image], valid_paths: List[str], 
                                  all_search_terms: List[str], save_queue: Queue = None):
        """Process a prefetched batch (images already loaded and preprocessed)"""
        batch_start_time = time.time()
        
        if not images:
            return []

        # Run GPU inference
        batch_results = self.process_batch(images, all_search_terms)
        all_results = []

        # Post-process results (fast - no I/O blocking)
        for image, results, image_path in zip(images, batch_results, valid_paths):
            if results is None:
                continue

            image_name = os.path.basename(image_path)
            annotated_image, detections = self.draw_detections(image, results, image_name, all_search_terms)

            # Queue image for saving (non-blocking) instead of saving immediately
            if save_queue is not None:
                queue_start = self.profiler.start_timer("queue_operation")
                annotated_folder = os.path.join(self.output_folder, "annotated")
                output_image_path = os.path.join(annotated_folder, f"annotated_{image_name}")
                try:
                    save_queue.put((annotated_image, output_image_path, annotated_folder), timeout=1.0)
                    self.profiler.end_timer("queue_operation", queue_start)
                except:
                    # Queue full, save immediately (fallback to prevent blocking)
                    self.profiler.end_timer("queue_operation", queue_start)
                    save_start = self.profiler.start_timer("image_save_immediate")
                    os.makedirs(annotated_folder, exist_ok=True)
                    annotated_image.save(output_image_path)
                    self.profiler.end_timer("image_save_immediate", save_start)
            else:
                # Fallback: save immediately if no queue provided
                save_start = self.profiler.start_timer("image_save_immediate")
                annotated_folder = os.path.join(self.output_folder, "annotated")
                os.makedirs(annotated_folder, exist_ok=True)
                output_image_path = os.path.join(annotated_folder, f"annotated_{image_name}")
                annotated_image.save(output_image_path)
                self.profiler.end_timer("image_save_immediate", save_start)

            for det in detections:
                if det["score"] >= self.confidence_threshold:
                    detected_class = None
                    prediction_value = 0
                    for config in self.class_configs:
                        if det["label"] in config['search_terms']:
                            detected_class = config['class_name']
                            prediction_value = config['prediction_value']
                            break

                    if detected_class:
                        image_result = {
                            "img": image_name,
                            "prediction": prediction_value,
                            "label": detected_class,
                            "latency": round(time.time() - batch_start_time, 2),
                            "confidence": round(det["score"], 2),
                            "bounding_box": det["box"]
                        }
                        all_results.append(image_result)

        batch_time = time.time() - batch_start_time
        print(f"  - Batch processed in {batch_time:.2f}s ({len(valid_paths)} images)")
        print(f"  - Detections found: {len([d for d in all_results if d.get('confidence', 0) >= self.confidence_threshold])}")

        return all_results

    def _image_saver_worker(self, save_queue: Queue, stop_event: threading.Event):
        """Background thread for saving annotated images (non-blocking)"""
        from queue import Empty
        saved_count = 0
        last_annotated_folder = None
        while not stop_event.is_set():
            try:
                annotated_image, output_image_path, annotated_folder = save_queue.get(timeout=0.1)
                if annotated_image is None:  # Sentinel
                    break
                last_annotated_folder = annotated_folder
                # Save image (I/O operation - happens in background)
                save_start = self.profiler.start_timer("image_save_background")
                os.makedirs(annotated_folder, exist_ok=True)
                annotated_image.save(output_image_path)
                self.profiler.end_timer("image_save_background", save_start)
                saved_count += 1
            except Empty:
                continue
            except Exception as e:
                print(f"Error saving annotated image: {e}")
        if saved_count > 0 and last_annotated_folder:
            print(f"Saved {saved_count} annotated images to {last_annotated_folder}")

    def _start_gpu_monitoring(self):
        """Start GPU monitoring using nvidia-smi"""
        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU monitoring")
            return
        
        # Check if nvidia-smi is available
        try:
            subprocess.run(['nvidia-smi', '--version'], 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                         timeout=5, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: nvidia-smi not available, GPU monitoring disabled")
            return
        
        # Set up log file
        self.gpu_log_file = os.path.join(self.output_folder, "gpu_utilization.csv")
        gpu_detailed_log = os.path.join(self.output_folder, "gpu_detailed.csv")
        
        print(f"Starting GPU monitoring...")
        print(f"GPU logs will be saved to: {self.gpu_log_file} and {gpu_detailed_log}")
        
        # Start nvidia-smi continuous monitoring in background
        try:
            # Method 1: Use nvidia-smi with continuous query (most reliable)
            cmd = [
                'nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
                '--format=csv',
                '-l', '1'  # Loop every 1 second
            ]
            with open(gpu_detailed_log, 'w') as f:
                self._gpu_monitor_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    bufsize=1
                )
            
            # Method 2: Also create a simpler log with just utilization
            self._gpu_monitor_running = True
            def monitor_thread():
                with open(self.gpu_log_file, 'w') as f:
                    f.write("timestamp,gpu_utilization(%),memory_utilization(%),memory_used(MB),memory_total(MB),power_draw(W),temperature(C)\n")
                    while self._gpu_monitor_running:
                        try:
                            result = subprocess.run(
                                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            if result.returncode == 0:
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                values = result.stdout.strip().replace(' ', '')
                                f.write(f"{timestamp},{values}\n")
                                f.flush()
                        except Exception as e:
                            pass
                        time.sleep(1)
            
            self._gpu_monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
            self._gpu_monitor_thread.start()
            print("GPU monitoring started successfully")
        except Exception as e:
            print(f"Warning: Could not start GPU monitoring: {e}")

    def _stop_gpu_monitoring(self):
        """Stop GPU monitoring"""
        self._gpu_monitor_running = False
        if self._gpu_monitor_thread is not None:
            self._gpu_monitor_thread.join(timeout=2)
        if self._gpu_monitor_process is not None:
            try:
                self._gpu_monitor_process.terminate()
                self._gpu_monitor_process.wait(timeout=2)
            except:
                try:
                    self._gpu_monitor_process.kill()
                except:
                    pass
        if self.gpu_log_file and os.path.exists(self.gpu_log_file):
            print(f"GPU monitoring stopped. Log saved to: {self.gpu_log_file}")

    def process_images(self):
        # Set environment variable to suppress tokenizers warning
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print(f"Starting optimized object identification process...")
        print(f"Configuration: 1 GPU, {self.num_workers} CPU cores, batch size {self.batch_size}")

        overall_start_time = time.time()

        image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.tif', '*.TIF']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))

        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return

        print(f"Found {len(image_files)} images in {self.input_folder}")

        all_search_terms = self.prepare_search_terms()
        batches = self.create_batches(image_files)
        print(f"Created {len(batches)} batches of size {self.batch_size}")

        all_results = []

        # OPTIMIZATION 3: Async Prefetching + Deferred Image Saving
        use_async_prefetch = True
        if use_async_prefetch and len(batches) > 1:
            # Queues for prefetching and image saving
            batch_queue = Queue()  # Batches to prefetch
            prefetch_queue = Queue(maxsize=2)  # Prefetched batches (limit to prevent memory buildup)
            save_queue = Queue(maxsize=100)  # Images to save (bounded to prevent memory issues at scale)
            stop_prefetch = threading.Event()
            stop_saver = threading.Event()
            
            # Start prefetch worker thread (silent)
            prefetch_thread = threading.Thread(
                target=self._prefetch_batch_worker,
                args=(batch_queue, prefetch_queue, all_search_terms, stop_prefetch),
                daemon=True
            )
            prefetch_thread.start()
            
            # Start image saver worker thread (saves images in background)
            saver_thread = threading.Thread(
                target=self._image_saver_worker,
                args=(save_queue, stop_saver),
                daemon=True
            )
            saver_thread.start()
            
            # Enqueue all batches for prefetching
            for idx, batch in enumerate(batches):
                batch_queue.put((idx, batch))
            batch_queue.put((None, None))  # Sentinel to stop worker
            
            # Process batches using prefetched data
            processed_batches = set()
            batch_results_dict = {}
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_batch_idx = {}
                # Must drain futures after all batches are *submitted*; otherwise the loop exits
                # while GPU work is still running and overall_results.json stays empty.
                while len(processed_batches) < len(batches) or future_to_batch_idx:
                    try:
                        batch_idx, images, valid_paths, search_terms = prefetch_queue.get(timeout=0.5)

                        if batch_idx not in processed_batches:
                            future = executor.submit(
                                self._process_prefetched_batch,
                                images,
                                valid_paths,
                                search_terms,
                                save_queue,
                            )
                            future_to_batch_idx[future] = batch_idx
                            processed_batches.add(batch_idx)
                    except Exception:
                        pass

                    for future in list(future_to_batch_idx.keys()):
                        if future.done():
                            batch_idx = future_to_batch_idx.pop(future)
                            try:
                                batch_results_dict[batch_idx] = future.result()
                            except Exception as e:
                                print(f"Error processing batch {batch_idx}: {str(e)}")
                                batch_results_dict[batch_idx] = []
            
            # Stop prefetch thread
            stop_prefetch.set()
            prefetch_thread.join(timeout=2)
            
            # Wait for all images to be saved
            print("Waiting for image saving to complete...")
            save_queue.put((None, None, None))  # Sentinel to stop saver
            saver_thread.join(timeout=30)
            
            # Collect results in order
            for idx in range(len(batches)):
                if idx in batch_results_dict:
                    all_results.extend(batch_results_dict[idx])
        else:
            # Fallback: Original ThreadPoolExecutor approach with deferred saving
            # Setup deferred image saving
            save_queue = Queue(maxsize=100)  # Bounded to prevent memory issues at scale
            stop_saver = threading.Event()
            saver_thread = threading.Thread(
                target=self._image_saver_worker,
                args=(save_queue, stop_saver),
                daemon=True
            )
            saver_thread.start()
            
            # Update process_batch_worker to use save_queue
            def process_with_deferred_save(batch_data):
                image_paths, all_search_terms = batch_data
                batch_start_time = time.time()

                images, valid_paths = self.load_and_preprocess_batch(image_paths)
                
                if not images:
                    return []

                batch_results = self.process_batch(images, all_search_terms)
                all_results = []

                for image, results, image_path in zip(images, batch_results, valid_paths):
                    if results is None:
                        continue

                    image_name = os.path.basename(image_path)
                    annotated_image, detections = self.draw_detections(image, results, image_name, all_search_terms)

                    # Queue for deferred saving
                    annotated_folder = os.path.join(self.output_folder, "annotated")
                    output_image_path = os.path.join(annotated_folder, f"annotated_{image_name}")
                    queue_start = self.profiler.start_timer("queue_operation")
                    try:
                        save_queue.put((annotated_image, output_image_path, annotated_folder), timeout=1.0)
                        self.profiler.end_timer("queue_operation", queue_start)
                    except:
                        # Queue full, save immediately (fallback to prevent blocking)
                        self.profiler.end_timer("queue_operation", queue_start)
                        save_start = self.profiler.start_timer("image_save_immediate")
                        os.makedirs(annotated_folder, exist_ok=True)
                        annotated_image.save(output_image_path)
                        self.profiler.end_timer("image_save_immediate", save_start)

                    for det in detections:
                        if det["score"] >= self.confidence_threshold:
                            detected_class = None
                            prediction_value = 0
                            for config in self.class_configs:
                                if det["label"] in config['search_terms']:
                                    detected_class = config['class_name']
                                    prediction_value = config['prediction_value']
                                    break

                            if detected_class:
                                image_result = {
                                    "img": image_name,
                                    "prediction": prediction_value,
                                    "label": detected_class,
                                    "latency": round(time.time() - batch_start_time, 2),
                                    "confidence": round(det["score"], 2),
                                    "bounding_box": det["box"]
                                }
                                all_results.append(image_result)

                batch_time = time.time() - batch_start_time
                detections_count = len([d for d in all_results if d.get('confidence', 0) >= self.confidence_threshold])
                print(f"Batch {len(valid_paths)} images: {batch_time:.2f}s, {detections_count} detections")

                return all_results
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_batch = {
                    executor.submit(process_with_deferred_save, (batch, all_search_terms)): batch
                    for batch in batches
                }

                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        print(f"Error processing batch {batch}: {str(e)}")
            
            # Wait for all images to be saved
            save_queue.put((None, None, None))  # Sentinel
            saver_thread.join(timeout=30)

        overall_end_time = time.time()

        overall_json_path = os.path.join(self.output_folder, "overall_results.json")
        with open(overall_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("OPTIMIZED PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total results: {len(all_results)}")
        print(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")
        print(f"Average time per image: {(overall_end_time - overall_start_time) / max(len(image_files), 1):.2f} seconds")
        print(f"Images per second: {len(image_files) / (overall_end_time - overall_start_time):.2f}")
        print(f"Results saved to: {overall_json_path}")
        print(f"Annotated images saved in: {self.output_folder}")

# --- Grounding DINO (inlined) ---

def _import_grounding_dino():
    try:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        return GroundingDinoForObjectDetection, GroundingDinoProcessor
    except ImportError as e:
        raise ImportError(
            "grounding_dino_classifier requires transformers with Grounding DINO. "
            "Try: pip install transformers torchvision"
        ) from e

class GroundingDinoClassifier:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        confidence_threshold: float = 0.12,
        text_threshold: float = 0.25,
        class_configs: Optional[List[Dict[str, Any]]] = None,
        preserve_metadata: bool = True,
        num_workers: int = 12,
        batch_size: int = 1,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        io_workers: int = 8,
        save_workers: int = 4,
        use_amp: bool = False,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.box_threshold = confidence_threshold
        self.text_threshold = text_threshold
        self.class_configs = class_configs or []
        self.preserve_metadata = preserve_metadata
        self.num_workers = min(num_workers, 24)
        self.batch_size = max(1, batch_size)
        self.model_id = model_id
        self.io_workers = max(1, min(io_workers, 24))
        self.save_workers = max(1, min(save_workers, 16))
        self.use_amp = bool(use_amp and torch.cuda.is_available())

        self.max_bbox_size = 700
        self.nms_iou_threshold = 0.25
        self.nms_expand_ratio = 0.10
        self.nms_max_coverage = 0.95

        self.max_concurrent_batches = 1
        self._gpu_semaphore = Semaphore(self.max_concurrent_batches)
        self._gpu_monitor_running = False
        self._gpu_monitor_thread = None
        self._gpu_monitor_process = None
        self.gpu_log_file = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._save_executor: Optional[ThreadPoolExecutor] = None

        os.makedirs(output_folder, exist_ok=True)

        print("Initializing Grounding DINO classifier")
        print(f"Device: {self.device}, model: {self.model_id}")
        print(f"Box threshold: {self.box_threshold}, text_threshold: {self.text_threshold}")
        print(
            f"Batch workers: {self.num_workers}, batch_size: {self.batch_size}, "
            f"io_workers: {self.io_workers}, save_workers: {self.save_workers}, AMP: {self.use_amp}"
        )
        print(f"GPU semaphore: {self.max_concurrent_batches} concurrent forward pass(es)")

        self._load_model()

    def _load_model(self):
        mid_l = self.model_id.lower()
        # Only flag real typo "rounding-dino" as repo name, not substring inside "grounding-dino"
        if re.search(r"(?:^|/)rounding-dino", mid_l):
            raise ValueError(
                f"model_id typo: {self.model_id!r}. "
                "Correct: IDEA-Research/grounding-dino-tiny (grounding, not rounding)."
            )
        if "grouding-dino" in mid_l:
            raise ValueError(
                f"model_id typo: {self.model_id!r}. "
                "Correct: IDEA-Research/grounding-dino-tiny (spell grounding with 'ground')."
            )
        # Common typo: missing 'r' in grounding
        if "gounding-dino" in mid_l:
            old = self.model_id
            self.model_id = re.sub(r"(?i)gounding(?=-dino)", "grounding", self.model_id)
            print(f"Note: corrected model_id {old!r} -> {self.model_id!r}")
        ModelCls, ProcCls = _import_grounding_dino()
        t0 = time.time()
        self.processor = ProcCls.from_pretrained(self.model_id)
        self.model = ModelCls.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"Model loaded in {time.time() - t0:.2f}s")

    @staticmethod
    def _iou(boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a = (ax2 - ax1) * (ay2 - ay1)
        b = (bx2 - bx1) * (by2 - by1)
        return inter / (a + b - inter + 1e-9)

    @staticmethod
    def _phrase_to_term_index(phrase: str, search_terms: List[str]) -> Optional[int]:
        """Map model text label to index in search_terms (exact / substring)."""
        p = phrase.strip().lower()
        for i, t in enumerate(search_terms):
            if p == t.strip().lower():
                return i
        for i, t in enumerate(search_terms):
            tl = t.strip().lower()
            if tl in p or p in tl:
                return i
        return None

    def _apply_nms_and_merge(
        self, results: Dict[str, torch.Tensor], image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        if results is None or len(results.get("boxes", [])) == 0:
            return results

        W, H = image_size
        img_area = float(W * H)
        boxes = results["boxes"].detach().cpu().numpy().astype(float)
        scores = results["scores"].detach().cpu().numpy().astype(float)
        labels = results["labels"].detach().cpu().numpy().astype(int)

        keep_boxes, keep_scores, keep_labels = [], [], []

        def clip(b):
            x1 = max(0.0, min(b[0], W - 1))
            y1 = max(0.0, min(b[1], H - 1))
            x2 = max(0.0, min(b[2], W - 1))
            y2 = max(0.0, min(b[3], H - 1))
            return [x1, y1, x2, y2]

        def is_inside(a, b):
            return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]

        for lab in set(labels.tolist()):
            idxs = [i for i, L in enumerate(labels) if L == lab]
            if not idxs:
                continue
            idxs.sort(key=lambda i: float(scores[i]), reverse=True)
            suppressed = set()

            for i in idxs:
                if i in suppressed:
                    continue
                b = clip(boxes[i])
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 0 or h <= 0:
                    continue
                side = max(w, h)
                if self.max_bbox_size is not None and side > self.max_bbox_size:
                    continue
                coverage = (w * h) / (img_area + 1e-9)
                if coverage > self.nms_max_coverage:
                    continue
                if self.nms_expand_ratio > 0:
                    cx, cy = b[0] + w / 2.0, b[1] + h / 2.0
                    dw, dh = w * self.nms_expand_ratio, h * self.nms_expand_ratio
                    b = clip(
                        [
                            cx - (w / 2.0 + dw),
                            cy - (h / 2.0 + dh),
                            cx + (w / 2.0 + dw),
                            cy + (h / 2.0 + dh),
                        ]
                    )
                    w, h = b[2] - b[0], b[3] - b[1]
                    if w <= 0 or h <= 0:
                        continue
                    side = max(w, h)
                    if self.max_bbox_size is not None and side > self.max_bbox_size:
                        continue

                keep_boxes.append([round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)])
                keep_scores.append(round(float(scores[i]), 6))
                keep_labels.append(int(lab))

                for j in idxs:
                    if j == i or j in suppressed:
                        continue
                    bj = clip(boxes[j])
                    if self._iou(b, bj) >= self.nms_iou_threshold or is_inside(bj, b) or is_inside(b, bj):
                        suppressed.add(j)

        if not keep_boxes:
            return results

        dev = results["boxes"].device
        return {
            "boxes": torch.tensor(keep_boxes, device=dev, dtype=torch.float32),
            "scores": torch.tensor(keep_scores, device=dev, dtype=torch.float32),
            "labels": torch.tensor(keep_labels, device=dev, dtype=torch.int64),
        }

    def prepare_search_terms(self) -> List[str]:
        terms: List[str] = []
        for c in self.class_configs:
            raw_terms = c.get("search_terms", [])
            if not isinstance(raw_terms, list):
                continue
            for t in raw_terms:
                ts = str(t).strip()
                if ts:
                    terms.append(ts)
        return terms

    @staticmethod
    def _build_grounding_prompt(all_search_terms: List[str]) -> str:
        """
        Grounding DINO expects natural-language text per image (string),
        not list-of-lists token input.
        """
        cleaned: List[str] = []
        for term in all_search_terms:
            t = str(term).strip().rstrip(".")
            if t:
                cleaned.append(t)
        # Standard prompt shape for grounded detection.
        return ". ".join(cleaned) + "."

    @staticmethod
    def _load_one_rgb(args: Tuple[int, str]) -> Tuple[int, str, Optional[Image.Image]]:
        idx, p = args
        try:
            return idx, p, Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return idx, p, None

    def load_and_preprocess_batch(self, image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
        """Load images in parallel (CPU I/O + decode); preserves input order."""
        if not image_paths:
            return [], []
        n_workers = min(len(image_paths), self.io_workers)
        indexed: List[Tuple[int, str, Optional[Image.Image]]] = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for item in pool.map(self._load_one_rgb, enumerate(image_paths)):
                indexed.append(item)
        indexed.sort(key=lambda x: x[0])
        images, valid_paths = [], []
        for _i, p, img in indexed:
            if img is not None:
                images.append(img)
                valid_paths.append(p)
        return images, valid_paths

    @staticmethod
    def _save_pil_image(pil_image: Image.Image, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pil_image.save(out_path)

    def _submit_save(self, pil_image: Image.Image, out_path: str) -> None:
        """Non-blocking save when a save executor is active."""
        if self._save_executor is not None:
            self._save_executor.submit(self._save_pil_image, pil_image, out_path)
        else:
            self._save_pil_image(pil_image, out_path)

    def _raw_result_to_tensor_dict(
        self, result: Dict[str, Any], all_search_terms: List[str]
    ) -> Optional[Dict[str, torch.Tensor]]:
        boxes = result.get("boxes")
        scores = result.get("scores")
        text_labels = result.get("text_labels") or result.get("labels") or []
        if boxes is None or len(boxes) == 0:
            return None

        keep_boxes, keep_scores, keep_labels = [], [], []
        for box, score, tl in zip(boxes, scores, text_labels):
            idx = self._phrase_to_term_index(str(tl), all_search_terms)
            if idx is None:
                continue
            b = [float(x) for x in box.tolist()]
            keep_boxes.append(b)
            keep_scores.append(float(score.item()))
            keep_labels.append(idx)

        if not keep_boxes:
            return None

        return {
            "boxes": torch.tensor(keep_boxes, dtype=torch.float32, device="cpu"),
            "scores": torch.tensor(keep_scores, dtype=torch.float32, device="cpu"),
            "labels": torch.tensor(keep_labels, dtype=torch.int64, device="cpu"),
        }

    def _post_process_grounding_outputs(
        self,
        outputs: Any,
        input_ids: torch.Tensor,
        target_sizes: List[Tuple[int, int]],
    ):
        fn = self.processor.post_process_grounded_object_detection
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        kwargs = {
            "outputs": outputs,
            "input_ids": input_ids,
            "text_threshold": self.text_threshold,
            "target_sizes": target_sizes,
        }
        if "box_threshold" in params:
            kwargs["box_threshold"] = self.box_threshold
            return fn(**kwargs)
        if "threshold" in params:
            kwargs["threshold"] = self.box_threshold
            return fn(**kwargs)
        try:
            return fn(**{**kwargs, "box_threshold": self.box_threshold})
        except TypeError:
            return fn(**{**kwargs, "threshold": self.box_threshold})

    def predict_batch(
        self, images: List[Image.Image], all_search_terms: List[str]
    ) -> List[Optional[Dict[str, torch.Tensor]]]:
        if not images:
            return []
        prompt = self._build_grounding_prompt(all_search_terms)
        if not prompt.strip():
            return [None] * len(images)
        text_batch = [prompt for _ in images]
        inputs = self.processor(images=images, text=text_batch, return_tensors="pt")
        nb = self.device.type == "cuda"
        inputs = {k: v.to(self.device, non_blocking=nb) for k, v in inputs.items()}

        with torch.no_grad():
            with self._gpu_semaphore:
                if self.use_amp:
                    try:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            outputs = self.model(**inputs)
                    except Exception:
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

        target_sizes = [(im.height, im.width) for im in images]
        with torch.no_grad():
            processed = self._post_process_grounding_outputs(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                target_sizes=target_sizes,
            )

        out: List[Optional[Dict[str, torch.Tensor]]] = []
        for pr in processed:
            out.append(self._raw_result_to_tensor_dict(pr, all_search_terms))
        return out

    def draw_detections(
        self,
        image: Image.Image,
        results: Optional[Dict[str, torch.Tensor]],
        all_search_terms: List[str],
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        detections: List[Dict[str, Any]] = []
        if results is None:
            return image, detections

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            box = [round(float(i), 2) for i in box.tolist()]
            w, h = box[2] - box[0], box[3] - box[1]
            side = max(w, h)
            if self.max_bbox_size is not None and side > self.max_bbox_size:
                continue
            li = int(label.item())
            detected_term = all_search_terms[li]
            sc = round(float(score.item()), 2)
            draw.rectangle(box, outline="red", width=3)
            txt = f"{detected_term}: {sc}"
            if font:
                draw.text((box[0], box[1] - 10), txt, fill="red", font=font)
            else:
                draw.text((box[0], box[1] - 10), txt, fill="red")
            detections.append({"label": detected_term, "score": sc, "box": box})

        return image, detections

    def process_batch_worker(self, batch_data: Tuple[List[str], List[str]]):
        paths, all_search_terms = batch_data
        t0 = time.time()
        print(f"Processing batch of {len(paths)} images")

        images, valid_paths = self.load_and_preprocess_batch(paths)
        if not images:
            return []

        preds = self.predict_batch(images, all_search_terms)
        rows: List[Dict[str, Any]] = []

        for image, pred_dict, path in zip(images, preds, valid_paths):
            name = os.path.basename(path)
            out_img_path = os.path.join(self.output_folder, "annotated", f"annotated_{name}")

            if pred_dict is None:
                self._submit_save(image, out_img_path)
                continue

            merged = self._apply_nms_and_merge(pred_dict, image.size)
            annotated, dets = self.draw_detections(image, merged, all_search_terms)
            self._submit_save(annotated, out_img_path)

            for det in dets:
                if det["score"] > self.box_threshold:
                    detected_class = None
                    prediction_value = 0
                    for cfg in self.class_configs:
                        if det["label"] in cfg["search_terms"]:
                            detected_class = cfg["class_name"]
                            prediction_value = cfg["prediction_value"]
                            break
                    if detected_class:
                        rows.append(
                            {
                                "img": name,
                                "prediction": prediction_value,
                                "label": detected_class,
                                "latency": round(time.time() - t0, 2),
                                "confidence": round(det["score"], 2),
                                "bounding_box": det["box"],
                            }
                        )

        dt = time.time() - t0
        n = len(valid_paths)
        ips = (n / dt) if dt > 0 else 0.0
        spi = (dt / n) if n > 0 else 0.0
        print(
            f"  - Batch in {dt:.2f}s ({n} images, {ips:.2f} img/s, {spi:.3f} s/img); rows: {len(rows)}"
        )
        return rows

    def create_batches(self, files: List[str]) -> List[List[str]]:
        return [files[i : i + self.batch_size] for i in range(0, len(files), self.batch_size)]

    def _start_gpu_monitoring(self):
        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU monitoring")
            return
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: nvidia-smi not available, GPU monitoring disabled")
            return

        self.gpu_log_file = os.path.join(self.output_folder, "gpu_utilization.csv")
        detail = os.path.join(self.output_folder, "gpu_detailed.csv")
        print(f"GPU logs: {self.gpu_log_file}, {detail}")
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
            ]
            with open(detail, "w") as f:
                self._gpu_monitor_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, bufsize=1)
            self._gpu_monitor_running = True

            def loop():
                with open(self.gpu_log_file, "w") as gf:
                    gf.write(
                        "timestamp,gpu_utilization(%),memory_utilization(%),memory_used(MB),memory_total(MB),power_draw(W),temperature(C)\n"
                    )
                    while self._gpu_monitor_running:
                        try:
                            r = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                                    "--format=csv,noheader,nounits",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=2,
                            )
                            if r.returncode == 0:
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                gf.write(f"{ts},{r.stdout.strip().replace(' ', '')}\n")
                                gf.flush()
                        except Exception:
                            pass
                        time.sleep(1)

            self._gpu_monitor_thread = threading.Thread(target=loop, daemon=True)
            self._gpu_monitor_thread.start()
        except Exception as e:
            print(f"GPU monitoring failed: {e}")

    def _stop_gpu_monitoring(self):
        self._gpu_monitor_running = False
        if self._gpu_monitor_thread:
            self._gpu_monitor_thread.join(timeout=2)
        if self._gpu_monitor_process:
            try:
                self._gpu_monitor_process.terminate()
                self._gpu_monitor_process.wait(timeout=2)
            except Exception:
                try:
                    self._gpu_monitor_process.kill()
                except Exception:
                    pass

    def process_images(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("Starting Grounding DINO inference...")
        t0 = time.time()

        exts = [
            "*.jpg",
            "*.jpeg",
            "*.JPG",
            "*.JPEG",
            "*.png",
            "*.PNG",
            "*.bmp",
            "*.BMP",
            "*.tiff",
            "*.TIFF",
            "*.tif",
            "*.TIF",
        ]
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.input_folder, e)))

        if not files:
            print(f"No images in {self.input_folder}")
            return

        terms = self.prepare_search_terms()
        if not terms:
            print("No search_terms in class_configs")
            return

        print(f"Found {len(files)} images; phrases: {terms}")
        batches = self.create_batches(files)
        ann_root = os.path.join(self.output_folder, "annotated")
        os.makedirs(ann_root, exist_ok=True)

        all_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.save_workers) as save_ex:
            self._save_executor = save_ex
            try:
                with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                    fmap = {ex.submit(self.process_batch_worker, (b, terms)): b for b in batches}
                    for fut in as_completed(fmap):
                        try:
                            all_rows.extend(fut.result())
                        except Exception as e:
                            print(f"Batch error: {e}")
            finally:
                self._save_executor = None

        t1 = time.time()

        elapsed = t1 - t0
        n_img = len(files)
        ips = (n_img / elapsed) if elapsed > 0 else 0.0
        spi = (elapsed / n_img) if n_img > 0 else 0.0

        out_json = os.path.join(self.output_folder, "overall_results.json")
        with open(out_json, "w") as f:
            json.dump(all_rows, f, indent=2)

        print(f"\n{'='*60}")
        print("GROUNDING DINO COMPLETE")
        print(f"{'='*60}")
        print(f"Detection rows: {len(all_rows)}")
        print(f"Images processed: {n_img}")
        print(f"Wall time (inference): {elapsed:.2f}s")
        print(f"Throughput: {ips:.2f} images/s")
        print(f"Avg: {spi:.3f} s/image")
        print(f"Saved: {out_json}")

# --- YOLOE (inlined) ---

def _import_yoloe():
    try:
        from ultralytics import YOLOE
        return YOLOE
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for yoloe_classifier.py. "
            "Install with: pip install ultralytics"
        ) from e

class YOLOEClassifier:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        confidence_threshold: float = 0.12,
        class_configs: Optional[List[Dict[str, Any]]] = None,
        preserve_metadata: bool = True,
        num_workers: int = 12,
        batch_size: int = 1,
        weights: str = "yoloe-26s-seg.pt",
        imgsz: Optional[int] = 640,
        use_half: bool = False,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.confidence_threshold = confidence_threshold
        self.class_configs = class_configs or []
        self.preserve_metadata = preserve_metadata
        self.num_workers = min(num_workers, 24)
        self.batch_size = max(1, batch_size)
        self.weights = weights
        self.imgsz = imgsz
        # FP16 predict() mixes badly with YOLOE/MobileCLIP text embeddings (float32) → matmul Half vs float
        self.half = bool(use_half and torch.cuda.is_available())

        # Match owlv2 post-filter heuristics
        self.max_bbox_size = 700
        self.nms_iou_threshold = 0.25
        self.nms_expand_ratio = 0.10
        self.nms_max_coverage = 0.95

        self.max_concurrent_batches = 1
        self._gpu_semaphore = Semaphore(self.max_concurrent_batches)
        self._gpu_monitor_running = False
        self._gpu_monitor_thread = None
        self._gpu_monitor_process = None
        self.gpu_log_file = None

        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None

        os.makedirs(output_folder, exist_ok=True)

        print("Initializing YOLOE classifier")
        print(f"Device: {self._device}, workers: {self.num_workers}, batch_size: {self.batch_size}")
        print(f"Weights: {self.weights}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Predict half precision (FP16): {self.half}" + (" (default off for YOLOE stability)" if not self.half else ""))
        print(f"class_configs entries: {len(self.class_configs)}")

        self._load_model()

    def _load_model(self):
        YOLOE = _import_yoloe()
        print("Loading YOLOE model...")
        t0 = time.time()
        self.model = YOLOE(self.weights)
        self.model.to(self._device)
        all_terms = self.prepare_search_terms()
        if not all_terms:
            raise ValueError("class_configs must expose at least one search_terms string")
        self._ensure_clip_importable()
        self.model.set_classes(all_terms)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print(f"Model ready in {time.time() - t0:.2f}s; text classes: {all_terms}")

    @staticmethod
    def _ensure_clip_importable() -> None:
        try:
            import clip  # noqa: F401
            return
        except Exception:
            pass

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "git+https://github.com/ultralytics/CLIP.git",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            print(f"Warning: CLIP install attempt failed: {e}")

        user_site = site.getusersitepackages()
        if user_site and user_site not in sys.path:
            sys.path.append(user_site)
        importlib.invalidate_caches()

        try:
            import clip  # noqa: F401
        except Exception as e:
            raise ImportError(
                "YOLOE requires Python package `clip` (ultralytics CLIP). "
                "Install it in the active environment and rerun."
            ) from e

    @staticmethod
    def _iou(boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a = (ax2 - ax1) * (ay2 - ay1)
        b = (bx2 - bx1) * (by2 - by1)
        return inter / (a + b - inter + 1e-9)

    def _apply_nms_and_merge(
        self, results: Dict[str, torch.Tensor], image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        if results is None or len(results.get("boxes", [])) == 0:
            return results

        W, H = image_size
        img_area = float(W * H)
        boxes = results["boxes"].detach().cpu().numpy().astype(float)
        scores = results["scores"].detach().cpu().numpy().astype(float)
        labels = results["labels"].detach().cpu().numpy().astype(int)

        keep_boxes, keep_scores, keep_labels = [], [], []

        def clip(b):
            x1 = max(0.0, min(b[0], W - 1))
            y1 = max(0.0, min(b[1], H - 1))
            x2 = max(0.0, min(b[2], W - 1))
            y2 = max(0.0, min(b[3], H - 1))
            return [x1, y1, x2, y2]

        def is_inside(a, b):
            return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]

        for lab in set(labels.tolist()):
            idxs = [i for i, L in enumerate(labels) if L == lab]
            if not idxs:
                continue
            idxs.sort(key=lambda i: float(scores[i]), reverse=True)
            suppressed = set()

            for i in idxs:
                if i in suppressed:
                    continue
                b = clip(boxes[i])
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 0 or h <= 0:
                    continue
                side = max(w, h)
                if self.max_bbox_size is not None and side > self.max_bbox_size:
                    continue
                coverage = (w * h) / (img_area + 1e-9)
                if coverage > self.nms_max_coverage:
                    continue
                if self.nms_expand_ratio > 0:
                    cx, cy = b[0] + w / 2.0, b[1] + h / 2.0
                    dw, dh = w * self.nms_expand_ratio, h * self.nms_expand_ratio
                    b = clip(
                        [
                            cx - (w / 2.0 + dw),
                            cy - (h / 2.0 + dh),
                            cx + (w / 2.0 + dw),
                            cy + (h / 2.0 + dh),
                        ]
                    )
                    w, h = b[2] - b[0], b[3] - b[1]
                    if w <= 0 or h <= 0:
                        continue
                    side = max(w, h)
                    if self.max_bbox_size is not None and side > self.max_bbox_size:
                        continue

                keep_boxes.append([round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)])
                keep_scores.append(round(float(scores[i]), 6))
                keep_labels.append(int(lab))

                for j in idxs:
                    if j == i or j in suppressed:
                        continue
                    bj = clip(boxes[j])
                    if self._iou(b, bj) >= self.nms_iou_threshold or is_inside(bj, b) or is_inside(b, bj):
                        suppressed.add(j)

        if not keep_boxes:
            return results

        dev = results["boxes"].device
        return {
            "boxes": torch.tensor(keep_boxes, device=dev, dtype=torch.float32),
            "scores": torch.tensor(keep_scores, device=dev, dtype=torch.float32),
            "labels": torch.tensor(keep_labels, device=dev, dtype=torch.int64),
        }

    def prepare_search_terms(self) -> List[str]:
        all_search_terms: List[str] = []
        for config in self.class_configs:
            all_search_terms.extend(config["search_terms"])
        return all_search_terms

    def load_and_preprocess_batch(self, image_paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
        images, valid_paths = [], []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
        return images, valid_paths

    def _result_to_torch_dict(self, result, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        if result.boxes is None or len(result.boxes) == 0:
            return None
        b = result.boxes
        return {
            "boxes": b.xyxy.clone().to(device),
            "scores": b.conf.clone().to(device),
            "labels": b.cls.long().clone().to(device),
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Optional[Dict[str, torch.Tensor]]]:
        if not images:
            return []
        device = torch.device(self._device)
        predict_kwargs: Dict[str, Any] = {
            "source": images,
            "conf": self.confidence_threshold,
            "verbose": False,
            "device": self._device,
            "half": self.half,
            "stream": False,
            "batch": min(len(images), self.batch_size),
        }
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz

        with self._gpu_semaphore:
            raw = self.model.predict(**predict_kwargs)

        if not isinstance(raw, list):
            raw = [raw]

        out: List[Optional[Dict[str, torch.Tensor]]] = []
        for r in raw:
            d = self._result_to_torch_dict(r, device)
            out.append(d)
        return out

    def draw_detections(
        self,
        image: Image.Image,
        results: Optional[Dict[str, torch.Tensor]],
        image_name: str,
        all_search_terms: List[str],
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        detections: List[Dict[str, Any]] = []
        if results is None:
            return image, detections

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            box = [round(float(i), 2) for i in box.tolist()]
            w, h = box[2] - box[0], box[3] - box[1]
            side = max(w, h)
            if self.max_bbox_size is not None and side > self.max_bbox_size:
                continue

            li = int(label.item())
            detected_term = all_search_terms[li]
            sc = round(float(score.item()), 2)
            draw.rectangle(box, outline="red", width=3)
            text = f"{detected_term}: {sc}"
            if font:
                draw.text((box[0], box[1] - 10), text, fill="red", font=font)
            else:
                draw.text((box[0], box[1] - 10), text, fill="red")

            detections.append({"label": detected_term, "score": sc, "box": box})

        return image, detections

    def process_batch_worker(self, batch_data: Tuple[List[str], List[str]]):
        image_paths, all_search_terms = batch_data
        batch_start_time = time.time()
        print(f"Processing batch of {len(image_paths)} images")

        images, valid_paths = self.load_and_preprocess_batch(image_paths)
        if not images:
            return []

        preds = self.predict_batch(images)
        all_results: List[Dict[str, Any]] = []

        for image, pred_dict, image_path in zip(images, preds, valid_paths):
            if pred_dict is None:
                image_name = os.path.basename(image_path)
                annotated_folder = os.path.join(self.output_folder, "annotated")
                os.makedirs(annotated_folder, exist_ok=True)
                out_path = os.path.join(annotated_folder, f"annotated_{image_name}")
                image.save(out_path)
                continue

            res = self._apply_nms_and_merge(pred_dict, image.size)
            image_name = os.path.basename(image_path)
            annotated_image, dets = self.draw_detections(image, res, image_name, all_search_terms)

            annotated_folder = os.path.join(self.output_folder, "annotated")
            os.makedirs(annotated_folder, exist_ok=True)
            output_image_path = os.path.join(annotated_folder, f"annotated_{image_name}")
            annotated_image.save(output_image_path)

            for det in dets:
                if det["score"] > self.confidence_threshold:
                    detected_class = None
                    prediction_value = 0
                    for config in self.class_configs:
                        if det["label"] in config["search_terms"]:
                            detected_class = config["class_name"]
                            prediction_value = config["prediction_value"]
                            break
                    if detected_class:
                        all_results.append(
                            {
                                "img": image_name,
                                "prediction": prediction_value,
                                "label": detected_class,
                                "latency": round(time.time() - batch_start_time, 2),
                                "confidence": round(det["score"], 2),
                                "bounding_box": det["box"],
                            }
                        )

        batch_time = time.time() - batch_start_time
        print(f"  - Batch processed in {batch_time:.2f}s ({len(valid_paths)} images)")
        print(f"  - Detections (above thresh): {len(all_results)} in this batch")

        return all_results

    def create_batches(self, image_files: List[str]) -> List[List[str]]:
        batches = []
        for i in range(0, len(image_files), self.batch_size):
            batches.append(image_files[i : i + self.batch_size])
        return batches

    def _start_gpu_monitoring(self):
        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU monitoring")
            return
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: nvidia-smi not available, GPU monitoring disabled")
            return

        self.gpu_log_file = os.path.join(self.output_folder, "gpu_utilization.csv")
        gpu_detailed_log = os.path.join(self.output_folder, "gpu_detailed.csv")
        print("Starting GPU monitoring...")
        print(f"GPU logs: {self.gpu_log_file}, {gpu_detailed_log}")

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
            ]
            with open(gpu_detailed_log, "w") as f:
                self._gpu_monitor_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, bufsize=1)

            self._gpu_monitor_running = True

            def monitor_thread():
                with open(self.gpu_log_file, "w") as gf:
                    gf.write(
                        "timestamp,gpu_utilization(%),memory_utilization(%),memory_used(MB),memory_total(MB),power_draw(W),temperature(C)\n"
                    )
                    while self._gpu_monitor_running:
                        try:
                            result = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                                    "--format=csv,noheader,nounits",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=2,
                            )
                            if result.returncode == 0:
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                vals = result.stdout.strip().replace(" ", "")
                                gf.write(f"{ts},{vals}\n")
                                gf.flush()
                        except Exception:
                            pass
                        time.sleep(1)

            self._gpu_monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
            self._gpu_monitor_thread.start()
            print("GPU monitoring started")
        except Exception as e:
            print(f"Warning: Could not start GPU monitoring: {e}")

    def _stop_gpu_monitoring(self):
        self._gpu_monitor_running = False
        if self._gpu_monitor_thread is not None:
            self._gpu_monitor_thread.join(timeout=2)
        if self._gpu_monitor_process is not None:
            try:
                self._gpu_monitor_process.terminate()
                self._gpu_monitor_process.wait(timeout=2)
            except Exception:
                try:
                    self._gpu_monitor_process.kill()
                except Exception:
                    pass
        if self.gpu_log_file and os.path.exists(self.gpu_log_file):
            print(f"GPU monitoring stopped. Log: {self.gpu_log_file}")

    def process_images(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("Starting YOLOE object detection...")
        overall_start = time.time()

        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.JPG",
            "*.JPEG",
            "*.png",
            "*.PNG",
            "*.bmp",
            "*.BMP",
            "*.tiff",
            "*.TIFF",
            "*.tif",
            "*.TIF",
        ]
        image_files: List[str] = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))

        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return

        print(f"Found {len(image_files)} images in {self.input_folder}")
        all_search_terms = self.prepare_search_terms()
        batches = self.create_batches(image_files)
        print(f"Created {len(batches)} batches of size {self.batch_size}")

        all_results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_batch_worker, (b, all_search_terms)): b for b in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    print(f"Error processing batch {batch}: {e}")

        overall_end = time.time()

        overall_json_path = os.path.join(self.output_folder, "overall_results.json")
        with open(overall_json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("YOLOE PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total detection rows: {len(all_results)}")
        print(f"Total time: {overall_end - overall_start:.2f} s")
        print(f"Avg time / image: {(overall_end - overall_start) / max(1, len(image_files)):.2f} s")
        print(f"Images / s: {len(image_files) / max(overall_end - overall_start, 1e-9):.2f}")
        print(f"Results saved to: {overall_json_path}")
        print(f"Annotated images: {self.output_folder}")

# --- SAM3 (inlined) ---

def _import_predictor():
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor

        return SAM3SemanticPredictor
    except ImportError as e:
        raise ImportError(
            "sam3_classifier requires ultralytics with SAM3. "
            "Try: pip install -U ultralytics"
        ) from e

class SAM3Classifier:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        confidence_threshold: float = 0.10,
        class_configs: Optional[List[Dict[str, Any]]] = None,
        preserve_metadata: bool = True,
        num_workers: int = 12,
        batch_size: int = 1,
        imgsz: int = 644,  # multiple of SAM3 max stride 14 (640 triggers Ultralytics resize warning)
        iou: float = 0.7,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_repo_id: str = "facebook/sam3",
        hf_weights_filename: str = "sam3.pt",
        max_bbox_area_frac: float = 0.40,
        max_bbox_side_frac: float = 0.90,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.confidence_threshold = confidence_threshold
        self.class_configs = class_configs or []
        self.preserve_metadata = preserve_metadata
        self.num_workers = min(num_workers, 24)
        if batch_size > 1:
            print(f"Note: SAM3 runs one image per forward; processing up to {batch_size} images per worker sequentially.")
        self.batch_size = max(1, batch_size)
        self.weights = ""  # set in _load_predictor after Hub download
        self.imgsz = imgsz
        self.iou = iou
        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id
        self.hf_weights_filename = hf_weights_filename

        if device is None:
            self._device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # No fixed 700px cap (hurts valid large regions). Drop "background" giants by image fraction instead.
        self.max_bbox_size = None
        self.max_bbox_area_frac = max(0.01, min(max_bbox_area_frac, 0.99))
        self.max_bbox_side_frac = max(0.10, min(max_bbox_side_frac, 1.0))
        self.nms_iou_threshold = 0.25
        self.nms_expand_ratio = 0.10
        self.nms_max_coverage = 0.99

        self.max_concurrent_batches = 1
        self._gpu_semaphore = Semaphore(self.max_concurrent_batches)
        self._gpu_monitor_running = False
        self._gpu_monitor_thread = None
        self._gpu_monitor_process = None
        self.gpu_log_file = None

        self._predictor = None

        os.makedirs(output_folder, exist_ok=True)

        print("Initializing SAM3 semantic classifier")
        print(f"Device: {self._device}, workers: {self.num_workers}, batch_size: {self.batch_size}")
        print(
            f"Weights: Hugging Face {self.hf_repo_id!r} / {self.hf_weights_filename!r} (token required)"
        )
        print(f"conf: {confidence_threshold}, imgsz: {imgsz}, iou: {iou}")
        print(
            f"SAM3 skip if box area > {self.max_bbox_area_frac:.0%} of image "
            f"or max side > {self.max_bbox_side_frac:.0%} of longest image side"
        )

        self._load_predictor()

    def _sam3_box_too_large(self, b: List[float], W: int, H: int) -> bool:
        """Reject near-full-frame / half-frame false positives; keep small weed boxes."""
        w = b[2] - b[0]
        h = b[3] - b[1]
        if w <= 0 or h <= 0:
            return True
        img_area = float(W * H)
        if (w * h) / (img_area + 1e-9) > self.max_bbox_area_frac:
            return True
        side_im = float(max(W, H))
        if max(w, h) > self.max_bbox_side_frac * side_im + 1e-6:
            return True
        if self.max_bbox_size is not None and max(w, h) > self.max_bbox_size:
            return True
        return False

    def _load_predictor(self):
        SAM3SemanticPredictor = _import_predictor()
        if not hasattr(torch.nn, "attention"):
            raise RuntimeError(
                "SAM3 requires a newer PyTorch build that provides torch.nn.attention. "
                f"Found torch {torch.__version__}. "
                "Use torch>=2.4 (recommended with recent ultralytics SAM3) and rebuild the container."
            )
        token = (self.hf_token or os.environ.get("HF_TOKEN") or "").strip() or None
        if not token:
            raise ValueError(
                "SAM3 requires a Hugging Face token: set HF_TOKEN or pass hf_token= / --hf_token "
                "(accept the gated model on https://huggingface.co/facebook/sam3 first)."
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "SAM3 requires huggingface_hub for weight download. pip install huggingface_hub"
            ) from e
        print(
            f"SAM3: resolving {self.hf_weights_filename!r} from {self.hf_repo_id!r} "
            "(Hub cache used if already downloaded)"
        )
        wp = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=self.hf_weights_filename,
            token=token,
        )
        self.weights = os.path.abspath(os.path.expanduser(wp))
        print(f"SAM3: local weight path {self.weights!r}")
        overrides = dict(
            conf=self.confidence_threshold,
            iou=self.iou,
            task="segment",
            mode="predict",
            model=self.weights,
            half=False,
            device=self._device,
            imgsz=self.imgsz,
            save=False,
            verbose=False,
            batch=1,
        )
        t0 = time.time()
        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        print(f"SAM3SemanticPredictor ready in {time.time() - t0:.2f}s")

    @staticmethod
    def _iou(boxA, boxB):
        ax1, ay1, ax2, ay2 = boxA
        bx1, by1, bx2, by2 = boxB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a = (ax2 - ax1) * (ay2 - ay1)
        b = (bx2 - bx1) * (by2 - by1)
        return inter / (a + b - inter + 1e-9)

    def _apply_nms_and_merge(
        self, results: Dict[str, torch.Tensor], image_size: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        if results is None or len(results.get("boxes", [])) == 0:
            return results

        W, H = image_size
        img_area = float(W * H)
        boxes = results["boxes"].detach().cpu().numpy().astype(float)
        scores = results["scores"].detach().cpu().numpy().astype(float)
        labels = results["labels"].detach().cpu().numpy().astype(int)

        keep_boxes, keep_scores, keep_labels = [], [], []

        def clip(b):
            x1 = max(0.0, min(b[0], W - 1))
            y1 = max(0.0, min(b[1], H - 1))
            x2 = max(0.0, min(b[2], W - 1))
            y2 = max(0.0, min(b[3], H - 1))
            return [x1, y1, x2, y2]

        def is_inside(a, b):
            return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]

        for lab in set(labels.tolist()):
            idxs = [i for i, L in enumerate(labels) if L == lab]
            if not idxs:
                continue
            idxs.sort(key=lambda i: float(scores[i]), reverse=True)
            suppressed = set()

            for i in idxs:
                if i in suppressed:
                    continue
                b = clip(boxes[i])
                w, h = b[2] - b[0], b[3] - b[1]
                if w <= 0 or h <= 0:
                    continue
                side = max(w, h)
                if self.max_bbox_size is not None and side > self.max_bbox_size:
                    continue
                if self._sam3_box_too_large(b, W, H):
                    continue
                coverage = (w * h) / (img_area + 1e-9)
                if coverage > self.nms_max_coverage:
                    continue
                if self.nms_expand_ratio > 0:
                    cx, cy = b[0] + w / 2.0, b[1] + h / 2.0
                    dw, dh = w * self.nms_expand_ratio, h * self.nms_expand_ratio
                    b = clip(
                        [
                            cx - (w / 2.0 + dw),
                            cy - (h / 2.0 + dh),
                            cx + (w / 2.0 + dw),
                            cy + (h / 2.0 + dh),
                        ]
                    )
                    w, h = b[2] - b[0], b[3] - b[1]
                    if w <= 0 or h <= 0:
                        continue
                    side = max(w, h)
                    if self.max_bbox_size is not None and side > self.max_bbox_size:
                        continue
                    if self._sam3_box_too_large(b, W, H):
                        continue

                keep_boxes.append([round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)])
                keep_scores.append(round(float(scores[i]), 6))
                keep_labels.append(int(lab))

                for j in idxs:
                    if j == i or j in suppressed:
                        continue
                    bj = clip(boxes[j])
                    if self._iou(b, bj) >= self.nms_iou_threshold or is_inside(bj, b) or is_inside(b, bj):
                        suppressed.add(j)

        if not keep_boxes:
            return results

        dev = results["boxes"].device
        return {
            "boxes": torch.tensor(keep_boxes, device=dev, dtype=torch.float32),
            "scores": torch.tensor(keep_scores, device=dev, dtype=torch.float32),
            "labels": torch.tensor(keep_labels, device=dev, dtype=torch.int64),
        }

    def prepare_search_terms(self) -> List[str]:
        terms: List[str] = []
        for c in self.class_configs:
            terms.extend(c["search_terms"])
        return terms

    def load_batch_paths(self, image_paths: List[str]) -> List[str]:
        valid = []
        for p in image_paths:
            if os.path.isfile(p):
                valid.append(p)
            else:
                print(f"Missing file: {p}")
        return valid

    def _sam_result_to_dict(self, result) -> Optional[Dict[str, torch.Tensor]]:
        if result.boxes is None or len(result.boxes) == 0:
            return None
        b = result.boxes
        return {
            "boxes": b.xyxy.clone().cpu(),
            "scores": b.conf.clone().cpu(),
            "labels": b.cls.long().clone().cpu(),
        }

    def predict_paths(self, paths: List[str], text_terms: List[str]) -> List[Optional[Dict[str, torch.Tensor]]]:
        outputs: List[Optional[Dict[str, torch.Tensor]]] = []
        for path in paths:
            with self._gpu_semaphore:
                try:
                    raw = self._predictor(str(path), stream=False, text=text_terms)
                    if hasattr(self._predictor, "reset_image"):
                        self._predictor.reset_image()
                except Exception as e:
                    print(f"SAM3 predict failed for {path}: {e}")
                    outputs.append(None)
                    continue
            if not raw:
                outputs.append(None)
                continue
            outputs.append(self._sam_result_to_dict(raw[0]))
        return outputs

    def draw_detections(
        self,
        image: Image.Image,
        results: Optional[Dict[str, torch.Tensor]],
        all_search_terms: List[str],
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        detections: List[Dict[str, Any]] = []
        if results is None:
            return image, detections

        W, H = image.size[0], image.size[1]
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            box = [round(float(i), 2) for i in box.tolist()]
            w, h = box[2] - box[0], box[3] - box[1]
            side = max(w, h)
            if self.max_bbox_size is not None and side > self.max_bbox_size:
                continue
            if self._sam3_box_too_large(box, W, H):
                continue
            li = int(label.item())
            detected_term = all_search_terms[li] if li < len(all_search_terms) else str(li)
            sc = round(float(score.item()), 2)
            draw.rectangle(box, outline="red", width=3)
            txt = f"{detected_term}: {sc}"
            if font:
                draw.text((box[0], box[1] - 10), txt, fill="red", font=font)
            else:
                draw.text((box[0], box[1] - 10), txt, fill="red")
            detections.append({"label": detected_term, "score": sc, "box": box})

        return image, detections

    def process_batch_worker(self, batch_data: Tuple[List[str], List[str]]):
        image_paths, all_search_terms = batch_data
        batch_start = time.time()
        print(f"Processing batch of {len(image_paths)} images (SAM3 sequential forwards)")

        valid_paths = self.load_batch_paths(image_paths)
        if not valid_paths:
            return []

        pred_dicts = self.predict_paths(valid_paths, all_search_terms)
        all_rows: List[Dict[str, Any]] = []

        for path, pred_dict in zip(valid_paths, pred_dicts):
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

            if pred_dict is None:
                image_name = os.path.basename(path)
                ann = os.path.join(self.output_folder, "annotated")
                os.makedirs(ann, exist_ok=True)
                image.save(os.path.join(ann, f"annotated_{image_name}"))
                continue

            res = self._apply_nms_and_merge(pred_dict, image.size)
            image_name = os.path.basename(path)
            annotated, dets = self.draw_detections(image, res, all_search_terms)
            ann = os.path.join(self.output_folder, "annotated")
            os.makedirs(ann, exist_ok=True)
            annotated.save(os.path.join(ann, f"annotated_{image_name}"))

            for det in dets:
                if det["score"] >= self.confidence_threshold:
                    detected_class = None
                    prediction_value = 0
                    for cfg in self.class_configs:
                        if det["label"] in cfg["search_terms"]:
                            detected_class = cfg["class_name"]
                            prediction_value = cfg["prediction_value"]
                            break
                    if detected_class:
                        all_rows.append(
                            {
                                "img": image_name,
                                "prediction": prediction_value,
                                "label": detected_class,
                                "latency": round(time.time() - batch_start, 2),
                                "confidence": round(det["score"], 2),
                                "bounding_box": det["box"],
                            }
                        )

        dt = time.time() - batch_start
        n = len(valid_paths)
        ips = (n / dt) if dt > 0 else 0.0
        spi = (dt / n) if n > 0 else 0.0
        print(
            f"  - Batch done in {dt:.2f}s ({n} images, {ips:.2f} img/s, {spi:.3f} s/img)"
        )
        return all_rows

    def create_batches(self, files: List[str]) -> List[List[str]]:
        out = []
        for i in range(0, len(files), self.batch_size):
            out.append(files[i : i + self.batch_size])
        return out

    def _start_gpu_monitoring(self):
        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU monitoring")
            return
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: nvidia-smi not available, GPU monitoring disabled")
            return

        self.gpu_log_file = os.path.join(self.output_folder, "gpu_utilization.csv")
        gpu_detailed = os.path.join(self.output_folder, "gpu_detailed.csv")
        print(f"GPU logs: {self.gpu_log_file}, {gpu_detailed}")
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
            ]
            with open(gpu_detailed, "w") as f:
                self._gpu_monitor_process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, bufsize=0)
            self._gpu_monitor_running = True

            def loop():
                with open(self.gpu_log_file, "w") as gf:
                    gf.write(
                        "timestamp,gpu_utilization(%),memory_utilization(%),memory_used(MB),memory_total(MB),power_draw(W),temperature(C)\n"
                    )
                    while self._gpu_monitor_running:
                        try:
                            r = subprocess.run(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                                    "--format=csv,noheader,nounits",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=2,
                            )
                            if r.returncode == 0:
                                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                gf.write(f"{ts},{r.stdout.strip().replace(' ', '')}\n")
                                gf.flush()
                        except Exception:
                            pass
                        time.sleep(1)

            self._gpu_monitor_thread = threading.Thread(target=loop, daemon=True)
            self._gpu_monitor_thread.start()
        except Exception as e:
            print(f"GPU monitoring failed: {e}")

    def _stop_gpu_monitoring(self):
        self._gpu_monitor_running = False
        if self._gpu_monitor_thread:
            self._gpu_monitor_thread.join(timeout=2)
        if self._gpu_monitor_process:
            try:
                self._gpu_monitor_process.terminate()
                self._gpu_monitor_process.wait(timeout=2)
            except Exception:
                try:
                    self._gpu_monitor_process.kill()
                except Exception:
                    pass

    def process_images(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print("Starting SAM3 concept segmentation...")
        t0 = time.time()

        exts = [
            "*.jpg",
            "*.jpeg",
            "*.JPG",
            "*.JPEG",
            "*.png",
            "*.PNG",
            "*.bmp",
            "*.BMP",
            "*.tiff",
            "*.TIFF",
            "*.tif",
            "*.TIF",
        ]
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.input_folder, e)))

        if not files:
            print(f"No images in {self.input_folder}")
            return

        terms = self.prepare_search_terms()
        if not terms:
            print("No search_terms in class_configs")
            return

        print(f"Found {len(files)} images; text prompts: {terms}")
        batches = self.create_batches(files)
        print(f"{len(batches)} batches (chunk size {self.batch_size})")

        all_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            fmap = {ex.submit(self.process_batch_worker, (b, terms)): b for b in batches}
            for fut in as_completed(fmap):
                try:
                    all_rows.extend(fut.result())
                except Exception as e:
                    print(f"Batch error: {e}")

        t1 = time.time()

        elapsed = t1 - t0
        n_img = len(files)
        ips = (n_img / elapsed) if elapsed > 0 else 0.0
        spi = (elapsed / n_img) if n_img > 0 else 0.0

        out_json = os.path.join(self.output_folder, "overall_results.json")
        with open(out_json, "w") as f:
            json.dump(all_rows, f, indent=2)

        print(f"\n{'='*60}")
        print("SAM3 PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Detection rows: {len(all_rows)}")
        print(f"Images processed: {n_img}")
        print(f"Wall time (inference): {elapsed:.2f}s")
        print(f"Throughput: {ips:.2f} images/s")
        print(f"Avg: {spi:.3f} s/image")
        print(f"Saved: {out_json}")

def _backend_from_model_name(name: str) -> Tuple[int, str]:
    """
    Map Harvest / Patra UI display title to (internal_backend_index, label).
      1 = OWLv2, 2 = Grounding DINO, 3 = YOLOE, 4 = SAM3
    """
    raw = name.strip()
    if not raw:
        raise ValueError("model_name is empty")

    lowered = raw.lower()
    # Normalize common punctuation variants from model cards (em/en dash, etc.)
    lowered = (
        lowered.replace("—", "-")
        .replace("–", "-")
        .replace("_", " ")
    )
    compact = re.sub(r"\s+", "", lowered)
    alnum = re.sub(r"[^a-z0-9]", "", lowered)

    if (
        "sam3" in compact
        or re.search(r"sam\s*3\b", lowered)
        or "hvdsm3" in alnum
    ):
        return 4, "SAM3"
    if "yoloe" in lowered or "yoloe" in alnum:
        return 3, "YOLOE"
    if ("grounding" in lowered and "dino" in lowered) or "hvdgdn" in alnum:
        return 2, "Grounding DINO"
    if (
        "owlv2" in compact
        or "owl-v2" in lowered
        or "owlv2" in lowered.replace("-", "")
        or ("owl" in lowered and "v2" in lowered)
    ):
        return 1, "OWLv2"

    raise ValueError(
        f"Could not map model_name={raw!r} to a backend; "
        "title must contain SAM3, YOLOE, Grounding+DINO, or OWLv2 / Owl v2."
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified detector: --model_name or MODEL_NAME env "
            "(Patra card title; keywords select OWLv2, Grounding DINO, YOLOE, or SAM3)"
        )
    )
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=None,
        help=(
            "Patra/Harvest card display title; backend inferred from keywords "
            "(SAM3, OWLv2 / Owl v2, Grounding+DINO, YOLOE). "
            "If omitted, MODEL_NAME environment variable is used."
        ),
    )
    parser.add_argument("--input_folder", required=True, help="Folder containing input images")
    parser.add_argument("--output_folder", required=True, help="Folder for JSON + annotated/")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="Detection confidence (default: model-specific if omitted)",
    )
    parser.add_argument("--preserve_metadata", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--class_configs", required=True, help="Base64-encoded JSON list")

    # Grounding DINO
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="[model 2] Phrase token threshold for Grounding DINO",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="[model 2] Hugging Face model id",
    )
    parser.add_argument("--io_workers", type=int, default=8, help="[model 2] Parallel image load threads")
    parser.add_argument("--save_workers", type=int, default=4, help="[model 2] Async save threads")
    parser.add_argument("--amp", action="store_true", help="[model 2] CUDA autocast FP16")

    # YOLOE
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="[model 3] YOLOE weights (default yoloe-26s-seg.pt). SAM3 ignores this; use Hub + token.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="[model 3/4] Inference size (YOLOE: default 640, 0=omit; SAM3: default 644)",
    )
    parser.add_argument("--half", action="store_true", help="[model 3] FP16 predict (YOLOE)")

    # SAM3
    parser.add_argument("--iou", type=float, default=0.7, help="[model 4] NMS IoU")
    parser.add_argument("--device", type=str, default=None, help="[model 4] e.g. 0 or cpu")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="[model 4] Required for SAM3: Hugging Face token (or set HF_TOKEN; prefer env over CLI)",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="facebook/sam3",
        help="[model 4] Hub repo for hf_hub_download (default facebook/sam3)",
    )
    parser.add_argument(
        "--hf_weights_filename",
        type=str,
        default="sam3.pt",
        help="[model 4] Filename inside repo for hf_hub_download",
    )
    parser.add_argument(
        "--sam3_max_area_frac",
        type=float,
        default=0.40,
        help="[model 4] Drop SAM3 boxes if area exceeds this fraction of image (default 0.40; kills half-frame FPs)",
    )
    parser.add_argument(
        "--sam3_max_side_frac",
        type=float,
        default=0.90,
        help="[model 4] Drop SAM3 boxes if max(w,h) exceeds this fraction of longest image side (default 0.90)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: input folder does not exist: {args.input_folder}")
        return 1

    if isinstance(args.model_name, list):
        model_name_cli = " ".join(args.model_name).strip()
    else:
        model_name_cli = (args.model_name or "").strip()
    model_title = model_name_cli or (os.environ.get("MODEL_NAME") or "").strip()
    try:
        m, backend_label = _backend_from_model_name(model_title)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Backend from model name {model_title!r} -> internal index {m} ({backend_label})")

    try:
        raw = base64.b64decode(args.class_configs).decode("utf-8")
        class_configs = json.loads(raw)
        if not isinstance(class_configs, list):
            print("Error: class_configs must be a JSON list")
            return 1
    except Exception as e:
        print(f"Error parsing class_configs: {e}")
        return 1

    if m == 4:
        tok = (args.hf_token or os.environ.get("HF_TOKEN") or "").strip()
        if not tok:
            print(
                "Error: SAM3 (keyword in model_name) requires a Hugging Face token "
                "(export HF_TOKEN=... or pass --hf_token).",
                file=sys.stderr,
            )
            return 1

    try:
        if m == 1:
            conf = 0.1 if args.confidence_threshold is None else args.confidence_threshold
            OWLv2Classifier(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                confidence_threshold=conf,
                class_configs=class_configs,
                preserve_metadata=args.preserve_metadata,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
            ).process_images()
        elif m == 2:
            conf = 0.12 if args.confidence_threshold is None else args.confidence_threshold
            GroundingDinoClassifier(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                confidence_threshold=conf,
                text_threshold=args.text_threshold,
                class_configs=class_configs,
                preserve_metadata=args.preserve_metadata,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                model_id=args.model_id,
                io_workers=args.io_workers,
                save_workers=args.save_workers,
                use_amp=args.amp,
            ).process_images()
        elif m == 3:
            conf = 0.12 if args.confidence_threshold is None else args.confidence_threshold
            w = args.weights or "yoloe-26s-seg.pt"
            imgsz = 640 if args.imgsz is None else (None if args.imgsz == 0 else args.imgsz)
            YOLOEClassifier(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                confidence_threshold=conf,
                class_configs=class_configs,
                preserve_metadata=args.preserve_metadata,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                weights=w,
                imgsz=imgsz,
                use_half=args.half,
            ).process_images()
        else:  # m == 4
            conf = 0.10 if args.confidence_threshold is None else args.confidence_threshold
            imgsz = 644 if args.imgsz is None else args.imgsz
            SAM3Classifier(
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                confidence_threshold=conf,
                class_configs=class_configs,
                preserve_metadata=args.preserve_metadata,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                imgsz=imgsz,
                iou=args.iou,
                device=args.device,
                hf_token=args.hf_token,
                hf_repo_id=args.hf_repo_id,
                hf_weights_filename=args.hf_weights_filename,
                max_bbox_area_frac=args.sam3_max_area_frac,
                max_bbox_side_frac=args.sam3_max_side_frac,
            ).process_images()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
