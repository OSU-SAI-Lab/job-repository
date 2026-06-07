"""
proposal/embedding_utils.py  –  Unified embedding interface.

Supported backends
------------------
  bioclip  – BioCLIP  (ViT-B/16 CLIP, dim=512)
  dinov3   – DINOv3   (ViT-L pooler_output, dim=1024 default)
  owlv2    – OWLv2    (vision backbone pooler_output, dim=1024)

Design principles
-----------------
- ALL backends produce embeddings in the SAME space used by class_supports_utils.py.
  This is the fundamental contract: class supports and proposals MUST share the
  same vector space for cosine similarity to be valid.

- OWLv2: uses `owlv2.vision_model.pooler_output` (dim=1024) — NOT class_predictor
  (dim=512). Both supports and proposals use this identical path.

- Tensors remain on GPU throughout batch loops; a single .cpu() call happens
  once after concatenation, minimising CPU<->GPU transfers.

- Models are loaded once per process. Instantiate with get_embedder() and reuse.

Public API
----------
get_embedder(backend, **kwargs) -> BaseEmbedder
BaseEmbedder.embed(images)      -> Tensor (N, D) CPU float32 L2-normed
BaseEmbedder.embed_boxes(image, boxes) -> Tensor (N, D) CPU float32 L2-normed
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Return CPU float32 L2-normed tensor (N, D)."""

    def embed_boxes(self, image: Image.Image, boxes: List[List[float]]) -> torch.Tensor:
        """Crop boxes from image and embed. Returns (N, D) CPU float32."""
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            crops.append(image.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1
                         else Image.new("RGB", (1, 1)))
        return self.embed(crops)

    def embed_masks(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        masks: List,
        background: str = "zero",
    ) -> torch.Tensor:
        """
        Crop each box's tight region from `image`, zero out (or mean-fill) the
        pixels *outside* its segmentation mask, and embed.  This is the
        mask-aware counterpart of embed_boxes() and is what makes the pipeline a
        few-shot *segmentation* (not box) matcher: the embedding describes the
        object's pixels, not its bounding rectangle plus background.

        `masks[i]` must be a (H, W) boolean array in the SAME coordinate frame as
        `image` (i.e. the tile/full image the box was detected in).  If a mask is
        None or shape-incompatible, that entry falls back to a plain box crop so
        the call never fails.

        Returns (N, D) CPU float32 L2-normed — identical space to embed()/embed_boxes().
        """
        crops = [
            self._mask_crop(image, box, mask, background)
            for box, mask in zip(boxes, masks)
        ]
        return self.embed(crops)

    @staticmethod
    def _mask_crop(image: Image.Image, box, mask, background: str = "zero") -> Image.Image:
        """Crop `box` from `image` and suppress background pixels using `mask`.

        background: 'zero' → black, 'mean' → per-crop mean colour, 'none' → no
        masking (plain box crop). Falls back to a box crop on any shape mismatch.
        """
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return Image.new("RGB", (1, 1))

        crop = image.crop((x1, y1, x2, y2))
        if background == "none" or mask is None:
            return crop

        m = np.asarray(mask, dtype=bool)
        m = m[y1:y2, x1:x2]
        arr = np.array(crop)  # (h, w, 3) uint8
        if m.shape != arr.shape[:2] or not m.any():
            # Shape mismatch or empty mask → safest is the unmasked crop.
            return crop

        if background == "mean":
            fill = arr[m].mean(axis=0).astype(arr.dtype)
        else:  # 'zero'
            fill = 0
        arr[~m] = fill
        return Image.fromarray(arr)

    @staticmethod
    def _normalise(t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t.float(), dim=-1)


# ---------------------------------------------------------------------------
# BioCLIP  (dim=512)
# ---------------------------------------------------------------------------

class BioCLIPEmbedder(BaseEmbedder):
    """
    BioCLIP via bioclip.predict.BaseClassifier.
    Output dim: 512 (ViT-B/16 CLIP image encoder).
    """

    def __init__(self, batch_size: int = 32, **_):
        from bioclip.predict import BaseClassifier  # type: ignore
        self.batch_size = batch_size
        self._clf = BaseClassifier(device=DEVICE)
        self._clf.model.eval()
        logger.info(f"Loaded BioCLIP ViT-B/16 on {DEVICE} | dim=512")

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                emb = self._clf.create_image_features(
                    images[i: i + self.batch_size], normalize=False
                )  # (B, 512) on model device
                all_embs.append(emb)
        # Single normalise + single .cpu() transfer
        return self._normalise(torch.cat(all_embs, dim=0)).cpu()


# ---------------------------------------------------------------------------
# DINOv3  (dim=1024 for vitl, 384 for vits)
# ---------------------------------------------------------------------------

class DINOv3Embedder(BaseEmbedder):
    """
    DINOv3 via HuggingFace AutoModel (pooler_output).
    Default model: facebook/dinov3-vitl16-pretrain-lvd1689m (dim=1024).

    The model_id MUST match what class_supports_utils uses so that
    supports and proposals share the same embedding space.
    """

    DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, batch_size: int = 32, **_):
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        self.batch_size = batch_size
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
        dim = self._model.config.hidden_size
        logger.info(f"Loaded {model_id} on {DEVICE} | dim={dim}")

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                pv = self._processor(
                    images=images[i: i + self.batch_size], return_tensors="pt"
                )["pixel_values"].to(DEVICE)
                out = self._model(pixel_values=pv)
                all_embs.append(out.pooler_output)  # (B, D) stays on GPU
        return self._normalise(torch.cat(all_embs, dim=0)).cpu()


# ---------------------------------------------------------------------------
# OWLv2  (vision backbone, dim=1024)
# ---------------------------------------------------------------------------

class OWLv2Embedder(BaseEmbedder):
    """
    OWLv2 image embeddings using owlv2.vision_model.pooler_output (dim=1024).

    We bypass class_predictor entirely — that head projects into a 512-dim
    text-alignment space unsuitable for image-to-image cosine similarity.
    Using pooler_output ensures supports and proposals are in the same space.
    """

    DEFAULT_MODEL_ID = "google/owlv2-large-patch14-ensemble"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, batch_size: int = 16, **_):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection  # type: ignore
        self.batch_size = batch_size
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = (
            Owlv2ForObjectDetection
            .from_pretrained(model_id, torch_dtype=TORCH_DTYPE)
            .to(DEVICE)
            .eval()
        )
        logger.info(f"Loaded {model_id} on {DEVICE} | vision backbone pooler_output dim=1024")

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                pv = self._processor(
                    images=images[i: i + self.batch_size], return_tensors="pt"
                )["pixel_values"].to(DEVICE, dtype=TORCH_DTYPE)
                # Vision backbone only — pooler_output (B, 1024)
                out = self._model.owlv2.vision_model(pixel_values=pv)
                all_embs.append(out.pooler_output.float())  # keep on GPU
        return self._normalise(torch.cat(all_embs, dim=0)).cpu()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKEND_MAP = {
    "bioclip": BioCLIPEmbedder,
    "dinov3":  DINOv3Embedder,
    "owlv2":   OWLv2Embedder,
}
BACKENDS = list(_BACKEND_MAP.keys())


def get_embedder(backend: str, **kwargs) -> BaseEmbedder:
    """
    Instantiate and return an embedder for the given backend.

    Args:
        backend : 'bioclip' | 'dinov3' | 'owlv2'
        **kwargs: Forwarded to the embedder (e.g. model_id, batch_size).
    """
    key = backend.lower()
    if key not in _BACKEND_MAP:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {BACKENDS}")
    return _BACKEND_MAP[key](**kwargs)
