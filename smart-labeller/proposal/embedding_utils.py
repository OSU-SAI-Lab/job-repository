"""
proposal/embedding_utils.py  –  Unified embedding interface.

Supported backends:
  bioclip  – BioCLIP  (domain-specific vision-language model for biology)
  dinov3   – DINOv3   (self-supervised ViT from Meta)
  owlv2    – OWLv2    (open-vocabulary detection encoder from Google)

Public API
----------
get_embedder(backend: str) -> BaseEmbedder
    Returns an embedder instance for the requested backend.

BaseEmbedder.embed(images) -> torch.Tensor  shape (N, D)
    Accepts a single PIL Image or a list of PIL Images.
    Always returns a 2-D float32 CPU tensor normalised to unit length.

BaseEmbedder.embed_boxes(image, boxes) -> torch.Tensor  shape (N, D)
    Crops `boxes` ([x1,y1,x2,y2]) from `image` then calls embed().
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image

# Make sure the project root is on sys.path so sibling packages resolve
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    @abstractmethod
    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Return unit-normalised embeddings for one or more PIL images.

        Args:
            images: A single PIL Image or a list of PIL Images.

        Returns:
            Tensor of shape (N, D) on CPU, dtype float32.
        """

    def embed_boxes(
        self,
        image: Image.Image,
        boxes: List[List[float]],
    ) -> torch.Tensor:
        """
        Crop `boxes` from `image` and return their embeddings.

        Args:
            image: Full PIL Image.
            boxes: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            Tensor of shape (N, D) on CPU, dtype float32.
        """
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            if x2 > x1 and y2 > y1:
                crops.append(image.crop((x1, y1, x2, y2)))
            else:
                # Degenerate box – return a 1×1 black patch
                crops.append(Image.new("RGB", (1, 1)))
        return self.embed(crops)

    @staticmethod
    def _normalise(t: torch.Tensor) -> torch.Tensor:
        """L2-normalise along the feature dimension."""
        return F.normalize(t.float(), dim=-1)


# ---------------------------------------------------------------------------
# BioCLIP backend
# ---------------------------------------------------------------------------

class BioCLIPEmbedder(BaseEmbedder):
    """
    Embeddings from BioCLIP via bioclip.predict.BaseClassifier.

    Output dim: 512 (ViT-B/16 CLIP image encoder).
    """

    def __init__(self, batch_size: int = 32):
        from bioclip.predict import BaseClassifier  # type: ignore
        self.batch_size = batch_size
        self._clf = BaseClassifier(device=DEVICE)
        self._clf.model.eval()

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                emb = self._clf.create_image_features(batch, normalize=False)
                all_embs.append(emb.cpu())

        return self._normalise(torch.cat(all_embs, dim=0))


# ---------------------------------------------------------------------------
# DINOv3 backend
# ---------------------------------------------------------------------------

class DINOv3Embedder(BaseEmbedder):
    """
    Embeddings from a DINOv3 ViT via HuggingFace AutoModel.

    Default model : facebook/dinov3-vitl16-pretrain-lvd1689m  (dim 1024)
    Smaller option : facebook/dinov3-vits16-pretrain-lvd1689m  (dim 384)
    """

    DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, batch_size: int = 32):
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        self.batch_size = batch_size
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(DEVICE)
        self._model.eval()

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                inputs = self._processor(images=batch, return_tensors="pt").to(DEVICE)
                outputs = self._model(**inputs)
                # pooler_output: (B, D)
                all_embs.append(outputs.pooler_output.cpu())

        return self._normalise(torch.cat(all_embs, dim=0))


# ---------------------------------------------------------------------------
# OWLv2 backend
# ---------------------------------------------------------------------------

class OWLv2Embedder(BaseEmbedder):
    """
    Image embeddings from OWLv2 (Owlv2VisionModel).

    Uses the vision encoder only – no text / detection head.
    Default model : google/owlv2-large-patch14-ensemble  (dim 1024)
    """

    DEFAULT_MODEL_ID = "google/owlv2-large-patch14-ensemble"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, batch_size: int = 32):
        from transformers import Owlv2Processor, Owlv2VisionModel  # type: ignore
        self.batch_size = batch_size
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2VisionModel.from_pretrained(model_id).to(DEVICE)
        self._model.eval()

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                inputs = self._processor(images=batch, return_tensors="pt").to(DEVICE)
                outputs = self._model(**inputs)
                # pooler_output: (B, D)
                all_embs.append(outputs.pooler_output.cpu())

        return self._normalise(torch.cat(all_embs, dim=0))


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
    Return an initialised embedder for the requested backend.

    Args:
        backend: One of 'bioclip', 'dinov3', 'owlv2'.
        **kwargs: Forwarded to the embedder constructor
                  (e.g. model_id=..., batch_size=...).

    Returns:
        An instance of BaseEmbedder.

    Example:
        embedder = get_embedder("dinov3")
        embs = embedder.embed(pil_images)          # (N, 384)
        embs = embedder.embed_boxes(image, boxes)  # (N, 384)
    """
    backend = backend.lower()
    if backend not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown embedding backend '{backend}'. "
            f"Choose from: {BACKENDS}"
        )
    return _BACKEND_MAP[backend](**kwargs)
