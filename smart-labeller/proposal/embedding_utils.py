"""
proposal/embedding_utils.py  –  DINOv3 embedding interface (segmentation pipeline).

Only ONE backend exists: DINOv3 (ViT-L pooler_output, dim=1024).  OWLv2 and
BioCLIP were removed when the pipeline became SAM3 (masks) + DINOv3 (embeddings)
only.

Embedding-space contract
-------------------------
Class supports and proposals MUST be embedded through the identical path so that
cosine similarity at classification time is valid.  That path is:

    mask → mask_extent_crop (zero background, crop to the mask's tight extent)
         → DINOv3 pooler_output → L2-normalise

``class_supports_utils.py`` uses the SAME DINOv3 + ``mask_extent_crop`` path.

Public API
----------
get_embedder(backend="dinov3", **kwargs) -> DINOv3Embedder
DINOv3Embedder.embed(images)             -> Tensor (N, D) CPU float32 L2-normed
DINOv3Embedder.embed_masks(image, masks) -> Tensor (N, D) CPU float32 L2-normed
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Sequence, Union

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from proposal.mask_utils import mask_extent_crop

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ---------------------------------------------------------------------------
# DINOv3  (dim=1024 for vitl16)
# ---------------------------------------------------------------------------

class DINOv3Embedder:
    """
    DINOv3 via HuggingFace AutoModel (pooler_output).
    Default model: facebook/dinov3-vitl16-pretrain-lvd1689m (dim=1024).

    The model_id MUST match what class_supports_utils uses so that supports and
    proposals share the same embedding space.
    """

    DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, batch_size: int = 32, **_):
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
        self.batch_size = batch_size
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
        dim = self._model.config.hidden_size
        logger.info(f"Loaded {model_id} on {DEVICE} | dim={dim}")

    @staticmethod
    def _normalise(t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t.float(), dim=-1)

    def embed(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Return CPU float32 L2-normed tensor (N, D) for a list of PIL crops."""
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return torch.empty((0, self._model.config.hidden_size), dtype=torch.float32)
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                pv = self._processor(
                    images=images[i: i + self.batch_size], return_tensors="pt"
                )["pixel_values"].to(DEVICE)
                out = self._model(pixel_values=pv)
                all_embs.append(out.pooler_output)  # (B, D) stays on GPU
        return self._normalise(torch.cat(all_embs, dim=0)).cpu()

    def embed_masks(
        self,
        image: Image.Image,
        masks: Sequence[np.ndarray],
    ) -> torch.Tensor:
        """Embed each instance mask: zero background, crop to the mask extent, embed.

        Args:
            image: the full PIL image (or tile) the masks are defined on.
            masks: sequence of (H, W) boolean arrays in that image's coords.

        Returns (N, D) CPU float32 L2-normed.
        """
        crops = [mask_extent_crop(image, np.asarray(m).astype(bool)) for m in masks]
        return self.embed(crops)


# ---------------------------------------------------------------------------
# Factory (kept for API compatibility; DINOv3 is the only backend)
# ---------------------------------------------------------------------------

_BACKEND_MAP = {"dinov3": DINOv3Embedder}
BACKENDS = list(_BACKEND_MAP.keys())


def get_embedder(backend: str = "dinov3", **kwargs) -> DINOv3Embedder:
    """Instantiate the embedder for ``backend`` (only 'dinov3' is supported)."""
    key = backend.lower()
    if key not in _BACKEND_MAP:
        raise ValueError(f"Unknown backend '{backend}'. Only 'dinov3' is supported.")
    return _BACKEND_MAP[key](**kwargs)
