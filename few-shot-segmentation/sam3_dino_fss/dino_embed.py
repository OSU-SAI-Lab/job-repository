"""
sam3_dino_fss/dino_embed.py — DINOv3 crop embedder.

Embeds an image REGION (a proposal or support instance) as a single L2-normalised
vector by cropping its bounding box (with padding), letting the processor resize the
crop up to the model's input resolution, and taking the CLS token. Resizing the crop
is what makes this robust to sub-patch objects (e.g. tiny granules): the granule is
enlarged to fill the receptive field instead of being averaged into a soil patch.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

Box = Tuple[int, int, int, int]   # x1, y1, x2, y2


class DinoEmbedder:
    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: Optional[str] = None,
        amp_dtype: Optional[torch.dtype] = None,
        batch_size: int = 64,
    ):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = amp_dtype
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    def _autocast(self):
        if self.amp_dtype is not None and self.device == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    @torch.no_grad()
    def embed(self, image: Image.Image, boxes: Sequence[Box], pad: float = 0.15) -> torch.Tensor:
        """Return (N, D) L2-normalised CLS embeddings for the given boxes."""
        if not boxes:
            return torch.zeros((0, self.dim))
        W, H = image.size
        crops: List[Image.Image] = []
        for (x1, y1, x2, y2) in boxes:
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            px, py = int(round(bw * pad)), int(round(bh * pad))
            cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
            cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cy1, cx2, cy2 = x1, y1, min(W, x1 + 1), min(H, y1 + 1)
            crops.append(image.crop((cx1, cy1, cx2, cy2)).convert("RGB"))

        out: List[torch.Tensor] = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self._autocast():
                feats = self.model(**inputs).last_hidden_state[:, 0]   # CLS token (B, D)
            out.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(out, 0)
