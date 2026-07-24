"""
sam3_dino_fss/class_supports.py — build a class-support embedding from labelled
support (image, mask) pairs.

Each support mask is split into connected-component instances; each instance's
bounding box is DINOv3-crop-embedded (same embedder used on the query proposals,
so the spaces match), and the per-instance embeddings are averaged into one
L2-normalised class prototype. The raw instance embeddings are also returned (for
an optional k-NN style match instead of a single mean prototype).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from .dino_embed import DinoEmbedder, Box


def _instance_boxes(mask: np.ndarray, min_area: int) -> List[Box]:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    boxes: List[Box] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        if len(xs) < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def build_class_prototype(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    embedder: DinoEmbedder,
    min_area: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (prototype (D,), instance_embeddings (M, D)) — both L2-normalised."""
    all_embs: List[torch.Tensor] = []
    for img, m in zip(images, masks):
        boxes = _instance_boxes(np.asarray(m, dtype=bool), min_area)
        if not boxes:
            continue
        all_embs.append(embedder.embed(img, boxes))
    if not all_embs:
        raise ValueError("No support instances found to embed (check masks/min_area).")
    instances = torch.cat(all_embs, 0)                  # (M, D)
    prototype = F.normalize(instances.mean(0), dim=-1)  # (D,)
    return prototype, instances
