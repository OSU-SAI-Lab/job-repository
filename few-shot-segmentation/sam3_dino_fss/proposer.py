"""
sam3_dino_fss/proposer.py — SAM 3 "segment everything" proposer.

SAM 3 is concept-based (no pure geometric automatic-mask-generator in the HF API),
so "everything" is driven by a generic text concept (e.g. "fertilizer granules" /
"objects"). It returns ALL matching instance masks + boxes + scores; the class
decision is deferred to the DINOv3 cosine match downstream. This keeps SAM 3 as a
class-agnostic-ish proposer and the support set as the only label source.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

Box = Tuple[int, int, int, int]


@dataclass
class Proposal:
    mask: np.ndarray            # bool (H, W)
    box: Box                    # x1, y1, x2, y2
    score: float                # SAM 3 confidence


class Sam3Proposer:
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        amp_dtype: Optional[torch.dtype] = None,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ):
        from transformers import Sam3Model, Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = amp_dtype
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def _autocast(self):
        if self.amp_dtype is not None and self.device == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    @staticmethod
    def _box_from_mask(mask: np.ndarray) -> Optional[Box]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @torch.no_grad()
    def propose(self, image: Image.Image, text: str) -> List[Proposal]:
        """Return every SAM 3 instance for the concept ``text`` in ``image``."""
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self._autocast():
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = result.get("masks", [])
        boxes = result.get("boxes", None)
        scores = result.get("scores", None)
        out: List[Proposal] = []
        for i in range(len(masks)):
            m = masks[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            m = m.astype(bool)
            if m.sum() == 0:
                continue
            box = (tuple(int(v) for v in boxes[i].tolist())
                   if boxes is not None else self._box_from_mask(m))
            if box is None:
                continue
            score = float(scores[i]) if scores is not None else 1.0
            out.append(Proposal(mask=m, box=box, score=score))
        return out
