"""
sam3_sahi_fss/proposer.py — SAM 3 proposer in SAHI (sliced) mode.

SAM 3 is concept-based (no geometric automatic-mask-generator in the HF API), so
"every possible box" is driven by a deliberately generic text concept — the default
here is ``"visual"``. Running that prompt on overlapping slices instead of the whole
frame is what recovers tiny objects: a 20 px granule is ~2% of a 1024 px frame but
~8% of a 256 px slice, i.e. well above SAM 3's effective resolution floor.

    image ──slice(1024, 20% overlap)──▶ SAM3(text) per slice ──▶ instances
           ──map to global coords──▶ NMS across overlaps ──▶ proposals

The class decision is deferred entirely to the DINOv3 cosine match downstream, so the
prompt only has to be recall-oriented, not correct.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .slicer import Box, Instance, generate_slices, nms


class Sam3SahiProposer:
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        amp_dtype: Optional[torch.dtype] = None,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        slice_size: int = 256,
        overlap_ratio: float = 0.2,
        nms_iou: float = 0.5,
        min_area: int = 4,
        max_area_frac: float = 0.25,
        verbose: bool = False,
    ):
        from transformers import Sam3Model, Sam3Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = amp_dtype
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.nms_iou = nms_iou
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.verbose = verbose
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def _autocast(self):
        if self.amp_dtype is not None and self.device == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    @torch.no_grad()
    def _propose_slice(self, crop: Image.Image, text: str, origin: Box) -> List[Instance]:
        """Run SAM 3 on one slice; return instances in FULL-image coordinates."""
        inputs = self.processor(images=crop, text=text, return_tensors="pt").to(self.device)
        with self._autocast():
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = result.get("masks", [])
        scores = result.get("scores", None)
        ox, oy = origin[0], origin[1]
        slice_area = (origin[2] - origin[0]) * (origin[3] - origin[1])

        out: List[Instance] = []
        for i in range(len(masks)):
            m = masks[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            m = m.astype(bool)
            ys, xs = np.where(m)
            if len(xs) < self.min_area:
                continue
            if len(xs) > self.max_area_frac * slice_area:   # slice-wide blob, not an object
                continue
            lx1, ly1 = int(xs.min()), int(ys.min())
            lx2, ly2 = int(xs.max()) + 1, int(ys.max()) + 1
            out.append(Instance(
                mask=m[ly1:ly2, lx1:lx2].copy(),
                box=(lx1 + ox, ly1 + oy, lx2 + ox, ly2 + oy),
                score=float(scores[i]) if scores is not None else 1.0,
            ))
        return out

    def propose(self, image: Image.Image, text: str = "visual") -> List[Instance]:
        """SAHI-mode proposals: every SAM 3 instance for ``text`` over all slices."""
        W, H = image.size
        windows = generate_slices(W, H, self.slice_size, self.overlap_ratio)
        raw: List[Instance] = []
        for j, win in enumerate(windows):
            raw.extend(self._propose_slice(image.crop(win), text, win))
            if self.verbose and (j + 1) % 10 == 0:
                print(f"    [sahi] slice {j + 1}/{len(windows)} raw={len(raw)}", flush=True)
        kept = nms(raw, self.nms_iou)
        if self.verbose:
            print(f"    [sahi] {len(windows)} slices → {len(raw)} raw → {len(kept)} after NMS",
                  flush=True)
        return kept
