"""
sam3_sahi_fss/pipeline.py — SAHI-proposal-first few-shot segmentation.

    support (image, mask) ──SAHI SAM3("visual")──▶ boxes ──label by GT overlap──┐
                                                                               │
                                     DINOv3 crop-embed ──▶ prototype set ◀──────┘
    query ──SAHI SAM3("visual")──▶ boxes ──DINOv3 crop-embed──▶ cosine vs prototype
                                                                       │
                                              keep proposals ≥ thr ────┴──▶ union mask

Same idea as ../sam3_dino_fss but (a) SAM 3 runs on overlapping slices so small
objects are found at all, and (b) the prototype set is built from proposal boxes
rather than GT connected components, so support and query embeddings match.
Frozen models, no training; the support annotation is the only label source.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from sam3_dino_fss.dino_embed import DinoEmbedder

from .prototypes import PrototypeSet, build_prototype_set, score_against
from .proposer import Sam3SahiProposer
from .slicer import Instance, paint_union

ImageLike = Union[str, Image.Image, np.ndarray]


def _load_image(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    return Image.open(img).convert("RGB")


def _resolve_amp(dtype: Optional[str], device: str) -> Optional[torch.dtype]:
    if dtype in (None, "", "float32") or device != "cuda":
        return None
    if dtype == "auto":
        return torch.bfloat16
    return getattr(torch, dtype)


class Sam3SahiDinoFSS:
    """SAHI SAM 3 proposals + DINOv3 prototype-set matching."""

    def __init__(
        self,
        dinov3: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        sam3: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        proposal_text: str = "visual",
        slice_size: int = 256,
        overlap_ratio: float = 0.2,
        nms_iou: float = 0.5,
        score_threshold: float = 0.3,
        min_area: int = 4,
        max_area_frac: float = 0.25,
        cosine_threshold: float = 0.5,
        match_mode: str = "prototype",
        knn_k: int = 5,
        subtract_negatives: bool = False,
        pos_coverage: float = 0.6,
        neg_coverage: float = 0.05,
        crop_pad: float = 0.15,
        verbose: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        amp = _resolve_amp(dtype, self.device)
        self.proposal_text = proposal_text
        self.cosine_threshold = cosine_threshold
        self.match_mode = match_mode
        self.knn_k = knn_k
        self.subtract_negatives = subtract_negatives
        self.pos_coverage = pos_coverage
        self.neg_coverage = neg_coverage
        self.crop_pad = crop_pad
        self.verbose = verbose

        self.embedder = DinoEmbedder(dinov3, self.device, amp)
        self.proposer = Sam3SahiProposer(
            sam3, self.device, amp,
            score_threshold=score_threshold, slice_size=slice_size,
            overlap_ratio=overlap_ratio, nms_iou=nms_iou,
            min_area=min_area, max_area_frac=max_area_frac, verbose=verbose)

        self.protos: Optional[PrototypeSet] = None

    # ── stage 1+2: proposals on the support → labelled by GT → prototype set ──
    def set_support(self, images: Sequence[ImageLike],
                    masks: Sequence[np.ndarray]) -> "Sam3SahiDinoFSS":
        pil = [_load_image(im) for im in images]
        np_masks = [np.asarray(m, dtype=bool) for m in masks]
        self.protos = build_prototype_set(
            pil, np_masks, self.proposer, self.embedder,
            proposal_text=self.proposal_text,
            pos_coverage=self.pos_coverage, neg_coverage=self.neg_coverage,
            collect_negatives=True, pad=self.crop_pad, verbose=self.verbose)
        return self

    # ── stage 3: same proposals on the query → cosine to the prototype set ──
    def segment(self, query: ImageLike, proposal_text: Optional[str] = None) -> dict:
        if self.protos is None:
            raise RuntimeError("Call set_support(...) before segment(...).")
        img = _load_image(query)
        W, H = img.size

        proposals: List[Instance] = self.proposer.propose(
            img, proposal_text or self.proposal_text)
        if not proposals:
            return {"mask": np.zeros((H, W), bool), "proposals": [], "boxes": [],
                    "sims": [], "kept": [], "n_proposals": 0}

        embs = self.embedder.embed(img, [p.box for p in proposals], pad=self.crop_pad)
        sims = score_against(embs, self.protos, self.match_mode, self.knn_k,
                             self.subtract_negatives)

        thr = self.cosine_threshold
        kept = [i for i, s in enumerate(sims) if s >= thr]
        return {
            "mask": paint_union([proposals[i] for i in kept], H, W),
            "proposals": proposals,
            "boxes": [p.box for p in proposals],
            "sims": sims,
            "kept": kept,
            "n_proposals": len(proposals),
        }
