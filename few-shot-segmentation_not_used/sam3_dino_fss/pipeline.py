"""
sam3_dino_fss/pipeline.py — proposal-first few-shot segmentation.

    class supports ──DINOv3 crop-embed──▶ class prototype
    query ──SAM3 "everything"──▶ proposals ──DINOv3 crop-embed──▶ cosine vs prototype
                                                                       │
                                              keep proposals ≥ thr ────┴──▶ union mask

Contrast with ../fss (prior-driven): there DINOv3 builds a dense prior that PROMPTS
SAM. Here SAM proposes first (class-agnostic-ish) and DINOv3 cosine matching SELECTS
which proposals are the target class. No training; all models frozen.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .class_supports import build_class_prototype
from .dino_embed import DinoEmbedder
from .proposer import Sam3Proposer, Proposal

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


class SamDinoFSS:
    """SAM 3 proposals + DINOv3 cosine matching to class supports."""

    def __init__(
        self,
        dinov3: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        sam3: str = "facebook/sam3",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        proposal_text: str = "fertilizer granules",
        cosine_threshold: float = 0.5,
        score_threshold: float = 0.3,
        support_min_area: int = 4,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        amp = _resolve_amp(dtype, self.device)
        self.proposal_text = proposal_text
        self.cosine_threshold = cosine_threshold
        self.support_min_area = support_min_area

        self.embedder = DinoEmbedder(dinov3, self.device, amp)
        self.proposer = Sam3Proposer(sam3, self.device, amp, score_threshold=score_threshold)

        self._prototype: Optional[torch.Tensor] = None
        self._instances: Optional[torch.Tensor] = None

    def set_support(self, images: Sequence[ImageLike], masks: Sequence[np.ndarray]) -> "SamDinoFSS":
        pil = [_load_image(im) for im in images]
        np_masks = [np.asarray(m, dtype=bool) for m in masks]
        self._prototype, self._instances = build_class_prototype(
            pil, np_masks, self.embedder, self.support_min_area)
        return self

    def segment(self, query: ImageLike, proposal_text: Optional[str] = None) -> dict:
        """SAM-everything → embed proposals → cosine to prototype → keep ≥ threshold."""
        if self._prototype is None:
            raise RuntimeError("Call set_support(...) before segment(...).")
        img = _load_image(query)
        W, H = img.size

        proposals: List[Proposal] = self.proposer.propose(
            img, proposal_text or self.proposal_text)
        if not proposals:
            return {"mask": np.zeros((H, W), bool), "proposals": [], "sims": [],
                    "kept": [], "n_proposals": 0}

        boxes = [p.box for p in proposals]
        embs = self.embedder.embed(img, boxes)              # (N, D)
        sims = (embs @ self._prototype).tolist()            # cosine (both normalised)

        thr = self.cosine_threshold
        kept = [i for i, s in enumerate(sims) if s >= thr]
        masks = [proposals[i].mask for i in kept]
        mask = np.logical_or.reduce(masks) if masks else np.zeros((H, W), bool)

        return {
            "mask": mask,
            "proposals": proposals,
            "boxes": boxes,
            "sims": sims,                                   # cosine per proposal
            "kept": kept,                                   # indices kept at threshold
            "n_proposals": len(proposals),
        }
