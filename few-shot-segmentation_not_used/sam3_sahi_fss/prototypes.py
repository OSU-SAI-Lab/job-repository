"""
sam3_sahi_fss/prototypes.py — build the DINOv3 prototype set from support images.

The prototype set is built from the SAME proposal machinery used at query time: SAHI
SAM 3 proposes boxes on the support image, and the support MASK is only used to LABEL
those boxes (positive if the proposal is mostly inside the annotation). This keeps the
support and query embeddings in the same distribution — the alternative (embedding GT
connected components directly, as ../sam3_dino_fss does) builds prototypes from boxes
that SAM 3 would never have produced, so cosine scores are systematically offset.

Returns a positive bank (M, D) + its mean prototype, and optionally a negative bank
from the proposals that fell outside the annotation (free hard negatives — background
is exactly what a generic "visual" prompt over-produces).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3_dino_fss.dino_embed import DinoEmbedder

from .proposer import Sam3SahiProposer
from .slicer import Instance


@dataclass
class PrototypeSet:
    prototype: torch.Tensor                     # (D,) L2-normalised mean of positives
    positives: torch.Tensor                     # (M, D) per-instance positive bank
    negatives: Optional[torch.Tensor] = None    # (K, D) background bank, if collected

    @property
    def n_pos(self) -> int:
        return int(self.positives.shape[0])

    @property
    def n_neg(self) -> int:
        return 0 if self.negatives is None else int(self.negatives.shape[0])


def _coverage(ins: Instance, gt: np.ndarray) -> float:
    """Fraction of the proposal's pixels that fall inside the GT mask."""
    x1, y1, x2, y2 = ins.box
    inter = int(np.logical_and(ins.mask, gt[y1:y2, x1:x2]).sum())
    return inter / max(1, ins.area)


def build_prototype_set(
    images: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    proposer: Sam3SahiProposer,
    embedder: DinoEmbedder,
    proposal_text: str = "visual",
    pos_coverage: float = 0.6,
    neg_coverage: float = 0.05,
    collect_negatives: bool = True,
    pad: float = 0.15,
    verbose: bool = False,
) -> PrototypeSet:
    """Propose on each support image, label proposals by GT overlap, embed both banks."""
    pos: List[torch.Tensor] = []
    neg: List[torch.Tensor] = []

    for img, m in zip(images, masks):
        gt = np.asarray(m, dtype=bool)
        instances = proposer.propose(img, proposal_text)
        cov = [_coverage(ins, gt) for ins in instances]
        pos_boxes = [ins.box for ins, c in zip(instances, cov) if c >= pos_coverage]
        neg_boxes = [ins.box for ins, c in zip(instances, cov) if c <= neg_coverage]
        if verbose:
            print(f"  [support] proposals={len(instances)} pos={len(pos_boxes)} "
                  f"neg={len(neg_boxes)}", flush=True)
        if pos_boxes:
            pos.append(embedder.embed(img, pos_boxes, pad=pad))
        if collect_negatives and neg_boxes:
            neg.append(embedder.embed(img, neg_boxes, pad=pad))

    if not pos:
        raise ValueError(
            "No support proposal overlapped the annotation "
            f"(pos_coverage={pos_coverage}). Lower it, lower --score-threshold, or "
            "shrink --slice-size so SAM 3 actually finds the objects.")

    positives = torch.cat(pos, 0)
    negatives = torch.cat(neg, 0) if neg else None
    prototype = F.normalize(positives.mean(0), dim=-1)
    return PrototypeSet(prototype=prototype, positives=positives, negatives=negatives)


def score_against(
    embs: torch.Tensor,
    protos: PrototypeSet,
    match_mode: str = "prototype",
    knn_k: int = 5,
    subtract_negatives: bool = False,
) -> List[float]:
    """Score (N, D) query embeddings against the prototype set.

    match_mode: ``prototype`` cosine to the mean; ``knn`` mean of the top-k cosines
    against the positive bank (robust when the support instances are multi-modal).
    ``subtract_negatives`` returns a margin (positive score − best background score),
    which recentres the threshold around 0 instead of around the class-agnostic
    similarity floor.
    """
    if embs.numel() == 0:
        return []
    if match_mode == "prototype":
        sims = embs @ protos.prototype
    elif match_mode == "knn":
        k = min(knn_k, protos.n_pos)
        sims = (embs @ protos.positives.T).topk(k, dim=-1).values.mean(-1)
    else:
        raise ValueError(f"unknown match_mode {match_mode!r} (use 'prototype' or 'knn')")

    if subtract_negatives and protos.negatives is not None and protos.n_neg:
        sims = sims - (embs @ protos.negatives.T).max(dim=-1).values
    return sims.tolist()
