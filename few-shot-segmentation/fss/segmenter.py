"""
fss/segmenter.py — SAM 3 refinement (the "refine" stage).

Two entry points, both wrapping the official ``facebook/sam3`` weights:

  * ``segment_with_prompts`` — Promptable Visual Segmentation (PVS) via
    ``Sam3TrackerModel`` / ``Sam3TrackerProcessor``. Takes the point/box/mask
    prompts produced from the DINOv3 prior and returns one refined instance per
    prompt set. This is the matcher-driven path.

  * ``segment_concept`` — Promptable Concept Segmentation (PCS) via ``Sam3Model``
    / ``Sam3Processor``. Native text / image-exemplar prompting that returns ALL
    matching instances; lets a user bypass the matcher entirely.

SAM 3 uses its OWN preprocessing — kept fully separate from DINOv3's transforms.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import SegmenterConfig
from .prompts import PromptSet


@dataclass
class InstanceResult:
    """A single refined segmentation result.

    The ``candidate_*`` / ``region`` fields are only populated when
    ``segment_with_prompts(..., return_candidates=True)`` is used (eval mode), so
    callers can re-derive alternative selection/gating variants without re-running
    SAM 3.
    """

    mask: np.ndarray                       # bool (H, W)
    score: float
    box: Optional[Tuple[int, int, int, int]]  # x1, y1, x2, y2

    candidate_masks: Optional[List[np.ndarray]] = None    # raw multimask outputs
    candidate_scores: Optional[List[float]] = None        # SAM's predicted IoUs
    region: Optional[np.ndarray] = None                   # bool thresholded-prior region


def _box_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Candidate selection + gating (tasks 3 & 4). Module-level so the eval path in
# pipeline.py can reuse the exact same logic on cached candidate masks.
#
# PRINCIPLE: the prior localises and gates which instances are real; SAM owns the
# boundaries. Selection/gating may only PICK or REMOVE a candidate — they never
# grow, dilate, or reshape a mask, so the prior's blobby shape can't leak out.
# ──────────────────────────────────────────────────────────────────────────────

def _centroid_in_region(mask: np.ndarray, region: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    return bool(region[cy, cx])


def select_candidate(
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    region: Optional[np.ndarray],
    criterion: str,
    expected_area: Optional[float] = None,
    size_mult: float = 4.0,
) -> int:
    """Pick a multimask candidate index.

    'sam_score' → SAM's own predicted IoU (last-good default).
    'granule'   → keep candidates whose CENTROID lies in the prior, then prefer
                  the COMPACT one. Among those within the size prior
                  (area ≤ size_mult × expected_area) pick the highest SAM score;
                  if none qualify, pick the SMALLEST candidate. Never picks by
                  max prior overlap (that favours the over-inclusive blob).
    """
    masks = list(masks)
    scores = list(scores)
    if criterion != "granule":
        return int(np.argmax(np.asarray(scores)))

    idx = list(range(len(masks)))
    if region is not None and region.any():
        in_prior = [i for i in idx if _centroid_in_region(masks[i], region)]
        if in_prior:
            idx = in_prior

    areas = {i: int(masks[i].sum()) for i in idx}
    if expected_area and expected_area > 0:
        cap = size_mult * expected_area
        compact = [i for i in idx if areas[i] <= cap]
        if compact:
            return max(compact, key=lambda i: scores[i])     # best score among compact
    # No size prior (or all oversized): the smallest candidate is the tightest.
    return min(idx, key=lambda i: areas[i])


def gate_mask(
    mask: np.ndarray,
    region: Optional[np.ndarray],
    min_overlap: float,
    expected_area: Optional[float] = None,
    size_mult: float = 0.0,
) -> Optional[np.ndarray]:
    """Accept/reject test — returns the mask UNCHANGED if kept, or None if rejected.

    Rejects when (a) the fraction of the mask inside the UN-dilated prior region is
    below ``min_overlap`` (a hallucination off the prior), or (b) the mask area
    exceeds ``size_mult × expected_area`` when a size prior is supplied (a blob
    spanning many granules). Never dilates or reshapes the mask.
    """
    msize = int(mask.sum())
    if msize == 0:
        return None
    if region is not None and region.any() and min_overlap > 0:
        inside = int(np.logical_and(mask, region).sum())
        if inside / msize < min_overlap:
            return None
    if size_mult and expected_area and expected_area > 0:
        if msize > size_mult * expected_area:
            return None
    return mask


class SAM3Segmenter:
    """Frozen SAM 3 wrapper. The concept model is loaded lazily on first use."""

    def __init__(self, config: SegmenterConfig, device: str,
                 amp_dtype: Optional[torch.dtype] = None):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # local import

        self.config = config
        self.device = device
        # Weights stay fp32; half precision (if any) runs via autocast on forward.
        self.amp_dtype = amp_dtype

        self.tracker_processor = Sam3TrackerProcessor.from_pretrained(config.sam3)
        self.tracker = Sam3TrackerModel.from_pretrained(config.sam3).to(device).eval()

        # Concept (PCS) model is heavier and optional — load on demand.
        self._concept_model = None
        self._concept_processor = None

    def _autocast(self):
        if self.amp_dtype is not None and self.device == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    # ──────────────────────────────────────────────────────────────────────
    # PVS — matcher-driven point / box / mask prompts
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def segment_with_prompts(
        self,
        image: Image.Image,
        prompt_sets: Sequence[PromptSet],
        return_candidates: bool = False,
        expected_area: Optional[float] = None,
    ) -> List[InstanceResult]:
        """Refine each prompt set into a single best mask. One result per set.

        ``expected_area`` is the granule size prior (used by 'granule' selection
        and the size gate). ``return_candidates=True`` attaches the raw multimask
        candidates + the prompt's region to each result (used by the eval-variant
        path) and skips gating-based rejection so no instance is dropped before
        comparison.
        """
        results: List[InstanceResult] = []
        for ps in prompt_sets:
            res = self._segment_one(image, ps, return_candidates=return_candidates,
                                    expected_area=expected_area)
            if res is not None:
                results.append(res)
        return results

    def _segment_one(
        self, image: Image.Image, ps: PromptSet, return_candidates: bool = False,
        expected_area: Optional[float] = None,
    ) -> Optional[InstanceResult]:
        proc_kwargs = {"images": image, "return_tensors": "pt"}

        # Points: 4D (image, object, point, xy); labels: 3D (image, object, point).
        if ps.points:
            proc_kwargs["input_points"] = [[[list(p) for p in ps.points]]]
            proc_kwargs["input_labels"] = [[ps.labels]]
        # Box: 3D (image, num_boxes, 4).
        if ps.box is not None:
            proc_kwargs["input_boxes"] = [[list(ps.box)]]

        inputs = self.tracker_processor(**proc_kwargs).to(self.device)

        forward_kwargs = dict(multimask_output=self.config.multimask_output)
        mask_input = self._mask_prompt_tensor(ps)
        if mask_input is not None:
            forward_kwargs["input_masks"] = mask_input.to(self.device)

        with self._autocast():
            outputs = self.tracker(**inputs, **forward_kwargs)

        # masks: list per image; [0] -> (num_objects, num_masks, H, W) at full res.
        masks = self.tracker_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        ious = outputs.iou_scores[0].float().cpu()       # (num_objects, num_masks)

        # Candidate masks for the first (only) object on this prompt set.
        cand_masks = [masks[0, j].numpy().astype(bool) for j in range(masks.shape[1])]
        cand_scores = [float(ious[0, j]) for j in range(ious.shape[1])]

        # Task 3: granule-aware selection among the multimask candidates.
        sel = select_candidate(cand_masks, cand_scores, ps.region,
                               self.config.selection_criterion,
                               expected_area=expected_area,
                               size_mult=self.config.granule_size_mult)
        mask = cand_masks[sel]
        score = cand_scores[sel]

        # Tasks 1, 3 & 4: accept/reject gating (no dilation; only removes
        # instances). Skipped in eval-candidate mode, where pipeline.py applies
        # the variants itself.
        if not return_candidates and (self.config.prior_gating or self.config.size_gate):
            kept = gate_mask(
                mask, ps.region, self.config.min_prior_overlap,
                expected_area=expected_area if self.config.size_gate else None,
                size_mult=self.config.granule_size_mult if self.config.size_gate else 0.0,
            )
            if kept is None:
                return None
            mask = kept.astype(bool)

        result = InstanceResult(mask=mask, score=score, box=_box_from_mask(mask))
        if return_candidates:
            result.candidate_masks = cand_masks
            result.candidate_scores = cand_scores
            result.region = ps.region
        return result

    def _mask_prompt_tensor(self, ps: PromptSet) -> Optional[torch.Tensor]:
        """Best-effort low-res logit mask prompt; returns None if disabled.

        SAM 3's tracker accepts a previous-mask input as low-resolution logits
        (shape (B, 1, h, w)). We synthesise such logits from the coarse prior blob.
        Wrapped defensively because the exact low-res size is model-dependent.
        """
        if ps.mask is None:
            return None
        try:
            low = int(getattr(self.tracker.config.prompt_encoder_config, "image_size", 1024)) // 4
        except Exception:
            low = 256
        m = torch.from_numpy(ps.mask.astype("float32"))[None, None]
        m = F.interpolate(m, size=(low, low), mode="bilinear", align_corners=False)
        return (m * 20.0 - 10.0)            # ~[-10, +10] logits, shape (1, 1, low, low)

    # ──────────────────────────────────────────────────────────────────────
    # PCS — native concept prompting (text / image exemplars)
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_concept(self):
        if self._concept_model is None:
            from transformers import Sam3Model, Sam3Processor  # local import
            self._concept_processor = Sam3Processor.from_pretrained(self.config.sam3)
            self._concept_model = (
                Sam3Model.from_pretrained(self.config.sam3).to(self.device).eval()
            )

    @torch.no_grad()
    def segment_concept(
        self,
        image: Image.Image,
        text: Optional[str] = None,
        exemplar_boxes: Optional[List[List[float]]] = None,
        exemplar_labels: Optional[List[int]] = None,
    ) -> List[InstanceResult]:
        """Native SAM 3 PCS: text and/or box exemplars -> all matching instances."""
        if text is None and exemplar_boxes is None:
            raise ValueError("segment_concept requires `text` and/or `exemplar_boxes`.")
        self._ensure_concept()

        proc_kwargs = {"images": image, "return_tensors": "pt"}
        if text is not None:
            proc_kwargs["text"] = text
        if exemplar_boxes is not None:
            proc_kwargs["input_boxes"] = [exemplar_boxes]
            proc_kwargs["input_boxes_labels"] = [exemplar_labels or [1] * len(exemplar_boxes)]

        inputs = self._concept_processor(**proc_kwargs).to(self.device)
        with self._autocast():
            outputs = self._concept_model(**inputs)
        result = self._concept_processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.concept_threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        out: List[InstanceResult] = []
        masks = result.get("masks", [])
        boxes = result.get("boxes", None)
        scores = result.get("scores", None)
        for i in range(len(masks)):
            m = masks[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
            m = m.astype(bool)
            box = (
                tuple(int(v) for v in boxes[i].tolist())
                if boxes is not None else _box_from_mask(m)
            )
            score = float(scores[i]) if scores is not None else 1.0
            out.append(InstanceResult(mask=m, score=score, box=box))
        return out
