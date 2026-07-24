"""
fss/pipeline.py — FewShotSegmenter orchestration.

    seg = FewShotSegmenter(dinov3="base", sam3="facebook/sam3", device=...)
    seg.set_support(images=[...], masks=[...])   # 1..K (image, binary mask) pairs
    result = seg.segment(query_image)            # -> {"mask","score","box","prior",...}

``set_support`` caches DINOv3 prototypes so many queries reuse them. ``segment``
runs DINOv3 match -> prompt derivation -> SAM 3 refinement and also returns the
intermediate prior map and prompts for debugging / visualisation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from .config import FSSConfig
from .matcher import DINOv3Matcher, SupportPrototypes
from .postprocess import prior_only_mask
from .prompts import PromptGenerator, PromptSet
from .segmenter import (
    InstanceResult,
    SAM3Segmenter,
    gate_mask,
    select_candidate,
)

ImageLike = Union[str, Image.Image, np.ndarray]
MaskLike = Union[str, Image.Image, np.ndarray]


def _load_image(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    return Image.open(img).convert("RGB")


def _load_mask(mask: MaskLike) -> np.ndarray:
    """Return a 2D boolean mask."""
    if isinstance(mask, np.ndarray):
        arr = mask
    elif isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    else:
        arr = np.array(Image.open(mask).convert("L"))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > (0.5 if arr.max() <= 1 else 127)


def _granule_size_prior(masks: Sequence[np.ndarray]) -> Optional[float]:
    """Median connected-component area (px) across the support masks — the
    expected single-granule area used by 'granule' selection and the size gate.
    """
    from scipy import ndimage

    areas: List[float] = []
    structure = np.ones((3, 3), dtype=bool)          # 8-connectivity
    for m in masks:
        labels, n = ndimage.label(m, structure=structure)
        if n == 0:
            continue
        comp_areas = ndimage.sum(np.ones_like(m), labels, index=range(1, n + 1))
        areas.extend(float(a) for a in np.atleast_1d(comp_areas))
    if not areas:
        return None
    return float(np.median(areas))


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_amp_dtype(name: Optional[str], device: str) -> Optional[torch.dtype]:
    """Resolve config.dtype to an autocast dtype (or None for full fp32).

    'auto'   -> bfloat16 on CUDA (great on H100), fp32 on CPU.
    'float16'/'bfloat16' -> that dtype, but only honoured on CUDA.
    None/'float32' -> None (no autocast).
    """
    if name in (None, "", "float32"):
        return None
    if device != "cuda":
        return None  # half-precision autocast is only meaningful on CUDA
    if name == "auto":
        return torch.bfloat16
    return getattr(torch, name)


class FewShotSegmenter:
    """Training-free few-shot segmentation: DINOv3 match-first, SAM 3 refine."""

    def __init__(
        self,
        dinov3: Optional[str] = None,
        sam3: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[FSSConfig] = None,
    ):
        self.config = config or FSSConfig()
        if dinov3 is not None:
            self.config.matcher.dinov3 = dinov3
        if sam3 is not None:
            self.config.segmenter.sam3 = sam3
        if device is not None:
            self.config.device = device

        self.device = _resolve_device(self.config.device)
        self.amp_dtype = _resolve_amp_dtype(self.config.dtype, self.device)

        self.matcher = DINOv3Matcher(self.config.matcher, self.device, self.amp_dtype)
        self.prompt_generator = PromptGenerator(self.config.prompts)
        self.segmenter = SAM3Segmenter(self.config.segmenter, self.device, self.amp_dtype)

        self._protos: Optional[SupportPrototypes] = None
        self._expected_area: Optional[float] = None  # granule size prior (px)
        self._support_images: Optional[List[Image.Image]] = None  # for alignment check
        self._support_masks: Optional[List[np.ndarray]] = None     # for alignment check

    # ──────────────────────────────────────────────────────────────────────

    def set_support(
        self, images: Sequence[ImageLike], masks: Sequence[MaskLike]
    ) -> "FewShotSegmenter":
        """Compute and cache prototypes from 1..K (image, binary mask) pairs."""
        if len(images) != len(masks):
            raise ValueError(f"Got {len(images)} images but {len(masks)} masks.")
        if len(images) == 0:
            raise ValueError("At least one support (image, mask) pair is required.")
        pil_images = [_load_image(im) for im in images]
        np_masks = [_load_mask(m) for m in masks]
        self._protos = self.matcher.build_prototypes(pil_images, np_masks)
        self._expected_area = _granule_size_prior(np_masks)
        # Keep the raw support (image, mask) pairs for the reverse alignment pass.
        self._support_images = pil_images
        self._support_masks = np_masks
        return self

    def segment(self, query_image: ImageLike) -> dict:
        """Run the full pipeline; return final mask + intermediate prior/prompts."""
        if self._protos is None:
            raise RuntimeError("Call set_support(...) before segment(...).")

        query = _load_image(query_image)
        gray = np.asarray(query.convert("L"))
        prior = self.matcher.compute_prior(query, self._protos)         # (H, W) float
        prompt_sets: List[PromptSet] = self.prompt_generator.generate(
            prior, gray=gray, expected_area=self._expected_area
        )

        eval_variants = self.config.eval_variants
        instances: List[InstanceResult] = self.segmenter.segment_with_prompts(
            query, prompt_sets, return_candidates=eval_variants,
            expected_area=self._expected_area,
        )

        # DINOv3 feature verification: keep only SAM masks whose pooled DINOv3
        # feature matches the class-support prototype (cosine fg−bg). Filters masks
        # that drifted onto soil/non-class regions before they enter the union.
        cosine_sims: Optional[List[float]] = None
        if self.config.cosine_verify and instances:
            cosine_sims = self.matcher.mask_cosine_similarities(
                query, [r.mask for r in instances], self._protos)
            thr = self.config.cosine_verify_threshold
            keep = [i for i, s in enumerate(cosine_sims) if s >= thr]
            instances = [instances[i] for i in keep]
            cosine_sims = [cosine_sims[i] for i in keep]

        result = self._assemble(query, prior, prompt_sets, instances)
        if cosine_sims is not None:
            result["cosine_sims"] = cosine_sims

        # Final-assembly safety: intersect with the UN-dilated prior (never dilated).
        if self.config.segmenter.intersect_prior_safety and prior is not None:
            fg = prior >= self.config.prompts.threshold
            result["mask"] = np.logical_and(result["mask"], fg)
            result["masks"] = [np.logical_and(m, fg) for m in result["masks"]]

        if eval_variants:
            result["variant_masks"] = self._variant_masks(query, prior, instances)
        result["route_summary"] = self._route_summary(prompt_sets)
        if self.config.alignment_check:
            result["alignment"] = self._alignment_score(query, result["mask"])
        result["active_flags"] = self._active_flags()
        return result

    def _alignment_score(self, query: Image.Image, pred_mask: np.ndarray) -> dict:
        """Reverse prototype-alignment consistency (PANet-PAR idea, inference-only).

        Build a prototype from the PREDICTED query mask, re-segment every cached
        support image, and score reverse-IoU against the known support mask. A
        high reverse-IoU means the prediction captured a prototype that round-trips
        back to the support — a label-free confidence signal. Returns mean / per-
        support reverse-IoU; never modifies the forward mask.
        """
        from .eval import binary_iou

        thr = self.config.prompts.threshold
        out: dict = {"reverse_iou": 0.0, "per_support": []}
        if (self._support_images is None or not self._support_images
                or int(pred_mask.sum()) == 0):
            return out                                  # empty prediction → no signal

        rev_protos = self.matcher.build_prototypes([query], [pred_mask.astype(bool)])
        ious: List[float] = []
        for sup_img, sup_mask in zip(self._support_images, self._support_masks):
            rev_prior = self.matcher.compute_prior(sup_img, rev_protos)
            rev_pred = rev_prior >= thr
            fg_iou, _ = binary_iou(rev_pred, sup_mask)
            ious.append(float(fg_iou))
        out["per_support"] = ious
        out["reverse_iou"] = float(np.mean(ious)) if ious else 0.0
        return out

    @staticmethod
    def _route_summary(prompt_sets: List[PromptSet]) -> Optional[dict]:
        """Per-image routing tally (None when the router is off).

        Counts components routed dense vs scattered (each component contributes one
        box prompt set, or N per-granule sets sharing one density dict) so the eval
        can break results down by regime.
        """
        if not any(ps.route is not None for ps in prompt_sets):
            return None
        comps: dict = {}
        for ps in prompt_sets:
            if ps.route is None or ps.density is None:
                continue
            key = (round(ps.density["comp_area"], 1), ps.density["n_seeds"])
            comps[key] = ps.route                      # one entry per source component
        dense = sum(1 for r in comps.values() if r == "dense")
        scattered = sum(1 for r in comps.values() if r == "scattered")
        return {"n_components": len(comps), "dense": dense, "scattered": scattered}

    def _active_flags(self) -> dict:
        """Surface which downstream behaviours are ON (acceptance: be explicit)."""
        p, s = self.config.prompts, self.config.segmenter
        return {
            "router": p.router,
            "router_density_metric": p.router_density_metric,
            "router_seed_coverage_threshold": p.router_seed_coverage_threshold,
            "per_granule_prompts": p.per_granule_prompts,
            "component_prompts": p.component_prompts,
            "use_box_prompts": p.use_box_prompts,
            "use_mask_prompt": p.use_mask_prompt,
            "selection_criterion": s.selection_criterion,
            "prior_gating": s.prior_gating,
            "size_gate": s.size_gate,
            "intersect_prior_safety": s.intersect_prior_safety,
            "alignment_check": self.config.alignment_check,
            "cosine_verify": self.config.cosine_verify,
            "cosine_verify_threshold": self.config.cosine_verify_threshold,
            "expected_granule_area": self._expected_area,
            "output_is_prior_only": False,
        }

    # ──────────────────────────────────────────────────────────────────────

    def _variant_masks(
        self,
        query: Image.Image,
        prior: Optional[np.ndarray],
        instances: List[InstanceResult],
    ) -> dict:
        """Diagnostic comparison masks derived from one SAM 3 run (task 6):

        (a) prior_only       — threshold + cleanup of the prior, no SAM 3.
        (b) sam3             — SAM-score selection, no gating (over-inclusive ref).
        (c) granule_sam3     — granule (compact, centroid-in-prior) selection +
                               size gate; the regression fix's output.
        """
        w, h = query.size
        seg = self.config.segmenter
        exp = self._expected_area

        # (a) prior_only
        if prior is not None:
            gray = np.asarray(query.convert("L"))
            prior_only = prior_only_mask(prior, self.config.prior_mask, gray)
        else:
            prior_only = np.zeros((h, w), dtype=bool)

        # (b) & (c) re-derived from cached multimask candidates per instance.
        sam_masks: List[np.ndarray] = []
        granule_masks: List[np.ndarray] = []
        for r in instances:
            if r.candidate_masks is None:           # nothing cached → use final mask
                sam_masks.append(r.mask)
                granule_masks.append(r.mask)
                continue
            i_sam = select_candidate(
                r.candidate_masks, r.candidate_scores, r.region, "sam_score"
            )
            sam_masks.append(r.candidate_masks[i_sam])

            i_gran = select_candidate(
                r.candidate_masks, r.candidate_scores, r.region, "granule",
                expected_area=exp, size_mult=seg.granule_size_mult,
            )
            kept = gate_mask(
                r.candidate_masks[i_gran], r.region, seg.min_prior_overlap,
                expected_area=exp, size_mult=seg.granule_size_mult,
            )
            if kept is not None:
                granule_masks.append(kept)

        sam3 = (np.logical_or.reduce(sam_masks) if sam_masks
                else np.zeros((h, w), dtype=bool))
        granule = (np.logical_or.reduce(granule_masks) if granule_masks
                   else np.zeros((h, w), dtype=bool))

        # Safety intersect with the un-dilated prior (never dilated).
        if prior is not None:
            fg = prior >= self.config.prompts.threshold
            granule = np.logical_and(granule, fg)

        return {
            "prior_only": prior_only.astype(bool),
            "sam3": sam3.astype(bool),
            "granule_sam3": granule.astype(bool),
        }

    def segment_concept(self, query_image: ImageLike, **kwargs) -> dict:
        """Bypass the matcher and use SAM 3's native concept prompting (PCS).

        kwargs forwarded to SAM3Segmenter.segment_concept: text=..., and/or
        exemplar_boxes=[[x1,y1,x2,y2], ...], exemplar_labels=[1, 0, ...].
        """
        query = _load_image(query_image)
        instances = self.segmenter.segment_concept(query, **kwargs)
        return self._assemble(query, None, [], instances)

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _assemble(
        query: Image.Image,
        prior: Optional[np.ndarray],
        prompt_sets: List[PromptSet],
        instances: List[InstanceResult],
    ) -> dict:
        w, h = query.size
        masks = [r.mask for r in instances]
        scores = [r.score for r in instances]
        boxes = [r.box for r in instances]

        if masks:
            combined = np.logical_or.reduce(masks)
            best_i = int(np.argmax(scores))
        else:
            combined = np.zeros((h, w), dtype=bool)
            best_i = -1

        return {
            "mask": combined,                                  # union of instances
            "masks": masks,                                    # per-instance masks
            "score": float(scores[best_i]) if masks else 0.0,  # best instance score
            "scores": scores,
            "box": boxes[best_i] if masks else None,
            "boxes": boxes,
            "prior": prior,                                    # (H, W) or None
            "prompts": prompt_sets,
        }
