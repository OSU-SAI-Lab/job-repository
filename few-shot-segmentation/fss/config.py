"""
fss/config.py — configuration for the training-free few-shot segmentation pipeline.

Everything that changes behaviour (model variants, prototype mode, prompt
thresholds, device/dtype) lives here as plain dataclasses so that swapping a
DINOv3 / SAM 3 variant is config-only and CLI flags map 1:1 onto fields.

All models are FROZEN — these configs only affect inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# DINOv3 backbone variants (official Meta release on Hugging Face)
#   https://huggingface.co/docs/transformers/model_doc/dinov3
# All ViT/16 LVD-1689M weights share patch_size=16 and 4 register tokens; the
# matcher reads patch_size / num_register_tokens from the loaded model.config,
# so this map only needs the repo id.
# ──────────────────────────────────────────────────────────────────────────────

DINOV3_VARIANTS: dict[str, str] = {
    # small distilled — fastest, lowest VRAM (~1 GB), dim=384
    "small":      "facebook/dinov3-vits16-pretrain-lvd1689m",
    "small_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    # base — default, good accuracy/speed trade-off (~2 GB), dim=768
    "base":       "facebook/dinov3-vitb16-pretrain-lvd1689m",
    # large — most accurate dense features (~5 GB), dim=1024
    "large":      "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

DEFAULT_DINOV3 = "base"
DEFAULT_SAM3 = "facebook/sam3"


def resolve_dinov3(variant_or_id: str) -> str:
    """Map a friendly variant name to a HF repo id; pass through full ids."""
    return DINOV3_VARIANTS.get(variant_or_id, variant_or_id)


# ──────────────────────────────────────────────────────────────────────────────
# Component configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MatcherConfig:
    """DINOv3 dense-correspondence settings."""

    dinov3: str = DEFAULT_DINOV3
    """Variant key (small/base/large) or a full HF repo id."""

    use_bg_prototype: bool = True
    """Build a background prototype from the mask complement and suppress it."""

    bg_mode: str = "subtract"
    """How to combine fg/bg similarity: 'subtract' (fg - bg) or 'softmax'."""

    use_4d_correlation: bool = False
    """Stronger (slower) mode: correlate every support-fg patch against every
    query patch instead of a single averaged prototype. Default False."""

    corr_topk: int = 1
    """For 4D mode: per query patch, average the top-k support-fg similarities."""

    normalize_prior: bool = True
    """Min-max normalise the prior map to [0, 1] (stabilises prompt thresholds)."""


@dataclass
class PromptConfig:
    """prior-map → SAM-prompt derivation settings."""

    threshold: float = 0.4
    """Foreground threshold on the (normalised) prior map in [0, 1]. Sweep-
    calibrated for the per-granule path (mIoU 0.64 over the 7 fertilizer queries)."""

    num_pos_points: int = 3
    """Max positive point prompts per instance (similarity peaks)."""

    num_neg_points: int = 2
    """Negative points sampled from clearly-low-similarity regions (0 disables)."""

    neg_threshold: float = 0.4
    """Prior below this is considered clear background for negative sampling."""

    peak_min_distance: int = 8
    """Minimum pixel spacing between detected positive peaks."""

    min_blob_area: int = 64
    """Ignore thresholded blobs smaller than this many pixels."""

    multi_instance: bool = False
    """Emit one prompt set per separated foreground blob (else one global set)."""

    emit_box: bool = True
    """Include a tight bounding box around each blob as a prompt."""

    emit_mask_prompt: bool = False
    """Pass the thresholded prior as a low-res mask prompt to SAM 3. Off by
    default — points+box are more reliable; see segmenter.py for the caveat."""

    seed: int = 0
    """RNG seed for negative-point sampling (reproducibility)."""

    # ── Small-scattered-instance path (task 4; all OFF by default) ──────────
    # Tuned for diffuse fields of many small targets (e.g. fertilizer granules)
    # where boxes over-include background between granules. Enabling
    # ``component_prompts`` drives one prompt set per 8-connected component of the
    # thresholded prior and is best paired with ``multi_instance=True``.

    component_prompts: bool = False
    """Drive instances from 8-connected components of the thresholded prior
    (one prompt set per component) instead of the default 4-connected blobs."""

    points_per_component: int = 3
    """Positive points (1-3 typical) emitted at prior peaks inside each
    component when ``component_prompts`` is on."""

    use_box_prompts: bool = True
    """Emit a bounding box per component on the component path. Set False to
    rely on points only — boxes over-include background between granules."""

    box_erode_frac: float = 0.0
    """Shrink each component box toward its core by this fraction of its
    half-extent (0 = tight box, 0.25 = 25%% inset on each side)."""

    use_mask_prompt: bool = False
    """Feed a *heavily eroded high-confidence core* of the prior (a few patches)
    as SAM 3's low-res ``input_masks`` — NEVER the full thresholded blob, which
    makes SAM reproduce the prior's shape. Off by default; sparse points are
    safer. See ``mask_prompt_erosion``."""

    mask_prompt_erosion: int = 3
    """Binary-erosion iterations applied before a mask prompt is emitted, so only
    a tiny high-confidence core survives (prevents prior-shape leakage)."""

    # ── Per-granule prompting (the regression fix) ─────────────────────────
    # For diffuse fields of small bright granules on darker soil. The prior only
    # *localises* (which regions contain granules); SAM cuts tight boundaries per
    # seed. The prior's blobby shape must never become the output shape.

    per_granule_prompts: bool = False
    """Detect individual granule seeds inside each prior-positive region and emit
    ONE prompt per seed (single positive point + optional tiny box), unioning the
    per-granule SAM masks. Supersedes the whole-blob prompt for scattered targets."""

    seed_source: str = "brightness"
    """Where granule seeds come from: 'brightness' (granules are lighter than
    soil), 'prior', or 'both' (brightness * prior, restricted to prior fg).
    Sweep-calibrated: 'brightness' is the single biggest lever (mixing the prior
    diluted seeds); flipping it back to 'both' drops mIoU 0.64 → 0.49."""

    granule_min_distance: int = 5
    """Minimum pixel spacing between detected granule seeds (NMS radius). Tune to
    roughly the granule radius so adjacent granules get separate seeds. Sweep-
    calibrated to 5 (marginal over 6)."""

    granule_seed_z: float = 1.5
    """A seed is kept only if its score exceeds mean + granule_seed_z·std of the
    in-prior score, rejecting flat soil plateaus between granules. Lower = more
    seeds (higher recall, more soil risk); higher = fewer, brighter seeds.
    Sweep-calibrated to 1.5 (fewer, cleaner seeds)."""

    max_seeds_per_region: int = 256
    """Safety cap on seeds per prior region (avoids thousands of SAM calls)."""

    granule_box: bool = True
    """Emit a small tight box (sized from the granule size prior) around each seed
    in addition to the positive point. Constrains SAM to a single granule."""

    granule_box_scale: float = 1.0
    """Box half-side = granule_box_scale * sqrt(expected_granule_area)/2. Sweep-
    calibrated to 1.0 (a tight box hugging the granule includes less inter-granule
    soil than the looser 1.5)."""

    granule_neg_points: int = 2
    """Negative points sampled from the darker inter-granule gaps near each seed
    (tells SAM the soil between granules is background). Sweep-calibrated to 2
    (marginal over 4)."""

    # ── Per-region dense-vs-scattered router (primary deliverable) ──────────
    # A single image can hold BOTH a dense/clumped pile (boxes win) and scattered
    # granules separated by soil (per-granule wins). Routing per WHOLE image is
    # therefore suboptimal. The router decides PER connected component of the
    # thresholded prior whether that region is dense or scattered and dispatches
    # it to the box path or the per-granule path; results are unioned downstream.
    # Still strictly downstream of the prior — the prior localises and now ROUTES,
    # but never dictates mask boundaries.

    router: bool = False
    """Route each prior component independently: dense → box prompts, scattered →
    per-granule prompts. Supersedes per_granule_prompts / component_prompts when on."""

    router_density_metric: str = "seed_coverage"
    """Which clumping metric decides dense-vs-scattered per component:
      'seed_coverage' — estimated granule area (n_seeds × expected_area) / component
                        area; HIGH ⇒ packed granules with little soil ⇒ dense.
      'nn_distance'   — median inter-seed nearest-neighbour distance in granule-
                        diameter units; SMALL ⇒ granules touching ⇒ dense.
      'both'          — dense only if seed_coverage says dense AND nn_distance does."""

    router_seed_coverage_threshold: float = 0.7
    """A component is DENSE when its seed-coverage fraction ≥ this (else scattered).
    Calibrated on the fertilizer eval (q60 dense / q105 scattered): 0.7 dominated
    0.4 and 0.55 on both — fully recovers the dense pile while staying closest to
    the per-granule path on scattered. Surfaced per-component in the debug panel."""

    router_nn_distance_threshold: float = 2.0
    """A component is DENSE when its median normalised inter-seed NN distance ≤ this
    (units of granule diameter = sqrt(expected_area)). Used by 'nn_distance'/'both'."""

    router_min_seeds_dense: int = 2
    """Components with fewer than this many detected seeds default to the box path
    (a 1-seed component is a single granule; a tight box ≈ the granule). Calibrated:
    the original default of 4 forced small scattered clusters onto the box path and
    cost ~0.04 IoU on the scattered subset; 2 lets the router ACCEPT on both regimes
    (q60 dense ≈0.82, q105 scattered ≈0.45 > per-granule 0.44)."""


@dataclass
class SegmenterConfig:
    """SAM 3 refinement settings."""

    sam3: str = DEFAULT_SAM3
    """HF repo id for SAM 3 (used for both tracker and concept paths)."""

    multimask_output: bool = True
    """Ask SAM 3 for 3 candidate masks and keep the highest-IoU one."""

    concept_threshold: float = 0.7
    """Score threshold for the native concept (text/exemplar) path."""

    mask_threshold: float = 0.7
    """Mask logit threshold for the concept path post-processing."""

    # ── Prior-aware candidate selection + prior gating (tasks 2 & 3) ────────
    # All default to the original SAM-score behaviour so current results are
    # reproducible; turn these on to make SAM defer to the (good) prior.

    selection_criterion: str = "sam_score"
    """How to pick among multimask_output candidates:
      'sam_score' — SAM's own predicted IoU (last-good default).
      'granule'   — keep only candidates whose centroid lies in the prior, then
                    prefer the COMPACT one (within the size prior) by SAM score.
    NOTE: the old 'prior_overlap' rule is removed — it biased toward the most
    over-inclusive candidate because the prior itself is blobby."""

    prior_gating: bool = False
    """Reject SAM instances that barely overlap the (UN-dilated) prior — a pure
    accept/reject test. Gating may only REMOVE instances, never grow/reshape a
    mask, so it can no longer leak inter-granule background."""

    min_prior_overlap: float = 0.0
    """Reject a SAM instance whose fraction of pixels inside the un-dilated prior
    region is below this (0 = never reject on overlap)."""

    prior_dilation: int = 0
    """DEPRECATED / unused. Dilation in gating reintroduced inter-granule
    background; gating no longer dilates. Kept only so old CLI flags don't break."""

    # ── Granule size prior (task 4) ────────────────────────────────────────

    size_gate: bool = False
    """Reject any SAM mask whose area >> the expected granule area (estimated
    from the support masks). A mask covering many granules is invalid by
    construction — this directly kills the blob."""

    granule_size_mult: float = 4.0
    """Reject a mask whose area exceeds granule_size_mult × expected granule
    area (median support connected-component area)."""

    intersect_prior_safety: bool = False
    """Final-assembly safety: intersect the union of accepted masks with the
    UN-dilated thresholded prior (never dilated). Off by default."""


@dataclass
class PriorMaskConfig:
    """Prior-only diagnostic mask path + shared mask cleanup (task 1).

    Thresholds the (confirmed-good) prior directly into a mask with light
    morphology and an optional edge-snap, with NO SAM 3 in the loop. Used as the
    'prior_only' baseline variant and is entirely downstream of the prior.
    """

    prior_only_threshold: float = 0.5
    """Foreground threshold on the normalised prior for the prior-only mask."""

    morph_open_radius: int = 1
    """Binary-opening disk radius to drop speckle (0 disables)."""

    morph_close_radius: int = 1
    """Binary-closing disk radius to fill pinholes/bridge granules (0 disables)."""

    use_edge_snap: bool = False
    """Run one guided-filter pass against the grayscale image to snap the mask
    boundary to image edges (off by default)."""

    edge_snap_radius: int = 4
    """Guided-filter window radius (px) for edge snapping."""

    edge_snap_eps: float = 1e-2
    """Guided-filter regularisation; larger = smoother / less edge-aware."""


@dataclass
class FSSConfig:
    """Top-level aggregate config."""

    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    prior_mask: PriorMaskConfig = field(default_factory=PriorMaskConfig)

    eval_variants: bool = False
    """When True, segment() also returns 'variant_masks' with the prior_only /
    current-SAM3 / prior-gated-SAM3 masks for side-by-side IoU comparison."""

    alignment_check: bool = False
    """Inference-time prototype-alignment (PANet-PAR-style) CONSISTENCY check — NOT
    a training loss (all models stay frozen). After predicting the query mask,
    build a prototype FROM the predicted query mask and re-segment each support
    image, scoring reverse-IoU against the (known) support mask. Returns a
    label-free confidence signal in result['alignment']; never alters the mask."""

    device: Optional[str] = None
    """'cuda' / 'cpu' / None (auto-detect)."""

    dtype: Optional[str] = None
    """Autocast precision used on CUDA forward passes (weights stay fp32):
    'auto' (bfloat16 on CUDA), 'float16', 'bfloat16', or 'float32'/None (no
    autocast). Ignored on CPU. bfloat16 is recommended on H100."""
