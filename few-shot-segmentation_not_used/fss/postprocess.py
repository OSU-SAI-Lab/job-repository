"""
fss/postprocess.py — prior-only mask path + shared mask-cleanup utilities.

Everything here is strictly *downstream of the (confirmed-good) DINOv3 prior* —
it never touches the matcher or the prior computation. It provides:

  * ``prior_only_mask`` — threshold the prior, light morphological open/close,
    and an optional guided-filter edge-snap against the grayscale image. No SAM 3.
    This is the diagnostic baseline (task 1, variant "a").
  * ``disk`` / morphology helpers reused by the prompt + gating code.

The point of the prior-only path is to answer "is SAM 3 net-positive for this
class at all?" — for a diffuse field of small scattered granules the prior alone
may already beat SAM 3's object-snapping behaviour.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from .config import PriorMaskConfig


def disk(radius: int) -> np.ndarray:
    """A boolean disk structuring element of the given radius."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    """Edge-preserving guided filter (He et al. 2010).

    ``guide`` (the grayscale image) steers the smoothing of ``src`` (the soft
    mask); both are float arrays, ``guide`` expected in [0, 1].
    """
    size = 2 * radius + 1
    mean_I = ndimage.uniform_filter(guide, size)
    mean_p = ndimage.uniform_filter(src, size)
    mean_Ip = ndimage.uniform_filter(guide * src, size)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = ndimage.uniform_filter(guide * guide, size)
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = ndimage.uniform_filter(a, size)
    mean_b = ndimage.uniform_filter(b, size)
    return mean_a * guide + mean_b


def prior_only_mask(
    prior: np.ndarray,
    cfg: PriorMaskConfig,
    gray: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Threshold the prior into a mask with cleanup; no SAM 3 involved.

    Args:
        prior: (H, W) float prior map (normalised to [0, 1] upstream).
        cfg:   PriorMaskConfig with threshold / morphology / edge-snap knobs.
        gray:  optional (H, W) grayscale image for the edge-snap pass.
    """
    mask = prior >= cfg.prior_only_threshold

    if cfg.morph_open_radius > 0:
        mask = ndimage.binary_opening(mask, structure=disk(cfg.morph_open_radius))
    if cfg.morph_close_radius > 0:
        mask = ndimage.binary_closing(mask, structure=disk(cfg.morph_close_radius))

    if cfg.use_edge_snap and gray is not None and mask.any():
        g = gray.astype("float32")
        rng = float(g.max() - g.min())
        g = (g - g.min()) / (rng + 1e-6)
        refined = _guided_filter(
            g, mask.astype("float32"), cfg.edge_snap_radius, cfg.edge_snap_eps
        )
        mask = refined >= 0.5

    return mask.astype(bool)
