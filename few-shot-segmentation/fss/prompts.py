"""
fss/prompts.py — turn a DINOv3 prior map into SAM 3 prompts.

From the upsampled prior map we derive, per target instance:
  * positive points — local maxima / top-k peaks of the similarity map,
  * a tight bounding box around the thresholded foreground blob,
  * negative points — sampled from clearly-low-similarity regions,
  * optionally a coarse low-res mask prompt (the thresholded prior).

Multi-instance: if several separated high-similarity blobs exist, one prompt set
is emitted per blob so SAM 3 can segment each instance independently.

Coordinate convention: points and boxes are in (x, y) pixel coordinates of the
full-resolution query image (x = column, y = row), matching SAM 3's processor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .config import PromptConfig


@dataclass
class PromptSet:
    """One SAM 3 prompt bundle targeting a single instance."""

    positive_points: List[Tuple[int, int]] = field(default_factory=list)  # (x, y)
    negative_points: List[Tuple[int, int]] = field(default_factory=list)  # (x, y)
    box: Optional[Tuple[int, int, int, int]] = None                       # x1,y1,x2,y2
    mask: Optional[np.ndarray] = None                  # SAM mask-prompt input (H, W)
    region: Optional[np.ndarray] = None                # bool thresholded-prior region
    """The thresholded-prior region this set targets — used downstream for
    prior-aware candidate selection and prior gating (segmenter.py). Always set;
    independent of whether a mask prompt is emitted."""

    route: Optional[str] = None                        # 'dense' | 'scattered' | None
    """Set by the per-region router: which regime this prompt set's source
    component was routed to. None when the router is off."""

    density: Optional[dict] = None                     # router metrics for this component
    """The router's per-component density metrics (seed_coverage, nn_distance_norm,
    n_seeds, comp_area) — surfaced in the debug panel. None when the router is off."""

    @property
    def points(self) -> List[Tuple[int, int]]:
        return self.positive_points + self.negative_points

    @property
    def labels(self) -> List[int]:
        return [1] * len(self.positive_points) + [0] * len(self.negative_points)


class PromptGenerator:
    """Derive SAM-style prompts from a prior map. Thresholds are configurable."""

    def __init__(self, config: PromptConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate(
        self,
        prior: np.ndarray,
        gray: Optional[np.ndarray] = None,
        expected_area: Optional[float] = None,
    ) -> List[PromptSet]:
        """Return one PromptSet per target instance (>=1).

        ``gray`` is the full-res grayscale query (needed for the per-granule path,
        where granules are detected as brightness maxima). ``expected_area`` is the
        granule size prior (median support component area) used to size tiny boxes.
        """
        c = self.config
        fg_mask = prior >= c.threshold

        if not fg_mask.any():
            # Nothing crossed the threshold — fall back to the global argmax peak so
            # SAM 3 still receives a prompt instead of failing silently.
            y, x = np.unravel_index(int(np.argmax(prior)), prior.shape)
            return [PromptSet(positive_points=[(int(x), int(y))])]

        # Shared pool of negative candidate coordinates (clear background).
        neg_coords = np.argwhere(prior < c.neg_threshold)        # (K, 2) as (y, x)

        if c.router:
            return self._router_prompts(prior, fg_mask, gray, expected_area, neg_coords)

        if c.per_granule_prompts:
            return self._per_granule_prompts(prior, fg_mask, gray, expected_area)

        if c.component_prompts:
            return self._component_prompts(prior, fg_mask, neg_coords)

        labels, n = ndimage.label(fg_mask)
        blobs = [labels == i for i in range(1, n + 1)]
        blobs = [b for b in blobs if b.sum() >= c.min_blob_area] or [fg_mask]

        if not c.multi_instance:
            # Collapse to a single set covering all foreground.
            blobs = [np.logical_or.reduce(blobs)]

        prompt_sets: List[PromptSet] = []
        for blob in blobs:
            prompt_sets.append(self._prompts_for_blob(prior, blob, neg_coords))
        return prompt_sets

    # ──────────────────────────────────────────────────────────────────────
    # Per-granule path: one prompt per detected granule seed (the regression fix).
    # Prior localises; SAM cuts tight boundaries. The prior shape never becomes
    # the output shape.
    # ──────────────────────────────────────────────────────────────────────

    def _per_granule_prompts(
        self,
        prior: np.ndarray,
        fg_mask: np.ndarray,
        gray: Optional[np.ndarray],
        expected_area: Optional[float],
        seeds: Optional[List[Tuple[int, int]]] = None,
    ) -> List[PromptSet]:
        c = self.config

        # Seed-detection score, restricted to the prior-positive region. ``seeds``
        # may be passed in precomputed (the router detects them once for both the
        # density metric and prompting) to avoid a duplicate detection pass.
        if seeds is None:
            score = self._seed_score(prior, fg_mask, gray)
            seeds = self._detect_seeds(score, fg_mask, c.granule_min_distance,
                                       c.max_seeds_per_region,
                                       c.granule_seed_z)                 # [(x, y), ...]
        if not seeds:
            # Fall back to the global prior peak so SAM still gets a prompt.
            y, x = np.unravel_index(int(np.argmax(np.where(fg_mask, prior, -np.inf))),
                                    prior.shape)
            seeds = [(int(x), int(y))]

        # Inter-granule gap pixels: prior fg that is *darker* than the local seeds
        # (the soil between granules) — strong negatives.
        if gray is not None:
            gnorm = self._norm(gray)
            gap_mask = fg_mask & (gnorm < float(np.median(gnorm[fg_mask])))
        else:
            gap_mask = np.zeros_like(fg_mask)
        gap_coords = np.argwhere(gap_mask)                                # (K, 2) (y, x)

        # Tiny box half-side from the granule size prior.
        half = None
        if c.granule_box and expected_area and expected_area > 0:
            half = max(2, int(round(c.granule_box_scale * (expected_area ** 0.5) / 2)))

        h, w = prior.shape
        seed_arr = np.array(seeds)                       # (N, 2) as (x, y)
        prompt_sets: List[PromptSet] = []
        for (x, y) in seeds:
            ps = PromptSet(positive_points=[(int(x), int(y))])

            if half is not None:
                ps.box = (max(0, x - half), max(0, y - half),
                          min(w, x + half + 1), min(h, y + half + 1))

            # Negatives: nearest other seeds (other granules) + nearby gap/soil
            # points, so SAM cuts THIS granule out of its neighbours.
            negs: List[Tuple[int, int]] = []
            if c.granule_neg_points > 0:
                negs += self._nearest_other_seeds(seed_arr, (x, y),
                                                  c.granule_neg_points // 2)
                negs += self._nearest_coords(gap_coords, (x, y),
                                             c.granule_neg_points - len(negs))
            ps.negative_points = negs

            # Per-seed region (small disk around the seed) for centroid-in-prior
            # selection & gating — NOT the whole blob, so the prior shape can't leak.
            ps.region = self._seed_region(fg_mask, (x, y), c.granule_min_distance)
            prompt_sets.append(ps)
        return prompt_sets

    # ──────────────────────────────────────────────────────────────────────
    # Per-region router: dense components → box prompts; scattered → per-granule.
    # The prior localises and now ROUTES; SAM still owns every boundary. Each
    # 8-connected prior component is classified independently so one image can mix
    # a dense pile (boxes win) with scattered granules (per-seed wins).
    # ──────────────────────────────────────────────────────────────────────

    def _router_prompts(
        self,
        prior: np.ndarray,
        fg_mask: np.ndarray,
        gray: Optional[np.ndarray],
        expected_area: Optional[float],
        neg_coords: np.ndarray,
    ) -> List[PromptSet]:
        c = self.config
        structure = np.ones((3, 3), dtype=bool)             # 8-connectivity
        labels, n = ndimage.label(fg_mask, structure=structure)
        comps = [labels == i for i in range(1, n + 1)]
        comps = [b for b in comps if b.sum() >= c.min_blob_area] or [fg_mask]

        img_area = float(fg_mask.size)
        prompt_sets: List[PromptSet] = []
        for comp in comps:
            # Detect seeds once; reused for the density metric and (if scattered)
            # for per-granule prompting.
            score = self._seed_score(prior, comp, gray)
            seeds = self._detect_seeds(score, comp, c.granule_min_distance,
                                       c.max_seeds_per_region, c.granule_seed_z)
            density = self._component_density(comp, seeds, expected_area, img_area)
            route = "dense" if self._is_dense(density) else "scattered"

            if route == "dense":
                # Box path = the original box pipeline restricted to this component.
                sets = [self._prompts_for_blob(prior, comp, neg_coords)]
            else:
                # Per-granule path restricted to this component (reuse the seeds).
                sets = self._per_granule_prompts(prior, comp, gray, expected_area,
                                                 seeds=seeds)
            for ps in sets:
                ps.route = route
                ps.density = density
            prompt_sets.extend(sets)
        return prompt_sets

    def _component_density(
        self,
        comp: np.ndarray,
        seeds: List[Tuple[int, int]],
        expected_area: Optional[float],
        img_area: Optional[float] = None,
    ) -> dict:
        """Clumping metrics for one prior component.

        seed_coverage    — estimated granule area (n_seeds × expected_area) divided
                           by the component area; HIGH ⇒ granules pack the region
                           with little inter-granule soil ⇒ dense.
        nn_distance_norm — median inter-seed nearest-neighbour distance in units of
                           granule diameter (sqrt(expected_area)); SMALL ⇒ granules
                           are touching ⇒ dense. inf when <2 seeds.
        comp_area_frac   — component area / image area; a true pile is one BIG region
                           (the area gate in _is_dense uses this).
        """
        comp_area = float(comp.sum())
        n = len(seeds)
        comp_area_frac = (comp_area / img_area) if (img_area and img_area > 0) else 1.0

        # Solidity = fraction of the component's bounding box that is foreground.
        # A solid pile fills its bbox (≈1); granules spread across an area fill
        # little of it (≪1) — the decisive dense/scattered signal.
        ys, xs = np.where(comp)
        if len(xs):
            bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
            solidity = comp_area / bbox_area if bbox_area > 0 else 1.0
        else:
            solidity = 1.0

        if expected_area and expected_area > 0:
            gran_area = float(expected_area)
        else:
            gran_area = 1.0
        seed_coverage = (n * gran_area / comp_area) if comp_area > 0 else 0.0

        diam = (gran_area ** 0.5) if gran_area > 0 else 1.0
        if n >= 2:
            pts = np.asarray(seeds, dtype="float32")        # (N, 2) as (x, y)
            d2 = ((pts[:, None, 0] - pts[None, :, 0]) ** 2 +
                  (pts[:, None, 1] - pts[None, :, 1]) ** 2)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(d2.min(axis=1))                     # nearest neighbour per seed
            nn_distance_norm = float(np.median(nn) / (diam + 1e-6))
        else:
            nn_distance_norm = float("inf")

        return {
            "seed_coverage": float(seed_coverage),
            "nn_distance_norm": nn_distance_norm,
            "n_seeds": int(n),
            "comp_area": comp_area,
            "comp_area_frac": float(comp_area_frac),
            "solidity": float(solidity),
        }

    def _is_dense(self, density: dict) -> bool:
        """Apply the configured density metric + thresholds to a component."""
        c = self.config
        # Area gate: a component too small to be a 'pile' is always scattered.
        if density.get("comp_area_frac", 1.0) < c.router_min_dense_area_frac:
            return False
        # Solidity gate: a component that doesn't fill its bounding box is granules
        # spread over an area (a box would swallow soil) → scattered. This is the
        # decisive gate; it catches LARGE scattered fields the area gate lets through.
        if density.get("solidity", 1.0) < c.router_min_dense_solidity:
            return False
        if density["n_seeds"] < c.router_min_seeds_dense:
            return True            # big, solid, few seeds → a packed pile (box-friendly)
        cov_dense = density["seed_coverage"] >= c.router_seed_coverage_threshold
        nn_dense = density["nn_distance_norm"] <= c.router_nn_distance_threshold
        metric = c.router_density_metric
        if metric == "seed_coverage":
            return cov_dense
        if metric == "nn_distance":
            return nn_dense
        if metric == "both":
            return cov_dense and nn_dense
        return cov_dense

    def _seed_score(
        self, prior: np.ndarray, fg_mask: np.ndarray, gray: Optional[np.ndarray]
    ) -> np.ndarray:
        """Build the granule-seed detection map inside the prior region."""
        c = self.config
        if gray is not None and c.seed_source in ("brightness", "both"):
            b = self._norm(gray)
            score = b * self._norm(prior) if c.seed_source == "both" else b
        else:
            score = self._norm(prior)            # 'prior' source or no image
        return np.where(fg_mask, score, -np.inf)

    @staticmethod
    def _detect_seeds(
        score: np.ndarray, fg_mask: np.ndarray, min_distance: int, max_seeds: int,
        seed_z: float = 1.0,
    ) -> List[Tuple[int, int]]:
        """Local maxima of ``score`` within fg_mask, NMS'd by min_distance.

        Only peaks whose score exceeds ``mean + seed_z*std`` of the in-region score
        are kept — this rejects the flat darker soil between granules (whose pixels
        would otherwise all register as plateau maxima), so seeds land on granules.
        """
        size = 2 * min_distance + 1
        finite = np.where(np.isfinite(score), score, -np.inf)
        local_max = ndimage.maximum_filter(finite, size=size, mode="nearest")
        peaks = fg_mask & (finite >= local_max) & np.isfinite(finite)

        in_region = score[fg_mask & np.isfinite(score)]
        if in_region.size:
            thr = float(in_region.mean() + seed_z * in_region.std())
            peaks &= finite >= thr
        ys, xs = np.where(peaks)
        if len(xs) == 0:
            # Threshold removed everything (uniform region) — fall back to the
            # single strongest in-region peak.
            yx = np.where(fg_mask & np.isfinite(score))
            if len(yx[0]) == 0:
                return []
            i = int(np.argmax(score[yx]))
            return [(int(yx[1][i]), int(yx[0][i]))]
        order = np.argsort(finite[ys, xs])[::-1]         # strongest first
        ys, xs = ys[order], xs[order]
        # Greedy NMS to avoid clustered duplicates from plateaus.
        chosen: List[Tuple[int, int]] = []
        taken = np.zeros(0)
        cy: List[int] = []
        cx: List[int] = []
        d2 = min_distance * min_distance
        for y, x in zip(ys, xs):
            if all((x - px) ** 2 + (y - py) ** 2 >= d2 for px, py in zip(cx, cy)):
                chosen.append((int(x), int(y)))
                cx.append(x); cy.append(y)
                if len(chosen) >= max_seeds:
                    break
        return chosen

    @staticmethod
    def _norm(a: np.ndarray) -> np.ndarray:
        a = a.astype("float32")
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-6)

    def _nearest_other_seeds(
        self, seed_arr: np.ndarray, seed: Tuple[int, int], k: int
    ) -> List[Tuple[int, int]]:
        if k <= 0 or len(seed_arr) <= 1:
            return []
        d2 = ((seed_arr[:, 0] - seed[0]) ** 2 + (seed_arr[:, 1] - seed[1]) ** 2)
        order = np.argsort(d2)
        out = []
        for i in order:
            if d2[i] == 0:
                continue                     # skip self
            out.append((int(seed_arr[i, 0]), int(seed_arr[i, 1])))
            if len(out) >= k:
                break
        return out

    @staticmethod
    def _nearest_coords(coords: np.ndarray, seed: Tuple[int, int], k: int) -> List[Tuple[int, int]]:
        if k <= 0 or len(coords) == 0:
            return []
        d2 = (coords[:, 1] - seed[0]) ** 2 + (coords[:, 0] - seed[1]) ** 2
        order = np.argsort(d2)[:k]
        return [(int(coords[i, 1]), int(coords[i, 0])) for i in order]

    @staticmethod
    def _seed_region(fg_mask: np.ndarray, seed: Tuple[int, int], radius: int) -> np.ndarray:
        """A small disk around the seed, clipped to the prior fg (for gating)."""
        h, w = fg_mask.shape
        x, y = seed
        r = max(2, radius)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        region = np.zeros_like(fg_mask)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        region[y0:y1, x0:x1] = ((xx - x) ** 2 + (yy - y) ** 2) <= r * r
        return region & fg_mask

    # ──────────────────────────────────────────────────────────────────────
    # Small-scattered-instance path: one prompt set per 8-connected component.
    # ──────────────────────────────────────────────────────────────────────

    def _component_prompts(
        self, prior: np.ndarray, fg_mask: np.ndarray, neg_coords: np.ndarray
    ) -> List[PromptSet]:
        c = self.config
        structure = np.ones((3, 3), dtype=bool)            # 8-connectivity
        labels, n = ndimage.label(fg_mask, structure=structure)
        comps = [labels == i for i in range(1, n + 1)]
        comps = [b for b in comps if b.sum() >= c.min_blob_area] or [fg_mask]

        if not c.multi_instance:
            # Honour the global-set request even on this path.
            comps = [np.logical_or.reduce(comps)]

        prompt_sets: List[PromptSet] = []
        for comp in comps:
            other = fg_mask & ~comp        # gaps/other components → negatives
            prompt_sets.append(
                self._prompts_for_component(prior, comp, other, neg_coords)
            )
        return prompt_sets

    def _prompts_for_component(
        self,
        prior: np.ndarray,
        comp: np.ndarray,
        other_fg: np.ndarray,
        neg_coords: np.ndarray,
    ) -> PromptSet:
        c = self.config
        ps = PromptSet(region=comp.copy())

        # 1-3 positive points at the prior peaks inside this component.
        masked_prior = np.where(comp, prior, -np.inf)
        ps.positive_points = self._peak_points(
            masked_prior, max(1, c.points_per_component), c.peak_min_distance
        )

        # Boxes over-include the background between granules — optional & erodable.
        if c.use_box_prompts:
            ps.box = self._component_box(comp, c.box_erode_frac)

        # Negatives: prefer the gaps *between* components (clearest "not this
        # instance" signal), then top up from clear background.
        negs: List[Tuple[int, int]] = []
        if c.num_neg_points > 0:
            gap_coords = np.argwhere(other_fg)
            negs += self._sample_coords(gap_coords, c.num_neg_points)
            remaining = c.num_neg_points - len(negs)
            if remaining > 0:
                negs += self._sample_coords(neg_coords, remaining)
        ps.negative_points = negs

        # Optional mask prompt: ONLY a heavily eroded high-confidence core, never
        # the full blob (which would make SAM reproduce the prior's shape).
        if c.use_mask_prompt:
            ps.mask = self._eroded_core(comp, prior, c.mask_prompt_erosion)

        return ps

    def _eroded_core(
        self, region: np.ndarray, prior: np.ndarray, iterations: int
    ) -> np.ndarray:
        """A tiny high-confidence core of ``region`` for use as a mask prompt.

        Erode the region hard, then keep only the highest-prior patches inside it.
        Falls back to the single argmax patch so the prompt is never empty.
        """
        core = ndimage.binary_erosion(region, iterations=max(1, iterations))
        if not core.any():
            core = region
        vals = prior[core]
        if vals.size:
            thr = np.quantile(vals, 0.75)        # top-quartile prior within the core
            core = core & (prior >= thr)
        if not core.any():
            y, x = np.unravel_index(int(np.argmax(np.where(region, prior, -np.inf))),
                                    prior.shape)
            core = np.zeros_like(region)
            core[y, x] = True
        return core.astype("float32")

    @staticmethod
    def _component_box(comp: np.ndarray, erode_frac: float):
        ys, xs = np.where(comp)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if erode_frac > 0:
            dx = (x2 - x1) * erode_frac / 2.0
            dy = (y2 - y1) * erode_frac / 2.0
            nx1, ny1 = int(round(x1 + dx)), int(round(y1 + dy))
            nx2, ny2 = int(round(x2 - dx)), int(round(y2 - dy))
            if nx2 > nx1 and ny2 > ny1:           # keep the box non-empty
                x1, y1, x2, y2 = nx1, ny1, nx2, ny2
        return (x1, y1, x2, y2)

    def _sample_coords(self, coords: np.ndarray, k: int) -> List[Tuple[int, int]]:
        """Sample up to k (x, y) points from a (K, 2) array of (y, x) coords."""
        if k <= 0 or len(coords) == 0:
            return []
        n = min(k, len(coords))
        sel = self._rng.choice(len(coords), size=n, replace=False)
        return [(int(x), int(y)) for y, x in coords[sel]]

    # ──────────────────────────────────────────────────────────────────────

    def _prompts_for_blob(
        self, prior: np.ndarray, blob: np.ndarray, neg_coords: np.ndarray
    ) -> PromptSet:
        c = self.config
        ps = PromptSet(region=blob.copy())

        # Positive points: peaks of the prior restricted to this blob.
        masked_prior = np.where(blob, prior, -np.inf)
        ps.positive_points = self._peak_points(masked_prior, c.num_pos_points,
                                               c.peak_min_distance)

        # Bounding box: tight box around the blob.
        if c.emit_box:
            ys, xs = np.where(blob)
            ps.box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

        # Negative points: random sample from the clear-background pool.
        if c.num_neg_points > 0:
            ps.negative_points = self._sample_coords(neg_coords, c.num_neg_points)

        # Mask prompt (optional): eroded high-confidence core if use_mask_prompt,
        # else the legacy binary blob via emit_mask_prompt.
        if c.use_mask_prompt:
            ps.mask = self._eroded_core(blob, prior, c.mask_prompt_erosion)
        elif c.emit_mask_prompt:
            ps.mask = blob.copy()

        return ps

    @staticmethod
    def _peak_points(
        score: np.ndarray, k: int, min_distance: int
    ) -> List[Tuple[int, int]]:
        """Greedy non-maximum-suppressed top-k peaks of ``score`` (returns (x, y))."""
        flat_order = np.argsort(score, axis=None)[::-1]      # high → low
        h, w = score.shape
        peaks: List[Tuple[int, int]] = []
        for idx in flat_order:
            if len(peaks) >= k:
                break
            y, x = divmod(int(idx), w)
            if not np.isfinite(score[y, x]):
                break
            if all((x - px) ** 2 + (y - py) ** 2 >= min_distance ** 2
                   for px, py in peaks):
                peaks.append((x, y))
        return peaks
