"""
Smoke tests for fss.

Two tiers:
  * Lightweight tests (PromptGenerator, config) run with no model weights.
  * The full end-to-end test loads DINOv3 + SAM 3 and runs one forward pass on a
    tiny synthetic image; it is SKIPPED automatically if the weights cannot be
    downloaded/loaded (offline CI, missing transformers, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_image_and_mask(size=96, sq=(30, 30, 66, 66)):
    """A grey image with a bright square, plus the square's binary mask."""
    from PIL import Image

    img = np.full((size, size, 3), 110, dtype="uint8")
    x1, y1, x2, y2 = sq
    img[y1:y2, x1:x2] = (230, 60, 60)
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Image.fromarray(img), mask


# ── Lightweight: no weights required ────────────────────────────────────────

def test_prompt_generator_from_prior():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    prior = np.zeros((96, 96), dtype="float32")
    prior[30:66, 30:66] = 1.0  # one foreground blob
    gen = PromptGenerator(PromptConfig(num_pos_points=3, num_neg_points=2))
    sets = gen.generate(prior)

    assert len(sets) == 1
    ps = sets[0]
    assert len(ps.positive_points) >= 1
    # positive peaks land inside the blob
    for x, y in ps.positive_points:
        assert 30 <= x < 66 and 30 <= y < 66
    assert ps.box is not None
    x1, y1, x2, y2 = ps.box
    assert x1 >= 30 and x2 <= 66


def test_prompt_generator_multi_instance():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    prior = np.zeros((128, 128), dtype="float32")
    prior[10:30, 10:30] = 1.0
    prior[90:110, 90:110] = 1.0
    gen = PromptGenerator(PromptConfig(multi_instance=True, min_blob_area=16))
    sets = gen.generate(prior)
    assert len(sets) == 2
    # region is always populated for prior-aware selection/gating downstream.
    assert all(ps.region is not None for ps in sets)


# ── Prior-deferring path: prior-only mask, component prompts, selection, gating ─

def test_prior_only_mask_cleanup():
    from fss.config import PriorMaskConfig
    from fss.postprocess import prior_only_mask

    prior = np.zeros((64, 64), dtype="float32")
    prior[20:40, 20:40] = 1.0
    prior[0, 0] = 1.0  # isolated speckle dropped by morphological opening
    mask = prior_only_mask(prior, PriorMaskConfig(prior_only_threshold=0.5,
                                                  morph_open_radius=1))
    assert mask.dtype == bool and mask.shape == (64, 64)
    assert mask[30, 30] and not mask[0, 0]


def test_component_prompts_no_boxes():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    prior = np.zeros((120, 120), dtype="float32")
    for cy, cx in [(20, 20), (20, 90), (90, 20), (90, 90)]:
        prior[cy - 3:cy + 3, cx - 3:cx + 3] = 1.0
    gen = PromptGenerator(PromptConfig(component_prompts=True, multi_instance=True,
                                       min_blob_area=4, use_box_prompts=False,
                                       use_mask_prompt=True))
    sets = gen.generate(prior)
    assert len(sets) == 4
    for ps in sets:
        assert ps.box is None                       # boxes disabled
        assert ps.mask is not None and ps.mask.dtype == np.float32  # continuous prior
        assert len(ps.positive_points) >= 1


def test_granule_selection_compact_and_centroid():
    from fss.segmenter import select_candidate

    region = np.zeros((20, 20), dtype=bool)
    region[8:12, 8:12] = True                 # small prior region near the centre
    tight = np.zeros((20, 20), dtype=bool); tight[8:12, 8:12] = True    # area 16
    blob = np.zeros((20, 20), dtype=bool); blob[2:18, 2:18] = True       # area 256
    off = np.zeros((20, 20), dtype=bool); off[15:19, 15:19] = True       # centroid off prior
    cands = [tight, blob, off]
    scores = [0.4, 0.9, 0.95]   # SAM prefers the off-region / over-inclusive "object"

    # sam_score chases the highest-confidence candidate (here the off-region one).
    assert select_candidate(cands, scores, region, "sam_score") == 2
    # granule keeps centroid-in-prior + compact (within size prior) → the tight one.
    assert select_candidate(cands, scores, region, "granule",
                            expected_area=16, size_mult=4.0) == 0


def test_size_gate_and_overlap_gate():
    from fss.segmenter import gate_mask

    region = np.zeros((30, 30), dtype=bool); region[10:15, 10:15] = True
    tight = np.zeros((30, 30), dtype=bool); tight[10:15, 10:15] = True   # area 25
    blob = np.zeros((30, 30), dtype=bool); blob[0:30, 0:30] = True        # area 900
    off = np.zeros((30, 30), dtype=bool); off[25:29, 25:29] = True        # off prior

    # Size gate kills the blob (>> expected granule area); keeps the tight mask UNCHANGED.
    assert gate_mask(blob, region, 0.0, expected_area=25, size_mult=4.0) is None
    kept = gate_mask(tight, region, 0.0, expected_area=25, size_mult=4.0)
    assert kept is not None and int(kept.sum()) == 25       # never reshaped/grown

    # Overlap gate rejects a mask sitting off the prior.
    assert gate_mask(off, region, 0.5) is None


def test_per_granule_prompts_many_seeds():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    # One big prior blob containing several bright "granules" on a dark field.
    prior = np.zeros((80, 80), dtype="float32")
    prior[20:60, 20:60] = 1.0
    gray = np.full((80, 80), 40, dtype="float32")          # dark soil
    for cy, cx in [(28, 28), (28, 52), (52, 28), (52, 52), (40, 40)]:
        gray[cy - 2:cy + 3, cx - 2:cx + 3] = 230            # bright granules

    gen = PromptGenerator(PromptConfig(per_granule_prompts=True,
                                       granule_min_distance=6, granule_box=True))
    sets = gen.generate(prior, gray=gray, expected_area=25)
    assert len(sets) >= 5                                    # one prompt per granule
    for ps in sets:
        assert len(ps.positive_points) == 1                 # single seed each
        assert ps.box is not None                           # tiny granule box
        x1, y1, x2, y2 = ps.box
        assert (x2 - x1) <= 14 and (y2 - y1) <= 14          # small, not the blob
        assert ps.region is not None and ps.region.sum() < prior.sum()  # not whole blob


# ── Per-region dense-vs-scattered router (primary deliverable) ───────────────

def test_router_density_metric_and_decision():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    gen = PromptGenerator(PromptConfig(router_seed_coverage_threshold=0.5,
                                       router_min_seeds_dense=4))

    # Dense: 10 seeds packed into a 100-px component (coverage 0.9 ≫ 0.5).
    dense_comp = np.ones((10, 10), dtype=bool)
    dense_seeds = [(i, i) for i in range(10)]
    d = gen._component_density(dense_comp, dense_seeds, expected_area=9)
    assert d["n_seeds"] == 10
    assert d["seed_coverage"] > 0.5
    assert gen._is_dense(d) is True

    # Scattered: 4 spread seeds in a 1000-px component (coverage 0.036 ≪ 0.5).
    scat_comp = np.zeros((50, 50), dtype=bool)
    scat_comp[:40, :25] = True                       # area 1000
    scat_seeds = [(2, 2), (2, 45), (38, 2), (38, 45)]
    s = gen._component_density(scat_comp, scat_seeds, expected_area=9)
    assert s["seed_coverage"] < 0.5
    assert gen._is_dense(s) is False

    # Too-few-seeds guard: a 2-seed component defaults to the box (dense) path.
    tiny = gen._component_density(scat_comp, [(2, 2), (38, 45)], expected_area=9)
    assert tiny["n_seeds"] == 2
    assert gen._is_dense(tiny) is True


def test_router_area_gate():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    gen = PromptGenerator(PromptConfig(router_seed_coverage_threshold=0.3,
                                       router_min_seeds_dense=2,
                                       router_min_dense_area_frac=0.05))
    # Small but locally-packed cluster → scattered (fails the area gate) even though
    # its seed-coverage is high. This is the misroute the area gate fixes.
    small = dict(seed_coverage=0.9, nn_distance_norm=1.0, n_seeds=5,
                 comp_area=100.0, comp_area_frac=0.01)
    assert gen._is_dense(small) is False
    # Large packed pile → dense.
    big = dict(seed_coverage=0.9, nn_distance_norm=1.0, n_seeds=5,
               comp_area=10000.0, comp_area_frac=0.20)
    assert gen._is_dense(big) is True


def test_router_routes_mixed_image():
    from fss.config import PromptConfig
    from fss.prompts import PromptGenerator

    # Two disjoint prior components: a solid packed pile (dense) and a field of
    # sparse bright granules on dark soil (scattered).
    prior = np.zeros((120, 200), dtype="float32")
    prior[20:50, 20:50] = 1.0                        # dense pile component
    prior[20:90, 110:180] = 1.0                      # scattered field component
    gray = np.full((120, 200), 40, dtype="float32")  # dark soil
    gray[20:50, 20:50] = 230                         # packed bright pile
    for cy in range(28, 86, 18):                     # sparse granules in the field
        for cx in range(118, 176, 18):
            gray[cy - 2:cy + 3, cx - 2:cx + 3] = 230

    gen = PromptGenerator(PromptConfig(router=True, granule_min_distance=6,
                                       min_blob_area=64,
                                       router_seed_coverage_threshold=0.55,
                                       router_min_seeds_dense=4,
                                       router_min_dense_area_frac=0.03))
    sets = gen.generate(prior, gray=gray, expected_area=25)

    routes = {ps.route for ps in sets}
    assert routes == {"dense", "scattered"}          # both regimes present
    assert all(ps.density is not None for ps in sets)

    # Dense pile → a single box prompt set with a box (the original box pipeline).
    dense_sets = [ps for ps in sets if ps.route == "dense"]
    assert len(dense_sets) == 1
    assert dense_sets[0].box is not None
    dx1, dy1, dx2, dy2 = dense_sets[0].box
    assert (dx2 - dx1) >= 20 and (dy2 - dy1) >= 20    # spans the pile, not one granule

    # Scattered field → several per-granule sets, each a single seed + tiny box.
    scat_sets = [ps for ps in sets if ps.route == "scattered"]
    assert len(scat_sets) >= 4
    for ps in scat_sets:
        assert len(ps.positive_points) == 1
        if ps.box is not None:
            bx1, by1, bx2, by2 = ps.box
            assert (bx2 - bx1) <= 16 and (by2 - by1) <= 16  # tiny, not the component


# ── Alignment check (reverse-IoU) ────────────────────────────────────────────

def test_binary_iou_matches_expected():
    # binary_iou backs the reverse alignment score; verify fg IoU semantics.
    from fss.eval import binary_iou

    a = np.zeros((10, 10), dtype=bool); a[2:6, 2:6] = True      # 16 px
    b = np.zeros((10, 10), dtype=bool); b[4:8, 2:6] = True      # 16 px, half-overlap
    fg, bg = binary_iou(a, b)
    # intersection 8, union 24 → 1/3.
    assert abs(fg - (8 / 24)) < 1e-6
    assert 0.0 <= bg <= 1.0
    # Identical masks → fg IoU 1.0; disjoint → 0.0.
    assert binary_iou(a, a)[0] == 1.0
    c = np.zeros((10, 10), dtype=bool); c[0, 0] = True
    assert binary_iou(a, c)[0] == 0.0


# ── Full end-to-end: needs model weights (skippable) ─────────────────────────

@pytest.mark.slow
def test_end_to_end_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from fss import FewShotSegmenter

    try:
        seg = FewShotSegmenter(dinov3="small", device="cpu")
    except Exception as e:  # weights unavailable / offline
        pytest.skip(f"Model weights unavailable: {e}")

    img, mask = _synthetic_image_and_mask()
    seg.set_support(images=[img], masks=[mask])
    result = seg.segment(img)

    assert result["mask"].shape == (96, 96)
    assert result["mask"].dtype == bool
    assert result["prior"].shape == (96, 96)
    assert isinstance(result["score"], float)
