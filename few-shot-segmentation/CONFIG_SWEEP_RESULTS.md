# FSS per-granule config sweep — threshold & knob results

Record of the config sweeps run to maximise **mean IoU** of the `--per-granule-prompts`
path over the 7 held-out fertilizer COCO queries (support = image 0, 1024 px crops,
DINOv3-base + SAM 3, bf16). Kept so we can **backtrack to a more conservative
threshold if the tuned defaults turn out to be overfit.**

> ⚠️ **Overfitting caveat.** Every mIoU below is measured on the *same* 7 images the
> knobs were tuned on — there is no held-out split (only 8 labelled images exist,
> one is the support). Treat these as upper-bound, in-sample numbers. The winning
> knobs (bright seeds, tight boxes) are physically sensible for granules and should
> largely transfer, but the exact threshold may not. If new images score lower than
> expected, **backtrack the threshold** using the table below.

## Current defaults (written into `fss/config.py`)

| knob | value | line |
|---|---|---|
| `threshold` | **0.4** | config.py `PromptConfig.threshold` |
| `granule_seed_z` | **1.5** | `granule_seed_z` |
| `seed_source` | **brightness** | `seed_source` |
| `granule_box_scale` | **1.0** | `granule_box_scale` |
| `granule_min_distance` | **5** (r2) | `granule_min_distance` |
| `granule_neg_points` | **2** (r2) | `granule_neg_points` |
| `neg_threshold` | 0.4 (unchanged) | `neg_threshold` |

**Best mIoU = 0.6454** (round 2) · per-query: q15 0.373, q30 0.637, q45 0.709,
q60 0.744, q75 0.712, q90 0.737, q105 0.607.

## Threshold backtrack table (the key reference)

Threshold's effect depends on the other knobs, so two regimes are shown. **To back
off overfitting, raise the threshold toward 0.45–0.6** (more conservative prior, less
soil, lower in-sample mIoU but likely more robust).

| threshold | mIoU @ tuned knobs (brightness, box 1.0, z 1.5) | mIoU @ old baseline knobs (both, box 1.5, z 1.0) | source |
|---|---|---|---|
| 0.40 | **0.641** (current default) | 0.519 | sweep r1 |
| 0.45 | 0.606 | 0.526 | sweep r1 + full-pipeline run |
| 0.50 | — | 0.535 | sweep r1 |
| 0.60 | — | 0.513 | full-pipeline run (`thr06_eval`) |

Backtrack recommendation if overfit: **threshold 0.45** keeps most of the gain
(0.606 at the tuned knobs) and was the long-standing stable value. 0.5–0.6 are the
more conservative / higher-precision fallbacks.

## Round 1 sweep — full trajectory (23 configs, coordinate descent)

Start = old recommended config; each row varies one knob from the running best.

| # | config change | mIoU |
|---|---|---|
| 1 | baseline (thr 0.45, z 1.0, dist 6, both, box 1.5, neg 4) | 0.5264 |
| 2 | threshold=0.4 | 0.5186 |
| 3 | threshold=0.5 | 0.5345 |
| 4 | granule_seed_z=0.5 | 0.4900 |
| 5 | granule_seed_z=0.75 | 0.5227 |
| 6 | granule_seed_z=1.5 | 0.5216 |
| 7 | granule_min_distance=4 | 0.5456 |
| 8 | granule_min_distance=8 | 0.5149 |
| 9 | **seed_source=brightness** | **0.5837** ← biggest single jump |
| 10 | **granule_box_scale=1.0** | **0.6091** |
| 11 | granule_box_scale=2.0 | 0.5746 |
| 12 | granule_neg_points=0 | 0.6099 |
| 13 | threshold=0.4 (pass 1) | 0.6111 |
| 14 | threshold=0.45 (pass 1) | 0.6058 |
| 15 | granule_seed_z=0.5 (pass 1) | 0.5941 |
| 16 | granule_seed_z=0.75 (pass 1) | 0.5995 |
| 17 | **granule_seed_z=1.5 (pass 1)** | **0.6360** |
| 18 | granule_min_distance=6 (pass 1) | 0.6377 |
| 19 | granule_min_distance=8 (pass 1) | 0.6285 |
| 20 | seed_source=both (pass 1) | 0.4871 ← worst; confirms brightness |
| 21 | granule_box_scale=1.5 (pass 1) | 0.5927 |
| 22 | granule_box_scale=2.0 (pass 1) | 0.4967 |
| 23 | granule_neg_points=4 (pass 1) → **BEST** | **0.6410** |

Range explored: **0.487 → 0.641**. The `seed_source` knob alone accounts for most
of the spread (brightness 0.64 vs both 0.49).

## Round 2 — refinement (expanded ranges past round-1 edges)

Job 11907189, ranges: threshold 0.30–0.45, seed_z 1.5–2.5, box_scale 0.75–1.25,
min_dist 5–7, negs 2–6, seed_source re-check.

**Result: BEST mIoU = 0.6454** (+0.004 over round 1, converged after 1 pass). Only
two knobs nudged the optimum:

| config change | mIoU |
|---|---|
| round-1 best (thr 0.4, z 1.5, dist 6, brightness, box 1.0, neg 4) | 0.6410 |
| granule_min_distance 6 → **5** | 0.6447 |
| granule_neg_points 4 → **2** | **0.6454** ← final best |

Key takeaway: the **expanded threshold (0.30–0.45), seed_z (1.5–2.5) and box_scale
(0.75–1.25) ranges did NOT improve** — so threshold 0.4 / seed_z 1.5 / box_scale 1.0
are genuine optima, not grid-edge artifacts. The +0.004 from min_dist 5 / neg 2 is
noise-level; **round-1's config (dist 6, neg 4) is an equally good, marginally more
conservative fallback.** Search is effectively converged at ~0.64–0.645.

## How to revert / backtrack

Edit `fss/config.py` `PromptConfig`:
- **Safer threshold:** set `threshold = 0.45` (or 0.5 / 0.6 for more precision).
- **Full revert to pre-sweep:** `threshold=0.45, granule_seed_z=1.0,
  seed_source="both", granule_box_scale=1.5` (mIoU 0.526).

Re-validate after any change: `python -m pytest fss/tests/test_smoke.py -q`, then a
GPU eval via `sbatch scripts/fertilizer_router_eval_all.slurm` (or re-run the sweep
`scripts/sweep_fertilizer.slurm`).
