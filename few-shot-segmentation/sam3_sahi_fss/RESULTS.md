# sam3_sahi_fss — first eval (2026-07-20, job 12476837)

Apples-to-apples with `../sam3_dino_fss`: support = COCO image 0, queries
15/30/45/60/75/90/105, 1024 px crops, DINOv3-base + SAM 3, bf16, H100.
Config: `--proposal-text "visual" --slice-size 256 --overlap-ratio 0.2
--score-threshold 0.3 --match-mode prototype` (no `--subtract-negatives`).
Runtime **2 min 27 s** for support + 7 queries (25 SAM 3 forwards per image).

Support prototype set: **78 positive** / 58 background boxes from the support image's
SAHI proposals (`pos_coverage 0.6`).

## Cosine-threshold sweep (mean over the 7 GT queries)

| cos_thr | mIoU | precision | recall | keep% |
|---|---|---|---|---|
| ≤0.10 | 0.348 | 0.390 | 0.795 | 1.00 |
| 0.20 | 0.354 | 0.397 | 0.795 | 0.98 |
| 0.30 | 0.400 | 0.455 | 0.795 | 0.87 |
| 0.35 | 0.451 | 0.519 | 0.795 | 0.77 |
| 0.40 | 0.513 | 0.600 | 0.794 | 0.67 |
| 0.45 | 0.588 | 0.705 | 0.792 | 0.55 |
| 0.50 | 0.652 | 0.798 | 0.789 | 0.46 |
| **0.55** | **0.684** | **0.853** | **0.784** | **0.40** |
| 0.60 | 0.677 | 0.872 | 0.759 | 0.36 |
| 0.65 | 0.598 | 0.892 | 0.650 | 0.28 |
| 0.70 | 0.424 | 0.891 | 0.452 | 0.19 |

**BEST: cos_thr 0.55 → mIoU 0.6844, P 0.8532, R 0.7843.**

## Per query (at cos_thr 0.50)

| query | proposals (after NMS) | IoU |
|---|---|---|
| q15 | 121 | 0.688 |
| q30 | 145 | 0.595 |
| q45 | 207 | 0.720 |
| q60 | 338 | 0.606 |
| q75 | 293 | 0.604 |
| q90 | 175 | 0.722 |
| q105 | 291 | 0.627 |

## Comparison

| pipeline | mIoU | P | R | notes |
|---|---|---|---|---|
| `sam3_dino_fss` (SAM3 "seed", whole image) | **0.741** | 0.859 | 0.848 | 1 forward/image |
| `sam3_sahi_fss` (SAM3 "visual", 25 slices) | 0.684 | 0.853 | 0.784 | 25 forwards/image |
| `fss` per-granule (prior-driven, sweep-tuned) | 0.645 | 0.751 | — | |

SAHI lands **between** the two: clearly ahead of the prior-driven pipeline, ~0.06
behind whole-image `"seed"` proposals, and ~10× more expensive. The gap is entirely
**recall** (0.784 vs 0.848) — precision matches at 0.853.

## Reading of the result

- **The cosine match is finally doing real work.** In `sam3_dino_fss` the threshold was
  a near no-op (0.2–0.45 all gave 0.741, keep ≈ 100%) because `"seed"` proposals were
  already granule-clean. Here the sweep spans 0.348 → 0.684 and the best threshold keeps
  only **40%** of proposals — `"visual"` is a genuinely class-agnostic proposer and
  DINOv3 is the thing separating granules from soil, which is the intended design.
- **Where the recall goes.** Recall is flat at ~0.795 for every threshold below 0.5,
  so the ceiling is the *proposal* stage, not the match: ~20% of GT pixels are never
  proposed by `"visual"` at `--score-threshold 0.3`. That is the lever to pull, not
  the cosine threshold.
- **q60 regressed most** (0.606 here vs 0.82-class scores for the clean dense pile in
  earlier pipelines) — slicing a single large pile into 256 px tiles fragments it, and
  `max_area_frac 0.25` then drops the tile-spanning pieces. Dense piles want big slices.
- **Degenerate crops exist.** Transformers logged "channel dimension is ambiguous" for
  crops like 3×20 and 3×2 px — very thin proposals produce near-meaningless embeddings.
  A minimum box side before embedding would remove that noise.

## Next things worth trying (cheap → expensive)

1. `--score-threshold 0.15` and/or `--min-area 2` — directly targets the recall ceiling.
2. `--proposal-text "seed"` in SAHI mode — isolates *prompt* vs *slicing*; `"seed"` was
   worth a lot whole-image and has never been tested sliced.
3. `--slice-size 512` — halves cost, likely recovers q60's pile, may lose small granules.
4. `--subtract-negatives` / `--match-mode knn` — the 58-box background bank is collected
   but unused in this run; the margin score should sharpen the 0.5–0.6 threshold region.
