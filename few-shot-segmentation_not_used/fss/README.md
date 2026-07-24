# fss — Training-Free Few-Shot Segmentation (DINOv3 + SAM 3)

Segment a target object in a query image given a small support set (1–5 labeled
examples). No training, no fine-tuning — all models are **frozen, inference only**.

**Approach (match-first, then refine):** DINOv3 dense features *localize* the
target by matching the support example(s) to the query; SAM 3 then *refines* that
localization into a precise mask.

```
support(image+mask) ──DINOv3 match──▶ prior map ──▶ SAM prompts ──▶ SAM 3 ──▶ final mask
```

---

## Architecture

| Module | Class | Role |
|--------|-------|------|
| [matcher.py](matcher.py) | `DINOv3Matcher` | Dense patch features → support prototypes → query **prior map** |
| [prompts.py](prompts.py) | `PromptGenerator` | Prior map → SAM 3 prompts (points, box, optional coarse mask) |
| [segmenter.py](segmenter.py) | `SAM3Segmenter` | SAM 3 refinement (geometric prompts) + native concept (text/exemplar) path; prior-aware candidate selection + prior gating |
| [postprocess.py](postprocess.py) | — | Prior-only mask path (threshold + morphology + edge-snap); shared mask cleanup |
| [pipeline.py](pipeline.py) | `FewShotSegmenter` | Orchestration + prototype caching; eval-variant comparison |
| [config.py](config.py) | `FSSConfig` | All variants/thresholds — swapping models is config-only |
| [viz.py](viz.py) | — | Debug overlay: query + prior heatmap + prompts + final mask |
| [cli.py](cli.py) | — | `python -m fss ...` |
| [eval.py](eval.py) | — | Optional mIoU / FB-IoU harness over episode folders |

### How it works

1. **DINOv3 dense features.** The backbone is run on each image; the
   `last_hidden_state` is `[CLS] + register tokens + patch tokens`. We strip the
   first `1 + num_register_tokens` tokens (read from `model.config`) so only the
   **patch tokens** form the `(H/p, W/p, D)` grid, then L2-normalize per patch.
2. **Prototypes.** Each support mask is resized to the patch grid; foreground
   patches are masked-average-pooled into a foreground prototype (and the
   complement into a background prototype). K-shot averages prototypes across the
   K support items.
3. **Prior map.** Cosine similarity of every query patch vs the foreground
   prototype, minus the background prototype (`subtract`, default) or `softmax`
   over `{fg, bg}`. The coarse grid is bilinearly upsampled to full query
   resolution. An optional **dense 4D correlation** mode (`--corr-4d`) scores each
   query patch against *all* support foreground patches (top-k mean) instead of a
   single averaged prototype — stronger on intra-class appearance variation, at
   `O(N_query × M_support)` cost.
4. **Prompts.** Peaks of the prior → positive points; clear-background regions →
   negative points; the thresholded blob → a tight box (and, optionally, a coarse
   mask prompt). Multiple separated blobs → one prompt set per instance.
5. **SAM 3 refine.** Each prompt set is refined by `Sam3TrackerModel` (Promptable
   Visual Segmentation); the highest predicted-IoU mask is kept.

---

## Model sources (verified)

- **DINOv3** — official Meta release on Hugging Face, `facebook/dinov3-*`.
  Variants exposed as config: `small` (`dinov3-vits16`), `small_plus`,
  `base` (`dinov3-vitb16`, **default**), `large` (`dinov3-vitl16`), all LVD-1689M.
  Docs: <https://huggingface.co/docs/transformers/model_doc/dinov3>
- **SAM 3** — official Meta release `facebook/sam3`.
  - Geometric prompts (points/box/mask, one instance per prompt):
    `Sam3TrackerModel` / `Sam3TrackerProcessor` — Promptable Visual Segmentation.
    Docs: <https://huggingface.co/docs/transformers/model_doc/sam3_tracker>
  - Concept prompts (text / image exemplars, all matching instances):
    `Sam3Model` / `Sam3Processor` — Promptable Concept Segmentation.
    Docs: <https://huggingface.co/docs/transformers/model_doc/sam3>

Each model uses its **own** preprocessing — DINOv3 and SAM 3 transforms are never
shared.

---

## Install

```bash
cd few-shot-segmentation
pip install -r requirements.txt          # or: pip install -e .
```

Requires `transformers>=5.12.0` (SAM 3 / SAM 3 Tracker) and `torch>=2.4`.

### VRAM expectations (rough, fp32 inference)

| DINOv3 variant | + SAM 3 | Notes |
|----------------|---------|-------|
| `small`        | ~4–5 GB | fastest, lowest memory |
| `base` (default) | ~5–6 GB | balanced |
| `large`        | ~7–9 GB | best dense features |

Pass `--device cpu` to run without a GPU (slow), and `--dtype float16` on CUDA to
roughly halve memory. The pipeline auto-detects CUDA else CPU.

---

## End-to-end example

### Python

```python
from fss import FewShotSegmenter

seg = FewShotSegmenter(dinov3="base", sam3="facebook/sam3")   # device auto
seg.set_support(
    images=["support1.png", "support2.png"],   # paths, PIL images, or np arrays
    masks=["mask1.png", "mask2.png"],           # binary masks aligned to images
)
result = seg.segment("query.png")

result["mask"]    # bool (H, W) — union of all instances
result["masks"]   # list of per-instance bool masks
result["score"]   # best instance's predicted IoU
result["box"]     # (x1, y1, x2, y2) of the best instance
result["prior"]   # (H, W) float prior map (for debugging/viz)
```

Visualize:

```python
from fss.viz import save_debug_overlay
save_debug_overlay("query.png", result["prior"], result["prompts"], result, "debug.png")
```

Bypass the matcher with SAM 3's native concept prompt:

```python
result = seg.segment_concept("query.png", text="yellow school bus")
```

### CLI

```bash
python -m fss \
    --support img1.png:mask1.png img2.png:mask2.png \
    --query q.png --out out.png \
    --debug-out debug.png        # optional 4-panel overlay

# multi-instance + dense 4D correlation + larger backbone
python -m fss --support s.png:m.png --query q.png --out out.png \
    --dinov3 large --multi-instance --corr-4d

# concept (matcher-free) mode
python -m fss --query q.png --out out.png --concept-text "yellow school bus"
```

Key flags: `--dinov3 {small,base,large|repo_id}`, `--sam3`, `--device`, `--dtype`,
`--threshold`, `--pos-points`, `--neg-points`, `--multi-instance`, `--corr-4d`,
`--no-bg`, `--mask-prompt`, `--mask-out`, `--debug-out`.

---

## Diffuse / small-scattered targets (per-granule prompting)

For a **diffuse field of small scattered instances** (e.g. fertilizer granules),
SAM 3's object/boundary segmenter is adversarial: it snaps to the strongest-edged
nearby blob, over-includes the soil between granules inside boxes, or picks the
over-inclusive `multimask_output` candidate.

> **Principle.** The prior *localises* (which regions contain granules) and *gates*
> which instances are real. **SAM owns the boundaries** — free to cut tight around
> each granule. The prior's blobby shape must **never** become the output shape.
> An earlier "defer to the prior" experiment violated this (fed the coarse prior as
> SAM's mask input, selected the candidate with max prior overlap, dilated the prior
> in gating) and produced masks that were just the prior blob, labelling inter-granule
> soil. Those footguns are removed; selection/gating may now only **pick or reject** a
> candidate, never grow or reshape it.

**The fix: per-granule prompting.** Detect individual granule seeds (brightness ×
prior local maxima), prompt SAM **once per seed** (a single positive point + a tiny
box sized from the support granule size), and union the per-granule masks. A
**granule size prior** (median support connected-component area) rejects any mask
that spans many granules. **All flags default OFF** — defaults reproduce the
last-good box-based pipeline.

| Flag | Config field | What it does |
|------|--------------|--------------|
| `--per-granule-prompts` | `prompts.per_granule_prompts` | One prompt per detected granule seed; union per-granule masks. **The core fix.** |
| `--seed-source {brightness,prior,both}` | `prompts.seed_source` | Seed score: granule brightness, prior, or `brightness × prior` (default). |
| `--granule-min-distance` | `prompts.granule_min_distance` | NMS spacing between seeds (≈ granule radius). |
| `--no-granule-box` | `prompts.granule_box` | Point-only seeds (drop the tiny per-seed box). |
| `--selection granule` | `segmenter.selection_criterion` | Among the 3 candidates keep those whose **centroid is in the prior**, then pick the **compact** one within the size prior. Never picks by max prior overlap. |
| `--size-gate` | `segmenter.size_gate` | Reject any SAM mask whose area ≫ expected granule area — directly kills the blob. |
| `--granule-size-mult` | `segmenter.granule_size_mult` | Size-gate multiplier on the expected granule area (default 4×). |
| `--prior-gating` | `segmenter.prior_gating` | Reject instances barely overlapping the **un-dilated** prior (accept/reject only). |
| `--min-prior-overlap` | `segmenter.min_prior_overlap` | Overlap fraction below which an instance is rejected. |
| `--intersect-prior-safety` | `segmenter.intersect_prior_safety` | Intersect the final mask with the **un-dilated** prior (never dilated). |
| `--use-mask-prompt` | `prompts.use_mask_prompt` | If a mask prompt is used at all, feed only a heavily **eroded high-confidence core** (`prompts.mask_prompt_erosion`) — never the full blob. |
| `--eval-variants` (+`--gt`) | `eval_variants` | Compare `prior_only` / `sam3` / `granule_sam3`; with `--gt`, prints IoU / precision / recall / bgFP per variant. |
| `--prior-only-threshold`, `--edge-snap` | `prior_mask.*` | The SAM-free `prior_only` diagnostic baseline. |

```bash
# Recommended config for scattered granules (granule-tight, soil unlabeled):
python -m fss --support s.png:m.png --query q.png --out out.png \
    --per-granule-prompts --selection granule --size-gate \
    --intersect-prior-safety

# Diagnostic: granule_sam3 vs the blobby sam3 vs prior_only, with metrics:
python -m fss --support s.png:m.png --query q.png --out out.png \
    --per-granule-prompts --selection granule --size-gate \
    --eval-variants --gt q_gt.png
```

The eval harness reports **precision and recall separately plus bgFP**
(background-false-positive rate inside the mask = 1 − precision) — IoU alone hides
blobbiness — and flags whether `granule_sam3` improves precision over `sam3`
without collapsing recall:

```bash
python -m fss.eval --episodes /path/to/episodes --per-granule-prompts \
    --selection granule --size-gate          # variants on by default
```

---

## Tests

```bash
pytest fss/tests/                 # lightweight prompt/config tests always run
pytest fss/tests/ -m slow         # full end-to-end smoke (skips if weights absent)
```

---

## Evaluation (optional / stretch)

```bash
python -m fss.eval --episodes /path/to/episodes --dinov3 base
```

Each episode is a JSON (`support`/`query`/`gt`) — see [eval.py](eval.py). Reports
mIoU and FB-IoU; point it at episodes exported from PASCAL-5i or COCO-20i.

---

## Notes & caveats

- **Patch-token stripping is load-bearing.** The matcher asserts
  `Hp*Wp == n_patch_tokens`; if the register-token count is wrong the prior map is
  silently corrupted, so the assert fails loudly instead.
- **Mask prompt (`--mask-prompt`) is off by default.** SAM 3's tracker accepts a
  low-resolution logit mask, but the exact size is model-dependent; the coarse
  prior is converted best-effort and points+box are generally more reliable.
- **Multi-instance** emits one prompt set per separated prior blob; each is refined
  independently and the results are merged.
