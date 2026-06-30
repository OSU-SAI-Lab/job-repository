# sam3_dino_fss — proposal-first few-shot segmentation

A **separate** pipeline from `../fss` (which is prior-driven). Here SAM 3 proposes
first and DINOv3 cosine-matching selects the target class:

```
class supports ──DINOv3 crop-embed──▶ class prototype (mean CLS, L2-norm)
query ──SAM3 "everything" (concept text)──▶ proposals ──DINOv3 crop-embed──▶ cosine
                                                                              │
                                       keep proposals with cosine ≥ thr ──────┴──▶ union mask
```

| stage | file | in → out |
|---|---|---|
| class supports | `class_supports.py` | support (image, mask) → per-instance bbox crops → DINOv3 CLS → **mean prototype** |
| embed | `dino_embed.py` | image + boxes → cropped (+padded) regions resized up → **DINOv3 CLS, L2-norm** |
| propose | `proposer.py` | query + concept text → **all SAM 3 instance masks/boxes/scores** |
| match | `pipeline.py` `SamDinoFSS.segment` | embed each proposal → cosine vs prototype → keep ≥ `cosine_threshold` → union |

Key design choice vs the in-place `fss --cosine-verify`: proposals are embedded by
**cropping and resizing up**, so sub-patch granules are enlarged to fill DINOv3's
receptive field instead of being averaged into a soil patch.

Note: SAM 3 has no pure geometric "everything" generator in the HF API, so
`proposal_text` (default `"fertilizer granules"`) drives a generic concept; the
DINOv3 cosine match is what makes it few-shot (the support set is the only labels).

## Run

```bash
sbatch sam3_dino_fss/run.slurm          # OSC debug partition
# or:
python sam3_dino_fss/run_fertilizer.py --out-dir sam3_dino_fss/results
```

Outputs a cosine-threshold sweep (mIoU/precision/recall/keep%), `results.json`, and
`*_all_proposals.png` vs `*_matched_overlay.png` overlays per query.
