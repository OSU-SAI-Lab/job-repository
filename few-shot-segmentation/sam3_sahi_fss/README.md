# sam3_sahi_fss — SAHI SAM 3 proposals + DINOv3 prototype matching

A third pipeline alongside `../fss` (prior-driven) and `../sam3_dino_fss` (whole-image
SAM 3 proposals). Here SAM 3 runs with the deliberately generic text prompt `"visual"`
on **overlapping slices**, so small objects are found at all, and DINOv3 decides which
of those proposals are the target class.

```
support (image, mask) ──SAHI SAM3("visual")──▶ boxes ──labelled by GT overlap──┐
                                                                              │
                                    DINOv3 crop-embed ──▶ prototype set ◀──────┘
                                                (positives + background negatives)

query ──SAHI SAM3("visual")──▶ boxes ──DINOv3 crop-embed──▶ cosine vs prototype set
                                                                    │
                                           keep proposals ≥ thr ────┴──▶ union mask
```

| stage | file | in → out |
|---|---|---|
| slice/merge | `slicer.py` | image size → overlapping windows; instance boxes → NMS; masks → union |
| propose | `proposer.py` | image + text → SAM 3 per slice → **all instances in global coords** |
| prototypes | `prototypes.py` | support proposals + GT → **positive/negative DINOv3 banks + mean prototype** |
| match | `pipeline.py` `Sam3SahiDinoFSS.segment` | query proposals → cosine vs prototype set → keep ≥ `cosine_threshold` → union |

Two deliberate differences from `../sam3_dino_fss`:

1. **Sliced inference.** A 20 px granule is ~2% of a 1024 px frame but ~8% of a 256 px
   slice — above SAM 3's effective resolution floor. Slice overlap duplicates are
   removed with greedy box NMS.
2. **Prototypes come from proposal boxes, not GT components.** The support mask only
   *labels* SAHI proposals (positive if ≥ `pos_coverage` of the proposal's pixels are
   inside the annotation). Support and query embeddings then come from the same box
   distribution, so cosine scores are comparable. Proposals that miss the annotation
   entirely (≤ `neg_coverage`) become a free background bank — `--subtract-negatives`
   turns the score into a margin (positive − best background), which recentres the
   useful threshold near 0 instead of near the class-agnostic similarity floor.

Instance masks are stored cropped to their own bbox (`slicer.Instance`), because a
"visual" prompt over 25 slices returns hundreds of instances per image.

## Run

```bash
sbatch sam3_sahi_fss/run.slurm                        # OSC debug partition
python sam3_sahi_fss/run_fertilizer.py --max-queries 7 # smoke test
python sam3_sahi_fss/run_fertilizer.py                 # every annotated COCO image
```

Cost scales with slices: a 1024 crop at `--slice-size 256 --overlap-ratio 0.2` is 16
SAM 3 forwards per image (vs 1 for `sam3_dino_fss`). Raise `--slice-size` to 512 to
quarter that if recall on small objects already looks fine.

Outputs a cosine-threshold sweep (mIoU/precision/recall/keep%), `results.json` with
per-query stats, and `q*_all_proposals.png` vs `q*_matched_overlay.png` overlays.

## Knobs that matter

| flag | default | effect |
|---|---|---|
| `--proposal-text` | `visual` | the "everything" concept; recall-oriented, correctness is DINOv3's job |
| `--slice-size` / `--overlap-ratio` | 256 / 0.2 | proposal recall on small objects vs runtime |
| `--score-threshold` | 0.3 | SAM 3 confidence floor — lower = more (noisier) proposals |
| `--max-area-frac` | 0.25 | drops slice-wide blobs (`"visual"` often returns the whole tile) |
| `--pos-coverage` | 0.6 | how much of a support proposal must sit inside the GT to count as positive |
| `--match-mode` | `prototype` | `knn` (mean top-k over the positive bank) if the class is multi-modal |
