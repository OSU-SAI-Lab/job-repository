---
name: corn-residue-fewshot-setup
description: Per-piece few-shot residue pipeline — train/test/infer split, GT build, run commands
metadata:
  type: project
---

Corn-residue few-shot per-piece segmentation at `/fs/ess/PAS2699/kamath/corn_residue/v1/` ($BASE).

**Goal/split:** TRAIN supports on field_1+field_3 (20 imgs), TEST on field_2 (10 imgs),
then auto-annotate the 270 unlabeled `query/field_{1,2,3}` images (90 each).

**Method (the right one):** DINOv3 mask-cropped embeddings as class supports ↔ SAM mask
proposals embedded the same way ↔ cosine classify. The old OWLv2 box path scored 0 because
GT/supports were whole-frame; see [[corn-residue-eval-limitation]].

**SAM3 concept mode was wrong; use SAM-AMG.** SAM3 text-prompt ("corn residue") proposed only
~9 regions total (concept head returns few coarse instances, not per-piece). Replaced with a SAM
automatic-mask-generation "segment everything" proposer `proposal/sam_amg_proposal.py`
(facebook/sam-vit-base, point grid) registered as proposer `sam_amg` in generate_proposals.py.
Yields 600-1300 masks/image. Tunables via env SAM_AMG_{POINTS,PRED_IOU,STABILITY,MIN_AREA,MAX_AREA_FRAC}
+ NMS_IOU (all wired into mask_step2_proposals.sh).

**Two more bugs fixed:** (1) evaluate_annotations.py compute_mask_iou crashed silently on
uncompressed-RLE (list counts) GT → always returned IoU 0 → 0 TP; fixed to normalize via
frPyObjects. Also keyed by basename. (2) classifier had a hard top-100 proposals/image cap
(object_classification_utils.py) throttling recall on dense scenes; made configurable via
`--max_proposals` (mask_step3 default 1500).

**Results (test=field_2, IoU 0.5, max_proposals 1500):** per-piece R=0.65 P=0.19 F1=0.29;
pixel-coverage IoU=0.36 R=0.73 P=0.43. **Key limits:** cosine similarity is NON-DISCRIMINATIVE
(99% of proposals score >0.7 vs residue supports — soil/residue look alike to DINOv3, so
similarity_threshold does nothing; F1 flat 0.2-0.7), and ~3.5x over-proposal caps precision at
~0.29. Tuning AMG (fewer/tighter masks: PRED_IOU 0.88, STABILITY 0.92, MIN_AREA 120, NMS 0.3) to
lift precision is the chosen next step. Corn residue is a diffuse texture — semantic segmentation
is likely the higher-ceiling route if few-shot precision stays low.

**Key fix this session:** hand masks `field_*/masks/<img>_gt.png` were single binary blobs
(1 whole-image mask). Split into connected-component PIECES via `jobs/build_piece_annotations.py`
(--min_area 50, 8-conn, emits COCO uncompressed RLE) → `gt/train_field1_field3.json` (4369 pieces),
`gt/test_field2.json` (3288), `gt/field_{1,2,3}_pieces.json`. `image_path` is relative to $BASE.
`test/field_2/` = symlinks to the 10 annotated field_2 images (so steps run only on GT'd images).

**Run:** `bash $BASE/jobs/mask_run_all.sh` (step1 train→2 proposals→3 classify→4 eval, SLURM-chained;
metrics in `output/test_field2/eval/summary.json`). Then `bash $BASE/jobs/mask_run_inference.sh` for
the 270. `SKIP_STEP1=1` reuses supports. step2/3 are generalized to QUERY_DIR+TAG env vars.

**Why per-piece matters:** support crops must match SAM3 proposal granularity; whole-frame masks
broke cosine matching. **How to apply:** if recall low lower `--similarity_threshold` (step3, dflt 0.2);
tune SAM3 `TEXT_PROMPT` (dflt "corn residue"); `--min_area` is the piece noise floor.
Full details in `$BASE/jobs/HANDOFF.md`.
