# sam3_dino_fss — code walkthrough (file by file)

Proposal-first few-shot segmentation:

```
class supports ──DINOv3 crop-embed──▶ class prototype
query ──SAM3 "everything"──▶ proposals ──DINOv3 crop-embed──▶ cosine vs prototype
                                                                    │
                                       keep proposals ≥ thr ────────┴──▶ union mask
```

Frozen models, no training. Two frozen backbones: **SAM 3** (proposes masks) and
**DINOv3** (embeds crops). The support set is the only label source; the cosine
match is what makes it few-shot.

Read order: `dino_embed.py` → `proposer.py` → `class_supports.py` → `pipeline.py`
→ `run_fertilizer.py`. (`diag_proposer.py` is a side diagnostic.)

---

## `__init__.py`

Package marker. Re-exports the single public entry point so callers can write
`from sam3_dino_fss import SamDinoFSS`. Everything else is internal.

---

## `dino_embed.py` — turn an image REGION into one vector

Class **`DinoEmbedder`**. Wraps a frozen DINOv3 (`AutoModel` +
`AutoImageProcessor`). Its whole job: given an image and a list of boxes, return one
L2-normalised embedding per box.

- **`__init__(model_id, device, amp_dtype, batch_size)`** — loads the DINOv3 weights
  once (`.eval()`), records the hidden size as `self.dim`. `amp_dtype` enables
  bf16/fp16 autocast on CUDA for speed (weights stay fp32).
- **`embed(image, boxes, pad=0.15)`** — the key method.
  1. For each box, expand it by `pad` (15%) on each side so the crop includes a
     little context, clamp to the image bounds, and `image.crop(...)`.
  2. Batch the crops through the processor (which **resizes each crop up** to the
     model's input resolution — e.g. 224/256 px) and the model.
  3. Take **`last_hidden_state[:, 0]`** — the **CLS token** — as the crop's single
     summary vector, `F.normalize` it, move to CPU.
  4. Returns `(N, D)` normalised embeddings.

**Why crop + resize up matters:** a fertilizer granule is smaller than one DINOv3
patch (~16 px). If you pooled the *full-image* patch grid over a granule mask (what
the old `fss --cosine-verify` did) you'd average in surrounding soil and get a
near-background vector. Cropping the granule and letting the processor enlarge it
makes DINOv3 see the granule at full resolution → a clean, discriminative embedding.
This is the single change that made cosine matching work here.

---

## `proposer.py` — SAM 3 "segment everything"

Dataclass **`Proposal`** = `(mask: np.bool, box, score)`.

Class **`Sam3Proposer`**. Wraps the frozen SAM 3 concept model (`Sam3Model` +
`Sam3Processor`).

- **`__init__(model_id, device, amp_dtype, score_threshold, mask_threshold)`** —
  loads SAM 3 once. `score_threshold` drops low-confidence instances;
  `mask_threshold` binarises the predicted mask logits.
- **`propose(image, text)`** — the only real method.
  1. `processor(images=image, text=text, ...)` then `model(**inputs)`.
  2. `post_process_instance_segmentation(...)` → a dict with `masks`, `boxes`,
     `scores` for **every** instance SAM 3 found for the concept `text`.
  3. Convert each mask to a boolean numpy array, drop empties, take the box (from
     SAM, or recomputed from the mask), and wrap as a `Proposal`.
  4. Returns `List[Proposal]`.

**Why a text prompt:** SAM 3 (in the HF API) has no pure geometric
automatic-mask-generator; "everything" is driven by a generic concept word. The
diagnostic (`diag_proposer.py`) showed the word matters enormously — `"fertilizer
granules"` scores ~0.00 (→ 0 proposals), while **`"seed"`** scores up to 0.90 at
granule scale. The text only makes SAM *propose* granule-shaped blobs; the **class
decision is deferred to the DINOv3 cosine match**, so the support set stays the
label source.

---

## `class_supports.py` — build the class prototype

- **`_instance_boxes(mask, min_area)`** — `scipy.ndimage.label` splits a support
  mask into 8-connected components (individual granules) and returns each one's
  bounding box, skipping specks below `min_area`.
- **`build_class_prototype(images, masks, embedder, min_area=4)`** —
  1. For each support (image, mask): get per-instance boxes, then
     `embedder.embed(img, boxes)` → that support's instance embeddings.
  2. Concatenate across all supports → `(M, D)` matrix of instance embeddings.
  3. The **prototype** is the L2-normalised **mean** of all instance embeddings.
  4. Returns `(prototype (D,), instances (M, D))`.

So the class is represented by the average DINOv3 appearance of all labelled
granules. (The raw `instances` are also returned in case you want a k-NN match
instead of a single mean — not used by default.)

---

## `pipeline.py` — the model (`SamDinoFSS`)

The orchestrator that wires the three pieces together. Helpers at top:
`_load_image` (path/array/PIL → RGB PIL) and `_resolve_amp` (dtype string →
autocast dtype, CUDA only).

Class **`SamDinoFSS`**:

- **`__init__(dinov3, sam3, device, dtype, proposal_text, cosine_threshold,
  score_threshold, support_min_area)`** — constructs one `DinoEmbedder` and one
  `Sam3Proposer` (sharing device/amp). Stores the defaults: `proposal_text` (e.g.
  `"seed"`), `cosine_threshold` (keep proposals at/above this cosine), and
  `score_threshold` (SAM confidence floor).
- **`set_support(images, masks)`** — loads the support pairs and calls
  `build_class_prototype(...)`, caching `self._prototype` and `self._instances`.
- **`segment(query, proposal_text=None)`** — the forward pass:
  1. `self.proposer.propose(img, text)` → every SAM 3 proposal. (Empty → return a
     zero mask.)
  2. `self.embedder.embed(img, boxes)` → `(N, D)` proposal embeddings.
  3. **`sims = embs @ self._prototype`** — cosine similarity of each proposal to the
     class prototype (both are L2-normalised, so the dot product *is* cosine).
  4. Keep proposals with `sim >= cosine_threshold`; **union** their masks → final
     mask.
  5. Returns a dict: `mask`, `proposals`, `boxes`, `sims` (per-proposal cosine),
     `kept` (indices), `n_proposals`.

This is the whole model: stages 1–3 are SAM-propose → DINOv3-embed → cosine-select.

---

## `run_fertilizer.py` — the test driver

Self-contained (its own minimal COCO helpers, no dependency on `../fss`).

- **COCO helpers** — `load_coco`, `anns_by_image`, `build_mask` (rasterise polygon
  annotations), `best_crop_window` (pick the 1024² crop with the most granules),
  `prepare_case` (return cropped image + GT mask). `prf` computes precision/recall/
  IoU; `overlay`/`union` are small mask utilities.
- **`main()`**:
  1. Build `SamDinoFSS` with `cosine_threshold=-1` (keep all proposals; the sweep
     applies thresholds offline).
  2. `set_support` from one labelled COCO image.
  3. For each GT query: `segment()` once (SAM runs once), capture every proposal's
     mask + cosine sim + the GT. Print the per-image cosine range.
  4. **Offline cosine-threshold sweep** (`THRESHOLDS = 0.2…0.7`): for each
     threshold, union the kept proposals and average IoU/precision/recall/keep%
     across queries — no SAM reruns. Print the table and the best row.
  5. Save `q*_all_proposals.png` (everything SAM proposed) vs
     `q*_matched_overlay.png` (kept at best threshold), plus a couple of unseen
     overlays, and a `results.json`.

Running SAM once and sweeping offline is why the whole eval is one cheap job.

---

## `diag_proposer.py` — concept-text diagnostic

A standalone probe that loads only `Sam3Proposer` and tries ~14 candidate concept
words (`visual`, `object`, `granule`, `pellet`, `seed`, `small stones`, …) at
`score_threshold=0.0` on two crops, printing proposal counts + score/area ranges.
This is how `"seed"` was found to be the word SAM 3 scores granules highly on. Run
it first whenever moving to a new object class.

---

## Run scripts

- **`run.slurm`** — submits `run_fertilizer.py` on the OSC `debug` partition
  (`--proposal-text "seed" --score-threshold 0.3`), writing to
  `sam3_dino_fss/results/`.
- **`diag.slurm`** — submits `diag_proposer.py`.

Both `module load` conda, activate the `fss` env, set `HF_HOME` to scratch and
`HF_HUB_OFFLINE=1` (weights pre-cached).

---

## End-to-end, in one breath

`set_support` embeds every labelled granule (crop → DINOv3 CLS) and averages them
into a **prototype**. `segment` asks **SAM 3** for every "seed"-like blob in the
query, embeds each proposal crop with the **same DINOv3**, scores each by **cosine**
to the prototype, keeps the matches, and unions them. Best result on the fertilizer
data: **mIoU 0.741 / precision 0.859** — above the prior-driven `../fss` pipeline
(0.645).
