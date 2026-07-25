"""Sweep the similarity threshold on an existing detections file, loading and
rasterizing all masks ONCE (the slow part) and re-matching in memory per T."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "smart-labeller"))
import evaluate_annotations as ev

p = argparse.ArgumentParser()
p.add_argument("--gt_file", required=True)
p.add_argument("--generated_file", required=True)
p.add_argument("--iou_threshold", type=float, default=0.5)
p.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.72")
args = p.parse_args()

print("loading GT + detections (rasterizing masks once)...", flush=True)
gt  = ev.load_annotations(args.gt_file)
gen = ev.load_annotations(args.generated_file)

print(f"{'sim>':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'mIoU':>7}  TP/FP/FN")
for T in [float(x) for x in args.thresholds.split(",")]:
    r = ev.evaluate(gt, gen, similarity_threshold=T, iou_threshold=args.iou_threshold)
    print(f"{T:>6.2f} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} "
          f"{r['mIoU']:>7.3f}  {r['total_tp']}/{r['total_fp']}/{r['total_fn']}", flush=True)
