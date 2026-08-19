#!/bin/bash
# Stage 4 — compare generated masks against ground-truth masks (mask-IoU, mIoU).
# Picks up the most recent Stage 3 detections; override paths via env vars.
set -euo pipefail

OUT_DIR=${OUT_DIR:-/fs/ess/PAS2699/farmers_meet/output_seg}
GT_FILE=${GT_FILE:-/fs/ess/PAS2699/farmers_meet/annotation/cow_masks.json}

detections=$(ls -t "$OUT_DIR"/detections/detections_dinov3_*.json 2>/dev/null | head -1 || true)

if [ -z "$detections" ]; then
  echo "No detections in $OUT_DIR/detections — run object_classification/run.sh first" >&2
  exit 1
fi

echo "detections : $detections"

python evaluate_annotations.py \
  --gt_file "$GT_FILE" \
  --generated_file "$detections" \
  --similarity_threshold 0.2 \
  --iou_threshold 0.5 \
  --output_path ./results \
  --verbose
