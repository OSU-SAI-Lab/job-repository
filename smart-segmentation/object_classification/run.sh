#!/bin/bash
# Stage 3 — few-shot classification of mask proposals against class supports.
# Picks up the most recent Stage 1 / Stage 2 tensors; override paths via env vars.
set -euo pipefail

OUT_DIR=${OUT_DIR:-/fs/ess/PAS2699/farmers_meet/output_seg}
QRY_DIR=${QRY_DIR:-/fs/ess/PAS2699/farmers_meet/query}

class_supports=$(ls -t "$OUT_DIR"/class_supports/tensors/class_supports_dinov3_*.npz 2>/dev/null | head -1 || true)
object_features=$(ls -t "$OUT_DIR"/proposals/tensors/features_dinov3_sam3_*.npz 2>/dev/null | head -1 || true)

if [ -z "$class_supports" ]; then
  echo "No class supports in $OUT_DIR/class_supports/tensors — run class_support/run.sh first" >&2
  exit 1
fi
if [ -z "$object_features" ]; then
  echo "No proposal features in $OUT_DIR/proposals/tensors — run proposal/run.sh first" >&2
  exit 1
fi

echo "class supports : $class_supports"
echo "features       : $object_features"

python object_classification_main.py \
  --qry_path "$QRY_DIR" \
  --is_query_dir \
  --embedding_backend dinov3 \
  --class_support_file_path "$class_supports" \
  --object_features_file_path "$object_features" \
  --output_path "$OUT_DIR" \
  --similarity_threshold 0.2 \
  --objectness_threshold 0.1 \
  --nms_iou_threshold 0.5
