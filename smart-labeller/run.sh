python evaluate_annotations.py \
  --gt_file /fs/ess/PAS2699/farmers_meet/annotation/cow_masks.json \
  --generated_file /fs/ess/PAS2699/farmers_meet/output_seg/detections/detections_dinov3_0.2_0.1_TIMESTAMP.json \
  --similarity_threshold 0.2 \
  --iou_threshold 0.5 \
  --output_path ./results \
  --verbose
