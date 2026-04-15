python evaluate_annotations.py \
  --gt_file /fs/ess/PAS2699/brijesh/experimentation/mmla/gt/default/DJI_0210-partition_1.json \
  --generated_file /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0210/detect/detections_bioclip_0.1_0.1_20260402_003615.json \
  --similarity_threshold 0.88 \
  --iou_threshold 0.5 \
  --output_path ./results \
  --verbose