python object_classification_main.py \
  --qry_path /fs/ess/PAS2699/farmers_meet/query \
  --is_query_dir \
  --embedding_backend dinov3 \
  --class_support_file_path /fs/ess/PAS2699/farmers_meet/output_seg/class_supports/tensors/class_supports_dinov3_TIMESTAMP.npz \
  --object_features_file_path /fs/ess/PAS2699/farmers_meet/output_seg/proposals/tensors/features_dinov3_sam3_TIMESTAMP.npz \
  --output_path /fs/ess/PAS2699/farmers_meet/output_seg \
  --similarity_threshold 0.2 \
  --objectness_threshold 0.1 \
  --nms_iou_threshold 0.5
