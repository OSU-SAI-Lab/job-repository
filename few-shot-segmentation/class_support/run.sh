python generate_class_supports_main.py \
  --ann_path /fs/ess/PAS2699/farmers_meet/annotation/cow.json \
  --src_path /fs/ess/PAS2699/farmers_meet/source \
  --output_path /fs/ess/PAS2699/farmers_meet/output5 \
  --embedding_backend dinov3 \
  --crop_size '750' \
  --method image \
  --device cuda