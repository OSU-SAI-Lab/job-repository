python generate_class_supports_main.py \
  --ann_path /fs/ess/PAS2699/farmers_meet/annotation/cow.json \
  --src_path /fs/ess/PAS2699/farmers_meet/source \
  --output_path /fs/ess/PAS2699/farmers_meet/output_run \
  --embedding_backend owlv2 bioclip dinov3 \
  --crop_size '[2048,860]' \
  --method image \
  --device cuda