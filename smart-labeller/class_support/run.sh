python generate_class_supports_main.py \
  --ann_path /fs/ess/PAS2699/brijesh/object-detection/ann/mpala.json \
  --src_path /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/source \
  --output_path /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output \
  --embedding_backend owlv2 bioclip dinov3 \
  --crop_size 1024 \
  --method image \
  --device cuda