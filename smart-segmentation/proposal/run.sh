python generate_proposals.py \
  --image_dir /fs/ess/PAS2699/farmers_meet/query \
  --output_dir /fs/ess/PAS2699/farmers_meet/output_seg \
  --embedder dinov3 \
  --text_prompt visual \
  --confidence 0.1 \
  --batch_size 1 \
  --tile_size 1024 \
  --overlap_ratio 0.2 \
  --nms_iou 0.5
  # SAHI is ON by default; add --no_sahi to process whole images
