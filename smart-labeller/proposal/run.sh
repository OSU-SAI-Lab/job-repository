python generate_proposals.py \
  --backend sam3 \
  --image_dir /fs/ess/PAS2699/farmers_meet/query \
  --output_dir /fs/ess/PAS2699/farmers_meet/output_run \
  --embedding_backend bioclip \
  --confidence 0.2 \
  --batch_size 2 \
  # --is_sahi