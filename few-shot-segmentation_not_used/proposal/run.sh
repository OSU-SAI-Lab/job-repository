python generate_proposals.py \
  --proposers owlv2 sam3 \
  --image_dir /fs/ess/PAS2699/farmers_meet/query \
  --output_dir /fs/ess/PAS2699/farmers_meet/output5 \
  --embedders bioclip dinov3 owlv2\
  --confidence 0.1 \
  --batch_size 1 \
  --is_sahi \
  --tile_size 1024 \
  --overlap_ratio 0.2