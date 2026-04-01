python generate_proposals.py \
  --proposers sam3 \
  --image_dir /fs/ess/PAS2699/object_detection_datasets/pratham/session_5/DJI_0208/partition_1 \
  --output_dir /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208 \
  --embedders bioclip \
  --confidence 0.1 \
  --batch_size 1 \
  # --is_sahi