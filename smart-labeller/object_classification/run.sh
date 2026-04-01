python object_classification_main.py \
  --qry_path /fs/ess/PAS2699/object_detection_datasets/pratham/session_5/DJI_0208/partition_1 \
  --is_query_dir \
  --embedding_backends dinov3 \
  --class_support_file_paths /fs/ess/PAS2699/brijesh/experimentation/mmla/output_se/class_supports/tensors/class_supports_dinov3_1024_20260331_223353.npz \
  --object_features_file_paths /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208/proposals/tensors/features_dinov3_sam3_20260401_102137.npz \
  --output_path /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208 \
  --similarity_threshold 0.1 \
  --objectness_threshold 0.1


  # --class_support_file_paths /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/tensors/class_supports_owlv2_1024_20260303_015528.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/tensors/class_supports_bioclip_1024_20260303_015528.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/tensors/class_supports_dinov3_1024_20260303_015528.npz \
  # --object_features_file_paths /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/objects/features_owlv2_owlv2_20260303_015816.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/objects/features_bioclip_owlv2_20260303_015736.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/objects/features_dinov3_owlv2_20260303_020100.npz \

  # --class_support_file_paths /fs/ess/PAS2699/farmers_meet/output3/class_supports/tensors/class_supports_owlv2_1800_20260325_234712.npz \
  # --object_features_file_paths /fs/ess/PAS2699/farmers_meet/output3/proposals/tensors/features_owlv2_owlv2_20260326_002343.npz \

  # --class_support_file_paths /fs/ess/PAS2699/brijesh/experimentation/mmla/output_se/class_supports/tensors/class_supports_owlv2_1024_20260331_221256.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_se/class_supports/tensors/class_supports_dinov3_1024_20260331_223353.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_se/class_supports/tensors/class_supports_bioclip_1024_20260331_223353.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_se/class_supports/tensors/class_supports_dinov3_1024_20260331_223353.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_se/class_supports/tensors/class_supports_bioclip_1024_20260331_223353.npz \
  # --object_features_file_paths /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208/proposals/tensors/features_owlv2_owlv2_20260401_102137.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208/proposals/tensors/features_dinov3_owlv2_20260401_102137.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208/proposals/tensors/features_bioclip_owlv2_20260401_102137.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208/proposals/tensors/features_dinov3_sam3_20260401_102137.npz /fs/ess/PAS2699/brijesh/experimentation/mmla/output_DJI_0208/proposals/tensors/features_bioclip_sam3_20260401_102137.npz \