python object_detection_main.py \
  --qry_path /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/query \
  --is_query_dir \
  --embedding_backends owlv2 bioclip dinov3 \
  --class_support_file_paths /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/tensors/class_supports_owlv2_1024_20260303_015528.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/tensors/class_supports_bioclip_1024_20260303_015528.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/tensors/class_supports_dinov3_1024_20260303_015528.npz \
  --object_features_file_paths /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/objects/features_owlv2_owlv2_20260303_015816.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/objects/features_bioclip_owlv2_20260303_015736.npz /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output/objects/features_dinov3_owlv2_20260303_020100.npz \
  --output_path /users/PAS2699/brijeshnandaby/projects/advanced-auto-labeler/output \
