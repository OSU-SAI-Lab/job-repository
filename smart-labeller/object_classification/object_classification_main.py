import os
import time
import numpy as np
from PIL import Image
import argparse
import json
from object_detection.object_classification_utils import run_detection_for_backend, load_model_and_processor
import torch

METHOD = "image"  # image, text, RPN
SRC_PATH = None
QRY_PATH = None
OUTPUT_PATH = None
ANN_FILE_PATH = None
TORCH_DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE = 1024
MODEL_NAME = "google/owlv2-large-patch14-ensemble"

# Multi-backend support:
# EMBEDDING_BACKENDS  – ordered list of backends to run
# CLASS_SUPPORT_FILE_PATHS  – dict: backend → .npz path with class support embeddings
# OBJECT_FEATURES_FILE_PATHS – dict: backend → .npz path with proposal embeddings
EMBEDDING_BACKENDS         = ["owlv2"]
CLASS_SUPPORT_FILE_PATHS   = {}      # populated from args
OBJECT_FEATURES_FILE_PATHS = {}      # populated from args

OBJECTNESS_THRESHOLD  = 0.1
SIMILARITY_THRESHOLD  = 0.2
NMS_IOU_THRESHOLD     = 0.5
PROPOSAL_NMS_THRESHOLD = 0.5
IS_QUERY_DIR = False

## Process a single image across all selected backends
def process(
    query_path=None,
    all_class_supports: dict = None,       # backend → {class → tensor}
    all_object_features: dict = None,      # backend → {features, boxes, scores}
    owlv2_model_bundle=None,
) -> dict:                                 # backend → list of detections

    if not query_path:
        return {}

    print("Current working directory:", os.getcwd())

    image_name = '/workdir/' + query_path[query_path.rfind('/') + 1:]
    per_backend_detections = {}

    for backend in EMBEDDING_BACKENDS:
        class_supports  = all_class_supports.get(backend, {})
        object_features = all_object_features.get(backend, {})

        if not class_supports or object_features.get('features') is None:
            print(f"[{backend}] Skipping — missing class supports or proposal features.")
            per_backend_detections[backend] = []
            continue

        print(f"[{backend}] Running detection...")
        detections = run_detection_for_backend(
            backend=backend,
            class_supports=class_supports,
            object_features=object_features,
            device=DEVICE,
            owlv2_model_bundle=owlv2_model_bundle,
            torch_data_type=TORCH_DTYPE,
            objectness_threshold=OBJECTNESS_THRESHOLD,
            similarity_threshold=SIMILARITY_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )
        print(f"[{backend}] {len(detections)} detections.")
        for det in detections:
            det['image_path'] = image_name
        per_backend_detections[backend] = detections

    return per_backend_detections
        
## save to file system        
def save_detections(annotations, timestamp, backend):
    generated_boxes_dir = os.path.join(OUTPUT_PATH, "detections")
    detection_file_path = os.path.join(generated_boxes_dir, f"detections_{backend}_{SIMILARITY_THRESHOLD}_{NMS_IOU_THRESHOLD}_{PROPOSAL_NMS_THRESHOLD}_{timestamp}.json")
    os.makedirs(generated_boxes_dir, exist_ok=True)
    with open(detection_file_path, "w") as f:
        json.dump({"annotations": annotations}, f)
        print(f"Detections saved to {detection_file_path}")           

## Main function
def main() -> bool:
    global_annotations = {b: [] for b in EMBEDDING_BACKENDS}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Load OWLv2 model only when needed
    owlv2_model_bundle = None
    if "owlv2" in EMBEDDING_BACKENDS:
        print("Loading OWLv2 model...")
        processor, model = load_model_and_processor(MODEL_NAME, DEVICE, TORCH_DTYPE)
        model.to(DEVICE)
        owlv2_model_bundle = (processor, model)

    # Load class support embeddings for each backend
    all_class_supports = {}   # backend → {class_name → tensor}
    for backend, path in CLASS_SUPPORT_FILE_PATHS.items():
        if path and os.path.exists(path):
            print(f"[{backend}] Loading class supports from {path}")
            cs_np = np.load(path)
            all_class_supports[backend] = {
                k: torch.from_numpy(cs_np[k]).to(DEVICE) for k in cs_np.files
            }
        else:
            print(f"[{backend}] Class support file not found: {path}")
            all_class_supports[backend] = {}

    # Load proposal (object) feature arrays for each backend
    all_features_np = {}   # backend → np.NpzFile
    for backend, path in OBJECT_FEATURES_FILE_PATHS.items():
        if path and os.path.exists(path):
            print(f"[{backend}] Loading object features from {path}")
            all_features_np[backend] = np.load(path)
        else:
            print(f"[{backend}] Object features file not found: {path}")
            all_features_np[backend] = None

    def build_object_features(backend, query_file_name):
        """Extract per-image feature tensors for one backend."""
        npz = all_features_np.get(backend)
        if npz is None:
            return {'features': None, 'boxes': None, 'scores': None}
        try:
            return {
                'features': torch.from_numpy(npz[f'{query_file_name}_features']).to(DEVICE),
                'boxes':    torch.from_numpy(npz[f'{query_file_name}_boxes']).to(DEVICE),
                'scores':   torch.from_numpy(npz[f'{query_file_name}_scores']).to(DEVICE),
            }
        except KeyError:
            print(f"[{backend}] No features found for {query_file_name} in {OBJECT_FEATURES_FILE_PATHS[backend]}")
            return {'features': None, 'boxes': None, 'scores': None}

    if IS_QUERY_DIR:
        if not os.path.isdir(QRY_PATH):
            print(f"Provided query path {QRY_PATH} is not a directory.")
            return False
        query_files = [
            os.path.join(QRY_PATH, f) for f in os.listdir(QRY_PATH)
            if os.path.isfile(os.path.join(QRY_PATH, f))
            and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        for query_file in query_files:
            query_file_name = os.path.basename(query_file)
            all_object_features = {
                backend: build_object_features(backend, query_file_name)
                for backend in EMBEDDING_BACKENDS
            }
            per_backend = process(
                query_path=query_file,
                all_class_supports=all_class_supports,
                all_object_features=all_object_features,
                owlv2_model_bundle=owlv2_model_bundle,
            )
            for backend, dets in per_backend.items():
                if dets:
                    global_annotations[backend].extend(dets)
        for backend, annotations in global_annotations.items():
            save_detections(annotations, timestamp, backend)
    else:
        query_file_name = QRY_PATH[QRY_PATH.rfind('/') + 1:]
        all_object_features = {
            backend: build_object_features(backend, query_file_name)
            for backend in EMBEDDING_BACKENDS
        }
        per_backend = process(
            query_path=QRY_PATH,
            all_class_supports=all_class_supports,
            all_object_features=all_object_features,
            owlv2_model_bundle=owlv2_model_bundle,
        )
        for backend, annotations in per_backend.items():
            save_detections(annotations, timestamp, backend)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-backend Object Detector")
    parser.add_argument("--method",       type=str,  default="image", help="Detection method")
    parser.add_argument("--use_sahi",     action="store_true", default=False, help="Use SAHI for slicing")
    parser.add_argument("--qry_path",     type=str,  help="Query image path or directory")
    parser.add_argument("--output_path",  type=str,  default=None, help="Path to save output")
    parser.add_argument("--model_name",   type=str,  default=MODEL_NAME,
                        help="HuggingFace model ID for OWLv2 (only used when owlv2 backend is selected)")
    parser.add_argument("--device",       type=str,  default=DEVICE, help="Device to use")
    parser.add_argument("--embedding_backends", type=str, nargs="+",
                        default=["owlv2"], choices=["owlv2", "bioclip", "dinov3"],
                        help="One or more embedding backends. Example: --embedding_backends owlv2 bioclip dinov3")
    parser.add_argument("--class_support_file_paths", type=str, nargs="+", default=[],
                        help="Per-backend class support .npz paths, aligned with --embedding_backends. "
                             "Example: cs_owlv2.npz cs_bioclip.npz")
    parser.add_argument("--object_features_file_paths", type=str, nargs="+", default=[],
                        help="Per-backend proposal features .npz paths, aligned with --embedding_backends. "
                             "Example: of_owlv2.npz of_bioclip.npz")
    parser.add_argument("--objectness_threshold",  type=float, default=0.1,  help="Objectness threshold")
    parser.add_argument("--similarity_threshold",  type=float, default=0.2,  help="Similarity threshold")
    parser.add_argument("--nms_iou_threshold",      type=float, default=0.5,  help="NMS IoU threshold")
    parser.add_argument("--is_query_dir", action="store_true", help="Treat qry_path as a directory")
    args = parser.parse_args()

    METHOD     = args.method
    QRY_PATH   = args.qry_path
    OUTPUT_PATH = args.output_path
    MODEL_NAME = args.model_name
    DEVICE     = args.device
    OBJECTNESS_THRESHOLD = float(args.objectness_threshold)
    SIMILARITY_THRESHOLD = float(args.similarity_threshold)
    NMS_IOU_THRESHOLD    = float(args.nms_iou_threshold)
    IS_QUERY_DIR         = args.is_query_dir
    EMBEDDING_BACKENDS   = args.embedding_backends

    # Build backend → path dicts from aligned lists
    cs_paths = args.class_support_file_paths
    of_paths = args.object_features_file_paths
    if cs_paths and len(cs_paths) != len(EMBEDDING_BACKENDS):
        parser.error("--class_support_file_paths must have the same number of entries as --embedding_backends")
    if of_paths and len(of_paths) != len(EMBEDDING_BACKENDS):
        parser.error("--object_features_file_paths must have the same number of entries as --embedding_backends")

    CLASS_SUPPORT_FILE_PATHS   = dict(zip(EMBEDDING_BACKENDS, cs_paths)) if cs_paths else {b: None for b in EMBEDDING_BACKENDS}
    OBJECT_FEATURES_FILE_PATHS = dict(zip(EMBEDDING_BACKENDS, of_paths)) if of_paths else {b: None for b in EMBEDDING_BACKENDS}

    main()