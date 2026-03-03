import os
import time
from class_supports_utils import (
    calculate_change_in_iou,
    extract_support_embeddings,
    load_embedding_model,
    DEFAULT_OWLV2_MODEL,
    DEFAULT_DINOV3_MODEL,
)
import numpy as np
import argparse
import json
import torch


CROP_SIZE = '1024'
EMBEDDING_BACKENDS = ["owlv2"]       # one or more of: owlv2, bioclip, dinov3
MODEL_NAME = None                    # None → use backend default
METHOD = "image"                     # image, text, RPN
SRC_PATH = None
OUTPUT_PATH = None
ANN_FILE_PATH = None
TORCH_DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
patch_size_to_iou_score_map = {}

def main() -> bool:
    global patch_size_to_iou_score_map

    print("Current working directory:", os.getcwd())
    print(f"Embedding backends: {EMBEDDING_BACKENDS}")

    if ANN_FILE_PATH is None:
        print("No annotation file provided.")
        return False

    with open(ANN_FILE_PATH, "r") as f:
        support_examples = json.load(f).get("annotations", [])
    print(f"Loaded {len(support_examples)} support examples from {ANN_FILE_PATH}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generated_boxes_dir = os.path.join(OUTPUT_PATH, "generated_boxes")
    tensors_dir = os.path.join(OUTPUT_PATH, "tensors")
    os.makedirs(generated_boxes_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    # GT boxes are backend-independent — save them once
    boxes_saved   = False
    generated_boxes = []

    for backend in EMBEDDING_BACKENDS:
        print(f"\n── Running backend: {backend} ──")
        model_bundle = load_embedding_model(backend, DEVICE, TORCH_DTYPE, MODEL_NAME)

        class_supports, generated_boxes = extract_support_embeddings(
            support_examples, backend, model_bundle, DEVICE, src_path=SRC_PATH
        )

        # Save the GT-box JSON only on the first backend pass
        if not boxes_saved:
            gen_boxes_filename = f"generated_boxes_{CROP_SIZE}_{timestamp}.json"
            gen_boxes_path = os.path.join(generated_boxes_dir, gen_boxes_filename)
            with open(gen_boxes_path, "w") as f:
                json.dump({"annotations": generated_boxes}, f, indent=2)
            print(f"Saved generated boxes → {gen_boxes_path}")
            boxes_saved = True

        # Save per-backend embeddings
        tensor_filename = f"class_supports_{backend}_{CROP_SIZE}_{timestamp}.npz"
        tensor_path = os.path.join(tensors_dir, tensor_filename)
        class_supports_np = {k: v.cpu().numpy() for k, v in class_supports.items()}
        np.savez(tensor_path, **class_supports_np)
        print(f"Saved class supports [{backend}] → {tensor_path}")

    print("\nDone.")

    next_iter, iou_map = calculate_change_in_iou(CROP_SIZE, generated_boxes, patch_size_to_iou_score_map)
    patch_size_to_iou_score_map = iou_map
    print(f"Patch size to IoU score map: {patch_size_to_iou_score_map}")
    if next_iter:
        print(f"Significant improvement in IoU detected for patch size {CROP_SIZE}. Consider running with a smaller patch size.")
        return True
    else:
        print("No significant improvement in IoU detected.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-Support Embedding Generator")
    parser.add_argument("--method",            type=str,   default="image",
                        choices=["image", "text", "RPN"],  help="Detection method")
    parser.add_argument("--src_path",          type=str,   help="Source image path")
    parser.add_argument("--use_sahi",          action="store_true", default=False,
                        help="Use SAHI for slicing")
    parser.add_argument("--ann_path",          type=str,   default=None,
                        help="Annotation file path")
    parser.add_argument("--crop_size",         type=str,   default=CROP_SIZE,
                        help="Crop size (single int or JSON list, e.g. '[512,1024]')")
    parser.add_argument("--output_path",       type=str,   default=None,
                        help="Path to save output")
    parser.add_argument("--embedding_backend", type=str,   default=["owlv2"],
                        choices=["owlv2", "bioclip", "dinov3"],
                        nargs="+",
                        help="One or more embedding backends to run (default: owlv2). "
                             "Example: --embedding_backend owlv2 bioclip dinov3")
    parser.add_argument("--model_name",        type=str,   default=None,
                        help="HuggingFace model ID override for the chosen backend "
                             f"(owlv2 default: {DEFAULT_OWLV2_MODEL}; "
                             f"dinov3 default: {DEFAULT_DINOV3_MODEL}; "
                             "bioclip has no override)")
    parser.add_argument("--device",            type=str,   default=DEVICE,
                        help="Device to use (default: auto)")
    args = parser.parse_args()

    METHOD           = args.method
    SRC_PATH         = args.src_path
    OUTPUT_PATH      = args.output_path
    ANN_FILE_PATH    = args.ann_path
    EMBEDDING_BACKENDS = args.embedding_backend
    MODEL_NAME         = args.model_name
    DEVICE           = args.device
    CROP_SIZE_LIST   = json.loads(args.crop_size) if args.crop_size.startswith('[') else [args.crop_size]
    CROP_SIZE_LIST   = sorted(map(int, CROP_SIZE_LIST), reverse=True)
    for crop_size in CROP_SIZE_LIST:
        print(f"Processing with crop size: {crop_size}")
        CROP_SIZE = int(crop_size)
        res = main()
        if not res:
            break