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
MASK_BACKGROUND = "zero"             # 'zero'|'mean'|'none' — GT background suppression
TORCH_DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
patch_size_to_iou_score_map = {}

def main(crop_size: int, backends_to_run: list = None) -> tuple:
    """
    Main function to extract class support embeddings.
    
    Args:
        crop_size: The crop size to use for this iteration
        backends_to_run: List of backends to run for this crop_size. 
                        If None, runs all backends in EMBEDDING_BACKENDS.
    
    Returns:
        (success: bool, generated_boxes: list)
    """
    if backends_to_run is None:
        backends_to_run = EMBEDDING_BACKENDS
    
    print("Current working directory:", os.getcwd())
    print(f"Running backends: {backends_to_run}")
    print(f"Crop size: {crop_size}")

    if ANN_FILE_PATH is None:
        print("No annotation file provided.")
        return False, []
            

    with open(ANN_FILE_PATH, "r") as f:
        support_examples = json.load(f).get("annotations", [])
    print(f"Loaded {len(support_examples)} support examples from {ANN_FILE_PATH}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generated_boxes_dir = os.path.join(OUTPUT_PATH, "class_supports/generated_boxes")
    tensors_dir = os.path.join(OUTPUT_PATH, "class_supports/tensors")
    os.makedirs(generated_boxes_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    generated_boxes = []
    boxes_saved = False

    for backend in backends_to_run:
        print(f"\n── Running backend: {backend} ──")
        backend_model_name = MODEL_NAME
        if MODEL_NAME is not None:
            lname = MODEL_NAME.lower()
            if backend == "dinov3" and "owlv2" in lname:
                print(f"Warning: ignoring --model_name '{MODEL_NAME}' for dinov3 backend (incompatible model id).")
                backend_model_name = None
            elif backend == "owlv2" and "dinov3" in lname:
                print(f"Warning: ignoring --model_name '{MODEL_NAME}' for owlv2 backend (incompatible model id).")
                backend_model_name = None

        model_bundle = load_embedding_model(backend, DEVICE, TORCH_DTYPE, backend_model_name)

        class_supports, generated_boxes = extract_support_embeddings(
            support_examples, backend, model_bundle, DEVICE, src_path=SRC_PATH,
            crop_size=crop_size, mask_background=MASK_BACKGROUND,
        )

        if backend == "owlv2":
            gen_boxes_filename = f"generated_boxes_{crop_size}_{timestamp}.json"
            gen_boxes_path = os.path.join(generated_boxes_dir, gen_boxes_filename)
            with open(gen_boxes_path, "w") as f:
                json.dump({"annotations": generated_boxes}, f, indent=2)
            print(f"Saved generated boxes → {gen_boxes_path}")

        # Save per-backend embeddings
        tensor_filename = f"class_supports_{backend}_{crop_size}_{timestamp}.npz"
        tensor_path = os.path.join(tensors_dir, tensor_filename)
        class_supports_np = {k: v.cpu().numpy() for k, v in class_supports.items()}
        np.savez(tensor_path, **class_supports_np)
        print(f"Saved class supports [{backend}] → {tensor_path}")

    print("\nDone.")
    return True, generated_boxes


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
    parser.add_argument("--mask_background",   type=str,   default="zero",
                        choices=["zero", "mean", "none"],
                        help="Background suppression for GT crops when a segmentation "
                             "mask is present (dinov3/bioclip): 'zero' blacks out pixels "
                             "outside the mask, 'mean' fills with the crop mean, 'none' "
                             "disables masking (plain box crop). Default: zero.")
    args = parser.parse_args()

    METHOD           = args.method
    SRC_PATH         = args.src_path
    OUTPUT_PATH      = args.output_path
    ANN_FILE_PATH    = args.ann_path
    EMBEDDING_BACKENDS = args.embedding_backend
    MODEL_NAME         = args.model_name
    DEVICE           = args.device
    MASK_BACKGROUND  = args.mask_background
    # Parse crop_size: can be "1024" or "[1024,760]" or "[1024, 760]"
    crop_size_str = args.crop_size.strip()
    if crop_size_str.startswith('['):
        CROP_SIZE_LIST = json.loads(crop_size_str)  # Handles spaces fine
    else:
        CROP_SIZE_LIST = [int(crop_size_str)]  # Single value
    CROP_SIZE_LIST = sorted(map(int, CROP_SIZE_LIST), reverse=True)
    
    # Separate OWLv2 from other backends
    owlv2_backends = ["owlv2"] if "owlv2" in EMBEDDING_BACKENDS else []
    other_backends = [b for b in EMBEDDING_BACKENDS if b != "owlv2"]
    
    patch_size_to_iou_score_map = {}
    
    # Process OWLv2 across all crop sizes (with optimization)
    for crop_size in CROP_SIZE_LIST:
        print(f"\n{'='*60}")
        print(f"Processing crop size: {crop_size}")
        print(f"{'='*60}")
        
        if owlv2_backends:
            success, generated_boxes = main(crop_size, backends_to_run=owlv2_backends)
            
            if not success:
                print(f"Failed to process crop size {crop_size}")
                break
            
            # Check IoU progression for OWLv2
            if generated_boxes:
                next_iter, iou_map = calculate_change_in_iou(crop_size, generated_boxes, patch_size_to_iou_score_map)
                patch_size_to_iou_score_map = iou_map
                print(f"\nIoU Analysis (OWLv2):")
                print(f"  Patch size to IoU score map: {patch_size_to_iou_score_map}")
                
                if next_iter:
                    print(f"  ✓ Significant improvement detected for crop size {crop_size}")
                    print(f"    Continuing with smaller crop sizes...")
                else:
                    print(f"  ✗ No significant improvement in IoU detected")
                    print(f"    Stopping iteration (crop size {crop_size} is optimal)")
                    break
    
    # Run other backends only once (on the first/largest crop size)
    if other_backends and CROP_SIZE_LIST:
        largest_crop_size = CROP_SIZE_LIST[0]
        print(f"\n{'='*60}")
        print(f"Processing other backends (DINOv3, BioCLIP) with crop size: {largest_crop_size}")
        print(f"{'='*60}")
        
        success, _ = main(largest_crop_size, backends_to_run=other_backends)
        
        if not success:
            print(f"Failed to process other backends")
    
    print(f"\n{'='*60}")
    print("All processing completed successfully.")
    print(f"{'='*60}")