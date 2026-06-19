"""
evaluate_annotations.py - Evaluate object detection annotations against ground truth.

Calculates precision metrics by:
1. Loading ground truth annotations
2. Loading AI-generated annotations
3. Filtering by similarity threshold
4. Matching boxes using IoU (Intersection over Union)
5. Computing precision = TP / (TP + FP)

Usage:
    python evaluate_annotations.py \
      --gt_file path/to/ground_truth.json \
      --generated_file path/to/generated.json \
      --similarity_threshold 0.5 \
      --iou_threshold 0.5 \
      --output_path ./results

Output:
    - Precision score
    - Per-image breakdown
    - Detailed matching report (optional JSON)
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou


def load_annotations(file_path):
    """
    Load annotations from JSON file.
    
    Expected format:
    {
        "annotations": [
            {
                "image_path": "image_name.jpg",
                "bounding_box": [x1, y1, x2, y2],
                "class": "class_name",
                "score": 0.95  # Optional, for generated annotations
            },
            ...
        ]
    }
    
    Returns:
        dict: image_path -> list of annotations
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    annotations_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        img_path = ann["image_path"]
        annotations_by_image[img_path].append(ann)
    
    return annotations_by_image


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU score (0-1)
    """
    boxes1 = torch.tensor([box1], dtype=torch.float32)
    boxes2 = torch.tensor([box2], dtype=torch.float32)
    iou = box_iou(boxes1, boxes2)
    return iou.item()


def match_boxes(gt_boxes, generated_boxes, iou_threshold=0.5, debug=False):
    """
    Match generated boxes to GT boxes using IoU.
    
    Args:
        gt_boxes: list of [x1, y1, x2, y2]
        generated_boxes: list of [x1, y1, x2, y2]
        iou_threshold: minimum IoU to consider a match
        debug: print detailed matching info
    
    Returns:
        dict with:
            - matched_pairs: list of (gt_idx, gen_idx, iou) tuples
            - unmatched_gt: list of GT indices
            - unmatched_gen: list of generated indices
    """
    if debug:
        print(f"\n  [Matching] GT boxes: {len(gt_boxes)}, Generated boxes: {len(generated_boxes)}")
        if gt_boxes:
            print(f"    GT sample: {gt_boxes[0]}")
        if generated_boxes:
            print(f"    Gen sample: {generated_boxes[0]}")
    
    matched_pairs = []
    matched_gen_idxs = set()
    matched_gt_idxs = set()
    
    # For each generated box, find best matching GT box
    for gen_idx, gen_box in enumerate(generated_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt_idxs:
                continue
            
            iou = compute_iou(gt_box, gen_box)
            if debug and gen_idx == 0 and gt_idx < 3:  # Show first generated box matching attempts
                print(f"    gen[{gen_idx}] vs gt[{gt_idx}]: IoU={iou:.4f}")
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if debug and gen_idx < 3:
            print(f"  gen[{gen_idx}]: best_iou={best_iou:.4f} (threshold={iou_threshold}), matched={best_iou >= iou_threshold}")
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_pairs.append((best_gt_idx, gen_idx, best_iou))
            matched_gen_idxs.add(gen_idx)
            matched_gt_idxs.add(best_gt_idx)
    
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt_idxs]
    unmatched_gen = [i for i in range(len(generated_boxes)) if i not in matched_gen_idxs]
    
    return {
        "matched_pairs": matched_pairs,
        "unmatched_gt": unmatched_gt,
        "unmatched_gen": unmatched_gen,
    }


def evaluate(gt_annotations, generated_annotations, similarity_threshold=0.5, iou_threshold=0.5, debug=False):
    """
    Evaluate generated annotations against ground truth.
    
    Args:
        gt_annotations: dict[image_path -> list of GT annotations]
        generated_annotations: dict[image_path -> list of generated annotations]
        similarity_threshold: filter generated annotations by score (optional)
        iou_threshold: minimum IoU to consider a match
        debug: print detailed matching info
    
    Returns:
        dict with metrics and per-image breakdown
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    per_image_results = {}
    
    # Get all unique images from both GT and generated
    all_images = set(gt_annotations.keys()) | set(generated_annotations.keys())
    
    if debug:
        print(f"\n[Evaluation] GT images: {len(gt_annotations)}, Generated images: {len(generated_annotations)}")
        print(f"[Evaluation] Unique images: {len(all_images)}")
        if gt_annotations and generated_annotations:
            gt_keys = list(gt_annotations.keys())[:3]
            gen_keys = list(generated_annotations.keys())[:3]
            print(f"  Sample GT keys: {gt_keys}")
            print(f"  Sample Gen keys: {gen_keys}")
    
    for image_path in sorted(all_images):
        gt_anns = gt_annotations.get(image_path, [])
        gen_anns = generated_annotations.get(image_path, [])
        
        # Filter generated annotations by similarity threshold
        if similarity_threshold > 0:
            gen_anns = [ann for ann in gen_anns if ann.get("score", 1.0) >= similarity_threshold]
        
        # Extract bounding boxes
        gt_boxes = [ann["bounding_box"] for ann in gt_anns]
        gen_boxes = [ann["bounding_box"] for ann in gen_anns]
        
        if debug and (len(gt_boxes) > 0 or len(gen_boxes) > 0):
            print(f"\n[Image] {image_path}")
            print(f"  GT boxes: {gt_boxes}")
            print(f"  Gen boxes: {gen_boxes}")
        
        # Match boxes
        matching = match_boxes(gt_boxes, gen_boxes, iou_threshold=iou_threshold, debug=debug)
        
        # Count TP, FP, FN for this image
        num_tp = len(matching["matched_pairs"])
        num_fp = len(matching["unmatched_gen"])
        num_fn = len(matching["unmatched_gt"])
        
        total_tp += num_tp
        total_fp += num_fp
        total_fn += num_fn
        
        # Store per-image result
        per_image_results[image_path] = {
            "num_gt": len(gt_boxes),
            "num_generated": len(gen_boxes),
            "num_tp": num_tp,
            "num_fp": num_fp,
            "num_fn": num_fn,
            "matched_pairs": matching["matched_pairs"],
            "precision": num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0.0,
            "recall": num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0.0,
        }
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_image": per_image_results,
    }


def print_results(results, verbose=False):
    """Print evaluation results in a human-readable format."""
    print("\n" + "="*70)
    print("OBJECT DETECTION EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  True Positives (TP):  {results['total_tp']}")
    print(f"  False Positives (FP): {results['total_fp']}")
    print(f"  False Negatives (FN): {results['total_fn']}")
    print(f"\n  Precision: {results['precision']:.4f} (TP / (TP + FP))")
    print(f"  Recall:    {results['recall']:.4f} (TP / (TP + FN))")
    print(f"  F1-Score:  {results['f1']:.4f}")
    
    if verbose:
        print(f"\nPer-Image Breakdown:")
        print("-" * 70)
        for image_path, img_result in results['per_image'].items():
            print(f"\n  {Path(image_path).name}:")
            print(f"    GT boxes:       {img_result['num_gt']}")
            print(f"    Generated:      {img_result['num_generated']}")
            print(f"    Matched (TP):   {img_result['num_tp']}")
            print(f"    False Pos (FP): {img_result['num_fp']}")
            print(f"    False Neg (FN): {img_result['num_fn']}")
            print(f"    Precision:      {img_result['precision']:.4f}")
            print(f"    Recall:         {img_result['recall']:.4f}")
    
    print("\n" + "="*70)


def save_results(results, output_path):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary = {
        "total_tp": results['total_tp'],
        "total_fp": results['total_fp'],
        "total_fn": results['total_fn'],
        "precision": results['precision'],
        "recall": results['recall'],
        "f1": results['f1'],
    }
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = output_path / "detailed_results.json"
    with open(detailed_path, 'w') as f:
        # Convert per_image results to JSON-serializable format
        results_to_save = {
            "summary": summary,
            "per_image": {
                img_path: {
                    "num_gt": res['num_gt'],
                    "num_generated": res['num_generated'],
                    "num_tp": res['num_tp'],
                    "num_fp": res['num_fp'],
                    "num_fn": res['num_fn'],
                    "precision": res['precision'],
                    "recall": res['recall'],
                    "matched_pairs": res['matched_pairs'],
                }
                for img_path, res in results['per_image'].items()
            }
        }
        json.dump(results_to_save, f, indent=2)
    print(f"Detailed results saved to: {detailed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate object detection annotations against ground truth."
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to ground truth annotations JSON file"
    )
    parser.add_argument(
        "--generated_file",
        type=str,
        required=True,
        help="Path to AI-generated annotations JSON file"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.0,
        help="Filter generated annotations by score/confidence threshold (default: 0.0, no filtering)"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="Minimum IoU to consider a match between generated and GT box (default: 0.5)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation_results",
        help="Path to save detailed results (default: ./evaluation_results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image breakdown"
    )
    
    args = parser.parse_args()
    
    print(f"Loading ground truth annotations from: {args.gt_file}")
    gt_annotations = load_annotations(args.gt_file)
    print(f"  Loaded {len(gt_annotations)} images with {sum(len(v) for v in gt_annotations.values())} boxes")
    
    print(f"Loading generated annotations from: {args.generated_file}")
    generated_annotations = load_annotations(args.generated_file)
    print(f"  Loaded {len(generated_annotations)} images with {sum(len(v) for v in generated_annotations.values())} boxes")
    
    print(f"\nEvaluating with:")
    print(f"  Similarity threshold: {args.similarity_threshold}")
    print(f"  IoU threshold:        {args.iou_threshold}")
    
    # Evaluate
    results = evaluate(
        gt_annotations,
        generated_annotations,
        similarity_threshold=args.similarity_threshold,
        iou_threshold=args.iou_threshold,
        debug=args.verbose,
    )
    
    # Print results
    print_results(results, verbose=args.verbose)
    
    # Save results
    save_results(results, args.output_path)
