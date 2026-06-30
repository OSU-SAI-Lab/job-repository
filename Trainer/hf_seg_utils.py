"""
hf_seg_utils.py
===============
Segmentation dataset loader and utilities for HuggingFace SegFormer.

Dataset format expected:
    data/
        train/
            images/         - RGB images
            masks/          - Grayscale masks (pixel value = class ID)
        val/
            images/
            masks/
"""

import os
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation.
    Loads images and corresponding pixel-level mask images.
    
    Folder structure:
        root/
            images/   ← RGB images (.jpg, .png)
            masks/    ← Grayscale masks (.png) same filename as images
                         pixel value = class ID (0, 1, 2 ...)
    """

    def __init__(self, images_dir: str, masks_dir: str, processor, id2label: dict):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.processor  = processor
        self.id2label   = id2label

        # Match images with masks by filename (without extension)
        image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        self.samples = []
        for img_path in image_files:
            mask_path = self.masks_dir / (img_path.stem + ".png")
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                print(f"WARNING: No mask found for {img_path.name}, skipping.")

        print(f"INFO: Found {len(self.samples)} image-mask pairs in {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        mask  = Image.open(mask_path)


        # Process image and mask together
        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels"       : encoded["labels"].squeeze(0),
        }


def load_segmentation_datasets(data_path: str, processor):
    """
    Load train and val segmentation datasets from folder structure.
    
    Args:
        data_path : path to dataset root
        processor : SegformerImageProcessor
        
    Returns:
        train_dataset, val_dataset, id2label dict
    """
    data_path = Path(data_path)

    # Validate structure
    for split in ["train", "val"]:
        for subdir in ["images", "masks"]:
            if not (data_path / split / subdir).exists():
                print(f"ERROR: {split}/{subdir}/ not found in {data_path}")
                sys.exit(1)

    # Load label mapping if exists
    label_map_path = data_path / "id2label.json"
    if label_map_path.exists():
        with open(label_map_path) as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        print(f"INFO: Loaded {len(id2label)} classes from id2label.json")
    else:
        # Infer from mask pixel values
        print("INFO: No id2label.json found. Inferring classes from masks...")
        class_ids = set()
        for mask_path in (data_path / "train" / "masks").glob("*.png"):
            mask = np.array(Image.open(mask_path))
            class_ids.update(np.unique(mask).tolist())
        id2label = {int(i): str(i) for i in sorted(class_ids)}
        print(f"INFO: Found {len(id2label)} classes: {list(id2label.values())}")

    train_ds = SegmentationDataset(
        images_dir = str(data_path / "train" / "images"),
        masks_dir  = str(data_path / "train" / "masks"),
        processor  = processor,
        id2label   = id2label,
    )
    val_ds = SegmentationDataset(
        images_dir = str(data_path / "val" / "images"),
        masks_dir  = str(data_path / "val" / "masks"),
        processor  = processor,
        id2label   = id2label,
    )

    return train_ds, val_ds, id2label



def create_test_segmentation_dataset(output_dir: str, num_classes: int = 3, num_images: int = 10):
    """
    Create a small dummy segmentation dataset for testing.
    
    Args:
        output_dir  : where to create the dataset
        num_classes : number of segmentation classes
        num_images  : number of images per split
    """
    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            # Random RGB image
            img  = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            img.save(output_dir / split / "images" / f"{i:04d}.jpg")

            # Random mask with class IDs 0 to num_classes-1
            mask = Image.fromarray(
                np.random.randint(0, num_classes, (512, 512), dtype=np.uint8)
            )
            mask.save(output_dir / split / "masks" / f"{i:04d}.png")

    # Save label mapping
    id2label = {i: f"class_{i}" for i in range(num_classes)}
    with open(output_dir / "id2label.json", "w") as f:
        json.dump(id2label, f, indent=2)

    print(f"Test segmentation dataset created at: {output_dir}")
    print(f"  Classes    : {num_classes}")
    print(f"  Images/split: {num_images}")


def compute_segmentation_metrics(eval_pred):
    """
    Compute mean IoU for segmentation evaluation.
    
    IoU (Intersection over Union) per class:
        IoU = TP / (TP + FP + FN)
    
    Mean IoU = average IoU across all classes
    """
    import evaluate
    metric = evaluate.load("mean_iou")

    logits, labels = eval_pred
    # Upsample logits to label size
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = torch.nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics     = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=logits.shape[1],
        ignore_index=255,
        reduce_labels=False,
    )

    metrics = {
            k: v.tolist() if isinstance(v, np.ndarray)
            else round(float(v), 4) if isinstance(v, (float, np.floating))
            else v
            for k, v in metrics.items()
            }
    return metrics
