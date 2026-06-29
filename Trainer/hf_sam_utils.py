"""

hf_sam_utils.py

===============

SAM (Segment Anything Model) dataset loader and utilities.



Unlike standard semantic segmentation (SegFormer, BEiT, DPT), SAM is a

PROMPTABLE segmentation model — it needs an image AND a prompt (a point

or a bounding box) telling it WHERE to segment. It outputs a single mask

for whatever object the prompt points to, not a per-pixel class map.



Because of this, SAM does not fit the AutoModelForSemanticSegmentation

pattern used for SegFormer/BEiT/DPT, and needs its own dataset format,

collate function, and training loop logic.



Dataset format expected (same folder layout as standard segmentation):

    data/

        train/

            images/         - RGB images

            masks/          - binary masks (pixel value 0 or 255)

        val/

            images/

            masks/



For each image+mask pair, we auto-generate a prompt (a point sampled

from inside the mask) since SAM needs a prompt to know what to segment.

"""



import sys

from pathlib import Path

import numpy as np

from PIL import Image

import torch

from torch.utils.data import Dataset





def get_point_prompt_from_mask(mask: np.ndarray):

    """

    Pick a random point that lies INSIDE the foreground of the mask.

    This acts as the prompt telling SAM "segment the object at this point".



    Args:

        mask: 2D numpy array, foreground pixels are non-zero



    Returns:

        [x, y] coordinates of a point inside the foreground,

        or the image center if the mask is empty.

    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:

        h, w = mask.shape

        return [w // 2, h // 2]

    idx = np.random.randint(len(xs))

    return [int(xs[idx]), int(ys[idx])]





class SamDataset(Dataset):

    """

    Dataset for SAM fine-tuning.

    Loads an image + binary mask, and generates a point prompt

    sampled from inside the mask's foreground region.

    """



    def __init__(self, images_dir: str, masks_dir: str, processor):

        self.images_dir = Path(images_dir)

        self.masks_dir  = Path(masks_dir)

        self.processor  = processor



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

        mask  = np.array(Image.open(mask_path).convert("L"))

        # Binarize — anything non-zero counts as foreground

        binary_mask = (mask > 0).astype(np.float32)



        point = get_point_prompt_from_mask(mask)



        # SAM processor expects input_points shaped [batch, num_boxes, num_points, 2]

        inputs = self.processor(

            image,

            input_points=[[point]],

            return_tensors="pt",

        )

        # Drop the batch dim the processor adds — Trainer's collate will re-add it

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}



        # Ground truth mask resized to match what the model predicts

        inputs["ground_truth_mask"] = torch.from_numpy(binary_mask)



        return inputs





def load_sam_datasets(data_path: str, processor):

    """

    Load train and val SAM datasets from folder structure.



    Args:

        data_path : path to dataset root

        processor : SamProcessor



    Returns:

        train_dataset, val_dataset, categories (placeholder dict for

        compatibility with the rest of the trainer's category-counting code)

    """

    data_path = Path(data_path)



    for split in ["train", "val"]:

        for subdir in ["images", "masks"]:

            if not (data_path / split / subdir).exists():

                print(f"ERROR: {split}/{subdir}/ not found in {data_path}")

                sys.exit(1)



    train_ds = SamDataset(

        images_dir = str(data_path / "train" / "images"),

        masks_dir  = str(data_path / "train" / "masks"),

        processor  = processor,

    )

    val_ds = SamDataset(

        images_dir = str(data_path / "val" / "images"),

        masks_dir  = str(data_path / "val" / "masks"),

        processor  = processor,

    )



    # SAM is class-agnostic (foreground vs background only) — categories

    # dict kept just so the rest of the trainer's logging code still works

    categories = {0: "background", 1: "foreground"}



    return train_ds, val_ds, categories





def sam_collate_fn(batch):

    """

    Collate function for SAM batches.

    Stacks all tensors, including the variable-content input_points/input_labels

    that the SamProcessor produces.

    """

    keys = batch[0].keys()

    collated = {}

    for key in keys:

        collated[key] = torch.stack([item[key] for item in batch])

    return collated





def create_test_sam_dataset(output_dir: str, num_images: int = 10):

    """

    Create a small dummy SAM dataset for testing — random images with

    a random circular blob as the foreground mask.

    """

    output_dir = Path(output_dir)



    for split in ["train", "val"]:

        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)

        (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)



        for i in range(num_images):

            img = Image.fromarray(

                np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

            )

            img.save(output_dir / split / "images" / f"{i:04d}.jpg")



            # Random circular blob mask

            mask_arr = np.zeros((512, 512), dtype=np.uint8)

            cx, cy = np.random.randint(100, 412, size=2)

            radius = np.random.randint(40, 100)

            yy, xx = np.ogrid[:512, :512]

            blob = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

            mask_arr[blob] = 255



            mask = Image.fromarray(mask_arr)

            mask.save(output_dir / split / "masks" / f"{i:04d}.png")



    print(f"Test SAM dataset created at: {output_dir}")

    print(f"  Images/split: {num_images}")





def compute_dice_loss(pred_mask, gt_mask, smooth=1.0):

    """

    Dice loss — standard loss function for mask prediction tasks like SAM.

    Measures overlap between predicted and ground truth masks.

    """

    pred_mask = torch.sigmoid(pred_mask)

    pred_flat = pred_mask.reshape(-1)

    gt_flat   = gt_mask.reshape(-1)



    intersection = (pred_flat * gt_flat).sum()

    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)

    return 1.0 - dice





def compute_dice_score(pred_mask, gt_mask, threshold=0.5, smooth=1.0):

    """

    Dice score (= 1 - dice loss), reported as a metric rather than used

    for backprop. Ranges 0 (no overlap) to 1 (perfect overlap).

    """

    pred_prob = torch.sigmoid(pred_mask)

    pred_bin  = (pred_prob > threshold).float()



    pred_flat = pred_bin.reshape(-1)

    gt_flat   = gt_mask.reshape(-1)



    intersection = (pred_flat * gt_flat).sum()

    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)

    return dice.item()





def compute_iou_score(pred_mask, gt_mask, threshold=0.5, smooth=1.0):

    """

    Intersection over Union (IoU) — the standard metric for segmentation

    mask quality. Ranges 0 (no overlap) to 1 (perfect overlap).

    """

    pred_prob = torch.sigmoid(pred_mask)

    pred_bin  = (pred_prob > threshold).float()



    pred_flat = pred_bin.reshape(-1)

    gt_flat   = gt_mask.reshape(-1)



    intersection = (pred_flat * gt_flat).sum()

    union        = pred_flat.sum() + gt_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()
