"""
hf_trainer.py
=============
HuggingFace training backend for the Workflow Orchestrator.
Inherits from BaseTrainer and implements all abstract methods
using the HuggingFace Transformers library.

Supports:
    - Classification : ViT, Swin, DeiT (any AutoModelForImageClassification)
    - Detection      : DETR (any AutoModelForObjectDetection)
    - Any model available on HuggingFace Hub
    - Auto-download from HuggingFace Hub
    - Local COCO folder datasets
    - Single GPU and multi-GPU distributed training
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

from base_trainer import BaseTrainer


class HFTrainer(BaseTrainer):
    """
    HuggingFace training backend.
    Supports classification and detection using HuggingFace Transformers.
    """

    def __init__(self, args):
        super().__init__(args)
        self.framework  = "huggingface"
        self.processor  = None
        self.categories = None
        self.hf_trainer = None

    # ─────────────────────────────────────────────
    # Abstract method implementations
    # ─────────────────────────────────────────────

    def download_model(self):
        """
        Load model and image processor from HuggingFace Hub.
        Supports any model compatible with AutoImageProcessor.
        If model exists as a local path, loads directly.
        """
        from transformers import AutoImageProcessor

        print(f"INFO: Loading image processor for {self.args.model}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.args.model)
            print(f"INFO: Image size: {self.processor.size}")
        except Exception as e:
            print(f"ERROR: Could not load processor for '{self.args.model}': {e}")
            sys.exit(1)

        # Model is loaded in train() after we know num_classes from dataset
        print(f"INFO: Processor loaded. Model will be loaded after dataset is ready.")

    def load_data(self):
        """
        Load dataset based on task type.
        - Classification: folder with train/ and val/ subfolders per class
        - Detection: COCO format with images/ and annotations.json
        Supports both local paths and HuggingFace Hub dataset IDs.
        """
        if self.args.task == "classify":
            self.train_data, self.val_data, self.categories = self._load_classification_data()
        elif self.args.task == "detect":
            self.train_data, self.val_data, self.categories = self._load_detection_data()
        else:
            print(f"ERROR: Unsupported task '{self.args.task}' for HuggingFace trainer.")
            print(f"       Supported tasks: classify, detect")
            sys.exit(1)

        print(f"INFO: Train samples : {len(self.train_data)}")
        print(f"INFO: Val samples   : {len(self.val_data)}")
        print(f"INFO: Num classes   : {len(self.categories)}")

    def train(self):
        """
        Run HuggingFace training using the Trainer API.
        Automatically selects the right model class based on task.
        Supports single GPU and multi-GPU distributed training.
        """
        from transformers import (
            AutoModelForImageClassification,
            AutoModelForObjectDetection,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        import torch

        # Setup output directory
        output_dir = self.setup_output_dir(self.args.task)

        print(f"\nStarting HuggingFace Training...")
        print(f"Experiment : {self.args.name}")
        print(f"Saving to  : {output_dir}\n")

        # Distributed training info
        total_gpus     = self.args.nproc_per_node * self.args.num_nodes
        is_distributed = total_gpus > 1
        local_rank     = int(os.environ.get("LOCAL_RANK", 0))

        # Build label mappings
        num_classes = len(self.categories)
        id2label    = {i: name for i, name in enumerate(self.categories.values())}
        label2id    = {name: i for i, name in id2label.items()}

        # Load model based on task
        print(f"INFO: Loading model {self.args.model} for task: {self.args.task}...")
        if self.args.task == "classify":
            model = AutoModelForImageClassification.from_pretrained(
                self.args.model,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
        elif self.args.task == "detect":
            # DETR uses full 91 COCO classes to avoid index out of bounds
            num_classes = 91
            id2label    = {i: str(i) for i in range(91)}
            label2id    = {str(i): i for i in range(91)}
            model = AutoModelForObjectDetection.from_pretrained(
                self.args.model,
                num_labels=num_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )

        # Build TrainingArguments
        use_cuda = self.args.device == "cuda" and torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch,
            per_device_eval_batch_size=self.args.batch,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            optim=self.args.optimizer,
            lr_scheduler_type=self.args.lr_scheduler,
            warmup_steps=self.args.warmup_steps,
            eval_strategy=self.args.eval_strategy,
            save_strategy=self.args.save_strategy,
            save_total_limit=self.args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy" if self.args.task == "classify" else "eval_loss",
            logging_steps=self.args.logging_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            fp16=self.args.fp16 and use_cuda,
            seed=self.args.seed,
            dataloader_num_workers=0,
            report_to="none",
            ddp_backend=self.args.backend if is_distributed else None,
            local_rank=local_rank,
            remove_unused_columns=False,
        )

        # Build Trainer — use custom DetrTrainer for detection
        if self.args.task == "classify":
            from transformers import Trainer
            self.hf_trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.train_data,
                eval_dataset=self.val_data,
                compute_metrics=self._compute_classification_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.args.patience)],
            )
        elif self.args.task == "detect":
            self.hf_trainer = self._build_detr_trainer(
                model, training_args, self.args.patience
            )

        # Run training
        self.hf_trainer.train()
        # Save best model
        if local_rank == 0:
            best_model_path = output_dir / "best_model"
            self.hf_trainer.save_model(str(best_model_path))
            self.processor.save_pretrained(str(best_model_path))
            print(f"\nBest model saved to: {best_model_path}")

    def get_metrics(self) -> dict:
        """
        Extract and return standardized metrics from HuggingFace training.
        Returns accuracy for classification, eval_loss for detection.
        """
        perf_metrics = {}

        try:
            eval_results = self.hf_trainer.evaluate()
            for k, v in eval_results.items():
                if isinstance(v, float):
                    perf_metrics[k] = v
        except Exception as e:
            print(f"WARNING: Could not extract metrics: {e}")

        best_model_path = self.output_dir / "best_model"

        metrics = {
            "task"     : self.args.task,
            "framework": self.framework,
            "metrics"  : perf_metrics,
            "metadata" : {
                "model"      : self.args.model,
                "dataset"    : self.args.data,
                "output_path": str(self.output_dir),
                "epochs"     : self.args.epochs,
                "batch"      : self.args.batch,
                "lr"         : self.args.lr,
                "optimizer"  : self.args.optimizer,
                "fp16"       : self.args.fp16,
                "distributed": self.args.nproc_per_node * self.args.num_nodes > 1,
                "best_model" : str(best_model_path),
                "timestamp"  : datetime.now().isoformat(),
            }
        }

        self.metrics = metrics
        return metrics

    # ─────────────────────────────────────────────
    # Private helper methods
    # ─────────────────────────────────────────────

    def _load_classification_data(self):
        """Load classification dataset from folder structure."""
        import torch
        from torch.utils.data import Dataset
        from torchvision.datasets import ImageFolder

        class HFClassifyDataset(Dataset):
            def __init__(self, root, processor):
                self.dataset   = ImageFolder(root=root)
                self.processor = processor
                self.classes   = self.dataset.classes

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image, label = self.dataset[idx]
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = torch.tensor(label)
                return inputs

        data_path = Path(self.args.data)
        if not data_path.exists():
            print(f"ERROR: Dataset path not found: {self.args.data}")
            sys.exit(1)

        for split in ["train", "val"]:
            if not (data_path / split).exists():
                print(f"ERROR: {split}/ folder not found in {self.args.data}")
                sys.exit(1)

        train_ds = HFClassifyDataset(str(data_path / "train"), self.processor)
        val_ds   = HFClassifyDataset(str(data_path / "val"),   self.processor)
        classes  = {i: cls for i, cls in enumerate(train_ds.classes)}

        return train_ds, val_ds, classes

    def _load_detection_data(self):
        """Load detection dataset from local COCO folder or HuggingFace Hub."""
        data_path = Path(self.args.data)

        if data_path.exists():
            return self._load_local_coco(data_path)
        else:
            return self._load_hf_coco(self.args.data)

    def _load_local_coco(self, data_path: Path):
        """Load local COCO format dataset."""
        import torch
        from torch.utils.data import Dataset
        from PIL import Image

        class CocoDataset(Dataset):
            def __init__(self, images_dir, annotations_file, processor):
                with open(annotations_file) as f:
                    coco = json.load(f)
                self.processor   = processor
                self.images_dir  = Path(images_dir)
                self.images      = {img["id"]: img for img in coco["images"]}
                self.image_ids   = [img["id"] for img in coco["images"]]
                self.categories  = {cat["id"]: cat["name"] for cat in coco["categories"]}
                self.annotations = {}
                for ann in coco["annotations"]:
                    self.annotations.setdefault(ann["image_id"], []).append(ann)

            def __len__(self):
                return len(self.image_ids)

            def __getitem__(self, idx):
                image_id   = self.image_ids[idx]
                image_info = self.images[image_id]
                image      = Image.open(self.images_dir / image_info["file_name"]).convert("RGB")
                anns       = self.annotations.get(image_id, [])
                target     = {
                    "image_id": image_id,
                    "annotations": [{
                        "image_id"   : image_id,
                        "category_id": a["category_id"],
                        "bbox"       : a["bbox"],
                        "area"       : a.get("area", a["bbox"][2] * a["bbox"][3]),
                        "iscrowd"    : a.get("iscrowd", 0),
                    } for a in anns]
                }
                encoding = self.processor(images=image, annotations=target, return_tensors="pt")
                return {
                    "pixel_values": encoding["pixel_values"].squeeze(0),
                    "pixel_mask"  : encoding["pixel_mask"].squeeze(0),
                    "labels"      : encoding["labels"][0],
                }

        train_ds = CocoDataset(
            str(data_path / "train" / "images"),
            str(data_path / "train" / "annotations.json"),
            self.processor,
        )
        val_ds = CocoDataset(
            str(data_path / "val" / "images"),
            str(data_path / "val" / "annotations.json"),
            self.processor,
        )
        return train_ds, val_ds, train_ds.categories

    def _load_hf_coco(self, dataset_id: str):
        """Load COCO dataset from HuggingFace Hub."""
        from datasets import load_dataset
        from torch.utils.data import Dataset

        train_samples = getattr(self.args, "train_samples", 100)
        val_samples   = getattr(self.args, "val_samples", 20)

        print(f"INFO: Downloading '{dataset_id}' — train:{train_samples}, val:{val_samples}")
        hf_train = load_dataset(dataset_id, split=f"train[:{train_samples}]")
        hf_val   = load_dataset(dataset_id, split=f"val[:{val_samples}]")

        # Use full 91 COCO categories
        categories = {i: str(i) for i in range(91)}

        class HFCocoDataset(Dataset):
            def __init__(self, data, processor):
                self.data      = data
                self.processor = processor

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item    = self.data[idx]
                image   = item["image"].convert("RGB")
                img_id  = item["image_id"]
                objects = item["objects"]
                bboxes  = objects.get("bbox", []) if isinstance(objects, dict) else [o["bbox"] for o in objects]
                cats    = objects.get("category", []) if isinstance(objects, dict) else [o["category"] for o in objects]
                target  = {
                    "image_id": img_id,
                    "annotations": [{
                        "image_id"   : img_id,
                        "category_id": cat,
                        "bbox"       : [x, y, w, h],
                        "area"       : w * h,
                        "iscrowd"    : 0,
                    } for (x, y, w, h), cat in zip(bboxes, cats)]
                }
                encoding = self.processor(images=image, annotations=target, return_tensors="pt")
                return {
                    "pixel_values": encoding["pixel_values"].squeeze(0),
                    "pixel_mask"  : encoding["pixel_mask"].squeeze(0),
                    "labels"      : encoding["labels"][0],
                }

        return HFCocoDataset(hf_train, self.processor), HFCocoDataset(hf_val, self.processor), categories

    def _compute_classification_metrics(self, eval_pred):
        """Compute accuracy for classification evaluation."""
        import numpy as np
        logits, labels = eval_pred
        predictions    = np.argmax(logits, axis=-1)
        accuracy       = (predictions == labels).mean()
        return {"accuracy": float(accuracy)}

    def _detection_collate_fn(self, batch):
        """Collate function for DETR — pads images to same size."""
        import torch
        from torch.nn import functional as F

        pixel_values = [b["pixel_values"] for b in batch]
        max_h = max(p.shape[1] for p in pixel_values)
        max_w = max(p.shape[2] for p in pixel_values)

        padded_images = []
        padded_masks  = []
        for b in batch:
            p     = b["pixel_values"]
            pad_h = max_h - p.shape[1]
            pad_w = max_w - p.shape[2]
            padded_images.append(F.pad(p, (0, pad_w, 0, pad_h)))
            if "pixel_mask" in b:
                padded_masks.append(F.pad(b["pixel_mask"], (0, pad_w, 0, pad_h)))
            else:
                m = torch.zeros(max_h, max_w, dtype=torch.long)
                m[:p.shape[1], :p.shape[2]] = 1
                padded_masks.append(m)

        return {
            "pixel_values": torch.stack(padded_images),
            "pixel_mask"  : torch.stack(padded_masks),
            "labels"      : [b["labels"] for b in batch],
        }

    def _build_detr_trainer(self, model, training_args, patience):
        """Build custom DETR trainer that handles label device placement."""
        from transformers import Trainer, EarlyStoppingCallback
        import torch

        detection_collate_fn = self._detection_collate_fn

        class DetrTrainer(Trainer):
            def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
                labels = inputs.pop("labels")
                labels = [{k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                           for k, v in label.items()} for label in labels]
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    pixel_mask=inputs.get("pixel_mask"),
                    labels=labels,
                )
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

        return DetrTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            data_collator=detection_collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
