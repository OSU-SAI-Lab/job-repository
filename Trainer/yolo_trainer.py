"""
yolo_trainer.py
===============
YOLO training backend for the Workflow Orchestrator.
Inherits from BaseTrainer and implements all abstract methods
using the Ultralytics YOLO library.

Supports:
    - All YOLO versions: v8, v9, v10, v11
    - Tasks: detect, segment, classify
    - Auto-download of model weights
    - Image size validation (nearest x32)
    - Checkpoint reuse via --model path
"""

import sys
from pathlib import Path
from datetime import datetime

from base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    """
    YOLO training backend.
    Supports detection, segmentation, and classification
    using the Ultralytics YOLO library.
    """

    def __init__(self, args):
        super().__init__(args)
        self.framework = "yolo"
        self.results   = None

    # ─────────────────────────────────────────────
    # Abstract method implementations
    # ─────────────────────────────────────────────

    def download_model(self):
        """
        Download or load YOLO model weights.
        Supports any Ultralytics model: v8, v9, v10, v11.
        If model exists locally, loads directly.
        Otherwise, auto-downloads from Ultralytics GitHub.
        """
        from ultralytics import YOLO

        model_path = Path(self.args.model)

        if model_path.exists():
            print(f"INFO: Using local weights: {self.args.model}")
        else:
            print(f"INFO: Model '{self.args.model}' not found locally. Attempting auto-download...")
            try:
                YOLO(self.args.model)
                print(f"INFO: Model '{self.args.model}' downloaded successfully.")
            except Exception as e:
                print(f"ERROR: Could not load or download model '{self.args.model}': {e}")
                sys.exit(1)

        self.model = YOLO(self.args.model)
        print(f"INFO: Model loaded: {self.args.model}")

    def load_data(self):
        """
        Validate dataset path and adjust image size.
        - Detects built-in Ultralytics datasets (auto-downloaded)
        - Validates local dataset paths
        - Adjusts imgsz to nearest multiple of 32
        """
        # Validate image size
        if self.args.imgsz % 32 != 0:
            adjusted       = round(self.args.imgsz / 32) * 32
            print(f"WARNING: imgsz={self.args.imgsz} is not a multiple of 32. Adjusting to {adjusted}.")
            self.args.imgsz = adjusted

        data_path = Path(self.args.data)

        if data_path.exists():
            print(f"INFO: Using local dataset: {self.args.data}")
        else:
            print(f"INFO: Dataset '{self.args.data}' not found locally.")
            print(f"      Ultralytics will attempt to auto-download it.")

        # Validate task vs dataset format
        if self.args.task in ["detect", "segment"]:
            if data_path.suffix not in [".yaml", ".yml"] and not str(self.args.data).endswith((".yaml", ".yml")):
                if data_path.exists() and not data_path.is_dir():
                    print(f"WARNING: For detection/segmentation, dataset should be a .yaml file.")

        if self.args.task == "classify" and str(self.args.data).endswith((".yaml", ".yml")):
            print(f"WARNING: For classification, dataset should be a folder, not a .yaml file.")

        self.train_data = self.args.data
        self.val_data   = self.args.data

    def train(self):
        """
        Run YOLO training using Ultralytics model.train().
        All hyperparameters are passed from args.
        Outputs saved under output_path/task/experiment/
        """
        # Setup output directory
        output_dir     = self.setup_output_dir(self.args.task)
        task_output    = str(Path(self.args.output_path) / self.args.task)

        print(f"\nStarting YOLO Training...")
        print(f"Experiment : {self.args.name}")
        print(f"Saving to  : {task_output}/{self.args.name}\n")

        # Build training arguments
        train_args = {
            "data"            : self.args.data,
            "epochs"          : self.args.epochs,
            "imgsz"           : self.args.imgsz,
            "batch"           : self.args.batch,
            "lr0"             : self.args.lr0,
            "lrf"             : self.args.lrf,
            "momentum"        : self.args.momentum,
            "weight_decay"    : self.args.weight_decay,
            "warmup_epochs"   : self.args.warmup_epochs,
            "warmup_momentum" : self.args.warmup_momentum,
            "warmup_bias_lr"  : self.args.warmup_bias_lr,
            "optimizer"       : self.args.optimizer,
            "cos_lr"          : self.args.cos_lr,
            "dropout"         : self.args.dropout,
            "seed"            : self.args.seed,
            "fraction"        : self.args.fraction,
            "freeze"          : self.args.freeze,
            "patience"        : self.args.patience,
            "save_period"     : self.args.save_period,
            "close_mosaic"    : self.args.close_mosaic,
            "amp"             : self.args.amp,
            "resume"          : self.args.resume,
            "profile"         : self.args.profile,
            "workers"         : self.args.workers,
            "task"            : self.args.task,
            "device"          : self.args.device,
            "project"         : task_output,
            "name"            : self.args.name,
        }

        # Enable W&B if API key provided
        wandb_key = getattr(self.args, 'wandb_key', None)
        if wandb_key:
            try:
                import wandb
                wandb.login(key=wandb_key)
                train_args['project'] = getattr(self.args, 'wandb_project', 'workflow-orchestrator')
                print(f"INFO: W&B enabled for YOLO training")
            except Exception as e:
                print(f"WARNING: Could not enable W&B for YOLO: {e}")

        self.results = self.model.train(**train_args)

        # Resolve actual output directory
        if hasattr(self.results, "save_dir"):
            self.output_dir = Path(self.results.save_dir)

    def get_metrics(self) -> dict:
        """
        Extract and return standardized metrics from YOLO training results.
        Returns mAP50, mAP50-95 for detection/segmentation.
        Returns top1/top5 accuracy for classification.
        """
        perf_metrics = {}

        try:
            if hasattr(self.results, "results_dict"):
                raw = self.results.results_dict
                # Detection / Segmentation
                for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                            "metrics/mAP50(M)", "metrics/mAP50-95(M)"]:
                    if key in raw:
                        perf_metrics[key] = float(raw[key])
                # Classification
                for key in ["metrics/accuracy_top1", "metrics/accuracy_top5"]:
                    if key in raw:
                        perf_metrics[key] = float(raw[key])

        except Exception as e:
            print(f"WARNING: Could not extract metrics: {e}")

        best_weights = self.output_dir / "weights" / "best.pt"
        last_weights = self.output_dir / "weights" / "last.pt"

        metrics = {
            "task"     : self.args.task,
            "framework": self.framework,
            "metrics"  : perf_metrics,
            "metadata" : {
                "model"       : self.args.model,
                "dataset"     : self.args.data,
                "output_path" : str(self.output_dir),
                "epochs"      : self.args.epochs,
                "imgsz"       : self.args.imgsz,
                "batch"       : self.args.batch,
                "lr0"         : self.args.lr0,
                "device"      : self.args.device,
                "optimizer"   : self.args.optimizer,
                "best_weights": str(best_weights) if best_weights.exists() else "not found",
                "last_weights": str(last_weights) if last_weights.exists() else "not found",
                "timestamp"   : datetime.now().isoformat(),
            }
        }

        self.metrics = metrics
        return metrics
