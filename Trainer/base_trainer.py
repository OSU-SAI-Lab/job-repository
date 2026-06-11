"""
base_trainer.py
===============
Abstract base class for all trainers in the Workflow Orchestrator.
Every training backend (YOLO, HuggingFace, etc.) must inherit from
this class and implement all abstract methods.

This enforces a consistent interface across all training frameworks
so the orchestrator can call any trainer the same way.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
import json
import os


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    All training backends must implement:
        - download_model()   : pull model weights from registry
        - load_data()        : load and validate dataset
        - train()            : run training loop
        - get_metrics()      : return standardized performance metrics

    Shared utilities (available to all trainers):
        - save_metrics()     : save metrics.json to output directory
        - print_summary()    : print training summary
        - setup_output_dir() : create standardized output directory
    """

    def __init__(self, args):
        """
        Initialize trainer with parsed arguments.

        Args:
            args: argparse.Namespace with all training configuration
        """
        self.args       = args
        self.model      = None
        self.train_data = None
        self.val_data   = None
        self.metrics    = {}
        self.output_dir = None

    # ─────────────────────────────────────────────
    # Abstract methods — must be implemented by
    # every subclass (YOLO, HuggingFace, etc.)
    # ─────────────────────────────────────────────

    @abstractmethod
    def download_model(self):
        """
        Download or load model weights.
        
        - For YOLO: download from Ultralytics GitHub
        - For HuggingFace: load from HuggingFace Hub
        - For local weights: load from file path
        
        Must set self.model after loading.
        """
        pass

    @abstractmethod
    def load_data(self):
        """
        Load and validate the dataset.
        
        - For YOLO: validate .yaml file or dataset folder
        - For HuggingFace: load from local COCO folder or HuggingFace Hub
        
        Must set self.train_data and self.val_data after loading.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Run the training loop.
        
        - For YOLO: model.train(...)
        - For HuggingFace: trainer.train()
        
        Must update self.metrics after training.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Return standardized performance metrics.
        
        All trainers must return a dict with at least:
            {
                "task"     : "detect" | "segment" | "classify",
                "framework": "yolo"  | "huggingface",
                "metrics"  : { ... framework specific metrics ... },
                "metadata" : { model, dataset, epochs, timestamp, ... }
            }
        
        Returns:
            dict: standardized metrics dictionary
        """
        pass

    # ─────────────────────────────────────────────
    # Shared utilities — available to all trainers
    # ─────────────────────────────────────────────

    def setup_output_dir(self, task: str) -> Path:
        """
        Create standardized output directory.
        
        Structure: output_path / task / experiment_name /
        
        Args:
            task: "detect", "segment", or "classify"
            
        Returns:
            Path: output directory path
        """
        if self.args.name is None:
            timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short     = Path(self.args.model).stem if Path(self.args.model).exists() else self.args.model.replace("/", "_")
            self.args.name  = f"{model_short}_{task}_{timestamp}"

        output_dir = Path(self.args.output_path) / task / self.args.name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        return output_dir

    def save_metrics(self, metrics: dict) -> Path:
        """
        Save metrics to standardized metrics.json file.
        
        Args:
            metrics: dictionary of metrics to save
            
        Returns:
            Path: path to saved metrics.json
        """
        if self.output_dir is None:
            raise RuntimeError("setup_output_dir() must be called before save_metrics()")

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, default=str)

        print(f"\nMetrics saved to: {metrics_path}")
        return metrics_path

    def print_summary(self, metrics: dict):
        """
        Print a clean training summary.
        
        Args:
            metrics: dictionary of metrics to display
        """
        print("\n" + "=" * 50)
        print("TRAINING COMPLETE")
        print("=" * 50)
        print(f"\nOutput Directory : {self.output_dir}")
        print(f"Metrics File     : {self.output_dir / 'metrics.json'}")

        print("\nKey Metrics:")
        perf = metrics.get("metrics", {})
        if perf:
            for k, v in perf.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
        else:
            print("   (see metrics.json for full results)")

        print("=" * 50 + "\n")

    def run(self):
        """
        Run the full training pipeline in order.
        Calls all abstract methods in the correct sequence.
        """
        print(f"\nStarting training pipeline...")
        print(f"Framework : {self.__class__.__name__}")
        print(f"Model     : {self.args.model}\n")

        self.download_model()
        self.load_data()
        self.train()
        metrics = self.get_metrics()
        self.save_metrics(metrics)
        self.print_summary(metrics)
        return metrics
