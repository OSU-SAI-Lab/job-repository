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
        - setup_wandb()      : initialize Weights & Biases run
        - save_metrics()     : save metrics.json to output directory
        - print_summary()    : print training summary
        - setup_output_dir() : create standardized output directory
    """

    def __init__(self, args):
        self.args       = args
        self.model      = None
        self.train_data = None
        self.val_data   = None
        self.metrics    = {}
        self.output_dir = None
        self.wandb_run  = None

    # ─────────────────────────────────────────────
    # Abstract methods
    # ─────────────────────────────────────────────

    @abstractmethod
    def download_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        pass

    # ─────────────────────────────────────────────
    # Shared utilities
    # ─────────────────────────────────────────────

    def setup_wandb(self):
        """
        Initialize Weights & Biases run if API key is provided.
        Logs all training hyperparameters as config.
        W&B will track metrics live during training.
        """
        wandb_key     = getattr(self.args, "wandb_key", None)
        wandb_project = getattr(self.args, "wandb_project", "workflow-orchestrator")
        wandb_name    = getattr(self.args, "name", None)

        if not wandb_key:
            print("INFO: No W&B API key provided. Skipping W&B tracking.")
            print("      Pass --wandb_key <your_key> to enable live tracking.")
            return

        try:
            import wandb
            wandb.login(key=wandb_key)

            self.wandb_run = wandb.init(
                project = wandb_project,
                name    = wandb_name,
                config  = vars(self.args),
            )
            print(f"INFO: W&B tracking enabled!")
            print(f"      Project : {wandb_project}")
            print(f"      Run     : {self.wandb_run.name}")
            print(f"      URL     : {self.wandb_run.url}")

        except Exception as e:
            print(f"WARNING: Could not initialize W&B: {e}")
            print(f"         Training will continue without W&B tracking.")
            self.wandb_run = None

    def finish_wandb(self, metrics: dict):
        """
        Log final metrics and close W&B run.
        """
        if self.wandb_run is None:
            return
        try:
            import wandb
            # Log final metrics
            perf = metrics.get("metrics", {})
            if perf:
                self.wandb_run.log(perf)
            # Save best model as artifact if it exists
            if self.output_dir:
                best_model = self.output_dir / "best_model"
                if best_model.exists():
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_dir(str(best_model))
                    self.wandb_run.log_artifact(artifact)
            self.wandb_run.finish()
            print(f"INFO: W&B run finished. View at: {self.wandb_run.url}")
        except Exception as e:
            print(f"WARNING: Could not finish W&B run: {e}")

    def setup_output_dir(self, task: str) -> Path:
        """Create standardized output directory."""
        if self.args.name is None:
            timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short     = Path(self.args.model).stem if Path(self.args.model).exists() else self.args.model.replace("/", "_")
            self.args.name  = f"{model_short}_{task}_{timestamp}"

        output_dir = Path(self.args.output_path) / task / self.args.name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        return output_dir

    def save_metrics(self, metrics: dict) -> Path:
        """Save metrics to standardized metrics.json file."""
        if self.output_dir is None:
            raise RuntimeError("setup_output_dir() must be called before save_metrics()")

        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, default=str)

        print(f"\nMetrics saved to: {metrics_path}")
        return metrics_path

    def print_summary(self, metrics: dict):
        """Print a clean training summary."""
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

        if self.wandb_run:
            print(f"\nW&B Dashboard    : {self.wandb_run.url}")

        print("=" * 50 + "\n")

    def run(self):
        """
        Run the full training pipeline in order.
        Calls all abstract methods in the correct sequence.
        """
        print(f"\nStarting training pipeline...")
        print(f"Framework : {self.__class__.__name__}")
        print(f"Model     : {self.args.model}\n")

        self.setup_wandb()
        self.download_model()
        self.load_data()
        self.train()
        metrics = self.get_metrics()
        self.save_metrics(metrics)
        self.print_summary(metrics)
        self.finish_wandb(metrics)
        return metrics
