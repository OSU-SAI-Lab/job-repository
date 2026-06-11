"""
factory.py
==========
Factory pattern for creating the right trainer based on framework.

The TrainerFactory is the single point of entry for creating trainers.
It decides which trainer to instantiate based on the --framework argument
and returns it ready to use.

Supported frameworks:
    - yolo        : Ultralytics YOLO (detect, segment, classify)
    - huggingface : HuggingFace Transformers (classify, detect)

Usage:
    trainer = TrainerFactory.create("yolo", args)
    trainer = TrainerFactory.create("huggingface", args)
"""

from yolo_trainer import YOLOTrainer
from hf_trainer   import HFTrainer


# Registry of all supported frameworks
# Add new frameworks here without changing any other code
SUPPORTED_FRAMEWORKS = {
    "yolo"        : YOLOTrainer,
    "huggingface" : HFTrainer,
}


class TrainerFactory:
    """
    Factory class for creating trainer instances.

    Uses a registry pattern so adding a new framework only requires:
        1. Creating a new trainer class that inherits from BaseTrainer
        2. Adding it to SUPPORTED_FRAMEWORKS above

    No other code needs to change.
    """

    @staticmethod
    def create(framework: str, args):
        """
        Create and return the appropriate trainer for the given framework.

        Args:
            framework : str  — one of "yolo", "huggingface"
            args      : argparse.Namespace — parsed CLI arguments

        Returns:
            BaseTrainer subclass instance ready to train

        Raises:
            ValueError if framework is not supported
        """
        framework = framework.lower().strip()

        if framework not in SUPPORTED_FRAMEWORKS:
            supported = ", ".join(SUPPORTED_FRAMEWORKS.keys())
            raise ValueError(
                f"Unknown framework: '{framework}'. "
                f"Supported frameworks: {supported}"
            )

        trainer_class = SUPPORTED_FRAMEWORKS[framework]
        print(f"INFO: Creating {trainer_class.__name__} for framework '{framework}'")
        return trainer_class(args)

    @staticmethod
    def list_frameworks() -> list:
        """
        Return list of all supported framework names.

        Returns:
            list of supported framework strings
        """
        return list(SUPPORTED_FRAMEWORKS.keys())
