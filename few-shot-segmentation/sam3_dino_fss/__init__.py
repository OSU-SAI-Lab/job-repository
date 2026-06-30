"""sam3_dino_fss — proposal-first few-shot segmentation.

SAM 3 segments everything → DINOv3 embeds each proposal crop → cosine-match to a
DINOv3 class-support prototype selects the target-class proposals. Frozen models,
no training. See pipeline.SamDinoFSS.
"""

from .pipeline import SamDinoFSS

__all__ = ["SamDinoFSS"]
