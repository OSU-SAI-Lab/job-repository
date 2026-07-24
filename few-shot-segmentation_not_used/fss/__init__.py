"""
fss — training-free few-shot segmentation: DINOv3 dense matching + SAM 3 refine.

Pipeline:  support(image+mask) --DINOv3 match--> prior map --> SAM prompts --> SAM 3 --> mask

All models are frozen (inference only). Public API:

    from fss import FewShotSegmenter
    seg = FewShotSegmenter(dinov3="base", sam3="facebook/sam3")
    seg.set_support(images=[...], masks=[...])
    result = seg.segment(query_image)   # {"mask","score","box","prior",...}
"""

from __future__ import annotations

from .config import (
    DINOV3_VARIANTS,
    FSSConfig,
    MatcherConfig,
    PriorMaskConfig,
    PromptConfig,
    SegmenterConfig,
)

__all__ = [
    "FewShotSegmenter",
    "DINOv3Matcher",
    "PromptGenerator",
    "SAM3Segmenter",
    "FSSConfig",
    "MatcherConfig",
    "PromptConfig",
    "SegmenterConfig",
    "PriorMaskConfig",
    "DINOV3_VARIANTS",
]

__version__ = "0.1.0"


# Lazy attribute access so `import fss` (and `fss --help`) does not eagerly import
# torch / transformers. Heavy components are pulled in only when referenced.
def __getattr__(name: str):
    if name == "FewShotSegmenter":
        from .pipeline import FewShotSegmenter
        return FewShotSegmenter
    if name == "DINOv3Matcher":
        from .matcher import DINOv3Matcher
        return DINOv3Matcher
    if name == "PromptGenerator":
        from .prompts import PromptGenerator
        return PromptGenerator
    if name == "SAM3Segmenter":
        from .segmenter import SAM3Segmenter
        return SAM3Segmenter
    raise AttributeError(f"module 'fss' has no attribute '{name}'")
