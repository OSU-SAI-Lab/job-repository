"""sam3_sahi_fss — SAHI-sliced SAM 3 proposals + DINOv3 prototype matching.

SAM 3 runs with a generic text concept ("visual") over overlapping slices to produce
every candidate box; DINOv3 embeds the support-labelled boxes into a prototype set,
and cosine similarity against that set selects the target-class proposals on the whole
dataset. Frozen models, no training. See pipeline.Sam3SahiDinoFSS.
"""

from .pipeline import Sam3SahiDinoFSS
from .prototypes import PrototypeSet

__all__ = ["Sam3SahiDinoFSS", "PrototypeSet"]
