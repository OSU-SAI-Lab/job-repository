"""
fss/matcher.py — DINOv3 dense correspondence (the "match-first" stage).

Given support (image, binary mask) pairs we build a foreground prototype (and an
optional background prototype) from DINOv3 dense patch features, then score every
patch of a query image by cosine similarity to produce a coarse PRIOR MAP that is
upsampled to full query resolution.

Patch-token contract
---------------------
DINOv3 returns ``last_hidden_state`` of shape ``(B, 1 + R + P, D)`` where the first
token is the [CLS] token, the next ``R = config.num_register_tokens`` are register
tokens, and the remaining ``P`` are the spatial patch tokens. ONLY the patch tokens
form the (H/p, W/p) grid — the CLS and register tokens are stripped before reshape.
Getting this wrong silently corrupts the similarity map, so we assert the count.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import MatcherConfig, resolve_dinov3


@dataclass
class DenseFeatures:
    """L2-normalised dense patch features for one image."""

    grid: torch.Tensor          # (Hp, Wp, D) on CPU, float32, L2-normalised
    hp: int
    wp: int

    @property
    def flat(self) -> torch.Tensor:
        """(Hp*Wp, D) view."""
        return self.grid.reshape(self.hp * self.wp, -1)


@dataclass
class SupportPrototypes:
    """Cached prototypes (and raw fg patches for 4D mode) for a support set."""

    fg: torch.Tensor                       # (D,) L2-normalised
    bg: Optional[torch.Tensor]             # (D,) L2-normalised or None
    fg_patches: Optional[torch.Tensor]     # (M, D) all support fg patches (4D mode)


class DINOv3Matcher:
    """Frozen DINOv3 backbone used purely for dense feature matching."""

    def __init__(self, config: MatcherConfig, device: str,
                 amp_dtype: Optional[torch.dtype] = None):
        from transformers import AutoImageProcessor, AutoModel  # local import

        self.config = config
        self.device = device
        # Weights stay fp32; half precision (if any) is applied via autocast on the
        # forward pass only — avoids dtype-mismatch errors and keeps cosine stable.
        self.amp_dtype = amp_dtype
        model_id = resolve_dinov3(config.dinov3)

        # DINOv3 uses its OWN preprocessing — never share transforms with SAM 3.
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

        self.patch_size: int = int(self.model.config.patch_size)
        self.num_register_tokens: int = int(getattr(self.model.config, "num_register_tokens", 0))
        self.dim: int = int(self.model.config.hidden_size)

    def _autocast(self):
        if self.amp_dtype is not None and self.device == "cuda":
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    # ──────────────────────────────────────────────────────────────────────
    # Dense feature extraction
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_dense_features(self, image: Image.Image) -> DenseFeatures:
        """Run DINOv3 and return L2-normalised patch features on the (Hp, Wp) grid."""
        pv = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        pv = pv.to(self.device)

        with self._autocast():
            out = self.model(pixel_values=pv)
        last_hidden = out.last_hidden_state            # (1, 1 + R + P, D)

        # Strip CLS + register tokens — only patch tokens are spatial.
        n_skip = 1 + self.num_register_tokens
        patch_tokens = last_hidden[:, n_skip:, :]      # (1, P, D)

        _, _, h_px, w_px = pv.shape
        hp, wp = h_px // self.patch_size, w_px // self.patch_size
        n_patches = patch_tokens.shape[1]
        if hp * wp != n_patches:
            raise RuntimeError(
                f"Patch-grid mismatch: pixel grid {hp}x{wp}={hp * wp} tokens but model "
                f"returned {n_patches} patch tokens (after stripping 1 CLS + "
                f"{self.num_register_tokens} register tokens). Check num_register_tokens."
            )

        grid = patch_tokens.reshape(hp, wp, self.dim).float()
        grid = F.normalize(grid, dim=-1)               # L2-normalise per patch
        return DenseFeatures(grid=grid.cpu(), hp=hp, wp=wp)

    # ──────────────────────────────────────────────────────────────────────
    # Prototype construction
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _mask_to_grid(mask: np.ndarray, hp: int, wp: int) -> torch.Tensor:
        """Resize a binary HxW mask to the (Hp, Wp) patch grid as float coverage."""
        m = torch.from_numpy(mask.astype("float32"))[None, None]    # (1,1,H,W)
        m = F.interpolate(m, size=(hp, wp), mode="bilinear", align_corners=False)
        return m[0, 0]                                              # (Hp, Wp) in [0,1]

    def build_prototypes(
        self,
        images: Sequence[Image.Image],
        masks: Sequence[np.ndarray],
    ) -> SupportPrototypes:
        """Masked-average-pool foreground (and complement) patches across K shots."""
        fg_vecs: List[torch.Tensor] = []
        bg_vecs: List[torch.Tensor] = []
        all_fg_patches: List[torch.Tensor] = []

        for image, mask in zip(images, masks):
            feats = self.extract_dense_features(image)
            grid = feats.grid                           # (Hp, Wp, D)
            cov = self._mask_to_grid(mask, feats.hp, feats.wp)      # (Hp, Wp)
            fg = cov >= 0.5

            if fg.sum() == 0:
                # Mask is smaller than one patch — keep the single best-covered patch.
                fg = torch.zeros_like(cov, dtype=torch.bool)
                idx = torch.argmax(cov)
                fg.view(-1)[idx] = True

            fg_patches = grid[fg]                        # (m, D)
            all_fg_patches.append(fg_patches)
            fg_vecs.append(F.normalize(fg_patches.mean(0), dim=-1))

            if self.config.use_bg_prototype:
                bg = ~fg
                if bg.sum() > 0:
                    bg_vecs.append(F.normalize(grid[bg].mean(0), dim=-1))

        # K-shot: average prototypes across support items, then renormalise.
        fg_proto = F.normalize(torch.stack(fg_vecs, 0).mean(0), dim=-1)
        bg_proto = (
            F.normalize(torch.stack(bg_vecs, 0).mean(0), dim=-1)
            if bg_vecs else None
        )
        fg_patches = (
            torch.cat(all_fg_patches, 0) if self.config.use_4d_correlation else None
        )
        return SupportPrototypes(fg=fg_proto, bg=bg_proto, fg_patches=fg_patches)

    # ──────────────────────────────────────────────────────────────────────
    # Prior-map computation
    # ──────────────────────────────────────────────────────────────────────

    def compute_prior(
        self,
        query: Image.Image,
        protos: SupportPrototypes,
    ) -> np.ndarray:
        """Return a prior map upsampled to full query resolution (H, W) float32."""
        feats = self.extract_dense_features(query)

        if self.config.use_4d_correlation and protos.fg_patches is not None:
            sim = self._prior_4d(feats, protos.fg_patches)         # (Hp, Wp)
        else:
            sim = self._prior_prototype(feats, protos)             # (Hp, Wp)

        # Bilinearly upsample the coarse grid to the original query resolution.
        w, h = query.size
        prior = F.interpolate(
            sim[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        prior_np = prior.numpy()

        if self.config.normalize_prior:
            lo, hi = float(prior_np.min()), float(prior_np.max())
            prior_np = (prior_np - lo) / (hi - lo + 1e-8)
        return prior_np.astype("float32")

    def _prior_prototype(
        self, feats: DenseFeatures, protos: SupportPrototypes
    ) -> torch.Tensor:
        flat = feats.flat                                # (N, D)
        sim_fg = flat @ protos.fg                        # (N,)
        if protos.bg is not None:
            sim_bg = flat @ protos.bg
            if self.config.bg_mode == "softmax":
                stacked = torch.stack([sim_fg, sim_bg], dim=-1)   # (N, 2)
                sim = torch.softmax(stacked, dim=-1)[:, 0]
            else:  # subtract
                sim = sim_fg - sim_bg
        else:
            sim = sim_fg
        return sim.reshape(feats.hp, feats.wp)

    def _prior_4d(self, feats: DenseFeatures, fg_patches: torch.Tensor) -> torch.Tensor:
        """Dense 4D correlation: every query patch vs every support-fg patch.

        Tradeoff: captures intra-object appearance variation better than a single
        averaged prototype, at O(N_query * M_support) memory/compute. Use for hard
        cases; the prototype path is the cheaper default.
        """
        flat = feats.flat                                # (N, D)
        corr = flat @ fg_patches.t()                     # (N, M)
        k = min(self.config.corr_topk, corr.shape[1])
        topk = corr.topk(k, dim=1).values                # (N, k)
        sim = topk.mean(1)                               # (N,)
        return sim.reshape(feats.hp, feats.wp)
