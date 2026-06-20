"""
fss/viz.py — debugging / visualisation overlays.

The debug overlay (query + prior heatmap + sampled prompts + final mask) is the
key tool for tuning the prompt step, so it is a first-class utility here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .prompts import PromptSet


def _to_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return np.asarray(image)


def overlay_mask(
    image, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5
) -> np.ndarray:
    """Alpha-blend a binary mask over an RGB image; returns a uint8 array."""
    rgb = _to_rgb(image).astype("float32").copy()
    color_arr = np.array(color, dtype="float32")
    m = mask.astype(bool)
    rgb[m] = (1 - alpha) * rgb[m] + alpha * color_arr
    return np.clip(rgb, 0, 255).astype("uint8")


def save_debug_overlay(
    query_image,
    prior: Optional[np.ndarray],
    prompt_sets: Sequence[PromptSet],
    result: dict,
    out_path: str,
) -> str:
    """Save a 4-panel debug image: query | prior | prompts | final mask.

    Returns the output path. Requires matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rgb = _to_rgb(query_image)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("query")

    axes[1].imshow(rgb)
    if prior is not None:
        axes[1].imshow(prior, cmap="jet", alpha=0.55)
    axes[1].set_title("prior heatmap")

    axes[2].imshow(rgb)
    routed = any(ps.route is not None for ps in prompt_sets)
    axes[2].set_title("router decision" if routed else "sampled prompts")
    # When the router is active, colour each prompt set's box by regime and label
    # the component with its density metric so the routing call is inspectable.
    # dense → box path (cyan boxes), scattered → per-granule path (yellow boxes).
    route_color = {"dense": "cyan", "scattered": "yellow", None: "yellow"}
    seen_components = set()
    for ps in prompt_sets:
        for (x, y) in ps.positive_points:
            axes[2].plot(x, y, "o", color="lime", markersize=9, markeredgecolor="k")
        for (x, y) in ps.negative_points:
            axes[2].plot(x, y, "x", color="red", markersize=9)
        if ps.box is not None:
            x1, y1, x2, y2 = ps.box
            axes[2].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        edgecolor=route_color.get(ps.route, "yellow"),
                                        linewidth=2))
        # Label each routed component once, at its region centroid.
        if ps.route is not None and ps.density is not None and ps.region is not None:
            key = (round(ps.density["comp_area"], 1), ps.density["n_seeds"])
            if key not in seen_components:
                seen_components.add(key)
                ys, xs = np.where(ps.region)
                if len(xs):
                    cov = ps.density["seed_coverage"]
                    nn = ps.density["nn_distance_norm"]
                    nn_s = "inf" if not np.isfinite(nn) else f"{nn:.2f}"
                    axes[2].text(
                        float(xs.mean()), float(ys.min()) - 4,
                        f"{ps.route}\ncov={cov:.2f} nn={nn_s} n={ps.density['n_seeds']}",
                        color=route_color.get(ps.route, "yellow"), fontsize=7,
                        ha="center", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    axes[3].imshow(overlay_mask(rgb, result["mask"]))
    axes[3].set_title(f"final mask (score={result.get('score', 0.0):.3f})")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_mask(mask: np.ndarray, out_path: str) -> str:
    """Save a binary mask as an 8-bit PNG (0/255)."""
    Image.fromarray((mask.astype("uint8") * 255)).save(out_path)
    return out_path
