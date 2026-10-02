"""Estimate batch-size-1 runtime and choose how many equal image chunks to use.

Time for one job is ``startup + seconds_per_tile * tiles``. Tile count follows
OverlappingTileDataset for any tile size and overlap. Overlap changes the
count, not the rate. Seconds per tile is the Pass 1 sam3 + bioclip curve,
interpolated on tile size, times the Pass 2 ratio for the pair.

N is the smallest split whose heaviest chunk stays at or under 5 hours after a
2x margin. That margin is the spread seen when equal image counts still held
very different numbers of boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TARGET_SECONDS = 5 * 3600
CHUNK_SKEW = 2.0
REFERENCE_TILE = 1280
REFERENCE_OVERLAP = 0.2
SHARED_STARTUP = 60.0

# startup seconds, seconds per tile at tile 1280 / overlap 0.2 / batch 1.
# Fitted pairs use the N=1 wall time and the mean N=10 chunk time on 384 tiles.
# sam3+bioclip and owlv2+bioclip use the shared startup because they were not split.
PAIR_RATES: dict[tuple[str, str], tuple[float, float]] = {
    ("owlv2", "dinov3"): (54.1111, 0.843461),
    ("sam3", "dinov3"): (77.6667, 1.782118),
    ("owlv2", "owlv2"): (45.6667, 3.480035),
    ("sam3", "bioclip"): (SHARED_STARTUP, (885.0 - SHARED_STARTUP) / 384.0),
    ("owlv2", "bioclip"): (SHARED_STARTUP, (539.0 * (378.0 / 480.0) - SHARED_STARTUP) / 384.0),
}

# sam3 + bioclip seconds per tile at batch 1 and overlap 0.2.
# Overlap is not a separate axis: at a fixed tile size it barely changed this rate.
TILE_RATE_CURVE: tuple[tuple[int, float], ...] = (
    (384, 1.630133),
    (512, 1.685934),
    (640, 1.754195),
    (768, 1.811111),
    (1024, 1.945274),
    (1280, 2.148438),
)


@dataclass(frozen=True)
class SplitPlan:
    n: int
    startup: float
    seconds_per_tile: float
    chunk_tile_counts: tuple[int, ...]
    chunk_image_counts: tuple[int, ...]
    predicted_seconds: float


def tile_count(width: int, height: int, tile_size: int, overlap: float) -> int:
    """How many SAHI tiles one image produces. At least one."""
    if tile_size < 1:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    stride = int(tile_size * (1 - overlap))
    if stride < 1:
        raise ValueError(f"stride is {stride} for tile_size={tile_size} overlap={overlap}")
    cols = math.ceil((width - tile_size) / stride) + 1
    rows = math.ceil((height - tile_size) / stride) + 1
    return max(cols, 1) * max(rows, 1)


def interpolated_tile_rate(tile_size: int) -> float:
    """Sam3 + bioclip seconds per tile, linearly interpolated on tile size."""
    if tile_size < 1:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    curve = TILE_RATE_CURVE
    if tile_size <= curve[0][0] or tile_size >= curve[-1][0]:
        if tile_size <= curve[0][0]:
            (x0, y0), (x1, y1) = curve[0], curve[1]
            anchor_x, anchor_y = x0, y0
        else:
            (x0, y0), (x1, y1) = curve[-2], curve[-1]
            anchor_x, anchor_y = x1, y1
        slope = (y1 - y0) / (x1 - x0)
        return max(anchor_y + slope * (tile_size - anchor_x), 1e-6)
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= tile_size <= x1:
            weight = (tile_size - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    raise ValueError(f"No tile-rate segment for tile_size={tile_size}")


def pair_cost(proposer: str, embedder: str, tile_size: int, overlap: float) -> tuple[float, float]:
    """Return (startup seconds, seconds per tile) for one proposer/embedder pair.

    ``overlap`` is checked but does not change the rate. It changes tile count.
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    try:
        startup, rate = PAIR_RATES[(proposer, embedder)]
    except KeyError as exc:
        known = ", ".join(f"{p}+{e}" for p, e in sorted(PAIR_RATES))
        raise ValueError(f"No timing for {proposer}+{embedder}. Measured: {known}") from exc
    scale = interpolated_tile_rate(tile_size) / interpolated_tile_rate(REFERENCE_TILE)
    return startup, rate * scale


def combined_cost(
    pairs: list[tuple[str, str]],
    tile_size: int,
    overlap: float,
) -> tuple[float, float]:
    """Add pair costs. Each measured owlv2+dinov3 time already includes its extra embedder."""
    if not pairs:
        raise ValueError("At least one proposer/embedder pair is required")
    startup = 0.0
    rate = 0.0
    for proposer, embedder in pairs:
        pair_startup, pair_rate = pair_cost(proposer, embedder, tile_size, overlap)
        startup += pair_startup
        rate += pair_rate
    return startup, rate


def estimate_seconds(tile_counts: list[int], startup: float, seconds_per_tile: float) -> float:
    """Wall time for one job that runs these tiles, without the chunk-skew margin."""
    if startup < 0 or seconds_per_tile < 0:
        raise ValueError("startup and seconds_per_tile must be non-negative")
    return startup + seconds_per_tile * sum(tile_counts)


def equal_chunks(tile_counts: list[int], n: int) -> list[list[int]]:
    """Split a list into n pieces whose lengths differ by at most one."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if n > len(tile_counts):
        raise ValueError(f"Cannot split {len(tile_counts)} images into {n} chunks")
    base, extra = divmod(len(tile_counts), n)
    chunks: list[list[int]] = []
    start = 0
    for index in range(n):
        size = base + (1 if index < extra else 0)
        chunks.append(tile_counts[start : start + size])
        start += size
    return chunks


def choose_n(
    tile_counts: list[int],
    startup: float,
    seconds_per_tile: float,
    target_seconds: float = TARGET_SECONDS,
    skew: float = CHUNK_SKEW,
) -> SplitPlan:
    """Smallest equal image split whose heaviest chunk is predicted under the target.

    Prediction is ``startup + skew * seconds_per_tile * tiles_in_chunk``.
    """
    if not tile_counts:
        raise ValueError("No images to split")
    if skew < 1:
        raise ValueError(f"skew must be at least 1, got {skew}")
    if target_seconds <= 0:
        raise ValueError(f"target_seconds must be positive, got {target_seconds}")

    for index, tiles in enumerate(tile_counts):
        alone = startup + skew * seconds_per_tile * tiles
        if alone > target_seconds:
            raise ValueError(
                f"Image {index} estimates {alone:.0f}s, over the {target_seconds:.0f}s target. "
                "An image cannot be split."
            )

    for n in range(1, len(tile_counts) + 1):
        chunks = equal_chunks(tile_counts, n)
        heaviest = max(sum(chunk) for chunk in chunks)
        predicted = startup + skew * seconds_per_tile * heaviest
        if predicted <= target_seconds:
            return SplitPlan(
                n=n,
                startup=startup,
                seconds_per_tile=seconds_per_tile,
                chunk_tile_counts=tuple(sum(chunk) for chunk in chunks),
                chunk_image_counts=tuple(len(chunk) for chunk in chunks),
                predicted_seconds=predicted,
            )
    raise ValueError("No split fits the target")
