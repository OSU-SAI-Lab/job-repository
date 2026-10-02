"""Decide whether a proposal run should be split before the model starts."""

from __future__ import annotations

from split_estimate import (
    SplitPlan,
    choose_n,
    combined_cost,
    tile_count,
)


def pairs_to_estimate(proposers: list[str], embedders: list[str]) -> list[tuple[str, str]]:
    """Pairs whose measured times should be added.

    An OWLv2 proposer that also embeds with DINOv3 or BioCLIP already includes
    the extra OWLv2 embedder in that measurement, so the extra file is not added
    again. SAM3 cannot use the OWLv2 embedder.
    """
    pairs: list[tuple[str, str]] = []
    for proposer in proposers:
        usable = [embedder for embedder in embedders if not (proposer == "sam3" and embedder == "owlv2")]
        if proposer == "owlv2" and "owlv2" not in usable:
            usable = usable + ["owlv2"]
        includes_other = proposer == "owlv2" and any(embedder != "owlv2" for embedder in usable)
        for embedder in usable:
            if includes_other and embedder == "owlv2":
                continue
            pairs.append((proposer, embedder))
    if not pairs:
        raise ValueError("No proposer/embedder pair left to estimate")
    return pairs


def plan_proposal_run(
    image_sizes: list[tuple[int, int]],
    proposers: list[str],
    embedders: list[str],
    tile_size: int,
    overlap: float,
    batch_size: int,
    is_sahi: bool,
    target_seconds: float | None = None,
) -> SplitPlan | None:
    """Return a split plan, or None when the run should stay one chunk.

    Splitting applies only at batch size 1 with SAHI on. None means the caller
    runs every image in the existing loop.
    """
    if batch_size != 1 or not is_sahi:
        return None
    if not image_sizes:
        raise ValueError("No images to estimate")
    startup, rate = combined_cost(
        pairs_to_estimate(proposers, embedders),
        tile_size,
        overlap,
    )
    counts = [tile_count(width, height, tile_size, overlap) for width, height in image_sizes]
    kwargs = {}
    if target_seconds is not None:
        kwargs["target_seconds"] = target_seconds
    plan = choose_n(counts, startup, rate, **kwargs)
    if plan.n == 1:
        return None
    return plan
