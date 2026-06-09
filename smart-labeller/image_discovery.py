"""
image_discovery.py — find images under a root directory (multi-level support).

Used by proposal and object_classification entry points.  Returns paths relative
to the chosen root so NPZ keys stay unique across subfolders (e.g.
``batch_a/img.jpg`` vs ``batch_b/img.jpg``).
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff"})


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def discover_images(root: Path) -> list[tuple[Path, str]]:
    """
    Recursively find image files under ``root``.

    Returns a sorted list of ``(absolute_path, relative_key)`` where
    ``relative_key`` is ``path.relative_to(root).as_posix()``.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    found: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if is_image_file(path):
            found.append((path, path.relative_to(root).as_posix()))
    return found


def path_key_map(images: list[tuple[Path, str]]) -> dict[str, str]:
    """Map resolved absolute path strings to relative keys."""
    return {str(p.resolve()): key for p, key in images}
