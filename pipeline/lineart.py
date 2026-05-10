"""Stage 1: anime image to binary line-art sketch extraction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_images(
    input_root: Path,
    output_root: Path,
    max_per_folder: int,
) -> list[tuple[Path, Path]]:
    """Collect source/destination image pairs for line-art extraction.

    The output path mirrors the source folder layout and always uses ``.png``.
    Existing outputs are skipped, so the command can be resumed safely.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    images_by_folder: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_folder[path.parent].append(path)

    pairs: list[tuple[Path, Path]] = []
    for folder in sorted(images_by_folder):
        for src in images_by_folder[folder][:max_per_folder]:
            relative_path = src.relative_to(input_root).with_suffix(".png")
            dst = output_root / relative_path
            if not dst.exists():
                pairs.append((src, dst))

    return pairs


def process_image(
    src: Path,
    dst: Path,
    detector: Any,
    detect_resolution: int,
    image_resolution: int,
) -> bool:
    """Run a ControlNet line-art detector on one source image."""
    try:
        with Image.open(src) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            detected = detector(
                image,
                detect_resolution=detect_resolution,
                image_resolution=image_resolution,
            )

        sketch = _to_binary_grayscale(detected)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sketch.save(dst)
        return True
    except Exception as exc:
        print(f"[skip] {src}: {exc}")
        return False


def _to_binary_grayscale(image_like: Any) -> Image.Image:
    image = _to_pil_image(image_like)
    gray = ImageOps.grayscale(image)
    return gray.point(lambda pixel: 0 if pixel < 128 else 255, mode="L")


def _to_pil_image(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like

    try:
        return Image.fromarray(image_like)
    except Exception as exc:
        image_type = type(image_like)
        raise TypeError(f"Detector returned unsupported image type: {image_type!r}") from exc
