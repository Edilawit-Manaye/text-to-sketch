"""Step 2 — Vectorization + RDP Simplification.

Converts a binary line-art image into a list of simplified vector strokes
using OpenCV contour detection followed by the Ramer-Douglas-Peucker (RDP)
algorithm (cv2.approxPolyDP).

Public API
----------
    vectorize_image(image_path, epsilon=0.5)
        -> list[list[tuple[int, int]]]
    vectorize_image_with_stats(image_path, epsilon=0.5)
        -> tuple[list[list[tuple[int, int]]], VectorizationStats]
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import cv2

DEFAULT_RDP_EPSILON = 0.5


@dataclass(frozen=True)
class VectorizationStats:
    """Point-count summary for one vectorized sketch."""

    epsilon: float
    raw_stroke_count: int
    raw_point_count: int
    simplified_stroke_count: int
    simplified_point_count: int

    @property
    def removed_point_count(self) -> int:
        return self.raw_point_count - self.simplified_point_count

    @property
    def point_retention_ratio(self) -> float:
        if self.raw_point_count == 0:
            return 0.0
        return self.simplified_point_count / self.raw_point_count


def vectorize_image(
    image_path: str | Path,
    epsilon: float = DEFAULT_RDP_EPSILON,
) -> list[list[tuple[int, int]]]:
    """Vectorize a lineart image into a list of simplified strokes.

    Parameters
    ----------
    image_path : str or Path
        Path to the input lineart (grayscale) image.
    epsilon : float
        Absolute RDP simplification tolerance in image pixels.
        Higher values produce simpler strokes. The project default is 0.5.

    Returns
    -------
    list[list[tuple[int, int]]]
        Each element is a stroke: an ordered list of (x, y) pixel coordinates.

    Raises
    ------
    ValueError
        If the image cannot be read.
    """
    strokes, _ = vectorize_image_with_stats(image_path, epsilon=epsilon)
    return strokes


def vectorize_image_with_stats(
    image_path: str | Path,
    epsilon: float = DEFAULT_RDP_EPSILON,
) -> tuple[list[list[tuple[int, int]]], VectorizationStats]:
    """Vectorize a sketch and return raw/simplified stroke point counts."""
    img = _read_grayscale_image(image_path)

    # Invert so drawn lines become white (255) for findContours.
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    strokes: list[list[tuple[int, int]]] = []
    raw_stroke_count = 0
    raw_point_count = 0

    for contour in contours:
        if len(contour) < 2:
            continue

        raw_stroke_count += 1
        raw_point_count += len(contour)

        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        points = [tuple(pt[0]) for pt in approx]
        if len(points) > 1:
            strokes.append(points)

    simplified_point_count = sum(len(stroke) for stroke in strokes)
    stats = VectorizationStats(
        epsilon=epsilon,
        raw_stroke_count=raw_stroke_count,
        raw_point_count=raw_point_count,
        simplified_stroke_count=len(strokes),
        simplified_point_count=simplified_point_count,
    )

    return strokes, stats


def _read_grayscale_image(image_path: str | Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    return img
