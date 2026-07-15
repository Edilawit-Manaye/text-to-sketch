"""Fixed-canvas qualitative reconstruction plots."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from metrics.sketchformer.free_running import CANVAS_MARGIN, CANVAS_SIZE
from metrics.sketchformer.reconstruction import ReconstructionExample

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _stroke_array_and_pen_lift(strokes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stroke deltas and pen-lift mask for stroke3 or stroke5 arrays."""

    array = np.asarray(strokes, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] not in {3, 5}:
        raise ValueError(
            "Expected stroke array with shape (N, 3) or (N, 5), "
            f"got {array.shape}"
        )
    if array.shape[1] == 5:
        end_mask = array[:, 4] >= 0.5
        pen_lift = np.asarray((array[:, 3] >= 0.5) | end_mask, dtype=bool)
        return array[:, :2], pen_lift

    pen_lift = np.asarray(array[:, 2] >= 0.5, dtype=bool)
    return array[:, :2], pen_lift


def stroke3_to_points(stroke3: np.ndarray) -> np.ndarray:
    """Convert relative stroke3 or stroke5 deltas into absolute xy points."""

    deltas, _ = _stroke_array_and_pen_lift(stroke3)
    return np.cumsum(deltas, axis=0)


def _stroke_segments(strokes: np.ndarray) -> list[np.ndarray]:
    deltas, pen_lift = _stroke_array_and_pen_lift(strokes)
    points = np.cumsum(deltas, axis=0)
    segments: list[np.ndarray] = []
    start = 0
    for index, lifted in enumerate(pen_lift):
        if lifted:
            # Single-point relocation segments are not visible pen strokes.
            if index + 1 - start >= 2:
                segments.append(points[start : index + 1])
            start = index + 1
    if len(points) - start >= 2:
        segments.append(points[start:])
    return segments


def _shared_transform(
    target_segments: list[np.ndarray],
    *,
    coordinate_mode: str,
) -> tuple[float, np.ndarray]:
    if coordinate_mode == "canvas":
        return 1.0, np.zeros(2, dtype=np.float32)
    if coordinate_mode != "target-normalized":
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode!r}")
    if not target_segments:
        return 1.0, np.zeros(2, dtype=np.float32)
    points = np.concatenate(target_segments, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    available = float(CANVAS_SIZE - 2 * CANVAS_MARGIN - 1)
    scale = available / max(float(np.max(maximum - minimum)), 1e-6)
    offset = np.full(2, CANVAS_MARGIN, dtype=np.float32) - minimum * scale
    return scale, offset


def _plot_segments(
    ax: plt.Axes,
    segments: list[np.ndarray],
    *,
    scale: float,
    offset: np.ndarray,
    title: str,
) -> None:
    if not segments:
        ax.text(
            CANVAS_SIZE / 2,
            CANVAS_SIZE / 2,
            "empty",
            ha="center",
            va="center",
        )
    else:
        transformed = [segment * scale + offset for segment in segments]
        for segment in transformed:
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                linewidth=0.8,
                color="#222222",
                alpha=0.85,
                clip_on=True,
            )
        first = transformed[0][0]
        last = transformed[-1][-1]
        ax.scatter(first[0], first[1], s=10, color="#2f80ed", zorder=3)
        ax.scatter(last[0], last[1], s=10, color="#eb5757", zorder=3)

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, CANVAS_SIZE - 1)
    ax.set_ylim(CANVAS_SIZE - 1, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save_reconstruction_pair(
    example: ReconstructionExample,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a target and prediction on identical fixed 256-pixel axes."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_segments = _stroke_segments(example.target)
    prediction_segments = _stroke_segments(example.prediction)
    scale, offset = _shared_transform(
        target_segments,
        coordinate_mode=example.coordinate_mode,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    if title:
        fig.suptitle(title, fontsize=9)
    _plot_segments(
        axes[0],
        target_segments,
        scale=scale,
        offset=offset,
        title=f"target | tokens={example.length}",
    )
    statistics = example.statistics
    prediction_title = (
        f"prediction | tokens={example.prediction_length or 0} "
        f"eos={statistics.get('eos_position', 'n/a')} "
        f"strokes={statistics.get('generated_stroke_count', 'n/a')} "
        f"repeat={statistics.get('longest_repeated_token_run', 'n/a')}"
    )
    _plot_segments(
        axes[1],
        prediction_segments,
        scale=scale,
        offset=offset,
        title=prediction_title,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_reconstruction_examples(
    examples: list[ReconstructionExample],
    output_dir: str | Path,
    *,
    prefix: str = "reconstruction",
) -> list[Path]:
    """Save mode-qualified plots so teacher/free runs cannot overwrite."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for index, example in enumerate(examples, start=1):
        statistics = example.statistics
        metric_text = ""
        if statistics:
            metric_text = (
                f" f1@2={float(statistics.get('geometry_f1_2px', 0.0)):.3f}"
                f" chamfer={float(statistics.get('symmetric_chamfer_px', 0.0)):.2f}px"
            )
        title = (
            f"sample={example.sample_id or example.source_index} "
            f"mode={example.decode_mode} label={example.label}{metric_text}"
        )
        mode = example.decode_mode.replace("_", "-")
        output_path = directory / f"{prefix}_{mode}_{index:03d}.png"
        saved.append(save_reconstruction_pair(example, output_path, title=title))
    return saved
