"""
Evaluate Tok-Dict Encoder/Decoder.

This script encodes stroke-5 sketches to tokens, decodes them back to stroke-5, and
measures reconstruction quality in a quantitative, repeatable way.

It can evaluate a single sketch, a random sketch, or an entire stroke-5 corpus.
Reconstructed stroke-5 arrays and token sequences are saved alongside summary metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.tokdict.encoder import encode_stroke5
from pipeline.tokdict.decoder import decode_tokens
from pipeline.utils.io import (
    load_codebook,
    load_stroke5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantitatively evaluate Tok-Dict encoding and reconstruct stroke-5 sketches."
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "stroke5"),
        help="Directory containing input stroke-5 .npz files.",
    )
    parser.add_argument(
        "--sketch",
        default=None,
        help="Optional specific stroke-5 .npz file to evaluate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all stroke-5 files in the input directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "tokdict" / "reconstructions"),
        help="Directory where reconstructed stroke-5, tokens, and metrics are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
        help="Random seed for selecting a sketch when not using --all or --sketch.",
    )
    return parser.parse_args()


def collect_sketch_paths(input_dir: Path, sketch_path: Path | None, use_all: bool) -> list[Path]:
    if sketch_path is not None:
        if not sketch_path.exists() or not sketch_path.is_file():
            raise FileNotFoundError(f"Sketch file not found: {sketch_path}")
        return [sketch_path.resolve()]

    sketches = sorted(input_dir.rglob("*.npz"))
    if not sketches:
        raise FileNotFoundError(f"No stroke-5 files found in {input_dir}")

    if use_all:
        return sketches

    random.shuffle(sketches)
    return [sketches[0]]


def compute_reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float | int | bool]:
    pen_states_match = np.array_equal(original[:, 2:], reconstructed[:, 2:])

    drawing_mask = original[:, 2] == 1.0
    drawing_count = int(drawing_mask.sum())
    motion_error = np.zeros((0, 2), dtype=np.float32)

    if drawing_count > 0:
        orig_motion = original[drawing_mask, :2]
        recon_motion = reconstructed[drawing_mask, :2]
        motion_error = np.abs(orig_motion - recon_motion)

    mae_dx = float(np.mean(motion_error[:, 0])) if drawing_count > 0 else 0.0
    mae_dy = float(np.mean(motion_error[:, 1])) if drawing_count > 0 else 0.0
    max_dx = float(np.max(motion_error[:, 0])) if drawing_count > 0 else 0.0
    max_dy = float(np.max(motion_error[:, 1])) if drawing_count > 0 else 0.0
    mean_euclid = float(np.mean(np.linalg.norm(motion_error, axis=1))) if drawing_count > 0 else 0.0
    max_euclid = float(np.max(np.linalg.norm(motion_error, axis=1))) if drawing_count > 0 else 0.0

    return {
        "pen_states_match": pen_states_match,
        "drawing_points": drawing_count,
        "mae_dx": mae_dx,
        "mae_dy": mae_dy,
        "max_dx": max_dx,
        "max_dy": max_dy,
        "mean_euclidean_error": mean_euclid,
        "max_euclidean_error": max_euclid,
    }


def format_summary(metrics: dict[str, float | int | bool], total_points: int) -> str:
    return (
        f"  Total sequence length : {total_points} points\n"
        f"  Drawing points (p1=1) : {metrics['drawing_points']} points\n"
        f"  Pen states preserved  : {'✅ YES' if metrics['pen_states_match'] else '❌ NO'}\n"
        f"  Mean abs motion error : dx={metrics['mae_dx']:.4f}, dy={metrics['mae_dy']:.4f}\n"
        f"  Max abs motion error  : dx={metrics['max_dx']:.4f}, dy={metrics['max_dy']:.4f}\n"
        f"  Mean euclidean error  : {metrics['mean_euclidean_error']:.4f}\n"
        f"  Max euclidean error   : {metrics['max_euclidean_error']:.4f}\n"
    )


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    fieldnames = [
        "sketch",
        "relative_path",
        "n_points",
        "drawing_points",
        "tokens_count",
        "pen_states_match",
        "mae_dx",
        "mae_dy",
        "max_dx",
        "max_dy",
        "mean_euclidean_error",
        "max_euclidean_error",
    ]

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    metrics_path = output_root / "metrics.csv"

    codebook_path = PROJECT_ROOT / "data" / "processed" / "tokdict" / "codebook.npy"
    if not codebook_path.exists():
        print(f"Error: Codebook not found at {codebook_path}")
        print("Please run scripts/run_pipeline.py first to build the codebook.")
        sys.exit(1)

    if args.sketch is None and not args.all:
        random.seed(args.seed)

    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        sys.exit(1)

    try:
        sketch_paths = collect_sketch_paths(
            input_root,
            Path(args.sketch).resolve() if args.sketch else None,
            args.all,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    codebook = load_codebook(codebook_path)
    output_root.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, str | int | float | bool]] = []

    print("\n" + "=" * 68)
    print("          TOK-DICT ENCODER EVALUATION / RECONSTRUCTION")
    print("=" * 68)
    print(f"Using codebook: {codebook_path.name} ({len(codebook)} centroids)")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Sketches to evaluate: {len(sketch_paths)}")
    print("=" * 68 + "\n")

    for sketch_path in sketch_paths:
        relative_path = sketch_path.relative_to(input_root) if sketch_path.is_relative_to(input_root) else Path(sketch_path.name)
        print(f"Evaluating: {relative_path}")

        original_s5 = load_stroke5(sketch_path)
        tokens = encode_stroke5(original_s5, codebook)
        reconstructed_s5 = decode_tokens(tokens, codebook)

        metrics = compute_reconstruction_metrics(original_s5, reconstructed_s5)
        total_points = len(original_s5)

        metrics_rows.append(
            {
                "sketch": sketch_path.name,
                "relative_path": str(relative_path),
                "n_points": total_points,
                "drawing_points": metrics["drawing_points"],
                "tokens_count": len(tokens),
                "pen_states_match": metrics["pen_states_match"],
                "mae_dx": metrics["mae_dx"],
                "mae_dy": metrics["mae_dy"],
                "max_dx": metrics["max_dx"],
                "max_dy": metrics["max_dy"],
                "mean_euclidean_error": metrics["mean_euclidean_error"],
                "max_euclidean_error": metrics["max_euclidean_error"],
            }
        )

        print(format_summary(metrics, total_points))
        print("-" * 68)

    write_metrics_csv(metrics_path, metrics_rows)

    if metrics_rows:
        print(f"Reconstruction complete. Metrics saved to: {metrics_path}")
    else:
        print("No sketches were evaluated.")


if __name__ == "__main__":
    main()
