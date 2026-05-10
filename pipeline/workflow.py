"""Core workflow for the Hand Simulation Pipeline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pipeline.kinematics import generate_kinematics
from pipeline.ordering import (
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_tsp,
)
from pipeline.stroke5 import to_stroke5
from pipeline.vectorization import (
    DEFAULT_RDP_EPSILON,
    vectorize_image,
)
from prep_data.sketch_token.create_token_dict import build_codebook, save_codebook
from utils.tokenizer import encode_stroke5
from utils.io import save_stroke5, save_token_sequence

_ORDER_FN_MAP = {
    "directional": order_directional_bias,
    "greedy": order_greedy_nearest_neighbor,
    "tsp": order_tsp,
}


def run_pipeline(
    sketches_dir: Path,
    stroke5_dir: Path,
    sketch_token_dir: Path,
    n_sketches: int,
    ordering: str,
    rdp_epsilon: float = DEFAULT_RDP_EPSILON,
    codebook_K: int = 1000,
    seed: int = 42,
) -> None:
    """Run vectorization, ordering, kinematics, stroke-5, and Tok-Dict encoding."""

    all_sketches = sorted(sketches_dir.rglob("*.png"))
    if not all_sketches:
        raise FileNotFoundError(
            f"No sketches found in {sketches_dir}. Run the filter-sketches "
            "command first to populate the filtered sketch directory."
        )
    if ordering not in _ORDER_FN_MAP:
        valid = ", ".join(sorted(_ORDER_FN_MAP))
        raise ValueError(f"Unknown ordering '{ordering}'. Expected one of: {valid}.")

    n = min(n_sketches, len(all_sketches))
    random.seed(seed)
    samples = random.sample(all_sketches, n)

    order_fn = _ORDER_FN_MAP[ordering]

    print(
        f"\n[pipeline] {n} sketches  ·  ordering: {ordering}"
        f"  ·  RDP epsilon={rdp_epsilon}  ·  K={codebook_K}"
    )
    print(f"[pipeline] stroke-5 output → {stroke5_dir}")
    print()

    stroke5_arrays: list[np.ndarray] = []
    ok = skipped = 0

    for img_path in tqdm(samples, desc="Steps B-E", unit="sketch"):
        try:
            # B: Vectorize
            strokes = vectorize_image(img_path, epsilon=rdp_epsilon)
            if not strokes:
                skipped += 1
                continue

            # C: Order
            ordered = order_fn(strokes)

            # D: Kinematics
            timed = generate_kinematics(ordered)
            if not timed:
                skipped += 1
                continue

            # E: stroke-5
            s5 = to_stroke5(timed)
            stroke5_arrays.append(s5)

            out_path = stroke5_dir / (img_path.stem + ".npz")
            save_stroke5(s5, out_path)
            ok += 1

        except Exception as exc:
            tqdm.write(f"  [skip] {img_path.name}: {exc}")
            skipped += 1

    # Summary
    print()
    print("─" * 58)
    print(f"  Stroke-5 conversion  : {ok:>5} OK   {skipped:>5} skipped")
    print(f"  Output directory     : {stroke5_dir}")
    print("─" * 58)

    if not stroke5_arrays:
        print("[sketch_token] No stroke-5 data — skipping codebook build.")
        return

    # Sketch token dictionary
    n_drawing_pts = int(
        sum(int((s5[:, 2] == 1.0).sum()) for s5 in stroke5_arrays)
    )
    print("\n[sketch_token] Building K-means codebook ...")
    print(f"[sketch_token]   sequences    : {len(stroke5_arrays)}")
    print(f"[sketch_token]   drawing pts  : {n_drawing_pts:,}")
    print(f"[sketch_token]   requested K  : {codebook_K}")

    codebook = build_codebook(stroke5_arrays, K=codebook_K)
    npy_path, meta_path = save_codebook(
        codebook, sketch_token_dir,
        K=len(codebook),
        n_samples=n_drawing_pts,
    )

    print(f"[sketch_token]   actual K     : {len(codebook)}")
    print(f"[sketch_token]   codebook     : {npy_path}")
    print(f"[sketch_token]   metadata     : {meta_path}")

    # Encoding Step
    tokens_dir = stroke5_dir.parent / "tokens"
    print(f"\n[encoder] Encoding {len(stroke5_arrays)} stroke-5 arrays to discrete tokens...")
    print(f"[encoder] Output directory: {tokens_dir}")

    encoded_count = 0
    for s5, img_path in zip(stroke5_arrays, samples):
        try:
            tokens = encode_stroke5(s5, codebook)
            out_path = tokens_dir / (img_path.stem + ".npz")
            save_token_sequence(tokens, out_path)
            encoded_count += 1
        except Exception as exc:
            print(f"  [skip] Failed to encode {img_path.name}: {exc}")

    print(f"[encoder] Successfully encoded {encoded_count} token sequences.")
