"""Sketch token encoding helpers.

Encodes a stroke-5 array into a sequence of discrete token indices using a
K-means codebook built by ``prep_data.sketch_token.create_token_dict``.

Special tokens (appended after K regular motion tokens):
    K     →  pen-lift          (p2 == 1)
    K + 1 →  end-of-sketch     (p3 == 1)
"""

from __future__ import annotations

import numpy as np


def encode_stroke5(
    stroke5: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Map each stroke-5 row to a discrete token index.

    Pen-lift points (p2=1) emit **two** tokens: a codebook motion token
    preserving the (dx, dy) displacement, followed by a SEP token (K).
    This matches Sketchformer's tokenizer approach and prevents
    cumulative position drift at stroke boundaries.
    """
    K = len(codebook)
    tokens: list[int] = []

    for row in stroke5:
        p3 = row[4]
        p2 = row[3]

        if p3 == 1.0:
            tokens.append(K + 1)       # end-of-sketch
        else:
            # Quantize (dx, dy) for ALL points, including pen-lift.
            delta = codebook - row[:2]
            motion_token = int(np.argmin((delta * delta).sum(axis=1)))
            tokens.append(motion_token)

            if p2 == 1.0:
                tokens.append(K)       # SEP inserted after motion

    return np.array(tokens, dtype=np.int32)


def decode_tokens(
    tokens: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Approximate inverse of ``encode_stroke5``.

    Handles the two-token pen-lift encoding: a motion token followed by
    a SEP token (K) means the point has p2=1 (pen lifts after it).
    A standalone motion token has p1=1 (pen is drawing).
    """
    K = len(codebook)
    rows: list[list[float]] = []

    for tok in tokens:
        if tok == K + 1:
            rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
            break
        if tok == K:
            if rows:
                rows[-1][2] = 0.0
                rows[-1][3] = 1.0
        else:
            dx, dy = codebook[tok]
            rows.append([float(dx), float(dy), 1.0, 0.0, 0.0])

    if not rows or rows[-1][4] != 1.0:
        rows.append([0.0, 0.0, 0.0, 0.0, 1.0])

    return np.array(rows, dtype=np.float32)
