from __future__ import annotations

import numpy as np

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
            # End-of-sketch sentinel
            rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
            break
        elif tok == K:
            # SEP: retroactively mark previous point as pen-lift
            if rows:
                rows[-1][2] = 0.0      # p1 = 0
                rows[-1][3] = 1.0      # p2 = 1
        else:
            # Motion token: default to drawing (p1=1)
            dx, dy = codebook[tok]
            rows.append([float(dx), float(dy), 1.0, 0.0, 0.0])

    # Ensure sentinel exists
    if not rows or rows[-1][4] != 1.0:
        rows.append([0.0, 0.0, 0.0, 0.0, 1.0])

    return np.array(rows, dtype=np.float32)