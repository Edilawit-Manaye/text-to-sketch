"""
Tok-Dict Encoder.

Encodes a stroke-5 array into a sequence of discrete token indices using a
K-means codebook built by ``pipeline.tokdict.builder``.

Special tokens (appended after K regular motion tokens):
    K     →  pen-lift          (p2 == 1)
    K + 1 →  end-of-sketch     (p3 == 1)
"""

from __future__ import annotations

import numpy as np


# Encoder

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
