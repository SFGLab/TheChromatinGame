"""Scientific colormaps as 256-entry lookup tables (no matplotlib dependency).

`fall`     -- the white-to-black ramp used for raw Hi-C contact frequency.
`coolwarm` -- the diverging ramp used for the O/E correlation checkerboard.
"""
from __future__ import annotations

import numpy as np

_COOLWARM_ANCHORS = [
    (0.000, (58, 76, 192)),
    (0.125, (97, 128, 226)),
    (0.250, (140, 171, 244)),
    (0.375, (183, 207, 249)),
    (0.500, (221, 221, 221)),
    (0.625, (244, 197, 173)),
    (0.750, (244, 158, 125)),
    (0.875, (222, 108, 83)),
    (1.000, (180, 4, 38)),
]

_FALL_ANCHORS = [
    (0.000, (255, 255, 255)),
    (0.180, (255, 224, 158)),
    (0.380, (250, 158, 62)),
    (0.580, (214, 58, 40)),
    (0.790, (118, 20, 60)),
    (1.000, (18, 4, 28)),
]


def _build(anchors) -> np.ndarray:
    xs = np.array([a[0] for a in anchors])
    cols = np.array([a[1] for a in anchors], dtype=float)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(t, xs, cols[:, c]), 0, 255).astype(np.uint8)
    return lut


COOLWARM = _build(_COOLWARM_ANCHORS)
FALL = _build(_FALL_ANCHORS)


def apply(values: np.ndarray, lut: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a float array to an (H, W, 3) uint8 RGB image through `lut`."""
    if vmax - vmin < 1e-12:
        idx = np.zeros(values.shape, dtype=np.intp)
    else:
        norm = (values - vmin) / (vmax - vmin)
        idx = np.clip(norm * 255.0, 0, 255).astype(np.intp)
    return lut[idx]


def sample(lut: np.ndarray, t: float) -> tuple[int, int, int]:
    """Single colour lookup, t in 0..1."""
    i = int(np.clip(t, 0.0, 1.0) * 255)
    return tuple(int(v) for v in lut[i])
