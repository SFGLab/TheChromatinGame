"""Cheap structural metrics computed live from a Polymer conformation.

All functions take a Polymer and return a float. They are designed to be
called every few frames without measurable frame-rate impact at N <= 100.
"""
from __future__ import annotations
import numpy as np
from .physics import Polymer


def radius_of_gyration(poly: Polymer) -> float:
    """RMS distance of beads from the centroid."""
    c = poly.pos - poly.pos.mean(axis=0)
    return float(np.sqrt((c ** 2).sum(axis=1).mean()))


def end_to_end(poly: Polymer) -> float:
    """Euclidean distance between the first and last bead."""
    return float(np.linalg.norm(poly.pos[-1] - poly.pos[0]))


def mean_bond_length(poly: Polymer) -> float:
    """Average backbone bond length (should stay close to b0)."""
    d = np.linalg.norm(poly.pos[1:] - poly.pos[:-1], axis=1)
    return float(d.mean())


def contact_count(poly: Polymer) -> float:
    """Number of non-bonded bead pairs with contact probability > 0.5."""
    C = poly.contacts()
    n = poly.n
    # Upper triangle excluding diagonal and bonded neighbours (|i-j| >= 2)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 2, n):
            mask[i, j] = True
    return float((C[mask] > 0.5).sum())

def asphericity(poly: Polymer) -> float:
    """Asphericity from the gyration tensor eigenvalues.

    0 = perfect sphere, 1 = perfect rod.
    Cheap: 3×3 eigendecomposition regardless of N.
    """
    c = poly.pos - poly.pos.mean(axis=0)
    # Gyration tensor (3x3)
    T = (c.T @ c) / poly.n
    evals = np.linalg.eigvalsh(T)       # sorted ascending
    lam1, lam2, lam3 = sorted(evals)
    denom = (lam1 + lam2 + lam3) ** 2
    if denom < 1e-12:
        return 0.0
    num = ((lam1 - lam2) ** 2 + (lam2 - lam3) ** 2 + (lam1 - lam3) ** 2)
    return float(num / (2.0 * denom))


def compaction(poly: Polymer) -> float:
    """Rg relative to the ideal Gaussian chain value (b0 * sqrt(N/6)).

    < 1 means more compact than ideal (collapsed), > 1 means swollen.
    """
    rg_ideal = poly.p.b0 * np.sqrt(poly.n / 6.0)
    if rg_ideal < 1e-9:
        return 1.0
    return radius_of_gyration(poly) / rg_ideal


# Ordered list used by the panel: (key, display_label, y_axis_label, colour)
METRIC_DEFS = [
    ("rg",       radius_of_gyration, "Rg",            "radius of gyration",     (120, 180, 255)),
    ("ete",      end_to_end,         "end-to-end",     "end-to-end distance",    (210, 130, 140)),
    ("bond",     mean_bond_length,   "bond length",    "mean bond length",       (140, 210, 165)),
    ("contacts", contact_count,      "contacts",       "contact count  (>0.5)",  (240, 186, 92)),
    ("asph",     asphericity,        "asphericity",    "asphericity  [0–1]",     (214, 108, 232)),
    ("compact",  compaction,         "Rg / Rg_ideal",  "compaction index",       (92, 200, 255)),
]

MAX_HISTORY = 300   # frames of history kept per metric