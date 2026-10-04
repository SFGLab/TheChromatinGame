"""Shared 3D geometry for the polymer's 'chain' and 'ribbon' representations.

Both render3d.py (software) and render3d_gpu.py (Panda3D) use this so the
backbone looks identical regardless of which one is actually drawing --
same split as beadcolor.py does for bead colour.

spheres -- the original look: one sphere sprite per bead, thin thread.
chain   -- a continuous, shinier tube along the backbone (rounded joints).
ribbon  -- a flat, shaded band that follows the backbone's centreline.
"""
from __future__ import annotations

import numpy as np

REP_MODES = ("spheres", "chain", "ribbon")
REP_LABEL = {
    "spheres": "beads (spheres)",
    "chain":   "rigid chain (tube)",
    "ribbon":  "flat ribbon",
}
SHORT_LABEL = {
    "spheres": "Spheres",
    "chain":   "Chain",
    "ribbon":  "Ribbon",
}


def cycle(mode: str) -> str:
    return REP_MODES[(REP_MODES.index(mode) + 1) % len(REP_MODES)]


# Reused as a fixed, headlamp-style light direction for the ribbon/tube
# shading -- the same vector render3d.py bakes into its sphere sprites, so
# all three representations look lit from the same place.
LIGHT = np.array([-0.35, 0.65, 0.68])
LIGHT /= np.linalg.norm(LIGHT)


def ribbon_frame(pos: np.ndarray) -> np.ndarray:
    """One unit 'side' vector per bead, perpendicular to the backbone, used
    to offset the ribbon's two edges. Parallel-transported bond to bond
    (each side vector is the previous one projected back into the new
    bond's perpendicular plane) so the ribbon doesn't flip or twist as the
    chain bends -- a full minimal-torsion frame is overkill for a game.
    """
    n = len(pos)
    side = np.zeros((n, 3))
    if n < 2:
        side[:] = (1.0, 0.0, 0.0)
        return side

    bond = pos[1:] - pos[:-1]
    lens = np.linalg.norm(bond, axis=1, keepdims=True)
    tangents = bond / np.clip(lens, 1e-9, None)

    up = np.array([0.0, 1.0, 0.0])
    alt = np.array([1.0, 0.0, 0.0])
    bond_side = np.zeros((n - 1, 3))
    prev = None
    for i in range(n - 1):
        tz = tangents[i]
        ref = alt if abs(float(np.dot(tz, up))) > 0.95 else up
        base = ref if prev is None else prev
        s = base - np.dot(base, tz) * tz
        sl = np.linalg.norm(s)
        s = s / sl if sl > 1e-9 else ref
        bond_side[i] = s
        prev = s
    side[0] = bond_side[0]
    side[-1] = bond_side[-1]
    if n > 2:
        mid = bond_side[:-1] + bond_side[1:]
        mn = np.linalg.norm(mid, axis=1, keepdims=True)
        # beads where consecutive bonds' frames happen to cancel (a near
        # 180-degree kink) fall back to the earlier bond's own side
        flat = (mn[:, 0] < 1e-6)
        side[1:-1] = np.where(flat[:, None], bond_side[:-1], mid / np.clip(mn, 1e-9, None))
    return side


def lit_fraction(normal: np.ndarray) -> np.ndarray:
    """Double-sided diffuse term in [0.45, 1.0] from a world-space normal
    (abs() so the ribbon's underside isn't just black -- there is no real
    back light in this scene, only the one fixed lamp)."""
    ndl = np.abs(normal @ LIGHT)
    return 0.45 + 0.55 * ndl


# --------------------------------------------------------- smooth ribbons
# A ChimeraX/PyMOL-style cartoon ribbon is a spline through the backbone,
# not a straight facet from bead to bead, and it reads as a lit, gently
# rounded band rather than a flat painted strip. The two helpers below give
# both renderers that look without duplicating the maths: smooth_ribbon_path
# subdivides the backbone into a Catmull-Rom curve (so bends are curves, not
# kinks), and cross_section_shade fakes the rounded cross-section with a
# soft highlight down the centreline.

# Profile across the ribbon's width: 3 lines (2 strips) from edge to edge.
# -1/+1 are the outer edges, so both renderers share exactly the same rims.
# (Panda3D's per-pixel colour interpolation smooths this out further on
# the GPU path; the CPU path's flat-filled strips are why spline
# subdivision -- lots of short, slightly-different segments -- matters so
# much there: it's what approximates a shaded curve out of flat facets.)
U_LEVELS = (-1.0, 0.0, 1.0)


def _catmull_rom(p0, p1, p2, p3, t: float):
    """Point and tangent of a uniform Catmull-Rom segment at t in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    pt = 0.5 * ((2 * p1) + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    tan = 0.5 * ((-p0 + p2) + 2 * (2 * p0 - 5 * p1 + 4 * p2 - p3) * t
                 + 3 * (-p0 + 3 * p1 - 3 * p2 + p3) * t2)
    return pt, tan


def ribbon_samples_per_segment(n: int, target_total: int = 260) -> int:
    """Fewer subdivisions per bond as the chain gets long, so a 150-bead
    ribbon still draws a bounded number of quads per frame."""
    if n < 2:
        return 1
    return max(2, min(6, round(target_total / (n - 1))))


def smooth_ribbon_path(pos: np.ndarray, side: np.ndarray, colors: np.ndarray,
                        samples_per_segment: int | None = None):
    """Subdivide the backbone into a smooth curve instead of straight
    bead-to-bead facets. Side vectors (kept perpendicular to the local
    tangent) and bead colours are interpolated along the same curve, so
    shape, lighting frame and colour all stay continuous.

    Returns (path_pos, path_side, path_tan, path_col), each with one row
    per sample point (1 + (n-1)*samples_per_segment rows for n >= 3).
    """
    n = len(pos)
    colors = np.asarray(colors, dtype=float)
    if n < 2:
        return pos.copy(), side.copy(), np.tile([0.0, 0.0, 1.0], (n, 1)), colors
    if n == 2:
        tan = pos[1] - pos[0]
        tan = tan / max(1e-9, np.linalg.norm(tan))
        return pos.copy(), side.copy(), np.tile(tan, (2, 1)), colors

    if samples_per_segment is None:
        samples_per_segment = ribbon_samples_per_segment(n)

    P = np.vstack([pos[0:1], pos, pos[-1:]])
    S = np.vstack([side[0:1], side, side[-1:]])
    C = np.vstack([colors[0:1], colors, colors[-1:]])
    out_p, out_s, out_t, out_c = [], [], [], []
    for i in range(n - 1):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        s1, s2 = S[i + 1], S[i + 2]
        c1, c2 = C[i + 1], C[i + 2]
        steps = samples_per_segment if i < n - 2 else samples_per_segment + 1
        for k in range(steps):
            t = k / samples_per_segment
            pt, tan = _catmull_rom(p0, p1, p2, p3, t)
            tl = np.linalg.norm(tan)
            tan = tan / tl if tl > 1e-9 else (p2 - p1)
            sv = (1 - t) * s1 + t * s2
            sv = sv - np.dot(sv, tan) * tan
            sl = np.linalg.norm(sv)
            sv = sv / sl if sl > 1e-9 else s1
            out_p.append(pt)
            out_s.append(sv)
            out_t.append(tan)
            out_c.append((1 - t) * c1 + t * c2)
    return (np.array(out_p), np.array(out_s), np.array(out_t), np.array(out_c))


def cross_section_shade(u):
    """Brightness multiplier across the ribbon's width (u in [-1, 1], 0 =
    centreline): a soft highlight down the middle that fakes a gently
    rounded, glossy cross-section -- the detail that makes a ChimeraX-style
    ribbon read as a lit 3D band instead of a flat painted strip."""
    u = np.clip(u, -1.0, 1.0)
    return 0.6 + 0.4 * np.cos(u * (np.pi / 2)) ** 1.4
