"""Picks the best available 3D polymer renderer.

Panda3D (GPU, chromatin/render3d_gpu.py) is used when it's installed and can
actually open a graphics context on this machine; otherwise the game falls
back to the original pure NumPy/pygame software renderer
(chromatin/render3d.py). Either way the returned object has the same public
surface (cam, hover, pending, dragging, mark, bead_r, color_mode,
screen_positions, pick, draw, cycle_color_mode), so callers never need to
know which one they got.
"""
from __future__ import annotations

from . import render3d

_force_software = False   # flipped on after a GPU init failure, see below


def create_polymer_view():
    global _force_software
    if not _force_software:
        try:
            from . import render3d_gpu
            if render3d_gpu.HAVE_PANDA:
                return render3d_gpu.PandaPolymerView()
        except Exception:
            # Any failure (no GL driver, panda3d not installed, a buffer
            # that couldn't be created, ...) falls back to the software
            # renderer for the rest of this run -- the game should never
            # refuse to start just because GPU rendering isn't available.
            _force_software = True
    return render3d.PolymerView()
