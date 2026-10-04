"""Shared bead-colouring logic for both polymer renderers (the Panda3D GPU
view and the NumPy/pygame software fallback), so the three colour modes
look identical regardless of which renderer is actually drawing.

compartment -- the original pastel A/B scheme, tinted by |E1| when available.
rainbow     -- one hue per bead index (colorbar legend on the right).
loop        -- orange for beads outside every tied loop's [i, j] span, green
               for beads inside one -- i.e. the same "domain" a loop's
               eps_domain bonus actually pulls together in physics.py.
"""
from __future__ import annotations

import numpy as np

from . import theme, colormaps

MODES = ("compartment", "rainbow", "loop")
MODE_LABEL = {
    "compartment": "by compartment (A/B)",
    "rainbow":     "rainbow (bead index)",
    "loop":        "loop domains",
}
SHORT_LABEL = {   # compact text for the on-screen Colour button
    "compartment": "A/B",
    "rainbow":     "Rainbow",
    "loop":        "Loop",
}

LOOP_DOMAIN_ON = (120, 210, 150)     # green  -- bead inside a tied loop's span
LOOP_DOMAIN_OFF = (225, 160, 90)     # orange -- bead outside every loop


def bead_colors(n: int, types: np.ndarray, loops, e1, mode: str) -> list[tuple]:
    if mode == "rainbow":
        return [colormaps.sample(colormaps.RAINBOW, i / max(1, n - 1)) for i in range(n)]

    if mode == "loop":
        in_domain = np.zeros(n, dtype=bool)
        for (i, j) in loops:
            in_domain[i:j + 1] = True
        return [(LOOP_DOMAIN_ON if in_domain[i] else LOOP_DOMAIN_OFF) for i in range(n)]

    # "compartment" (default): pastel A/B, tinted toward grey as |E1| shrinks
    out = []
    for i in range(n):
        base = (210, 130, 140) if types[i] > 0 else (120, 150, 210)
        if e1 is not None and len(e1) == n:
            amp = 0.4 + 0.6 * min(1.0, abs(float(e1[i])))
            base = theme.lerp_col((110, 115, 130), base, amp)
        out.append(base)
    return out


def draw_colorbar(surf, rect, n: int) -> None:
    """Small vertical bead-index legend, drawn only in 'rainbow' mode.

    Starts below the Colour/Shape buttons pinned to this corner (see
    draw.draw_view_controls) so the two never overlap.
    """
    import pygame
    w, h = 14, min(160, rect.height - 130)
    x = rect.right - w - 16
    y = rect.top + 78
    for row in range(h):
        col = colormaps.sample(colormaps.RAINBOW, 1.0 - row / max(1, h - 1))
        pygame.draw.line(surf, col, (x, y + row), (x + w, y + row))
    pygame.draw.rect(surf, theme.TEXT_FAINT, (x, y, w, h), 1)
    fnt = theme.font(12, mono=True)
    top = fnt.render(str(n - 1), True, theme.TEXT_DIM)
    bot = fnt.render("0", True, theme.TEXT_DIM)
    surf.blit(top, (x - top.get_width() - 4, y - 2))
    surf.blit(bot, (x - bot.get_width() - 4, y + h - bot.get_height() + 2))
    lab = fnt.render("bead #", True, theme.TEXT_FAINT)
    surf.blit(lab, (x + w / 2 - lab.get_width() / 2, y - 16))


def cycle(mode: str) -> str:
    return MODES[(MODES.index(mode) + 1) % len(MODES)]
