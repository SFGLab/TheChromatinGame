"""Visual identity for The Chromatin Game.

Design direction: a laboratory instrument console. The palette is grounded in the
subject's own world -- Hi-C contact maps are read in the "fall" white-to-black
ramp, compartments in coolwarm, and loop calls are marked green. Those three are
non-negotiable scientific conventions, so the surrounding chrome stays quiet and
lets them carry the colour.
"""
from __future__ import annotations

import pygame

# --- surfaces -------------------------------------------------------------
INK        = (10, 13, 22)      # deepest background
INK_2      = (15, 19, 31)      # viewport background
PANEL      = (24, 30, 48)      # was (20, 25, 40) -- lifts panels off INK a touch
PANEL_HI   = (36, 44, 68)      # was (29, 36, 56)
RULE       = (58, 70, 100)     # was (43, 53, 79) -- dividers visible without shouting
RULE_SOFT  = (42, 52, 78)      # was (32, 40, 60)

# --- type -----------------------------------------------------------------
TEXT       = (238, 244, 255)   # was (223, 230, 244) -- primary text, near-white
TEXT_DIM   = (180, 192, 218)   # was (140, 153, 184) -- secondary labels
TEXT_FAINT = (140, 155, 188)   # was ( 92, 103, 132) -- captions, hints, mono numbers

# --- instrument accents ---------------------------------------------------
CYAN       = (92, 200, 255)    # interactive / selection
AMBER      = (240, 186, 92)    # warnings, timers
GREEN      = (58, 226, 141)    # loops (Hi-C convention)
MAGENTA    = (214, 108, 232)   # player two
ROYAL_RED  = (196, 30, 58)     # the MiNI-Lab button + its accent colour

# --- the science ----------------------------------------------------------
COMP_A     = (222, 74, 66)     # E1 > 0, red bead, weak attraction  (active)
COMP_B     = (58, 118, 212)    # E1 < 0, blue bead, strong attraction (inactive)

GOOD       = (58, 226, 141)
FAIR       = (240, 186, 92)
POOR       = (233, 88, 88)

# --- layout ---------------------------------------------------------------
WIN_W, WIN_H = 1500, 900
MIN_W, MIN_H = 1180, 760
HEADER_H = 56
FOOTER_H = 34
PAD = 14
RADIUS = 8

# Prefer fonts that ship with a strong regular weight or a dedicated Medium.
_MONO_STACK = [
    "jetbrainsmonomedium",   # JetBrains Mono Medium -- clean, strong on dark
    "jetbrainsmono",
    "firacodemedium",
    "firacode",
    "iosevkafixedmedium",
    "iosevka",
    "cascadiamono",          # ships with Windows Terminal
    "sfmono",
    "menlo",
    "consolas",
    "dejavusansmono",
    "liberationmono",
    "couriernew",
    "monospace",
]

_UI_STACK = [
    "intermedium",           # Inter Medium -- crisp at small sizes
    "inter",
    "robotomedium",
    "roboto",
    "sfprodisplaymedium",
    "sfprodisplay",
    "segoeuisemibold",       # Windows fallback with visible weight
    "segoeui",
    "helveticaneuemedium",
    "helveticaneue",
    "notosansmedium",
    "notosans",
    "dejavusans",
    "liberationsans",
    "arial",
    "sans",
]

_cache: dict[tuple, pygame.font.Font] = {}


def _resolve(stack: list[str]) -> str | None:
    for name in stack:
        path = pygame.font.match_font(name)
        if path:
            return path
    return None


_mono_path = None
_ui_path = None

FONT_SCALE = 1.0

def set_font_scale(mode: str) -> None:
    """Called by the settings menu. 'small' = default, 'large' = ~22% bigger."""
    global FONT_SCALE
    FONT_SCALE = 1.2 if mode == "large" else 1.0

def font(size: int, *, mono: bool = False, bold: bool = False) -> pygame.font.Font:
    """Fetch a cached font. Mono is used for every number in the app."""
    global _mono_path, _ui_path
    size = int(round(size * FONT_SCALE))     # <-- ADD THIS LINE
    key = (size, mono, bold)
    if key in _cache:
        return _cache[key]
    if _mono_path is None:
        _mono_path = _resolve(_MONO_STACK) or ""
    if _ui_path is None:
        _ui_path = _resolve(_UI_STACK) or ""
    path = _mono_path if mono else _ui_path
    if path:
        f = pygame.font.Font(path, size)
        f.set_bold(bold)
    else:
        f = pygame.font.SysFont(None, int(size * 1.25), bold=bold)
    _cache[key] = f
    return f


def score_color(x: float) -> tuple[int, int, int]:
    """Map a 0..1 quality value onto the instrument's good/fair/poor ramp."""
    x = max(0.0, min(1.0, x))
    if x < 0.5:
        t = x / 0.5
        a, b = POOR, FAIR
    else:
        t = (x - 0.5) / 0.5
        a, b = FAIR, GOOD
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def lerp_col(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))
