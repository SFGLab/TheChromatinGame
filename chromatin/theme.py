"""Visual identity for The Chromatin Game.

Design direction: a laboratory instrument console. The palette is grounded in the
subject's own world -- Hi-C contact maps are read in the "fall" white-to-black
ramp, compartments in coolwarm, and loop calls are marked green. Those three are
non-negotiable scientific conventions, so they stay fixed across every theme
below; only the surrounding "chrome" (panels, rules, text, accents) re-skins.

Call set_theme(name) to swap the whole UI palette at runtime -- every other
module reads these as `theme.INK`, `theme.TEXT`, etc. (attribute access, never
`from .theme import X`), so reassigning the module globals here is enough to
re-skin the entire app with no changes anywhere else.
"""
from __future__ import annotations

import os

import pygame
import pygame.gfxdraw as gfxdraw

# --- the science (fixed across every theme) --------------------------------
COMP_A     = (222, 74, 66)     # E1 > 0, red bead, weak attraction  (active)
COMP_B     = (58, 118, 212)    # E1 < 0, blue bead, strong attraction (inactive)
GREEN      = (58, 226, 141)    # loops (Hi-C convention)
GOOD       = (58, 226, 141)
FAIR       = (240, 186, 92)
POOR       = (233, 88, 88)

# --- layout (fixed) ---------------------------------------------------------
WIN_W, WIN_H = 1500, 900
MIN_W, MIN_H = 1180, 760
HEADER_H = 56
FOOTER_H = 34
PAD = 14
RADIUS = 8

# --- theme palettes ----------------------------------------------------------
# Each palette re-skins the "chrome": surfaces, rules, text and the four
# instrument accents (CYAN/AMBER/MAGENTA/ROYAL_RED). 3 dark + 2 light.
THEMES: dict[str, dict] = {
    "daylight": {   # light, default -- clean lab-whiteboard look
        "kind": "light", "label": "Daylight",
        "INK": (246, 247, 250), "INK_2": (255, 255, 255),
        "PANEL": (255, 255, 255), "PANEL_HI": (233, 238, 247),
        "RULE": (203, 211, 226), "RULE_SOFT": (222, 228, 239),
        "TEXT": (26, 31, 44), "TEXT_DIM": (76, 85, 106), "TEXT_FAINT": (122, 131, 151),
        "CYAN": (14, 116, 184), "AMBER": (188, 125, 16),
        "MAGENTA": (155, 55, 165), "ROYAL_RED": (196, 30, 58),
    },
    "paper": {      # light, warm cream -- softer on the eyes than pure white
        "kind": "light", "label": "Paper",
        "INK": (250, 246, 236), "INK_2": (255, 252, 245),
        "PANEL": (255, 253, 248), "PANEL_HI": (240, 231, 212),
        "RULE": (213, 201, 175), "RULE_SOFT": (230, 221, 200),
        "TEXT": (41, 35, 25), "TEXT_DIM": (94, 84, 68), "TEXT_FAINT": (138, 126, 106),
        "CYAN": (24, 108, 138), "AMBER": (176, 117, 18),
        "MAGENTA": (148, 68, 138), "ROYAL_RED": (178, 49, 39),
    },
    "midnight": {   # dark, the original console blue -- now one of three
        "kind": "dark", "label": "Midnight",
        "INK": (10, 13, 22), "INK_2": (15, 19, 31),
        "PANEL": (24, 30, 48), "PANEL_HI": (36, 44, 68),
        "RULE": (58, 70, 100), "RULE_SOFT": (42, 52, 78),
        "TEXT": (238, 244, 255), "TEXT_DIM": (180, 192, 218), "TEXT_FAINT": (140, 155, 188),
        "CYAN": (92, 200, 255), "AMBER": (240, 186, 92),
        "MAGENTA": (214, 108, 232), "ROYAL_RED": (196, 30, 58),
    },
    "carbon": {     # dark, neutral graphite with a teal accent
        "kind": "dark", "label": "Carbon",
        "INK": (12, 12, 13), "INK_2": (18, 18, 20),
        "PANEL": (28, 28, 31), "PANEL_HI": (40, 40, 45),
        "RULE": (70, 70, 76), "RULE_SOFT": (48, 48, 53),
        "TEXT": (240, 240, 238), "TEXT_DIM": (190, 190, 186), "TEXT_FAINT": (145, 145, 142),
        "CYAN": (110, 220, 190), "AMBER": (235, 170, 80),
        "MAGENTA": (200, 130, 220), "ROYAL_RED": (214, 70, 70),
    },
    "nebula": {     # dark, indigo/purple -- cooler, higher-contrast accents
        "kind": "dark", "label": "Nebula",
        "INK": (14, 10, 24), "INK_2": (20, 15, 34),
        "PANEL": (32, 24, 52), "PANEL_HI": (46, 34, 72),
        "RULE": (78, 62, 110), "RULE_SOFT": (56, 44, 82),
        "TEXT": (240, 236, 250), "TEXT_DIM": (195, 186, 220), "TEXT_FAINT": (150, 140, 180),
        "CYAN": (150, 140, 255), "AMBER": (245, 190, 110),
        "MAGENTA": (230, 110, 210), "ROYAL_RED": (214, 60, 100),
    },
}
THEME_ORDER = ("daylight", "paper", "midnight", "carbon", "nebula")
DEFAULT_THEME = "daylight"
CURRENT_THEME = DEFAULT_THEME   # name of the active theme, for Settings + persistence


def set_theme(name: str) -> None:
    """Swap the active palette. Unknown names fall back to the default."""
    global CURRENT_THEME
    pal = THEMES.get(name, THEMES[DEFAULT_THEME])
    name = name if name in THEMES else DEFAULT_THEME
    g = globals()
    for k, v in pal.items():
        if k not in ("kind", "label"):
            g[k] = v
    CURRENT_THEME = name


def cycle_theme(name: str) -> str:
    i = THEME_ORDER.index(name) if name in THEME_ORDER else 0
    return THEME_ORDER[(i + 1) % len(THEME_ORDER)]


def is_light() -> bool:
    """True for a light theme (daylight/paper) -- used where a colour tuned
    to sit on a dark background would look washed out on a light one."""
    return THEMES[CURRENT_THEME]["kind"] == "light"


set_theme(DEFAULT_THEME)   # populate INK/PANEL/TEXT/... module globals at import

# --- fonts ------------------------------------------------------------------
# Bundled so the UI looks the same everywhere, not just on machines that
# happen to have a nice sans installed. Quicksand is the default UI face
# (smooth, rounded, modern); the mono stack (numbers, aligned data) is
# unaffected -- a rounded display font is wrong for tabular figures.
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
_QS_MEDIUM = os.path.join(_FONT_DIR, "Quicksand-Medium.ttf")
_QS_BOLD = os.path.join(_FONT_DIR, "Quicksand-Bold.ttf")

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

# Fallback only -- used if the bundled Quicksand files are ever missing.
_UI_STACK = [
    "intermedium",
    "inter",
    "robotomedium",
    "roboto",
    "segoeuisemibold",
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

FONT_SCALE = 1.0

def set_font_scale(mode: str) -> None:
    """Called by the settings menu. 'small' = default, 'large' = ~22% bigger."""
    global FONT_SCALE
    FONT_SCALE = 1.2 if mode == "large" else 1.0

def font(size: int, *, mono: bool = False, bold: bool = False) -> pygame.font.Font:
    """Fetch a cached font. Mono is used for every number in the app."""
    global _mono_path
    size = int(round(size * FONT_SCALE))
    key = (size, mono, bold)
    if key in _cache:
        return _cache[key]
    if mono:
        if _mono_path is None:
            _mono_path = _resolve(_MONO_STACK) or ""
        if _mono_path:
            f = pygame.font.Font(_mono_path, size)
            f.set_bold(bold)
        else:
            f = pygame.font.SysFont(None, int(size * 1.25), bold=bold)
    else:
        qs = _QS_BOLD if bold else _QS_MEDIUM
        if os.path.isfile(qs):
            f = pygame.font.Font(qs, size)
        else:
            ui_path = _resolve(_UI_STACK) or ""
            if ui_path:
                f = pygame.font.Font(ui_path, size)
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


def circle(surf, color, center, radius, width: int = 0) -> None:
    """Anti-aliased circle -- drop-in swap for pygame.draw.circle. Beads,
    dots, handles and icon bubbles are everywhere in this UI, and
    gfxdraw's AA edge reads far smoother than the jagged default."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    if r <= 0:
        return
    col = (*color[:3], color[3] if len(color) > 3 else 255)
    if width <= 0:
        gfxdraw.filled_circle(surf, cx, cy, r, col)
        gfxdraw.aacircle(surf, cx, cy, r, col)
    else:
        for i in range(int(width)):
            gfxdraw.aacircle(surf, cx, cy, max(1, r - i), col)
