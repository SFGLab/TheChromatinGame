"""Instrument panels: heatmaps, the E1 track, the metrics table, buttons."""
from __future__ import annotations

import math

import numpy as np
import pygame

from . import colormaps as cm
from . import theme

MODES = ["split", "contact", "corr"]
MODE_LABEL = {
    "split": "contact / correlation",
    "contact": "contact frequency",
    "corr": "O/E correlation",
}


# ---------------------------------------------------------------- chrome
def panel(surf, rect, *, fill=theme.PANEL, border=theme.RULE, radius=theme.RADIUS):
    pygame.draw.rect(surf, fill, rect, border_radius=radius)
    pygame.draw.rect(surf, border, rect, 1, border_radius=radius)


def label(surf, text, x, y, *, size=12, col=theme.TEXT_DIM, mono=False, bold=False,
          center=False, right=False):
    f = theme.font(size, mono=mono, bold=bold)
    img = f.render(text, True, col)
    if center:
        x -= img.get_width() // 2
    elif right:
        x -= img.get_width()
    surf.blit(img, (x, y))
    return img.get_rect(topleft=(x, y))


def eyebrow(surf, text, x, y, col=theme.TEXT_FAINT):
    """Small tracked-out caps label. Used to name every panel."""
    f = theme.font(10, bold=True)
    cx = x
    for ch in text.upper():
        img = f.render(ch, True, col)
        surf.blit(img, (cx, y))
        cx += img.get_width() + 1.6
    return cx


class Button:
    def __init__(self, rect, text, *, key=None, accent=theme.CYAN, size=13, icon=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.key = key
        self.accent = accent
        self.size = size
        self.icon = icon
        self.hover = False
        self.active = False
        self.enabled = True

    def handle(self, ev) -> bool:
        if ev.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.enabled and self.rect.collidepoint(ev.pos):
                return True
        return False

    def draw(self, surf):
        if not self.enabled:
            bg, fg, bd = theme.PANEL, theme.TEXT_FAINT, theme.RULE_SOFT
        elif self.active:
            bg = theme.lerp_col(self.accent, theme.INK, 0.62)
            fg, bd = self.accent, self.accent
        elif self.hover:
            bg, fg, bd = theme.PANEL_HI, theme.TEXT, self.accent
        else:
            bg, fg, bd = theme.PANEL, theme.TEXT_DIM, theme.RULE
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, bd, self.rect, 1, border_radius=6)
        f = theme.font(self.size, bold=self.active)
        img = f.render(self.text, True, fg)
        surf.blit(img, img.get_rect(center=self.rect.center))
        if self.key:
            kf = theme.font(9, mono=True)
            ki = kf.render(self.key, True, theme.lerp_col(fg, theme.INK, 0.45))
            surf.blit(ki, (self.rect.right - ki.get_width() - 6, self.rect.top + 3))


# ---------------------------------------------------------------- heatmap
class Scale:
    """One colour scale, derived from the target and shared by both maps.

    Two Hi-C maps drawn on different scales cannot be compared by eye, which is
    the entire task here -- so the target defines the ramp and the player's map
    is rendered into it.
    """

    def __init__(self, P: np.ndarray, C: np.ndarray):
        n = P.shape[0]
        off = ~np.eye(n, dtype=bool)
        v = np.log10(np.clip(P[off], 1e-4, 1.0))
        self.vmin = float(np.percentile(v, 1.0))
        self.vmax = float(np.percentile(v, 99.5))
        if self.vmax - self.vmin < 0.35:
            self.vmax = self.vmin + 0.35
        self.cap = max(0.12, float(np.percentile(np.abs(C[off]), 97))) if n > 2 else 1.0

    @property
    def key(self):
        return (round(self.vmin, 4), round(self.vmax, 4), round(self.cap, 4))


def _matrix_rgb(P: np.ndarray, C: np.ndarray, mode: str, sc: Scale) -> np.ndarray:
    n = P.shape[0]
    logP = np.log10(np.clip(P, 1e-4, 1.0))
    img_c = cm.apply(logP, cm.FALL, sc.vmin, sc.vmax)
    img_r = cm.apply(C, cm.COOLWARM, -sc.cap, sc.cap)
    if mode == "contact":
        return img_c
    if mode == "corr":
        return img_r
    i = np.arange(n)
    upper = (i[None, :] > i[:, None])
    return np.where(upper[:, :, None], img_r, img_c)


def colorbars(surf, rect, sc: Scale, mode: str):
    """Compact key for whichever ramps are on screen."""
    x = rect.x
    if mode in ("split", "contact"):
        r = pygame.Rect(x, rect.y + 10, 116, 8)
        grad = np.linspace(sc.vmin, sc.vmax, 116)[None, :].repeat(8, 0)
        img = cm.apply(grad, cm.FALL, sc.vmin, sc.vmax)
        s = pygame.Surface((116, 8))
        pygame.surfarray.blit_array(s, np.transpose(img, (1, 0, 2)))
        surf.blit(s, r)
        pygame.draw.rect(surf, theme.RULE, r, 1)
        label(surf, "contact P", x, rect.y - 2, size=9, col=theme.TEXT_FAINT, mono=True)
        label(surf, f"{10 ** sc.vmin:.3f}", x, rect.y + 20, size=9, col=theme.TEXT_FAINT, mono=True)
        label(surf, f"{10 ** sc.vmax:.2f}", x + 116, rect.y + 20, size=9,
              col=theme.TEXT_FAINT, mono=True, right=True)
        x += 152
    if mode in ("split", "corr"):
        r = pygame.Rect(x, rect.y + 10, 116, 8)
        grad = np.linspace(-sc.cap, sc.cap, 116)[None, :].repeat(8, 0)
        img = cm.apply(grad, cm.COOLWARM, -sc.cap, sc.cap)
        s = pygame.Surface((116, 8))
        pygame.surfarray.blit_array(s, np.transpose(img, (1, 0, 2)))
        surf.blit(s, r)
        pygame.draw.rect(surf, theme.RULE, r, 1)
        label(surf, "O/E correlation", x, rect.y - 2, size=9, col=theme.TEXT_FAINT, mono=True)
        label(surf, f"{-sc.cap:+.2f}", x, rect.y + 20, size=9, col=theme.TEXT_FAINT, mono=True)
        label(surf, f"{sc.cap:+.2f}", x + 116, rect.y + 20, size=9,
              col=theme.TEXT_FAINT, mono=True, right=True)


class Heatmap:
    """A square Hi-C panel. Default view is split: contact frequency below the
    diagonal, O/E correlation above it -- loops and compartments in one square.
    """

    def __init__(self, title: str):
        self.title = title
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.grid = pygame.Rect(0, 0, 10, 10)
        self.n = 1
        self._surf: pygame.Surface | None = None
        self._sig = None
        self.hover_bin: tuple[int, int] | None = None

    def layout(self, rect: pygame.Rect, n: int):
        self.rect = pygame.Rect(rect)
        self.n = n
        pad_l, pad_t = 26, 30
        side = min(rect.w - pad_l - 6, rect.h - pad_t - 30)
        side = max(40, (side // n) * n if side // n > 0 else side)
        self.grid = pygame.Rect(rect.x + pad_l, rect.y + pad_t, side, side)

    def bin_at(self, mx, my):
        if not self.grid.collidepoint(mx, my):
            return None
        j = int((mx - self.grid.x) / self.grid.w * self.n)
        i = int((my - self.grid.y) / self.grid.h * self.n)
        if 0 <= i < self.n and 0 <= j < self.n:
            return (i, j)
        return None

    def draw(self, surf, P, C, mode, loops_true=None, loops_player=None, *,
             subtitle: str = "", accent=theme.TEXT_FAINT, live: bool = False,
             mouse=None, scale: "Scale | None" = None):
        n = P.shape[0]
        sc = scale or Scale(P, C)
        sig = (id(P), float(P.sum()), float(C.sum()), mode, n, self.grid.w, sc.key)
        if self._surf is None or sig != self._sig:
            rgb = _matrix_rgb(P, C, mode, sc)
            small = pygame.Surface((n, n))
            pygame.surfarray.blit_array(small, np.transpose(rgb, (1, 0, 2)))
            self._surf = pygame.transform.scale(small, (self.grid.w, self.grid.h))
            self._sig = sig

        eyebrow(surf, self.title, self.rect.x, self.rect.y + 2, accent)
        if subtitle:
            label(surf, subtitle, self.rect.x + 1, self.rect.y + 15, size=10,
                  col=theme.TEXT_FAINT)
        if live:
            r = pygame.Rect(self.rect.right - 44, self.rect.y + 1, 40, 13)
            pygame.draw.rect(surf, theme.lerp_col(theme.AMBER, theme.INK, 0.7), r,
                             border_radius=3)
            label(surf, "LIVE", r.centerx, r.y + 1, size=9, col=theme.AMBER,
                  mono=True, bold=True, center=True)

        surf.blit(self._surf, self.grid)
        pygame.draw.rect(surf, theme.RULE, self.grid, 1)

        cell = self.grid.w / n
        if cell >= 9:                       # bin gridlines only when they can breathe
            gs = pygame.Surface(self.grid.size, pygame.SRCALPHA)
            for k in range(1, n):
                x = int(k * cell)
                pygame.draw.line(gs, (0, 0, 0, 26), (x, 0), (x, self.grid.h))
                pygame.draw.line(gs, (0, 0, 0, 26), (0, x), (self.grid.w, x))
            surf.blit(gs, self.grid)

        # bin numbers -- the whole point of a small polymer is that you can count
        step = 1 if cell >= 17 else (2 if cell >= 10 else (5 if cell >= 6 else 10))
        f = theme.font(9, mono=True)
        for k in range(0, n, step):
            x = self.grid.x + (k + 0.5) * cell
            img = f.render(str(k), True, theme.TEXT_FAINT)
            surf.blit(img, (x - img.get_width() / 2, self.grid.y - 12))
            y = self.grid.y + (k + 0.5) * cell
            img = f.render(str(k), True, theme.TEXT_FAINT)
            surf.blit(img, (self.grid.x - img.get_width() - 4, y - img.get_height() / 2))

        # diagonal
        pygame.draw.line(surf, (255, 255, 255, 40), self.grid.topleft,
                         self.grid.bottomright, 1)

        if loops_true:
            for (i, j) in loops_true:
                self._dot(surf, i, j, theme.GREEN, filled=True)
                self._dot(surf, j, i, theme.GREEN, filled=True)
        if loops_player:
            for (i, j) in loops_player:
                self._dot(surf, i, j, (255, 255, 255), filled=False)
                self._dot(surf, j, i, (255, 255, 255), filled=False)

        mx, my = mouse if mouse is not None else pygame.mouse.get_pos()
        self.hover_bin = self.bin_at(mx, my)
        y = self.grid.bottom + 5
        if self.hover_bin:
            i, j = self.hover_bin
            cx = self.grid.x + (j + 0.5) * cell
            cy = self.grid.y + (i + 0.5) * cell
            pygame.draw.rect(surf, theme.CYAN,
                             (self.grid.x + j * cell, self.grid.y + i * cell,
                              max(2, cell), max(2, cell)), 1)
            txt = f"[{i:>2},{j:>2}]  P={P[i, j]:.3f}  r={C[i, j]:+.2f}"
            label(surf, txt, self.grid.x, y, size=10, col=theme.CYAN, mono=True)
        else:
            label(surf, MODE_LABEL[mode], self.grid.x, y, size=10, col=theme.TEXT_FAINT)

    def _dot(self, surf, i, j, col, filled=True):
        cell = self.grid.w / self.n
        cx = self.grid.x + (j + 0.5) * cell
        cy = self.grid.y + (i + 0.5) * cell
        r = max(2.5, min(6.0, cell * 0.34))
        if filled:
            g = pygame.Surface((int(r * 6), int(r * 6)), pygame.SRCALPHA)
            pygame.draw.circle(g, (*col, 55), (int(r * 3), int(r * 3)), int(r * 2.6))
            surf.blit(g, (cx - r * 3, cy - r * 3))
            pygame.draw.circle(surf, col, (int(cx), int(cy)), int(r))
            pygame.draw.circle(surf, (255, 255, 255), (int(cx), int(cy)), int(r), 1)
        else:
            pygame.draw.circle(surf, col, (int(cx), int(cy)), int(r + 1.5), 1)


# ------------------------------------------------------------- E1 track
def eigen_track(surf, rect, e1, types=None, *, title="E1", show_axis=True):
    pygame.draw.rect(surf, theme.INK_2, rect, border_radius=3)
    pygame.draw.rect(surf, theme.RULE_SOFT, rect, 1, border_radius=3)
    n = len(e1)
    if n == 0:
        return
    mid = rect.y + rect.h / 2
    w = rect.w / n
    for i in range(n):
        v = float(np.clip(e1[i], -1, 1))
        h = abs(v) * (rect.h / 2 - 2)
        x = rect.x + i * w
        col = theme.COMP_A if v >= 0 else theme.COMP_B
        if h < 0.7:
            h = 0.7
        r = pygame.Rect(int(x) + 0, int(mid - h if v >= 0 else mid), max(1, int(w) - 1),
                        max(1, int(h)))
        pygame.draw.rect(surf, col, r)
    if show_axis:
        pygame.draw.line(surf, theme.RULE, (rect.x, mid), (rect.right, mid), 1)
    label(surf, title, rect.x + 3, rect.y - 12, size=9, col=theme.TEXT_FAINT,
          mono=True, bold=True)


def type_track(surf, rect, types, hover=None):
    """The bead colouring the player controls, as a ribbon under the map."""
    n = len(types)
    w = rect.w / n
    for i in range(n):
        col = theme.COMP_A if types[i] > 0 else theme.COMP_B
        r = pygame.Rect(int(rect.x + i * w), rect.y, max(1, int(w) - 1), rect.h)
        pygame.draw.rect(surf, col, r)
        if hover == i:
            pygame.draw.rect(surf, theme.CYAN, r, 2)
    pygame.draw.rect(surf, theme.RULE_SOFT, rect, 1)


# ------------------------------------------------------------ metric table
ROWS = [
    ("scc", "SCC", "stratum-adjusted corr", 0.0, 1.0),
    ("pearson", "Pearson", "log contact, |i-j|>=2", 0.0, 1.0),
    ("spearman", "Spearman", "rank corr", 0.0, 1.0),
    ("eig_r", "E1 r", "compartment corr", -1.0, 1.0),
    ("checker", "Checker", "correlation-matrix corr", -1.0, 1.0),
    ("apa", "APA", "loop corner enrichment", 0.0, 1.0),
    ("loop_f1", "Loop F1", "anchors within +/-1 bin", 0.0, 1.0),
]


def metric_table(surf, rect, rep, *, live=False):
    eyebrow(surf, "simulated  vs  experimental", rect.x, rect.y)
    if live:
        label(surf, "updating", rect.right, rect.y - 1, size=9, col=theme.AMBER,
              mono=True, right=True)
    y = rect.y + 18
    row_h = 21
    for key, name, desc, lo, hi in ROWS:
        v = float(rep.get(key, 0.0)) if rep else 0.0
        q = (v - lo) / (hi - lo) if hi > lo else 0.0
        q = max(0.0, min(1.0, q))
        col = theme.score_color(q)
        label(surf, name, rect.x + 2, y + 3, size=11, col=theme.TEXT, mono=True)
        label(surf, desc, rect.x + 66, y + 4, size=10, col=theme.TEXT_FAINT)
        bar = pygame.Rect(rect.right - 148, y + 6, 86, 7)
        pygame.draw.rect(surf, theme.INK, bar, border_radius=3)
        if lo < 0:
            mid = bar.x + bar.w / 2
            pygame.draw.line(surf, theme.RULE, (mid, bar.y - 2), (mid, bar.bottom + 2), 1)
            wdt = abs(v) / hi * (bar.w / 2)
            x0 = mid if v >= 0 else mid - wdt
            pygame.draw.rect(surf, col, (x0, bar.y, max(1, wdt), bar.h), border_radius=3)
        else:
            pygame.draw.rect(surf, col, (bar.x, bar.y, max(1, q * bar.w), bar.h),
                             border_radius=3)
        label(surf, f"{v:+.3f}" if lo < 0 else f"{v:.3f}", rect.right - 2, y + 3,
              size=11, col=col, mono=True, right=True)
        y += row_h
        pygame.draw.line(surf, theme.RULE_SOFT, (rect.x, y - 3), (rect.right, y - 3), 1)
    return y


def big_score(surf, rect, value, caption, accent=theme.CYAN):
    panel(surf, rect, fill=theme.INK_2, border=theme.RULE)
    eyebrow(surf, caption, rect.x + 10, rect.y + 8, accent)
    f = theme.font(int(rect.h * 0.52), mono=True, bold=True)
    img = f.render(f"{value:5.1f}", True, theme.score_color(value / 100.0))
    surf.blit(img, (rect.right - img.get_width() - 10,
                    rect.centery - img.get_height() // 2 + 4))
    
class Slider:
    """A draggable horizontal parameter control: label above, track + handle,
    current value on the right. Click or drag anywhere on the track to set it."""

    def __init__(self, rect, label: str, value: float, lo: float, hi: float,
                step: float | None = None, fmt: str = "{:.3f}", integer: bool = False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = value
        self.lo, self.hi = lo, hi
        self.step = step
        self.fmt = fmt
        self.integer = integer
        self.dragging = False
        self.enabled = True
        self.key = ""          # set by the caller: which SimParams field this drives

    def _value_at(self, x: int) -> float:
        t = (x - self.rect.x) / max(1, self.rect.w)
        t = max(0.0, min(1.0, t))
        v = self.lo + t * (self.hi - self.lo)
        if self.step:
            v = round(v / self.step) * self.step
        if self.integer:
            v = int(round(v))
        return max(self.lo, min(self.hi, v))

    def handle(self, ev) -> bool:
        """Returns True the frame the value actually changes."""
        if not self.enabled:
            return False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
                and self.rect.collidepoint(ev.pos):
            self.dragging = True
            new = self._value_at(ev.pos[0])
            changed, self.value = new != self.value, new
            return changed
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        if ev.type == pygame.MOUSEMOTION and self.dragging:
            new = self._value_at(ev.pos[0])
            changed, self.value = new != self.value, new
            return changed
        return False

    def draw(self, sc) -> None:
        r = self.rect
        S = getattr(theme, "FONT_SCALE", 1.0)
        f = theme.font(max(10, int(11 * S)), mono=True)
        col = theme.CYAN if self.enabled else theme.TEXT_FAINT

        img = f.render(self.label, True, theme.TEXT_DIM)
        sc.blit(img, (r.x, r.y - img.get_height() - 3))
        vs = self.fmt.format(self.value)
        vimg = f.render(vs, True, theme.TEXT)
        sc.blit(vimg, (r.right - vimg.get_width(), r.y - img.get_height() - 3))

        track = pygame.Rect(r.x, r.centery - 3, r.w, 6)
        pygame.draw.rect(sc, theme.PANEL, track, border_radius=3)
        t = (self.value - self.lo) / max(1e-9, self.hi - self.lo)
        fw = int(track.w * max(0.0, min(1.0, t)))
        pygame.draw.rect(sc, col, (track.x, track.y, fw, track.h), border_radius=3)
        hx = track.x + fw
        pygame.draw.circle(sc, theme.TEXT, (hx, track.centery), 7)
        pygame.draw.circle(sc, col, (hx, track.centery), 7, 2)
