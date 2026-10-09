"""Live structural analysis panel: a 2×3 grid of time-series plots.

Designed to overlay any game screen as a semi-transparent panel.
Call .update(poly) every frame (throttled internally), then .draw(sc, W, H).
History accumulates from the moment the simulation starts -- opening and
closing the panel just shows/hides the window. The Clear button inside the
panel explicitly wipes and restarts the history.
"""
from __future__ import annotations
from collections import deque
import time
import pygame
import numpy as np
from . import theme
from .metrics import METRIC_DEFS, MAX_HISTORY
from .physics import Polymer


class AnalysisPanel:
    """Floating 2×3 grid of live structural metric plots.

    Usage
    -----
    panel = AnalysisPanel()

    # every simulation frame, regardless of whether the panel is visible:
    panel.update(poly)

    # every draw frame, only when the panel should be shown:
    panel.draw(sc, W, H)

    # to handle the Clear button click:
    if panel.handle_click(mouse_pos):
        pass   # already handled internally
    """

    SAMPLE_EVERY_MS = 150   # ms between metric samples

    def __init__(self):
        self.histories: dict[str, deque] = {
            key: deque(maxlen=MAX_HISTORY) for key, *_ in METRIC_DEFS
        }
        self.times: deque = deque(maxlen=MAX_HISTORY)
        self._last_sample = 0.0
        self._t0 = time.time()
        self._clear_btn: pygame.Rect | None = None   # set during draw()

    def reset(self) -> None:
        """Wipe all history and restart the time axis from now."""
        for d in self.histories.values():
            d.clear()
        self.times.clear()
        self._t0 = time.time()
        self._last_sample = 0.0

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Call from handle_events on MOUSEBUTTONDOWN.

        Returns True if the click was consumed by the panel (Clear button).
        """
        if self._clear_btn is not None and self._clear_btn.collidepoint(pos):
            self.reset()
            return True
        return False

    def update(self, poly: Polymer) -> None:
        """Sample all metrics. Throttled; safe to call every frame."""
        now = time.time()
        if (now - self._last_sample) * 1000 < self.SAMPLE_EVERY_MS:
            return
        self._last_sample = now
        self.times.append(now - self._t0)
        for key, fn, *_ in METRIC_DEFS:
            try:
                val = float(fn(poly))
            except Exception:
                val = float("nan")
            self.histories[key].append(val)

    # ---------------------------------------------------------------- draw
    def draw(self, sc: pygame.Surface, W: int, H: int) -> None:
        """Draw the panel as a semi-transparent overlay."""
        pw = int(W * 0.82)
        ph = int(H * 0.80)
        px = (W - pw) // 2
        py = (H - ph) // 2

        # Background
        bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(bg, (*theme.INK, 230), bg.get_rect(), border_radius=10)
        pygame.draw.rect(bg, (*theme.RULE, 200), bg.get_rect(), 1, border_radius=10)
        sc.blit(bg, (px, py))

        # Title row
        sc.blit(theme.font(14, bold=True).render(
            "STRUCTURAL ANALYSIS — live metrics", True, theme.TEXT),
            (px + 20, py + 12))
        n_pts = len(self.times)
        elapsed = f"{self.times[-1]:.1f} s" if self.times else "0.0 s"
        sc.blit(theme.font(11, mono=True).render(
            f"samples: {n_pts} / {MAX_HISTORY}  ·  elapsed: {elapsed}"
            f"  ·  every {self.SAMPLE_EVERY_MS} ms  ·  ESC to close",
            True, theme.TEXT_DIM),
            (px + 20, py + 32))

        # Clear button (top-right corner of the panel)
        self._clear_btn = pygame.Rect(px + pw - 82, py + 12, 68, 26)
        pygame.draw.rect(sc, theme.PANEL_HI, self._clear_btn, border_radius=5)
        pygame.draw.rect(sc, theme.RULE, self._clear_btn, 1, border_radius=5)
        clbl = theme.font(12, bold=True).render("Clear", True, theme.TEXT_DIM)
        sc.blit(clbl, (self._clear_btn.x + (self._clear_btn.w - clbl.get_width()) // 2,
                       self._clear_btn.y + (self._clear_btn.h - clbl.get_height()) // 2))

        # 2 rows × 3 columns of plots
        COLS, ROWS = 3, 2
        pad = 14
        header_h = 56
        gw = (pw - pad * (COLS + 1)) // COLS
        gh = (ph - header_h - pad * (ROWS + 1)) // ROWS

        for idx, (key, _fn, _short, long_label, col) in enumerate(METRIC_DEFS):
            col_i = idx % COLS
            row_i = idx // COLS
            gx = px + pad + col_i * (gw + pad)
            gy = py + header_h + pad + row_i * (gh + pad)
            self._draw_plot(sc, gx, gy, gw, gh, key, long_label, col)

    def _draw_plot(self, sc, gx, gy, gw, gh,
                   key: str, label: str, col: tuple) -> None:
        hist  = list(self.histories[key])
        times = list(self.times)

        # Plot background
        bg = pygame.Surface((gw, gh), pygame.SRCALPHA)
        pygame.draw.rect(bg, (*theme.PANEL, 200), bg.get_rect(), border_radius=6)
        pygame.draw.rect(bg, (*theme.RULE_SOFT, 160), bg.get_rect(), 1, border_radius=6)
        sc.blit(bg, (gx, gy))

        # Metric name (top-left)
        sc.blit(theme.font(11, bold=True).render(label, True, col), (gx + 8, gy + 6))

        # Waiting for data
        if len(hist) < 2:
            sc.blit(theme.font(11, mono=True).render("collecting…", True, theme.TEXT_DIM),
                    (gx + 8, gy + gh // 2 - 6))
            return

        vals   = np.array(hist, dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            sc.blit(theme.font(11, mono=True).render("no data", True, theme.TEXT_DIM),
                    (gx + 8, gy + gh // 2 - 6))
            return

        vmin, vmax = float(finite.min()), float(finite.max())
        span = vmax - vmin
        if span < 1e-6:
            vmin -= 0.5; vmax += 0.5; span = 1.0

        # Inner plot area padding
        lpad, rpad, tpad, bpad = 48, 10, 32, 20
        plot_w = gw - lpad - rpad
        plot_h = gh - tpad - bpad

        t0 = times[0]  if times else 0.0
        t1 = times[-1] if times else 1.0
        dt = t1 - t0 or 1.0

        def to_screen(t_val: float, v_val: float) -> tuple[int, int]:
            sx = gx + lpad + int((t_val - t0) / dt * plot_w)
            sy = gy + tpad + plot_h - int((v_val - vmin) / span * plot_h)
            return sx, sy

        # Y-axis grid lines and labels
        gf = theme.font(10, mono=True)
        for frac in (0.0, 0.5, 1.0):
            v  = vmin + frac * span
            _, sy = to_screen(t0, v)
            pygame.draw.line(sc, theme.RULE_SOFT,
                             (gx + lpad, sy), (gx + lpad + plot_w, sy), 1)
            lbl = gf.render(f"{v:.2f}", True, theme.TEXT_DIM)
            sc.blit(lbl, (gx + lpad - lbl.get_width() - 4, sy - 5))

        # Current value (top-right of subplot)
        cur = vals[-1]
        cur_txt = f"{cur:.3f}" if np.isfinite(cur) else "—"
        cur_lbl = theme.font(12, mono=True, bold=True).render(cur_txt, True, col)
        sc.blit(cur_lbl, (gx + gw - rpad - cur_lbl.get_width() - 2, gy + 6))

        # Time axis label (bottom-right)
        sc.blit(gf.render(f"{t1:.0f} s", True, theme.TEXT_DIM),
                (gx + lpad + plot_w - 20, gy + tpad + plot_h + 4))

        # Clip to the plot rectangle so lines don't bleed into neighbours
        clip      = pygame.Rect(gx + lpad, gy + tpad, plot_w, plot_h)
        prev_clip = sc.get_clip()
        sc.set_clip(clip)

        # Line plot + dot at the latest point
        pts = [to_screen(t_val, v_val)
               for t_val, v_val in zip(times, vals)
               if np.isfinite(v_val)]
        if len(pts) >= 2:
            pygame.draw.lines(sc, col, False, pts, 2)
        if pts:
            theme.circle(sc, col,       pts[-1], 4)
            theme.circle(sc, theme.TEXT, pts[-1], 4, 1)

        sc.set_clip(prev_clip)