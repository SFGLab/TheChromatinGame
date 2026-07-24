# chromatin/draw.py
"""All draw_* methods for the Game class.

Attached to Game via setattr() in app.py. Each function takes `self`
(a Game instance) as its first argument, exactly like a regular method.

Organisation
------------
  SHARED          compute_layout, paint_background, draw, draw_toast
  MENU / SETTINGS draw_menu, draw_settings
  LOADING         draw_loading
  PLAY SESSION    draw_play, draw_header, draw_footer, draw_view_hud,
                  draw_settle_overlay, draw_results, draw_help
  MiNI-LAB        draw_lab, draw_lab_panel, compute_lab_layout
"""
from __future__ import annotations
import time
import numpy as np
import pygame

from . import theme, widgets
from .levels import LEVELS
from .physics import B_TYPE
from .constants import *

# ===========================================================================
# SHARED -- called on every frame regardless of game state
# ===========================================================================

def draw(self):
    """Main dispatch: clear → layout → background → state-specific screen."""
    W, H = self.screen.get_size()
    self.compute_layout(W, H)
    self.paint_background(W, H)

    if self.state == MENU:
        self.draw_menu(W, H)
    elif self.state == SETTINGS:
        self.draw_settings(W, H)
    elif self.state == LOADING:
        self.draw_loading(W, H)
    elif self.state in (PLAY, SETTLE):
        self.draw_play(W, H)
        if self.state == SETTLE:
            self.draw_settle_overlay(W, H)
    elif self.state == RESULTS:
        self.draw_results(W, H)
    elif self.state == LAB:
        self.draw_lab(W, H)

    if self.show_help:
        self.draw_help(W, H)
    self.draw_toast(W, H)


def compute_layout(self, W, H):
    """Compute rects for the PLAY screen: header/footer/viewport/heatmaps.

    All sizes scale with theme.FONT_SCALE so Large font mode gets more room.
    The Lab has its own layout function (compute_lab_layout) because it has
    a completely different panel arrangement.
    """
    S = getattr(theme, "FONT_SCALE", 1.0)
    P     = int(theme.PAD * S)
    hdr_h = int(theme.HEADER_H * S)
    ftr_h = int(theme.FOOTER_H * S)

    # Right column widens slightly at Large so bin labels don't crowd.
    right_frac = 0.45 + (S - 1.0) * 0.10
    right_w    = max(int(620 * S), int(W * right_frac))

    body  = pygame.Rect(0, hdr_h, W, H - hdr_h - ftr_h)
    view  = pygame.Rect(P, body.y + P, W - right_w - P, body.h - 2 * P)
    right = pygame.Rect(view.right + P, body.y + P, right_w - 2 * P, body.h - 2 * P)

    # Heatmap panel: title strip + square grid + label strip, all scaled.
    title_h = int(30 * S)
    label_h = int(22 * S)
    inner   = int(32 * S)
    hm_w    = (right.w - P) // 2
    side    = max(80, hm_w - inner)
    hm_h    = title_h + side + label_h

    self.rects = {
        "header": pygame.Rect(0, 0, W, hdr_h),
        "footer": pygame.Rect(0, H - ftr_h, W, ftr_h),
        "body":   body,
        "view":   view,
        "right":  right,
        "tgt":    pygame.Rect(right.x, right.y, hm_w, hm_h),
        "sim":    pygame.Rect(right.x + hm_w + P, right.y, hm_w, hm_h),
    }


def paint_background(self, W, H):
    """Subtle top-to-bottom gradient from INK_2 to INK."""
    sc = self.screen
    sc.fill(theme.INK)
    grad = pygame.Surface((1, H))
    for y in range(H):
        t = y / max(1, H - 1)
        grad.set_at((0, y), theme.lerp_col(theme.INK_2, theme.INK, t ** 0.7))
    sc.blit(pygame.transform.scale(grad, (W, H)), (0, 0))


def draw_toast(self, W, H):
    """Transient notification that fades out after ~3 s."""
    if not self.toast:
        return
    age = time.time() - self.toast_t
    if age > 3.0:
        self.toast = ""
        return
    a = 255 if age < 2.3 else int(255 * (1 - (age - 2.3) / 0.7))
    f   = theme.font(12, mono=True)
    img = f.render(self.toast, True, theme.TEXT)
    pad = 10
    r = pygame.Rect(W // 2 - img.get_width() // 2 - pad,
                    H - theme.FOOTER_H - 44,
                    img.get_width() + pad * 2, 26)
    s = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*theme.PANEL_HI, min(a, 235)), s.get_rect(), border_radius=6)
    pygame.draw.rect(s, (*theme.RULE,     min(a, 235)), s.get_rect(), 1, border_radius=6)
    s.blit(img, (pad, 6))
    s.set_alpha(a)
    self.screen.blit(s, r)


# ===========================================================================
# MENU + SETTINGS
# ===========================================================================

def draw_menu(self, W, H):
    """Main menu: title, level chooser, mode/difficulty toggles, action row."""
    sc = self.screen
    self.buttons = {}

    # Background: rotating demo polymer on the right half, veiled on the left.
    vr = pygame.Rect(int(W * 0.50), 0, int(W * 0.50), H)
    self.demo_view.cam.dist   = 9.5
    self.demo_view.cam.center = self.demo.pos.mean(axis=0)
    self.demo_view.draw(sc, vr, self.demo.pos, self.demo.types, self.demo.loops,
                        self.demo.box, t=time.time() - self.t0,
                        show_index=False, mouse=self.mouse, show_box=False)
    veil = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.rect(veil, (*theme.INK, 150), (0, 0, int(W * 0.55), H))
    sc.blit(veil, (0, 0))

    x = 70
    y = int(H * 0.16)
    widgets.eyebrow(sc, "a polymer physics game about reading hi-c", x, y, theme.CYAN)
    f = theme.font(64, bold=True)
    sc.blit(f.render("THE CHROMATIN", True, theme.TEXT), (x - 3, y + 18))
    sc.blit(f.render("       GAME",   True, theme.GREEN), (x - 3, y + 88))
    widgets.label(sc, "Build the fold. Match the map.", x, y + 182, size=16,
                  col=theme.TEXT_DIM)

    # ---- Level chooser
    y2 = y + 222
    widgets.eyebrow(sc, "locus", x, y2)
    for i, lvl in enumerate(LEVELS):
        r = pygame.Rect(x, y2 + 22 + i * 46, 340, 28)
        b = widgets.Button(r, lvl.name, size=13)
        b.active = (i == self.sel_level)
        self.buttons[f"lvl{i}"] = b
        b.draw(sc)
        plural = "loop" if lvl.n_loops == 1 else "loops"
        widgets.label(sc, f"{lvl.n} beads · {lvl.n_loops} {plural}",
                      r.right + 12, r.y + 7, size=11,
                      col=theme.TEXT_FAINT, mono=True)
        # Show best score for the current difficulty.
        rec_key = lvl.name + ("  [hard]" if self.sel_hard else "")
        rec = self.records.get(rec_key, {}).get("best")
        if rec:
            widgets.label(sc, f"best {rec:.1f}", r.right + 200, r.y + 7,
                          size=11, col=theme.AMBER, mono=True)

    # ---- Mode row: Solo / Versus + difficulty Easy / Hard
    y3 = y2 + 18 + len(LEVELS) * 46 + 16
    widgets.eyebrow(sc, "mode", x, y3)

    b1 = widgets.Button((x,       y3 + 22, 110, 30), "Solo")
    b2 = widgets.Button((x + 118, y3 + 18, 110, 30), "Versus", accent=theme.MAGENTA)
    b1.active = not self.sel_two
    b2.active = self.sel_two
    self.buttons["solo"]   = b1
    self.buttons["versus"] = b2
    b1.draw(sc); b2.draw(sc)

    # Difficulty buttons sit on the same row, right of the mode buttons.
    bd1 = widgets.Button((x + 480, y3 + 18, 90, 30), "Easy")
    bd2 = widgets.Button((x + 574, y3 + 18, 90, 30), "Hard", accent=theme.POOR)
    bd1.active = not self.sel_hard
    bd2.active = self.sel_hard
    self.buttons["easy"] = bd1
    self.buttons["hard"] = bd2
    bd1.draw(sc); bd2.draw(sc)
    if self.sel_hard:
        widgets.label(sc, "hard: no loop dots, no E1 track on the target",
                      x + 480, y3 + 56, size=11, col=theme.POOR)

    if self.sel_two:
        br = widgets.Button((x + 244, y3 + 18, 108, 30), f"Rounds: {self.sel_rounds}")
        ts = "off" if not self.sel_turn_sec else f"{self.sel_turn_sec}s"
        bt = widgets.Button((x + 360, y3 + 18, 116, 30), f"Turn: {ts}")
        self.buttons["rounds"] = br
        self.buttons["turn"]   = bt
        br.draw(sc); bt.draw(sc)
        widgets.label(sc, "P1 plays loops · P2 plays compartments · same polymer",
                      x, y3 + 56, size=11, col=theme.TEXT_FAINT)

    # ---- Action row: Start / Shuffle / How to play / Settings / Quit
    y4   = y3 + 86
    bs   = widgets.Button((x,       y4, 150, 38), "Start",      key="ENTER", size=15)
    bsh  = widgets.Button((x + 158, y4, 130, 38), "Shuffle seed",             size=13)
    bh   = widgets.Button((x + 296, y4, 130, 38), "How to play", key="H",    size=13)
    bset = widgets.Button((x + 434, y4, 110, 38), "Settings",                 size=13)
    bq   = widgets.Button((x + 552, y4,  80, 38), "Quit",                     size=13)
    bs.active = True
    for k, b in (("start", bs), ("seed", bsh), ("help", bh),
                 ("settings", bset), ("quit", bq)):
        self.buttons[k] = b
        b.draw(sc)
    widgets.label(sc, f"seed {self.sel_seed}", x + 640, y4 + 12,
                  size=11, col=theme.TEXT_FAINT, mono=True)

    # ---- Chromatin MiNI-Lab shortcut
    y5   = y4 + 56
    blab = widgets.Button((x, y5, 260, 42), "Chromatin MiNI-Lab",
                          size=15, accent=theme.ROYAL_RED)
    blab.active = True
    self.buttons["lab"] = blab
    blab.draw(sc)
    widgets.label(sc, "free-play sandbox — every force, every parameter, live",
                  x + 272, y5 + 14, size=11, col=theme.TEXT_FAINT)

    # ---- Music strip (bottom-left)
    m = self.music
    widgets.label(sc,
                  ("♪  " + m.title()) if m.has_music
                  else "♪  drop your piano .mp3 files into  music/",
                  x, H - 54, size=12,
                  col=theme.TEXT_DIM if m.has_music else theme.TEXT_FAINT)
    if m.has_music:
        widgets.label(sc, f"{len(m.tracks)} track(s)  ·  M mute  ·  N next",
                      x, H - 36, size=11, col=theme.TEXT_FAINT, mono=True)


def draw_settings(self, W, H):
    """Settings screen: volume, font size, resolution, windowed/fullscreen."""
    sc = self.screen
    self.buttons = {}

    x = 80
    y = int(H * 0.10)
    widgets.eyebrow(sc, "settings", x, y, theme.CYAN)
    sc.blit(theme.font(42, bold=True).render("SETTINGS", True, theme.TEXT), (x - 2, y + 12))
    widgets.label(sc, "Changes are saved when you go back to the menu.",
                  x, y + 62, size=13, col=theme.TEXT_DIM)
    y += 118

    # --- Volume
    widgets.eyebrow(sc, "music volume", x, y)
    vol = self.settings["volume"]
    bvm = widgets.Button((x,       y + 22, 48, 32), "Vol -", size=13)
    bvp = widgets.Button((x + 274, y + 22, 48, 32), "Vol +", size=13)
    self.buttons["vol_down"] = bvm
    self.buttons["vol_up"]   = bvp
    bvm.draw(sc); bvp.draw(sc)
    bar = pygame.Rect(x + 56, y + 34, 214, 8)
    pygame.draw.rect(sc, theme.PANEL, bar, border_radius=4)
    pygame.draw.rect(sc, theme.CYAN,
                     (bar.x, bar.y, int(bar.w * vol), bar.h), border_radius=4)
    widgets.label(sc, f"{int(vol * 100):3d}%", x + 332, y + 28,
                  size=13, col=theme.TEXT_DIM, mono=True)
    y += 82

    # --- Font size
    widgets.eyebrow(sc, "font size", x, y)
    bfs = widgets.Button((x,       y + 22, 110, 32), "Small", size=13)
    bfl = widgets.Button((x + 118, y + 22, 110, 32), "Large", size=13)
    bfs.active = self.settings["font_mode"] == "small"
    bfl.active = self.settings["font_mode"] == "large"
    self.buttons["font_small"] = bfs
    self.buttons["font_large"] = bfl
    bfs.draw(sc); bfl.draw(sc)
    widgets.label(sc, "large adds ~22% to every label",
                  x + 240, y + 30, size=11, col=theme.TEXT_FAINT, mono=True)
    y += 82

    # --- Resolution (windowed only)
    widgets.eyebrow(sc, "resolution (windowed only)", x, y)
    for i, (rw, rh) in enumerate(RESOLUTIONS):
        b = widgets.Button((x + i * 132, y + 22, 124, 32), f"{rw}x{rh}", size=13)
        b.active  = (self.settings["resolution"] == i
                     and not self.settings["fullscreen"])
        b.enabled = not self.settings["fullscreen"]
        self.buttons[f"res{i}"] = b
        b.draw(sc)
    y += 82

    # --- Display mode
    widgets.eyebrow(sc, "display mode", x, y)
    bw = widgets.Button((x,       y + 22, 120, 32), "Windowed",   size=13)
    bf = widgets.Button((x + 128, y + 22, 120, 32), "Fullscreen", size=13)
    bw.active = not self.settings["fullscreen"]
    bf.active =     self.settings["fullscreen"]
    self.buttons["windowed"]   = bw
    self.buttons["fullscreen"] = bf
    bw.draw(sc); bf.draw(sc)
    y += 96

    # --- Back
    bb = widgets.Button((x, y, 140, 40), "Back", key="ESC", size=15)
    bb.active = True
    self.buttons["back"] = bb
    bb.draw(sc)


# ===========================================================================
# LOADING -- target simulation runs in a background thread
# ===========================================================================

def draw_loading(self, W, H):
    """Progress bar while the hidden target polymer equilibrates."""
    sc = self.screen
    self.buttons = {}
    p  = self._work_progress[0]
    cx, cy = W // 2, H // 2
    widgets.eyebrow(sc, "running the hidden ground truth", cx - 160, cy - 60, theme.CYAN)
    widgets.label(sc, LEVELS[self.sel_level].name, cx, cy - 40, size=26,
                  col=theme.TEXT, bold=True, center=True)
    widgets.label(sc, LEVELS[self.sel_level].blurb, cx, cy - 6, size=13,
                  col=theme.TEXT_DIM, center=True)
    bar = pygame.Rect(cx - 220, cy + 28, 440, 6)
    pygame.draw.rect(sc, theme.PANEL, bar, border_radius=3)
    pygame.draw.rect(sc, theme.GREEN, (bar.x, bar.y, int(bar.w * p), bar.h),
                     border_radius=3)
    widgets.label(sc, f"{int(p * 100):3d}%   equilibrating the target polymer",
                  cx, cy + 44, size=11, col=theme.TEXT_FAINT, mono=True, center=True)


# ===========================================================================
# PLAY SESSION -- normal single-player and versus game screens
# ===========================================================================

def draw_play(self, W, H):
    """Main play screen: 3D viewport + two heatmaps + E1 tracks + scores."""
    sc = self.screen
    s  = self.session
    self.buttons = {}
    R  = self.rects

    self.draw_header(W)
    self.draw_footer(W, H)

    # Heatmaps first: hovering a bin cross-highlights the beads in 3D.
    t = s.target
    self.hm_tgt.layout(R["tgt"], t.n)
    self.hm_sim.layout(R["sim"], t.n)
    mx, my = self.mouse
    hb = self.hm_sim.bin_at(mx, my) or self.hm_tgt.bin_at(mx, my)
    rb = (self.ribbon_bin(mx)
          if self.rects.get("ribbon") and self.rects["ribbon"].collidepoint(mx, my)
          else None)
    self.view.mark = (hb if hb and hb[0] != hb[1] else
                      ((rb,) if rb is not None else ()))

    # 3D viewport
    widgets.panel(sc, R["view"], fill=theme.INK_2)
    self.view.draw(sc, R["view"], s.poly.pos, s.poly.types, s.poly.loops,
                   s.poly.box, e1=None, t=time.time() - self.t0,
                   mouse=self.mouse)
    self.draw_view_hud(R["view"])

    # Heatmaps: in hard mode the target loses its loop-anchor overlay.
    if self._scale is None:
        self._scale = widgets.Scale(t.P, t.C)
    self.hm_tgt.draw(sc, t.P, t.C, s.map_mode,
                     loops_true=(None if s.hard else t.loops),
                     subtitle=("hard mode -- no ground truth overlay"
                               if s.hard else "what you must reproduce"),
                     accent=(theme.POOR if s.hard else theme.AMBER),
                     mouse=self.mouse, scale=self._scale)
    self.hm_sim.draw(sc, s.P_live, s.C_live, s.map_mode,
                     loops_player=s.poly.loops,
                     subtitle=f"ensemble at t={s.sim_time:6.1f}",
                     accent=theme.CYAN,
                     live=(self.state == PLAY and not s.paused),
                     mouse=self.mouse, scale=self._scale)

    # E1 tracks (target track hidden in hard mode -- it encodes the answer).
    g1, g2 = self.hm_tgt.grid, self.hm_sim.grid
    y = R["tgt"].bottom + 14
    if s.hard:
        ph = pygame.Rect(g1.x, y, g1.w, 34)
        pygame.draw.rect(sc, theme.PANEL, ph, border_radius=6)
        widgets.label(sc, "E1 target — hidden in hard mode",
                      ph.x + 10, ph.centery - 6,
                      size=10, col=theme.TEXT_FAINT, mono=True)
    else:
        widgets.eigen_track(sc, pygame.Rect(g1.x, y, g1.w, 34),
                            t.e1, title="E1 target")
    widgets.eigen_track(sc, pygame.Rect(g2.x, y, g2.w, 34),
                        s.e1_live, title="E1 yours")

    # Colour ribbon (always visible -- it shows the player's own colouring).
    yr     = y + 42
    ribbon = pygame.Rect(g2.x, yr, g2.w, 14)
    self.rects["ribbon"] = ribbon
    rb = (self.ribbon_bin(self.mouse[0])
          if ribbon.collidepoint(self.mouse) else None)
    widgets.type_track(sc, ribbon, s.poly.types,
                       hover=rb if rb is not None else self.view.hover)
    widgets.label(sc, "your colouring — click or drag to paint",
                  g2.x, yr + 17, size=9, col=theme.TEXT_FAINT, mono=True)
    if not s.hard:
        widgets.label(sc, "green dots = target anchors",
                      g1.x, yr + 2, size=9, col=theme.GREEN, mono=True)
    widgets.label(sc, "click a cell to tie that loop",
                  g2.x + g2.w, yr - 16, size=9, col=theme.CYAN,
                  mono=True, right=True)

    # Metric table
    mt = pygame.Rect(R["right"].x, yr + 34, R["right"].w, 180)
    widgets.metric_table(sc, mt, s.rep_live,
                         live=(self.state == PLAY and not s.paused))

    # Score display (different layout for solo vs versus)
    ys = mt.y + 190
    if s.two_player:
        hw = (R["right"].w - theme.PAD) // 2
        widgets.big_score(sc, pygame.Rect(R["right"].x, ys, hw, 56),
                          s.best[P_LOOP], "p1 best · loops", theme.GREEN)
        widgets.big_score(sc, pygame.Rect(R["right"].x + hw + theme.PAD, ys, hw, 56),
                          s.best[P_COMP], "p2 best · compartments", theme.MAGENTA)
        widgets.label(sc,
                      f"live  loop {s.rep_live['loop_score']:5.1f}   "
                      f"comp {s.rep_live['comp_score']:5.1f}",
                      R["right"].x, ys + 62,
                      size=11, col=theme.TEXT_FAINT, mono=True)
    else:
        widgets.big_score(sc, pygame.Rect(R["right"].x, ys, R["right"].w, 56),
                          s.rep["total"] if s.rep else s.rep_live["total"],
                          "score  ·  press enter to measure", theme.CYAN)
        rec = self.records.get(s.level.name, {}).get("best", 0.0)
        widgets.label(sc,
                      f"live {s.rep_live['total']:5.1f}   ·   record {rec:5.1f}"
                      f"   ·   {s.measurements} measurement(s)",
                      R["right"].x, ys + 62,
                      size=11, col=theme.TEXT_FAINT, mono=True)

    widgets.colorbars(sc,
                      pygame.Rect(R["right"].x, ys + 92, R["right"].w, 30),
                      self._scale, s.map_mode)


def draw_header(self, W):
    """Play-screen header: title, mode buttons, turn banner, measure button."""
    sc = self.screen
    s  = self.session
    r  = self.rects["header"]
    pygame.draw.rect(sc, theme.INK, r)
    pygame.draw.line(sc, theme.RULE, (0, r.bottom - 1), (W, r.bottom - 1), 1)

    S     = getattr(theme, "FONT_SCALE", 1.0)
    btn_h = int(32 * S)
    btn_y = (r.h - btn_h) // 2
    gap   = int(10 * S)
    edge  = int(16 * S)

    # Title + level name (hard mode flagged in red)
    widgets.label(sc, "THE CHROMATIN GAME", edge, int(9 * S),
                  size=13, col=theme.TEXT, bold=True)
    diff_tag = "  ·  HARD" if s.hard else ""
    widgets.label(sc, s.level.name + diff_tag, edge, int(29 * S),
                  size=11, col=theme.POOR if s.hard else theme.TEXT_FAINT)

    # Mode toggles (enabled/disabled depending on whose turn it is in versus)
    x       = int(220 * S)
    w_loops = int(108 * S)
    w_comp  = int(150 * S)
    w_view  = int(80 * S)
    bl = widgets.Button((x, btn_y, w_loops, btn_h),
                        "Loops", key="L", accent=theme.GREEN)
    bc = widgets.Button((x + w_loops + gap, btn_y, w_comp, btn_h),
                        "Compartments", key="C", accent=theme.CYAN)
    bv = widgets.Button((x + w_loops + gap + w_comp + gap, btn_y, w_view, btn_h),
                        "View", key="V")
    bl.active  = s.mode == "loop"
    bc.active  = s.mode == "comp"
    bl.enabled = s.allowed("loop")
    bc.enabled = s.allowed("comp")
    for k, b in (("loops", bl), ("comp", bc), ("view", bv)):
        self.buttons[k] = b
        b.draw(sc)

    modes_right = x + w_loops + gap + w_comp + gap + w_view

    # Turn banner (versus mode only)
    if s.two_player:
        acc  = PLAYER_ACCENT[s.turn]
        br_w = int(260 * S)
        br   = pygame.Rect(modes_right + int(20 * S), btn_y, br_w, btn_h)
        pygame.draw.rect(sc, theme.lerp_col(acc, theme.INK, 0.75), br, border_radius=6)
        pygame.draw.rect(sc, acc, br, 1, border_radius=6)
        widgets.label(sc,
                      f"ROUND {min(s.round, s.rounds)}/{s.rounds}  ·  "
                      f"{PLAYER_SHORT[s.turn]}",
                      br.x + int(12 * S), br.y + int(9 * S),
                      size=12, col=acc, mono=True, bold=True)
        tl = s.time_left()
        if tl is not None:
            col = theme.POOR if tl < 10 else theme.AMBER
            widgets.label(sc, f"{int(tl // 60)}:{int(tl % 60):02d}",
                          br.right + int(14 * S), br.y + int(8 * S),
                          size=15, col=col, mono=True, bold=True)

    # Right-side controls (laid out inward from the right edge)
    w_menu  = int(84 * S)
    w_help  = int(34 * S)
    w_meas  = int(140 * S)
    bm_x    = W - edge - w_menu
    bq_x    = bm_x - gap - w_help
    bmeas_x = bq_x - gap - w_meas
    bm    = widgets.Button((bm_x,    btn_y, w_menu, btn_h), "Menu",    key="ESC")
    bq    = widgets.Button((bq_x,    btn_y, w_help, btn_h), "?",       key="")
    bmeas = widgets.Button((bmeas_x, btn_y, w_meas, btn_h),
                           "End turn" if s.two_player else "Measure",
                           key="ENTER", accent=theme.AMBER)
    bmeas.active = True
    for k, b in (("menu", bm), ("helpb", bq), ("measure", bmeas)):
        self.buttons[k] = b
        b.draw(sc)
    if self.music.has_music:
        widgets.label(sc, "♪ " + self.music.title()[:28],
                      bmeas_x - gap, int(30 * S),
                      size=10, col=theme.TEXT_FAINT, right=True)


def draw_footer(self, W, H):
    """Play-screen footer: context-sensitive hint + camera controls."""
    sc = self.screen
    r  = self.rects["footer"]
    pygame.draw.rect(sc, theme.INK, r)
    pygame.draw.line(sc, theme.RULE, (0, r.y), (W, r.y), 1)
    s = self.session
    hint = (
        "LOOPS   click cell (i,j) on your map to tie that loop  ·  or click "
        "bead i then bead j in 3D  ·  click again to untie  ·  BACKSPACE clears"
        if s.mode == "loop" else
        "COMPARTMENTS   click or drag the colour ribbon to paint  ·  or click "
        "a bead in 3D to flip it  ·  watch E1 follow"
    )
    widgets.label(sc, hint, 16, r.y + 9, size=11, col=theme.TEXT_DIM)
    widgets.label(sc,
                  "drag polymer to move · drag space to orbit · wheel zoom · "
                  "SPACE pause · F fast-forward · R reset view",
                  W - 16, r.y + 9, size=11, col=theme.TEXT_FAINT, right=True)


def draw_view_hud(self, rect):
    """Overlay in the 3D viewport: polymer state box + legend."""
    sc = self.screen
    s  = self.session
    p  = s.poly
    nb = int((p.types == B_TYPE).sum())
    lines = [
        f"beads      {p.n}",
        f"loops      {len(p.loops)}",
        f"A red      {p.n - nb}",
        f"B blue     {nb}",
        f"Rg         {p.radius_of_gyration():5.2f}",
        f"box        {p.box:5.2f}",
    ]
    w, h = 116, 14 * len(lines) + 22
    r    = pygame.Rect(rect.x + 10, rect.y + 10, w, h)
    s2   = pygame.Surface(r.size, pygame.SRCALPHA)
    pygame.draw.rect(s2, (*theme.INK, 165), s2.get_rect(), border_radius=6)
    sc.blit(s2, r)
    widgets.eyebrow(sc, "state", r.x + 8, r.y + 6)
    for i, ln in enumerate(lines):
        widgets.label(sc, ln, r.x + 8, r.y + 20 + i * 14,
                      size=10, col=theme.TEXT_DIM, mono=True)

    # Legend (bottom-left of viewport)
    ly = rect.bottom - 58
    for i, (col, txt) in enumerate([
            (theme.COMP_A, "A · red · E1>0 · weak attraction"),
            (theme.COMP_B, "B · blue · E1<0 · strong attraction"),
            (theme.GREEN,  "loop bond · harmonic, non-consecutive")]):
        pygame.draw.circle(sc, col, (rect.x + 18, ly + i * 16 + 5), 5)
        widgets.label(sc, txt, rect.x + 30, ly + i * 16 - 2,
                      size=10, col=theme.TEXT_FAINT)
    if s.paused:
        widgets.label(sc, "PAUSED", rect.centerx, rect.y + 14,
                      size=14, col=theme.AMBER, bold=True, center=True)


def draw_settle_overlay(self, W, H):
    """Semi-transparent overlay shown while the score measurement runs."""
    sc = self.screen
    veil = pygame.Surface((W, H), pygame.SRCALPHA)
    veil.fill((*theme.INK, 190))
    sc.blit(veil, (0, 0))
    cx, cy = W // 2, H // 2
    p = self._work_progress[0]
    widgets.eyebrow(sc, "measuring", cx - 40, cy - 54, theme.AMBER)
    widgets.label(sc, "Equilibrating, then averaging the contact map",
                  cx, cy - 32, size=17, col=theme.TEXT, center=True)
    widgets.label(sc,
                  f"{MEAS_SAMPLES} conformations · a single structure is not a Hi-C map",
                  cx, cy - 8, size=12, col=theme.TEXT_DIM, center=True)
    bar = pygame.Rect(cx - 200, cy + 22, 400, 6)
    pygame.draw.rect(sc, theme.PANEL, bar, border_radius=3)
    pygame.draw.rect(sc, theme.AMBER, (bar.x, bar.y, int(bar.w * p), bar.h),
                     border_radius=3)


def draw_results(self, W, H):
    """Results screen shown after the final round (solo or versus)."""
    sc = self.screen
    s  = self.session
    self.buttons = {}
    cx = W // 2
    y  = H // 2 - 190

    widgets.eyebrow(sc, "final call", cx - 34, y, theme.CYAN)
    widgets.label(sc, s.level.name, cx, y + 16, size=30,
                  col=theme.TEXT, bold=True, center=True)

    if s.two_player:
        win = P_LOOP if s.best[P_LOOP] > s.best[P_COMP] else P_COMP
        tie = abs(s.best[P_LOOP] - s.best[P_COMP]) < 0.05
        widgets.label(sc, "Draw" if tie else f"{PLAYER_NAME[win]} wins",
                      cx, y + 58, size=22,
                      col=theme.TEXT if tie else PLAYER_ACCENT[win],
                      center=True, bold=True)
        for i in (P_LOOP, P_COMP):
            x = cx - 320 + i * 340
            r = pygame.Rect(x, y + 100, 300, 92)
            widgets.panel(sc, r, fill=theme.INK_2)
            widgets.eyebrow(sc, PLAYER_NAME[i], r.x + 12, r.y + 10, PLAYER_ACCENT[i])
            f   = theme.font(44, mono=True, bold=True)
            img = f.render(f"{s.best[i]:.1f}", True,
                           theme.score_color(s.best[i] / 100))
            sc.blit(img, (r.x + 12, r.y + 28))
            hs = "  ".join(f"{v:.0f}" for v in s.history[i]) or "-"
            widgets.label(sc, f"rounds  {hs}", r.x + 12, r.y + 74,
                          size=10, col=theme.TEXT_FAINT, mono=True)
        widgets.label(sc,
                      "The loop player is scored on corner enrichment and anchor accuracy; "
                      "the compartment player on E1 and the checkerboard.",
                      cx, y + 208, size=12, col=theme.TEXT_DIM, center=True)
        widgets.label(sc, "You were both folding the same polymer.",
                      cx, y + 228, size=12, col=theme.TEXT_FAINT, center=True)
    else:
        widgets.label(sc, f"{s.best_total:.1f}", cx, y + 60, size=64,
                      col=theme.score_color(s.best_total / 100),
                      center=True, bold=True)

    b = widgets.Button((cx - 80, y + 270, 160, 40), "Back to menu", key="ENTER")
    b.active = True
    self.buttons["back"] = b
    b.draw(sc)


def draw_help(self, W, H):
    """Overlay help panel -- appears on top of whatever screen is active."""
    sc = self.screen
    veil = pygame.Surface((W, H), pygame.SRCALPHA)
    veil.fill((*theme.INK, 225))
    sc.blit(veil, (0, 0))
    r = pygame.Rect(W // 2 - 430, H // 2 - 280, 860, 560)
    widgets.panel(sc, r, fill=theme.PANEL)
    x = r.x + 34
    y = r.y + 26
    widgets.eyebrow(sc, "how to play", x, y, theme.CYAN)
    widgets.label(sc, "You are given a Hi-C map. Build the polymer that makes it.",
                  x, y + 16, size=19, col=theme.TEXT, bold=True)

    blocks = [
        ("the map", theme.AMBER, [
            "Lower triangle: contact frequency. Upper triangle: O/E correlation.",
            "Green dots mark the target's loop anchors. Bin numbers run along both edges.",
            "The red/blue checkerboard is compartments. E1 is the first eigenvector",
            "of the correlation matrix -- red above the line, blue below. Press V to switch view.",
        ]),
        ("your two moves", theme.GREEN, [
            "LOOPS (L)  -- see a green dot at (i,j)? Click that same cell on YOUR map.",
            "             It ties a harmonic bond between two non-consecutive beads and pulls",
            "             the segment between them into a TAD. Anchors must be >= 3 apart.",
            "             You can also click bead i then bead j directly in 3D.",
            "COMPARTMENTS (C) -- paint the ribbon under your map, or click beads in 3D.",
            "             Red A attracts A weakly; blue B attracts B strongly; A and B repel.",
            "             Blue blocks collapse into a core and drive the checkerboard.",
        ]),
        ("the physics", theme.CYAN, [
            "Overdamped Langevin. Stiff harmonic backbone, soft excluded volume (chains can",
            "cross), Gaussian block-copolymer attraction, soft box walls. Your map is an",
            "ensemble average over simulation time -- one structure is never a Hi-C map.",
        ]),
        ("scoring", theme.MAGENTA, [
            "ENTER runs a long measurement and locks in a score. SCC is the headline number.",
            "Versus: P1 only places loops and is scored on APA + anchor F1; P2 only paints",
            "colours and is scored on E1 + checkerboard. Same polymer, different report cards.",
        ]),
    ]
    y += 54
    for title, col, lines in blocks:
        widgets.eyebrow(sc, title, x, y, col)
        for i, ln in enumerate(lines):
            widgets.label(sc, ln, x + 4, y + 16 + i * 16,
                          size=12, col=theme.TEXT_DIM)
        y += 22 + len(lines) * 16 + 12

    widgets.label(sc,
                  "L loops   C compartments   V map view   SPACE pause   F fast-forward   "
                  "R reset view   ENTER measure   M mute   N next track   ESC back",
                  x, r.bottom - 34, size=11, col=theme.TEXT_FAINT, mono=True)
    widgets.label(sc, "H or ESC to close", r.right - 34, r.y + 26,
                  size=11, col=theme.TEXT_FAINT, right=True)


# ===========================================================================
# MiNI-LAB -- free-play sandbox with live parameter editing
# ===========================================================================

def compute_lab_layout(self, W, H):
    """Compute rects for the Lab screen: header/footer/3D/maps/param panel.

    Separate from compute_layout because the Lab has a fundamentally different
    arrangement: no right-column heatmap pair, instead a scrollable parameter
    panel on the right and stacked 3D + two heatmaps on the left.
    """
    S     = getattr(theme, "FONT_SCALE", 1.0)
    P     = int(theme.PAD * S)
    hdr_h = int(theme.HEADER_H * S)
    ftr_h = int(theme.FOOTER_H * S)
    body  = pygame.Rect(0, hdr_h, W, H - hdr_h - ftr_h)

    panel_w = max(340, int(W * 0.27))
    left    = pygame.Rect(P, body.y + P, W - panel_w - P, body.h - 2 * P)
    panel   = pygame.Rect(left.right + P, body.y + P, panel_w - 2 * P, body.h - 2 * P)

    view_h  = int(left.h * 0.55)
    view    = pygame.Rect(left.x, left.y, left.w, view_h)
    maps_y  = view.bottom + P
    hm_w    = (left.w - P) // 2
    hm_h    = left.bottom - maps_y

    self.rects = {
        "header": pygame.Rect(0, 0, W, hdr_h),
        "footer": pygame.Rect(0, H - ftr_h, W, ftr_h),
        "view":   view,
        "map_a":  pygame.Rect(left.x, maps_y, hm_w, hm_h),
        "map_b":  pygame.Rect(left.x + hm_w + P, maps_y, hm_w, hm_h),
        "panel":  panel,
    }


def draw_lab(self, W, H):
    """Chromatin MiNI-Lab screen: live polymer + two heatmaps + param panel."""
    sc  = self.screen
    lab = self.lab
    self.buttons = {}
    self.compute_lab_layout(W, H)
    R = self.rects

    # Header -- minimal controls, no mode toggles (loops+compartments always active)
    r = R["header"]
    pygame.draw.rect(sc, theme.INK, r)
    pygame.draw.line(sc, theme.RULE, (0, r.bottom - 1), (W, r.bottom - 1), 1)
    widgets.label(sc, "CHROMATIN MiNI-LAB", 16, 9, size=13,
                  col=theme.ROYAL_RED, bold=True)
    widgets.label(sc, "free-play sandbox — every parameter is live", 16, 27,
                  size=11, col=theme.TEXT_FAINT)
    bp  = widgets.Button((260, 12, 100, 32),
                         "Pause" if not lab.paused else "Resume", key="SPACE")
    bp.active = lab.paused
    br  = widgets.Button((368, 12, 100, 32), "Randomize")
    brs = widgets.Button((476, 12,  90, 32), "Reset")
    bx  = widgets.Button((W - 120, 12, 104, 32), "Exit Lab", key="ESC")
    for k, b in (("lab_pause", bp), ("lab_rand", br),
                 ("lab_reset", brs), ("lab_exit", bx)):
        self.buttons[k] = b
        b.draw(sc)

    # Footer
    fr = R["footer"]
    pygame.draw.rect(sc, theme.INK, fr)
    pygame.draw.line(sc, theme.RULE, (0, fr.y), (W, fr.y), 1)
    widgets.label(sc,
                  "click any cell on a map to tie/untie a loop  ·  "
                  "paint the ribbon to set compartments  ·  "
                  "shift+click a bead in 3D to flip it",
                  16, fr.y + 9, size=11, col=theme.TEXT_DIM)

    # 3D viewport -- positions sanitised in case of a diverged integrator
    widgets.panel(sc, R["view"], fill=theme.INK_2)
    pos_safe = np.nan_to_num(lab.poly.pos,
                             nan=0.0, posinf=lab.poly.box, neginf=-lab.poly.box)
    self.lab_view.draw(sc, R["view"], pos_safe, lab.poly.types, lab.poly.loops,
                       lab.poly.box, t=time.time() - self.t0, mouse=self.mouse)

    # Two heatmaps: contact frequency (Reds) + O/E correlation (coolwarm)
    self.hm_lab_a.layout(R["map_a"], lab.poly.n)
    self.hm_lab_b.layout(R["map_b"], lab.poly.n)
    self.hm_lab_a.draw(sc, lab.P_live, lab.C_live, "contact",
                       loops_player=lab.poly.loops,
                       subtitle="live contact frequency",
                       accent=theme.ROYAL_RED, live=not lab.paused,
                       mouse=self.mouse, scale=lab.scale)
    self.hm_lab_b.draw(sc, lab.P_live, lab.C_live, "corr",
                       loops_player=lab.poly.loops,
                       subtitle="O/E correlation",
                       accent=theme.CYAN, live=not lab.paused,
                       mouse=self.mouse, scale=lab.scale)

    # Colour ribbon spans the full left column width
    ry     = R["map_a"].bottom + int(10 * getattr(theme, "FONT_SCALE", 1.0))
    ribbon = pygame.Rect(R["view"].x, ry, R["view"].w, 14)
    self.rects["ribbon"] = ribbon
    widgets.type_track(sc, ribbon, lab.poly.types, hover=self.lab_view.hover)

    # Scrollable parameter panel on the right
    self.draw_lab_panel(R["panel"])


def draw_lab_panel(self, panel_rect):
    """Scrollable parameter panel for the MiNI-Lab.

    Layout (top to bottom, not scrolled):
      - Bead count: eyebrow label + min/max hint + text field (always visible)
      - Stability warning (shown when dt·k_bond/γ > 0.5)
      - Grouped sliders, one per SimParams field (scrollable)
    """
    sc  = self.screen
    lab = self.lab
    S   = getattr(theme, "FONT_SCALE", 1.0)
    widgets.panel(sc, panel_rect, fill=theme.PANEL)

    x     = panel_rect.x + 16
    row_h = int(40 * S)
    self.lab_sliders = {}

    # --- Bead count (pinned at top, not part of the scroll region)
    from .physics import N_MIN, N_MAX
    n = lab.poly.n
    widgets.eyebrow(sc, f"beads: {n}", x, panel_rect.y + 14, theme.ROYAL_RED)
    widgets.label(sc, f"min {N_MIN} · max {N_MAX}", x, panel_rect.y + 30,
                  size=10, col=theme.TEXT_FAINT, mono=True)
    if self.lab_bead_field is None or self.lab_bead_field.value != n:
        self.lab_bead_field = widgets.TextField(
            (panel_rect.right - 100, panel_rect.y + 6, 80, 30), n, N_MIN, N_MAX)
    self.lab_bead_field.rect.topleft = (panel_rect.right - 100, panel_rect.y + 6)
    self.lab_bead_field.draw(sc)

    # --- Scrollable content starts here
    y        = panel_rect.y + 14 + int(46 * S) - self.lab_scroll
    clip_top = y

    # Stability guardrail -- warns before things explode
    ratio = lab.params.dt * lab.params.k_bond / lab.params.gamma
    if ratio > 0.5:
        widgets.label(sc,
                      f"⚠ dt·k_bond/γ = {ratio:.2f} -- integrator may be unstable",
                      x, y, size=11, col=theme.POOR, bold=True)
        y += int(20 * S)

    # Parameter groups: Langevin / Backbone / Loops / EV / Compartments / Confinement
    for group_name, fields in LAB_PARAM_GROUPS:
        widgets.eyebrow(sc, group_name, x, y, theme.TEXT_DIM)
        y += int(20 * S)
        for key, label, lo, hi, step, fmt, is_int in fields:
            val = getattr(lab.params, key)
            r   = pygame.Rect(x, y + int(14 * S), panel_rect.w - 32, int(8 * S))
            sl  = widgets.Slider(r, label, val, lo, hi,
                                 step=step, fmt=fmt, integer=is_int)
            sl.key = key
            self.lab_sliders[key] = sl
            # Only draw rows that are actually visible inside the panel.
            if panel_rect.y <= y <= panel_rect.bottom:
                sl.draw(sc)
            y += row_h
        y += int(10 * S)

    # Track total content height so the scroll handler knows the limit.
    self._lab_panel_content_h = y - clip_top + self.lab_scroll