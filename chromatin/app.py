"""The Chromatin Game -- game states, layout and interaction."""
from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
import pygame

from . import analysis as an
from . import theme, widgets
from .audio import Music
from .levels import LEVELS, build_target
from .physics import A_TYPE, B_TYPE, MIN_LOOP_SPAN, SimParams, SimulationUnstable
from .physics_jax import FastPolymer   # same engine as Session/LabSession -- see physics_jax.py
from .render3d import Camera
from .polymer_view import create_polymer_view
from . import beadcolor, chainshape
from .session import Session, LabSession
from .manual_panel import ManualPanel
from .constants import *

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSIC_DIR = os.path.join(ROOT, "music")
RECORDS_PATH = os.path.join(ROOT, "records.json")

SETTINGS_PATH = os.path.join(ROOT, "settings.json")

RESOLUTIONS = [(1280, 800), (1600, 900), (1920, 1080)]

DEFAULT_SETTINGS = {
    "volume": 1.0,
    "font_mode": "small",     # "small" | "large"
    "resolution": 1,           # index into RESOLUTIONS
    "fullscreen": False,
    "theme": theme.DEFAULT_THEME,   # light by default; see theme.THEME_ORDER
}

def load_records() -> dict:
    try:
        with open(RECORDS_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_records(rec: dict) -> None:
    try:
        with open(RECORDS_PATH, "w") as fh:
            json.dump(rec, fh, indent=2)
    except Exception:
        pass

def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as fh:
            return {**DEFAULT_SETTINGS, **json.load(fh)}
    except Exception:
        return dict(DEFAULT_SETTINGS)


def save_settings(s: dict) -> None:
    try:
        with open(SETTINGS_PATH, "w") as fh:
            json.dump(s, fh, indent=2)
    except Exception:
        pass

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("The Chromatin Game")
        self.settings = load_settings()
        theme.set_theme(self.settings["theme"])
        self.screen = None
        self.apply_display_settings()
        self.clock = pygame.time.Clock()
        self.music = Music(MUSIC_DIR)
        self.music.set_volume(self.settings["volume"])
        self.records = load_records()

        self.state = MENU
        self.running = True
        self.t0 = time.time()
        self.view = create_polymer_view()   # GPU (Panda3D) if available, else software
        self.hm_tgt = widgets.Heatmap("experimental  /  target")
        self.hm_sim = widgets.Heatmap("simulated  /  yours")

        self.show_manual = False
        self.manual = ManualPanel()

        # menu selections
        self.sel_hard = False
        self.sel_level = 0
        self.sel_two = False
        self.sel_rounds = 3
        self.sel_turn_sec = 90
        self.sel_seed = int(np.random.default_rng().integers(1, 9999))
        self.show_help = False

        self.session: Session | None = None
        self._scale: widgets.Scale | None = None
        self.toast = ""
        self.toast_t = 0.0

        self.lab: LabSession | None = None
        self.lab_view = create_polymer_view()
        self.hm_lab_a = widgets.Heatmap("contact  ·  reds")
        self.hm_lab_b = widgets.Heatmap("correlation  ·  coolwarm")
        self.lab_sliders: dict[str, widgets.Slider] = {}
        self.lab_bead_slider: widgets.Slider | None = None
        self.lab_scroll = 0
        
        # workers
        self._work_progress = [0.0]
        self._work_result = [None]
        self._work_thread: threading.Thread | None = None

        # menu ambience -- FastPolymer too, so every simulation in the game
        # (menu, Lab, Play) runs on the exact same JAX-fused engine.
        self.demo = FastPolymer(30, SimParams(), seed=5)
        rng = np.random.default_rng(3)
        t = np.empty(30, dtype=np.int8)
        i, cur = 0, A_TYPE
        while i < 30:
            L = int(rng.integers(4, 8))
            t[i:i + L] = cur
            cur = -cur
            i += L
        self.demo.load_config(t, [(3, 12), (15, 26)])
        self.demo.step(400)
        self.demo.step(6)   # pre-compile the per-frame step count too (see update())
        self.demo_view = create_polymer_view()
        self.demo_view.cam.dist = 13.0

        # interaction
        self.mouse: tuple[int, int] = (0, 0)
        self.press: tuple[int, tuple[int, int]] | None = None
        self.orbiting = False
        self.dragged = False
        self.paint: int | None = None
        self.buttons: dict[str, widgets.Button] = {}

    
    # ============================================================== buttons
    def on_button(self, b: widgets.Button):
        n = b.text
        if self.state == MENU:
            if n == "Start":
                self.start_session()
            elif n == "Solo":
                self.sel_two = False
            elif n == "Versus":
                self.sel_two = True
            elif n == "How to play":
                self.show_help = True
            elif n == "Chromatin MiNI-Lab":
                self.start_lab()
            elif n == "Quit":
                self.running = False
            elif n == "Settings":
                self.state = SETTINGS
            elif n == "Manual":
                self.show_manual = True
                self.manual.select(0)
            elif n.startswith("Rounds"):
                self.sel_rounds = self.sel_rounds % 5 + 1
            elif n.startswith("Turn"):
                opts = [60, 90, 120, 0]
                i = opts.index(self.sel_turn_sec) if self.sel_turn_sec in opts else 1
                self.sel_turn_sec = opts[(i + 1) % len(opts)]
            elif n == "Shuffle seed":
                self.sel_seed = int(np.random.default_rng().integers(1, 9999))
            elif n == "Easy":
                self.sel_hard = False
            elif n == "Hard":
                self.sel_hard = True
            elif n in [l.name for l in LEVELS]:
                self.sel_level = [l.name for l in LEVELS].index(n)
                self.sel_seed = int(np.random.default_rng().integers(1, 9999))

        elif self.state == PLAY:
            s = self.session
            if n == "View":
                i = widgets.MODES.index(s.map_mode)
                s.map_mode = widgets.MODES[(i + 1) % len(widgets.MODES)]
            elif n in ("Measure", "End turn"):
                self.begin_measure()
            elif n == "Menu":
                self.state = MENU
                self.session = None
            elif n == "?":
                self.show_help = True
            elif n == "Analysis":
                s.show_analysis = not s.show_analysis
                # if s.show_analysis:
                #     s.analysis_panel.reset()
            elif n.startswith("Colour:"):
                mode = self.view.cycle_color_mode()
                self.say(f"colour mode: {beadcolor.MODE_LABEL[mode]}")
            elif n.startswith("Shape:"):
                mode = self.view.cycle_rep_mode()
                self.say(f"shape: {chainshape.REP_LABEL[mode]}")

        elif self.state == RESULTS:
            if n == "Back to menu":
                self.state = MENU
                self.session = None

        elif self.state == SETTINGS:
            if n == "Back":
                save_settings(self.settings)
                self.state = MENU
            elif n == "Vol -":
                self.settings["volume"] = max(0.0, self.settings["volume"] - 0.05)
                self.music.set_volume(self.settings["volume"])
            elif n == "Vol +":
                self.settings["volume"] = min(1.0, self.settings["volume"] + 0.05)
                self.music.set_volume(self.settings["volume"])
            elif n in ("Small", "Large"):
                self.settings["font_mode"] = n.lower()
                theme.set_font_scale(self.settings["font_mode"])
            elif n.startswith("Theme:"):
                name = n.split(":", 1)[1]
                self.settings["theme"] = name
                theme.set_theme(name)
            elif n in ("Play music", "Pause music", "Resume music"):
                self.music.toggle_pause()
            elif n == "Windowed":
                self.settings["fullscreen"] = False
                self.apply_display_settings()
            elif n == "Fullscreen":
                self.settings["fullscreen"] = True
                self.apply_display_settings()
            else:
                # Resolution buttons — button text is "WIDTHxHEIGHT"
                for i, (rw, rh) in enumerate(RESOLUTIONS):
                    if n == f"{rw}x{rh}":
                        self.settings["resolution"] = i
                        self.settings["fullscreen"] = False
                        self.apply_display_settings()
                        break

        elif self.state == LAB:
            lab = self.lab
            if n == "Exit Lab":
                self.state = MENU
                self.lab = None
            elif n == "Pause":
                lab.paused = not lab.paused
            elif n == "Randomize":
                lab.randomize_all()
            elif n == "Reset":
                self.lab = LabSession(n=lab.poly.n,
                                      seed=int(np.random.default_rng().integers(1, 9999)))
            elif n == "+1 bead":
                lab.set_bead_count(lab.poly.n + 1)
            elif n == "-1 bead":
                lab.set_bead_count(lab.poly.n - 1)
            elif n == "+10":
                lab.set_bead_count(lab.poly.n + 10)
            elif n == "-10":
                lab.set_bead_count(lab.poly.n - 10)
            elif n == "Analysis":
                lab.show_analysis = not lab.show_analysis
                # if lab.show_analysis:
                #     lab.analysis_panel.reset()
            elif n.startswith("Colour:"):
                mode = self.lab_view.cycle_color_mode()
                self.say(f"colour mode: {beadcolor.MODE_LABEL[mode]}")
            elif n.startswith("Shape:"):
                mode = self.lab_view.cycle_rep_mode()
                self.say(f"shape: {chainshape.REP_LABEL[mode]}")

    def apply_display_settings(self) -> None:
        """(Re)create the window with the current settings and apply font scale."""
        theme.set_font_scale(self.settings["font_mode"])
        if self.settings["fullscreen"]:
            self.screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        else:
            w, h = RESOLUTIONS[self.settings["resolution"]]
            self.screen = pygame.display.set_mode(
                (w, h), pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.event.clear()   # drop spurious events from the mode switch

    # ================================================================ toast
    def say(self, msg: str):
        self.toast = msg
        self.toast_t = time.time()

    # ================================================================ frame
    def run(self):
        while self.running:
            # let the worker thread have the CPU while it equilibrates
            fps = 30 if self.state in (LOADING, SETTLE) else 60
            dt = self.clock.tick(fps) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
            pygame.display.flip()
        self.music.stop()
        pygame.quit()

    def _get_close_rect(self, W: int, H: int) -> pygame.Rect:
        mw = int(W * 0.92)
        mh = int(H * 0.90)
        mx = (W - mw) // 2
        my = (H - mh) // 2
        close_f   = theme.font(16, bold=True)
        close_img = close_f.render("×", True, theme.TEXT_FAINT)   # match draw() exactly
        return pygame.Rect(
            mx + mw - close_img.get_width() - self.PAD,
            my + self.PAD - 2,
            close_img.get_width() + 8,
            close_img.get_height() + 4)

    def handle_events(self):
        for ev in pygame.event.get():
            # ---- pygame.QUIT must always be handled, even if the manual
            # is open — this is the OS window close button (the X in the
            # window title bar), not the in-game manual close button.
            if ev.type == pygame.QUIT:
                self.running = False
                continue

            # ---- Manual overlay consumes ALL other events while open.
            if self.show_manual:
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self.show_manual = False
                elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEWHEEL):
                    W, H = self.screen.get_size()
                    result = self.manual.handle_event(ev, W, H)
                    if result == "close":
                        self.show_manual = False
                continue

            # ---- Normal event routing (manual is closed) ------------------
            self.music.handle(ev)
            if ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                           pygame.MOUSEBUTTONUP):
                self.mouse = ev.pos

            if ev.type == pygame.VIDEORESIZE:
                w = max(theme.MIN_W, ev.w)
                h = max(theme.MIN_H, ev.h)
                self.screen = pygame.display.set_mode(
                    (w, h), pygame.RESIZABLE | pygame.DOUBLEBUF)

            elif ev.type == pygame.KEYDOWN:
                self.on_key(ev)

            # ---- Analysis panel Clear button (before game buttons so a
            # click on the panel doesn't also fire buttons beneath it)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if (self.state == PLAY
                        and self.session is not None
                        and self.session.show_analysis):
                    if self.session.analysis_panel.handle_click(ev.pos):
                        continue
                if (self.state == LAB
                        and self.lab is not None
                        and self.lab.show_analysis):
                    if self.lab.analysis_panel.handle_click(ev.pos):
                        continue

            # ---- Game buttons (header, footer, menu rows, etc.)
            for b in self.buttons.values():
                if b.handle(ev):
                    self.on_button(b)

            # ---- Play-screen mouse (heatmap clicks, ribbon, 3D orbit/drag)
            if self.state == PLAY and not self.show_help:
                self.on_play_mouse(ev)

            # ---- Lab-screen controls (sliders, bead count, 3D + heatmap)
            if self.state == LAB and self.lab is not None:
                for sl in self.lab_sliders.values():
                    if sl.handle(ev):
                        self.lab.apply_param(sl.key, sl.value)
                if self.lab_bead_slider is not None and self.lab_bead_slider.handle(ev):
                    self.lab.set_bead_count(int(self.lab_bead_slider.value))
                self.on_lab_mouse(ev)

    def handle_event(self, ev) -> str:
        """Returns 'close', 'consumed', or 'outside'.

        Callers should only close the manual on 'close'. Scroll and nav
        clicks return 'consumed'. Anything else returns 'consumed' too so
        the caller's unconditional continue fires and nothing leaks through.
        """
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self._close_rect and self._close_rect.collidepoint(ev.pos):
                return "close"
            for i, r in enumerate(self._nav_rects):
                if r.collidepoint(ev.pos):
                    self.select(i)
                    return "consumed"
            return "consumed"

        if ev.type == pygame.MOUSEWHEEL:
            self._scroll = max(0, min(
                self._scroll - ev.y * self.SCROLL_STEP,
                max(0, self._content_h)))
            return "consumed"

        return "consumed"

    def on_key(self, ev):
        k = ev.key

        # ----------------------------------------------------------------
        # ESC: close panels in priority order before changing state
        # ----------------------------------------------------------------
        if k == pygame.K_ESCAPE:
            if self.show_manual:
                self.show_manual = False
            elif self.show_help:
                self.show_help = False
            elif self.state == PLAY and self.session and self.session.show_analysis:
                # Close the analysis panel first; a second ESC exits to menu
                self.session.show_analysis = False
            elif self.state == LAB and self.lab and self.lab.show_analysis:
                self.lab.show_analysis = False
            elif self.state == SETTINGS:
                save_settings(self.settings)
                self.state = MENU
            elif self.state in (PLAY, RESULTS):
                self.state = MENU
                self.session = None
            elif self.state == LAB:
                self.state = MENU
                self.lab = None
            else:
                self.running = False
            return

        # ----------------------------------------------------------------
        # Global keys (work in any state)
        # ----------------------------------------------------------------
        if k in (pygame.K_h, pygame.K_F1):
            self.show_help = not self.show_help
            return
        if k == pygame.K_m:
            # one key for both jobs: first press starts the music,
            # every press after that just mutes / unmutes it
            if not self.music.started:
                self.music.toggle_pause()
                self.say(self.music.title())
            else:
                self.music.toggle_mute()
                self.say("music muted" if self.music.muted else "music on")
            return
        if k == pygame.K_n:
            self.music.next()
            self.say(self.music.title())
            return
        if k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.music.nudge_volume(+0.08)
            self.say(f"volume {int(self.music.volume * 100)}%")
            return
        if k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.music.nudge_volume(-0.08)
            self.say(f"volume {int(self.music.volume * 100)}%")
            return

        # ----------------------------------------------------------------
        # State-specific keys
        # ----------------------------------------------------------------
        if self.state == MENU:
            if k == pygame.K_RETURN:
                self.start_session()
            elif k == pygame.K_LEFT:
                self.sel_level = (self.sel_level - 1) % len(LEVELS)
            elif k == pygame.K_RIGHT:
                self.sel_level = (self.sel_level + 1) % len(LEVELS)
            return

        if self.state == RESULTS:
            if k == pygame.K_RETURN:
                self.state = MENU
                self.session = None
            return

        if self.state == LAB:
            lab = self.lab
            if k == pygame.K_SPACE:
                lab.paused = not lab.paused
            elif k == pygame.K_a:
                # A toggles the analysis panel (mirrors the header button)
                lab.show_analysis = not lab.show_analysis
                if lab.show_analysis:
                    lab.analysis_panel.reset()
            elif k == pygame.K_t:
                mode = self.lab_view.cycle_color_mode()
                self.say(f"colour mode: {beadcolor.MODE_LABEL[mode]}")
            elif k == pygame.K_g:
                mode = self.lab_view.cycle_rep_mode()
                self.say(f"shape: {chainshape.REP_LABEL[mode]}")
            return

        if self.state != PLAY:
            return

        # ----------------------------------------------------------------
        # PLAY keys
        # ----------------------------------------------------------------
        s = self.session

        if k == pygame.K_a:
            # A toggles the analysis panel (mirrors the header button)
            s.show_analysis = not s.show_analysis
            if s.show_analysis:
                s.analysis_panel.reset()
            return

        if k == pygame.K_v:
            i = widgets.MODES.index(s.map_mode)
            s.map_mode = widgets.MODES[(i + 1) % len(widgets.MODES)]
            self.say(f"map view: {widgets.MODE_LABEL[s.map_mode]}")
        elif k == pygame.K_SPACE:
            s.paused = not s.paused
            self.say("paused" if s.paused else "running")
        elif k == pygame.K_r:
            self.view.cam = Camera()
            self.view.cam.dist = s.poly.box * 3.4
            self.say("view reset")
        elif k == pygame.K_t:
            mode = self.view.cycle_color_mode()
            self.say(f"colour mode: {beadcolor.MODE_LABEL[mode]}")
        elif k == pygame.K_g:
            mode = self.view.cycle_rep_mode()
            self.say(f"shape: {chainshape.REP_LABEL[mode]}")
        elif k == pygame.K_f:
            s.poly.step(600)
            s.sim_time += 600 * s.poly.p.dt
            self.say("fast-forward")
        elif k == pygame.K_RETURN:
            self.begin_measure()
        elif k == pygame.K_BACKSPACE:
            if s.allowed("loop"):
                s.poly.clear_loops()
                self.say("all loops removed")

    # =========================================================== session flow
    def start_session(self):
        lvl = LEVELS[self.sel_level]
        self.session = Session(lvl, self.sel_seed, self.sel_two,
                            self.sel_rounds, self.sel_turn_sec, hard=self.sel_hard)
        self.state = LOADING
        self._work_progress = [0.0]
        self._work_result = [None]

        def worker():
            tgt = build_target(lvl, self.sel_seed,
                               progress=lambda p: self._work_progress.__setitem__(0, p))
            self._work_result[0] = tgt

        self._work_thread = threading.Thread(target=worker, daemon=True)
        self._work_thread.start()
        # music no longer autoplays -- the user starts it (Settings, or P)

    def start_lab(self) -> None:
        seed = int(np.random.default_rng().integers(1, 9999))
        self.lab = LabSession(n=40, seed=seed)
        self.lab_scroll = 0
        # Sliders are built once here and reused every frame in draw_lab_panel
        # (not rebuilt per-frame), so a fresh Lab session starts with a clean set.
        self.lab_sliders = {}
        self.lab_bead_slider = None
        self.state = LAB

    def begin_measure(self):
        s = self.session
        if self.state != PLAY:
            return
        self.state = SETTLE
        self._work_progress = [0.0]
        self._work_result = [None]
        poly = s.poly
        poly.grab = None

        def worker():
            total = MEAS_BURN + MEAS_SAMPLES * MEAS_EVERY
            done = 0
            while done < MEAS_BURN:
                poly.step(200)
                done += 200
                self._work_progress[0] = done / total
            acc = np.zeros((poly.n, poly.n))
            for i in range(MEAS_SAMPLES):
                poly.step(MEAS_EVERY)
                acc += poly.contacts()
                done += MEAS_EVERY
                if i % 16 == 0:
                    self._work_progress[0] = done / total
            self._work_result[0] = acc / MEAS_SAMPLES
            self._work_progress[0] = 1.0

        self._work_thread = threading.Thread(target=worker, daemon=True)
        self._work_thread.start()

    def finish_measure(self, P: np.ndarray):
        s = self.session
        p, t = s.poly, s.target
        oe, C, e1 = an.pipeline(P, p.types.astype(float))
        rep = an.evaluate(P, t.P, C, t.C, e1, t.e1, list(p.loops), list(t.loops),
                          calib=t.calib)
        s.rep = rep
        s.P_live = P.copy()
        s.C_live, s.e1_live = C, e1
        s.rep_live = rep
        s.measurements += 1
        s.best_total = max(s.best_total, rep["total"])

        if s.two_player:
            who = s.turn
            val = rep["loop_score"] if who == P_LOOP else rep["comp_score"]
            s.last[who] = val
            s.best[who] = max(s.best[who], val)
            s.history[who].append(val)
            self.say(f"{PLAYER_NAME[who]} scores {val:.1f}")
            if who == P_LOOP:
                s.turn = P_COMP
            else:
                s.turn = P_LOOP
                s.round += 1
            s.turn_start = time.time()
            self.view.pending = None
            if s.round > s.rounds:
                self.state = RESULTS
                return
            self.state = PLAY
        else:
            # Easy and hard mode keep separate records -- hard mode hides the
            key = s.level.name + ("  [hard]" if s.hard else "")
            rec = self.records.setdefault(key, {})
            prev = float(rec.get("best", 0.0))
            if rep["total"] > prev:
                rec["best"] = round(float(rep["total"]), 2)
                rec["seed"] = s.seed
                save_records(self.records)
                self.say(f"new record: {rep['total']:.1f}")
            else:
                self.say(f"measured: {rep['total']:.1f}  (record {prev:.1f})")
            self.state = PLAY

    # =============================================================== update
    def update(self, dt: float):
        if self.state == LOADING:
            if self._work_result[0] is not None:
                self._scale = None
                self.session.begin(self._work_result[0])
                self.view.cam.dist = self.session.poly.box * 3.4
                self.view.cam.center = np.zeros(3)
                self.state = PLAY
                self.session.turn_start = time.time()
            return

        if self.state == SETTLE:
            self.demo_spin()
            if self._work_result[0] is not None:
                self.finish_measure(self._work_result[0])
            return

        if self.state == MENU:
            self.demo.step(6)
            self.demo_view.cam.yaw += dt * 0.22
            return

        if self.state == LAB:
            if not self.show_help:
                try:
                    self.lab.step()
                except Exception as e:
                    self.lab.unstable = True
                    self.lab.unstable_reason = f"unexpected error: {e}"
                if self.lab.unstable:
                    self.say(f"simulation unstable: {self.lab.unstable_reason}")
                    self.state = MENU
                    self.lab = None
                    return
                # Always collect metrics so the history builds up from the
                # moment the Lab starts, not just while the panel is open.
                if self.lab is not None:
                    self.lab.analysis_panel.update(self.lab.poly)
            return

        if self.state != PLAY:
            return

        s = self.session
        if not s.paused and not self.show_help:
            s.poly.step(s.poly.p.steps_per_frame)
            s.sim_time += s.poly.p.steps_per_frame * s.poly.p.dt
            f = pygame.time.get_ticks() // 16
            if f % 2 == 0:
                c = s.poly.contacts()
                s.P_live *= (1.0 - LIVE_ALPHA)
                s.P_live += LIVE_ALPHA * c
            if f % 20 == 0:
                s.refresh_live()
            # Always collect metrics so opening the panel reveals the full
            # history since the simulation started, not just since it opened.
            s.analysis_panel.update(s.poly)

        tl = s.time_left()
        if s.two_player and tl is not None and tl <= 0.0:
            self.say("time up")
            self.begin_measure()

    def demo_spin(self):
        pass

from .draw import (
    draw, compute_layout, paint_background,
    draw_menu, draw_settings, draw_loading, draw_play,
    draw_lab, draw_lab_panel, compute_lab_layout,   # <-- compute_lab_layout added
    draw_view_hud, draw_view_controls, draw_header, draw_footer,
    draw_settle_overlay, draw_results, draw_help, draw_toast,
)
from .interact import (
    on_play_mouse, on_lab_mouse, map_click, click_bead,
    ribbon_bin, ribbon_bin_n,
)

for _fn in [
    draw, compute_layout, paint_background,
    draw_menu, draw_settings, draw_loading, draw_play,
    draw_lab, draw_lab_panel, compute_lab_layout,   # <-- added here too
    draw_view_hud, draw_view_controls, draw_header, draw_footer,
    draw_settle_overlay, draw_results, draw_help, draw_toast,
    on_play_mouse, on_lab_mouse, map_click, click_bead,
    ribbon_bin, ribbon_bin_n,
]:
    setattr(Game, _fn.__name__, _fn)

def main():
    Game().run()