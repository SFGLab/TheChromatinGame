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
from .physics import A_TYPE, B_TYPE, MIN_LOOP_SPAN, Polymer, SimParams
from .render3d import PolymerView

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
}

MENU, SETTINGS, LOADING, PLAY, SETTLE, RESULTS, LAB = "menu", "settings", "loading", "play", "settle", "results", "lab"

P_LOOP, P_COMP = 0, 1
PLAYER_NAME = ["Loop player", "Compartment player"]
PLAYER_SHORT = ["LOOPS", "COMPARTMENTS"]
PLAYER_ACCENT = [theme.GREEN, theme.MAGENTA]

# measurement ensemble (the score-of-record)
MEAS_BURN, MEAS_SAMPLES, MEAS_EVERY = 2500, 520, 20
LIVE_ALPHA = 0.006

# (SimParams field, display label, lo, hi, step, format, is_integer)
LAB_PARAM_GROUPS = [
    ("Langevin integrator", [
        ("dt",              "timestep  dt",       0.001, 0.02, 0.001, "{:.3f}", False),
        ("gamma",            "friction  γ",         0.1,  10.0, 0.1,   "{:.2f}", False),
        ("kT",               "temperature  kT",     0.1,   3.0, 0.1,  "{:.2f}", False),
        ("steps_per_frame",  "steps / frame",       5,   200,   5,     "{:d}",   True),
    ]),
    ("Backbone & bending", [
        ("b0",       "bond length  b0",     0.5,  2.0, 0.05, "{:.2f}", False),
        ("k_bond",   "bond stiffness",       20,  500,  5,   "{:.0f}", False),
        ("k_angle",  "bending  k_angle",     0.0, 100.0, 5, "{:.1f}", False),
    ]),
    ("Loops & the player's hand", [
        ("k_loop",   "loop stiffness",       5,   150,  5,   "{:.0f}", False),
        ("k_grab",   "hand stiffness",       10,  200,  5,   "{:.0f}", False),
    ]),
    ("Excluded volume", [
        ("ev_eps",   "EV height  ε",         5,   150,  5,   "{:.0f}", False),
        ("ev_rc",    "EV diameter  rc",      0.6,  2.0, 0.05, "{:.2f}", False),
    ]),
    ("Compartments (copolymer)", [
        ("sigma",      "attraction range σ",   0.5,  2.5, 0.1, "{:.2f}", False),
        ("eps_AA",     "A–A  ε",               0.0,  4.0, 0.25, "{:.2f}", False),
        ("eps_BB",     "B–B  ε",               0.0,  4.0, 0.25, "{:.2f}", False),
        ("eps_AB",     "A–B  ε",              -1.0,  1.0, 0.25, "{:.2f}", False),
        ("eps_domain", "loop-domain bonus",    0.0,  1.0, 0.25, "{:.2f}", False),
    ]),
    ("Confinement & contact call", [
        ("k_wall",     "wall stiffness",     5,   200,  5,    "{:.0f}", False),
        ("contact_rc", "contact cutoff",     0.8,  3.0, 0.1, "{:.2f}", False),
        ("contact_w",  "contact softness",   0.02, 0.5, 0.05, "{:.2f}", False),
    ]),
]

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

class Session:
    """One playthrough of one level."""

    def __init__(self, level, seed: int, two_player: bool, rounds: int, turn_seconds: int,
                hard: bool = False):
        self.level = level
        self.seed = seed
        self.two_player = two_player
        self.rounds = rounds
        self.turn_seconds = turn_seconds
        self.hard = hard

        self.target = None
        self.poly: Polymer | None = None
        self.mode = "loop"
        self.map_mode = "split"

        self.round = 1
        self.turn = P_LOOP
        self.turn_start = 0.0
        self.best = [0.0, 0.0]
        self.last = [0.0, 0.0]
        self.history: list[list[float]] = [[], []]

        self.P_live: np.ndarray | None = None
        self.C_live: np.ndarray | None = None
        self.e1_live: np.ndarray | None = None
        self.rep_live: an.Report | None = None
        self.rep: an.Report | None = None
        self.best_total = 0.0
        self.measurements = 0
        self.sim_time = 0.0
        self.paused = False

    # -------------------------------------------------------------- helpers
    def begin(self, target):
        self.target = target
        n = target.n
        self.poly = Polymer(n, SimParams(), seed=self.seed * 31 + 7)
        self.P_live = self.poly.contacts()
        self.refresh_live(force=True)
        self.turn_start = time.time()

    def allowed(self, kind: str) -> bool:
        if not self.two_player:
            return True
        return (kind == "loop") == (self.turn == P_LOOP)

    def refresh_live(self, force: bool = False):
        p = self.poly
        oe, C, e1 = an.pipeline(self.P_live, p.types.astype(float))
        self.C_live, self.e1_live = C, e1
        t = self.target
        self.rep_live = an.evaluate(self.P_live, t.P, C, t.C, e1, t.e1,
                                    list(p.loops), list(t.loops))

    def time_left(self) -> float | None:
        if not self.turn_seconds:
            return None
        return max(0.0, self.turn_seconds - (time.time() - self.turn_start))

class LabSession:
    """Chromatin MiNI-Lab: no target, no scoring -- just a live polymer whose
    every force-field parameter and bead count you can change on the fly."""

    def __init__(self, n: int, seed: int):
        self.seed = seed
        self.params = SimParams()
        self.poly = Polymer(n, self.params, seed=seed)

        rng = np.random.default_rng(seed)
        t = np.empty(n, dtype=np.int8)
        i, cur = 0, A_TYPE
        while i < n:
            L = int(rng.integers(3, 7))
            t[i:i + L] = cur
            cur = -cur
            i += L
        loops = [(2, min(n - 3, max(6, n // 2)))] if n >= 12 else []
        self.poly.load_config(t, loops)

        self.paused = False
        self.paint: int | None = None
        self.P_live = self.poly.contacts()
        self.C_live = None
        self.e1_live = None
        self.scale: widgets.Scale | None = None
        self._frame = 0
        self._refresh(rebuild_scale=True)

    def _refresh(self, rebuild_scale: bool = False) -> None:
        oe, C, e1 = an.pipeline(self.P_live, self.poly.types.astype(float))
        self.C_live, self.e1_live = C, e1
        # Rescale the colour range from the live map itself -- this is what
        # makes the heatmaps visibly "breathe" as you drag a slider. Throttled
        # to every few frames so the colours don't flicker every tick.
        if rebuild_scale or (self._frame % 6 == 0):
            self.scale = widgets.Scale(self.P_live, C)

    def step(self) -> None:
        if self.paused:
            return
        self.poly.step(self.poly.p.steps_per_frame)
        c = self.poly.contacts()
        self.P_live = 0.985 * self.P_live + 0.015 * c
        self._frame += 1
        self._refresh()

    def apply_param(self, key: str, value) -> None:
        import dataclasses
        self.params = dataclasses.replace(self.params, **{key: value})
        self.poly.set_params(self.params)

    def randomize_all(self) -> None:
        """Reroll compartments and loops only. Force-field parameters are left
        untouched, so the current physics stays fixed while you explore
        different starting configurations under it."""
        n = self.poly.n
        rng = np.random.default_rng(np.random.default_rng().integers(1, 999999))

        # --- random compartment blocks
        types = np.empty(n, dtype=np.int8)
        cur = int(rng.choice([A_TYPE, B_TYPE]))
        i = 0
        while i < n:
            L = int(rng.integers(3, 8))
            types[i:i + L] = cur
            cur = -cur
            i += L
        if len(np.unique(types)) == 1:
            types[n // 2:] = -types[0]

        # --- random loops, a handful, non-overlapping-ish
        n_loops = int(rng.integers(1, max(2, n // 6)))
        loops: list[tuple[int, int]] = []
        tries = 0
        while len(loops) < n_loops and tries < 200:
            tries += 1
            i0 = int(rng.integers(0, n - MIN_LOOP_SPAN - 1))
            span = int(rng.integers(MIN_LOOP_SPAN, max(MIN_LOOP_SPAN + 1, n // 2)))
            j0 = i0 + span
            if j0 >= n:
                continue
            if any(max(abs(i0 - a), abs(j0 - b)) < 2 for a, b in loops):
                continue
            loops.append((i0, j0))
        loops.sort()

        self.poly.load_config(types, loops)
        self._refresh(rebuild_scale=True)

    def set_bead_count(self, new_n: int) -> None:
        from .physics import N_MIN, N_MAX
        new_n = int(np.clip(new_n, N_MIN, N_MAX))
        if new_n == self.poly.n:
            return
        old_types, old_loops = self.poly.clone_config()
        old_n = self.poly.n

        if new_n < old_n:
            types = old_types[:new_n].copy()
            # any loop touching a removed bead is dropped -- this is the
            # "if he removes some beads, the forces that used them go too"
            # behaviour you asked for.
            loops = [(i, j) for (i, j) in old_loops if j < new_n]
        else:
            fill = old_types[-1] if old_n else A_TYPE
            types = np.concatenate([old_types, np.full(new_n - old_n, fill, dtype=np.int8)])
            loops = list(old_loops)

        self.poly = Polymer(new_n, self.params, seed=self.seed)
        self.poly.load_config(types, loops)
        self.P_live = self.poly.contacts()
        self._refresh(rebuild_scale=True)

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("The Chromatin Game")
        self.settings = load_settings()
        self.screen = None
        self.apply_display_settings()
        self.clock = pygame.time.Clock()
        self.music = Music(MUSIC_DIR)
        self.music.set_volume(self.settings["volume"])
        self.records = load_records()

        self.state = MENU
        self.running = True
        self.t0 = time.time()
        self.view = PolymerView()
        self.hm_tgt = widgets.Heatmap("experimental  /  target")
        self.hm_sim = widgets.Heatmap("simulated  /  yours")

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
        self.lab_view = PolymerView()
        self.hm_lab_a = widgets.Heatmap("contact  ·  reds")
        self.hm_lab_b = widgets.Heatmap("correlation  ·  coolwarm")
        self.lab_sliders: dict[str, widgets.Slider] = {}
        self.lab_bead_field: widgets.TextField | None = None    # <-- ADD THIS
        self.lab_scroll = 0
        
        # workers
        self._work_progress = [0.0]
        self._work_result = [None]
        self._work_thread: threading.Thread | None = None

        # menu ambience
        self.demo = Polymer(30, SimParams(), seed=5)
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
        self.demo_view = PolymerView()
        self.demo_view.cam.dist = 13.0

        # interaction
        self.mouse: tuple[int, int] = (0, 0)
        self.press: tuple[int, tuple[int, int]] | None = None
        self.orbiting = False
        self.dragged = False
        self.paint: int | None = None
        self.buttons: dict[str, widgets.Button] = {}

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

    # =============================================================== events
    def handle_events(self):
        for ev in pygame.event.get():
            self.music.handle(ev)
            if ev.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN,
                           pygame.MOUSEBUTTONUP):
                self.mouse = ev.pos
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.VIDEORESIZE:
                w = max(theme.MIN_W, ev.w)
                h = max(theme.MIN_H, ev.h)
                self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE | pygame.DOUBLEBUF)
            elif ev.type == pygame.KEYDOWN:
                self.on_key(ev)
            for b in self.buttons.values():
                if b.handle(ev):
                    self.on_button(b)
            if self.state == PLAY and not self.show_help:
                self.on_play_mouse(ev)
            if self.state == LAB and self.lab is not None:
                for sl in self.lab_sliders.values():
                    if sl.handle(ev):
                        self.lab.apply_param(sl.key, sl.value)
                if self.lab_bead_field is not None:
                    new_n = self.lab_bead_field.handle(ev)
                    if new_n is not None:
                        self.lab.set_bead_count(new_n)
                self.on_lab_mouse(ev)

    def on_key(self, ev):
        k = ev.key
        # global
        if k == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
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
        if k in (pygame.K_h, pygame.K_F1):
            self.show_help = not self.show_help
            return
        if k == pygame.K_m:
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
            if self.lab_bead_field is not None and self.lab_bead_field.focused:
                return          # let the text field own all keys while typing
            if k == pygame.K_SPACE:
                lab.paused = not lab.paused
            return

        if self.state != PLAY:
            return

        s = self.session
        if k == pygame.K_l:
            self.set_mode("loop")
        elif k == pygame.K_c:
            self.set_mode("comp")
        elif k == pygame.K_TAB:
            self.set_mode("comp" if s.mode == "loop" else "loop")
        elif k == pygame.K_v:
            i = widgets.MODES.index(s.map_mode)
            s.map_mode = widgets.MODES[(i + 1) % len(widgets.MODES)]
            self.say(f"map view: {widgets.MODE_LABEL[s.map_mode]}")
        elif k == pygame.K_SPACE:
            s.paused = not s.paused
            self.say("paused" if s.paused else "running")
        elif k == pygame.K_r:
            self.view.cam = PolymerView().cam
            self.view.cam.dist = s.poly.box * 3.4
            self.say("view reset")
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

    def set_mode(self, mode: str):
        s = self.session
        if s.two_player and not s.allowed(mode):
            self.say(f"not your move -- it is the {PLAYER_NAME[s.turn].lower()}'s turn")
            return
        s.mode = mode
        self.view.pending = None

    # ---------------------------------------------------------- interaction
    def on_play_mouse(self, ev):
        """Three control surfaces: the 3D view, either heatmap, and the colour
        ribbon. The maps are not just a readout -- a loop is placed by clicking
        the cell where you want the corner to appear, on either one."""
        s = self.session
        rect = self.rects["view"]
        grid_sim = self.hm_sim.grid
        grid_tgt = self.hm_tgt.grid
        ribbon = self.rects.get("ribbon")

        if ev.type == pygame.MOUSEWHEEL:
            mx, my = self.mouse
            if rect.collidepoint(mx, my):
                self.view.cam.zoom(ev.y)
            return
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            self.view.pending = None
            return

        # ---- either heatmap: click a cell to tie/untie that loop
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
                and (grid_sim.collidepoint(ev.pos) or grid_tgt.collidepoint(ev.pos)):
            hm = self.hm_sim if grid_sim.collidepoint(ev.pos) else self.hm_tgt
            b = hm.bin_at(*ev.pos)
            if b:
                self.map_click(*b)
            return

        # ---- the colour ribbon: click to flip, drag to paint
        if ribbon and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
                and ribbon.collidepoint(ev.pos):
            i = self.ribbon_bin(ev.pos[0])
            if i is not None and s.allowed("comp"):
                s.poly.flip_type(i)
                self.paint = int(s.poly.types[i])
            elif i is not None:
                self.say(f"colours are the {PLAYER_NAME[P_COMP].lower()}'s move")
            return
        if ribbon and ev.type == pygame.MOUSEMOTION and ev.buttons[0] \
                and self.paint is not None and ribbon.collidepoint(ev.pos):
            i = self.ribbon_bin(ev.pos[0])
            if i is not None and s.allowed("comp"):
                s.poly.set_type(i, self.paint)
            return

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if not rect.collidepoint(ev.pos):
                return
            i = self.view.pick(s.poly.pos, rect, *ev.pos)
            if i is None:
                self.orbiting = True
                self.press = None
            else:
                self.press = (i, ev.pos)
                self.dragged = False
                mods = pygame.key.get_mods()
                if s.mode == "comp" and (mods & pygame.KMOD_SHIFT) and s.allowed("comp"):
                    self.paint = B_TYPE if s.poly.types[i] == A_TYPE else A_TYPE
                    s.poly.set_type(i, self.paint)
            return

        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            self.view.hover = self.view.pick(s.poly.pos, rect, mx, my) \
                if rect.collidepoint(mx, my) else None
            if self.orbiting and (ev.buttons[0] or ev.buttons[2]):
                self.view.cam.orbit(ev.rel[0], ev.rel[1])
            elif self.press is not None and ev.buttons[0]:
                i, p0 = self.press
                if self.paint is not None:
                    if self.view.hover is not None:
                        s.poly.set_type(self.view.hover, self.paint)
                    return
                if not self.dragged and math.hypot(mx - p0[0], my - p0[1]) > 5:
                    self.dragged = True
                    self.view.dragging = i
                if self.dragged:
                    _, depth = self.view.screen_positions(s.poly.pos, rect)
                    tgt = self.view.cam.unproject(mx, my, float(depth[i]),
                                                  rect.centerx, rect.centery)
                    tgt = np.clip(tgt, -s.poly.box, s.poly.box)
                    s.poly.grab = (i, tgt)
            return

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.orbiting = False
            self.paint = None
            if self.dragged:
                s.poly.grab = None
                self.view.dragging = None
                self.dragged = False
                self.press = None
                return
            if self.press is not None:
                i, _ = self.press
                self.press = None
                self.click_bead(i)

    def ribbon_bin(self, mx: int) -> int | None:
        r = self.rects.get("ribbon")
        if not r:
            return None
        i = int((mx - r.x) / r.w * self.session.poly.n)
        return i if 0 <= i < self.session.poly.n else None

    def map_click(self, i: int, j: int):
        """A click on your own contact map. In loop mode that cell IS the move."""
        s = self.session
        if not s.allowed("loop"):
            self.say(f"loops are the {PLAYER_NAME[P_LOOP].lower()}'s move")
            return
        if s.mode != "loop":
            self.say("switch to Loops (L) to place anchors from the map")
            return
        if i == j:
            return
        if not s.poly.valid_loop(i, j):
            self.say(f"anchors must be at least {MIN_LOOP_SPAN} beads apart")
            return
        lo, hi = min(i, j), max(i, j)
        res = s.poly.toggle_loop(lo, hi)
        self.view.pending = None
        self.say(f"loop ({lo},{hi}) {res} from the map")

    def click_bead(self, i: int):
        s = self.session
        if s.mode == "loop":
            if not s.allowed("loop"):
                self.say(f"loops are the {PLAYER_NAME[P_LOOP].lower()}'s move")
                return
            if self.view.pending is None:
                self.view.pending = i
                self.say(f"anchor {i} selected -- pick its partner")
            elif self.view.pending == i:
                self.view.pending = None
            else:
                a = self.view.pending
                res = s.poly.toggle_loop(a, i)
                self.view.pending = None
                if res == "invalid":
                    self.say(f"anchors must be at least {MIN_LOOP_SPAN} beads apart")
                else:
                    lo, hi = min(a, i), max(a, i)
                    self.say(f"loop ({lo},{hi}) {res}")
        else:
            if not s.allowed("comp"):
                self.say(f"colours are the {PLAYER_NAME[P_COMP].lower()}'s move")
                return
            s.poly.flip_type(i)
            self.say(f"bead {i} -> {'A red' if s.poly.types[i] > 0 else 'B blue'}")

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
            if n == "Loops":
                self.set_mode("loop")
            elif n == "Compartments":
                self.set_mode("comp")
            elif n == "View":
                i = widgets.MODES.index(self.session.map_mode)
                self.session.map_mode = widgets.MODES[(i + 1) % len(widgets.MODES)]
            elif n in ("Measure", "End turn"):
                self.begin_measure()
            elif n == "Menu":
                self.state = MENU
                self.session = None
            elif n == "?":
                self.show_help = True
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
            elif n == "Windowed":
                self.settings["fullscreen"] = False
                self.apply_display_settings()
            elif n == "Fullscreen":
                self.settings["fullscreen"] = True
                self.apply_display_settings()
            else:
                # resolution buttons — text is "WIDTHxHEIGHT"
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
        self.music.start_if_idle()

    def start_lab(self) -> None:
        seed = int(np.random.default_rng().integers(1, 9999))
        self.lab = LabSession(n=40, seed=seed)
        self.lab_scroll = 0
        self.state = LAB
        self.music.start_if_idle()

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
        rep = an.evaluate(P, t.P, C, t.C, e1, t.e1, list(p.loops), list(t.loops))
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
                s.mode = "comp"
            else:
                s.turn = P_LOOP
                s.mode = "loop"
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
                self.lab.step()
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

        tl = s.time_left()
        if s.two_player and tl is not None and tl <= 0.0:
            self.say("time up")
            self.begin_measure()

    def demo_spin(self):
        pass

    # ================================================================= draw
    def draw(self):
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
        S = getattr(theme, "FONT_SCALE", 1.0)

        P     = int(theme.PAD * S)
        hdr_h = int(theme.HEADER_H * S)
        ftr_h = int(theme.FOOTER_H * S)

        right_frac = 0.45 + (S - 1.0) * 0.10 
        right_w = max(int(620 * S), int(W * right_frac))

        body = pygame.Rect(0, hdr_h, W, H - hdr_h - ftr_h)
        view = pygame.Rect(P, body.y + P, W - right_w - P, body.h - 2 * P)
        right = pygame.Rect(view.right + P, body.y + P,
                        right_w - 2 * P, body.h - 2 * P)

        
        title_h = int(30 * S)
        label_h = int(22 * S)
        inner   = int(32 * S)

        hm_w = (right.w - P) // 2
        side = max(80, hm_w - inner)
        hm_h = title_h + side + label_h

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
        sc = self.screen
        sc.fill(theme.INK)
        grad = pygame.Surface((1, H))
        for y in range(H):
            t = y / max(1, H - 1)
            grad.set_at((0, y), theme.lerp_col(theme.INK_2, theme.INK, t ** 0.7))
        sc.blit(pygame.transform.scale(grad, (W, H)), (0, 0))

    # ---------------------------------------------------------------- menu
    def draw_menu(self, W, H):
        sc = self.screen
        self.buttons = {}

        # ambient structure on the right half
        vr = pygame.Rect(int(W * 0.50), 0, int(W * 0.50), H)
        self.demo_view.cam.dist = 9.5
        self.demo_view.cam.center = self.demo.pos.mean(axis=0)
        self.demo_view.draw(sc, vr, self.demo.pos, self.demo.types, self.demo.loops,
                            self.demo.box, t=time.time() - self.t0, show_index=False,
                            mouse=self.mouse, show_box=False)
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(veil, (*theme.INK, 150), (0, 0, int(W * 0.55), H))
        sc.blit(veil, (0, 0))

        x = 70
        y = int(H * 0.16)
        widgets.eyebrow(sc, "a polymer physics game about reading hi-c", x, y, theme.CYAN)
        f = theme.font(64, bold=True)
        sc.blit(f.render("THE CHROMATIN", True, theme.TEXT), (x - 3, y + 18))
        f2 = theme.font(64, bold=True)
        sc.blit(f2.render("       GAME", True, theme.GREEN), (x - 3, y + 88))
        widgets.label(sc, "Build the fold. Match the map.", x, y + 182, size=16,
                      col=theme.TEXT_DIM)

        # level chooser
        y2 = y + 222
        widgets.eyebrow(sc, "locus", x, y2)
        for i, lvl in enumerate(LEVELS):
            r = pygame.Rect(x, y2 + 22 + i * 46, 340, 28)
            b = widgets.Button(r, lvl.name, size=13)
            b.active = (i == self.sel_level)
            self.buttons[f"lvl{i}"] = b
            b.draw(sc)
            plural = "loop" if lvl.n_loops == 1 else "loops"
            widgets.label(sc, f"{lvl.n} beads · {lvl.n_loops} {plural}", r.right + 12,
                          r.y + 7, size=11, col=theme.TEXT_FAINT, mono=True)
            rec_key = lvl.name + ("  [hard]" if self.sel_hard else "")
            rec = self.records.get(rec_key, {}).get("best")
            if rec:
                widgets.label(sc, f"best {rec:.1f}", r.right + 200, r.y + 7, size=11,
                              col=theme.AMBER, mono=True)

        # mode
        y3 = y2 + 18 + len(LEVELS) * 46 + 16
        b1 = widgets.Button((x, y3 + 22, 110, 30), "Solo")
        b1.active = not self.sel_two
        b2 = widgets.Button((x + 118, y3 + 18, 110, 30), "Versus", accent=theme.MAGENTA)
        b2.active = self.sel_two
        self.buttons["solo"] = b1
        self.buttons["versus"] = b2
        b1.draw(sc)
        b2.draw(sc)

        # --- difficulty: hard mode hides the target's loops and compartment signal
        bd1 = widgets.Button((x + 480, y3 + 18, 90, 30), "Easy")
        bd2 = widgets.Button((x + 574, y3 + 18, 90, 30), "Hard", accent=theme.POOR)
        bd1.active = not self.sel_hard
        bd2.active = self.sel_hard
        self.buttons["easy"] = bd1
        self.buttons["hard"] = bd2
        bd1.draw(sc)
        bd2.draw(sc)
        if self.sel_hard:
            widgets.label(sc, "hard: no loop dots, no E1 track on the target",
                        x + 480, y3 + 56, size=11, col=theme.POOR)

        if self.sel_two:
            br = widgets.Button((x + 244, y3 + 18, 108, 30), f"Rounds: {self.sel_rounds}")
            ts = "off" if not self.sel_turn_sec else f"{self.sel_turn_sec}s"
            bt = widgets.Button((x + 360, y3 + 18, 116, 30), f"Turn: {ts}")
            self.buttons["rounds"] = br
            self.buttons["turn"] = bt
            br.draw(sc)
            bt.draw(sc)
            widgets.label(sc, "P1 plays loops · P2 plays compartments · same polymer",
                          x, y3 + 56, size=11, col=theme.TEXT_FAINT)

        y4 = y3 + 86
        bs   = widgets.Button((x, y4, 150, 38), "Start", key="ENTER", size=15)
        bs.active = True
        bsh  = widgets.Button((x + 158, y4, 130, 38), "Shuffle seed", size=13)
        bh   = widgets.Button((x + 296, y4, 130, 38), "How to play", key="H", size=13)
        bset = widgets.Button((x + 434, y4, 110, 38), "Settings", size=13)
        bq   = widgets.Button((x + 552, y4, 80, 38), "Quit", size=13)
        for k, b in (("start", bs), ("seed", bsh), ("help", bh),
                    ("settings", bset), ("quit", bq)):
            self.buttons[k] = b
            b.draw(sc)
        widgets.label(sc, f"seed {self.sel_seed}", x + 640, y4 + 12, size=11,
                    col=theme.TEXT_FAINT, mono=True)

        # music strip
        m = self.music
        widgets.label(sc, ("♪  " + m.title()) if m.has_music else
                      "♪  drop your piano .mp3 files into  music/",
                      x, H - 54, size=12, col=theme.TEXT_DIM if m.has_music else theme.TEXT_FAINT)
        if m.has_music:
            widgets.label(sc, f"{len(m.tracks)} track(s)  ·  M mute  ·  N next",
                          x, H - 36, size=11, col=theme.TEXT_FAINT, mono=True)
            
        y5 = y4 + 56
        blab = widgets.Button((x, y5, 260, 42), "Chromatin MiNI-Lab", size=15,
                            accent=theme.ROYAL_RED)
        blab.active = True
        self.buttons["lab"] = blab
        blab.draw(sc)
        widgets.label(sc, "free-play sandbox — every force, every parameter, live",
                    x + 272, y5 + 14, size=11, col=theme.TEXT_FAINT)
            
    # ---------------------------------------------------------------- settings
    def draw_settings(self, W, H):
        sc = self.screen
        self.buttons = {}

        x = 80
        y = int(H * 0.10)
        widgets.eyebrow(sc, "settings", x, y, theme.CYAN)
        f = theme.font(42, bold=True)
        sc.blit(f.render("SETTINGS", True, theme.TEXT), (x - 2, y + 12))
        widgets.label(sc, "Changes are saved when you go back to the menu.",
                    x, y + 62, size=13, col=theme.TEXT_DIM)

        y += 118

        # --- music volume -------------------------------------------------
        widgets.eyebrow(sc, "music volume", x, y)
        vol = self.settings["volume"]
        bvm = widgets.Button((x, y + 22, 48, 32), "Vol -", size=13)
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

        # --- font size ----------------------------------------------------
        widgets.eyebrow(sc, "font size", x, y)
        bfs = widgets.Button((x, y + 22, 110, 32), "Small", size=13)
        bfl = widgets.Button((x + 118, y + 22, 110, 32), "Large", size=13)
        bfs.active = self.settings["font_mode"] == "small"
        bfl.active = self.settings["font_mode"] == "large"
        self.buttons["font_small"] = bfs
        self.buttons["font_large"] = bfl
        bfs.draw(sc); bfl.draw(sc)
        widgets.label(sc, "large adds ~22% to every label",
                    x + 240, y + 30, size=11,
                    col=theme.TEXT_FAINT, mono=True)

        y += 82

        # --- resolution ---------------------------------------------------
        widgets.eyebrow(sc, "resolution (windowed only)", x, y)
        for i, (rw, rh) in enumerate(RESOLUTIONS):
            b = widgets.Button((x + i * 132, y + 22, 124, 32),
                            f"{rw}x{rh}", size=13)
            b.active  = (self.settings["resolution"] == i
                        and not self.settings["fullscreen"])
            b.enabled = not self.settings["fullscreen"]
            self.buttons[f"res{i}"] = b
            b.draw(sc)

        y += 82

        # --- display mode -------------------------------------------------
        widgets.eyebrow(sc, "display mode", x, y)
        bw = widgets.Button((x, y + 22, 120, 32), "Windowed", size=13)
        bf = widgets.Button((x + 128, y + 22, 120, 32), "Fullscreen", size=13)
        bw.active = not self.settings["fullscreen"]
        bf.active = self.settings["fullscreen"]
        self.buttons["windowed"]   = bw
        self.buttons["fullscreen"] = bf
        bw.draw(sc); bf.draw(sc)

        y += 96

        # --- back ---------------------------------------------------------
        bb = widgets.Button((x, y, 140, 40), "Back", key="ESC", size=15)
        bb.active = True
        self.buttons["back"] = bb
        bb.draw(sc)

    # -------------------------------------------------------------- loading
    def draw_loading(self, W, H):
        sc = self.screen
        self.buttons = {}
        p = self._work_progress[0]
        cx, cy = W // 2, H // 2
        widgets.eyebrow(sc, "running the hidden ground truth", cx - 130, cy - 60, theme.CYAN)
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

    # ----------------------------------------------------------------- play
    def draw_play(self, W, H):
        sc = self.screen
        s = self.session
        self.buttons = {}
        R = self.rects

        self.draw_header(W)
        self.draw_footer(W, H)

        # heatmaps are laid out first: hovering a bin lights up the beads it
        # refers to, so the map and the structure read as one object.
        t = s.target
        self.hm_tgt.layout(R["tgt"], t.n)
        self.hm_sim.layout(R["sim"], t.n)
        mx, my = self.mouse
        hb = self.hm_sim.bin_at(mx, my) or self.hm_tgt.bin_at(mx, my)
        rb = self.ribbon_bin(mx) if self.rects.get("ribbon") \
            and self.rects["ribbon"].collidepoint(mx, my) else None
        if hb and hb[0] != hb[1]:
            self.view.mark = (hb[0], hb[1])
        elif rb is not None:
            self.view.mark = (rb,)
        else:
            self.view.mark = ()

        # ---- 3D viewport
        widgets.panel(sc, R["view"], fill=theme.INK_2)
        self.view.draw(sc, R["view"], s.poly.pos, s.poly.types, s.poly.loops,
                       s.poly.box, e1=None, t=time.time() - self.t0,
                       mouse=self.mouse)
        self.draw_view_hud(R["view"])

        # ---- heatmaps
        if self._scale is None:
            self._scale = widgets.Scale(t.P, t.C)
        self.hm_tgt.draw(sc, t.P, t.C, s.map_mode,
                        loops_true=(None if s.hard else t.loops),
                        subtitle=("hard mode -- no ground truth overlay" if s.hard
                                else "what you must reproduce"),
                        accent=(theme.POOR if s.hard else theme.AMBER),
                        mouse=self.mouse, scale=self._scale)
        self.hm_sim.draw(sc, s.P_live, s.C_live, s.map_mode,
                         loops_player=s.poly.loops,
                         subtitle=f"ensemble at t={s.sim_time:6.1f}", accent=theme.CYAN,
                         live=(self.state == PLAY and not s.paused),
                         mouse=self.mouse, scale=self._scale)

        # ---- E1 tracks (target E1 is hidden in hard mode -- it directly reveals
        # compartment assignment, which the player is meant to infer from the map)
        g1, g2 = self.hm_tgt.grid, self.hm_sim.grid
        y = R["tgt"].bottom + 14
        if s.hard:
            ph = pygame.Rect(g1.x, y, g1.w, 34)
            pygame.draw.rect(sc, theme.PANEL, ph, border_radius=6)
            widgets.label(sc, "E1 target — hidden in hard mode", ph.x + 10, ph.centery - 6,
                        size=10, col=theme.TEXT_FAINT, mono=True)
        else:
            widgets.eigen_track(sc, pygame.Rect(g1.x, y, g1.w, 34), t.e1, title="E1 target")
        widgets.eigen_track(sc, pygame.Rect(g2.x, y, g2.w, 34), s.e1_live, title="E1 yours")
        yr = y + 42
        ribbon = pygame.Rect(g2.x, yr, g2.w, 14)
        self.rects["ribbon"] = ribbon
        rb = self.ribbon_bin(self.mouse[0]) if ribbon.collidepoint(self.mouse) else None
        widgets.type_track(sc, ribbon, s.poly.types,
                           hover=rb if rb is not None else self.view.hover)
        widgets.label(sc, "your colouring — click or drag to paint", g2.x, yr + 17,
                      size=9, col=theme.TEXT_FAINT, mono=True)
        if not s.hard:
            widgets.label(sc, "green dots = target anchors", g1.x, yr + 2, size=9,
                        col=theme.GREEN, mono=True)
        widgets.label(sc, "click a cell to tie that loop", g2.x + g2.w, yr - 16, size=9,
                      col=theme.CYAN, mono=True, right=True)

        # ---- metrics
        mt = pygame.Rect(R["right"].x, yr + 34, R["right"].w, 180)
        widgets.metric_table(sc, mt, s.rep_live,
                             live=(self.state == PLAY and not s.paused))

        # ---- scores
        ys = mt.y + 190
        if s.two_player:
            hw = (R["right"].w - theme.PAD) // 2
            widgets.big_score(sc, pygame.Rect(R["right"].x, ys, hw, 56), s.best[P_LOOP],
                              "p1 best · loops", theme.GREEN)
            widgets.big_score(sc, pygame.Rect(R["right"].x + hw + theme.PAD, ys, hw, 56),
                              s.best[P_COMP], "p2 best · compartments", theme.MAGENTA)
            widgets.label(sc, f"live  loop {s.rep_live['loop_score']:5.1f}   "
                              f"comp {s.rep_live['comp_score']:5.1f}",
                          R["right"].x, ys + 62, size=11, col=theme.TEXT_FAINT, mono=True)
        else:
            widgets.big_score(sc, pygame.Rect(R["right"].x, ys, R["right"].w, 56),
                              s.rep["total"] if s.rep else s.rep_live["total"],
                              "score  ·  press enter to measure", theme.CYAN)
            rec = self.records.get(s.level.name, {}).get("best", 0.0)
            widgets.label(sc, f"live {s.rep_live['total']:5.1f}   ·   record {rec:5.1f}"
                              f"   ·   {s.measurements} measurement(s)",
                          R["right"].x, ys + 62, size=11, col=theme.TEXT_FAINT, mono=True)
        widgets.colorbars(sc, pygame.Rect(R["right"].x, ys + 92, R["right"].w, 30),
                          self._scale, s.map_mode)
        
    def on_lab_mouse(self, ev):
        lab = self.lab
        if lab is None or "view" not in self.rects:
            return
        rect = self.rects["view"]
        ribbon = self.rects.get("ribbon")
        grid_a = self.hm_lab_a.grid
        grid_b = self.hm_lab_b.grid

        if ev.type == pygame.MOUSEWHEEL:
            if rect.collidepoint(self.mouse):
                self.lab_view.cam.zoom(ev.y)
                return
            panel_r = self.rects.get("panel")
            if panel_r and panel_r.collidepoint(self.mouse):
                content_h = getattr(self, "_lab_panel_content_h", 0)
                self.lab_scroll = max(0, min(self.lab_scroll - ev.y * 30,
                                            max(0, content_h - panel_r.h)))
            return
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            self.lab_view.pending = None
            return

        # ---- either heatmap: click a cell to tie/untie that loop
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
                and (grid_a.collidepoint(ev.pos) or grid_b.collidepoint(ev.pos)):
            hm = self.hm_lab_a if grid_a.collidepoint(ev.pos) else self.hm_lab_b
            b = hm.bin_at(*ev.pos)
            if b and b[0] != b[1]:
                lo, hi = min(b), max(b)
                if lab.poly.valid_loop(lo, hi):
                    lab.poly.toggle_loop(lo, hi)
                    self.say(f"loop ({lo},{hi}) toggled")
                else:
                    self.say(f"anchors must be at least {MIN_LOOP_SPAN} beads apart")
            return

        # ---- the colour ribbon: click to flip, drag to paint
        if ribbon and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
                and ribbon.collidepoint(ev.pos):
            i = self.ribbon_bin_n(ev.pos[0], lab.poly.n)
            if i is not None:
                lab.poly.flip_type(i)
                self.paint = int(lab.poly.types[i])
            return
        if ribbon and ev.type == pygame.MOUSEMOTION and ev.buttons[0] \
                and self.paint is not None and ribbon.collidepoint(ev.pos):
            i = self.ribbon_bin_n(ev.pos[0], lab.poly.n)
            if i is not None:
                lab.poly.set_type(i, self.paint)
            return

        # ---- 3D view: click bead to flip/loop, drag to move, drag empty space to orbit
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if not rect.collidepoint(ev.pos):
                return
            i = self.lab_view.pick(lab.poly.pos, rect, *ev.pos)
            if i is None:
                self.orbiting = True
                self.press = None
            else:
                self.press = (i, ev.pos)
                self.dragged = False
                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:
                    self.paint = B_TYPE if lab.poly.types[i] == A_TYPE else A_TYPE
                    lab.poly.set_type(i, self.paint)
            return
        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            self.lab_view.hover = self.lab_view.pick(lab.poly.pos, rect, mx, my) \
                if rect.collidepoint(mx, my) else None
            if self.orbiting and (ev.buttons[0] or ev.buttons[2]):
                self.lab_view.cam.orbit(ev.rel[0], ev.rel[1])
            elif self.press is not None and ev.buttons[0]:
                i, p0 = self.press
                if self.paint is not None:
                    if self.lab_view.hover is not None:
                        lab.poly.set_type(self.lab_view.hover, self.paint)
                    return
                if not self.dragged and math.hypot(mx - p0[0], my - p0[1]) > 5:
                    self.dragged = True
                    self.lab_view.dragging = i
                if self.dragged:
                    _, depth = self.lab_view.screen_positions(lab.poly.pos, rect)
                    tgt = self.lab_view.cam.unproject(mx, my, float(depth[i]),
                                                      rect.centerx, rect.centery)
                    tgt = np.clip(tgt, -lab.poly.box, lab.poly.box)
                    lab.poly.grab = (i, tgt)
            return
        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.orbiting = False
            self.paint = None
            if self.dragged:
                lab.poly.grab = None
                self.lab_view.dragging = None
                self.dragged = False
                self.press = None
                return
            if self.press is not None:
                i, _ = self.press
                self.press = None
                if self.lab_view.pending is None:
                    self.lab_view.pending = i
                elif self.lab_view.pending == i:
                    self.lab_view.pending = None
                else:
                    a = self.lab_view.pending
                    self.lab_view.pending = None
                    lab.poly.toggle_loop(*sorted((a, i)))

    def ribbon_bin_n(self, mx: int, n: int) -> int | None:
        r = self.rects.get("ribbon")
        if not r:
            return None
        i = int((mx - r.x) / r.w * n)
        return i if 0 <= i < n else None
    
    def compute_lab_layout(self, W, H):
        S = getattr(theme, "FONT_SCALE", 1.0)
        P = int(theme.PAD * S)
        hdr_h = int(theme.HEADER_H * S)
        ftr_h = int(theme.FOOTER_H * S)
        body = pygame.Rect(0, hdr_h, W, H - hdr_h - ftr_h)

        panel_w = max(340, int(W * 0.27))                # parameter panel, right side
        left = pygame.Rect(P, body.y + P, W - panel_w - P, body.h - 2 * P)
        panel = pygame.Rect(left.right + P, body.y + P, panel_w - 2 * P, body.h - 2 * P)

        view_h = int(left.h * 0.55)
        view = pygame.Rect(left.x, left.y, left.w, view_h)
        maps_y = view.bottom + P
        hm_w = (left.w - P) // 2
        hm_h = left.bottom - maps_y

        self.rects = {
            "header": pygame.Rect(0, 0, W, hdr_h),
            "footer": pygame.Rect(0, H - ftr_h, W, ftr_h),
            "view": view,
            "map_a": pygame.Rect(left.x, maps_y, hm_w, hm_h),
            "map_b": pygame.Rect(left.x + hm_w + P, maps_y, hm_w, hm_h),
            "panel": panel,
        }

    def draw_lab(self, W, H):
        sc = self.screen
        lab = self.lab
        self.buttons = {}
        self.compute_lab_layout(W, H)
        R = self.rects

        # ---- header
        r = R["header"]
        pygame.draw.rect(sc, theme.INK, r)
        pygame.draw.line(sc, theme.RULE, (0, r.bottom - 1), (W, r.bottom - 1), 1)
        widgets.label(sc, "CHROMATIN MiNI-LAB", 16, 9, size=13,
                    col=theme.ROYAL_RED, bold=True)
        widgets.label(sc, "free-play sandbox — every parameter is live", 16, 27,
                    size=11, col=theme.TEXT_FAINT)
        bp = widgets.Button((260, 12, 100, 32), "Pause" if not lab.paused else "Resume",
                    key="SPACE")
        bp.active = lab.paused
        br = widgets.Button((368, 12, 100, 32), "Randomize")
        brs = widgets.Button((476, 12, 90, 32), "Reset")
        bx = widgets.Button((W - 120, 12, 104, 32), "Exit Lab", key="ESC")
        for k, b in (("lab_pause", bp), ("lab_rand", br), ("lab_reset", brs), ("lab_exit", bx)):
            self.buttons[k] = b
            b.draw(sc)

        # ---- footer
        fr = R["footer"]
        pygame.draw.rect(sc, theme.INK, fr)
        pygame.draw.line(sc, theme.RULE, (0, fr.y), (W, fr.y), 1)
        hint = ("click any cell on a map to tie/untie a loop  ·  paint the ribbon to set compartments  ·  "
                "shift+click a bead in 3D to flip it")
        widgets.label(sc, hint, 16, fr.y + 9, size=11, col=theme.TEXT_DIM)

        # ---- 3D view
        widgets.panel(sc, R["view"], fill=theme.INK_2)
        self.lab_view.draw(sc, R["view"], lab.poly.pos, lab.poly.types, lab.poly.loops,
                        lab.poly.box, t=time.time() - self.t0, mouse=self.mouse)

        # ---- two heatmaps, pinned modes: reds (contact) + coolwarm (correlation)
        self.hm_lab_a.layout(R["map_a"], lab.poly.n)
        self.hm_lab_b.layout(R["map_b"], lab.poly.n)
        self.hm_lab_a.draw(sc, lab.P_live, lab.C_live, "contact",
                        loops_player=lab.poly.loops, subtitle="live contact frequency",
                        accent=theme.ROYAL_RED, live=not lab.paused,
                        mouse=self.mouse, scale=lab.scale)
        self.hm_lab_b.draw(sc, lab.P_live, lab.C_live, "corr",
                        loops_player=lab.poly.loops, subtitle="O/E correlation",
                        accent=theme.CYAN, live=not lab.paused,
                        mouse=self.mouse, scale=lab.scale)

        # ---- colour ribbon under the left column
        g = self.hm_lab_a.grid
        ry = R["map_a"].bottom + int(10 * getattr(theme, "FONT_SCALE", 1.0))
        ribbon = pygame.Rect(R["view"].x, ry, R["view"].w, 14)
        self.rects["ribbon"] = ribbon
        widgets.type_track(sc, ribbon, lab.poly.types, hover=self.lab_view.hover)

        # ---- parameter panel, scrollable
        self.draw_lab_panel(R["panel"])

    def draw_lab_panel(self, panel_rect):
        sc = self.screen
        lab = self.lab
        S = getattr(theme, "FONT_SCALE", 1.0)
        widgets.panel(sc, panel_rect, fill=theme.PANEL)

        x = panel_rect.x + 16
        y = panel_rect.y + 14 - self.lab_scroll
        row_h = int(40 * S)
        self.lab_sliders = {}

        # bead-count control, pinned at the top (not part of the scroll)
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

        y += int(46 * S)
        clip_top = y

        # stability guardrail: dt * k_bond / gamma should stay below ~0.5
        ratio = lab.params.dt * lab.params.k_bond / lab.params.gamma
        if ratio > 0.5:
            widgets.label(sc, f"⚠ dt·k_bond/γ = {ratio:.2f} -- integrator may be unstable",
                        x, y, size=11, col=theme.POOR, bold=True)
            y += int(20 * S)

        for group_name, fields in LAB_PARAM_GROUPS:
            widgets.eyebrow(sc, group_name, x, y, theme.TEXT_DIM)
            y += int(20 * S)
            for key, label, lo, hi, step, fmt, is_int in fields:
                val = getattr(lab.params, key)
                r = pygame.Rect(x, y + int(14 * S), panel_rect.w - 32, int(8 * S))
                sl = widgets.Slider(r, label, val, lo, hi, step=step, fmt=fmt, integer=is_int)
                sl.key = key
                self.lab_sliders[key] = sl
                if panel_rect.y <= y <= panel_rect.bottom:   # only draw visible rows
                    sl.draw(sc)
                y += row_h
            y += int(10 * S)

        self._lab_panel_content_h = y - clip_top + self.lab_scroll

    def draw_view_hud(self, rect):
        sc = self.screen
        s = self.session
        p = s.poly
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
        r = pygame.Rect(rect.x + 10, rect.y + 10, w, h)
        s2 = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(s2, (*theme.INK, 165), s2.get_rect(), border_radius=6)
        sc.blit(s2, r)
        widgets.eyebrow(sc, "state", r.x + 8, r.y + 6)
        for i, ln in enumerate(lines):
            widgets.label(sc, ln, r.x + 8, r.y + 20 + i * 14, size=10,
                          col=theme.TEXT_DIM, mono=True)

        # legend, bottom-left
        ly = rect.bottom - 58
        for i, (col, txt) in enumerate([
                (theme.COMP_A, "A · red · E1>0 · weak attraction"),
                (theme.COMP_B, "B · blue · E1<0 · strong attraction"),
                (theme.GREEN, "loop bond · harmonic, non-consecutive")]):
            pygame.draw.circle(sc, col, (rect.x + 18, ly + i * 16 + 5), 5)
            widgets.label(sc, txt, rect.x + 30, ly + i * 16 - 2, size=10,
                          col=theme.TEXT_FAINT)

        if s.paused:
            widgets.label(sc, "PAUSED", rect.centerx, rect.y + 14, size=14,
                          col=theme.AMBER, bold=True, center=True)

    def draw_header(self, W):
        sc = self.screen
        s = self.session
        r = self.rects["header"]
        pygame.draw.rect(sc, theme.INK, r)
        pygame.draw.line(sc, theme.RULE, (0, r.bottom - 1), (W, r.bottom - 1), 1)

        # Every pixel offset scales with the current font size so the header keeps
        # its air in Large mode. `S` is 1.0 in Small (identical to before) and
        # ~1.22 in Large (Settings > font size).
        S = getattr(theme, "FONT_SCALE", 1.0)
        btn_h = int(32 * S)                              # button height
        btn_y = (r.h - btn_h) // 2                        # vertically centred
        gap   = int(10 * S)                               # gap between buttons
        edge  = int(16 * S)                               # side margin

        # ---- title block, top-left
        widgets.label(sc, "THE CHROMATIN GAME",
                    edge, int(9 * S),
                    size=13, col=theme.TEXT, bold=True)
        diff_tag = "  ·  HARD" if s.hard else ""
        widgets.label(sc, s.level.name + diff_tag, edge, int(29 * S), size=11,
                    col=theme.POOR if s.hard else theme.TEXT_FAINT)

        # ---- mode toggles: sized to their text so bigger fonts don't get clipped
        x = int(220 * S)
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
        self.buttons["loops"] = bl
        self.buttons["comp"]  = bc
        self.buttons["view"]  = bv
        for b in (bl, bc, bv):
            b.draw(sc)

        modes_right = x + w_loops + gap + w_comp + gap + w_view

        # ---- turn banner (versus mode only)
        if s.two_player:
            acc = PLAYER_ACCENT[s.turn]
            br_w = int(260 * S)
            br = pygame.Rect(modes_right + int(20 * S), btn_y, br_w, btn_h)
            pygame.draw.rect(sc, theme.lerp_col(acc, theme.INK, 0.75), br,
                            border_radius=6)
            pygame.draw.rect(sc, acc, br, 1, border_radius=6)
            widgets.label(sc,
                        f"ROUND {min(s.round, s.rounds)}/{s.rounds}  ·  "
                        f"{PLAYER_SHORT[s.turn]}",
                        br.x + int(12 * S), br.y + int(9 * S),
                        size=12, col=acc, mono=True, bold=True)
            tl = s.time_left()
            if tl is not None:
                col = theme.POOR if tl < 10 else theme.AMBER
                widgets.label(sc,
                            f"{int(tl // 60)}:{int(tl % 60):02d}",
                            br.right + int(14 * S), br.y + int(8 * S),
                            size=15, col=col, mono=True, bold=True)

        # ---- right side, laid out from the right edge inward
        # Order right-to-left: Menu, ?, Measure/End-turn.
        w_menu  = int(84 * S)
        w_help  = int(34 * S)
        w_meas  = int(140 * S)

        bm_x = W - edge - w_menu
        bq_x = bm_x - gap - w_help
        bmeas_x = bq_x - gap - w_meas

        bm    = widgets.Button((bm_x,    btn_y, w_menu, btn_h), "Menu",  key="ESC")
        bq    = widgets.Button((bq_x,    btn_y, w_help, btn_h), "?",     key="")
        bmeas = widgets.Button((bmeas_x, btn_y, w_meas, btn_h),
                            "End turn" if s.two_player else "Measure",
                            key="ENTER", accent=theme.AMBER)
        bmeas.active = True
        for k, b in (("menu", bm), ("helpb", bq), ("measure", bmeas)):
            self.buttons[k] = b
            b.draw(sc)

        # ---- music title, tucked left of the Measure button
        if self.music.has_music:
            widgets.label(sc, "♪ " + self.music.title()[:28],
                        bmeas_x - gap, int(30 * S),
                        size=10, col=theme.TEXT_FAINT, right=True)

    def draw_footer(self, W, H):
        sc = self.screen
        r = self.rects["footer"]
        pygame.draw.rect(sc, theme.INK, r)
        pygame.draw.line(sc, theme.RULE, (0, r.y), (W, r.y), 1)
        s = self.session
        if s.mode == "loop":
            hint = ("LOOPS   click cell (i,j) on your map to tie that loop  ·  or click "
                    "bead i then bead j in 3D  ·  click again to untie  ·  BACKSPACE clears")
        else:
            hint = ("COMPARTMENTS   click or drag the colour ribbon to paint  ·  or click "
                    "a bead in 3D to flip it  ·  watch E1 follow")
        widgets.label(sc, hint, 16, r.y + 9, size=11, col=theme.TEXT_DIM)
        widgets.label(sc, "drag polymer to move · drag space to orbit · wheel zoom · "
                          "SPACE pause · F fast-forward · R reset view",
                      W - 16, r.y + 9, size=11, col=theme.TEXT_FAINT, right=True)

    # --------------------------------------------------------------- settle
    def draw_settle_overlay(self, W, H):
        sc = self.screen
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((*theme.INK, 190))
        sc.blit(veil, (0, 0))
        cx, cy = W // 2, H // 2
        p = self._work_progress[0]
        widgets.eyebrow(sc, "measuring", cx - 40, cy - 54, theme.AMBER)
        widgets.label(sc, "Equilibrating, then averaging the contact map",
                      cx, cy - 32, size=17, col=theme.TEXT, center=True)
        widgets.label(sc, f"{MEAS_SAMPLES} conformations · a single structure is not a Hi-C map",
                      cx, cy - 8, size=12, col=theme.TEXT_DIM, center=True)
        bar = pygame.Rect(cx - 200, cy + 22, 400, 6)
        pygame.draw.rect(sc, theme.PANEL, bar, border_radius=3)
        pygame.draw.rect(sc, theme.AMBER, (bar.x, bar.y, int(bar.w * p), bar.h),
                         border_radius=3)

    # -------------------------------------------------------------- results
    def draw_results(self, W, H):
        sc = self.screen
        s = self.session
        self.buttons = {}
        cx = W // 2
        y = H // 2 - 190
        widgets.eyebrow(sc, "final call", cx - 34, y, theme.CYAN)
        widgets.label(sc, s.level.name, cx, y + 16, size=30, col=theme.TEXT,
                      bold=True, center=True)

        if s.two_player:
            win = P_LOOP if s.best[P_LOOP] > s.best[P_COMP] else P_COMP
            tie = abs(s.best[P_LOOP] - s.best[P_COMP]) < 0.05
            widgets.label(sc, "Draw" if tie else f"{PLAYER_NAME[win]} wins",
                          cx, y + 58, size=22,
                          col=theme.TEXT if tie else PLAYER_ACCENT[win], center=True,
                          bold=True)
            for i in (P_LOOP, P_COMP):
                x = cx - 320 + i * 340
                r = pygame.Rect(x, y + 100, 300, 92)
                widgets.panel(sc, r, fill=theme.INK_2)
                widgets.eyebrow(sc, PLAYER_NAME[i], r.x + 12, r.y + 10, PLAYER_ACCENT[i])
                f = theme.font(44, mono=True, bold=True)
                img = f.render(f"{s.best[i]:.1f}", True, theme.score_color(s.best[i] / 100))
                sc.blit(img, (r.x + 12, r.y + 28))
                hs = "  ".join(f"{v:.0f}" for v in s.history[i]) or "-"
                widgets.label(sc, f"rounds  {hs}", r.x + 12, r.y + 74, size=10,
                              col=theme.TEXT_FAINT, mono=True)
            note = ("The loop player is scored on corner enrichment and anchor accuracy; "
                    "the compartment player on E1 and the checkerboard.")
            widgets.label(sc, note, cx, y + 208, size=12, col=theme.TEXT_DIM, center=True)
            widgets.label(sc, "You were both folding the same polymer.", cx, y + 228,
                          size=12, col=theme.TEXT_FAINT, center=True)
        else:
            widgets.label(sc, f"{s.best_total:.1f}", cx, y + 60, size=64,
                          col=theme.score_color(s.best_total / 100), center=True, bold=True)

        b = widgets.Button((cx - 80, y + 270, 160, 40), "Back to menu", key="ENTER")
        b.active = True
        self.buttons["back"] = b
        b.draw(sc)

    # ----------------------------------------------------------------- help
    def draw_help(self, W, H):
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
                widgets.label(sc, ln, x + 4, y + 16 + i * 16, size=12, col=theme.TEXT_DIM)
            y += 22 + len(lines) * 16 + 12

        keys = "L loops   C compartments   V map view   SPACE pause   F fast-forward   " \
               "R reset view   ENTER measure   M mute   N next track   ESC back"
        widgets.label(sc, keys, x, r.bottom - 34, size=11, col=theme.TEXT_FAINT, mono=True)
        widgets.label(sc, "H or ESC to close", r.right - 34, r.y + 26, size=11,
                      col=theme.TEXT_FAINT, right=True)

    # ---------------------------------------------------------------- toast
    def draw_toast(self, W, H):
        if not self.toast:
            return
        age = time.time() - self.toast_t
        if age > 3.0:
            self.toast = ""
            return
        a = 255 if age < 2.3 else int(255 * (1 - (age - 2.3) / 0.7))
        f = theme.font(12, mono=True)
        img = f.render(self.toast, True, theme.TEXT)
        pad = 10
        r = pygame.Rect(W // 2 - img.get_width() // 2 - pad,
                        H - theme.FOOTER_H - 44, img.get_width() + pad * 2, 26)
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*theme.PANEL_HI, min(a, 235)), s.get_rect(), border_radius=6)
        pygame.draw.rect(s, (*theme.RULE, min(a, 235)), s.get_rect(), 1, border_radius=6)
        s.blit(img, (pad, 6))
        s.set_alpha(a)
        self.screen.blit(s, r)


def main():
    Game().run()
