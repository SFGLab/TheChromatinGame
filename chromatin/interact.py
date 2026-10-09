# chromatin/interact.py
from __future__ import annotations
import math
import numpy as np
import pygame
from . import widgets
from .physics import A_TYPE, B_TYPE, MIN_LOOP_SPAN
from .constants import *

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
            if (mods & pygame.KMOD_SHIFT) and s.allowed("comp"):
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
    """A click on your own contact map: that cell IS the loop you're tying."""
    s = self.session
    if not s.allowed("loop"):
        self.say(f"loops are the {PLAYER_NAME[P_LOOP].lower()}'s move")
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
    """A plain bead click always starts/finishes a loop (shift+click flips
    its compartment instead -- see on_play_mouse). Both moves are always
    live; only whose turn it is (in versus) can block one of them."""
    s = self.session
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
