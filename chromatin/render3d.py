"""A small software 3D renderer built on pygame surfaces.

Aesthetic direction: scientific illustration rather than game graphics.
Beads are small, softly lit pastels on a dark ground. The backbone is a
near-white thread so it reads as a neutral scaffold. Loop bonds are fine
dashed lines with a small coloured dot at each anchor -- no glowing quads.
The confinement box is gone; a single faint horizon plane anchors the scene.

No OpenGL; no external dependencies beyond numpy and pygame-ce.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from . import theme, beadcolor, chainshape

# ------------------------------------------------------------------ palette
# Bead A/B, bond, hover and pending tones live in beadcolor.py (theme-aware
# -- see _COMPARTMENT_DARK/_COMPARTMENT_LIGHT, bond_color(), hover_color()
# and pending_color() there). The rest of this quiet, dark-INK_2-tuned
# palette is unique to this module.
_LOOP_COL= (110, 210, 165)   # sage green      -- loop anchor dots
_MARK    = (240, 200, 110)   # warm amber      -- cross-highlight from map


class Camera:
    def __init__(self, dist: float = 16.0):
        self.yaw   = 0.7
        self.pitch = 0.35
        self.dist  = dist
        self.center = np.zeros(3)
        self.focal  = 760.0

    def rot(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw),   math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return Rx @ Ry

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw   += dx * 0.008
        self.pitch  = max(-1.45, min(1.45, self.pitch + dy * 0.008))

    def zoom(self, amount: float) -> None:
        self.dist = max(5.0, min(60.0, self.dist * (0.90 ** amount)))

    def view(self, pts: np.ndarray) -> np.ndarray:
        v = (pts - self.center) @ self.rot().T
        v[:, 2] += self.dist
        return v

    def project(self, pts: np.ndarray, cx: float, cy: float):
        v = self.view(np.atleast_2d(pts))
        z = np.maximum(v[:, 2], 0.05)
        sx = cx + self.focal * v[:, 0] / z
        sy = cy - self.focal * v[:, 1] / z
        return np.stack([sx, sy], axis=1), v[:, 2]

    def unproject(self, sx: float, sy: float, depth: float,
                  cx: float, cy: float) -> np.ndarray:
        z  = max(depth, 0.05)
        vx = (sx - cx) * z / self.focal
        vy = -(sy - cy) * z / self.focal
        v  = np.array([vx, vy, z - self.dist])
        return self.rot().T @ v + self.center


# --------------------------------------------------------- orientation gizmo
_AXIS_COLORS = {
    "X": (222, 90, 90),     # red
    "Y": (96, 196, 120),    # green
    "Z": (92, 150, 232),    # blue
}


def draw_axis_gizmo(surf: pygame.Surface, rect: pygame.Rect, cam: "Camera") -> None:
    """Small orientation compass, bottom-left of the viewport: three short
    arms that rotate with the camera so it's always clear which way is
    which -- the same idea as the axis indicator in ChimeraX or Blender.
    Bottom-right: bottom-left is taken by the state/legend HUD in PLAY (see
    draw.draw_view_hud). Shared by both renderers (see render3d_gpu.py's
    identical overlay step)."""
    cx, cy = rect.right - 42, rect.bottom - 42
    arm = 20

    # faint backing disc so the gizmo reads against any backbone colour
    pad = arm + 14
    disc = pygame.Surface((pad * 2, pad * 2), pygame.SRCALPHA)
    theme.circle(disc, (*theme.PANEL_HI, 140), (pad, pad), pad)
    surf.blit(disc, (cx - pad, cy - pad))

    R = cam.rot()
    axes = []
    for name, vec in (("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0))):
        d = R @ np.array(vec)
        axes.append((d[2], name, d))
    axes.sort(key=lambda a: a[0], reverse=True)   # far arm first, near arm drawn on top

    fnt = theme.font(11, mono=True, bold=True)
    for depth_z, name, d in axes:
        tip = (cx + d[0] * arm, cy - d[1] * arm)
        near = depth_z < 0                 # arm points toward the viewer
        col = _AXIS_COLORS[name] if near else theme.lerp_col(_AXIS_COLORS[name], theme.PANEL_HI, 0.55)
        pygame.draw.line(surf, col, (cx, cy), tip, 3 if near else 2)
        r = 8 if near else 5
        theme.circle(surf, col, (int(tip[0]), int(tip[1])), r)
        lab = fnt.render(name, True, theme.TEXT if near else theme.TEXT_FAINT)
        surf.blit(lab, (tip[0] - lab.get_width() / 2, tip[1] - lab.get_height() / 2))
    theme.circle(surf, theme.TEXT_FAINT, (int(cx), int(cy)), 2)


# ------------------------------------------------------------ sphere cache
_SS = 2                       # supersample factor (2 is plenty for pastels)
_sphere_cache: dict[tuple, pygame.Surface] = {}
_LIGHT = np.array([-0.35, 0.65, 0.68])
_LIGHT /= np.linalg.norm(_LIGHT)


def _bake_sphere_rgba(size: int, color, fog: float, bg, shiny: bool = False) -> np.ndarray:
    """Lambert sphere as an (H, W, 4) uint8 RGBA array. Shared by the CPU
    sprite cache below and the GPU bead-texture cache in render3d_gpu.py, so
    both renderers bake the exact same look.

    shiny=False (default, used by the 'spheres' representation): a soft
    Lambert shade with just a gentle creamy glow at the centre.
    shiny=True (used for the 'chain' representation's tube joints): adds a
    tight, bright Blinn-Phong highlight on top, like a glossy plastic bead.
    """
    s = max(4, int(size)) * _SS
    r = s / 2.0
    y, x = np.mgrid[0:s, 0:s]
    xx = (x - r + 0.5) / r
    yy = (y - r + 0.5) / r
    d2 = xx * xx + yy * yy
    inside = d2 <= 1.0
    zz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
    ny = -yy
    ndl = np.clip(xx * _LIGHT[0] + ny * _LIGHT[1] + zz * _LIGHT[2], 0.0, 1.0)

    # softer shading: raise the ambient floor, narrow the diffuse range
    shade = 0.45 + 0.55 * ndl

    base = np.array(color, dtype=float)
    rgb  = base[None, None, :] * shade[:, :, None]

    if shiny:
        # Blinn-Phong: half-vector between the light and the (fixed, head-on)
        # view direction -- a small, bright, tight highlight reads as glossy.
        view = np.array([0.0, 0.0, 1.0])
        half = _LIGHT + view
        half /= np.linalg.norm(half)
        ndh  = np.clip(xx * half[0] + ny * half[1] + zz * half[2], 0.0, 1.0)
        spec = ndh ** 40
        rgb += 235.0 * spec[:, :, None]
    else:
        # very subtle inner glow at the lit centre instead of hard specular
        glow = np.exp(-d2 * 3.5) * 0.18
        rgb += 255.0 * glow[:, :, None]            # creamy highlight centre

    if fog > 0.0:
        rgb = rgb * (1.0 - fog) + np.array(bg, dtype=float)[None, None, :] * fog
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[:, :, None]], axis=2)

    out = max(4, int(size))
    if out == s:
        return rgba
    # nearest-ish downsample to the target size via pygame (keeps one code
    # path for the resize instead of depending on e.g. scipy/PIL here)
    surf = pygame.Surface((s, s), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, np.transpose(rgba[:, :, :3], (1, 0, 2)))
    pygame.surfarray.pixels_alpha(surf)[:, :] = alpha.T
    surf = pygame.transform.smoothscale(surf, (out, out))
    out_rgb = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    out_a = pygame.surfarray.array_alpha(surf).transpose(1, 0)
    return np.concatenate([out_rgb, out_a[:, :, None]], axis=2).astype(np.uint8)


def _bake_sphere(size: int, color, fog: float, bg, shiny: bool = False) -> pygame.Surface:
    """Lambert (or shiny) sphere as a pygame Surface (CPU-renderer sprite cache)."""
    rgba = _bake_sphere_rgba(size, color, fog, bg, shiny)
    out = max(4, int(size))
    surf = pygame.Surface((out, out), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, np.transpose(rgba[:, :, :3], (1, 0, 2)))
    pygame.surfarray.pixels_alpha(surf)[:, :] = rgba[:, :, 3].T
    return surf


def sphere_sprite(size: int, color, fog: float, bg=None, shiny: bool = False) -> pygame.Surface:
    bg = theme.INK_2 if bg is None else bg
    size = max(4, min(120, int(round(size / 2.0)) * 2))
    fb   = round(max(0.0, min(0.85, fog)) * 8) / 8.0
    # theme name is part of the key: the baked fog blend depends on bg, which
    # changes when the user switches theme in Settings.
    key  = (size, tuple(int(c) for c in color), fb, shiny, theme.CURRENT_THEME)
    surf = _sphere_cache.get(key)
    if surf is None:
        surf = _bake_sphere(size, color, fb, bg, shiny)
        _sphere_cache[key] = surf
    return surf


# --------------------------------------------------------------- fog helper
def _fog_of(depth: float, near: float, far: float) -> float:
    if far <= near:
        return 0.0
    # Fading toward a light INK_2 (near-white) desaturates colour much faster
    # than fading toward a dark one, so cap how far light themes fog out --
    # otherwise distant beads wash out to near-white instead of just dimming.
    cap = 0.40 if theme.is_light() else 0.72
    return float(np.clip((depth - near) / (far - near), 0.0, 1.0)) * cap

def _fog_col(col, fog: float) -> tuple:
    return theme.lerp_col(col, theme.INK_2, fog)


class PolymerView:
    """Draws the polymer chain into a rect of the screen.

    Changes from the previous version
    ----------------------------------
    * No confinement box -- just a faint horizon grid plane.
    * Backbone is a thin near-white thread (neutral against dark bg).
    * Loop bonds are dotted lines + small coloured dot at each anchor;
      no glow quad in the middle.
    * Beads are smaller and softer (bead_r 0.42 -> 0.30).
    * Pastel A/B colours (rose / periwinkle).
    * Lambert only -- no sharp specular, no CYAN rim.
    """

    def __init__(self):
        self.cam      = Camera()
        self.hover    : int | None = None
        self.pending  : int | None = None
        self.dragging : int | None = None
        self.mark     : tuple[int, ...] = ()
        self.bead_r   = 0.30          # smaller: polymer-like, not ball-pit
        self.color_mode = "compartment"   # "compartment" | "rainbow" | "loop"
        self.rep_mode   = "spheres"       # "spheres" | "chain" | "ribbon"

    def cycle_color_mode(self) -> str:
        self.color_mode = beadcolor.cycle(self.color_mode)
        return self.color_mode

    def cycle_rep_mode(self) -> str:
        self.rep_mode = chainshape.cycle(self.rep_mode)
        return self.rep_mode

    # ------------------------------------------------------------ picking
    def screen_positions(self, pos: np.ndarray, rect: pygame.Rect):
        return self.cam.project(pos, rect.centerx, rect.centery)

    def pick(self, pos: np.ndarray, rect: pygame.Rect,
             mx: int, my: int) -> int | None:
        pts, depth = self.screen_positions(pos, rect)
        best, bd = None, 1e18
        for i in range(len(pts)):
            if depth[i] <= 0.1:
                continue
            rad = max(5.0, self.cam.focal * self.bead_r / depth[i])
            d2  = (pts[i][0] - mx) ** 2 + (pts[i][1] - my) ** 2
            if d2 <= (rad + 6) ** 2 and depth[i] < bd:
                bd, best = depth[i], i
        return best

    # ---------------------------------------------------------- horizon
    def _draw_horizon(self, surf, rect, box):
        """A faint grid in the -Y plane only -- grounds the scene without
        the visual noise of a full wireframe cube."""
        g    = 4
        near = self.cam.dist - box * 1.8
        far  = self.cam.dist + box * 1.8
        for k in range(g + 1):
            tval = -box + 2 * box * k / g
            for seg in (
                ([tval, -box, -box], [tval, -box, box]),
                ([-box, -box, tval], [box, -box, tval]),
            ):
                pts, d = self.cam.project(
                    np.array(seg, dtype=float), rect.centerx, rect.centery)
                if d[0] <= 0.1 or d[1] <= 0.1:
                    continue
                f   = _fog_of((d[0] + d[1]) / 2, near, far)
                col = theme.lerp_col((38, 44, 62), theme.INK_2, f)
                pygame.draw.line(surf, col, pts[0], pts[1], 1)

    # --------------------------------------------------------------- draw
    def draw(self, surf: pygame.Surface, rect: pygame.Rect,
             pos: np.ndarray, types: np.ndarray, loops, box: float,
             *, e1=None, t: float = 0.0, show_index: bool = True,
             dim: bool = False, mouse=None, show_box: bool = True):

        prev = surf.get_clip()
        surf.set_clip(rect)

        if show_box:
            self._draw_horizon(surf, rect, box)

        pts, depth = self.screen_positions(pos, rect)
        near = self.cam.dist - box * 1.6
        far  = self.cam.dist + box * 1.6
        n    = len(pos)
        bead_cols = beadcolor.bead_colors(n, types, loops, e1, self.color_mode)

        prims = []   # (sort_depth, kind, payload)

        if self.rep_mode == "ribbon":
            # ---- ChimeraX-style ribbon: a Catmull-Rom spline through the
            # backbone (curves, not kinks), drawn as several width-strips
            # per segment so a soft highlight runs down the centreline --
            # the two things (chainshape.smooth_ribbon_path /
            # cross_section_shade) that make it read as a glossy, gently
            # rounded band instead of a flat painted strip. No separate bead
            # sprites -- the strip itself is the body (see "bead" below for
            # why rings/labels still work).
            side = chainshape.ribbon_frame(pos)
            hw = self.bead_r * 0.9
            path_pos, path_side, path_tan, path_col = chainshape.smooth_ribbon_path(
                pos, side, bead_cols)
            m = len(path_pos)
            levels = [self.screen_positions(path_pos + path_side * (hw * u), rect)
                      for u in chainshape.U_LEVELS]
            for i in range(m - 1):
                if (levels[0][1][i] <= 0.1 or levels[0][1][i + 1] <= 0.1 or
                        levels[-1][1][i] <= 0.1 or levels[-1][1][i + 1] <= 0.1):
                    continue
                normal = np.cross(path_tan[i] + path_tan[i + 1], path_side[i] + path_side[i + 1])
                nl = np.linalg.norm(normal)
                bend = chainshape.lit_fraction(normal / nl) if nl > 1e-9 else 0.7
                dm = (levels[1][1][i] + levels[1][1][i + 1] +
                      levels[2][1][i] + levels[2][1][i + 1]) / 4
                f  = _fog_of(dm, near, far)
                base = (path_col[i] + path_col[i + 1]) / 2
                for s in range(len(chainshape.U_LEVELS) - 1):
                    u_mid = (chainshape.U_LEVELS[s] + chainshape.U_LEVELS[s + 1]) / 2
                    shade = bend * chainshape.cross_section_shade(u_mid)
                    lit = tuple(min(255.0, c * shade) for c in base)
                    col = _fog_col(lit, f)
                    quad = [levels[s][0][i], levels[s][0][i + 1],
                            levels[s + 1][0][i + 1], levels[s + 1][0][i]]
                    prims.append((dm, "ribbon", (quad, col)))

        elif self.rep_mode == "chain":
            # ---- rigid chain: a thick, shiny tube -- gradient-coloured
            # quad per bond, rounded off at each bead by a glossy joint
            # sphere (drawn in the "bead" branch below).
            for i in range(n - 1):
                if depth[i] <= 0.1 or depth[i + 1] <= 0.1:
                    continue
                ri = max(3.0, self.cam.focal * self.bead_r / depth[i])
                rj = max(3.0, self.cam.focal * self.bead_r / depth[i + 1])
                mid_col = theme.lerp_col(bead_cols[i], bead_cols[i + 1], 0.5)
                for k, (c0, c1) in enumerate(((bead_cols[i], mid_col), (mid_col, bead_cols[i + 1]))):
                    t0, t1 = k / 2, (k + 1) / 2
                    a  = pts[i] * (1 - t0) + pts[i + 1] * t0
                    b  = pts[i] * (1 - t1) + pts[i + 1] * t1
                    dm = depth[i] * (1 - (t0+t1)/2) + depth[i+1] * ((t0+t1)/2)
                    w  = (ri * (1 - (t0+t1)/2) + rj * ((t0+t1)/2)) * 0.85
                    f  = _fog_of(dm, near, far)
                    col = _fog_col(theme.lerp_col(c0, c1, 0.5), f)
                    prims.append((dm, "tube", (a, b, w, col)))

        else:  # "spheres" -- thin near-white thread, split into halves for z-sort
            for i in range(n - 1):
                if depth[i] <= 0.1 or depth[i + 1] <= 0.1:
                    continue
                for k in range(2):
                    t0, t1 = k / 2, (k + 1) / 2
                    a  = pts[i] * (1 - t0) + pts[i + 1] * t0
                    b  = pts[i] * (1 - t1) + pts[i + 1] * t1
                    dm = depth[i] * (1 - (t0+t1)/2) + depth[i+1] * ((t0+t1)/2)
                    w  = max(1.0, self.cam.focal * 0.07 / dm)   # thinner than before
                    f  = _fog_of(dm, near, far)
                    col = _fog_col(beadcolor.bond_color(), f * 0.85)
                    prims.append((dm, "bond", (a, b, w, col)))

        # ---- loop bonds: dashed thin line + small anchor dots
        for (i, j) in loops:
            if depth[i] <= 0.1 or depth[j] <= 0.1:
                continue
            dm = (depth[i] + depth[j]) / 2
            f  = _fog_of(dm, near, far)
            col = _fog_col(_LOOP_COL if self.color_mode != "loop" else beadcolor.LOOP_DOMAIN_ON, f)
            w  = max(1.0, self.cam.focal * 0.045 / dm)
            prims.append((dm, "loop", (pts[i], pts[j], w, col, depth[i], depth[j], f, i, j)))

        # ---- beads
        for i in range(n):
            if depth[i] <= 0.1:
                continue
            prims.append((depth[i], "bead", i))

        prims.sort(key=lambda p: -p[0])

        for dm, kind, payload in prims:
            if kind == "bond":
                a, b, w, col = payload
                self._thick_line(surf, a, b, w, col)

            elif kind == "tube":
                a, b, w, col = payload
                self._thick_line(surf, a, b, w, col)
                self._shiny_highlight(surf, a, b, w, col)

            elif kind == "ribbon":
                quad, col = payload
                pygame.draw.polygon(surf, col, quad)

            elif kind == "loop":
                a, b, w, col, da, db, f, li, lj = payload
                # dashed line -- no glow quad
                self._dashed_line(surf, a, b, w, col)
                # small coloured dot at each anchor (drawn at bead layer depth,
                # but we do it here so they don't z-fight)
                dot_r = max(3, self.cam.focal * 0.055 / da)
                theme.circle(surf, col, (int(a[0]), int(a[1])), int(dot_r))
                dot_r = max(3, self.cam.focal * 0.055 / db)
                theme.circle(surf, col, (int(b[0]), int(b[1])), int(dot_r))

            else:  # bead
                i = payload
                d   = depth[i]
                rad = max(3.0, self.cam.focal * self.bead_r / d)
                f   = _fog_of(d, near, far)

                base = bead_cols[i]
                if dim:
                    base = theme.lerp_col(base, theme.INK_2, 0.5)

                # "ribbon": the strip itself is the body -- no bead sprite,
                # just the rings/labels below, drawn right on the ribbon.
                # "chain": a smaller, shiny joint sphere rounds off the tube.
                # "spheres": the original full-size bead.
                if self.rep_mode != "ribbon":
                    joint = self.rep_mode == "chain"
                    draw_rad = rad * 0.68 if joint else rad
                    spr = sphere_sprite(int(draw_rad * 2), base, f, shiny=joint)
                    surf.blit(spr,
                              (pts[i][0] - spr.get_width()  / 2,
                               pts[i][1] - spr.get_height() / 2))

                # interaction rings
                if i == self.pending:
                    pulse = 0.5 + 0.5 * math.sin(t * 6.5)
                    self._ring(surf, pts[i], rad + 3 + 2 * pulse, beadcolor.pending_color(), 2)
                elif i in self.mark:
                    pulse = 0.5 + 0.5 * math.sin(t * 6.5)
                    self._ring(surf, pts[i], rad + 4 + 2 * pulse, _MARK, 2)
                    self._ring(surf, pts[i], rad + 1, _MARK, 1)
                elif i == self.hover:
                    self._ring(surf, pts[i], rad + 3, beadcolor.hover_color(), 2)

                # index label: only when big enough and not too many beads
                if show_index and n <= 80 and rad > 9:
                    # ribbon mode has no bead sprite -- give the number a small
                    # disc to sit inside, same as every other rep mode
                    if self.rep_mode == "ribbon":
                        theme.circle(surf, _fog_col(base, f), pts[i], rad * 0.62)
                    fnt = theme.font(max(10, int(rad * 0.8)), mono=True, bold=True)
                    lab = fnt.render(str(i), True, (255, 255, 255))
                    lab.set_alpha(int(220 * (1 - f)))
                    sh  = fnt.render(str(i), True, (0, 0, 0))
                    sh.set_alpha(int(130 * (1 - f)))
                    ox  = pts[i][0] - lab.get_width()  / 2
                    oy  = pts[i][1] - lab.get_height() / 2
                    surf.blit(sh,  (ox + 1, oy + 1))
                    surf.blit(lab, (ox,     oy))

        # rubber-band while choosing the second anchor
        if self.pending is not None and depth[self.pending] > 0.1 and mouse is not None:
            mx, my = mouse
            if rect.collidepoint(mx, my):
                self._dashed(surf, pts[self.pending], (mx, my), beadcolor.pending_color(), t)

        if self.color_mode == "rainbow":
            beadcolor.draw_colorbar(surf, rect, n)

        if show_box:
            draw_axis_gizmo(surf, rect, self.cam)

        surf.set_clip(prev)

    # --------------------------------------------------------- primitives
    @staticmethod
    def _thick_line(surf, a, b, w, col):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.5:
            return
        px, py = -dy / L * w / 2, dx / L * w / 2
        quad = [
            (a[0] + px, a[1] + py), (b[0] + px, b[1] + py),
            (b[0] - px, b[1] - py), (a[0] - px, a[1] - py),
        ]
        pygame.draw.polygon(surf, col, quad)

    @staticmethod
    def _shiny_highlight(surf, a, b, w, col):
        """A bright streak along one edge of a 'chain' tube segment -- the
        2D trick that reads as a glossy, lit cylinder without real 3D
        lighting. The side is picked from a fixed on-screen light direction
        so it stays consistent as the segment's own direction changes."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.5:
            return
        px, py = -dy / L, dx / L
        if px * -0.4 + py * -0.9 < 0:      # keep the highlight on the lit side
            px, py = -px, -py
        off = w * 0.22
        hl  = theme.lerp_col(col, (255, 255, 255), 0.55)
        hw  = max(1.0, w * 0.22)
        PolymerView._thick_line(surf,
                                (a[0] + px * off, a[1] + py * off),
                                (b[0] + px * off, b[1] + py * off),
                                hw, hl)

    @staticmethod
    def _dashed_line(surf, a, b, w, col):
        """Short dashes for loop bonds -- visually distinct from backbone."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 2:
            return
        ux, uy = dx / L, dy / L
        dash, gap = 7, 5
        d = 0.0
        while d < L:
            e = min(d + dash, L)
            PolymerView._thick_line(
                surf,
                (a[0] + ux * d,  a[1] + uy * d),
                (a[0] + ux * e,  a[1] + uy * e),
                max(1.0, w), col)
            d += dash + gap

    @staticmethod
    def _ring(surf, p, r, col, w):
        r = int(max(2, r))
        s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        theme.circle(s, (*col, 210), (r + 2, r + 2), r, w)
        surf.blit(s, (p[0] - r - 2, p[1] - r - 2))

    @staticmethod
    def _dashed(surf, a, b, col, t):
        """Animated dashed rubber-band while picking the second anchor."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 2:
            return
        ux, uy = dx / L, dy / L
        step = 8
        off  = (t * 38) % (step * 2)
        d    = off
        while d < L:
            e = min(d + step, L)
            pygame.draw.line(surf, col,
                             (a[0] + ux * d,  a[1] + uy * d),
                             (a[0] + ux * e,  a[1] + uy * e), 2)
            d += step * 2