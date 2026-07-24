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

from . import theme

# ------------------------------------------------------------------ palette
# Pastel tones that sit quietly on INK_2 without fighting the Hi-C maps.
_BEAD_A  = (210, 130, 140)   # soft rose       -- A-type / E1 > 0
_BEAD_B  = (120, 150, 210)   # periwinkle      -- B-type / E1 < 0
_BOND    = (210, 215, 230)   # near-white      -- backbone thread
_LOOP_COL= (110, 210, 165)   # sage green      -- loop anchor dots
_HOVER   = (140, 210, 255)   # light blue      -- hover ring
_PENDING = (160, 240, 180)   # mint            -- pending anchor pulse
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


# ------------------------------------------------------------ sphere cache
_SS = 2                       # supersample factor (2 is plenty for pastels)
_sphere_cache: dict[tuple, pygame.Surface] = {}
_LIGHT = np.array([-0.35, 0.65, 0.68])
_LIGHT /= np.linalg.norm(_LIGHT)


def _bake_sphere(size: int, color, fog: float, bg) -> pygame.Surface:
    """Soft Lambert sphere -- no harsh specular, just a gentle highlight."""
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

    # very subtle inner glow at the lit centre instead of hard specular
    glow  = np.exp(-d2 * 3.5) * 0.18

    base = np.array(color, dtype=float)
    rgb  = base[None, None, :] * shade[:, :, None]
    rgb += 255.0 * glow[:, :, None]            # creamy highlight centre
    if fog > 0.0:
        rgb = rgb * (1.0 - fog) + np.array(bg, dtype=float)[None, None, :] * fog
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    surf = pygame.Surface((s, s), pygame.SRCALPHA)
    pygame.surfarray.blit_array(surf, np.transpose(rgb, (1, 0, 2)))
    alpha = np.where(inside, 255, 0).astype(np.uint8)
    pygame.surfarray.pixels_alpha(surf)[:, :] = alpha.T
    out = max(4, int(size))
    return pygame.transform.smoothscale(surf, (out, out))


def sphere_sprite(size: int, color, fog: float, bg=theme.INK_2) -> pygame.Surface:
    size = max(4, min(120, int(round(size / 2.0)) * 2))
    fb   = round(max(0.0, min(0.85, fog)) * 8) / 8.0
    key  = (size, tuple(int(c) for c in color), fb)
    surf = _sphere_cache.get(key)
    if surf is None:
        surf = _bake_sphere(size, color, fb, bg)
        _sphere_cache[key] = surf
    return surf


# --------------------------------------------------------------- fog helper
def _fog_of(depth: float, near: float, far: float) -> float:
    if far <= near:
        return 0.0
    return float(np.clip((depth - near) / (far - near), 0.0, 1.0)) * 0.72

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

        prims = []   # (sort_depth, kind, payload)

        # ---- backbone: thin near-white thread, split into halves for z-sort
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
                col = _fog_col(_BOND, f * 0.85)
                prims.append((dm, "bond", (a, b, w, col)))

        # ---- loop bonds: dashed thin line + small anchor dots
        for (i, j) in loops:
            if depth[i] <= 0.1 or depth[j] <= 0.1:
                continue
            dm = (depth[i] + depth[j]) / 2
            f  = _fog_of(dm, near, far)
            col = _fog_col(_LOOP_COL, f)
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

            elif kind == "loop":
                a, b, w, col, da, db, f, li, lj = payload
                # dashed line -- no glow quad
                self._dashed_line(surf, a, b, w, col)
                # small coloured dot at each anchor (drawn at bead layer depth,
                # but we do it here so they don't z-fight)
                dot_r = max(3, self.cam.focal * 0.055 / da)
                pygame.draw.circle(surf, col, (int(a[0]), int(a[1])), int(dot_r))
                dot_r = max(3, self.cam.focal * 0.055 / db)
                pygame.draw.circle(surf, col, (int(b[0]), int(b[1])), int(dot_r))

            else:  # bead
                i = payload
                d   = depth[i]
                rad = max(3.0, self.cam.focal * self.bead_r / d)
                f   = _fog_of(d, near, far)

                # base colour: pastel A or B, tinted by E1 amplitude if present
                if types[i] > 0:
                    base = _BEAD_A
                else:
                    base = _BEAD_B
                if e1 is not None and len(e1) == n:
                    amp  = 0.4 + 0.6 * min(1.0, abs(float(e1[i])))
                    grey = (110, 115, 130)
                    base = theme.lerp_col(grey, base, amp)
                if dim:
                    base = theme.lerp_col(base, theme.INK_2, 0.5)

                spr = sphere_sprite(int(rad * 2), base, f)
                surf.blit(spr,
                          (pts[i][0] - spr.get_width()  / 2,
                           pts[i][1] - spr.get_height() / 2))

                # interaction rings
                if i == self.pending:
                    pulse = 0.5 + 0.5 * math.sin(t * 6.5)
                    self._ring(surf, pts[i], rad + 3 + 2 * pulse, _PENDING, 2)
                elif i in self.mark:
                    pulse = 0.5 + 0.5 * math.sin(t * 6.5)
                    self._ring(surf, pts[i], rad + 4 + 2 * pulse, _MARK, 2)
                    self._ring(surf, pts[i], rad + 1, _MARK, 1)
                elif i == self.hover:
                    self._ring(surf, pts[i], rad + 3, _HOVER, 2)

                # index label: only when big enough and not too many beads
                if show_index and n <= 80 and rad > 9:
                    fnt = theme.font(max(9, int(rad * 0.75)), mono=True, bold=True)
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
                self._dashed(surf, pts[self.pending], (mx, my), _PENDING, t)

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
        pygame.draw.circle(s, (*col, 210), (r + 2, r + 2), r, w)
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