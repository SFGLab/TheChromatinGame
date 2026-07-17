"""A small software 3D renderer built on pygame surfaces.

No OpenGL: at N <= 56 beads a painter's-algorithm pass over pre-shaded sprites
is fast, dependency-free, and lets every pixel be art-directed. Beads are
Lambert+specular spheres baked once per (colour, radius, fog) bucket; bonds are
depth-split screen-space quads so they interleave correctly with the beads.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from . import theme


class Camera:
    def __init__(self, dist: float = 16.0):
        self.yaw = 0.7
        self.pitch = 0.35
        self.dist = dist
        self.center = np.zeros(3)
        self.focal = 760.0

    def rot(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return Rx @ Ry

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.008
        self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.008))

    def zoom(self, amount: float) -> None:
        self.dist = max(5.0, min(60.0, self.dist * (0.90 ** amount)))

    def view(self, pts: np.ndarray) -> np.ndarray:
        v = (pts - self.center) @ self.rot().T
        v[:, 2] += self.dist
        return v

    def project(self, pts: np.ndarray, cx: float, cy: float):
        """world -> (screen xy, depth). Depth <= near is flagged by depth value."""
        v = self.view(np.atleast_2d(pts))
        z = np.maximum(v[:, 2], 0.05)
        sx = cx + self.focal * v[:, 0] / z
        sy = cy - self.focal * v[:, 1] / z
        return np.stack([sx, sy], axis=1), v[:, 2]

    def unproject(self, sx: float, sy: float, depth: float, cx: float, cy: float) -> np.ndarray:
        z = max(depth, 0.05)
        vx = (sx - cx) * z / self.focal
        vy = -(sy - cy) * z / self.focal
        v = np.array([vx, vy, z - self.dist])
        return self.rot().T @ v + self.center


# --------------------------------------------------------------- sphere cache
_SS = 3          # supersample factor for antialiased edges
_sphere_cache: dict[tuple, pygame.Surface] = {}
_LIGHT = np.array([-0.45, 0.60, 0.68])
_LIGHT /= np.linalg.norm(_LIGHT)
_HALF = _LIGHT + np.array([0.0, 0.0, 1.0])
_HALF /= np.linalg.norm(_HALF)


def _bake_sphere(size: int, color, fog: float, bg) -> pygame.Surface:
    s = max(4, int(size)) * _SS
    r = s / 2.0
    y, x = np.mgrid[0:s, 0:s]
    xx = (x - r + 0.5) / r
    yy = (y - r + 0.5) / r
    d2 = xx * xx + yy * yy
    inside = d2 <= 1.0
    zz = np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
    ny = -yy                                      # screen y is down; light in world-up
    ndl = np.clip(xx * _LIGHT[0] + ny * _LIGHT[1] + zz * _LIGHT[2], 0.0, 1.0)
    ndh = np.clip(xx * _HALF[0] + ny * _HALF[1] + zz * _HALF[2], 0.0, 1.0)
    spec = ndh ** 36
    rim = (1.0 - zz) ** 3

    base = np.array(color, dtype=float)
    shade = 0.24 + 0.80 * ndl
    rgb = base[None, None, :] * shade[:, :, None]
    rgb += 235.0 * spec[:, :, None] * 0.60
    rgb += np.array(theme.CYAN, dtype=float)[None, None, :] * (rim * 0.30)[:, :, None]
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
    size = max(4, min(150, int(round(size / 2.0)) * 2))
    fb = round(max(0.0, min(0.85, fog)) * 8) / 8.0
    key = (size, tuple(int(c) for c in color), fb)
    surf = _sphere_cache.get(key)
    if surf is None:
        surf = _bake_sphere(size, color, fb, bg)
        _sphere_cache[key] = surf
    return surf


_glow_cache: dict[tuple, pygame.Surface] = {}


def glow_sprite(size: int, color) -> pygame.Surface:
    size = max(6, min(220, int(round(size / 4.0)) * 4))
    key = (size, tuple(int(c) for c in color))
    g = _glow_cache.get(key)
    if g is not None:
        return g
    s = size
    r = s / 2.0
    y, x = np.mgrid[0:s, 0:s]
    d = np.sqrt(((x - r + 0.5) / r) ** 2 + ((y - r + 0.5) / r) ** 2)
    a = np.clip(1.0 - d, 0.0, 1.0) ** 2.6
    surf = pygame.Surface((s, s), pygame.SRCALPHA)
    rgb = np.empty((s, s, 3), dtype=np.uint8)
    rgb[:, :, :] = np.array(color, dtype=np.uint8)[None, None, :]
    pygame.surfarray.blit_array(surf, np.transpose(rgb, (1, 0, 2)))
    pygame.surfarray.pixels_alpha(surf)[:, :] = np.transpose((a * 255).astype(np.uint8))
    _glow_cache[key] = surf
    return surf


# ------------------------------------------------------------------- helpers
def _fog_of(depth: float, near: float, far: float) -> float:
    if far <= near:
        return 0.0
    return float(np.clip((depth - near) / (far - near), 0.0, 1.0)) * 0.78


class PolymerView:
    """Draws the chain, its box and its bonds into a rect of the screen."""

    def __init__(self):
        self.cam = Camera()
        self.hover: int | None = None
        self.pending: int | None = None   # first anchor of a loop being drawn
        self.dragging: int | None = None
        self.mark: tuple[int, ...] = ()   # cross-highlight from the heatmap
        self.bead_r = 0.42

    # ------------------------------------------------------------- picking
    def screen_positions(self, pos: np.ndarray, rect: pygame.Rect):
        return self.cam.project(pos, rect.centerx, rect.centery)

    def pick(self, pos: np.ndarray, rect: pygame.Rect, mx: int, my: int) -> int | None:
        pts, depth = self.screen_positions(pos, rect)
        best, bd = None, 1e18
        for i in range(len(pts)):
            if depth[i] <= 0.1:
                continue
            rad = max(5.0, self.cam.focal * self.bead_r / depth[i])
            d2 = (pts[i][0] - mx) ** 2 + (pts[i][1] - my) ** 2
            if d2 <= (rad + 4) ** 2 and depth[i] < bd:
                bd, best = depth[i], i
        return best

    # -------------------------------------------------------------- drawing
    def _draw_box(self, surf, rect, box, order_far_first=True):
        c = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float) * box
        pts, depth = self.cam.project(c, rect.centerx, rect.centery)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        near, far = self.cam.dist - box * 1.8, self.cam.dist + box * 1.8
        items = []
        for a, b in edges:
            if depth[a] <= 0.1 or depth[b] <= 0.1:
                continue
            dm = (depth[a] + depth[b]) / 2
            items.append((dm, a, b))
        items.sort(key=lambda t: -t[0] if order_far_first else t[0])
        for dm, a, b in items:
            f = _fog_of(dm, near, far)
            col = theme.lerp_col(theme.RULE, theme.INK_2, f * 0.9)
            pygame.draw.line(surf, col, pts[a], pts[b], 1)
        # floor grid on the -Y face, quietly
        g = 4
        for k in range(g + 1):
            t = -box + 2 * box * k / g
            for seg in (([t, -box, -box], [t, -box, box]), ([-box, -box, t], [box, -box, t])):
                p, d = self.cam.project(np.array(seg, dtype=float), rect.centerx, rect.centery)
                if d[0] <= 0.1 or d[1] <= 0.1:
                    continue
                f = _fog_of((d[0] + d[1]) / 2, near, far)
                pygame.draw.line(surf, theme.lerp_col(theme.RULE_SOFT, theme.INK_2, f), p[0], p[1], 1)

    def draw(self, surf: pygame.Surface, rect: pygame.Rect, pos: np.ndarray,
             types: np.ndarray, loops, box: float, *, e1=None, t: float = 0.0,
             show_index: bool = True, dim: bool = False, mouse=None,
             show_box: bool = True):
        prev = surf.get_clip()
        surf.set_clip(rect)
        if show_box:
            self._draw_box(surf, rect, box)

        pts, depth = self.screen_positions(pos, rect)
        near = self.cam.dist - box * 1.6
        far = self.cam.dist + box * 1.6
        n = len(pos)

        prims = []   # (depth, kind, payload)

        # backbone: split each bond so it sorts against beads properly
        for i in range(n - 1):
            if depth[i] <= 0.1 or depth[i + 1] <= 0.1:
                continue
            for k in range(2):
                t0, t1 = k / 2, (k + 1) / 2
                a = pts[i] * (1 - t0) + pts[i + 1] * t0
                b = pts[i] * (1 - t1) + pts[i + 1] * t1
                dm = depth[i] * (1 - (t0 + t1) / 2) + depth[i + 1] * ((t0 + t1) / 2)
                w = max(1.5, self.cam.focal * 0.13 / dm)
                ca = theme.COMP_A if types[i] > 0 else theme.COMP_B
                cb = theme.COMP_A if types[i + 1] > 0 else theme.COMP_B
                col = theme.lerp_col(ca, cb, (t0 + t1) / 2)
                col = theme.lerp_col(col, (150, 160, 190), 0.45)
                prims.append((dm, "bond", (a, b, w, col, _fog_of(dm, near, far))))

        # loop bonds
        for (i, j) in loops:
            if depth[i] <= 0.1 or depth[j] <= 0.1:
                continue
            dm = (depth[i] + depth[j]) / 2
            w = max(1.5, self.cam.focal * 0.075 / dm)
            prims.append((dm, "loop", (pts[i], pts[j], w, _fog_of(dm, near, far))))

        # beads
        for i in range(n):
            if depth[i] <= 0.1:
                continue
            prims.append((depth[i], "bead", i))

        prims.sort(key=lambda p: -p[0])

        for dm, kind, payload in prims:
            if kind == "bond":
                a, b, w, col, f = payload
                col = theme.lerp_col(col, theme.INK_2, f)
                self._thick_line(surf, a, b, w, col)
            elif kind == "loop":
                a, b, w, f = payload
                col = theme.lerp_col(theme.GREEN, theme.INK_2, f)
                g = glow_sprite(int(max(10, w * 7)), theme.GREEN)
                mid = (a + b) / 2
                surf.blit(g, (mid[0] - g.get_width() / 2, mid[1] - g.get_height() / 2),
                          special_flags=pygame.BLEND_ADD)
                self._thick_line(surf, a, b, w, col)
            else:
                i = payload
                d = depth[i]
                rad = max(4.0, self.cam.focal * self.bead_r / d)
                f = _fog_of(d, near, far)
                base = theme.COMP_A if types[i] > 0 else theme.COMP_B
                if e1 is not None and len(e1) == n:
                    amp = 0.45 + 0.55 * min(1.0, abs(float(e1[i])))
                    base = theme.lerp_col((90, 96, 118), base, amp)
                if dim:
                    base = theme.lerp_col(base, theme.INK_2, 0.45)
                spr = sphere_sprite(int(rad * 2), base, f)
                surf.blit(spr, (pts[i][0] - spr.get_width() / 2,
                                pts[i][1] - spr.get_height() / 2))
                if i == self.pending:
                    pulse = 0.55 + 0.45 * math.sin(t * 7.0)
                    self._ring(surf, pts[i], rad + 4 + 2 * pulse, theme.GREEN, 2)
                elif i in self.mark:
                    pulse = 0.55 + 0.45 * math.sin(t * 7.0)
                    self._ring(surf, pts[i], rad + 5 + 3 * pulse, theme.AMBER, 2)
                    self._ring(surf, pts[i], rad + 2, theme.AMBER, 1)
                elif i == self.hover:
                    self._ring(surf, pts[i], rad + 3, theme.CYAN, 2)
                if show_index and rad > 9:
                    fnt = theme.font(max(9, int(rad * 0.82)), mono=True, bold=True)
                    lab = fnt.render(str(i), True, (255, 255, 255))
                    lab.set_alpha(int(235 * (1 - f)))
                    sh = fnt.render(str(i), True, (0, 0, 0))
                    sh.set_alpha(int(150 * (1 - f)))
                    surf.blit(sh, (pts[i][0] - lab.get_width() / 2 + 1,
                                   pts[i][1] - lab.get_height() / 2 + 1))
                    surf.blit(lab, (pts[i][0] - lab.get_width() / 2,
                                    pts[i][1] - lab.get_height() / 2))

        # rubber band while choosing the second anchor
        if self.pending is not None and depth[self.pending] > 0.1:
            mx, my = mouse if mouse is not None else pygame.mouse.get_pos()
            if rect.collidepoint(mx, my):
                self._dashed(surf, pts[self.pending], (mx, my), theme.GREEN, t)

        surf.set_clip(prev)

    # ---------------------------------------------------------- primitives
    @staticmethod
    def _thick_line(surf, a, b, w, col):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 0.5:
            return
        px, py = -dy / L * w / 2, dx / L * w / 2
        quad = [(a[0] + px, a[1] + py), (b[0] + px, b[1] + py),
                (b[0] - px, b[1] - py), (a[0] - px, a[1] - py)]
        pygame.draw.polygon(surf, col, quad)

    @staticmethod
    def _ring(surf, p, r, col, w):
        r = int(max(2, r))
        s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*col, 235), (r + 2, r + 2), r, w)
        surf.blit(s, (p[0] - r - 2, p[1] - r - 2))

    @staticmethod
    def _dashed(surf, a, b, col, t):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 2:
            return
        ux, uy = dx / L, dy / L
        step = 9
        off = (t * 40) % (step * 2)
        d = off
        while d < L:
            e = min(d + step, L)
            pygame.draw.line(surf, col, (a[0] + ux * d, a[1] + uy * d),
                             (a[0] + ux * e, a[1] + uy * e), 2)
            d += step * 2
