"""GPU-accelerated polymer view, backed by Panda3D, offscreen-rendered into
the existing pygame window.

Design note -- why this is low-risk despite swapping the renderer:
All of the actual 3D math (orbit camera, perspective projection, fog,
mouse picking) stays in the exact same `Camera` class from render3d.py,
completely unchanged. That class already reduces every bead to a cheap,
pure-NumPy (screen_x, screen_y, depth) triple every frame -- that part was
never the bottleneck. What was expensive was turning a few hundred of
those into pixels by hand: baking/blitting sphere sprites and drawing,
dashing and z-sorting line segments in pure Python, every frame.

So Panda3D is used here purely as a fast 2D compositor: an orthographic
camera over a plane in *pixel* coordinates, drawing GPU-textured billboard
quads for beads (the same Lambert-shaded sprite bitmap as before, just
uploaded to the GPU once per colour and reused, instead of CPU-blitted
every frame) and batched GPU line geometry for bonds. The back-to-front
draw order is still decided on the CPU exactly as before (cheap -- a sort
of a few hundred numbers) and handed to Panda3D as explicit draw-order
bins, so the result looks the same, it is just rasterised by the GPU
instead of pygame's software blitter.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from . import theme, colormaps, beadcolor, chainshape
from .render3d import Camera, _bake_sphere_rgba, _fog_of, _fog_col, draw_axis_gizmo

try:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        loadPrcFileData, CardMaker, Texture, TransparencyAttrib,
        GraphicsOutput, LineSegs, NodePath, SamplerState,
        GeomVertexFormat, GeomVertexData, GeomVertexWriter,
        GeomTriangles, Geom, GeomNode,
    )
    HAVE_PANDA = True
except Exception:                      # pragma: no cover -- panda3d optional
    HAVE_PANDA = False

# Bond colour lives in beadcolor.py (theme-aware -- see bond_color()).
_LOOP_COL = (110, 210, 165)


# ------------------------------------------------------------- shared app
class _PandaApp:
    """One Panda3D ShowBase for the whole process (Panda3D is a singleton by
    design). Each PolymerView owns a private NodePath subtree under here and
    is shown/hidden on demand so three logical views (play / lab / menu demo)
    can share one GPU context without paying for three."""

    def __init__(self, w: int, h: int):
        loadPrcFileData("", """
            window-type none
            load-display pandagl
            aux-display p3tinydisplay
            audio-library-name null
            sync-video 0
            model-cache-dir
            notify-level fatal
        """)
        # ShowBase with window-type none creates no window/buffer of its own
        # -- we make our own, explicitly resizeable, offscreen buffer below.
        self.base = ShowBase()
        self.w, self.h = w, h

        from panda3d.core import (
            GraphicsPipe, GraphicsPipeSelection, FrameBufferProperties,
            WindowProperties, OrthographicLens, AmbientLight,
        )
        fb_props = FrameBufferProperties()
        fb_props.setRgbaBits(8, 8, 8, 8)
        fb_props.setDepthBits(24)
        win_props = WindowProperties.size(w, h)
        flags = GraphicsPipe.BFRefuseWindow | GraphicsPipe.BFResizeable

        # Try the configured hardware pipe first (pandagl -- real GPU on the
        # player's machine); if it can't actually create a buffer (e.g. no
        # display/driver at all, as in a headless test container), fall back
        # to Panda's bundled software rasterizer explicitly. `aux-display`
        # only helps pipe *selection*, not a failed makeOutput, so this
        # fallback is done by hand.
        sel = GraphicsPipeSelection.getGlobalPtr()
        self.pipe = self.base.pipe or sel.makeDefaultPipe()
        self.buf = None
        if self.pipe is not None:
            self.buf = self.base.graphicsEngine.makeOutput(
                self.pipe, "chromatin-offscreen", 0, fb_props, win_props, flags)
        if self.buf is None:
            self.pipe = sel.makeModulePipe("p3tinydisplay")
            if self.pipe is not None:
                self.buf = self.base.graphicsEngine.makeOutput(
                    self.pipe, "chromatin-offscreen-sw", 0, fb_props, win_props, flags)
        if self.buf is None:
            raise RuntimeError("Panda3D could not create an offscreen buffer "
                                "(no usable GL driver found, not even software)")

        self.tex = Texture()
        self.tex.setFormat(Texture.FRgba8)
        self.buf.addRenderTexture(self.tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)

        self.lens = OrthographicLens()
        self.lens.setNearFar(-1.0, 100000.0)
        from panda3d.core import Camera as PCamera
        cam_node = PCamera("chromatin-cam")
        cam_node.setLens(self.lens)
        self.cam_np = self.base.render.attachNewNode(cam_node)
        self.dr = self.buf.makeDisplayRegion()
        self.dr.setCamera(self.cam_np)

        amb = AmbientLight("amb")
        amb.setColor((1, 1, 1, 1))     # shading is pre-baked into sprites
        self.base.render.setLight(self.base.render.attachNewNode(amb))

        self._set_lens(w, h)
        self._views = []   # registered PolymerView roots

    def _set_lens(self, w, h):
        self.lens.setFilmSize(w, h)
        # camera looks down +Y in Panda's Z-up world; geometry is placed at
        # (screen_x, depth, -screen_y) so screen-space x/y map to Panda x/z
        # and our precomputed depth maps to Panda's y (forward) axis.
        self.cam_np.setPos(w / 2.0, -1000.0, -h / 2.0)
        self.cam_np.setHpr(0, 0, 0)

    def resize(self, w: int, h: int):
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = w, h
        self.buf.setSize(w, h)
        self._set_lens(w, h)

    def register(self, view: "PandaPolymerView"):
        self._views.append(view)

    def activate(self, view: "PandaPolymerView"):
        for v in self._views:
            v.root.hide() if v is not view else v.root.show()

    def render_to(self, surf: pygame.Surface, rect: pygame.Rect):
        """Render one frame and blit the result into `surf` at `rect`."""
        self.base.graphicsEngine.renderFrame()
        self.base.graphicsEngine.syncFrame()
        if not self.tex.hasRamImage():
            return
        w, h = self.tex.getXSize(), self.tex.getYSize()
        buf = bytes(self.tex.getRamImageAs("RGBA"))
        img = pygame.image.frombuffer(buf, (w, h), "RGBA")
        img = pygame.transform.flip(img, False, True)   # Panda's RAM image is bottom-up
        if (w, h) != (rect.width, rect.height):
            img = pygame.transform.smoothscale(img, (rect.width, rect.height))
        surf.blit(img, rect.topleft)


_app: _PandaApp | None = None


def _ensure_app(w: int, h: int) -> _PandaApp:
    global _app
    if _app is None:
        _app = _PandaApp(w, h)
    else:
        _app.resize(w, h)
    return _app


# ---------------------------------------------------------- sprite cache
_tex_cache: dict[tuple, "Texture"] = {}


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _bead_texture(size: int, color, fog: float, shiny: bool = False):
    """Bucket to a handful of distinct (power-of-two, colour, fog) textures
    and cache them -- NPOT textures aren't reliably supported by every GL
    driver, so we always bake at a power-of-two resolution and let the
    on-screen quad (card.setScale) be whatever size is actually needed.
    shiny=True bakes the glossy variant used for 'chain' tube joints."""
    size = max(4, min(120, int(round(size / 2.0)) * 2))
    tex_size = _next_pow2(size)
    fb = round(max(0.0, min(0.85, fog)) * 8) / 8.0
    # theme name is part of the key: baked against theme.INK_2, which changes
    # when the user switches theme in Settings.
    key = (tex_size, tuple(int(c) for c in color), fb, shiny, theme.CURRENT_THEME)
    tex = _tex_cache.get(key)
    if tex is None:
        rgba = _bake_sphere_rgba(tex_size, color, fb, theme.INK_2, shiny)   # (H,W,4) uint8 RGBA
        bgra = rgba[:, :, [2, 1, 0, 3]]            # Texture.FRgba8's RAM layout is B,G,R,A
        bgra = np.ascontiguousarray(bgra[::-1])    # Panda's RAM image is bottom-up
        tex = Texture()
        tex.setup2dTexture(rgba.shape[1], rgba.shape[0], Texture.TUnsignedByte, Texture.FRgba8)
        tex.setRamImage(bgra.tobytes())
        tex.setMagfilter(SamplerState.FTLinear)
        tex.setMinfilter(SamplerState.FTLinear)
        _tex_cache[key] = tex
    return tex


# -------------------------------------------------------------- the view
class PandaPolymerView:
    """GPU-backed drop-in replacement for render3d.PolymerView. Same public
    surface (cam/hover/pending/dragging/mark/bead_r, screen_positions, pick,
    draw) so draw.py / interact.py need no changes beyond construction."""

    def __init__(self):
        app = _ensure_app(960, 720)
        self.cam = Camera()
        self.hover: int | None = None
        self.pending: int | None = None
        self.dragging: int | None = None
        self.mark: tuple[int, ...] = ()
        self.bead_r = 0.30
        self.color_mode = "compartment"
        self.rep_mode = "spheres"

        self.root = NodePath("polymer-view")
        self.root.reparentTo(app.base.render)
        app.register(self)

        self._bead_cards: list[NodePath] = []
        self._line_np: NodePath | None = None
        self._mesh_np: NodePath | None = None   # ribbon triangle mesh, rebuilt per frame
        self._cm = CardMaker("bead")
        self._cm.setFrame(-0.5, 0.5, -0.5, 0.5)

    # ------------------------------------------------------------ picking
    # identical to the software renderer -- same Camera, same formulas.
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
            if d2 <= (rad + 6) ** 2 and depth[i] < bd:
                bd, best = depth[i], i
        return best

    # --------------------------------------------------------- bead pool
    def _ensure_beads(self, n: int):
        while len(self._bead_cards) < n:
            np_card = NodePath(self._cm.generate())
            np_card.reparentTo(self.root)
            np_card.setTransparency(TransparencyAttrib.MAlpha)
            np_card.setBillboardAxis()
            self._bead_cards.append(np_card)
        for i in range(n, len(self._bead_cards)):
            self._bead_cards[i].hide()

    # ------------------------------------------------------------ colour
    def cycle_color_mode(self) -> str:
        self.color_mode = beadcolor.cycle(self.color_mode)
        return self.color_mode

    def cycle_rep_mode(self) -> str:
        self.rep_mode = chainshape.cycle(self.rep_mode)
        return self.rep_mode

    # --------------------------------------------------------------- draw
    def draw(self, surf: pygame.Surface, rect: pygame.Rect,
              pos: np.ndarray, types: np.ndarray, loops, box: float,
              *, e1=None, t: float = 0.0, show_index: bool = True,
              dim: bool = False, mouse=None, show_box: bool = True):

        app = _ensure_app(rect.width, rect.height)
        app.activate(self)

        pts, depth = self.screen_positions(pos, rect)
        near = self.cam.dist - box * 1.6
        far = self.cam.dist + box * 1.6
        n = len(pos)
        self._ensure_beads(n)
        colors = beadcolor.bead_colors(n, types, loops, e1, self.color_mode)

        # ---- backbone + loop anchors, one batched LineSegs -- only the
        # "spheres" representation uses this for its body (a thin line);
        # "chain" and "ribbon" draw real filled triangles below instead,
        # since a GL line's on-screen width isn't reliable enough across
        # drivers to read as a solid tube. Loop bonds are always drawn here.
        ls = LineSegs("bonds")
        if self.rep_mode == "spheres":
            for i in range(n - 1):
                if depth[i] <= 0.1 or depth[i + 1] <= 0.1:
                    continue
                dm = (depth[i] + depth[i + 1]) / 2
                f = _fog_of(dm, near, far)
                col = _fog_col(beadcolor.bond_color(), f * 0.85)
                w = max(1.0, self.cam.focal * 0.07 / dm)
                ls.setThickness(w)
                ls.setColor(col[0] / 255, col[1] / 255, col[2] / 255, 1.0)
                ls.moveTo(pts[i][0], dm, -pts[i][1])
                ls.drawTo(pts[i + 1][0], dm, -pts[i + 1][1])
        for (i, j) in loops:
            if depth[i] <= 0.1 or depth[j] <= 0.1:
                continue
            dm = (depth[i] + depth[j]) / 2
            f = _fog_of(dm, near, far)
            col = _fog_col(_LOOP_COL if self.color_mode != "loop" else beadcolor.LOOP_DOMAIN_ON, f)
            w = max(1.0, self.cam.focal * 0.045 / dm)
            ls.setThickness(w)
            ls.setColor(col[0] / 255, col[1] / 255, col[2] / 255, 1.0)
            ls.moveTo(pts[i][0], dm - 0.01, -pts[i][1])
            ls.drawTo(pts[j][0], dm - 0.01, -pts[j][1])
        if self._line_np is not None:
            self._line_np.removeNode()
        self._line_np = NodePath(ls.create())
        self._line_np.reparentTo(self.root)

        # ---- chain/ribbon body: real filled triangles rebuilt every frame
        # (a GL line's width isn't reliable enough across drivers to read
        # as a solid tube, so both of these use actual mesh geometry).
        if self._mesh_np is not None:
            self._mesh_np.removeNode()
            self._mesh_np = None
        if self.rep_mode == "ribbon":
            self._mesh_np = self._build_ribbon(pos, pts, depth, colors, rect, near, far)
        elif self.rep_mode == "chain":
            self._mesh_np = self._build_chain(pts, depth, colors, near, far)

        # ---- beads: draw back-to-front via explicit bin order (depth sort
        # happens here on the CPU exactly like the old renderer -- it's a
        # sort of <= a few hundred floats, never the expensive part).
        # "ribbon" mode has no bead sprites at all -- the strip is the body.
        order = sorted(range(n), key=lambda i: -depth[i])
        for bin_i, i in enumerate(order):
            card = self._bead_cards[i]
            if depth[i] <= 0.1 or self.rep_mode == "ribbon":
                card.hide()
                continue
            card.show()
            d = depth[i]
            joint = self.rep_mode == "chain"
            rad = max(3.0, self.cam.focal * self.bead_r / d) * (0.68 if joint else 1.0)
            f = _fog_of(d, near, far)
            base = colors[i]
            if dim:
                base = theme.lerp_col(base, theme.INK_2, 0.5)
            tex = _bead_texture(int(rad * 2), base, f, shiny=joint)
            card.setTexture(tex, 1)
            card.setScale(rad * 2, 1, rad * 2)
            card.setPos(pts[i][0], d, -pts[i][1])
            card.setBin("fixed", bin_i)

        app.render_to(surf, rect)

        # ---- thin 2D overlay layer: hover/pending/mark rings + index labels
        # + rubber-band -- identical to the original, cheap pygame drawing.
        prev = surf.get_clip()
        surf.set_clip(rect)
        for i in range(n):
            if depth[i] <= 0.1:
                continue
            rad = max(3.0, self.cam.focal * self.bead_r / depth[i])
            f = _fog_of(depth[i], near, far)
            if i == self.pending:
                pulse = 0.5 + 0.5 * math.sin(t * 6.5)
                self._ring(surf, pts[i], rad + 3 + 2 * pulse, beadcolor.pending_color(), 2)
            elif i in self.mark:
                pulse = 0.5 + 0.5 * math.sin(t * 6.5)
                self._ring(surf, pts[i], rad + 4 + 2 * pulse, (240, 200, 110), 2)
                self._ring(surf, pts[i], rad + 1, (240, 200, 110), 1)
            elif i == self.hover:
                self._ring(surf, pts[i], rad + 3, beadcolor.hover_color(), 2)
            if show_index and n <= 80 and rad > 9:
                fnt = theme.font(max(9, int(rad * 0.75)), mono=True, bold=True)
                lab = fnt.render(str(i), True, (255, 255, 255))
                lab.set_alpha(int(220 * (1 - f)))
                sh = fnt.render(str(i), True, (0, 0, 0))
                sh.set_alpha(int(130 * (1 - f)))
                ox = pts[i][0] - lab.get_width() / 2
                oy = pts[i][1] - lab.get_height() / 2
                surf.blit(sh, (ox + 1, oy + 1))
                surf.blit(lab, (ox, oy))
        if self.pending is not None and depth[self.pending] > 0.1 and mouse is not None:
            mx, my = mouse
            if rect.collidepoint(mx, my):
                self._dashed(surf, pts[self.pending], (mx, my), beadcolor.pending_color(), t)
        if self.color_mode == "rainbow":
            beadcolor.draw_colorbar(surf, rect, n)
        if show_box:
            draw_axis_gizmo(surf, rect, self.cam)
        surf.set_clip(prev)

    # ------------------------------------------------------------ ribbon
    def _build_ribbon(self, pos, pts, depth, colors, rect, near, far) -> NodePath | None:
        """ChimeraX-style ribbon: a Catmull-Rom spline through the backbone
        plus a soft cross-section highlight (chainshape.smooth_ribbon_path /
        cross_section_shade -- shared with render3d.py's CPU ribbon, see
        there for the full explanation). Panda interpolates per-vertex
        colour across each triangle for free, so unlike the CPU renderer
        this only needs the profile's own lines (U_LEVELS) as geometry --
        the smooth gradient between them comes from the GPU rasteriser."""
        side = chainshape.ribbon_frame(pos)
        hw = self.bead_r * 0.9
        path_pos, path_side, path_tan, path_col = chainshape.smooth_ribbon_path(pos, side, colors)
        m = len(path_pos)
        levels = [self.screen_positions(path_pos + path_side * (hw * u), rect)
                  for u in chainshape.U_LEVELS]
        nlev = len(chainshape.U_LEVELS)

        vdata = GeomVertexData("ribbon", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        vwriter = GeomVertexWriter(vdata, "vertex")
        cwriter = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        vi = 0
        for i in range(m - 1):
            if (levels[0][1][i] <= 0.1 or levels[0][1][i + 1] <= 0.1 or
                    levels[-1][1][i] <= 0.1 or levels[-1][1][i + 1] <= 0.1):
                continue
            normal = np.cross(path_tan[i] + path_tan[i + 1], path_side[i] + path_side[i + 1])
            nl = np.linalg.norm(normal)
            bend = chainshape.lit_fraction(normal / nl) if nl > 1e-9 else 0.7
            dm = (levels[1][1][i] + levels[1][1][i + 1] +
                  levels[2][1][i] + levels[2][1][i + 1]) / 4
            f = _fog_of(dm, near, far)

            # one lit colour per (profile line, endpoint) -- Panda blends
            # the rest across each triangle itself.
            col_at = {}
            for lv in range(nlev):
                shade = bend * chainshape.cross_section_shade(chainshape.U_LEVELS[lv])
                ci = tuple(min(1.0, c * shade / 255.0) for c in _fog_col(tuple(path_col[i]), f))
                cj = tuple(min(1.0, c * shade / 255.0) for c in _fog_col(tuple(path_col[i + 1]), f))
                col_at[lv] = (ci, cj)

            for s in range(nlev - 1):
                (sx0, sy0), d0 = levels[s][0][i],     levels[s][1][i]
                (sx1, sy1), d1 = levels[s][0][i + 1], levels[s][1][i + 1]
                (sx2, sy2), d2 = levels[s + 1][0][i + 1], levels[s + 1][1][i + 1]
                (sx3, sy3), d3 = levels[s + 1][0][i],     levels[s + 1][1][i]
                corners = [((sx0, sy0), d0, col_at[s][0]),
                           ((sx1, sy1), d1, col_at[s][1]),
                           ((sx2, sy2), d2, col_at[s + 1][1]),
                           ((sx3, sy3), d3, col_at[s + 1][0])]
                for (sx, sy), d, c in corners:
                    vwriter.addData3(sx, d, -sy)
                    cwriter.addData4(c[0], c[1], c[2], 1.0)
                tris.addVertices(vi, vi + 1, vi + 2)
                tris.addVertices(vi, vi + 2, vi + 3)
                vi += 4

        if vi == 0:
            return None
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("ribbon")
        node.addGeom(geom)
        np_ribbon = NodePath(node)
        np_ribbon.setTwoSided(True)     # visible from both sides, no culling
        np_ribbon.reparentTo(self.root)
        return np_ribbon

    # ------------------------------------------------------------- chain
    def _build_chain(self, pts, depth, colors, near, far) -> NodePath | None:
        """A thick, shiny tube along the backbone: per-bond quads (gradient
        colour, two halves like the software renderer) plus a brighter
        highlight quad offset to one side -- real triangles rather than a
        GL line, so the thickness actually shows up on every driver."""
        n = len(pts)
        vdata = GeomVertexData("chain", GeomVertexFormat.getV3c4(), Geom.UHDynamic)
        vwriter = GeomVertexWriter(vdata, "vertex")
        cwriter = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHDynamic)

        def quad(a, b, half_w, dm, colf):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L < 0.5:
                return None
            px, py = -dy / L * half_w, dx / L * half_w
            pts4 = [(a[0]+px, a[1]+py), (b[0]+px, b[1]+py),
                    (b[0]-px, b[1]-py), (a[0]-px, a[1]-py)]
            return pts4, (px / half_w, py / half_w) if half_w else (0.0, 0.0)

        vi = 0
        for i in range(n - 1):
            if depth[i] <= 0.1 or depth[i + 1] <= 0.1:
                continue
            ri = max(3.0, self.cam.focal * self.bead_r / depth[i])
            rj = max(3.0, self.cam.focal * self.bead_r / depth[i + 1])
            mid_col = theme.lerp_col(colors[i], colors[i + 1], 0.5)
            for k, (c0, c1) in enumerate(((colors[i], mid_col), (mid_col, colors[i + 1]))):
                t0, t1 = k / 2, (k + 1) / 2
                a  = (pts[i][0]*(1-t0)+pts[i+1][0]*t0, pts[i][1]*(1-t0)+pts[i+1][1]*t0)
                b  = (pts[i][0]*(1-t1)+pts[i+1][0]*t1, pts[i][1]*(1-t1)+pts[i+1][1]*t1)
                dm = depth[i] * (1 - (t0+t1)/2) + depth[i+1] * ((t0+t1)/2)
                w  = (ri * (1 - (t0+t1)/2) + rj * ((t0+t1)/2)) * 0.85
                f  = _fog_of(dm, near, far)
                col = _fog_col(theme.lerp_col(c0, c1, 0.5), f)
                colf = (col[0] / 255, col[1] / 255, col[2] / 255, 1.0)

                res = quad(a, b, w / 2, dm, colf)
                if res is None:
                    continue
                corners, (px_u, py_u) = res
                for (sx, sy) in corners:
                    vwriter.addData3(sx, dm, -sy)
                    cwriter.addData4(*colf)
                tris.addVertices(vi, vi + 1, vi + 2)
                tris.addVertices(vi, vi + 2, vi + 3)
                vi += 4

                # shiny highlight streak, offset to a fixed, consistent
                # "lit" side so it stays put regardless of segment heading.
                if px_u * -0.4 + py_u * -0.9 < 0:
                    px_u, py_u = -px_u, -py_u
                off = w * 0.22
                hl  = theme.lerp_col(col, (255, 255, 255), 0.55)
                hlf = (hl[0] / 255, hl[1] / 255, hl[2] / 255, 1.0)
                ha = (a[0] + px_u * off, a[1] + py_u * off)
                hb = (b[0] + px_u * off, b[1] + py_u * off)
                hres = quad(ha, hb, max(1.0, w * 0.22) / 2, dm, hlf)
                if hres is None:
                    continue
                hcorners, _ = hres
                for (sx, sy) in hcorners:
                    vwriter.addData3(sx, dm - 0.01, -sy)
                    cwriter.addData4(*hlf)
                tris.addVertices(vi, vi + 1, vi + 2)
                tris.addVertices(vi, vi + 2, vi + 3)
                vi += 4

        if vi == 0:
            return None
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("chain")
        node.addGeom(geom)
        np_chain = NodePath(node)
        np_chain.setTwoSided(True)
        np_chain.reparentTo(self.root)
        return np_chain

    # --------------------------------------------------------- primitives
    @staticmethod
    def _ring(surf, p, r, col, w):
        r = int(max(2, r))
        s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        theme.circle(s, (*col, 210), (r + 2, r + 2), r, w)
        surf.blit(s, (p[0] - r - 2, p[1] - r - 2))

    @staticmethod
    def _dashed(surf, a, b, col, t):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 2:
            return
        ux, uy = dx / L, dy / L
        step = 8
        off = (t * 38) % (step * 2)
        d = off
        while d < L:
            e = min(d + step, L)
            pygame.draw.line(surf, col,
                              (a[0] + ux * d, a[1] + uy * d),
                              (a[0] + ux * e, a[1] + uy * e), 2)
            d += step * 2
