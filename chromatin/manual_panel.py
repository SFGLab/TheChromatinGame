import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0
matplotlib.rcParams["mathtext.fontset"] = "cm"        # Computer Modern -- proper math font
matplotlib.rcParams["mathtext.rm"]      = "serif"
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame
from . import theme
from .manual_content import SECTIONS

try:
    from PIL import Image as _PIL_Image
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False

_eq_cache: dict[tuple, pygame.Surface] = {}


def render_latex(expr: str, fontsize: int = 18,
                 color: tuple = (220, 230, 255),
                 dpi: int = 220) -> pygame.Surface:
    key = (expr, fontsize, color, dpi)
    if key in _eq_cache:
        return _eq_cache[key]

    col = tuple(c / 255.0 for c in color)

    try:
        fig = Figure(figsize=(10, 1.6), facecolor=(0, 0, 0, 0))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax.text(0.5, 0.5, f"${expr}$",
                fontsize=fontsize, color=col,
                va="center", ha="center",
                transform=ax.transAxes)
        canvas.draw()

        rgba  = np.array(canvas.buffer_rgba())
        alpha = rgba[:, :, 3]
        rows  = np.any(alpha > 8, axis=1)
        cols  = np.any(alpha > 8, axis=0)

        if not rows.any() or not cols.any():
            raise ValueError("no visible pixels")

        r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
        c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
        pad    = max(8, int(fontsize * 0.35))
        r0 = max(0, r0 - pad);    r1 = min(rgba.shape[0] - 1, r1 + pad)
        c0 = max(0, c0 - pad);    c1 = min(rgba.shape[1] - 1, c1 + pad)
        cropped = rgba[r0:r1 + 1, c0:c1 + 1]   # (H, W, 4)
        h_px, w_px = cropped.shape[:2]

        if _HAVE_PIL:
            img = _PIL_Image.fromarray(cropped, "RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            surf = pygame.image.load(buf, "png").convert_alpha()
        else:
            # Build surface directly from numpy without PIL
            surf = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
            rgb  = np.ascontiguousarray(cropped[:, :, :3].transpose(1, 0, 2))
            alp  = np.ascontiguousarray(cropped[:, :, 3].T)
            pygame.surfarray.blit_array(surf, rgb)
            pygame.surfarray.pixels_alpha(surf)[:] = alp

        _eq_cache[key] = surf
        return surf

    except Exception as e:
        # Render a clean styled fallback -- NOT raw LaTeX source.
        # Use a unicode approximation of the expression so it's at least
        # readable, drawn with the game's own mono font.
        import traceback
        traceback.print_exc()
        # Make a small surface with a styled placeholder
        f    = theme.font(max(11, fontsize - 6), mono=True)
        # Strip LaTeX commands to get something human-readable
        import re
        readable = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', expr)
        readable = re.sub(r'\\[a-zA-Z]+', '', readable)
        readable = readable.replace('{','').replace('}','').strip()
        readable = readable[:80] + ("…" if len(readable) > 80 else "")
        img  = f.render(readable, True, color)
        surf = pygame.Surface((img.get_width() + 20,
                               img.get_height() + 12), pygame.SRCALPHA)
        surf.blit(img, (10, 6))
        _eq_cache[key] = surf
        return surf
    
# Colours specific to the manual -- functions, not constants, so they track
# theme.set_theme() at draw time instead of freezing at import time.
def _NAV_ACTIVE_COL(): return theme.GREEN           # loop-convention green, fixed
def _EQ_BG():           return theme.PANEL_HI        # chip behind equations
def _EQ_COL():          return theme.CYAN            # equation text accent
def _EQ_LABEL_COL():   return theme.TEXT_FAINT
def _HEADING_COL():    return theme.TEXT
def _SUBH_COL():        return theme.GREEN
def _BULLET_DOT():      return theme.GREEN


def _wrap(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    """Word-wrap `text` to fit within `max_w` pixels."""
    words = text.split()
    lines, line = [], []
    for w in words:
        test = " ".join(line + [w])
        if font.size(test)[0] <= max_w:
            line.append(w)
        else:
            if line:
                lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return lines or [""]


class ManualPanel:
    """Two-column manual overlay.

    Left column  : section navigation (narrow, fixed)
    Right column : scrollable rich text content

    Call draw(sc, W, H) each frame while visible.
    Call handle_event(ev) for mouse clicks and wheel scrolling.
    """

    NAV_W       = 188       # width of the left navigation column
    PAD         = 20        # outer padding
    INNER_PAD   = 16        # padding inside the content column
    SCROLL_STEP = 32        # pixels per mouse-wheel tick

    def __init__(self):
        self._section  = 0          # currently selected section index
        self._scroll   = 0          # scroll offset of the content column (px)
        self._content_h= 0          # total height of rendered content
        self._nav_rects: list[pygame.Rect] = []   # hit-rects for nav items
        self._close_rect: pygame.Rect | None = None

    def select(self, idx: int) -> None:
        self._section = max(0, min(idx, len(SECTIONS) - 1))
        self._scroll  = 0           # reset scroll when switching sections

    def handle_event(self, ev, W: int = 0, H: int = 0) -> str:
        """Returns 'close', 'consumed'.

        W and H let us compute the × hit-rect independently of whether
        draw() has already run this frame.
        """
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            close_r = (self._close_rect
                       if self._close_rect is not None
                       else (self._get_close_rect(W, H) if W and H else None))
            if close_r and close_r.collidepoint(ev.pos):
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

    # ---------------------------------------------------------------- draw
    def draw(self, sc: pygame.Surface, W: int, H: int) -> bool:
        """Draw the panel. Returns False if the close button was clicked."""
        # Outer rect: 92% of screen, centred
        mw = int(W * 0.92)
        mh = int(H * 0.90)
        mx = (W - mw) // 2
        my = (H - mh) // 2

        # Veil behind the panel
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((*theme.INK, 210))
        sc.blit(veil, (0, 0))

        # Panel background
        panel_surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (*theme.INK, 245),
                         panel_surf.get_rect(), border_radius=14)
        pygame.draw.rect(panel_surf, (*theme.RULE, 180),
                         panel_surf.get_rect(), 1, border_radius=14)
        sc.blit(panel_surf, (mx, my))

        # Title bar
        title_f = theme.font(16, bold=True)
        sc.blit(title_f.render("THE CHROMATIN GAME — Manual", True, _HEADING_COL()),
                (mx + self.PAD, my + self.PAD))
        sub_f = theme.font(10, mono=True)
        sc.blit(sub_f.render("click a section · scroll to read", True, theme.TEXT_FAINT),
                (mx + self.PAD, my + self.PAD + 22))

        # Close button (×) — top right
        close_f   = theme.font(16, bold=True)
        close_img = close_f.render("×", True, theme.TEXT_FAINT)
        self._close_rect = pygame.Rect(
            mx + mw - close_img.get_width() - self.PAD,
            my + self.PAD - 2,
            close_img.get_width() + 8,
            close_img.get_height() + 4)
        if self._close_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(sc, theme.PANEL_HI, self._close_rect, border_radius=4)
            close_img = close_f.render("×", True, theme.TEXT)   # brighter on hover
        sc.blit(close_img, (self._close_rect.x + 4, self._close_rect.y + 2))

        # Divider below title bar
        bar_y = my + self.PAD + 44
        pygame.draw.line(sc, theme.RULE,
                         (mx + self.PAD, bar_y),
                         (mx + mw - self.PAD, bar_y), 1)

        body_y  = bar_y + 10
        body_h  = mh - (bar_y - my) - self.PAD - 10
        nav_x   = mx + self.PAD
        cont_x  = nav_x + self.NAV_W + 12
        cont_w  = mw - self.NAV_W - self.PAD * 2 - 12

        # Vertical divider between nav and content
        pygame.draw.line(sc, theme.RULE,
                         (cont_x - 6, body_y),
                         (cont_x - 6, body_y + body_h), 1)

        self._draw_nav(sc, nav_x, body_y, body_h)
        self._draw_content(sc, cont_x, body_y, cont_w, body_h)
        return True

    def _draw_rich_line(self, sc, text: str, x: int, y: int,
                        body_f, w: int) -> int:
        """Draw a line of text that may contain inline $math$ segments.
        
        Returns the height used.
        """
        import re
        parts = re.split(r'(\$[^$]+\$)', text)
        
        # First pass: measure everything so we can word-wrap + baseline-align
        line_h = body_f.get_height()
        segments = []
        for part in parts:
            if part.startswith('$') and part.endswith('$') and len(part) > 2:
                expr = part[1:-1]
                try:
                    surf = render_latex(expr, fontsize=13, color=_EQ_COL(), dpi=180)
                    segments.append(('math', surf))
                    line_h = max(line_h, surf.get_height())
                except Exception:
                    segments.append(('text', part))
            else:
                segments.append(('text', part))
        
        # Second pass: lay out left to right, wrapping at w
        cx = x
        cy = y
        
        for kind, content in segments:
            if kind == 'math':
                sw = content.get_width()
                sh = content.get_height()
                if cx + sw > x + w and cx > x:
                    cy += line_h + 4
                    cx = x
                # Vertically centre the math surface on the text baseline
                offset_y = (line_h - sh) // 2
                sc.blit(content, (cx, cy + offset_y))
                cx += sw + 4
            else:
                words = content.split(' ')
                for word in words:
                    if not word:
                        continue
                    img = body_f.render(word, True, theme.TEXT_DIM)
                    if cx + img.get_width() > x + w and cx > x:
                        cy += line_h + 4
                        cx = x
                    sc.blit(img, (cx, cy))
                    cx += img.get_width() + body_f.size(' ')[0]
        
        return (cy - y) + line_h + 4

    def _get_close_rect(self, W: int, H: int) -> pygame.Rect:
        """Compute the × rect from screen dimensions -- must match draw() exactly."""
        mw        = int(W * 0.92)
        mh        = int(H * 0.90)
        mx        = (W - mw) // 2
        my        = (H - mh) // 2
        close_f   = theme.font(16, bold=True)
        close_img = close_f.render("×", True, theme.TEXT_FAINT)
        return pygame.Rect(
            mx + mw - close_img.get_width() - self.PAD,
            my + self.PAD - 2,
            close_img.get_width() + 8,
            close_img.get_height() + 4)

    def _draw_nav(self, sc, x, y, h):
        """Left column: section list with coloured icon circles."""
        self._nav_rects = []
        nav_f  = theme.font(12)
        icon_f = theme.font(11, bold=True)
        row_h  = 40
        pad    = 8

        # One accent colour per section — cycles through the game palette
        ICON_COLS = [
            (88, 210, 140),    # Inspiration  -- green
            (120, 150, 210),   # Polymer      -- periwinkle
            (240, 186, 92),    # Force Field  -- amber
            (92, 200, 255),    # Validation   -- cyan
            (210, 130, 140),   # Structural   -- rose
        ]

        for i, sec in enumerate(SECTIONS):
            active   = (i == self._section)
            col      = ICON_COLS[i % len(ICON_COLS)]
            dim_col  = tuple(int(c * 0.55) for c in col)
            icon_col = col if active else dim_col
            r = pygame.Rect(x - 6, y + i * row_h, self.NAV_W, row_h - 4)
            self._nav_rects.append(r)

            if active:
                # Tinted background for the active row
                bg_col = tuple(int(c * 0.18) for c in col)
                pygame.draw.rect(sc, bg_col, r, border_radius=6)
                pygame.draw.rect(sc, icon_col, r, 1, border_radius=6)

            # Small filled circle with the icon letter inside
            cx = x + pad + 10
            cy_icon = y + i * row_h + (row_h - 4) // 2
            theme.circle(sc, icon_col, (cx, cy_icon), 10)
            letter = sec["icon"]
            limg   = icon_f.render(letter, True, theme.INK)
            sc.blit(limg, (cx - limg.get_width() // 2,
                           cy_icon - limg.get_height() // 2))

            # Section title
            title_col = col if active else tuple(int(c * 0.65) for c in col)
            timg = nav_f.render(sec["title"], True, title_col)
            sc.blit(timg, (x + pad + 24, y + i * row_h + 12))

    def _draw_content(self, sc, x, y, w, h):
        """Right column: scrollable content for the selected section."""
        # Clip to the content column
        clip_rect = pygame.Rect(x, y, w, h)
        old_clip  = sc.get_clip()
        sc.set_clip(clip_rect)

        section = SECTIONS[self._section]
        blocks  = section["blocks"]

        # Fonts
        heading_f = theme.font(17, bold=True)
        subh_f    = theme.font(13, bold=True)
        body_f    = theme.font(12)
        eq_f      = theme.font(12, mono=True)
        label_f   = theme.font(10, mono=True)
        bullet_f  = theme.font(12)

        IP = self.INNER_PAD
        cy = y + IP - self._scroll   # current y, offset by scroll
        line_gap = 6

        for block in blocks:
            bt = block["type"]

            if bt == "heading":
                img = heading_f.render(block["text"], True, _HEADING_COL())
                sc.blit(img, (x + IP, cy))
                cy += img.get_height() + line_gap + 4
                # Underline
                pygame.draw.line(sc, _NAV_ACTIVE_COL(),
                                 (x + IP, cy - 3),
                                 (x + IP + img.get_width(), cy - 3), 1)
                cy += 6

            elif bt == "subheading":
                img = subh_f.render(block["text"], True, _SUBH_COL())
                sc.blit(img, (x + IP, cy + 6))
                cy += img.get_height() + line_gap + 8

            elif bt == "body":
                # Check if it contains any inline math
                if '$' in block["text"]:
                    used_h = self._draw_rich_line(
                        sc, block["text"], x + IP, cy, body_f, w - IP * 2)
                    cy += used_h + line_gap
                else:
                    # Plain text -- original fast path
                    lines = _wrap(block["text"], body_f, w - IP * 2)
                    for ln in lines:
                        if y <= cy <= y + h:
                            sc.blit(body_f.render(ln, True, theme.TEXT_DIM),
                                    (x + IP, cy))
                        cy += body_f.get_height() + 2
                    cy += line_gap

            elif bt == "equation":
                latex   = block.get("latex", block.get("text", ""))
                lbl_txt = block.get("label", "")
                try:
                    eq_surf = render_latex(latex, color=_EQ_COL())
                    eq_w, eq_h_px = eq_surf.get_size()

                    # Box is exactly the equation surface size + small padding.
                    # Centred horizontally in the content column.
                    pad_x, pad_y = 16, 8
                    box_w = eq_w + pad_x * 2
                    box_h = eq_h_px + pad_y * 2
                    # Centre within available width: from x+IP to x+IP+(w-IP*2)
                    avail_x = x + IP
                    avail_w = w - IP * 2
                    box_x   = avail_x + max(0, (avail_w - box_w) // 2)

                    box_r = pygame.Rect(box_x, cy + 6, box_w, box_h)

                    lbl_h = 0
                    lbl_img = None
                    if lbl_txt:
                        lbl_img = label_f.render(f"[{lbl_txt}]", True,
                                                 _EQ_LABEL_COL())
                        lbl_h = lbl_img.get_height() + 4

                    if y <= cy <= y + h:
                        pygame.draw.rect(sc, _EQ_BG(), box_r, border_radius=6)
                        pygame.draw.rect(sc, theme.RULE_SOFT, box_r, 1,
                                         border_radius=6)
                        sc.blit(eq_surf, (box_r.x + pad_x, box_r.y + pad_y))
                        if lbl_img:
                            sc.blit(lbl_img,
                                    (box_r.right - lbl_img.get_width(),
                                     box_r.bottom + 2))

                    cy += box_h + lbl_h + 14

                except Exception as e:
                    err_text = f"[equation error: {str(e)[:60]}]"
                    lines = _wrap(err_text, eq_f, w - IP * 3)
                    eq_h_fb = len(lines) * (eq_f.get_height() + 2) + 16
                    eq_rect = pygame.Rect(x + IP, cy + 4, w - IP * 2, eq_h_fb)
                    if y <= cy <= y + h:
                        pygame.draw.rect(sc, _EQ_BG(), eq_rect, border_radius=6)
                        ey = cy + 10
                        for ln in lines:
                            sc.blit(eq_f.render(ln, True, theme.POOR),
                                    (x + IP + 10, ey))
                            ey += eq_f.get_height() + 2
                    cy += eq_h_fb + 10

                except Exception as e:
                    err_text = f"[equation error: {str(e)[:60]}]"
                    lines = _wrap(err_text, eq_f, w - IP * 3)
                    eq_h_fb = len(lines) * (eq_f.get_height() + 2) + 16
                    eq_rect = pygame.Rect(x + IP, cy + 4, w - IP * 2, eq_h_fb)
                    if y <= cy <= y + h:
                        pygame.draw.rect(sc, _EQ_BG(), eq_rect, border_radius=6)
                        ey = cy + 10
                        for ln in lines:
                            sc.blit(eq_f.render(ln, True, theme.POOR),
                                    (x + IP + 10, ey))
                            ey += eq_f.get_height() + 2
                    cy += eq_h_fb + 10

            elif bt == "bullet":
                for item in block["items"]:
                    lines = _wrap(item, bullet_f, w - IP * 2 - 16)
                    for li, ln in enumerate(lines):
                        if y <= cy <= y + h:
                            if li == 0:
                                theme.circle(sc, _BULLET_DOT(),
                                            (x + IP + 6, cy + 8), 3)
                            sc.blit(bullet_f.render(ln, True, theme.TEXT_DIM),
                                    (x + IP + 16, cy))
                        cy += bullet_f.get_height() + 2
                    cy += 3
                cy += line_gap

            elif bt == "spacer":
                cy += 18

        # Total content height (used for scroll clamping)
        self._content_h = max(0, cy - (y + IP - self._scroll) - h  + 15)

        # Subtle scroll indicator on the right edge
        if self._content_h > 0:
            total = self._content_h + h
            thumb_h  = max(30, int(h * h / total))
            thumb_y  = y + int(self._scroll / self._content_h * (h - thumb_h))
            pygame.draw.rect(sc, theme.RULE,
                             pygame.Rect(x + w - 4, thumb_y, 3, thumb_h),
                             border_radius=2)

        sc.set_clip(old_clip)