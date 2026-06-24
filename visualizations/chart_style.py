"""Shared chart styling: GLM-inspired palette, rounded bars, clean layout."""

import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path
from matplotlib.legend_handler import HandlerPatch

# ── GLM-inspired palette (existing — kept for backward compat) ──
PALETTE = {
    'blue':   '#5B8FF9',
    'green':  '#61DDAA',
    'yellow': '#F6BD16',
    'red':    '#E8684A',
    'purple': '#7262FD',
    'orange': '#F6903D',
    'cyan':   '#78D3F8',
    'grey':   '#A0A0A0',
    'text':   '#2C2C2C',
    'annot':  '#555555',
}

# ── GLM-vivid palette — higher-saturation hues for the 3-engine
# QPS-vs-recall comparison plots so each curve is unmistakable on
# log-scale axes. Tuned for white background + dense overlap zones.
# References the brighter accent style used in the official GLM
# documentation (zhipuai.cn / open.bigmodel.cn).
PALETTE_VIVID = {
    'orion':    '#0EA5E9',  # sky-bright cyan-blue — "ours" line
    'parlay':   '#F43F5E',  # rose / bright magenta — high contrast accent
    'diskann':  '#F59E0B',  # amber — distinct hue, separated from blue + rose
    'orion_d':  '#0284C7',  # 1-shade-darker variant for annotation arrow
    'text':     '#1F2937',  # near-black for crisp labels on bright lines
    'grid':     '#E5E7EB',  # very light grey for grid + spines
}

DATASET_COLORS = {
    'sift':    PALETTE['blue'],
    'glove25': PALETTE['green'],
    'glove100':PALETTE['purple'],
    'gist':    PALETTE['red'],
}

def style_ax(ax):
    """Apply clean GLM-style axis styling."""
    ax.set_facecolor('white')
    ax.tick_params(colors=PALETTE['text'], labelsize=9)
    ax.xaxis.label.set_color(PALETTE['text'])
    ax.yaxis.label.set_color(PALETTE['text'])
    ax.title.set_color(PALETTE['text'])
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#E0E0E0')
    ax.grid(True, alpha=0.2, color='#E0E0E0', axis='y')

def style_fig(fig):
    """Apply white background to figure."""
    fig.patch.set_facecolor('white')

class RoundedTopBar(PathPatch):
    """Bar with rounded top corners. Corner radius is a fraction of the bar's
    display width so rounding looks visually uniform across charts regardless
    of axis aspect ratio. Recomputes its path at draw time."""

    def __init__(self, ax, x, y, width, *, facecolor, alpha=0.9, radius_frac=0.3, zorder=3):
        self._bar_ax = ax
        self._bar_x = x
        self._bar_y = y
        self._bar_w = width
        self._radius_frac = radius_frac
        left, right = x - width / 2, x + width / 2
        placeholder = Path(
            [(left, 0), (right, 0), (right, y), (left, y), (left, 0)],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        )
        super().__init__(
            placeholder, facecolor=facecolor, edgecolor='none',
            alpha=alpha, zorder=zorder, linewidth=0,
        )

    def draw(self, renderer):
        self._recompute_path()
        super().draw(renderer)

    def _recompute_path(self):
        ax = self._bar_ax
        x, y, w = self._bar_x, self._bar_y, self._bar_w
        if y == 0:
            return
        trans = ax.transData
        inv = trans.inverted()

        p_left = trans.transform((x - w / 2, 0))
        p_right = trans.transform((x + w / 2, 0))
        bar_width_px = abs(p_right[0] - p_left[0])
        radius_px = min(bar_width_px * self._radius_frac, 14.0)

        p0 = trans.transform((x, 0))
        px_data = inv.transform((p0[0] + radius_px, p0[1]))
        py_data = inv.transform((p0[0], p0[1] + radius_px))
        rx = min(abs(px_data[0] - x), w / 2)
        ry = min(abs(py_data[1] - 0), abs(y) / 2)

        left, right = x - w / 2, x + w / 2
        top, bottom = y, 0

        verts = [
            (left, bottom),
            (right, bottom),
            (right, top - ry),
            (right, top),
            (right - rx, top),
            (left + rx, top),
            (left, top),
            (left, top - ry),
            (left, bottom),
        ]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.LINETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CLOSEPOLY,
        ]
        self.set_path(Path(verts, codes))


def rounded_bar(ax, x, y, width, color, alpha=0.9, radius_frac=0.3):
    """Draw a bar with rounded top corners. radius_frac is fraction of bar width."""
    if y == 0:
        return
    bar = RoundedTopBar(ax, x, y, width,
                        facecolor=color, alpha=alpha, radius_frac=radius_frac)
    ax.add_patch(bar)

class RoundedPatch(mpatches.Patch):
    """Legend-only marker indicating the patch should be drawn with rounded corners."""
    pass

class _HandlerRoundedPatch(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        r = min(height * 0.35, width * 0.15, 4)
        p = FancyBboxPatch(
            (-xdescent, -ydescent), width, height,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=orig_handle.get_facecolor(),
            edgecolor='none',
            alpha=orig_handle.get_alpha(),
            transform=trans,
        )
        return [p]

_HANDLER_MAP = {RoundedPatch: _HandlerRoundedPatch()}

def rounded_patch(color, alpha=0.9, label=None):
    """Create a legend handle that renders with rounded corners."""
    return RoundedPatch(facecolor=color, edgecolor='none', alpha=alpha, label=label)

def rounded_bars(ax, positions, values, width, color, label=None, alpha=0.9):
    """Draw a group of rounded bars and return a legend handle."""
    for x, y in zip(positions, values):
        rounded_bar(ax, x, y, width, color, alpha)
    if label:
        return rounded_patch(color, alpha=alpha, label=label)
    return None

def apply_ylim(ax, values, headroom=1.25, bottom=0):
    """Set y-axis limits with headroom. Patches don't autoscale, so this is required."""
    vals = [v for v in values if v is not None]
    if not vals:
        return
    top = max(vals)
    if top <= 0:
        top = 1
    ax.set_ylim(bottom, top * headroom)

def make_legend(ax, handles, **kwargs):
    """Standard legend styling. Uses rounded-corner handler for RoundedPatch."""
    defaults = dict(fontsize=9, facecolor='white', edgecolor='#E0E0E0',
                    labelcolor=PALETTE['text'], handler_map=_HANDLER_MAP)
    defaults.update(kwargs)
    ax.legend(handles=handles, **defaults)


def _scale_fig_text(fig, scale):
    """Multiply every text element's fontsize by `scale`. Idempotent across
    calls because we just read-back and multiply — no state stashed."""
    for ax in fig.axes:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + list(ax.get_xticklabels())
            + list(ax.get_yticklabels())
            + list(ax.texts)
        ):
            item.set_fontsize(item.get_fontsize() * scale)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(t.get_fontsize() * scale)
            ttl = leg.get_title()
            if ttl is not None and ttl.get_text():
                ttl.set_fontsize(ttl.get_fontsize() * scale)
    # Figure-level suptitle if any.
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(fig._suptitle.get_fontsize() * scale)


def save_png_and_pdf(fig, png_path, pdf_font_scale=1.5, dpi=150):
    """Save a PNG with unchanged font sizes and a sibling PDF with all text
    uniformly scaled up (for embedding in a paper at half-page width).

    Emits both `{name}.png` and `{name}.pdf` side by side.
    """
    assert png_path.endswith('.png'), f"png_path must end with .png, got {png_path}"
    pdf_path = png_path[:-4] + '.pdf'
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    _scale_fig_text(fig, pdf_font_scale)
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    _scale_fig_text(fig, 1.0 / pdf_font_scale)
    return png_path, pdf_path
