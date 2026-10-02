"""Figure style shared by every panel in the manuscript.

Sizes follow the arXiv preprint page (US letter, 1 in margins: text width 6.5 in). Type
follows Nature-style practice: a sans-serif face, 7 pt labels, 6 pt ticks and legend,
8 pt bold lowercase panel letters, TrueType embedding so text stays editable.
Colours are the Okabe-Ito colour-blind-safe set, fixed per model family across all
figures; strengths of one family share its hue and vary in lightness.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

SINGLE = 3.4
DOUBLE = 6.5
# Panel coordinates in figures.py were designed on a 7.16 in canvas; they are scaled
# horizontally to the current width so type sizes stay fixed while panels narrow.
DESIGN_W = 7.16
SX = DOUBLE / DESIGN_W
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

FAMILY_COLOUR = {
    "PixelSeal": "#0072B2",
    "VideoSeal-1.0": "#56B4E9",
    "TrustMark": "#D55E00",
    "WAM": "#009E73",
    "ChunkySeal": "#CC79A7",
    "InvisMark": "#E69F00",
    "invisible-watermark": "#7F7F7F",
    "RivaGAN": "#3B3B3B",
    "AudioSeal": "#0072B2",
    "WavMark": "#009E73",
    "SilentCipher": "#E69F00",
    "Perth": "#CC79A7",
}
TRUSTMARK_MARKER = {"B": "s", "C": "v", "P": "^", "Q": "o"}
NEUTRAL = "#4D4D4D"
GRID = "#E6E6E6"
GATE_LINE = "#9A9A9A"


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "axes.linewidth": 0.6,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": "black",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "xtick.color": NEUTRAL,
        "ytick.color": NEUTRAL,
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "lines.linewidth": 0.9,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.02,
        "figure.constrained_layout.use": False,
    })


def shade(hex_colour: str, t: float) -> str:
    """Lightness ramp within one hue: t=0 lightest, t=1 darkest (for strength sweeps)."""
    r, g, b = mcolors.to_rgb(hex_colour)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l2 = 0.78 - 0.5 * t if l > 0.2 else 0.25 + 0.35 * (1 - t)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, max(0.15, min(0.85, l2)), s))


def panel_label(ax, letter: str, dx: float = -0.02, dy: float = 0.02) -> None:
    """Bold lowercase panel letter placed outside the plotting area, top left."""
    fig = ax.figure
    bb = ax.get_position()
    fig.text(bb.x0 + dx - 0.035 * (SINGLE / fig.get_figwidth()), bb.y1 + dy, letter, fontsize=8,
             fontweight="bold", va="bottom", ha="left")


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
    plt.close(fig)
    return pdf
