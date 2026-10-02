"""Render every data figure of the manuscript from analysis.load().

Layout is explicit: every axes rectangle is placed in inches, so comparable panels
share edges exactly and nothing depends on tight_layout heuristics. No numbers or
annotations are drawn inside a plotting area; values live in the tables and legends
sit outside the axes.

    uv run --with matplotlib --with numpy --with scipy --with pillow python figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import analysis as A
import style as S

S.apply()

IMAGE_ATTACKS = [
    ("none", "None"), ("jpeg90", "JPEG 90"), ("jpeg75", "JPEG 75"), ("jpeg60", "JPEG 60"), ("jpeg50", "JPEG 50"),
    ("webp80", "WebP 80"), ("webp60", "WebP 60"), ("resize0.75", "Resize 0.75"), ("resize0.5", "Resize 0.5"),
    ("crop90", "Crop 90%"), ("crop70", "Crop 70%"), ("crop50", "Crop 50%"), ("blur1", "Blur σ=1"),
    ("bright0.8", "Brightness 0.8"), ("bright1.2", "Brightness 1.2"),
]
AUDIO_ATTACKS = [
    ("none", "None"), ("mp3_320", "MP3 320k"), ("mp3_128", "MP3 128k"), ("mp3_64", "MP3 64k"), ("aac_256", "AAC 256k"),
    ("aac_128", "AAC 128k"), ("aac_64", "AAC 64k"), ("resample_22k", "Resample 22k"), ("resample_8k", "Resample 8k"),
    ("gain_-6db", "Gain −6 dB"), ("noise_30db", "Noise 30 dB"),
]
VIDEO_ATTACKS = [
    ("h264_crf18", "H.264 CRF18"), ("h264_crf22", "H.264 CRF22"), ("h264_crf23", "H.264 CRF23"),
    ("h264_crf28", "H.264 CRF28"), ("h265_crf23", "H.265 CRF23"), ("h265_crf28", "H.265 CRF28"),
    ("resize0.5_h264_crf23", "Resize 0.5 + H.264"), ("crop75_h264_crf23", "Crop 75% + H.264"),
]

BITACC_CMAP = LinearSegmentedColormap.from_list(
    "bitacc", ["#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"])


def plain_log(axis, ticks) -> None:
    """Log axis with a few plain-number ticks (no 2x10^0 style labels)."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def ax_at(fig, x: float, y: float, w: float, h: float, square: bool = False):
    """Axes at (x, y, w, h) in design inches from the bottom-left of the figure.

    Horizontal coordinates are scaled from the 7.16 in design canvas to the page width;
    `square` scales vertical ones too (image plates keep their aspect ratio).
    """
    W, H = fig.get_size_inches()
    sy = S.SX if square else 1.0
    return fig.add_axes((x * S.SX / W, y * sy / H, w * S.SX / W, h * sy / H))


def letter(fig, ax, s: str, dx: float = 0.34, dy: float = 0.06) -> None:
    W, H = fig.get_size_inches()
    bb = ax.get_position()
    fig.text(bb.x0 - dx / W, bb.y1 + dy / H, s, fontsize=8, fontweight="bold", ha="left", va="bottom")


def family_style(c: A.Config, fams: list[A.Config]) -> dict:
    """Colour by family, lightness by strength rank within the family, marker by TrustMark variant."""
    same = sorted({x.strength for x in fams if x.family == c.family and x.strength is not None})
    t = 0.6 if len(same) <= 1 or c.strength is None else same.index(c.strength) / (len(same) - 1)
    base = S.FAMILY_COLOUR[c.family]
    marker = "o"
    if c.family == "TrustMark":
        marker = S.TRUSTMARK_MARKER[c.network[-1]]
    elif c.family == "AudioSeal" and "base" in c.network:
        marker = "s"
    colour = S.shade(base, 0.25 + 0.6 * t) if len(same) > 1 else base
    return {"color": colour, "marker": marker}


def family_legend(cfgs: list[A.Config], extra: list = (), ncol: int = 6):
    handles, seen = [], set()
    for c in cfgs:
        key = c.network if c.family in ("TrustMark", "AudioSeal") else c.family
        if key in seen:
            continue
        seen.add(key)
        st = family_style(c, cfgs)
        label = key if c.family in ("TrustMark", "AudioSeal") else A.FAMILY_LABEL.get(c.family, c.family)
        if not c.open_licence:
            label += " (ref.)"
        handles.append(Line2D([], [], color=S.FAMILY_COLOUR[c.family], marker=st["marker"], lw=0.9, ms=3.5,
                              mfc=S.FAMILY_COLOUR[c.family] if c.open_licence else "white", label=label))
    return handles + list(extra)


def gate_handle(label: str = "Gate, 0.95 bit accuracy"):
    return Line2D([], [], color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), label=label)


def scatter_families(ax, cfgs, x, y, connect: bool = True) -> None:
    by_net: dict[str, list[A.Config]] = {}
    for c in cfgs:
        by_net.setdefault(c.network, []).append(c)
    for net, members in by_net.items():
        members = sorted(members, key=lambda c: (c.strength or 0))
        xs = [x(c) for c in members]
        ys = [y(c) for c in members]
        if connect and len(members) > 1:
            ax.plot(xs, ys, color=S.FAMILY_COLOUR[members[0].family], lw=0.6, alpha=0.55, zorder=2)
        for c, xv, yv in zip(members, xs, ys):
            st = family_style(c, cfgs)
            ax.plot([xv], [yv], ls="none", marker=st["marker"], ms=3.8, mec=st["color"], mew=0.8,
                    mfc=st["color"] if c.open_licence else "white", zorder=3)


def winner_ring(ax, x, y, kind: str) -> None:
    """Outline marker for the best-combined configuration (explained in the legend)."""
    ax.plot([x], [y], ls="none", marker="o", ms=8.5, mfc="none", mec="black", mew=0.8, zorder=4)


# ---------------------------------------------------------------------------------- Fig. 2
def fig_image_tradeoffs(data) -> None:
    cfgs = data["image"]
    win = A.winners(cfgs)
    fig = plt.figure(figsize=(S.DOUBLE, 2.62))
    h = 1.72
    a = ax_at(fig, 0.47, 0.42, 1.95, h)
    b = ax_at(fig, 2.95, 0.42, 1.95, h)
    c = ax_at(fig, 5.62, 0.42, 1.42, h)

    scatter_families(a, cfgs, lambda c: c.quality["flip"], lambda c: c.gate_worst)
    a.axhline(0.95, color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    w = win["combined"]
    winner_ring(a, w.quality["flip"], w.gate_worst, "combined")
    a.set_xscale("log")
    a.set_xlim(0.012, 0.11)
    a.xaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.1]))
    a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_ylim(0.48, 1.01)
    a.set_xlabel("FLIP error, mean (lower is better)")
    a.set_ylabel("Bit accuracy after JPEG 75,\nworst subset")
    S.light_grid(a, "both")

    scatter_families(b, cfgs, lambda c: c.gpu_ms, lambda c: c.quality["flip"])
    winner_ring(b, w.gpu_ms, w.quality["flip"], "combined")
    b.set_xscale("log")
    plain_log(b.xaxis, [10, 30, 100, 300, 1000, 3000])
    b.set_yscale("log")
    b.yaxis.set_major_locator(FixedLocator([0.02, 0.03, 0.05, 0.1]))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_ylim(0.012, 0.11)
    b.set_xlabel("GPU embedding time (ms per megapixel)")
    b.set_ylabel("FLIP error, mean")
    S.light_grid(b, "both")

    ranking_panel(c, cfgs, "image")

    ring = Line2D([], [], ls="none", marker="o", ms=8.5, mfc="none", mec="black", mew=0.8, label="Best combined score")
    fig.legend(handles=family_legend(cfgs, [gate_handle(), ring]), loc="upper left", bbox_to_anchor=(0.055, 1.0),
               ncol=7, frameon=False)
    for ax, s in ((a, "a"), (b, "b"), (c, "c")):
        letter(fig, ax, s)
    S.save(fig, "fig2_image_tradeoffs")


def ranking_panel(ax, cfgs, modality: str) -> None:
    """Combined score of qualifying configurations: filled dot equal weights, open dot quality-weighted."""
    pool = sorted([c for c in cfgs if c.scores], key=lambda c: c.scores["combined"])
    y = np.arange(len(pool))
    for i, c in enumerate(pool):
        col = S.FAMILY_COLOUR[c.family]
        ax.plot([c.scores["combined_q"], c.scores["combined"]], [i, i], color=col, lw=0.6, alpha=0.6, zorder=2)
        ax.plot([c.scores["combined"]], [i], "o", color=col, ms=3.6, zorder=3)
        ax.plot([c.scores["combined_q"]], [i], "o", mfc="white", mec=col, mew=0.8, ms=3.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([c.label for c in pool], fontsize=5.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, len(pool) - 0.3)
    ax.set_xlabel("Combined score")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    S.light_grid(ax, "x")
    handles = [Line2D([], [], ls="none", marker="o", color=S.NEUTRAL, ms=3.6, label="Equal weights"),
               Line2D([], [], ls="none", marker="o", mfc="white", mec=S.NEUTRAL, ms=3.6, label="Quality-weighted")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, handletextpad=0.2,
              labelspacing=0.2)


# ---------------------------------------------------------------------------------- Fig. 3
def heatmap(ax, cfgs, attacks, key=lambda c, a: c.robustness.get(a)):
    M = np.array([[np.nan if key(c, a) is None else key(c, a) for a, _ in attacks] for c in cfgs], dtype=float)
    im = ax.imshow(M, cmap=BITACC_CMAP, vmin=0.5, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([lab for _, lab in attacks], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([c.label + ("" if c.open_licence else " †") for c in cfgs])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(attacks)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cfgs)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    return im, M


def order_for_heatmap(cfgs):
    fam_order = ["PixelSeal", "VideoSeal-1.0", "ChunkySeal", "WAM", "TrustMark", "InvisMark", "invisible-watermark",
                 "RivaGAN", "AudioSeal", "WavMark", "SilentCipher", "Perth"]
    return sorted(cfgs, key=lambda c: (fam_order.index(c.family), c.network, c.strength or 0))


def colourbar(fig, ax, im, label: str) -> None:
    W, H = fig.get_size_inches()
    bb = ax.get_position()
    cax = fig.add_axes((bb.x1 + 0.08 / W, bb.y0 + bb.height * 0.25, 0.07 / W, bb.height * 0.5))
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0.5, 0.75, 1.0])
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=6)
    cb.set_label(label, fontsize=6)


def fig_image_heatmap(data) -> None:
    cfgs = order_for_heatmap(data["image"])
    fig = plt.figure(figsize=(S.DOUBLE, 3.85))
    ax = ax_at(fig, 1.35, 0.62, 4.9, 3.18)
    im, _ = heatmap(ax, cfgs, IMAGE_ATTACKS)
    colourbar(fig, ax, im, "Mean bit accuracy")
    S.save(fig, "fig3_image_robustness")


# ---------------------------------------------------------------------------------- Fig. 4
def fig_audio(data) -> None:
    cfgs = data["audio"]
    win = A.winners(cfgs)
    fig = plt.figure(figsize=(S.DOUBLE, 4.12))
    top = 2.12
    a = ax_at(fig, 0.62, top, 1.8, 1.5)
    b = ax_at(fig, 2.95, top, 1.6, 1.5)
    c = ax_at(fig, 5.85, top, 1.2, 1.5)
    d = ax_at(fig, 1.45, 0.52, 4.9, 1.0)

    scatter_families(a, cfgs, lambda c: c.quality["pesq"], lambda c: c.gate_worst, connect=True)
    a.axhline(0.95, color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    w = win["combined"]
    winner_ring(a, w.quality["pesq"], w.gate_worst, "combined")
    a.set_xlim(4.0, 4.65)
    a.set_ylim(0.8, 1.01)
    a.set_xlabel("Wideband PESQ, mean (higher is better)")
    a.set_ylabel("Bit accuracy after MP3 128k,\nworst subset")
    S.light_grid(a, "both")

    scatter_families(b, cfgs, lambda c: c.gpu_ms, lambda c: c.quality["si_snr"], connect=True)
    winner_ring(b, w.gpu_ms, w.quality["si_snr"], "combined")
    b.set_xscale("log")
    b.set_xlim(1.2, 50)
    plain_log(b.xaxis, [2, 5, 10, 20, 50])
    b.set_xlabel("GPU embedding time (ms per s of audio)")
    b.set_ylabel("SI-SNR (dB)")
    S.light_grid(b, "both")

    ranking_panel(c, cfgs, "audio")

    # Perth carries no bit payload (presence detection only), so it has no bit accuracy.
    hm = [c for c in order_for_heatmap(cfgs) if c.family != "Perth"]
    im, _ = heatmap(d, hm, AUDIO_ATTACKS)
    colourbar(fig, d, im, "Mean bit accuracy")

    ring = Line2D([], [], ls="none", marker="o", ms=8.5, mfc="none", mec="black", mew=0.8, label="Best combined score")
    fig.legend(handles=family_legend(cfgs, [gate_handle(), ring]), loc="upper left", bbox_to_anchor=(0.055, 1.0),
               ncol=7, frameon=False)
    letter(fig, a, "a", dx=0.52)
    letter(fig, b, "b")
    letter(fig, c, "c", dx=0.85)
    letter(fig, d, "d", dx=1.3)
    S.save(fig, "fig4_audio")


# ---------------------------------------------------------------------------------- Fig. 5
def fig_video(data) -> None:
    cfgs = data["video"]
    win = A.winners(cfgs)
    fig = plt.figure(figsize=(S.DOUBLE, 4.18))
    top = 2.2
    a = ax_at(fig, 0.66, top, 1.76, 1.5)
    b = ax_at(fig, 3.05, top, 1.85, 1.5)
    c = ax_at(fig, 5.62, top, 1.42, 1.5)
    d = ax_at(fig, 1.45, 0.66, 4.9, 1.0)

    scatter_families(a, cfgs, lambda c: c.quality["vmaf"], lambda c: c.gate_worst)
    a.axhline(0.95, color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    w = win["combined"]
    winner_ring(a, w.quality["vmaf"], w.gate_worst, "combined")
    a.set_xlabel("VMAF, mean (higher is better)")
    a.set_ylabel("Bit accuracy after H.264 CRF23,\nworst clip")
    a.set_ylim(0.6, 1.01)
    S.light_grid(a, "both")

    # Per-clip bit accuracy after the gate attack: shows which clip fails, not only the mean.
    order = order_for_heatmap(cfgs)
    rng = np.random.default_rng(1)
    for i, cfg in enumerate(order):
        v = cfg.per_item_gate[np.isfinite(cfg.per_item_gate)]
        st = family_style(cfg, cfgs)
        jitter = rng.uniform(-0.18, 0.18, size=len(v))
        b.plot(i + jitter, v, ls="none", marker=st["marker"], ms=2.6, mfc=st["color"], mec="white", mew=0.3, zorder=3)
        b.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="black", lw=0.8, zorder=4)
    b.axhline(0.95, color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    b.set_xticks(range(len(order)))
    b.set_xticklabels([c.label for c in order], rotation=45, ha="right", rotation_mode="anchor", fontsize=5.5)
    b.set_ylabel("Bit accuracy after H.264 CRF23,\nper clip")
    b.set_ylim(0.6, 1.01)
    b.set_xlim(-0.6, len(order) - 0.4)
    S.light_grid(b, "y")

    ranking_panel(c, cfgs, "video")

    im, _ = heatmap(d, order, VIDEO_ATTACKS)
    colourbar(fig, d, im, "Mean bit accuracy")

    ring = Line2D([], [], ls="none", marker="o", ms=8.5, mfc="none", mec="black", mew=0.8, label="Best combined score")
    mean_h = Line2D([], [], color="black", lw=0.8, label="Mean over clips")
    fig.legend(handles=family_legend(cfgs, [gate_handle(), ring, mean_h]), loc="upper left",
               bbox_to_anchor=(0.055, 1.0), ncol=7, frameon=False)
    letter(fig, a, "a", dx=0.56)
    letter(fig, b, "b", dx=0.5)
    letter(fig, c, "c", dx=0.62)
    letter(fig, d, "d", dx=1.3)
    S.save(fig, "fig5_video")


# ---------------------------------------------------------------------------------- Fig. 6
def fig_fpr(data) -> None:
    fig = plt.figure(figsize=(S.DOUBLE, 2.55))
    pa = ax_at(fig, 1.3, 0.42, 1.75, 1.55)
    pb = ax_at(fig, 3.55, 0.42, 1.45, 1.55)
    pc = ax_at(fig, 5.55, 0.42, 1.4, 1.55)

    blind = [c for m in ("image", "audio") for c in data[m] if c.blind_fpr is not None and c.blind_fpr_ci]
    # One row per network: strengths share the detector and the unmarked trials.
    rows, seen = [], set()
    for c in blind:
        if c.network in seen:
            continue
        seen.add(c.network)
        rows.append(c)
    # The unmarked source is the trial unit: transforms and sample-rate renderings of one source
    # are not independent (A.fpr_by_source).
    src = {c.network: A.fpr_by_source(c) for c in rows}
    rate = {n: s["blind_src_fm"] / s["blind_src_n"] for n, s in src.items()}
    rows.sort(key=lambda c: rate[c.network])
    floor = 1e-3
    for i, c in enumerate(rows):
        lo, hi = src[c.network]["blind_src_ci"]
        col = S.FAMILY_COLOUR[c.family]
        pa.plot([max(lo, floor), hi], [i, i], color=col, lw=1.0, solid_capstyle="butt")
        if rate[c.network] > 0:
            pa.plot([rate[c.network]], [i], "o", color=col, ms=3.6, mfc=col if c.open_licence else "white")
        else:
            pa.plot([hi], [i], marker=4, ls="none", color=col, ms=4.5)
    pa.axvline(1e-3, color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)))
    pa.set_xscale("log")
    pa.set_xlim(5e-4, 0.4)
    plain_log(pa.xaxis, [0.001, 0.01, 0.1])
    pa.set_yticks(range(len(rows)))
    pa.set_yticklabels([r.network if r.family in ("TrustMark", "AudioSeal") else A.FAMILY_LABEL[r.family]
                       for r in rows])
    pa.set_ylim(-0.6, len(rows) - 0.4)
    pa.tick_params(axis="y", length=0)
    pa.spines["left"].set_visible(False)
    pa.set_xlabel("Blind false-positive rate, per source")
    S.light_grid(pa, "x")

    # TrustMark blind FPR per image subset: images on which any trial fired, exact 95 % interval.
    tm = [c for c in data["image"] if c.family == "TrustMark"]
    nets = sorted({c.network for c in tm})
    sets = ["kodak", "div2k", "clic", "hdr16"]
    set_label = {"kodak": "Kodak", "div2k": "DIV2K", "clic": "CLIC", "hdr16": "HDR"}
    raw_by_net = {}
    for net in nets:
        c0 = next(c for c in tm if c.network == net)
        raw = A._raw("image", c0.name)
        recs = raw["records"]
        fired: dict[str, bool] = {}
        for t in raw["fpr"]:
            fired[t["item"]] = fired.get(t["item"], False) or bool(t["native_detect"])
        counts = {s: [0, 0] for s in sets}
        for item, f in fired.items():
            s = recs[item]["set"]
            counts[s][0] += f
            counts[s][1] += 1
        raw_by_net[net] = counts
    width = 0.8 / len(nets)
    for j, net in enumerate(nets):
        col = S.shade(S.FAMILY_COLOUR["TrustMark"], 0.15 + 0.7 * j / max(1, len(nets) - 1))
        for i, s in enumerate(sets):
            k, n = raw_by_net[net][s]
            lo, hi = A.clopper_pearson(k, n)
            x = i - 0.4 + width * (j + 0.5)
            pb.plot([x, x], [lo, hi], color=col, lw=0.8)
            pb.plot([x], [k / n], ls="none", marker=S.TRUSTMARK_MARKER[net[-1]], color=col, ms=3.2)
    pb.set_xticks(range(len(sets)))
    pb.set_xticklabels([set_label[s] for s in sets])
    pb.set_ylabel("Blind false-positive rate, per image")
    pb.set_ylim(0, 0.8)
    pb.set_xlim(-0.55, len(sets) - 0.45)
    S.light_grid(pb, "y")
    pb.legend(handles=[Line2D([], [], ls="none", marker=S.TRUSTMARK_MARKER[n[-1]],
                             color=S.shade(S.FAMILY_COLOUR["TrustMark"], 0.15 + 0.7 * j / max(1, len(nets) - 1)),
                             ms=3.2, label=n) for j, n in enumerate(nets)],
             loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=0.6)

    # Expected-payload matching: matched bits on unmarked content vs the decision threshold.
    ref = next(c for c in data["image"] if c.name == "PixelSeal@0.15")
    raw = A._raw("image", ref.name)
    k_thr = json.loads(A.RESULTS.read_text())["image"]["configs"][
        [x["meta"]["name"] for x in json.loads(A.RESULTS.read_text())["image"]["configs"]].index(ref.name)]["fpr"]["k_threshold"]
    matches = np.array([m for t in raw["fpr"] for m in t["matches"]])
    n_bits = 256
    bins = np.arange(90, 180, 2)
    pc.hist(matches, bins=bins, density=True, color=S.FAMILY_COLOUR["PixelSeal"], alpha=0.75, lw=0)
    from scipy import stats as st
    xs = np.arange(90, 180)
    pc.plot(xs, st.binom.pmf(xs, n_bits, 0.5) * 1.0, color="black", lw=0.8)
    pc.axvline(k_thr, color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)))
    pc.set_xlabel("Matching bits of 256, unmarked")
    pc.set_ylabel("Density")
    pc.set_xlim(90, 180)
    S.light_grid(pc, "y")
    pc.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=S.FAMILY_COLOUR["PixelSeal"], alpha=0.75, lw=0,
                                    label="PixelSeal, observed"),
                      Line2D([], [], color="black", lw=0.8, label="Binomial(256, 0.5)"),
                      Line2D([], [], color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), label="Threshold k")],
             loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, labelspacing=0.2)
    pa.legend(handles=[Line2D([], [], color=S.NEUTRAL, marker="o", ms=3.4, lw=1.0, label="Rate, exact 95% CI"),
                      Line2D([], [], color=S.NEUTRAL, marker=4, ms=4.5, ls="none", label="Zero observed: upper bound"),
                      Line2D([], [], color=S.GATE_LINE, lw=0.7, ls=(0, (3, 2)), label="0.001 criterion")],
             loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, labelspacing=0.2)
    for ax, s in ((pa, "a"), (pb, "b"), (pc, "c")):
        letter(fig, ax, s, dx=0.34 if ax is not pa else 1.15)
    S.save(fig, "fig6_false_positives")


# ---------------------------------------------------------------------------------- Fig. 7
def fig_cost(data) -> None:
    fig = plt.figure(figsize=(S.DOUBLE, 2.5))
    spec = [("image", "USD per 1,000 megapixels", 0.95), ("audio", "USD per 1,000 hours of audio", 3.35),
            ("video", "USD per 1,000 MP (1080p)", 5.75)]
    axes = []
    for modality, xlabel, x0 in spec:
        ax = ax_at(fig, x0, 0.42, 1.25, 1.62)
        axes.append(ax)
        nets = {}
        for c in data[modality]:
            nets.setdefault(c.network, c)
        rows = sorted(nets.values(), key=lambda c: c.cost)
        for i, c in enumerate(rows):
            col = S.FAMILY_COLOUR[c.family]
            vals = [v for v in (c.cost_gpu, c.cost_cpu) if v is not None]
            if len(vals) == 2:
                ax.plot(vals, [i, i], color=col, lw=0.6, alpha=0.6)
            if c.cost_gpu is not None:
                ax.plot([c.cost_gpu], [i], "o", color=col, ms=3.4, mfc=col if c.open_licence else "white")
            if c.cost_cpu is not None:
                ax.plot([c.cost_cpu], [i], "s", color=col, ms=3.2, mfc="white", mew=0.8)
        if modality == "video":
            ax.set_xlim(0.0035, 0.0075)
            ax.xaxis.set_major_locator(FixedLocator([0.004, 0.005, 0.006, 0.007]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_xscale("log")
            ticks = {"image": [0.01, 0.1, 1], "audio": [2, 5, 10, 20, 50]}[modality]
            plain_log(ax.xaxis, ticks)
            vals = [v for c in rows for v in (c.cost_gpu, c.cost_cpu) if v is not None]
            ax.set_xlim(min(min(ticks), min(vals)) / 1.25, max(max(ticks), max(vals)) * 1.25)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.network if r.family in ("TrustMark", "AudioSeal") else A.FAMILY_LABEL[r.family]
                            for r in rows], fontsize=5.5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlabel)
        S.light_grid(ax, "x")
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", color=S.NEUTRAL, ms=3.4, label="GPU (g5.xlarge, A10G)"),
                        Line2D([], [], ls="none", marker="s", mfc="white", mec=S.NEUTRAL, ms=3.2,
                               label="CPU (m6a.xlarge, 4 vCPU)"),
                        Line2D([], [], ls="none", marker="o", mfc="white", mec=S.NEUTRAL, ms=3.4,
                               label="Reference only (licence)")],
               loc="upper left", bbox_to_anchor=(0.12, 1.0), ncol=3)
    for ax, s in zip(axes, "abc"):
        letter(fig, ax, s, dx=0.8)
    S.save(fig, "fig7_cost")


# ---------------------------------------------------------------------------------- Fig. 8
def fig_stability(data) -> None:
    st = A.stability()
    files = sorted(k for k in st if not k.startswith("_"))
    fig = plt.figure(figsize=(S.SINGLE, 2.5))
    a = ax_at(fig, 0.5 / S.SX, 1.05, 2.8 / S.SX, 1.1)
    within = [st[f]["same_host_max_dlogit"] for f in files]
    across = [st[f]["cross_host_max_dlogit"] for f in files]
    flips = [st[f]["cross_host_bit_flips"] for f in files]
    x = np.arange(len(files))
    a.vlines(x, within, across, color=S.GATE_LINE, lw=0.6)
    a.plot(x, within, "o", ms=3, color=S.FAMILY_COLOUR["VideoSeal-1.0"], mfc="white", mew=0.8)
    a.plot(x, across, "o", ms=3, color=S.FAMILY_COLOUR["TrustMark"])
    for i, f in enumerate(flips):
        if f:
            a.plot([i], [across[i]], marker="x", color="black", ms=4, mew=0.8)
    a.set_yscale("log")
    plain_log(a.yaxis, [0.001, 0.01, 0.1, 1])
    a.set_ylabel("Max |Δ logit| between decodes")
    a.set_xticks(x)
    def tick(f: str) -> str:
        if f.endswith(".jpg"):
            return "TrustMark-Q, image, JPEG 75"
        clip, rest = f.split("_5s_sw")
        strength, crf = rest.replace(".mp4", "").split("_crf")
        return f"{clip.replace('_', ' ')}, {strength}, CRF {crf}"
    a.set_xticklabels([tick(f) for f in files], rotation=60, ha="right", rotation_mode="anchor", fontsize=5.5)
    a.set_xlim(-0.6, len(files) - 0.4)
    S.light_grid(a, "y")
    a.legend(handles=[Line2D([], [], ls="none", marker="o", mfc="white", mec=S.FAMILY_COLOUR["VideoSeal-1.0"], ms=3,
                             label="Same host (CPU, CUDA, MPS)"),
                      Line2D([], [], ls="none", marker="o", color=S.FAMILY_COLOUR["TrustMark"], ms=3,
                             label="x86-64 vs arm64"),
                      Line2D([], [], ls="none", marker="x", color="black", ms=4, label="Raw bit flipped")],
             loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, columnspacing=0.8, labelspacing=0.2)
    S.save(fig, "fig8_stability")


# ---------------------------------------------------------------------------------- Fig. 9
def fig_gallery() -> None:
    from PIL import Image

    g = Path.home() / "op-wm-bench-work" / "out" / "gallery"
    if not (g / "summary.json").exists():
        print("gallery inputs missing; run gallery_embed.py first")
        return
    names = [("PixelSeal@0.15", "PixelSeal (0.15)"), ("VideoSeal-1.0@default", "Video Seal (0.2)"),
             ("WAM-MIT@default", "WAM (2)"), ("trustmark-Q@1.2", "TrustMark-Q (1.2)"), ("DWT-DCT-SVD", "DWT-DCT-SVD")]
    ref = np.asarray(Image.open(g / "original.png"), dtype=np.float32) / 255.0
    # A 256 x 256 crop with texture and flat regions, so both kinds of residual show.
    y0, x0, s = 120, 330, 256
    crop = (slice(y0, y0 + s), slice(x0, x0 + s))
    fig = plt.figure(figsize=(S.DOUBLE, 3.95 * S.SX))
    cell, gap, colgap = 1.14, 0.06, 0.1
    left = 0.3
    heads = ["Watermarked", "Residual ×20", "FLIP error map"]
    ims = []
    for r, (name, lab) in enumerate(names):
        m = np.asarray(Image.open(g / f"{name}.png"), dtype=np.float32) / 255.0
        fmap = np.load(g / f"{name}.flipmap.npy")
        col = r
        x = left + col * (cell + colgap)
        panels = [m[crop], np.clip(0.5 + 20 * (m[crop] - ref[crop]), 0, 1), fmap[crop]]
        for k, p in enumerate(panels):
            ax = ax_at(fig, x, 0.08 + (2 - k) * (cell + gap), cell, cell, square=True)
            if k == 2:
                ims.append(ax.imshow(p, cmap="magma", vmin=0, vmax=0.3, interpolation="nearest"))
            else:
                ax.imshow(p, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
                sp.set_color(S.NEUTRAL)
            if k == 0:
                ax.set_title(lab, fontsize=6.5, pad=3)
            if col == 0:
                ax.set_ylabel(heads[k], fontsize=6.5)
    W, H = fig.get_size_inches()
    sx = S.SX
    cax = fig.add_axes(((left + 4 * (cell + colgap) + cell + 0.08) * sx / W, (0.08 + 0.2 * cell) * sx / H, 0.07 / W,
                        0.6 * cell * sx / H))
    cb = fig.colorbar(ims[-1], cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=6)
    cb.set_label("FLIP", fontsize=6)
    S.save(fig, "fig9_gallery")


RESIDUALS = Path.home() / "op-wm-bench-work" / "out" / "residuals"


def _spectrogram_db(x: np.ndarray, sr: int, ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT in dB relative to `ref` (Hann, 2048-point, hop 512)."""
    from scipy.signal import stft

    f, t, z = stft(x, fs=sr, window="hann", nperseg=2048, noverlap=2048 - 512, boundary=None)
    return f, t, 20 * np.log10(np.maximum(np.abs(z), 1e-12) / ref)


def fig_audio_residuals() -> None:
    """Spectrograms of the original clips and of each model's residual (marked minus original)."""
    d = RESIDUALS / "audio"
    rows = [("AudioSeal-base@1", "AudioSeal base (1)"), ("AudioSeal-streaming@0.5", "AudioSeal streaming (0.5)"),
            ("WavMark", "WavMark"), ("Perth", "Perth")]
    rows = [(n, lab) for n, lab in rows if (d / f"{n}__speech.npz").exists()]
    if not rows:
        print("audio residual inputs missing; run residual_embed.py first")
        return
    clips = [("speech", "Speech (LibriSpeech, 44.1 kHz)"), ("music", "Music (44.1 kHz)")]
    nrow = len(rows) + 1
    fig = plt.figure(figsize=(S.DOUBLE, 0.95 * nrow + 0.55))
    w, h, gx, gy, x0, y0 = 2.45, 0.8, 0.25, 0.15, 1.2, 0.42
    im = None
    for j, (clip, title) in enumerate(clips):
        base = np.load(d / f"{rows[0][0]}__{clip}.npz")
        y, sr = base["y"], int(base["sr"])
        _, _, zref = _spectrogram_db(y, sr, 1.0)
        ref = float(np.max(10 ** (zref / 20)))
        panels = [("Original", y)] + [(lab, np.load(d / f"{n}__{clip}.npz")["yw"] - np.load(d / f"{n}__{clip}.npz")["y"])
                                      for n, lab in rows]
        for i, (lab, sig) in enumerate(panels):
            ax = ax_at(fig, x0 + j * (w + gx), y0 + (nrow - 1 - i) * (h + gy), w, h)
            f, t, z = _spectrogram_db(sig, sr, ref)
            im = ax.imshow(z, origin="lower", aspect="auto", cmap="magma", vmin=-100, vmax=0,
                           extent=(0, t[-1] if len(t) else 0, 0, f[-1] / 1000), interpolation="nearest")
            ax.set_ylim(0, 16)
            ax.set_yticks([0, 8, 16])
            if i == 0:
                ax.set_title(title, fontsize=7, pad=3)
            if j == 0:
                ax.set_ylabel(lab + "\n(kHz)", fontsize=6.5)
            else:
                ax.set_yticklabels([])
            if i == nrow - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])
    W, H = fig.get_size_inches()
    right = (x0 + 2 * w + gx) * S.SX
    cax = fig.add_axes(((right + 0.12) / W, y0 / H + 0.25, 0.07 / W, 0.4))
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=6)
    cb.set_label("Level relative to\noriginal peak (dB)", fontsize=6)
    S.save(fig, "fig10_audio_residuals")


def fig_video_residuals() -> None:
    """One 1080p frame after video-mode embedding: marked crop, residual x20, FLIP map."""
    d = RESIDUALS / "video"
    cols = [("PixelSeal@0.2", "PixelSeal (0.2)"), ("PixelSeal@0.4", "PixelSeal (0.4)"),
            ("VideoSeal-1.0@0.2", "Video Seal (0.2)"), ("VideoSeal-1.0@0.4", "Video Seal (0.4)")]
    cols = [(n, lab) for n, lab in cols if (d / f"{n}.npz").exists()]
    if not cols:
        print("video residual inputs missing; run residual_embed.py first")
        return
    fig = plt.figure(figsize=(S.DOUBLE, 4.55 * S.SX))
    cell, gap, colgap, left = 1.4, 0.06, 0.1, 0.35
    y0, x0c, s = 380, 760, 320
    crop = (slice(y0, y0 + s), slice(x0c, x0c + s))
    heads = ["Watermarked", "Residual ×20", "FLIP error map"]
    last = None
    for c, (name, lab) in enumerate(cols):
        z = np.load(d / f"{name}.npz")
        ref = z["ref"].astype(np.float32) / 255
        mk = z["marked"].astype(np.float32) / 255
        panels = [mk[crop], np.clip(0.5 + 20 * (mk[crop] - ref[crop]), 0, 1), z["flip"][crop]]
        for k, p in enumerate(panels):
            ax = ax_at(fig, left + c * (cell + colgap), 0.08 + (2 - k) * (cell + gap), cell, cell, square=True)
            if k == 2:
                last = ax.imshow(p, cmap="magma", vmin=0, vmax=0.3, interpolation="nearest")
            else:
                ax.imshow(p, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.4)
                sp.set_color(S.NEUTRAL)
            if k == 0:
                ax.set_title(lab, fontsize=6.5, pad=3)
            if c == 0:
                ax.set_ylabel(heads[k], fontsize=6.5)
    W, H = fig.get_size_inches()
    sx = S.SX
    cax = fig.add_axes(((left + len(cols) * (cell + colgap) + 0.02) * sx / W, (0.08 + 0.2 * cell) * sx / H,
                        0.07 / W, 0.6 * cell * sx / H))
    cb = fig.colorbar(last, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=6)
    cb.set_label("FLIP", fontsize=6)
    S.save(fig, "fig11_video_residuals")


def main() -> None:
    data = A.load()
    fig_image_tradeoffs(data)
    fig_image_heatmap(data)
    fig_audio(data)
    fig_video(data)
    fig_fpr(data)
    fig_cost(data)
    fig_stability(data)
    fig_gallery()
    fig_audio_residuals()
    fig_video_residuals()


if __name__ == "__main__":
    main()
