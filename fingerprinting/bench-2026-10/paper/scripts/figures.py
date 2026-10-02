"""Render every data figure of the manuscript from analysis.load().

Layout follows the 2026-09 watermark paper: every axes rectangle is placed in inches,
comparable panels share edges exactly, no numbers are drawn inside plotting areas, and
legends sit outside the axes. Colour = method class (Okabe-Ito); filled markers =
product-eligible (tier P), open markers = reference tier.

    PAPER_RESULTS=<dir> uv run --with matplotlib --with numpy --with scipy --with pillow python figures.py [names]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import analysis as A
import style as S

S.apply()

CLASS_COLOUR = {"hash": "#0072B2", "standard": "#E69F00", "general": "#009E73", "copydet": "#D55E00"}
R1_CMAP = LinearSegmentedColormap.from_list("r1", ["#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"])


def ax_at(fig, x, y, w, h, square=False):
    W, H = fig.get_size_inches()
    sy = S.SX if square else 1.0
    return fig.add_axes((x * S.SX / W, y * sy / H, w * S.SX / W, h * sy / H))


def ax_in(fig, x, y, w, h):
    """Axes at absolute page inches (no design-canvas scaling): used for image plates."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, y / H, w / W, h / H))


def letter(fig, ax, s, dx=0.34, dy=0.06):
    W, H = fig.get_size_inches()
    bb = ax.get_position()
    fig.text(bb.x0 - dx / W, bb.y1 + dy / H, s, fontsize=8, fontweight="bold", ha="left", va="bottom")


def plain_log(axis, ticks):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def colour(m: A.Method) -> str:
    return CLASS_COLOUR[m.cls]


def mstyle(m: A.Method) -> dict:
    c = colour(m)
    return {"color": c, "marker": "o" if m.tier == "P" else "s", "mec": c, "mfc": c if m.tier == "P" else "white"}


DASHES = ["-", (0, (4, 1.5)), (0, (1.5, 1)), (0, (5, 1, 1, 1)), (0, (2.5, 2.5))]


def lstyle(m: A.Method, group: list[A.Method]) -> dict:
    """Line style: class colour and marker, dash pattern by rank within its class in `group`."""
    same = [x.name for x in group if x.cls == m.cls]
    return {**mstyle(m), "ls": DASHES[same.index(m.name) % len(DASHES)]}


def class_legend(extra=(), classes=A.CLASS_ORDER):
    h = [Line2D([], [], color=CLASS_COLOUR[c], marker="o", lw=0, ms=4, label=A.CLASS_LABEL[c]) for c in classes]
    h += [Line2D([], [], color="#666", marker="o", lw=0, ms=4, label="Product-eligible (P)"),
          Line2D([], [], color="#666", marker="s", mfc="white", lw=0, ms=4, label="Reference tier (R)")]
    return h + list(extra)


def heatmap(ax, methods, attacks, key="R@1", vmin=0.0):
    M = np.array([[m.at(a, key) for m in methods] for a in attacks], float)
    im = ax.imshow(M, cmap=R1_CMAP, vmin=vmin, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.label + ("†" if m.tier == "R" else "") for m in methods], rotation=55, ha="right",
                       rotation_mode="anchor")
    for t, m in zip(ax.get_xticklabels(), methods):
        t.set_color(colour(m))
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([A.ATTACK_LABEL.get(a, a) for a in attacks])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(methods)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(attacks)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    return im


def family_breaks(ax, attacks, fam, n_cols):
    prev = None
    for i, a in enumerate(attacks):
        if prev is not None and fam[a] != prev:
            ax.axhline(i - 0.5, color="black", lw=0.6)
        prev = fam[a]


def colourbar(fig, ax, im, label, ticks=(0, 0.5, 1.0)):
    W, H = fig.get_size_inches()
    bb = ax.get_position()
    cax = fig.add_axes((bb.x1 + 0.08 / W, bb.y0 + bb.height * 0.3, 0.07 / W, bb.height * 0.4))
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.set_ticks(list(ticks))
    cb.ax.tick_params(length=1.5, width=0.4, labelsize=6)
    cb.set_label(label, fontsize=6)


# ---------------------------------------------------------------------------------- Fig. 2
def fig_image_heatmap(data):
    ms = data["image"]
    fam = ms[0].rec["attack_family"]
    attacks = A.attack_order(ms[0])
    fig = plt.figure(figsize=(S.DOUBLE, 5.6))
    ax = ax_at(fig, 1.05, 1.25, 5.5, 4.25)
    im = heatmap(ax, ms, attacks)
    family_breaks(ax, attacks, fam, len(ms))
    colourbar(fig, ax, im, "Recall@1 (ABO, 100k-image index)")
    S.save(fig, "fig2_image_heatmap")


# ---------------------------------------------------------------------------------- Fig. 3
def fig_image_tradeoffs(data):
    ms = data["image"]
    fig = plt.figure(figsize=(S.DOUBLE, 2.35))
    y = lambda m: m.pooled.get(f"TPR@{A.OP}", np.nan)
    panels = [("a", lambda m: m.bits, "Descriptor size (bits per asset)", [64, 256, 1024, 4096, 32768], True),
              ("b", lambda m: m.ms_per_item(), "Extraction time (ms per image)", [0.3, 1, 3, 10, 30, 100], True),
              ("c", lambda m: m.rec.get("disc21", {}).get("muAP", np.nan), "DISC21 subset μAP", None, False)]
    for i, (lab, xf, xl, ticks, logx) in enumerate(panels):
        ax = ax_at(fig, 0.55 + i * 2.2, 0.45, 1.75, 1.6)
        for m in ms:
            xv = xf(m)
            if not np.isfinite(xv):
                continue
            yv = y(m) if lab != "c" else m.pooled.get("muAP", np.nan)
            ax.plot([xv], [yv], ls="none", ms=3.8, mew=0.8, zorder=3, **mstyle(m))
        if logx:
            ax.set_xscale("log")
            plain_log(ax.xaxis, ticks)
        ax.set_xlabel(xl)
        ax.set_ylabel(f"TPR at pair FPR 10$^{{{int(A.OP.split('e')[-1])}}}$" if lab != "c" else "ABO μAP")
        ax.set_ylim(-0.02, 1.02)
        if lab == "c":
            ax.set_xlim(-0.02, 1.02)
            ax.plot([0, 1], [0, 1], color=S.GRID, lw=0.6, zorder=1)
        S.light_grid(ax, "both")
        letter(fig, ax, lab)
    fig.legend(handles=class_legend(), loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), frameon=False,
               handletextpad=0.2, columnspacing=0.8)
    S.save(fig, "fig3_image_tradeoffs")


# ---------------------------------------------------------------------------------- Fig. 4
SHOW = ["PDQ", "pHash-64", "ISCC-Image-64", "DINOv2-S", "DINOv2-S-LSH256", "OpenCLIP-B32", "SSCD-mixup", "ISC21-1st"]


def fig_image_families(data):
    ms = A.by_name(data["image"])
    show = [ms[n] for n in SHOW if n in ms]
    fig = plt.figure(figsize=(S.DOUBLE, 2.45))
    ax = ax_at(fig, 0.55, 0.45, 2.0, 1.65)
    keep = [("none", 1.0), ("crop90", 0.9), ("crop70", 0.7), ("crop50", 0.5), ("crop30", 0.3)]
    for m in show:
        ax.plot([k for _, k in keep], [m.at(a) for a, _ in keep], lw=0.9, ms=3, **lstyle(m, show), label=m.label)
    ax.invert_xaxis()
    ax.set_xlabel("Share of each side kept (centre crop)")
    ax.set_ylabel("Recall@1")
    ax.set_ylim(-0.02, 1.02)
    S.light_grid(ax)
    letter(fig, ax, "a")
    ax2 = ax_at(fig, 3.2, 0.45, 3.75, 1.65)
    fams = [f for f in A.attack_families(show[0]) if f != "none"]
    w = 0.8 / len(show)
    for i, m in enumerate(show):
        ax2.bar(np.arange(len(fams)) + (i - len(show) / 2 + 0.5) * w, [A.family_mean(m, f) for f in fams], width=w,
                color=colour(m) if m.tier == "P" else "white", edgecolor=colour(m), lw=0.5,
                alpha=0.35 + 0.65 * (i % 3) / 2)
    ax2.set_xticks(range(len(fams)))
    ax2.set_xticklabels([f.capitalize() for f in fams])
    ax2.set_ylabel("Mean Recall@1 over family")
    ax2.set_ylim(0, 1.02)
    S.light_grid(ax2)
    letter(fig, ax2, "b")
    fig.legend(handles=[Line2D([], [], lw=0.9, ms=3, **lstyle(m, show), label=m.label) for m in show],
               loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=0.8)
    S.save(fig, "fig4_image_families")


# ---------------------------------------------------------------------------------- Fig. 5
def fig_scores(data):
    ms = A.by_name(data["image"])
    show = [ms[n] for n in SHOW if n in ms]
    fig = plt.figure(figsize=(S.DOUBLE, 2.55))
    for i, m in enumerate(show):
        ax = ax_at(fig, 0.3 + (i % 4) * 1.06, 1.42 - (i // 4) * 0.97, 0.9, 0.62)
        h = m.rec["score_hist"]
        for key, c in (("neg_top1", "#9A9A9A"), ("pos_true", colour(m))):
            if not h.get(key):
                continue
            e = np.array(h[key]["edges"])
            cnt = np.array(h[key]["counts"], float)
            ax.stairs(cnt / cnt.sum(), e, color=c, fill=True, alpha=0.55 if key == "neg_top1" else 0.7, lw=0)
        t = m.rec["thresholds"].get(A.OP)
        if t is not None and np.isfinite(t):
            ax.axvline(t, color="black", lw=0.6, ls=(0, (3, 2)))
        ax.set_yticks([])
        ax.set_title(m.label, fontsize=6, pad=2)
        ax.tick_params(labelsize=5)
    letter(fig, fig.axes[0], "a", dx=0.22)
    ax = ax_at(fig, 5.55, 0.45, 1.3, 1.55)
    allm = data["image"]
    rates = [(m, m.rec.get("hard_negative", {}).get(A.OP_Q, {}).get("rate", np.nan)) for m in allm]
    rates = [(m, r) for m, r in rates if np.isfinite(r)]
    ax.barh(range(len(rates)), [r for _, r in rates], color=[colour(m) for m, _ in rates], height=0.7)
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([m.label for m, _ in rates], fontsize=4.5)
    ax.invert_yaxis()
    ax.set_xlabel("False binding rate,\nother photo of product")
    S.light_grid(ax, "x")
    letter(fig, ax, "b", dx=0.95)
    fig.legend(handles=[Line2D([], [], color="#9A9A9A", lw=4, label="Best score, never-registered query"),
                        Line2D([], [], color="#555", lw=4, alpha=0.7, label="Score to true asset (attacked)"),
                        Line2D([], [], color="black", lw=0.6, ls=(0, (3, 2)), label="Threshold, pair FPR 10$^{-7}$")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.36, 1.0), frameon=False, columnspacing=1.5)
    S.save(fig, "fig5_scores")


# ---------------------------------------------------------------------------------- Fig. 6/7
def fig_timebased(data, mod, name):
    ms = data[mod]
    if not ms:
        return
    fam = ms[0].rec["attack_family"]
    attacks = A.attack_order(ms[0])
    fig = plt.figure(figsize=(S.DOUBLE, 3.6))
    ax = ax_at(fig, 1.2, 0.95, 2.2 if mod == "audio" else 2.7, 2.5)
    im = heatmap(ax, ms, attacks)
    family_breaks(ax, attacks, fam, len(ms))
    colourbar(fig, ax, im, "Recall@1")
    letter(fig, ax, "a", dx=1.1)
    ax2 = ax_at(fig, 4.55 if mod == "audio" else 4.95, 0.95, 1.45 if mod == "audio" else 1.2, 2.5)
    targets = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for m in ms:
        ys = [m.pooled.get(f"TPR@pair@{t:g}", np.nan) for t in targets]
        ax2.plot(targets, ys, lw=0.9, ms=3, **lstyle(m, ms), label=m.label)
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("Pair-level false-positive rate")
    ax2.set_ylabel("TPR, all attacks pooled")
    ax2.set_ylim(-0.02, 1.02)
    S.light_grid(ax2, "both")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    letter(fig, ax2, "b")
    S.save(fig, name)


# ---------------------------------------------------------------------------------- Fig. 12
def fig_det(data):
    fig = plt.figure(figsize=(S.DOUBLE, 2.45))
    ms = A.by_name(data["image"])
    show = [ms[n] for n in SHOW if n in ms]
    targets = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    ax = ax_at(fig, 0.55, 0.45, 2.3, 1.6)
    for m in show:
        ax.plot(targets, [m.pooled.get(f"TPR@pair@{t:g}", np.nan) for t in targets], lw=0.9, ms=3, **lstyle(m, show),
                label=m.label)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Pair-level false-positive rate")
    ax.set_ylabel("TPR, image attacks pooled")
    ax.set_ylim(-0.02, 1.02)
    S.light_grid(ax, "both")
    letter(fig, ax, "a")
    # registry scale: expected false bindings per query = N x pair FPR
    ax2 = ax_at(fig, 3.55, 0.45, 2.1, 1.6)
    N = np.logspace(3, 10, 50)
    for t, c in zip(targets, ["#CCCCCC", "#A0A0A0", "#707070", "#383838", "#000000"]):
        ax2.plot(N, np.minimum(1, N * t), color=c, lw=0.9, label=f"pair FPR 10$^{{{int(np.log10(t))}}}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Registry size (assets)")
    ax2.set_ylabel("False bindings per query")
    S.light_grid(ax2, "both")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    letter(fig, ax2, "b")
    fig.legend(handles=[Line2D([], [], lw=0.9, ms=3, **lstyle(m, show), label=m.label) for m in show],
               loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.02), frameon=False, columnspacing=0.8)
    S.save(fig, "fig12_operating_points")


# ---------------------------------------------------------------------------------- Fig. 13
def fig_stability(data):
    st = data.get("stability", {}).get("methods")
    if not st:
        return
    # binary codes only: float descriptors are never bit-identical across hosts but agree to
    # 1 - cos < 2e-4 (reported in the text), so a "differs" bar would mislead
    ms = [m for m in data["image"] if m.name in st and "bits_differing_max" in st[m.name]]
    fig = plt.figure(figsize=(S.SINGLE, 2.2))
    ax = ax_at(fig, 1.0, 0.4, 2.0, 1.6)
    ax.barh(range(len(ms)), [1 - st[m.name]["identical_share"] for m in ms], color=[colour(m) for m in ms], height=0.7)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([m.label for m in ms], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("Share of binary codes differing across hosts")
    S.light_grid(ax, "x")
    S.save(fig, "fig13_stability")


FIGS = {"fig2": fig_image_heatmap, "fig3": fig_image_tradeoffs, "fig4": fig_image_families, "fig5": fig_scores,
        "fig6": lambda d: fig_timebased(d, "audio", "fig6_audio"), "fig7": lambda d: fig_timebased(d, "video", "fig7_video"),
        "fig12": fig_det, "fig13": fig_stability}


# ---------------------------------------------------------------------------------- media location
import os

MEDIA = Path(os.environ.get("PAPER_MEDIA", Path.home() / "op-fp-bench-work" / "gpu-media"))
EDIT_TYPES = ["splice", "copymove", "inpaint-telea", "inpaint-lama", "recolor", "textreplace"]
MAP_LABEL = {"pixel": "Pixel", "ssim": "SSIM", "lpips": "LPIPS", "dino": "DINOv2 patch", "pdq": "PDQ tiles",
             "fused": "Fused"}
MAP_COLOUR = {"pixel": "#0072B2", "ssim": "#56B4E9", "lpips": "#009E73", "dino": "#E69F00", "pdq": "#CC79A7",
              "fused": "#000000"}


def _heat(ax, img, cmap="magma"):
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _norm_map(m):
    m = np.asarray(m, np.float32)
    lo, hi = np.nanpercentile(m, 1), np.nanpercentile(m, 99.5)
    return np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)


# ---------------------------------------------------------------------------------- Fig. 8
def fig_localize(data):
    loc = data.get("localize", {}).get("spatial_sift")
    if not loc:
        return
    gal_dir = MEDIA / "localize" if (MEDIA / "localize").exists() else Path.home() / "op-fp-bench-work" / "out" / "localize"
    gallery = json.loads((gal_dir / "gallery.json").read_text()) if (gal_dir / "gallery.json").exists() else []
    order = {t: i for i, t in enumerate(EDIT_TYPES)}
    picks, seen = [], set()
    for g in sorted(gallery, key=lambda g: (order.get(g["type"], 9), -g.get("fused_f1", 0))):
        if g["type"] not in seen:
            picks.append(g)
            seen.add(g["type"])
    picks = picks[:6]
    H = 4.35
    fig = plt.figure(figsize=(S.DOUBLE, H))
    cols = [("orig", "Original"), ("query", "Edited query"), ("gt", "Edited region"), ("pixel", "Pixel map"),
            ("dino", "DINOv2 map"), ("fused", "Fused map")]
    cw, gap = 0.6, 0.03
    for r, g in enumerate(picks):
        z = np.load(gal_dir / (g["key"].replace("|", "__") + ".npz"))
        for c, (k, lab) in enumerate(cols):
            ax = ax_in(fig, 0.78 + c * (cw + gap), H - 0.22 - (r + 1) * (cw + gap), cw, cw)
            if k in ("orig", "query"):
                ax.imshow(z[k])
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            elif k == "gt":
                _heat(ax, z[k].astype(np.float32), cmap="Greys")
            else:
                _heat(ax, _norm_map(z[k]))
            if r == 0:
                ax.set_title(lab, fontsize=6, pad=2)
            if c == 0:
                post = {"none": "no post", "jpeg75": "JPEG 75", "resize0.75_jpeg80": "resize+JPEG",
                        "crop90": "crop 90%"}.get(g["post"], g["post"])
                kind = {"splice": "Splice", "copymove": "Copy-move", "inpaint-telea": "Inpaint (Telea)",
                        "inpaint-lama": "Inpaint (LaMa)", "recolor": "Recolour", "textreplace": "Text replace"}[g["type"]]
                ax.text(-0.08, 0.5, f"{kind}\n{int(round(100 * g['area']))}%, {post}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=5.5)
    if picks:
        letter(fig, fig.axes[0], "a", dx=0.7)
    # b: F1 by edit area, per map (oracle retrieval, SIFT alignment)
    ax = ax_in(fig, 5.05, 2.55, 1.35, 1.45)
    areas = ["0.01", "0.03", "0.1", "0.25"]
    for k in ["pixel", "ssim", "lpips", "dino", "pdq", "fused"]:
        by = loc["oracle"].get(k, {}).get("by", {}).get("area", {})
        ax.plot([float(a) * 100 for a in areas], [by.get(a, {}).get("f1", np.nan) for a in areas], color=MAP_COLOUR[k],
                marker="o", ms=2.5, lw=0.9, label=MAP_LABEL[k])
    ax.set_xscale("log")
    plain_log(ax.xaxis, [1, 3, 10, 25])
    ax.set_xlabel("Edited area (%)")
    ax.set_ylabel("Pixel F1 at 1% FPR")
    ax.set_ylim(0, 1)
    S.light_grid(ax, "both")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False, fontsize=5)
    letter(fig, ax, "b")
    # c: retrieval under edit by area
    ax2 = ax_in(fig, 5.05, 0.4, 1.35, 1.45)
    ret = data.get("localize_retrieval", {})
    ms = A.by_name(data["image"])
    for n, r in ret.items():
        by = r.get("at_threshold", {}).get(A.OP, {}).get("by", {}).get("area") or r.get("R@1_by", {}).get("area", {})
        if not by or n not in ms:
            continue
        ax2.plot([float(a) * 100 for a in areas], [by.get(a, {}).get("rate", np.nan) for a in areas], lw=0.9, ms=2.5,
                 **lstyle(ms[n], [ms[x] for x in ret if x in ms]), label=ms[n].label)
    ax2.set_xscale("log")
    plain_log(ax2.xaxis, [1, 3, 10, 25])
    ax2.set_xlabel("Edited area (%)")
    ax2.set_ylabel("Original retrieved")
    ax2.set_ylim(0, 1.02)
    S.light_grid(ax2, "both")
    ax2.legend(loc="lower left", frameon=False, fontsize=5)
    letter(fig, ax2, "c")
    S.save(fig, "fig8_localization")


# ---------------------------------------------------------------------------------- Fig. 9
def fig_security(data):
    ev = data.get("security", {}).get("evasion")
    col = data.get("security", {}).get("collision")
    if not ev:
        return
    ms = A.by_name(data["image"])
    fig = plt.figure(figsize=(S.DOUBLE, 2.55))
    eps = sorted({int(e) for m in ev.get("whitebox", {}).values() for e in m if e.isdigit()})
    ax = ax_at(fig, 0.55, 0.45, 1.8, 1.55)
    groups = [(n, ev["whitebox"][n]) for n in ev.get("whitebox", {})] + [(n, ev["surrogate"][n]) for n in ev.get("surrogate", {})]
    col_names = list((col or {}).get("methods", {}))
    # one style per method across all panels and the shared legend
    shown = [ms[n] for n in A.ORDER if n in ms and (n in dict(groups) or n in col_names)]
    for n, d in groups:
        if n not in ms:
            continue
        t7 = ms[n].rec["thresholds"].get(A.OP)

        def rate7(x):
            # re-scored at the headline threshold from the stored final scores (see build_numbers)
            ok = [r for r in x.get("rows", []) if t7 is not None and np.isfinite(t7) and r["score_clean"] >= t7]
            return sum(r["score_adv"] < t7 for r in ok) / len(ok) if ok else np.nan
        ys = [rate7(d.get(str(e), {})) for e in eps]
        if all(not np.isfinite(y) for y in ys):
            continue  # no operating point to evade (threshold unreachable for this code)
        ax.plot(eps, ys, lw=0.9, ms=3, **lstyle(ms[n], shown), label=ms[n].label)
    ax.set_xscale("log", base=2)
    plain_log(ax.xaxis, eps)
    ax.set_xlabel("$L_\\infty$ budget (grey levels)")
    ax.set_ylabel("Evasion success (pair FPR $10^{-7}$)")
    ax.set_ylim(-0.02, 1.02)
    S.light_grid(ax, "both")
    letter(fig, ax, "a")
    # b: transfer from the ensemble at a fixed budget, all methods
    ax2 = ax_at(fig, 3.45, 0.45, 1.45, 1.55)
    tr_eps = "8" if "8" in ev.get("transfer", {}) else (sorted(ev.get("transfer", {}), key=int)[-1] if ev.get("transfer") else None)
    if tr_eps:
        rows = [(ms[n], ev["transfer"][tr_eps][n]["summary"].get("rate", np.nan)) for n in ms if n in ev["transfer"][tr_eps]]
        ax2.barh(range(len(rows)), [r for _, r in rows], color=[colour(m) if m.tier == "P" else "white" for m, _ in rows],
                 edgecolor=[colour(m) for m, _ in rows], lw=0.6, height=0.7)
        ax2.set_yticks(range(len(rows)))
        ax2.set_yticklabels([m.label + (" (n/a)" if not np.isfinite(r) else "") for m, r in rows], fontsize=4.5)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel(f"Transfer evasion, budget {tr_eps}")
        S.light_grid(ax2, "x")
    letter(fig, ax2, "b", dx=0.75)
    # c: forgery (targeted collision)
    ax3 = ax_at(fig, 5.5, 0.45, 1.55, 1.55)
    if col:
        ce = sorted({int(e) for m in col["methods"].values() for e in m if str(e).isdigit()})
        cm = [n for n in col["methods"] if n in ms]
        for n in cm:
            d = col["methods"][n]
            ax3.plot(ce, [d.get(str(e), {}).get("summary", {}).get("rate", np.nan) for e in ce], lw=0.9, ms=3,
                     **lstyle(ms[n], shown))
        ax3.set_xscale("log", base=2)
        plain_log(ax3.xaxis, ce)
    ax3.set_xlabel("$L_\\infty$ budget (grey levels)")
    ax3.set_ylabel("Forgery success (top-1)")
    ax3.set_ylim(-0.02, 1.02)
    S.light_grid(ax3, "both")
    letter(fig, ax3, "c")
    fig.legend(handles=[Line2D([], [], lw=0.9, ms=3, **lstyle(m, shown), label=m.label) for m in shown],
               loc="upper center", ncol=7, bbox_to_anchor=(0.5, 0.995), frameon=False, columnspacing=0.8)
    S.save(fig, "fig9_security")


# ---------------------------------------------------------------------------------- Fig. 10
INV_ORDER = ["original", "mean-image", "PDQ", "pHash-64", "ISCC-Image-64", "DINOv2-S", "SSCD-mixup", "OpenCLIP-B32",
             "DINOv2-S-LSH256", "DINOv2-S-LSH256 (wrong key)"]


def fig_inversion(data):
    inv = data.get("security", {}).get("inversion")
    gal = MEDIA / "security" / "inversion_gallery.npz"
    if not gal.exists():
        gal = Path.home() / "op-fp-bench-work" / "out" / "security" / "inversion_gallery.npz"
    if not inv or not gal.exists():
        return
    z = np.load(gal)
    rows = [k for k in INV_ORDER if k in z.files]
    n = z[rows[0]].shape[0]
    cw = 0.3
    H = 0.25 + len(rows) * (cw + 0.02) + 0.1
    fig = plt.figure(figsize=(S.DOUBLE, H))
    for r, k in enumerate(rows):
        for c in range(n):
            ax = ax_in(fig, 1.15 + c * (cw + 0.02), H - 0.15 - (r + 1) * (cw + 0.02), cw, cw)
            ax.imshow(z[k][c])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                lab = {"original": "Original", "mean-image": "Mean image", "DINOv2-S-LSH256 (wrong key)": "LSH-256, wrong key",
                       "DINOv2-S-LSH256": "LSH-256, right key"}.get(k, A.LABEL.get(k, k))
                ax.text(-0.1, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=5.5)
    letter(fig, fig.axes[0], "a", dx=1.05)
    ax = ax_in(fig, 4.85, 0.45, 1.5, H - 0.8)
    meth = [k for k in INV_ORDER if k in inv["methods"]]
    vals = [inv["methods"][k]["reid_top1"] for k in meth]
    cis = [inv["methods"][k].get("reid_ci", [np.nan, np.nan]) for k in meth]
    ax.barh(range(len(meth)), vals, color="#4D4D4D", height=0.65,
            xerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]], error_kw={"lw": 0.5})
    ch = next(iter(inv["methods"].values())).get("chance")
    if ch:
        ax.axvline(ch, color="black", lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(range(len(meth)))
    ax.set_yticklabels([{"mean-image": "Mean image", "DINOv2-S-LSH256": "LSH-256, right key",
                         "DINOv2-S-LSH256 (wrong key)": "LSH-256, wrong key"}.get(k, A.LABEL.get(k, k)) for k in meth],
                       fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("Re-identification from reconstruction")
    S.light_grid(ax, "x")
    letter(fig, ax, "b", dx=0.9)
    S.save(fig, "fig10_inversion")


# ---------------------------------------------------------------------------------- Fig. 11
def fig_wmcombo(data):
    wc = data.get("wmcombo")
    if not wc:
        return
    fig = plt.figure(figsize=(S.DOUBLE, 2.7))
    panels = [("image", "DINOv2-S", "TrustMark-Q", 0.55, 3.05), ("audio", "Chromaprint", "AudioSeal", 4.05, 1.4),
              ("video", "ISCC-Video-64", "Video Seal", 5.8, 1.3)]
    for i, (mod, name, wmname, x0, w) in enumerate(panels):
        res = wc.get(mod, {}).get("methods", {})
        if name not in res:
            continue
        s_ = res[name]["summary"]
        order = [a_ for a_ in A.ORDER_BY_MODALITY[mod] if a_ in s_]
        both = np.array([s_[a_]["both"] for a_ in order])
        wmo = np.array([s_[a_]["wm_only"] for a_ in order])
        fpo = np.array([s_[a_]["fp_only"] for a_ in order])
        ax = ax_at(fig, x0, 0.95, w, 1.3)
        xs = np.arange(len(order))
        ax.bar(xs, both, color="#4D4D4D", width=0.8)
        ax.bar(xs, wmo, bottom=both, color="#56B4E9", width=0.8)
        ax.bar(xs, fpo, bottom=both + wmo, color="#E69F00", width=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([A.ATTACK_LABEL.get(a_, a_) for a_ in order], rotation=90, fontsize=4)
        ax.tick_params(axis="x", length=0, pad=1)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(0, 1.02)
        if i == 0:
            ax.set_ylabel("Share meeting success criteria")
        ax.set_title(f"{wmname} + {A.LABEL.get(name, name).replace('ISCC Video-64', 'ISCC-64')}", fontsize=6, pad=2)
        S.light_grid(ax)
        letter(fig, ax, "abc"[i], dx=0.25)
    fig.legend(handles=[Line2D([], [], color=c, lw=4, label=l) for c, l in
                        (("#4D4D4D", "Both"), ("#56B4E9", "Watermark only"), ("#E69F00", "Fingerprint only"))],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.5)
    S.save(fig, "fig11_wm_complementarity")


FIGS.update({"fig8": fig_localize, "fig9": fig_security, "fig10": fig_inversion, "fig11": fig_wmcombo})


def main() -> None:
    data = A.load()
    for k, f in FIGS.items():
        if len(sys.argv) > 1 and k not in sys.argv[1:]:
            continue
        f(data)
        print("rendered", k)


if __name__ == "__main__":
    main()
