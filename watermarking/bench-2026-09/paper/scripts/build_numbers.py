"""Write every table and every number the manuscript cites, from analysis.load().

    uv run --with numpy --with scipy python build_numbers.py          # write generated/
    uv run --with numpy --with scipy python build_numbers.py --check  # fail if generated/ is stale

The manuscript cites values through \\val{key}; `generated/numbers.tex` defines each
key and `\\val` raises a LaTeX error for an undefined one, so a typo cannot silently
print nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import analysis as A

GEN = A.PAPER / "generated"
TAB = GEN / "tables"

COST_MARK = r"\textsuperscript{\costmark}"
SPEED_MARK = r"\textsuperscript{\speedmark}"
BEST_MARK = r"\textsuperscript{\bestmark}"


def fmt(v: float | None, nd: int = 2, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if pct:
        return f"{100 * v:.{nd}f}"
    return f"{v:.{nd}f}"


def fmt_sig(v: float | None, sig: int = 2) -> str:
    """Significant-figure format for quantities spanning orders of magnitude (cost, latency)."""
    if v is None:
        return "--"
    if v == 0:
        return "0"
    digits = max(0, sig - 1 - int(math.floor(math.log10(abs(v)))))
    return f"{v:.{digits}f}"


def tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def name_cell(c: A.Config, win: dict) -> str:
    marks = ""
    if win.get("cost") is c:
        marks += COST_MARK
    if win.get("speed") is c:
        marks += SPEED_MARK
    if win.get("combined") is c:
        marks += BEST_MARK
    dag = "" if c.open_licence else r"\textsuperscript{\dag}"
    return tex_escape(c.label) + dag + marks


def best_of(cfgs, key, higher: bool):
    pool = [c for c in cfgs if c.qualifies and key(c) is not None]
    if not pool:
        return None
    return (max if higher else min)(key(c) for c in pool)


def bold_if(v: float | None, best: float | None, s: str) -> str:
    return rf"\textbf{{{s}}}" if v is not None and best is not None and abs(v - best) < 1e-12 else s


def bq(c: A.Config, v: float | None, best: float | None, s: str) -> str:
    """Bold only a qualifying configuration's value: the best column value is chosen among
    qualifying rows, and a non-qualifying row that ties it is not a best performer."""
    return bold_if(v, best if c.qualifies else None, s)


def qualifies_cell(c: A.Config) -> str:
    return r"\checkmark" if c.qualifies else "--"


def table_env(caption: str, label: str, colspec: str, header: list[str], rows: list[str], note: str,
              wide: bool = True, size: str = r"\footnotesize", colsep: str = "3.4pt") -> str:
    env = "table*" if wide else "table"
    return "\n".join([
        rf"\begin{{{env}}}[!t]",
        r"\centering",
        size,
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\setlength{{\tabcolsep}}{{{colsep}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\par\vspace{{2pt}}\parbox{{\linewidth}}{{\raggedright\scriptsize {note}}}",
        rf"\end{{{env}}}",
        "",
    ])


MARK_NOTE = (r"\costmark\ lowest cost, \speedmark\ fastest GPU embedding, \bestmark\ highest combined score, each among "
             r"qualifying configurations (Q). Ties between strengths of one network are broken by quality. "
             r"Bold: best qualifying value per column. \dag\ Code or weights lack an open licence; reported for "
             r"reference and excluded from ranking.")


def image_table(cfgs) -> str:
    win = A.winners(cfgs)
    order = sorted(cfgs, key=lambda c: (-c.scores.get("combined", -1), c.quality["flip"]))
    bests = {
        "psnr": best_of(cfgs, lambda c: c.quality["psnr"], True),
        "ssim": best_of(cfgs, lambda c: c.quality["ssim"], True),
        "lpips": best_of(cfgs, lambda c: c.quality["lpips"], False),
        "flip": best_of(cfgs, lambda c: c.quality["flip"], False),
        "cvvdp": best_of(cfgs, lambda c: c.quality["cvvdp"], True),
        "pool": best_of(cfgs, lambda c: c.gate_pooled, True),
        "worst": best_of(cfgs, lambda c: c.gate_worst, True),
        "gpu": best_of(cfgs, lambda c: c.gpu_ms, False),
        "cost": best_of(cfgs, lambda c: c.cost, False),
        "comb": best_of(cfgs, lambda c: c.scores.get("combined"), True),
    }
    rows = []
    for c in order:
        q = c.quality
        rows.append(" & ".join([
            name_cell(c, win), qualifies_cell(c),
            bq(c, q["psnr"], bests["psnr"], fmt(q["psnr"], 1)),
            bq(c, q["ssim"], bests["ssim"], fmt(q["ssim"], 4)),
            bq(c, q["lpips"], bests["lpips"], fmt(q["lpips"], 4)),
            bq(c, q["flip"], bests["flip"], fmt(q["flip"], 3)),
            bq(c, q["cvvdp"], bests["cvvdp"], fmt(q["cvvdp"], 2)),
            bq(c, c.gate_pooled, bests["pool"], fmt(c.gate_pooled, 3)),
            bq(c, c.gate_worst, bests["worst"], fmt(c.gate_worst, 3)),
            fmt(c.blind_fpr, 3) if c.blind_fpr is not None else "n/a",
            bq(c, c.gpu_ms, bests["gpu"], fmt_sig(c.gpu_ms, 3)),
            bq(c, c.cost, bests["cost"], fmt_sig(c.cost, 2)) + (r"\textsuperscript{c}" if c.cost_device == "CPU" else ""),
            bq(c, c.scores.get("combined"), bests["comb"], fmt(c.scores.get("combined"), 2) if c.scores else "--"),
        ]) + r" \\")
    header = ["Configuration", "Q", "PSNR", "SSIM", "LPIPS", "FLIP", "CVVDP", r"\multicolumn{2}{c}{JPEG 75 bit acc.}",
              "Blind", "GPU", "Cost", "Comb."]
    sub = [" ", " ", r"(dB)$\uparrow$", r"$\uparrow$", r"$\downarrow$", r"$\downarrow$", r"$\uparrow$",
           r"pooled", r"worst set", r"FPR$\downarrow$", r"(ms/MP)", r"(\$/k MP)", r"$\uparrow$"]
    head = " & ".join(header) + r" \\" + "\n" + " & ".join(sub)
    return table_env(
        r"Image watermarking on 85 images (24 Kodak, 25 DIV2K, 28 CLIC, 8 16-bit HDR). Means over images (FLIP: 60 "
        r"images without DIV2K; CVVDP: 32 Kodak and HDR images); bit accuracy after JPEG at quality 75, pooled and "
        r"on the worst content subset.",
        "tab:image", "lcrrrrrrrrrrr", [head], rows,
        MARK_NOTE + r" Blind FPR: share of unmarked decodes (four per image) on which the model's own detector fires, per "
        r"decode; Table~\ref{tab:fpr} gives it per source (n/a: no blind "
        r"detector). Cost: estimated USD per 1{,}000 megapixels on the cheaper of GPU and CPU (\textsuperscript{c}: "
        r"CPU cheaper). Latency is the median over the network's strength settings (Section~\ref{sec:cost}).")


def audio_table(cfgs) -> str:
    win = A.winners(cfgs)
    order = sorted(cfgs, key=lambda c: (-c.scores.get("combined", -1), -c.quality["pesq"]))
    b = {
        "snr": best_of(cfgs, lambda c: c.quality["snr"], True),
        "si": best_of(cfgs, lambda c: c.quality["si_snr"], True),
        "pesq": best_of(cfgs, lambda c: c.quality["pesq"], True),
        "stoi": best_of(cfgs, lambda c: c.quality["stoi"], True),
        "lufs": best_of(cfgs, lambda c: c.quality["abs_dlufs"], False),
        "pool": best_of(cfgs, lambda c: c.gate_pooled, True),
        "worst": best_of(cfgs, lambda c: c.gate_worst, True),
        "gpu": best_of(cfgs, lambda c: c.gpu_ms, False),
        "cost": best_of(cfgs, lambda c: c.cost, False),
        "comb": best_of(cfgs, lambda c: c.scores.get("combined"), True),
    }
    rows = []
    for c in order:
        q = c.quality
        pooled = "--" if c.family == "Perth" else fmt(c.gate_pooled, 3)
        worst = "--" if c.family == "Perth" else fmt(c.gate_worst, 3)
        rows.append(" & ".join([
            name_cell(c, win), qualifies_cell(c),
            bq(c, q["snr"], b["snr"], fmt(q["snr"], 1)),
            bq(c, q["si_snr"], b["si"], fmt(q["si_snr"], 1)),
            bq(c, q["pesq"], b["pesq"], fmt(q["pesq"], 2)),
            fmt(q["pesq_music"], 2),
            bq(c, q["stoi"], b["stoi"], fmt(q["stoi"], 4)),
            bq(c, q["abs_dlufs"], b["lufs"], fmt(q["abs_dlufs"], 3)),
            bq(c, c.gate_pooled, b["pool"], pooled),
            bq(c, c.gate_worst, b["worst"], worst),
            fmt(c.blind_fpr, 3) if c.blind_fpr is not None else "n/a",
            bq(c, c.gpu_ms, b["gpu"], fmt_sig(c.gpu_ms, 3)),
            bq(c, c.cost, b["cost"], fmt_sig(c.cost, 2)) + (r"\textsuperscript{c}" if c.cost_device == "CPU" else ""),
            bq(c, c.scores.get("combined"), b["comb"], fmt(c.scores.get("combined"), 2) if c.scores else "--"),
        ]) + r" \\")
    header = ["Configuration", "Q", "SNR", "SI-SNR", r"\multicolumn{2}{c}{PESQ}", "STOI", r"$|\Delta$LUFS$|$",
              r"\multicolumn{2}{c}{MP3 128k bit acc.}", "Blind", "GPU", "Cost", "Comb."]
    sub = [" ", " ", r"(dB)$\uparrow$", r"(dB)$\uparrow$", r"speech$\uparrow$", r"music", r"$\uparrow$", r"$\downarrow$",
           "pooled", "worst set", r"FPR$\downarrow$", "(ms/s)", r"(\$/kh)", r"$\uparrow$"]
    head = " & ".join(header) + r" \\" + "\n" + " & ".join(sub)
    return table_env(
        r"Audio watermarking on 138 inputs: 46 source clips (40 LibriSpeech utterances, 6 music excerpts), each "
        r"rendered at 16, 44.1 and 48~kHz. Means over inputs. PESQ is a speech model: the primary PESQ is the mean "
        r"over the 120 speech inputs, and music PESQ (18 inputs) is shown for reference only. Bit accuracy after MP3 "
        r"at 128~kb/s.",
        "tab:audio", "lcrrrrrrrrrrrr", [head], rows,
        MARK_NOTE + r" Perth carries no bit payload, so it has no bit accuracy and cannot qualify; its detection "
        r"rate is reported in the text. Cost: estimated USD per 1{,}000 hours of audio. GPU: ms of compute per "
        r"second of audio. Blind FPR: share of unmarked decodes (nine per clip) on which the model's own detector "
        r"fires; Table~\ref{tab:fpr} gives it per source.", size=r"\scriptsize")


def video_table(cfgs) -> str:
    win = A.winners(cfgs)
    order = sorted(cfgs, key=lambda c: (-c.scores.get("combined", -1), -c.quality["vmaf"]))
    b = {
        "psnr": best_of(cfgs, lambda c: c.quality["psnr"], True),
        "ssim": best_of(cfgs, lambda c: c.quality["ssim"], True),
        "lpips": best_of(cfgs, lambda c: c.quality["lpips"], False),
        "flip": best_of(cfgs, lambda c: c.quality["flip"], False),
        "vmaf": best_of(cfgs, lambda c: c.quality["vmaf"], True),
        "pool": best_of(cfgs, lambda c: c.gate_pooled, True),
        "worst": best_of(cfgs, lambda c: c.gate_worst, True),
        "gpu": best_of(cfgs, lambda c: c.gpu_ms, False),
        "cost": best_of(cfgs, lambda c: c.cost, False),
        "comb": best_of(cfgs, lambda c: c.scores.get("combined"), True),
    }
    rows = []
    for c in order:
        q = c.quality
        rows.append(" & ".join([
            name_cell(c, win), qualifies_cell(c),
            bq(c, q["psnr"], b["psnr"], fmt(q["psnr"], 1)),
            bq(c, q["ssim"], b["ssim"], fmt(q["ssim"], 4)),
            bq(c, q["lpips"], b["lpips"], fmt(q["lpips"], 4)),
            bq(c, q["flip"], b["flip"], fmt(q["flip"], 3)),
            bq(c, q["vmaf"], b["vmaf"], fmt(q["vmaf"], 1)),
            bq(c, c.gate_pooled, b["pool"], fmt(c.gate_pooled, 3)),
            bq(c, c.gate_worst, b["worst"], fmt(c.gate_worst, 3)),
            bq(c, c.gpu_ms, b["gpu"], fmt_sig(c.gpu_ms, 3)),
            fmt(1000.0 / c.gpu_ms / A.VIDEO_MP_PER_FRAME, 1),
            bq(c, c.cost, b["cost"], fmt_sig(c.cost, 2)),
            bq(c, c.scores.get("combined"), b["comb"], fmt(c.scores.get("combined"), 2) if c.scores else "--"),
        ]) + r" \\")
    header = ["Configuration", "Q", "PSNR", "SSIM", "LPIPS", "FLIP", "VMAF", r"\multicolumn{2}{c}{H.264 CRF 23 bit acc.}",
              "GPU", "fps", "Cost", "Comb."]
    sub = [" ", " ", r"(dB)$\uparrow$", r"$\uparrow$", r"$\downarrow$", r"$\downarrow$", r"$\uparrow$", "pooled",
           "worst clip", r"(ms/MP)", r"1080p", r"(\$/k MP)", r"$\uparrow$"]
    head = " & ".join(header) + r" \\" + "\n" + " & ".join(sub)
    return table_env(
        r"Video watermarking on six 2.5-s 1080p clips. Bit accuracy after H.264 at CRF 23, pooled and on the "
        r"worst clip.",
        "tab:video", "lcrrrrrrrrrrr", [head], rows,
        MARK_NOTE + r" fps: embedding throughput for 1080p frames on one A10G. Cost: estimated USD per 1{,}000 "
        r"megapixels of 1080p video (GPU only; no CPU run).")


def fpr_table(data) -> str:
    """False positives, with the theoretical per-key probability and the observed counts in
    separate columns. Observed bounds use the unmarked source as the trial unit (see
    A.fpr_by_source); the random-payload comparisons are reported as counts only, because they
    reuse the same decodes and add no independent media."""
    rows = []
    for modality in ("image", "audio", "video"):
        if rows:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{12}}{{l}}{{\emph{{{modality.capitalize()}}}}} \\")
        seen = set()
        for c in sorted(data[modality], key=lambda c: c.network):
            if c.network in seen:
                continue
            seen.add(c.network)
            src = A.fpr_by_source(c) or {}
            if c.blind_fpr_ci:
                k = round(c.blind_fpr * c.blind_trials)
                blind = (rf"{k}/{c.blind_trials} & {src['blind_src_fm']}/{src['blind_src_n']} & "
                         rf"{fmt_sig(src['blind_src_ci'][1], 2)}")
            else:
                blind = r"\multicolumn{3}{c}{none}"
            split = A.key_trial_split(c)
            if split:
                m_, e_ = f"{A.payload_null(c.nbits, c.k_threshold):.1e}".split("e")
                theory = rf"{c.k_threshold}/{c.nbits} & ${m_}\times10^{{{int(e_)}}}$"
                observed = (rf"{split['own_fm']}/{split['own_n']} & {src['own_src_fm']}/{src['own_src_n']} & "
                            rf"{fmt_sig(src['own_src_upper'], 2)} & {split['rnd_fm']}/{split['rnd_n']} & "
                            rf"{A.max_matched_bits(c)}")
            elif c.nbits:
                theory = rf"\multicolumn{{2}}{{c}}{{none ({c.nbits} bits)}}"
                observed = r"\multicolumn{5}{c}{--}"
            else:
                theory = r"\multicolumn{2}{c}{no payload}"
                observed = r"\multicolumn{5}{c}{--}"
            name = c.network if c.family in ("TrustMark", "AudioSeal") else A.FAMILY_LABEL[c.family]
            if not c.open_licence:
                name += r"\textsuperscript{\dag}"
            ev = {"analytic": "analytic", "observed": "observed", "fails": "fails"}[c.fpr_evidence]
            rows.append(rf"{tex_escape(name)} & {theory} & {observed} & {blind} & {ev} \\")
    header = ["Network", r"\multicolumn{2}{c}{Theoretical}", r"\multicolumn{3}{c}{Own payload}",
              r"\multicolumn{2}{c}{Random payloads}", r"\multicolumn{3}{c}{Blind detector}", "Evidence"]
    sub = [" ", "$k/n$", r"$P_{\mathrm{FM}}$", "trials", "sources", "95\\% upper", "trials", "max", "trials",
           "sources", "95\\% upper", " "]
    head = (" & ".join(header) + r" \\" + "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}"
            + "\n" + " & ".join(sub))
    return table_env(
        r"False positives on unmarked content. Theoretical: the per-key false-match probability $P_{\mathrm{FM}}$ of "
        r"an expected-payload test that requires $k$ of $n$ bits to match, if unmarked content decodes to independent "
        r"fair bits. Own payload: false matches against the payload recorded for the item, the check a verifier "
        r"makes. Random payloads: the same decodes compared with unrelated payloads, which tests agreement with "
        r"payloads the content was never marked with, conditional on these decodes. Blind detector: false "
        r"detections of the model's own detector.",
        "tab:fpr", "lrrrrrrrrrrl", [head], rows,
        r"Trials are unmarked items with no transform and under three benign ones (images and audio), or the "
        r"H.264 CRF 18 encode of each unmarked clip (video); strength settings of one network share the detector, so each network appears once. "
        r"Sources: unmarked source items with at least one false match or detection, over all sources (an audio "
        r"source is one clip rendered at three sample rates). 95\% upper: upper endpoint of the two-sided exact "
        r"Clopper-Pearson 95\% interval with the source as the trial unit. Random payloads: false matches over "
        r"comparisons, using one fixed list of 50 payloads (200 for video) for every item; max: the largest number "
        r"of matching bits in any own- or random-payload comparison, to be read against $k$. Analytic: an "
        r"expected-payload test with $P_{\mathrm{FM}}\leq10^{-6}$ exists and no unmarked trial reached $k$. "
        r"Observed: no such test exists at this payload size, and the evidence is the blind detector's. None ($n$ bits): "
        r"even an exact match of all $n$ bits has probability $2^{-n}>10^{-6}$.",
        wide=True, size=r"\scriptsize", colsep="3.4pt")


def exact_table(data) -> str:
    """Payload recovery after decoding (and BCH correction, for TrustMark), next to bit accuracy."""
    rows = []
    for modality in ("image", "audio", "video"):
        gate = A.GATE_ATTACK[modality]
        for c in sorted(data[modality], key=lambda c: (-c.scores.get("combined", -1), c.label)):
            if not c.exact:
                continue
            ecc = r"BCH\_5" if c.family == "TrustMark" else "none"
            exact_all = [v for a, v in c.exact.items() if a != "none"]
            rows.append(" & ".join([
                modality.capitalize(), tex_escape(c.label) + (r"\textsuperscript{\dag}" if not c.open_licence else ""),
                str(c.nbits), ecc, fmt(c.robustness.get(gate), 3),
                fmt(c.keymatch.get(gate), 3) if c.keymatch else "--", fmt(c.exact.get(gate), 3),
                fmt(float(np.mean(exact_all)) if exact_all else None, 3),
            ]) + r" \\")
        rows.append(r"\midrule")
    rows.pop()
    return table_env(
        r"Exact payload recovery. Bit acc.: share of payload bits recovered at the gate transformation (JPEG 75, MP3 "
        r"128~kb/s, H.264 CRF 23). Exact: share of items whose decoded payload equals the embedded one, at the gate and "
        r"averaged over all transformations. For TrustMark exact recovery is after BCH decoding; no other method has "
        r"an error-correcting layer, so exact recovery requires every bit. $\geq k$: share of items whose decoded bits "
        r"match the embedded payload in at least $k$ of $n$ positions, the expected-payload decision of Table~\ref{tab:fpr} "
        r"(-- where no such $k$ exists).",
        "tab:exact", "llrlrrrr", [r"Modality & Configuration & Bits & ECC & Bit acc. & $\geq k$ (gate) & Exact (gate) & Exact (mean)"], rows,
        r"Bits: transmitted bits. TrustMark's 100-bit BCH\_5 codeword carries a 61-bit payload; bit accuracy and $\geq k$ "
        r"use the 100 transmitted bits, and exact recovery compares the 61 decoded payload bits. Perth carries "
        r"no payload and is omitted. \dag\ Not openly licensed.", wide=True)


def cost_table(data) -> str:
    rows = []
    for modality, unit in (("image", "MP"), ("audio", "s audio"), ("video", "MP")):
        cfgs = data[modality]
        nets: dict[str, list[A.Config]] = {}
        for c in cfgs:
            nets.setdefault(c.network, []).append(c)
        qual = {n: [c for c in m if c.qualifies] for n, m in nets.items()}
        best_speed = min((m[0].gpu_ms for n, m in nets.items() if qual[n]), default=None)
        best_cost = min((m[0].cost for n, m in nets.items() if qual[n]), default=None)
        comb_net = A.winners(cfgs)["combined"].network if A.winners(cfgs)["combined"] else None
        for n, m in sorted(nets.items(), key=lambda kv: kv[1][0].cost):
            c = m[0]
            marks = ""
            if qual[n] and abs(c.cost - best_cost) < 1e-12:
                marks += COST_MARK
            if qual[n] and abs(c.gpu_ms - best_speed) < 1e-9:
                marks += SPEED_MARK
            if n == comb_net:
                marks += BEST_MARK
            name = n if c.family in ("TrustMark", "AudioSeal") else A.FAMILY_LABEL[c.family]
            if not c.open_licence:
                name += r"\textsuperscript{\dag}"
            cpu = fmt_sig(c.cpu_ms, 3) + (r"\textsuperscript{i}" if c.cpu_imputed else "") if c.cpu_ms else "--"
            rows.append(" & ".join([
                modality.capitalize(), tex_escape(name) + marks, str(len(m)), "yes" if qual[n] else "no",
                fmt_sig(c.gpu_ms, 3), cpu, fmt_sig(c.cost_gpu, 2), fmt_sig(c.cost_cpu, 2) if c.cost_cpu else "--",
                c.cost_device,
            ]) + r" \\")
        rows.append(r"\midrule")
    rows.pop()
    header = ["Modality", "Network", "Configs", "Any Q", "GPU (ms/unit)", "CPU (ms/unit)", r"GPU \$", r"CPU \$",
              "Cheaper"]
    return table_env(
        r"Embedding cost and speed per network. Unit: one megapixel (image, video) or one second of audio. Cost "
        r"in USD per 1{,}000 units for image and video, and per 1{,}000 hours for audio, at on-demand prices "
        r"(g5.xlarge \$" + f"{A.PRICE_GPU_USD_H:.3f}" + r"/h, m6a.xlarge \$" + f"{A.PRICE_CPU_USD_H:.4f}" +
        r"/h, us-east-1, effective " + A.PRICE_DATE + r").",
        "tab:cost", "llrlrrrrl", [" & ".join(header)], rows,
        r"Latency is the median over the network's strength settings. \textsuperscript{i}~CPU latency measured "
        r"on another strength of the same network. Costs exclude I/O, storage, model loading and idle capacity. "
        + MARK_NOTE.split(" Bold")[0] + ".")


def stability_table() -> str:
    st = A.stability()
    files = sorted(k for k in st if not k.startswith("_"))
    fh = st["_framehash"]["files"]
    rows = []
    for f in files:
        e = st[f]
        yuv = fh.get(f, {}).get("yuv420p_equal")
        rgb = fh.get(f, {}).get("rgb24_equal")
        if f.endswith(".jpg"):
            label = "TrustMark-Q, image after JPEG 75"
        else:
            clip, rest = f.split("_5s_sw")
            s, crf = rest.replace(".mp4", "").split("_crf")
            label = f"{clip.replace('_', ' ')}, strength {s}, CRF {crf}"
        rows.append(" & ".join([
            label, fmt_sig(e.get("same_host_max_dlogit"), 2) if e.get("same_host_max_dlogit") is not None else "--",
            fmt(e["cross_host_max_dlogit"], 3), str(e["cross_host_bit_flips"]), str(e["distinct_bitstrings"]),
            "--" if yuv is None else ("yes" if yuv else "no"), "--" if rgb is None else ("yes" if rgb else "no"),
        ]) + r" \\")
    header = ["Decoded file", r"Same host", r"x86-64 vs arm64", "Bit flips", "Bit strings", "YUV equal", "RGB equal"]
    sub = [" ", r"max $|\Delta\ell|$", r"max $|\Delta\ell|$", " ", "distinct", " ", " "]
    head = " & ".join(header) + r" \\" + "\n" + " & ".join(sub)
    return table_env(
        r"Cross-device determinism of watermark decoding. $\Delta\ell$: difference in detector logits between "
        r"decodes of the same file.",
        "tab:stability", "lrrrrcc", [head], rows,
        r"Same host: CPU (1 and 4 threads), CUDA or MPS on one machine. Across hosts: AMD EPYC (x86-64, Linux) vs "
        r"Apple M-series (arm64, macOS). YUV/RGB equal: whether decoded frames are byte-identical across hosts "
        r"before and after the YUV to RGB conversion.", wide=True)


def ci_table(data) -> str:
    rows = []
    spec = {"image": [("psnr", 1), ("lpips", 4), ("flip", 4)], "audio": [("si_snr", 1), ("pesq", 3), ("stoi", 4)],
            "video": [("psnr", 1), ("lpips", 4), ("vmaf", 1)]}
    for modality, metrics in spec.items():
        for c in data[modality]:
            cells = []
            for m, nd in metrics:
                lo, hi = c.quality_ci.get(m, (float("nan"), float("nan")))
                key = "pesq_speech" if (modality, m) == ("audio", "pesq") else m
                n_m = int(np.isfinite(c.per_item.get(key, np.array([]))).sum())
                note = rf" \tiny($n$={n_m})" if n_m and n_m != len(c.items) else ""
                cells.append(rf"{fmt(c.quality[m], nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]{note}")
            n_src = len(set(c.groups)) if c.groups else len(c.items)
            n_cell = str(len(c.items)) if n_src == len(c.items) else f"{len(c.items)} ({n_src})"
            rows.append(" & ".join([modality.capitalize(), tex_escape(c.label), n_cell] + cells) + r" \\")
        rows.append(r"\midrule")
    rows.pop()
    return table_env(
        r"Quality means with bootstrap 95\% confidence intervals (10{,}000 resamples of sources). Columns: image PSNR "
        r"(dB), LPIPS, FLIP; audio SI-SNR (dB), PESQ on speech, STOI; video PSNR (dB), LPIPS, VMAF.",
        "tab:ci", "llrlll", [r"Modality & Configuration & $n$ & Metric 1 & Metric 2 & Metric 3"], rows,
        r"Percentile intervals. $n$: items (independent sources in parentheses). Audio sources are the 46 clips, each "
        r"rendered at three sample rates; the three renderings of a clip are resampled together. A metric computed on "
        r"a subset shows its own $n$ (FLIP on 60 images, PESQ on 120 speech inputs).", wide=True)


def paired_tests(data) -> tuple[str, dict[str, str]]:
    pairs = [
        ("image", "PixelSeal@0.15", "trustmark-P@0.8", "flip"),
        ("image", "PixelSeal@0.15", "VideoSeal-1.0@default", "flip"),
        ("image", "PixelSeal@0.15", "trustmark-Q@1.2", "flip"),
        ("audio", "AudioSeal-base@1", "AudioSeal-streaming@0.5", "pesq_speech"),
        ("video", "PixelSeal@0.4", "VideoSeal-1.0@0.4", "vmaf"),
    ]
    res = []
    for modality, a, b, metric in pairs:
        ca = next(c for c in data[modality] if c.name == a)
        cb = next(c for c in data[modality] if c.name == b)
        d, p, n = A.paired_wilcoxon(ca, cb, metric)
        res.append((modality, ca, cb, metric, d, p, n))
    # Holm-Bonferroni over the family of planned comparisons.
    m = len(res)
    order = sorted(range(m), key=lambda i: res[i][5])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * res[i][5]))
        adj[i] = running
    rows, vals = [], {}
    for i, (modality, ca, cb, metric, d, p, n) in enumerate(res):
        ptxt = r"$<$0.001" if adj[i] < 1e-3 else fmt(adj[i], 3)
        rows.append(" & ".join([modality.capitalize(), tex_escape(ca.label), tex_escape(cb.label),
                                metric.upper() if not metric.startswith("pesq") else "PESQ (speech)", str(n), fmt(d, 4), ptxt]) + r" \\")
        vals[f"test/{modality}/{ca.name}/{cb.name}/{metric}/p"] = ptxt
        vals[f"test/{modality}/{ca.name}/{cb.name}/{metric}/d"] = fmt(d, 4)
        vals[f"test/{modality}/{ca.name}/{cb.name}/{metric}/n"] = str(n)
    tab = table_env(
        r"Planned paired comparisons (two-sided Wilcoxon signed-rank on per-source values, Holm-adjusted over the "
        r"five tests). Audio inputs of one clip are averaged first, so $n$ counts clips; audio PESQ is on speech.",
        "tab:tests", "lllrrrr", [r"Modality & A & B & Metric & $n$ & median(A$-$B) & $p_{\mathrm{Holm}}$"], rows,
        r"Negative FLIP differences favour A; positive PESQ and VMAF differences favour A.", wide=True)
    return tab, vals


def values(data) -> dict[str, str]:
    v: dict[str, str] = {}
    for modality, cfgs in data.items():
        v[f"{modality}/n_configs"] = str(len(cfgs))
        v[f"{modality}/n_qualify"] = str(sum(c.qualifies for c in cfgs))
        v[f"{modality}/n_items"] = str(max(c.n for c in cfgs))
        w = A.winners(cfgs)
        for k, c in w.items():
            if c is not None:
                v[f"{modality}/win/{k}"] = tex_escape(c.label)
        for c in cfgs:
            key = f"{modality}/{c.name}"
            for m, x in c.quality.items():
                if isinstance(x, (int, float)):
                    nd = 4 if m in ("ssim", "ms_ssim", "lpips", "stoi") else (3 if m in ("flip", "abs_dlufs") else (2 if m in ("pesq", "cvvdp") else 1))
                    v[f"{key}/{m}"] = fmt(x, nd)
                    lo, hi = c.quality_ci.get(m, (None, None))
                    if lo is not None and math.isfinite(lo):
                        v[f"{key}/{m}/lo"] = fmt(lo, nd)
                        v[f"{key}/{m}/hi"] = fmt(hi, nd)
            v[f"{key}/gate"] = fmt(c.gate_pooled, 3)
            v[f"{key}/worst"] = fmt(c.gate_worst, 3)
            v[f"{key}/worstset"] = tex_escape(c.gate_worst_set)
            v[f"{key}/gpu"] = fmt_sig(c.gpu_ms, 3)
            v[f"{key}/cpu"] = fmt_sig(c.cpu_ms, 3)
            v[f"{key}/cost"] = fmt_sig(c.cost, 2)
            v[f"{key}/costgpu"] = fmt_sig(c.cost_gpu, 2)
            v[f"{key}/costcpu"] = fmt_sig(c.cost_cpu, 2)
            if c.blind_fpr is not None:
                v[f"{key}/blind"] = fmt(c.blind_fpr, 4)
                v[f"{key}/blindpct"] = fmt(c.blind_fpr, 1, pct=True)
            if c.blind_fpr_ci:
                v[f"{key}/blindhi"] = fmt(c.blind_fpr_ci[1], 4)
            if c.key_trials:
                v[f"{key}/keytrials"] = f"{c.key_trials:,}".replace(",", "{,}")
                v[f"{key}/keyupper"] = fmt_sig(c.key_fpr_upper, 2)
            for a, x in c.robustness.items():
                if isinstance(x, (int, float)):
                    v[f"{key}/rob/{a}"] = fmt(x, 3)
            if c.scores:
                v[f"{key}/comb"] = fmt(c.scores["combined"], 2)
                v[f"{key}/combq"] = fmt(c.scores["combined_q"], 2)
            if modality == "video":
                v[f"{key}/fps"] = fmt(1000.0 / c.gpu_ms / A.VIDEO_MP_PER_FRAME, 1)
                # USD per hour of 1080p video at 30 fps: megapixels per hour / 1,000 x cost per 1,000 MP.
                mp_per_hour = A.VIDEO_MP_PER_FRAME * A.VIDEO_FPS * 3600
                v[f"{key}/costhour"] = fmt(c.cost * mp_per_hour / 1000.0, 2)
    # TrustMark blind FPR range, pooled and on the HDR subset.
    tm = [c for c in data["image"] if c.family == "TrustMark" and c.blind_fpr is not None]
    # Per image (any of its four decodes fired), the unit of Table tab:fpr and Fig. fig:fpr.
    tm_src = [A.fpr_by_source(c) for c in tm]
    v["tm/blind/min"] = fmt(min(s["blind_src_fm"] / s["blind_src_n"] for s in tm_src), 1, pct=True)
    v["tm/blind/max"] = fmt(max(s["blind_src_fm"] / s["blind_src_n"] for s in tm_src), 1, pct=True)
    hdr = []
    for c in tm:
        raw = A._raw("image", c.name)
        recs = raw["records"]
        fired: dict[int, bool] = {}
        for x in raw["fpr"]:
            if recs[x["item"]]["set"] == "hdr16":
                fired[x["item"]] = fired.get(x["item"], False) or bool(x["native_detect"])
        hdr.append((sum(fired.values()), len(fired)))
    k_hdr, n_hdr = max(hdr)
    v["tm/blind/hdrmax"] = fmt(k_hdr / n_hdr, 1, pct=True)
    v["tm/blind/hdrk"], v["tm/blind/hdrn"] = str(k_hdr), str(n_hdr)
    v["tm/keytrials"] = next(v[k] for k in v if k.endswith("/keytrials") and "trustmark" in k)
    # False-positive evidence with the unmarked source as the trial unit (Table tab:fpr).
    for modality in ("image", "audio", "video"):
        for c in data[modality]:
            src = A.fpr_by_source(c) or {}
            key = f"fpr/{modality}/{tex_escape(c.network)}"
            if "own_src_n" in src:
                v[f"{key}/own_src_n"] = str(src["own_src_n"])
                v[f"{key}/own_src_upper"] = fmt(src["own_src_upper"], 1, pct=True)
            if "blind_src_n" in src:
                v[f"{key}/blind_src_fm"] = str(src["blind_src_fm"])
                v[f"{key}/blind_src_n"] = str(src["blind_src_n"])
                v[f"{key}/blind_src_upper"] = fmt(src["blind_src_ci"][1], 1, pct=True)
    # Largest AudioSeal detection score (fraction of frames detected) on unmarked audio, against the
    # 0.8 blind threshold here and the 0.5 used with exact-key matching in the fingerprinting track.
    v["audioseal/unmarked_max_score"] = fmt(max(
        t["native_score"] for c in data["audio"] if c.family == "AudioSeal"
        for t in A._raw("audio", c.name)["fpr"]), 2)
    # Sources with no false positive needed before the two-sided 95 % upper bound reaches 1e-3.
    n = 1
    while A.clopper_pearson(0, n)[1] > 1e-3:
        n += 1
    v["fpr/sources_for_1e-3"] = f"{n:,}".replace(",", "{,}")
    st = A.stability()
    files = [k for k in st if not k.startswith("_")]
    v["stab/files"] = str(len(files))
    v["stab/maxcross"] = fmt(max(st[f]["cross_host_max_dlogit"] for f in files), 2)
    vid = [f for f in files if f.endswith(".mp4")]
    v["stab/maxsame"] = fmt_sig(max(st[f]["same_host_max_dlogit"] for f in vid), 1)
    v["stab/videofiles"] = str(len(vid))
    img_f = [f for f in files if not f.endswith(".mp4")]
    if img_f:  # the TrustMark image decode differs even between backends of one host
        v["stab/tm/maxsame"] = fmt(max(st[f]["same_host_max_dlogit"] for f in img_f), 3)
        v["stab/tm/maxcross"] = fmt(max(st[f]["cross_host_max_dlogit"] for f in img_f), 3)
        cpu = [d["max_abs_logit_diff_vs_first"] for f in img_f for d in st[f]["decodes"] if d["backend"].startswith("cpu")]
        m, e = f"{max(cpu):.0e}".split("e")
        v["stab/tm/cpu_dlogit"] = rf"${m}\times10^{{{int(e)}}}$"
        v["stab/tm/flips"] = str(max(st[f]["cross_host_bit_flips"] for f in img_f))
    v["stab/maxflips"] = str(max(st[f]["cross_host_bit_flips"] for f in files))
    v["stab/nflipfiles"] = str(sum(1 for f in files if st[f]["cross_host_bit_flips"]))
    img = {c.name: c for c in data["image"]}
    ps, tq = img["PixelSeal@0.15"], img["trustmark-Q@1"]
    v["ratio/ps_tq/flip"] = fmt(ps.quality["flip"] / tq.quality["flip"], 2)
    v["ratio/ps_tq/lpips"] = fmt(tq.quality["lpips"] / ps.quality["lpips"], 1)
    v["ps/lpips"], v["tq/lpips"] = fmt(ps.quality["lpips"], 4), fmt(tq.quality["lpips"], 4)
    v["ps/flip"], v["tq/flip"] = fmt(ps.quality["flip"], 3), fmt(tq.quality["flip"], 3)
    aud = data["audio"]
    v["audio/n_sources"] = str(len({g for c in aud for g in c.groups}))
    v["audio/n_speech"] = str(max(int(np.isfinite(c.per_item["pesq_speech"]).sum()) for c in aud))
    v["audio/n_music"] = str(max(int(np.isfinite(c.per_item["pesq_music"]).sum()) for c in aud))
    for c in aud:
        v[f"audio/{c.name}/pesq_music"] = fmt(c.quality["pesq_music"], 2)
        v[f"audio/{c.name}/pesq_pooled"] = fmt(c.quality["pesq_pooled"], 2)
    for mod in ("image", "audio", "video"):
        for c in data[mod]:
            g = A.GATE_ATTACK[mod]
            if c.exact:
                v[f"{mod}/{c.name}/exact_gate"] = fmt(c.exact.get(g), 3)
            if c.keymatch:
                v[f"{mod}/{c.name}/keymatch_gate"] = fmt(c.keymatch.get(g), 3)
    v["ratio/ps_tq/gpu"] = fmt(tq.gpu_ms / ps.gpu_ms, 1)
    v["count/image"] = str(len(data["image"]))
    v["count/audio"] = str(len(data["audio"]))
    v["count/video"] = str(len(data["video"]))
    fams = {c.family for m in data.values() for c in m}
    v["count/families"] = str(len(fams))
    n_tr = {"image": 14, "audio": 10, "video": 8}
    for m in n_tr:
        got = len([a for a in data[m][0].robustness if a != "none"])
        assert got == n_tr[m], (m, got)
        v[f"count/transforms/{m}"] = str(got)
    v["count/transforms"] = str(sum(n_tr.values()))
    v["price/gpu"] = f"{A.PRICE_GPU_USD_H:.3f}"
    v["price/cpu"] = f"{A.PRICE_CPU_USD_H:.4f}"
    v["price/date"] = A.PRICE_DATE
    corpus = A.corpus()
    v["corpus/image"] = str(len(corpus["image"]))
    v["corpus/audio"] = str(len(corpus["audio"]))
    v["corpus/video"] = str(len(corpus["video"]))
    return v


FOLLOWUP = A.BENCH / "results-followup"


def rgb_grid_values() -> dict[str, str]:
    """2x2 follow-up (scripts/stability_rgb.py): each host's RGB conversion of the stability
    fixtures, decoded on each host. Same input on both hosts isolates the model; both inputs
    on one host isolate the colour conversion."""
    d = FOLLOWUP / "stability"
    files = sorted(d.glob("rgb-decode-*.json")) if d.exists() else []
    if len(files) < 2:
        return {}
    runs = [json.loads(f.read_text()) for f in files]
    inputs = sorted(runs[0]["inputs"])
    clips = sorted(runs[0]["inputs"][inputs[0]])
    L = lambda D, i, c, b: np.array(D["inputs"][i][c][b]["avg_logits"])
    model = max(float(np.abs(L(runs[0], i, c, "cpu1") - L(runs[1], i, c, "cpu1")).max()) for i in inputs for c in clips)
    conv, flips, flip_pairs = 0.0, 0, 0
    for D in runs:
        for c in clips:
            a, b = L(D, inputs[0], c, "cpu1"), L(D, inputs[1], c, "cpu1")
            conv = max(conv, float(np.abs(a - b).max()))
            n = int(((a > 0) != (b > 0)).sum())
            flips, flip_pairs = max(flips, n), flip_pairs + (n > 0)
    backend = 0.0
    for D in runs:
        for i in inputs:
            for c in clips:
                bes = [k for k in D["inputs"][i][c] if k != "rgb_sha256"]
                backend = max(backend, max(float(np.abs(L(D, i, c, k) - L(D, i, c, bes[0])).max()) for k in bes))
    fh = {f.stem: json.loads(f.read_text()) for f in d.glob("framehash-*-rerun.json")}
    m, e = f"{model:.0e}".split("e")
    v = {"rgb/model_dlogit": rf"${m}\times10^{{{int(e)}}}$", "rgb/conv_dlogit": fmt(conv, 2), "rgb/backend_dlogit": fmt_sig(backend, 1),
         "rgb/maxflips": str(flips), "rgb/flippairs": str(flip_pairs), "rgb/nclips": str(len(clips)),
         "rgb/nframes": str(runs[0]["n_frames"])}
    if len(fh) == 2:
        a, b = fh.values()
        v["rgb/av"] = a["av"]
        v["rgb/swscale"] = ".".join(map(str, a["ffmpeg"]))
        v["rgb/same_libs"] = "yes" if (a["av"], a["ffmpeg"]) == (b["av"], b["ffmpeg"]) else "no"
        v["rgb/rgb_equal"] = str(sum(a["files"][k]["rgb24_16f"] == b["files"][k]["rgb24_16f"] for k in a["files"]))
        v["rgb/yuv_equal"] = str(sum(a["files"][k]["yuv420p_16f"] == b["files"][k]["yuv420p_16f"] for k in a["files"]))
    return v


NOALIGN_CODECS = ["mp3_320", "mp3_128", "mp3_64", "aac_256", "aac_128", "aac_64"]


def noalign_table(data) -> tuple[str, dict[str, str]]:
    """Follow-up (scripts/audio_noalign.py): codec round trips decoded with the benchmark's
    alignment to the original and without it, re-run on one host; the aligned column is a
    reproducibility check against the published robustness values (Table 3)."""
    d = FOLLOWUP / "audio_noalign"
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        return "", {}
    cfg = {c.name: c for c in data["audio"]}
    rows, vals, host = [], {}, ""
    for f in files:
        r = json.loads(f.read_text())
        name = r["meta"]["name"]
        c = cfg.get(name)
        if c is None or not c.nbits:
            continue
        host = r["host"].get("cpu_model", "")
        recs = r["records"]
        cells = [tex_escape(c.label)]
        for codec in ("mp3_128", "aac_64"):
            al = float(np.mean([x["codecs"][codec]["aligned"]["bit_acc"] for x in recs]))
            un = float(np.mean([x["codecs"][codec]["unaligned"]["bit_acc"] for x in recs]))
            cells += [fmt(c.robustness.get(codec), 3), fmt(al, 3), fmt(un, 3)]
            vals[f"noalign/{name}/{codec}/aligned"] = fmt(al, 3)
            vals[f"noalign/{name}/{codec}/unaligned"] = fmt(un, 3)
        allc = [x["codecs"][k] for x in recs for k in NOALIGN_CODECS]
        al_all = float(np.mean([e["aligned"]["bit_acc"] for e in allc]))
        un_all = float(np.mean([e["unaligned"]["bit_acc"] for e in allc]))
        ex_al = float(np.mean([e["aligned"]["matches"] == c.nbits for e in allc]))
        ex_un = float(np.mean([e["unaligned"]["matches"] == c.nbits for e in allc]))
        cells += [fmt(al_all, 3), fmt(un_all, 3), fmt(ex_al, 3), fmt(ex_un, 3)]
        vals[f"noalign/{name}/all/aligned"], vals[f"noalign/{name}/all/unaligned"] = fmt(al_all, 3), fmt(un_all, 3)
        vals[f"noalign/{name}/all/exact_aligned"], vals[f"noalign/{name}/all/exact_unaligned"] = fmt(ex_al, 3), fmt(ex_un, 3)
        vals[f"noalign/{name}/n"] = str(len(recs))
        rows.append(" & ".join(cells) + r" \\")
    vals["noalign/host"] = tex_escape(host)
    tab = table_env(
        r"Audio decoding after codec round trips with and without alignment to the original. Pub.: the published "
        r"bit accuracy (Table~\ref{tab:audio}, aligned, benchmark host). Al./Un.: re-run on one host, decoded after "
        r"cross-correlation alignment to the original (the benchmark protocol) or from the codec output cut to the "
        r"original length with no offset search. All: the six MP3 and AAC settings pooled; exact: all payload bits "
        r"correct.",
        "tab:noalign", "lrrrrrrrrrr",
        [r" & \multicolumn{3}{c}{MP3 128k bit acc.} & \multicolumn{3}{c}{AAC 64k bit acc.} & "
         r"\multicolumn{2}{c}{All codecs bit acc.} & \multicolumn{2}{c}{All codecs exact} \\"
         "\n" r"Configuration & Pub. & Al. & Un. & Pub. & Al. & Un. & Al. & Un. & Al. & Un."], rows,
        rf"Re-run on {tex_escape(host)} with the speech inputs byte-identical to the benchmark corpus; the music "
        r"inputs were re-decoded from the same source files by a different FFmpeg build (same content, different "
        r"bytes). Perth carries no payload and is omitted.", wide=True)
    return tab, vals


def numbers_tex(vals: dict[str, str]) -> str:
    lines = [
        "% Generated by scripts/build_numbers.py from results.json and results-gpu/. Do not edit.",
        r"\makeatletter",
        r"\newcommand{\val}[1]{\ifcsname val@#1\endcsname\csname val@#1\endcsname"
        r"\else\PackageError{numbers}{Undefined value #1}{Run build_numbers.py}\fi}",
    ]
    for k in sorted(vals):
        lines.append(rf"\expandafter\def\csname val@{k}\endcsname{{{vals[k]}}}")
    lines.append(r"\makeatother")
    return "\n".join(lines) + "\n"


def build() -> dict[Path, str]:
    data = A.load()
    tests_tab, test_vals = paired_tests(data)
    noalign_tab, noalign_vals = noalign_table(data)
    vals = values(data) | test_vals | rgb_grid_values() | noalign_vals
    return {
        GEN / "numbers.tex": numbers_tex(vals),
        TAB / "image.tex": image_table(data["image"]),
        TAB / "audio.tex": audio_table(data["audio"]),
        TAB / "video.tex": video_table(data["video"]),
        TAB / "fpr.tex": fpr_table(data),
        TAB / "exact.tex": exact_table(data),
        TAB / "noalign.tex": noalign_tab,
        TAB / "cost.tex": cost_table(data),
        TAB / "stability.tex": stability_table(),
        TAB / "ci.tex": ci_table(data),
        TAB / "tests.tex": tests_tab,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = build()
    if args.check:
        stale = [p for p, s in out.items() if not p.exists() or p.read_text() != s]
        for p in stale:
            print(f"stale: {p.relative_to(A.PAPER)}")
        return 1 if stale else 0
    TAB.mkdir(parents=True, exist_ok=True)
    for p, s in out.items():
        p.write_text(s)
    print(f"wrote {len(out)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
