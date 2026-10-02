"""Write every table and every number the manuscript cites, from analysis.load().

    uv run --with numpy --with scipy python build_numbers.py          # write generated/
    uv run --with numpy --with scipy python build_numbers.py --check  # fail if generated/ is stale

The manuscript cites values through \\val{key}; `generated/numbers.tex` defines each key
and \\val raises a LaTeX error for an undefined one, so a typo cannot silently print
nothing. Tables are written to generated/tables/*.tex and \\input by the sections.
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


G5_USD_PER_H, G5_VCPUS = 1.212, 8  # measurement host: AWS g5.2xlarge on-demand, us-east-1


def fmt(x, nd=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt_sig(x, sig=2) -> str:
    if x is None or not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    d = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    return f"{x:.{d}f}"


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def table_env(caption, label, cols, header, rows, note="", wide=False, size=r"\footnotesize"):
    env = "table*" if wide else "table"
    body = "\n".join(r if r.startswith("\\addlinespace") else r + r" \\" for r in rows)
    note_tex = rf"\par\vspace{{2pt}}\parbox{{\linewidth}}{{\raggedright\scriptsize {note}}}" if note else ""
    return (f"\\begin{{{env}}}[!t]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{cols}}}\n\\toprule\n" + "\n".join(h + r" \\" for h in header)
            + f"\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n{note_tex}\n\\end{{{env}}}\n")


def mlabel(m: A.Method) -> str:
    return tex(m.label) + (r"\reftier" if m.tier == "R" else "")


def cluster_ci(m: A.Method, key: str) -> list[float] | None:
    """Source-level bootstrap interval (cluster_ci.py) for a pooled TPR@t, FPRq@t or R@1."""
    c = A.load()["cluster"].get("methods", {}).get(f"{m.modality}/{m.name}", {})
    if key == "R@1":
        return c.get("r1_ci")
    kind, _, target = key.partition("@")
    return c.get(target, {}).get({"TPR": "tpr_ci", "FPRq": "fpr_ci"}.get(kind, ""))


def with_ci(m: A.Method, key: str, attack: str = "_all") -> str:
    """Value with its 95% interval. Pooled values use the source-level bootstrap; the
    per-query Clopper-Pearson interval in the record is not used, because queries of one
    source are not independent."""
    r = m.rec["per_attack"].get(attack, {})
    v = r.get(key)
    if v is None or not np.isfinite(v):
        return "--"
    ci = cluster_ci(m, key) if attack == "_all" else None
    if ci:
        return f"{fmt(v, 3)} \\tiny[{fmt(ci[0], 2)}, {fmt(ci[1], 2)}]"
    return fmt(v, 3)


# ------------------------------------------------------------------ tables
def inventory_table(data) -> str:
    rows, prev = [], None
    for mod in ("image", "audio", "video"):
        for m in data[mod]:
            meta = m.rec["method"]
            ms = m.ms_per_item()
            dev = "GPU" if m.device.startswith("cuda") else "CPU"
            bits = m.bits
            rows.append(" & ".join([mod.capitalize() if mod != prev else "", mlabel(m), A.CLASS_LABEL[m.cls],
                                    tex(meta.get("code_licence", "")[:38]), tex(meta.get("weights_licence", "")[:44]),
                                    fmt_sig(bits, 3) if math.isfinite(bits) else "--",
                                    f"{fmt_sig(ms, 2)} {dev}" if math.isfinite(ms) else "--"]))
            prev = mod
        rows.append(r"\addlinespace")
    return table_env(
        r"Methods, licences, stored size and extraction time. Bits: stored bits per registered asset (float "
        r"descriptors count 32 bits per dimension; time-based systems report the mean over the registered items). "
        r"Time: per image (1024 px), per second of audio or per 5\,s clip.",
        "tab:methods", "lllp{3.3cm}p{3.6cm}rr",
        [r"Modality & Method & Class & Code licence & Weights / data & Bits & Time (ms)"], rows[:-1],
        r"\reftier~Reference tier: weights without a licence, trained on non-commercial data, or copyleft code; "
        r"measured for comparison, not product-eligible.", wide=True, size=r"\scriptsize")


def image_table(data) -> str:
    rows = []
    for m in data["image"]:
        p = m.pooled
        hard = m.rec.get("hard_negative", {}).get(A.OP_Q, {})
        disc = m.rec.get("disc21", {})
        rows.append(" & ".join([mlabel(m), fmt(p["R@1"]), fmt(p["muAP"]), fmt(p.get(f"TPR@{A.OP_Q}")),
                                with_ci(m, f"TPR@{A.OP}"), fmt(hard.get("rate"), 3), fmt(disc.get("muAP"))]))
    return table_env(
        r"Image retrieval over all attacks (ABO; registry of registered originals and distractors). "
        r"TPR counts a query only if its top-1 is the true asset and the score passes a threshold fixed on "
        r"held-out calibration negatives: query-level FPR 1\% at this registry size, or pair-level FPR $10^{-7}$ "
        r"(95\% interval from a bootstrap over sources, which redraws the calibration negatives and so the threshold)."
        r" Hard: rate at which another photograph of a registered product binds "
        r"to that product at the 1\% threshold.",
        "tab:image", "lcccccc",
        [r"Method & R@1 & $\mu$AP & TPR$_{q,1\%}$ & TPR$_{p,10^{-7}}$ & Hard FB & DISC21 $\mu$AP"], rows, wide=True)


def timebased_table(data, mod, cols) -> str:
    """Audio and video: the operating points their calibration sets resolve. The query-level 1%
    threshold is reported next to the false-match rate it actually produced on held-out
    negatives, and the strictest resolved pair-level target next to it."""
    strict = {"audio": "pair@1e-06", "video": "pair@0.0001"}[mod]
    strict_tex = {"audio": "10^{-6}", "video": "10^{-4}"}[mod]
    rows = []
    for m in data[mod]:
        p = m.pooled
        extra = []
        for key, label in cols:
            if key.startswith("music:") or key.startswith("speech:"):
                sub, a = key.split(":")
                extra.append(fmt(m.rec.get(f"per_attack_{sub}", {}).get(a, {}).get("R@1")))
            else:
                extra.append(fmt(m.at(key)))
        rows.append(" & ".join([mlabel(m), fmt(p["R@1"]), fmt(p["muAP"]), with_ci(m, f"TPR@{A.OP_Q}"),
                                with_ci(m, f"FPRq@{A.OP_Q}"), with_ci(m, f"TPR@{strict}"), *extra]))
    head = (r"Method & R@1 & $\mu$AP & TPR$_{q,1\%}$ & FPR$_{q}$ (test) & TPR$_{p," + strict_tex + r"}$ & "
            + " & ".join(l for _, l in cols))
    what = "Audio" if mod == "audio" else "Video"
    n_cal = max(m.rec["n_cal_neg"] for m in data[mod])
    n_ref = max(m.rec["n_ref"] for m in data[mod])
    return table_env(
        rf"{what} retrieval over all attacks, and Recall@1 under selected attacks. TPR$_{{q,1\%}}$: detection at the "
        rf"threshold set for a 1\% query-level false-match rate on the calibration negatives; FPR$_q$ (test): the "
        rf"rate that threshold actually gave on the held-out negatives. TPR$_{{p,{strict_tex}}}$: the strictest "
        rf"pair-level target that {n_cal} calibration queries against {n_ref} references resolve; stricter targets "
        rf"are not reported. Intervals: 95\% bootstrap over sources.",
        f"tab:{mod}", "l" + "c" * (5 + len(cols)), [head], rows, wide=True, size=r"\scriptsize")


def paired_table(data) -> tuple[str, dict]:
    """Planned image comparisons, tested at the level of sources (cluster_ci.py): a sign-flip
    test on each source's difference in successes, Holm-adjusted, with a bootstrap interval of
    the TPR difference. The per-query McNemar result (_paired.json) is kept in the values for
    reference only."""
    rows, vals = [], {}
    mc = {(r["a"], r["b"]): r for r in A._j(A.RAW / "image" / "_paired.json").get("rows", [])}
    for r in data["cluster"].get("paired", []):
        la, lb = A.LABEL.get(r["a"], r["a"]), A.LABEL.get(r["b"], r["b"])
        ptxt = fmt_sig(r["p_holm"], 2) if r["p_holm"] >= 1e-3 else r"$<10^{-3}$"
        ci = r.get("diff_ci") or [float("nan")] * 2
        rows.append(" & ".join([tex(la), tex(lb), fmt(r["tpr_a"]), fmt(r["tpr_b"]),
                                f"{fmt(r['diff'], 3)} \\tiny[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}]",
                                fmt_int(r["n_sources"]), ptxt]))
        vals[f"paired/{r['a']}/{r['b']}/p"] = ptxt
        if (r["a"], r["b"]) in mc:
            m = mc[(r["a"], r["b"])]
            vals[f"paired/{r['a']}/{r['b']}/p_mcnemar"] = fmt_sig(m["p_holm"], 2) if m["p_holm"] >= 1e-4 else r"$<10^{-4}$"
    tab = table_env(
        r"Planned paired comparisons on the same image queries at pair-level FPR $10^{-7}$. The unit is the registered "
        r"source with all its transformed queries: $p_{\mathrm{Holm}}$ from a sign-flip test on per-source "
        r"differences in detections (Holm-adjusted over the eight comparisons); the interval of the TPR difference is "
        r"a bootstrap over sources that also redraws the calibration negatives.", "tab:paired", "llcccrr",
        [r"A & B & TPR$_A$ & TPR$_B$ & A$-$B [95\% CI] & sources & $p_{\mathrm{Holm}}$"], rows)
    return tab, vals


SENS_METHODS = ["PDQ", "PDQ-dihedral", "pHash-256", "ISCC-Image-256", "DINOv2-S", "DINOv2-S-LSH256", "DINOv2-B",
                "OpenCLIP-B32", "SSCD-mixup", "ISC21-1st"]


def sensitivity_table(data) -> tuple[str, dict]:
    ms = A.by_name(data["image"])
    rows, vals = [], {}
    for n in SENS_METHODS:
        if n not in ms:
            continue
        cells = [mlabel(ms[n])]
        for rule, rec in (("any", ms[n].rec), ("two", data["image_sens"].get("two", {}).get(n)),
                          ("none", data["image_sens"].get("none", {}).get(n))):
            p = (rec or {}).get("per_attack", {}).get("_all", {})
            for key in (f"TPR@{A.OP_Q}", f"TPR@{A.OP}"):
                cells.append(fmt(p.get(key), 3))
                vals[f"sens/{rule}/{n}/{key}"] = fmt(p.get(key), 3)
        rows.append(" & ".join(cells))
    tab = table_env(
        r"Sensitivity of the image operating points to the treatment of catalogue duplicates. \emph{Any}: the rule "
        r"fixed in advance (a pair counts as a duplicate if PDQ, SSCD or DINOv2-B says so); \emph{two}: two of the "
        r"three must agree; \emph{none}: no cleanup, so recoloured or re-mounted versions of a registered design count "
        r"as false matches. R@1 changes by at most a few points; the calibrated TPR does not.",
        "tab:sens", "l" + "cc" * 3,
        [r" & \multicolumn{2}{c}{Any (primary)} & \multicolumn{2}{c}{Two of three} & \multicolumn{2}{c}{None}",
         r"Method & TPR$_{q,1\%}$ & TPR$_{p,10^{-7}}$ & TPR$_{q,1\%}$ & TPR$_{p,10^{-7}}$ & TPR$_{q,1\%}$ & TPR$_{p,10^{-7}}$"],
        rows)
    return tab, vals


def localization_values(data) -> tuple[str, dict]:
    v, rows = {}, []
    loc = data.get("localize", {})
    maps = ["pixel", "ssim", "lpips", "dino", "pdq", "fused"]
    label = {"pixel": "Pixel difference", "ssim": "SSIM", "lpips": "LPIPS", "dino": "DINOv2 patches",
             "pdq": "PDQ tiles", "fused": "Fused"}
    for al in ("sift", "lightglue"):
        o = loc.get(f"spatial_{al}", {}).get("oracle")
        if not o:
            continue
        v[f"loc/{al}/aligned"] = fmt(o["aligned_rate"], 1, pct=True)
        v[f"loc/{al}/n"] = fmt_int(o["n"])
        for m in maps:
            for k in ("auc", "f1", "best_f1", "iou"):
                if k in o[m]:
                    v[f"loc/{al}/{m}/{k}"] = fmt(o[m][k], 2)
            for dim, by in o[m].get("by", {}).items():
                for lev, r in by.items():
                    v[f"loc/{al}/{m}/{dim}/{lev}/f1"] = fmt(r["f1"], 2)
                    v[f"loc/{al}/{m}/{dim}/{lev}/auc"] = fmt(r["auc"], 2)
    for m in maps:
        cells = [label[m]]
        for al in ("sift", "lightglue"):
            o = loc.get(f"spatial_{al}", {}).get("oracle", {}).get(m, {})
            cells += [fmt(o.get("auc"), 3), fmt(o.get("f1"), 2), fmt(o.get("best_f1"), 2)]
        rows.append(" & ".join(cells))
    tab = table_env(
        r"Localization of partial edits against the registered original (oracle retrieval; evaluation half of the "
        r"sources). F1 at the threshold fixed for a 1\% pixel false-positive rate on unedited, post-processed "
        r"originals; best F1 chooses the threshold per map with hindsight.",
        "tab:loc", "lcccccc",
        [r" & \multicolumn{3}{c}{SIFT + RANSAC} & \multicolumn{3}{c}{DISK + LightGlue}",
         r"Map & AUC & F1 & best F1 & AUC & F1 & best F1"], rows)
    # End to end: the localization track assumes the correct original is known (oracle). Here an
    # edit whose original the fingerprint does not rank first counts as a failure (F1 = 0, not aligned).
    for al in ("sift", "lightglue"):
        per = loc.get(f"spatial_{al}", {}).get("per_edit", [])
        for n, r in data.get("localize_retrieval", {}).items():
            hit = r.get("per_edit_hit", {})
            rows_e = [e for e in per if e["key"] in hit]
            if not rows_e:
                continue
            for mp in ("pixel", "fused", "lpips"):
                f1 = np.array([e[mp]["f1"] if hit[e["key"]] else 0.0 for e in rows_e], float)
                v[f"loc/e2e/{al}/{n}/{mp}/f1"] = fmt(float(np.nanmean(f1)), 2)
            v[f"loc/e2e/{al}/{n}/retrieved_aligned"] = fmt(float(np.mean([hit[e["key"]] and e["aligned"] for e in rows_e])), 1, pct=True)
            v[f"loc/e2e/{al}/{n}/retrieved"] = fmt(float(np.mean([hit[e["key"]] for e in rows_e])), 1, pct=True)
            v[f"loc/e2e/{al}/n"] = fmt_int(len(rows_e))
    ta = loc.get("temporal_audio", {}).get("summary", {})
    for sysn, d in ta.items():
        for a, r in d.items():
            v[f"tloc/audio/{sysn}/{a}/median_ms"] = fmt(1000 * r["median_abs_err_s"], 0)
            v[f"tloc/audio/{sysn}/{a}/within05"] = fmt(r["within_0.5s"], 1, pct=True)
    tv = loc.get("temporal_video", {}).get("summary", {})
    for sysn, d in tv.items():
        for a, r in d.items():
            v[f"tloc/video/{sysn}/{a}/median_ms"] = fmt(1000 * r["median_abs_err_s"], 0)
            v[f"tloc/video/{sysn}/{a}/within1"] = fmt(r["within_1s"], 1, pct=True)
            if "interval_iou_mean" in r:
                v[f"tloc/video/{sysn}/{a}/iou"] = fmt(r["interval_iou_mean"], 2)
    for n, r in data.get("localize_retrieval", {}).items():
        v[f"editret/{n}/R@1"] = fmt(r.get("R@1"), 3)
        for dim, by in r.get("R@1_by", {}).items():
            for lev, x in by.items():
                v[f"editret/{n}/{dim}/{lev}"] = fmt(x["rate"], 2)
        for opn in (A.OP, A.OP_Q):
            at = r.get("at_threshold", {}).get(opn, {})
            if "TPR" in at:
                v[f"editret/{n}/{opn}/tpr"] = fmt(at["TPR"], 3)
                for dim, by in at.get("by", {}).items():
                    for lev, x in by.items():
                        v[f"editret/{n}/{opn}/{dim}/{lev}"] = fmt(x["rate"], 2)
    return tab, v


def security_threshold_values(data, sec: dict) -> dict:
    """Which retrieval threshold each security experiment was judged at, as recorded in its output.

    bench_security.threshold() takes pair@1e-6 when the record resolves it and falls back to
    query@0.01 otherwise. Image and audio resolve pair@1e-6; the 20 video calibration clips do not,
    so every video attack was judged at query@0.01. The audio security run read the first-run audio
    records; the audio retrieval was re-run afterwards, so CLAP-PGD (which keeps every row) is also
    re-scored here at the re-run thresholds."""
    v: dict[str, str] = {}
    keys = {
        "image": {x[1] for x in sec["evasion"]["thresholds"].values()}
                 | {m["threshold_source"] for m in sec["collision"]["methods"].values()},
        "audio": {x[1] for x in sec["audio"]["thresholds"].values()},
        "video": {x[1] for x in sec["video"]["thresholds"].values()},
    }
    assert keys == {"image": {"pair@1e-06"}, "audio": {"pair@1e-06"}, "video": {"query@0.01"}}, keys
    current = {n: m.rec["thresholds"] for n, m in A.by_name(data["audio"]).items()}
    deltas = [abs(th - current[n]["pair@1e-06"]) for n, (th, _) in sec["audio"]["thresholds"].items()]
    v["sec/audio/thdelta"] = fmt(max(deltas), 4)
    changed = 0
    for budget, per in sec["audio"]["clap_pgd"].items():
        for n in sec["audio"]["thresholds"]:
            rows = [r[n] for r in per["rows"] if n in r]
            t = current[n]["pair@1e-06"]
            ok = [r for r in rows if r["score_clean"] >= t]
            rescored = sum(r["score_adv"] < t for r in ok)
            changed += rescored != per[n]["success"] or len(ok) != per[n]["n"] - per[n]["n_unmatched_clean"]
    v["sec/audio/rescore_changed"] = str(changed)
    vid = A.by_name(data["video"])
    looser = all(vid[n].rec["thresholds"]["pair@0.0001"] <= th for n, (th, _) in sec["video"]["thresholds"].items())
    assert looser, "pair@1e-4 is not looser than query@0.01 for every video method"
    return v


def security_values(data) -> dict:
    """Security macros. Evasions ran against the pair-1e-6 threshold; white-box and surrogate
    rows keep the final adversarial score, so success at the stricter headline threshold is
    re-scored exactly (score_adv below the pair-1e-7 threshold, among items that matched
    before the attack at that threshold)."""
    v: dict[str, str] = {}
    sec = data.get("security", {})
    ms = A.by_name(data["image"])
    t7 = {n: m.rec["thresholds"].get(A.OP) for n, m in ms.items()}
    ev = sec.get("evasion", {})
    if ev.get("params"):
        v["sec/n"] = str(ev["params"]["n"])
    for grp in ("whitebox", "surrogate"):
        for n, per in ev.get(grp, {}).items():
            for e, x in per.items():
                if not e.isdigit():
                    continue
                sm = x.get("summary", {})
                k = f"sec/{grp}/{n}/{e}"
                v[f"{k}/rate"] = fmt(sm.get("rate"), 2)
                v[f"{k}/pct"] = fmt(sm.get("rate"), 0, pct=True)
                v[f"{k}/psnr"] = fmt(sm.get("psnr_mean"), 1)
                v[f"{k}/lpips"] = fmt(sm.get("lpips_mean"), 3)
                t = t7.get(n)
                rows = x.get("rows", [])
                if t is not None and math.isfinite(t) and rows:
                    ok = [r for r in rows if r["score_clean"] >= t]
                    if ok:
                        rate = sum(r["score_adv"] < t for r in ok) / len(ok)
                        v[f"{k}/rate7"] = fmt(rate, 2)
                        v[f"{k}/pct7"] = fmt(rate, 0, pct=True)
    for e, per in ev.get("transfer", {}).items():
        for n, x in per.items():
            r = x.get("summary", {}).get("rate")
            v[f"sec/transfer/{e}/{n}/pct"] = fmt(r, 0, pct=True) if r is not None else "--"
    col = sec.get("collision", {})
    for n, per in col.get("methods", {}).items():
        for e, x in per.items():
            if str(e).isdigit():
                sm = x.get("summary", {})
                v[f"sec/collision/{n}/{e}/pct"] = fmt(sm.get("rate"), 0, pct=True)
                v[f"sec/collision/{n}/{e}/psnr"] = fmt(sm.get("psnr_mean"), 1)
    inv = sec.get("inversion", {})
    for n, x in inv.get("methods", {}).items():
        key = n.replace(" (wrong key)", "-wrongkey")
        for f in ("reid_top1", "lpips", "ssim", "chance"):
            if f in x:
                v[f"sec/inv/{key}/{f}"] = fmt(x[f], 3)
        if "reid_top1" in x:
            v[f"sec/inv/{key}/reid/pct"] = fmt(x["reid_top1"], 1, pct=True)
    if inv.get("methods"):
        v["sec/inv/ntest"] = fmt_int(inv.get("params", {}).get("test", int(round(1 / next(iter(inv["methods"].values()))["chance"]))))
        v["sec/inv/ntrain"] = fmt_int(inv.get("params", {}).get("train", 0))
    au = sec.get("audio", {})
    if au.get("params"):
        v["sec/audio/n"] = fmt_int(au["params"]["n"])
    for snr, per in au.get("clap_pgd", {}).items():
        ach = [r["snr_achieved"] for r in per.get("rows", []) if "snr_achieved" in r]
        if ach:
            v[f"sec/audio/snr/{snr}"] = fmt(float(np.median(ach)), 0)
        for n, x in per.items():
            if isinstance(x, dict) and "rate" in x:
                v[f"sec/audio/pgd/{snr}/{n}/pct"] = fmt(x["rate"], 0, pct=True)
    for n, x in au.get("magnitude", {}).items():
        v[f"sec/audio/mag/{n}/pitch"] = str(x.get("pitch_semitones_median"))
        v[f"sec/audio/mag/{n}/tempo"] = str(x.get("tempo_factor_median"))
    vi = sec.get("video", {})
    if vi.get("params"):
        v["sec/video/n"] = fmt_int(vi["params"]["n"])
        for nm, x in vi.get("magnitude", {}).items():
            v[f"sec/video/magnitude/{nm}/speed_never"] = fmt_int(sum(r["speed"] is None for r in x["rows"]))
    for k, per in vi.items():
        if isinstance(per, dict) and k in ("frame_pgd", "magnitude"):
            for a, b in per.items():
                if isinstance(b, dict):
                    for n, x in b.items():
                        if isinstance(x, dict) and "rate" in x:
                            v[f"sec/video/{k}/{a}/{n}/pct"] = fmt(x["rate"], 0, pct=True)
                        elif not isinstance(x, (dict, list)):
                            v[f"sec/video/{k}/{a}/{n}"] = str(x)
    v.update(security_threshold_values(data, sec))
    return v


def audioseal_key_values(d: dict) -> dict:
    """AudioSeal key uniqueness, re-derived from the seeded key assignment of bench_wmcombo.py.

    Asset i (in registration order, the order of the per-query `source` list) was given
    payload(16, i, salt="wm-audio"). Nothing checked the keys for collisions, and a 16-bit key
    identifies an asset only if no other asset shares it, so exact-key recovery on an asset whose
    key collides verifies the key but does not identify the asset."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from common import payload  # the generator the benchmark used

    r = next(iter(d["methods"].values()))
    order = list(dict.fromkeys(r["source"]))
    keys = [tuple(payload(16, i, salt="wm-audio").tolist()) for i in range(len(order))]
    count: dict[tuple, int] = {}
    for k in keys:
        count[k] = count.get(k, 0) + 1
    colliding = {order[i] for i, k in enumerate(keys) if count[k] > 1}
    ok = np.array(r["wm_ok"], dtype=bool)
    shared = np.isin(np.array(r["source"]), sorted(colliding))
    n_attacks = len(set(r["attack"]))
    assert len(set(keys)) == d["wm_n_registered_keys"], "re-derived keys disagree with the recorded key count"
    return {
        "wm/audio/n_assets": fmt_int(len(order)),
        "wm/audio/n_colliding": fmt_int(len(colliding)),
        "wm/audio/keyok/n": fmt_int(int(ok.sum())),
        "wm/audio/keyok/colliding": fmt_int(int((ok & shared).sum())),
        "wm/audio/keyok/unique/pct": fmt(float((ok & ~shared).mean()), 1, pct=True),
        "wm/audio/keyok/pct1": fmt(float(ok.mean()), 1, pct=True),
        "wm/audio/n_queries": fmt_int(len(ok)),
        "wm/audio/falsekey/nsrc": fmt_int(d["wm_false_key_n"] // n_attacks),
    }


def wm_values(data) -> tuple[str, dict]:
    v, rows = {}, []
    wc = data.get("wmcombo", {})
    lab = {"image": "TrustMark-Q", "audio": "AudioSeal", "video": "Video Seal"}
    for mod in ("image", "audio", "video"):
        d = wc.get(mod)
        if not d:
            continue
        v[f"wm/{mod}/falsekey/pct"] = fmt(d.get("wm_false_key_rate"), 2, pct=True)
        v[f"wm/{mod}/falsekey/n"] = fmt_int(d.get("wm_false_key_n", 0))
        if "wm_false_key_registry_rate" in d:
            v[f"wm/{mod}/falsekey_registry/pct"] = fmt(d["wm_false_key_registry_rate"], 2, pct=True)
            v[f"wm/{mod}/nkeys"] = fmt_int(d.get("wm_n_registered_keys", 0))
        v[f"wm/{mod}/falsekey/k"] = fmt_int(round(d["wm_false_key_rate"] * d["wm_false_key_n"]))
        if mod == "audio":
            v.update(audioseal_key_values(d))
        first = True
        for n, r in d["methods"].items():
            a = r["summary"]["_all"]
            k = f"wm/{mod}/{n}"
            for f in ("wm", "fp", "either", "both", "wm_only", "fp_only"):
                v[f"{k}/{f}/pct"] = fmt(a[f], 0, pct=True)
            for att, x in r["summary"].items():
                v[f"{k}/att/{att}/wm/pct"] = fmt(x["wm"], 0, pct=True)
                v[f"{k}/att/{att}/fp/pct"] = fmt(x["fp"], 0, pct=True)
            if "drift_below_threshold" in r:
                v[f"{k}/drift/pct"] = fmt(r["drift_below_threshold"], 1, pct=True)
                v[f"{k}/unmarkedreg/fp/pct"] = fmt(r["summary_unmarked_registry"]["_all"]["fp"], 0, pct=True)
            if n in {"image": ("PDQ", "ISCC-Image-64", "DINOv2-S", "DINOv2-S-LSH256", "SSCD-mixup", "ISC21-1st"),
                     "audio": ("Chromaprint", "ISCC-Audio-64", "audfprint", "CLAP"),
                     "video": ("vPDQ", "TMK+PDQF", "ISCC-Video-64", "SSCD-mixup-seq")}[mod]:
                label = A.LABEL.get(n, n) + (r"\reftier" if A.CLASS.get(n) == "copydet" else "")
                rows.append(" & ".join([lab[mod] if first else "", tex(label) if "reftier" not in label else label,
                                        fmt(a["wm"], 2), fmt(a["fp"], 2), fmt(a["both"], 2), fmt(a["either"], 2),
                                        fmt(r.get("drift_below_threshold"), 2) if "drift_below_threshold" in r else "--"]))
                first = False
        rows.append(r"\addlinespace")
    tab = table_env(
        r"Watermark and fingerprint on the same attacked copies of watermarked assets, pooled over all transformations. "
        r"WM: the watermark yields the embedded key; FP: the fingerprint identifies the asset at its operating point "
        r"(images pair-level $10^{-7}$, audio and video query-level 1\%); either: the cascade recovers the asset. "
        r"Drift: share of watermarked assets whose own fingerprint falls below the threshold against the unmarked original.",
        "tab:wm", "llccccc", [r"Watermark & Fingerprint & WM & FP & Both & Either & Drift"], rows[:-1])
    return tab, v


def ci_table(data) -> str:
    rows = []
    for mod in ("image", "audio", "video"):
        for m in data[mod]:
            r = m.pooled
            c = data["cluster"].get("methods", {}).get(f"{mod}/{m.name}", {})
            src = f"{fmt_int(r['n_pos'])} ({fmt_int(c['n_pos_sources'])})" if c else fmt_int(r["n_pos"])
            neg = f"{fmt_int(r['n_neg'])} ({fmt_int(c['n_test_sources'])})" if c else fmt_int(r["n_neg"])
            rows.append(" & ".join([mod, mlabel(m), with_ci(m, "R@1"), with_ci(m, f"TPR@{A.OP_Q}"),
                                    with_ci(m, f"FPRq@{A.OP_Q}"), with_ci(m, f"TPR@{A.OP}"), src, neg]))
    return table_env(
        r"Pooled results with 95\% intervals from a bootstrap over sources: registered sources are drawn with all "
        r"their transformed queries, and calibration and held-out negative sources likewise, with the threshold "
        r"recomputed in every replicate. FPR$_q$: share of held-out never-registered queries whose best score passes "
        r"the query-level 1\% threshold (out-of-sample check of the calibration). TPR$_{p,10^{-7}}$ is not resolved by "
        r"the audio and video calibration sets (--).",
        "tab:ci", "llccccrr",
        [r"Mod. & Method & R@1 & TPR$_{q,1\%}$ & FPR$_{q,1\%}$ (test) & TPR$_{p,10^{-7}}$ & $n_+$ (sources) & "
         r"$n_-$ (sources)"], rows,
        wide=True, size=r"\scriptsize")


def cluster_values(data) -> tuple[str, dict]:
    """Values from the source-level bootstrap: video calibration facts and the verification
    experiment (cluster_ci.py), plus the verification table."""
    v, rows = {}, []
    c = data.get("cluster", {})
    meth = c.get("methods", {})
    vid = {k.split("/", 1)[1]: r for k, r in meth.items() if k.startswith("video/")}
    if vid:
        any_r = next(iter(vid.values()))
        v["video/cal_sources"] = str(any_r["n_cal_sources"])
        v["video/test_sources"] = str(any_r["n_test_sources"])
        fpr = [r[A.OP_Q]["fpr_q_test"] for r in vid.values() if "fpr_q_test" in r[A.OP_Q]]
        his = [r[A.OP_Q]["fpr_ci"][1] for r in vid.values() if "fpr_ci" in r[A.OP_Q]]
        v["video/fprq/min"], v["video/fprq/max"] = fmt(min(fpr), 1, pct=True), fmt(max(fpr), 1, pct=True)
        v["video/fprq/cihi"] = fmt(max(his), 0, pct=True)
        n_cal = max(m.rec["n_cal_neg"] for m in data["video"])
        v["video/q_per_source"] = str(round(n_cal / any_r["n_cal_sources"]))
    for mod in ("image", "audio"):
        rs = [r for k, r in meth.items() if k.startswith(mod + "/")]
        if rs:
            v[f"{mod}/pos_sources"] = fmt_int(rs[0]["n_pos_sources"])
            v[f"{mod}/cal_sources"] = fmt_int(rs[0]["n_cal_sources"])
    label = {"one": "retrieval only", "k_match": "matched", "k12": "$k=12$", "k_zero": "zero cal."}
    for name, r in c.get("verify", {}).items():
        one = r[f"one_stage_{A.OP}"]
        v[f"verify/{name}/one/tpr"] = fmt(one["tpr"], 3)
        v[f"verify/{name}/one/fp"] = fmt_int(one["fp_test"])
        v["verify/n_test"] = fmt_int(one["n_test"])
        rows.append(" & ".join([tex(A.LABEL.get(name, name)), r"pair $10^{-7}$", "--", fmt(one["tpr"], 3),
                                fmt_int(one["fp_test"]), fmt(one["fp_test"] / one["n_test"], 4)]))
        for ret, rlab in (("query@0.01", r"query 1\%"), ("pair@0.0001", r"pair $10^{-4}$")):
            for kn in ("k_match", "k12", "k_zero"):
                x = r.get(f"{ret}/{kn}")
                if not x:
                    continue
                key = f"verify/{name}/{ret}/{kn}"
                v[f"{key}/tpr"], v[f"{key}/k"] = fmt(x["tpr"], 3), str(x["k"])
                v[f"{key}/fp"] = fmt_int(x["fp_test"])
                v[f"{key}/ret_tpr"], v[f"{key}/ret_fp"] = fmt(x["retrieval_only_tpr"], 3), fmt_int(x["retrieval_only_fp_test"])
                ci = x["tpr_ci"]
                rows.append(" & ".join(["", rlab, f"{x['k']} ({label[kn]})",
                                        f"{fmt(x['tpr'], 3)} \\tiny[{fmt(ci[0], 2)}, {fmt(ci[1], 2)}]",
                                        fmt_int(x["fp_test"]), fmt(x["fp_test"] / x["n_test"], 4)]))
        rows.append(r"\addlinespace")
    if rows:
        rows.pop()
    tab = table_env(
        r"Second-stage verification of image matches. The top-1 candidate is accepted only if its retrieval score "
        r"passes a threshold and SIFT+RANSAC finds at least $k$ geometric inliers between the query and the stored "
        r"original. Both thresholds are fixed on the calibration negatives: matched, the smallest $k$ whose "
        r"calibration false bindings do not exceed those of retrieval alone at pair-level $10^{-7}$; zero cal., the "
        r"smallest $k$ no calibration negative reaches; and the localization track's $k=12$. FB: false bindings among "
        rf"the {v.get('verify/n_test', '')} held-out never-registered queries. TPR interval: bootstrap over sources.",
        "tab:verify", "llrccc", [r"Method & Retrieval threshold & $k$ & TPR & FB & FB rate"], rows) if rows else ""
    return tab, v


# ------------------------------------------------------------------ values
def values(data) -> dict[str, str]:
    v: dict[str, str] = {}
    for mod in ("image", "audio", "video"):
        ms = data[mod]
        v[f"count/{mod}/methods"] = str(len(ms))
        v[f"count/{mod}/methods_p"] = str(sum(m.tier == "P" for m in ms))
        if ms:
            v[f"count/{mod}/attacks"] = str(len([a for a in A.attack_order(ms[0]) if a != "none"]))
            p = ms[0].pooled
            v[f"count/{mod}/pos"] = fmt_int(p["n_pos"])
            v[f"count/{mod}/neg_test"] = fmt_int(ms[0].rec["n_test_neg"])
            v[f"count/{mod}/neg_cal"] = fmt_int(ms[0].rec["n_cal_neg"])
            v[f"count/{mod}/nref"] = fmt_int(ms[0].rec["n_ref"])
        for m in ms:
            k = f"{mod}/{m.name}"
            p = m.pooled
            for key in ("R@1", "R@10", "muAP", f"TPR@{A.OP_Q}", f"FPRq@{A.OP_Q}", f"FPRq@{A.OP}",
                        *(f"TPR@{t}" for t in A.PAIR_TARGETS)):
                if key in p:
                    v[f"{k}/{key}"] = fmt(p[key], 3)
                    v[f"{k}/{key}/pct"] = fmt(p[key], 1, pct=True)
            for a in A.attack_order(m):
                v[f"{k}/att/{a}"] = fmt(m.at(a), 2)
                v[f"{k}/att/{a}/pct"] = fmt(m.at(a), 0, pct=True)
            for ct in ("music", "speech"):
                for a, r in m.rec.get(f"per_attack_{ct}", {}).items():
                    v[f"{k}/{ct}/att/{a}"] = fmt(r.get("R@1"), 2)
                    v[f"{k}/{ct}/att/{a}/pct"] = fmt(r.get("R@1"), 0, pct=True)
                    if a == "_all":
                        v[f"{k}/{ct}/tpr"] = fmt(r.get(f"TPR@{A.OP_Q}"), 3)
            if "dedup" in m.rec:
                v[f"{mod}/dedup/neg"] = str(m.rec["dedup"]["n_neg_flagged"])
                v[f"{mod}/dedup/items"] = str(m.rec["dedup"]["n_items_with_duplicate"])
            for fam in A.attack_families(m):
                v[f"{k}/fam/{fam}"] = fmt(A.family_mean(m, fam), 2)
            v[f"{k}/ms"] = fmt_sig(m.ms_per_item(), 2)
            v[f"{k}/bits"] = fmt_sig(m.bits, 3) if math.isfinite(m.bits) else "--"
            h = m.rec.get("hard_negative", {}).get(A.OP_Q)
            if h:
                v[f"{k}/hard"] = fmt(h["rate"], 3)
                v[f"{k}/hard/pct"] = fmt(h["rate"], 1, pct=True)
            if "disc21" in m.rec:
                v[f"{k}/disc"] = fmt(m.rec["disc21"]["muAP"], 3)
    # out-of-sample false-match rate at the 1% query threshold, over methods whose threshold is
    # informative (TPR > 0: a degenerate code that ties every query at distance 0 is excluded)
    fq = [m.pooled.get(f"FPRq@{A.OP_Q}") for m in data["image"] if (m.pooled.get(f"TPR@{A.OP_Q}") or 0) > 0]
    fq = [x for x in fq if x is not None]
    if fq:
        v["image/fprq/min"], v["image/fprq/max"] = fmt(min(fq), 1, pct=True), fmt(max(fq), 1, pct=True)
    for f in sorted((A.RAW / "audio_dedup-none").glob("*.json")) if (A.RAW / "audio_dedup-none").exists() else []:
        r = json.loads(f.read_text())["per_attack"]["_all"]
        v[f"audio/{f.stem}/nodedup/tpr"] = fmt(r.get(f"TPR@{A.OP_Q}"), 3)
    st = data.get("stability", {}).get("methods", {})
    for n, r in st.items():
        v[f"stab/{n}/identical/pct"] = fmt(r["identical_share"], 0, pct=True)
        v[f"stab/{n}/differ/pct"] = fmt(1 - r["identical_share"], 0, pct=True)
        if "bits_differing_max" in r:
            v[f"stab/{n}/bitsmax"] = str(r["bits_differing_max"])
        if "one_minus_cos_max" in r:
            v[f"stab/{n}/cosmax"] = fmt_sig(r["one_minus_cos_max"], 2)
        if "self_below_threshold" in r:
            v[f"stab/{n}/below"] = str(r["self_below_threshold"])
        v["stab/n"] = str(r["n"])
    lat = A._j(A.RAW / "latency.json")
    for mod in ("image", "audio", "video"):
        for n, r in lat.get(mod, {}).items():
            v[f"lat/{mod}/{n}"] = fmt_sig(r["ms_per_item"] if mod == "image" else r["ms_per_media_second"], 2)
            if mod == "image":
                # Instance-hours per million images on the measurement host (g5.2xlarge, on-demand
                # USD 1.212/h): GPU methods use the one GPU, CPU methods one thread on each of 8 vCPUs.
                h = r["ms_per_item"] * 1e6 / 3.6e6 / (1 if r["device"] != "cpu" else G5_VCPUS)
                v[f"lat/image/{n}/usd1m"] = f"{h * G5_USD_PER_H:.2f}"
    tot = sum(len(data[m]) for m in ("image", "audio", "video"))
    v["count/methods"] = str(tot)
    v["count/attacks"] = str(sum(int(v.get(f"count/{m}/attacks", 0)) for m in ("image", "audio", "video")))
    d = data.get("dedup", {})
    for k in ("n_dist_flagged", "n_neg_flagged", "n_hard_near_duplicate", "n_hard", "n_dist"):
        if k in d:
            v[f"dedup/{k}"] = fmt_int(int(d[k]))
    man = data.get("manifest", {})
    if "image" in man:
        for k, n in man["image"]["sets"].items():
            v[f"corpus/abo/{k}"] = fmt_int(len(n))
        d = man["disc21"]
        v["corpus/disc/gt"], v["corpus/disc/negq"], v["corpus/disc/refs"] = (fmt_int(len(d["pairs"])),
                                                                            fmt_int(len(d["negq"])), fmt_int(len(d["refs"])))
        a = man["audio"]["sets"]
        for k in ("reg", "dist", "neg"):
            v[f"corpus/audio/{k}"] = fmt_int(len(a[k]))
        v["corpus/audio/music"] = fmt_int(sum(i.startswith("m") for s_ in a.values() for i in s_))
        v["corpus/audio/speech"] = fmt_int(sum(i.startswith("s") for s_ in a.values() for i in s_))
        for k in ("reg", "dist", "neg"):
            v[f"corpus/video/{k}"] = fmt_int(len(man["video"]["sets"][k]))
        v["corpus/video/xiph"] = str(sum(i.startswith("xiph") for i in man["video"]["items"]))
    ed = A._j(A.RAW / "locks" / "edits_index.json")
    if ed:
        v["edit/n_edits"] = fmt_int(len(ed["edits"]))
        v["edit/n_sources"] = fmt_int(len({e["key"].split("|")[0] for e in ed["edits"]}))
    # soft-binding list snapshot (commit 5c864efb, 2026-09-16)
    v["c2pa/list_total"], v["c2pa/list_watermark"], v["c2pa/list_fingerprint"] = "53", "44", "9"
    v["price/gpu"] = f"{A.PRICE_GPU_USD_H:.3f}"
    return v


def numbers_tex(vals: dict[str, str]) -> str:
    lines = ["% Generated by scripts/build_numbers.py from results-gpu/. Do not edit.", r"\makeatletter",
             r"\newcommand{\val}[1]{\ifcsname val@#1\endcsname\csname val@#1\endcsname"
             r"\else\PackageError{numbers}{Undefined value #1}{Run build_numbers.py}\fi}"]
    for k in sorted(vals):
        lines.append(rf"\expandafter\def\csname val@{k}\endcsname{{{vals[k]}}}")
    lines.append(r"\makeatother")
    return "\n".join(lines) + "\n"


def build() -> dict[Path, str]:
    data = A.load()
    paired_tab, paired_vals = paired_table(data)
    sens_tab, sens_vals = sensitivity_table(data)
    loc_tab, loc_vals = localization_values(data)
    wm_tab, wm_vals = wm_values(data)
    verify_tab, cluster_vals = cluster_values(data)
    vals = values(data) | paired_vals | sens_vals | loc_vals | security_values(data) | wm_vals | cluster_vals
    out = {GEN / "numbers.tex": numbers_tex(vals), TAB / "methods.tex": inventory_table(data),
           TAB / "verify.tex": verify_tab,
           TAB / "image.tex": image_table(data), TAB / "paired.tex": paired_tab, TAB / "ci.tex": ci_table(data),
           TAB / "sensitivity.tex": sens_tab, TAB / "localization.tex": loc_tab,
           TAB / "wm.tex": wm_tab}
    if data["audio"]:
        out[TAB / "audio.tex"] = timebased_table(data, "audio", [("music:excerpt10", "Music 10\\,s exc."),
                                                                 ("speech:excerpt10", "Speech 10\\,s exc."),
                                                                 ("pitch+2", "Pitch +2"), ("rerecord", "Re-record")])
    if data["video"]:
        out[TAB / "video.tex"] = timebased_table(data, "video", [("crop50", "Crop 50\\%"), ("pip", "PiP"),
                                                                 ("insert", "Inserted"), ("screenrec", "Screen rec.")])
    return out


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
