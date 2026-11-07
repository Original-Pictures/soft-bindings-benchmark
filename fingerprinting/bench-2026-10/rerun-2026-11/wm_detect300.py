"""v1.1: TrustMark decoder-only versus detect-first in the watermark-combination track (image part).

Same queries as bench_wmcombo.decode_image: the 300 marked product photos (drift300.py render) under the 36 copy
attacks, and the first 500 `neg` photos under the same attacks for the false-key rate. Each query is decoded with
both paths of models_image.TrustMarkAdapter (Q, strength 1, BCH_5). The fingerprint side is not rerun: the union
with each fingerprint is recomputed from the paper's per-query fingerprint outcomes in results-gpu/wmcombo, keyed by
"source|attack", so only the watermark column changes.

    BENCH_WORK=~/op-v11-work/full python wm_detect300.py decode [--device mps]   (needs the full ABO corpus)
    python wm_detect300.py analyze                                           writes results-wm/wm_detect300.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
WM300 = Path(os.environ.get("V11_WORK", Path.home() / "op-v11-work")) / "wm300"
sys.path.insert(0, str(BENCH / "scripts"))

import numpy as np  # noqa: E402

OUT = HERE / "results-wm"
MODES = {"decoder": False, "detect": True}


def decode(device: str, n_neg: int, n_pos: int = 300) -> None:
    import attacks
    import bench_image
    from bench_wmcombo import TRUSTMARK
    from common import CORPUS, load_rgb, payload
    from models_image import TrustMarkAdapter

    man = json.loads((CORPUS / "corpus_manifest.json").read_text())["image"]
    tm = TrustMarkAdapter(*TRUSTMARK, detect_first=True)
    tm.load(device)
    srcs = man["sets"]["pos"][:300]
    keys = {s: tm._codeword(payload(tm.nbits, i, salt="wm-image"))[0] for i, s in enumerate(srcs)}
    attacks.BACKGROUNDS = [load_rgb(CORPUS / man["items"][i]["path"], 1024) for i in man["sets"]["bg"]]
    OUT.mkdir(exist_ok=True)
    part = OUT / "decode.partial.json"
    rec = json.loads(part.read_text()) if part.exists() else {"pos": {}, "neg": {}}
    timing = {m: [] for m in MODES}
    jobs = [("pos", s, WM300 / "marked" / f"{s}.png") for s in srcs[:n_pos]] + \
           [("neg", s, CORPUS / man["items"][s]["path"]) for s in man["sets"]["neg"][:n_neg]]
    for j, (kind, s, path) in enumerate(jobs):
        for atk in attacks.COPY_ATTACKS:
            k = f"{s}|{atk}"
            if k in rec[kind]:
                continue
            x = bench_image.make_image(str(path), atk, k)
            r = {}
            for mode, flag in MODES.items():
                tm.detect_first = flag
                t0 = time.perf_counter()
                d = tm.decode(x)
                timing[mode].append(time.perf_counter() - t0)
                r[mode] = ({"present": bool(d["native_detect"]), "key_ok": bool(d["native_detect"]) and d["native_payload"] == keys[s]}
                           if kind == "pos" else {"present": bool(d["native_detect"])})
            rec[kind][k] = r
        if j % 20 == 19:
            part.write_text(json.dumps(rec))
            print(f"{j + 1}/{len(jobs)} sources; median decode s: " +
                  ", ".join(f"{m} {np.median(v):.3f}" for m, v in timing.items() if v), flush=True)
    rec["timing_median_s"] = {m: float(np.median(v)) for m, v in timing.items() if v}
    rec["device"] = device
    (OUT / "decode.json").write_text(json.dumps(rec))
    part.unlink(missing_ok=True)


def analyze() -> None:
    from analysis_stats import cp_interval

    dec = json.loads((OUT / "decode.json").read_text())
    paper = json.loads(gzip.decompress((BENCH / "results-gpu/wmcombo/wmcombo.json.gz").read_bytes()))["image"]
    res = {"paper_wm_key_rate": None, "modes": {}}
    for mode in MODES:
        pos = {k: v[mode]["key_ok"] for k, v in dec["pos"].items()}
        neg = [v[mode]["present"] for v in dec["neg"].values()]
        negsrc = {}
        for k, v in dec["neg"].items():
            negsrc[k.split("|")[0]] = negsrc.get(k.split("|")[0], False) or v[mode]["present"]
        out = {"wm_key_rate": float(np.mean(list(pos.values()))), "n_pos": len(pos),
               "false_key_rate": float(np.mean(neg)), "false_key_n": len(neg),
               "false_key_sources": int(sum(negsrc.values())), "n_neg_sources": len(negsrc),
               "false_key_sources_ci": cp_interval(int(sum(negsrc.values())), len(negsrc)), "union": {}}
        for name, m in paper["methods"].items():
            keys = [f"{s}|{a}" for s, a in zip(m["source"], m["attack"])]
            sel = np.array([k in pos for k in keys])  # a run over fewer than 300 sources covers a subset
            wm = np.array([pos[k] for k, x in zip(keys, sel) if x])
            fp = np.array(m["fp_ok_registry_marked"])[sel]
            fpu = np.array(m["fp_ok_registry_unmarked"])[sel]
            wm_paper = np.array(m["wm_ok"])[sel]
            out["union"][name] = {"n": int(sel.sum()), "wm": float(wm.mean()), "fp": float(fp.mean()),
                                  "either": float((wm | fp).mean()), "either_unmarked_registry": float((wm | fpu).mean()),
                                  "paper_wm_same_queries": float(wm_paper.mean()),
                                  "paper_either_same_queries": float((wm_paper | fp).mean())}
        res["modes"][mode] = out
    m = paper["methods"]["DINOv2-S"]
    res["paper_wm_key_rate"] = float(np.mean(m["wm_ok"]))
    res["paper_false_key_rate"] = paper["wm_false_key_rate"]
    keys = [f"{s}|{a}" for s, a in zip(m["source"], m["attack"])]
    res["agreement_decoder_vs_paper"] = float(np.mean([dec["pos"][k]["decoder"]["key_ok"] == w
                                                       for k, w in zip(keys, m["wm_ok"]) if k in dec["pos"]]))
    res["timing_median_s"] = dec.get("timing_median_s")
    (OUT / "wm_detect300.json").write_text(json.dumps(res, indent=1) + "\n")
    for mode, o in res["modes"].items():
        print(f"{mode}: key {100 * o['wm_key_rate']:.1f} %, false key {100 * o['false_key_rate']:.2f} % of {o['false_key_n']} "
              f"({o['false_key_sources']}/{o['n_neg_sources']} sources); DINOv2-S either {100 * o['union']['DINOv2-S']['either']:.1f} %")
    print(f"paper: key {100 * res['paper_wm_key_rate']:.1f} %, false key {100 * res['paper_false_key_rate']:.2f} %; "
          f"decoder-only agrees with the paper on {100 * res['agreement_decoder_vs_paper']:.1f} % of queries")


def export_decode(dst: str, mode: str = "decoder") -> None:
    """One decode path in bench_wmcombo's decode.json format, for bench_wmcombo analyze."""
    dec = json.loads((OUT / "decode.json").read_text())
    out = {kind: {k: v[mode] for k, v in dec[kind].items()} for kind in ("pos", "neg")}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(out))
    print(f"wrote {dst} ({mode}: {len(out['pos'])} pos, {len(out['neg'])} neg)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["decode", "analyze", "export-decode"])
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--mode", default="decoder", choices=list(MODES))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-neg", type=int, default=500)
    ap.add_argument("--n-pos", type=int, default=300, help="marked sources decoded (all 300 by default)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)
    if a.stage == "decode":
        decode(a.device, a.n_neg, a.n_pos)
    elif a.stage == "export-decode":
        export_decode(a.dst, a.mode)
    else:
        analyze()
