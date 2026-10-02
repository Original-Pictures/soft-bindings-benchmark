"""Aggregate raw per-config result files into results.json (the single source for the report).

Detection rules (stated in the report):
  * multi-bit models without a native detector: "watermarked with key K" when the
    decoded bits match K on >= k_n bits, k_n = smallest k with P[Bin(n, 1/2) >= k] <= 1e-6
    per key (theoretical false-match rate per key; the empirical rate is measured).
  * models with a native presence decision (TrustMark ECC decode, AudioSeal detected-
    frames >= 0.8, WavMark pattern sync, Perth score >= 0.5, SilentCipher confidence
    >= 0.5) additionally get a native false-detection rate on unmarked media.
Default-pick rule: among DEPLOYABLE configs with mean bit
accuracy >= 0.95 after JPEG-75 / H.264 CRF-23 / MP3-128 (detection rate for the
presence-only Perth) and FPR <= 1e-3, pick best perceptual quality. Tightening applied
here: the 0.95 must hold on the pooled corpus AND on every content subset, pick best perceptual quality: image FLIP+LPIPS
rank sum, audio PESQ then SI-SNR, video FLIP+LPIPS rank sum; PSNR / VMAF break ties.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

from common import BENCH_DIR

GATE = {"image": "jpeg75", "audio": "mp3_128", "video": "h264_crf23"}
FPR_MAX = 1e-3
BITACC_MIN = 0.95


def k_threshold(n: int, p: float = 1e-6) -> int:
    for k in range(n // 2, n + 1):
        if binom.sf(k - 1, n, 0.5) <= p:
            return k
    return n + 1


def cp_upper(x: int, n: int, conf: float = 0.95) -> float:
    if n == 0:
        return float("nan")
    return 1.0 if x == n else float(beta.ppf(conf, x + 1, n - x))


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def load_dir(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.json")) + sorted(d.glob("*.json.gz")):
        raw = gzip.open(f).read() if f.suffix == ".gz" else f.read_bytes()
        out.append(json.loads(raw))
    return out


def fpr_block(fpr: list[dict], nbits: int) -> dict:
    res: dict = {}
    if nbits:
        k = k_threshold(nbits)
        trials = sum(len(e["matches"]) for e in fpr)
        hits = sum(sum(1 for m in e["matches"] if m >= k) for e in fpr)
        allm = [m for e in fpr for m in e["matches"]]
        res.update({"k_threshold": k, "key_trials": trials, "key_false_matches": hits,
                    "key_fpr": hits / trials if trials else None, "key_fpr_cp95": cp_upper(hits, trials),
                    "unmarked_match_mean": mean(allm), "unmarked_match_max": max(allm) if allm else None,
                    "theoretical_per_key": float(binom.sf(k - 1, nbits, 0.5))})
    nat = [e["native_detect"] for e in fpr if e.get("native_detect") is not None]
    if nat:
        x = int(sum(bool(v) for v in nat))
        res.update({"native_trials": len(nat), "native_false_detections": x, "native_fpr": x / len(nat),
                    "native_fpr_cp95": cp_upper(x, len(nat))})
        by: dict = {}
        for e in fpr:
            if e.get("native_detect") is not None:
                by.setdefault(e.get("attack", "none"), []).append(bool(e["native_detect"]))
        res["native_fpr_by_attack"] = {a: float(np.mean(v)) for a, v in by.items()}
    return res


def attack_block(records: list[dict], nbits: int) -> dict:
    k = k_threshold(nbits) if nbits else None
    names = sorted({a for r in records for a in r.get("attacks", {})})
    out = {}
    for a in names:
        rs = [r["attacks"][a] for r in records if a in r.get("attacks", {})]
        blk = {"n": len(rs)}
        if nbits:
            blk["bit_acc"] = mean([x.get("bit_acc") for x in rs])
            blk["bit_acc_min"] = min(x.get("bit_acc", 0) for x in rs)
            blk["key_detect_rate"] = mean([float(x.get("matches", 0) >= k) for x in rs])
        if any(x.get("native_detect") is not None for x in rs):
            blk["native_detect_rate"] = mean([float(bool(x.get("native_detect"))) for x in rs if x.get("native_detect") is not None])
        if any("payload_exact" in x for x in rs):
            blk["payload_exact_rate"] = mean([float(x.get("payload_exact", False)) for x in rs])
        for extra in ("mean_abs_logit", "min_abs_logit", "frame_bit_acc", "decode_fps"):
            if any(extra in x for x in rs):
                blk[extra] = mean([x.get(extra) for x in rs])
        out[a] = blk
    return out


def summarize(kind: str, res: dict, cpu: dict | None) -> dict:
    meta = res["meta"]
    recs = [r for r in res.get("records", []) if "error" not in r and not r.get("excluded")]
    errs = [r for r in res.get("records", []) if "error" in r and not r.get("excluded")]
    s = {"meta": meta, "n_items": len(recs), "n_errors": len(errs), "errors": sorted({r["error"][:160] for r in errs})[:5],
         "host": res.get("host", {}), "wall_s": res.get("wall_s")}
    if "error" in res:
        s["fatal"] = res["error"]
        return s
    qkeys = {"image": ["psnr", "ssim", "ms_ssim", "lpips", "flip", "cvvdp"],
             "audio": ["snr", "si_snr", "pesq", "stoi", "dlufs", "spec_diff_db"],
             "video": ["psnr", "ssim", "lpips", "flip", "vmaf", "vmaf_identical", "embed_fps"]}[kind]  # video CVVDP: CUDA OOM with 2 workers, not reported
    s["quality"] = {q: mean([r.get(q) for r in recs]) for q in qkeys}
    if kind == "audio":
        s["quality"]["abs_dlufs"] = mean([abs(r["dlufs"]) for r in recs if r.get("dlufs") is not None])
    group = True  # content classes: image/audio sets; each video clip is its own class
    if group:
        by: dict = {}
        for r in recs:
            key = Path(r["item"]).stem if kind == "video" else r["set"] + (f"@{r['sr']}" if kind == "audio" else "")
            by.setdefault(key, []).append(r)
        s["quality_by_set"] = {g: {q: mean([r.get(q) for r in rs]) for q in qkeys} for g, rs in sorted(by.items())}
        s["robustness_by_set"] = {g: {a: (v.get("bit_acc") if meta["nbits"] else v.get("native_detect_rate")) for a, v in attack_block(rs, meta["nbits"]).items()} for g, rs in sorted(by.items())}
    s["robustness"] = attack_block(recs, meta["nbits"])
    s["fpr"] = fpr_block(res.get("fpr", []), meta["nbits"])
    allrec = res.get("records", [])
    by_set: dict = {}
    for e in res.get("fpr", []):
        if e.get("native_detect") is not None and e["item"] < len(allrec):
            g = allrec[e["item"]].get("set", "?")
            by_set.setdefault(g, []).append(bool(e["native_detect"]))
    s["fpr_by_set"] = {g: float(np.mean(v)) for g, v in by_set.items()}
    s["latency"] = {k: res.get(k) for k in ("embed_ms_per_mp_median", "decode_ms_median", "embed_ms_per_audio_s_median",
                                            "decode_ms_per_audio_s_median", "load_s") if res.get(k) is not None}
    if kind == "video":
        s["latency"]["embed_fps"] = s["quality"].get("embed_fps")
        s["latency"]["decode_fps"] = s["robustness"].get("h264_crf18", {}).get("decode_fps")
    s["latency"]["device"] = res.get("device")
    if cpu and "error" not in cpu:
        s["latency_cpu"] = {k: v for k, v in cpu.items() if k not in ("meta",)}
    # gate
    g = s["robustness"].get(GATE[kind], {})
    gate_value = g.get("bit_acc") if meta["nbits"] else g.get("native_detect_rate")
    native = s["fpr"].get("native_fpr")
    key = s["fpr"].get("key_fpr")
    fpr_used = native if native is not None else key
    # Robust = the gate holds on the pooled corpus AND on every content subset (Kodak, CLIC,
    # DIV2K, HDR16 / speech, music at each rate): a pooled mean can hide a failing class.
    per_set = {g: v.get(GATE[kind]) for g, v in s.get("robustness_by_set", {}).items()}
    worst_set = min((v for v in per_set.values() if v is not None), default=gate_value)
    s["gate"] = {"attack": GATE[kind], "value": gate_value, "worst_set_value": worst_set,
                 "worst_set": min(per_set, key=lambda g: per_set[g] if per_set[g] is not None else 9) if per_set else None,
                 "robust": gate_value is not None and gate_value >= BITACC_MIN and (worst_set is None or worst_set >= BITACC_MIN),
                 "fpr_used": fpr_used, "fpr_source": "native" if native is not None else "key-match",
                 "fpr_ok": fpr_used is not None and fpr_used <= FPR_MAX,
                 "fpr_resolvable": (s["fpr"].get("native_fpr_cp95") or s["fpr"].get("key_fpr_cp95") or 1) <= FPR_MAX}
    # Secondary view: when the verifier already knows the expected payload (C2PA manifest
    # present), the relevant false-positive is the key-match rate, not blind presence.
    s["gate_keymatch"] = {"fpr_used": key, "fpr_ok": key is not None and key <= FPR_MAX}
    s["eligible_keymatch"] = bool(meta["deployable"] and s["gate"]["robust"] and s["gate_keymatch"]["fpr_ok"] and s["n_errors"] == 0)
    s["eligible"] = bool(meta["deployable"] and s["gate"]["robust"] and s["gate"]["fpr_ok"] and s["n_errors"] == 0)
    return s


def rank(kind: str, rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if "quality" in r]

    def rk(vals, reverse):
        order = sorted(range(len(vals)), key=lambda i: (vals[i] is None, -(vals[i] or 0) if reverse else (vals[i] or 0)))
        out = [0] * len(vals)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out

    if kind == "audio":
        key = lambda r: (-(r["quality"]["pesq"] or 0), -(r["quality"]["si_snr"] or 0))  # noqa: E731
        ok.sort(key=key)
    else:
        f = rk([r["quality"].get("flip") for r in ok], False)
        l = rk([r["quality"].get("lpips") for r in ok], False)
        for r, a, b in zip(ok, f, l):
            r["perceptual_rank_sum"] = a + b
        tie = "vmaf" if kind == "video" else "psnr"
        ok.sort(key=lambda r: (r["perceptual_rank_sum"], -(r["quality"].get(tie) or 0)))
    for i, r in enumerate(ok):
        r["quality_rank"] = i + 1
    return ok


def stability_block(d: Path) -> dict:
    out = {}
    fh = {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("framehash-*.json"))}
    if len(fh) >= 2:
        hosts_ = list(fh)
        a, b = fh[hosts_[0]], fh[hosts_[1]]
        out["_framehash"] = {"hosts": hosts_, "pyav": [a["av"], b["av"]], "libswscale": [a["ffmpeg"], b["ffmpeg"]],
                             "files": {k: {"yuv420p_equal": a["files"][k]["yuv420p_16f"] == b["files"][k]["yuv420p_16f"],
                                           "rgb24_equal": a["files"][k]["rgb24_16f"] == b["files"][k]["rgb24_16f"]}
                                       for k in a["files"] if k in b["files"]}}
    files = sorted(f for f in d.glob("*.json") if not f.name.startswith("framehash-"))
    hosts = {f.stem: json.loads(f.read_text()) for f in files}
    if not hosts:
        return out
    payloads = {}
    pf = Path.home() / "op-wm-bench-work" / "stability" / "payloads.json"
    if pf.exists():
        payloads = json.loads(pf.read_text())
    rows = []
    for host, data in hosts.items():
        for fx, v in data.get("video", {}).items():
            for backend, dec in v["decodes"].items():
                L = np.asarray(dec["avg_logits"])
                stem = fx.split("_sw")[0]
                acc = None
                if stem in payloads:
                    acc = float(((L > 0).astype(int) == np.asarray(payloads[stem])).mean())
                rows.append({"host": host, "file": fx, "backend": backend, "bits_sha": _bits_hash(L > 0),
                             "bit_acc": acc, "mean_abs_logit": float(np.abs(L).mean()), "logits": L})
        for fx, v in data.get("image", {}).items():
            for backend, logits in v["decodes"].items():
                L = np.asarray(logits)
                rows.append({"host": host, "file": fx, "backend": backend, "bits_sha": _bits_hash(L > 0),
                             "bit_acc": None, "mean_abs_logit": float(np.abs(L).mean()), "logits": L})
    by: dict = {}
    for r in rows:
        by.setdefault(r["file"], []).append(r)
    for fx, rs in by.items():
        base = rs[0]["logits"]
        out[fx] = {"decodes": [{"host": r["host"], "backend": r["backend"], "bits_sha": r["bits_sha"], "bit_acc": r["bit_acc"],
                                "mean_abs_logit": r["mean_abs_logit"],
                                "max_abs_logit_diff_vs_first": float(np.max(np.abs(r["logits"] - base))),
                                "bit_flips_vs_first": int(np.sum((r["logits"] > 0) != (base > 0)))} for r in rs],
                   "distinct_bitstrings": len({r["bits_sha"] for r in rs})}
        same = cross = 0.0
        flips = 0
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                dl = float(np.max(np.abs(a["logits"] - b["logits"])))
                if a["host"] == b["host"]:
                    same = max(same, dl)
                else:
                    cross = max(cross, dl)
                    flips = max(flips, int(np.sum((a["logits"] > 0) != (b["logits"] > 0))))
        out[fx].update({"same_host_max_dlogit": same, "cross_host_max_dlogit": cross, "cross_host_bit_flips": flips})
    return out


def _bits_hash(b) -> str:
    import hashlib

    return hashlib.sha256(np.packbits(np.asarray(b, dtype=np.uint8)).tobytes()).hexdigest()[:12]


def main(results_dir: str) -> None:
    root = Path(results_dir)
    out = {"schema": "op.watermark-bench.results.v1", "rules": {"gate_attacks": GATE, "bit_acc_min": BITACC_MIN,
                                                                  "fpr_max": FPR_MAX, "key_threshold_p": 1e-6}}
    for kind in ("image", "audio", "video"):
        d = root / kind
        if not d.exists():
            continue
        cpu = {c["meta"]["name"]: c for c in load_dir(d / "cpu_latency")} if (d / "cpu_latency").exists() else {}
        raw = load_dir(d)
        # An item that fails in EVERY config is a corpus defect (e.g. a clip shorter than the
        # 400 ms loudness block after trimming), not a model failure: exclude it everywhere.
        failing = [{r["item"] for r in x.get("records", []) if "error" in r} for x in raw if "records" in x]
        corpus_bad = set.intersection(*failing) if failing else set()
        for x in raw:
            for r in x.get("records", []):
                r["excluded"] = r["item"] in corpus_bad
        out.setdefault("corpus_excluded", {})[kind] = sorted(corpus_bad)
        rows = [summarize(kind, r, cpu.get(r["meta"]["name"])) for r in raw]
        rows = rank(kind, rows) + [r for r in rows if "quality" not in r]
        elig = [r for r in rows if r.get("eligible")]
        out[kind] = {"configs": rows,
                     "default": elig[0]["meta"]["name"] if elig else None,
                     "runner_up": next((r["meta"]["name"] for r in elig[1:] if r["meta"]["family"] != elig[0]["meta"]["family"]), elig[1]["meta"]["name"] if len(elig) > 1 else None),
                     "eligible": [r["meta"]["name"] for r in elig]}
    if (root / "stability").exists():
        out["stability"] = stability_block(root / "stability")
    narr = BENCH_DIR / "scripts" / "narrative.json"  # hand-written interpretation, cites results.json numbers
    if narr.exists():
        out["narrative"] = json.loads(narr.read_text())
    lock = BENCH_DIR / "scripts" / "weights.lock.json"
    if lock.exists():
        out["weights"] = json.loads(lock.read_text())
    src = BENCH_DIR / "scripts" / "sources.lock.json"
    if src.exists():
        out["sources"] = json.loads(src.read_text())
    (BENCH_DIR / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    for kind in ("image", "audio", "video"):
        if kind in out:
            print(kind, "default:", out[kind]["default"], "runner-up:", out[kind]["runner_up"], "eligible:", out[kind]["eligible"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(BENCH_DIR / "results-gpu"))
