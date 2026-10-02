"""Watermark + fingerprint combination track.

A provenance service can bind a manifest to content twice: an invisible watermark that
carries a key (TrustMark for images, AudioSeal for audio, Video Seal for video, all MIT)
and a fingerprint looked up in a registry. This track measures how the two interact:

  drift          does embedding the watermark move the fingerprint? (score of marked vs
                 unmarked original, against each method's calibrated threshold)
  where to hash  registry built from the unmarked original vs from the marked asset
                 that is actually published (queries are always derived from the marked asset)
  complementarity per attack: P(watermark recovers the key), P(fingerprint identifies the
                 asset), P(either) and P(both)
  cascade        read the watermark first; fall back to the fingerprint if no valid key.
                 False positives of the cascade = watermark false keys + fingerprint false
                 matches on never-registered (unmarked) queries.

Stages:
  python bench_wmcombo.py render {image,audio,video} [--n N]  marked assets + attacked copies,
                                                            registers descriptor sets wm_ref/wm_pos
  (then bench_image/audio/video.py extract --sets wm_ref,wm_pos  picks those sets up)
  python bench_wmcombo.py decode {image,audio,video}         watermark decode on wm_pos and neg
  python bench_wmcombo.py analyze                           -> $BENCH_RESULTS/wmcombo/*.json

The attacked copies use the same per-source random draws as the unmarked retrieval
queries (common.rng_for keyed by source id), so marked and unmarked results are paired.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

import attacks
from common import (CORPUS, DESC, OUT, RESULTS, Timer, host_info, load_rgb, payload, register_set, rng_for, to_u8,
                    write_json)

WM = OUT / "wm"
N_DEFAULT = {"image": 300, "audio": 200, "video": 40}
TRUSTMARK = ("Q", 1.0)          # variant, WM_STRENGTH (library default strength)
AUDIOSEAL = ("base", 1.0)       # variant, alpha
VIDEOSEAL_BITS = 256


def _man() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())


# ------------------------------------------------------------------ watermark models
def trustmark(device):
    from models_image import TrustMarkAdapter

    a = TrustMarkAdapter(*TRUSTMARK)
    a.load(device)
    return a


def audioseal(device):
    from models_audio import AudioSealAdapter

    a = AudioSealAdapter(*AUDIOSEAL)
    a.load(device)
    return a


def videoseal(device):
    from models_image import MetaSealAdapter

    a = MetaSealAdapter("videoseal", None)
    a.load(device)
    return a


def vs_embed(a, frames: np.ndarray, bits: np.ndarray, device: str) -> np.ndarray:
    """Video Seal in native video mode, chunked (as in the 2026-09 watermark bench)."""
    import torch

    m = a.m
    win = int(getattr(m, "step_size", 1) or 1) * 8
    out = np.empty_like(frames)
    msg = torch.from_numpy(bits.astype(np.float32))[None].to(device)
    for s in range(0, len(frames), win):
        x = torch.from_numpy(frames[s:s + win]).permute(0, 3, 1, 2).float().div(255).to(device)
        with torch.no_grad():
            y = m.embed(x, msgs=msg, is_video=True, lowres_attenuation=True)["imgs_w"]
        out[s:s + win] = y.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return out


def vs_decode(a, frames: np.ndarray) -> np.ndarray:
    import torch

    logits = []
    for s in range(0, len(frames), 16):
        x = torch.from_numpy(frames[s:s + 16]).permute(0, 3, 1, 2).float().div(255).to(a.device)
        with torch.no_grad():
            logits.append(a.m.detect(x, is_video=True)["preds"][:, 1:].float().cpu())
    return (torch.cat(logits).numpy().mean(0) > 0).astype(np.uint8)


def k_threshold(n: int, p: float = 1e-6) -> int:
    """Smallest k with P(Binomial(n, 1/2) >= k) <= p: bit-agreement needed for a key match."""
    from scipy import stats

    for k in range(n // 2, n + 1):
        if stats.binom.sf(k - 1, n, 0.5) <= p:
            return k
    return n + 1


# ------------------------------------------------------------------ render
def render_image(n: int, device: str) -> None:
    man = _man()["image"]
    srcs = man["sets"]["pos"][:n]
    tm = trustmark(device)
    out = WM / "image" / "marked"
    out.mkdir(parents=True, exist_ok=True)
    keys = {}
    for i, s in enumerate(srcs):
        dst = out / f"{s}.png"
        bits = payload(tm.nbits, i, salt="wm-image")
        data, _ = tm._codeword(bits)
        keys[s] = data
        if not dst.exists():
            from PIL import Image

            x = load_rgb(CORPUS / man["items"][s]["path"], 1024)
            Image.fromarray(to_u8(tm.embed(x, bits))).save(dst)
    (WM / "image" / "keys.json").write_text(json.dumps(keys))
    register_set("image", "wm_ref", [(s, out / f"{s}.png", "none") for s in srcs])
    register_set("image", "wm_pos", [(f"{s}|{a}", out / f"{s}.png", a) for a in attacks.COPY_ATTACKS for s in srcs])
    print(f"image: {len(srcs)} marked; wm_pos = {len(srcs) * len(attacks.COPY_ATTACKS)} queries")


def render_audio(n: int, device: str) -> None:
    import soundfile as sf

    import bench_audio

    man = _man()["audio"]
    srcs = man["sets"]["reg"][:n]
    a = audioseal(device)
    out = WM / "audio" / "marked"
    out.mkdir(parents=True, exist_ok=True)
    keys = {}
    for i, s in enumerate(srcs):
        bits = payload(16, i, salt="wm-audio")
        keys[s] = bits.tolist()
        dst = out / f"{s}.flac"
        if not dst.exists():
            y, sr = sf.read(CORPUS / man["items"][s]["path"], dtype="float32")
            sf.write(dst, np.clip(a.embed(y, sr, bits), -1, 1), sr, subtype="PCM_16")
    (WM / "audio" / "keys.json").write_text(json.dumps(keys))
    bench_audio._render_init(bench_audio.babble_pool(man))
    items = []
    for atk in attacks.AUDIO_COPY_ATTACKS:
        for s in srcs:
            dst = WM / "audio" / "pos" / f"{s}__{atk}.flac"
            bench_audio._render((f"{s}|{atk}", out / f"{s}.flac", atk, dst))
            items.append((f"{s}|{atk}", dst, atk))
    register_set("audio", "wm_ref", [(s, out / f"{s}.flac", "ref") for s in srcs])
    register_set("audio", "wm_pos", items)
    print(f"audio: {len(srcs)} marked; {len(items)} queries")


def render_video(n: int, device: str) -> None:
    import bench_video
    from fingerprints_video import sample_frames  # noqa: F401  (PyAV import check)

    man = _man()["video"]
    srcs = man["sets"]["reg"][:n]
    a = videoseal(device)
    out = WM / "video" / "marked"
    out.mkdir(parents=True, exist_ok=True)
    keys = {}
    for i, s in enumerate(srcs):
        bits = payload(VIDEOSEAL_BITS, i, salt="wm-video")
        keys[s] = bits.tolist()
        dst = out / f"{s}.mp4"
        if dst.exists():
            continue
        frames, fps = _read_all(CORPUS / man["items"][s]["path"])
        marked = vs_embed(a, frames, bits, device)
        _write_h264(marked, fps, dst)
    (WM / "video" / "keys.json").write_text(json.dumps(keys))
    dist = man["sets"]["dist"]
    items = []
    for atk in attacks.VIDEO_COPY_ATTACKS:
        for s in srcs:
            bg = CORPUS / man["items"][dist[int(rng_for("video-bg", s).integers(0, len(dist)))]]["path"]
            dst = WM / "video" / "pos" / f"{s}__{atk}.mp4"
            bench_video._render((f"{s}|{atk}", out / f"{s}.mp4", atk, dst, bg, man["clip_seconds"]))
            items.append((f"{s}|{atk}", dst, atk))
    register_set("video", "wm_ref", [(s, out / f"{s}.mp4", "ref") for s in srcs])
    register_set("video", "wm_pos", items)
    print(f"video: {len(srcs)} marked; {len(items)} queries")


def _read_all(path: Path) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(s)]), fps


def _write_h264(frames: np.ndarray, fps: float, dst: Path) -> None:
    """Delivery encode of the marked clip: H.264 High CRF 18 yuv420p (the watermark bench's ship profile)."""
    h, w = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-profile:v", "high", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    assert p.wait() == 0


# ------------------------------------------------------------------ decode
def decode_image(device: str, n_neg: int) -> None:
    import bench_image

    man = _man()["image"]
    tm = trustmark(device)
    keys = json.loads((WM / "image" / "keys.json").read_text())
    bench_image_bg = [str(CORPUS / man["items"][i]["path"]) for i in man["sets"]["bg"]]
    attacks.BACKGROUNDS = [load_rgb(Path(p), 1024) for p in bench_image_bg]
    rec = {"pos": {}, "neg": {}}
    for s in keys:
        for atk in attacks.COPY_ATTACKS:
            x = bench_image.make_image(str(WM / "image" / "marked" / f"{s}.png"), atk, f"{s}|{atk}")
            d = tm.decode(x)
            rec["pos"][f"{s}|{atk}"] = {"present": d["native_detect"], "key_ok": bool(d["native_detect"]) and d["native_payload"] == keys[s]}
    for s in man["sets"]["neg"][:n_neg]:
        for atk in attacks.COPY_ATTACKS:
            x = bench_image.make_image(str(CORPUS / man["items"][s]["path"]), atk, f"{s}|{atk}")
            d = tm.decode(x)
            # a false key = the BCH layer accepts a payload on an unmarked image
            rec["neg"][f"{s}|{atk}"] = {"present": bool(d["native_detect"])}
    write_json(WM / "image" / "decode.json", rec)
    print("image decode:", sum(v["key_ok"] for v in rec["pos"].values()), "/", len(rec["pos"]), "keys;",
          sum(v["present"] for v in rec["neg"].values()), "/", len(rec["neg"]), "false")


def decode_audio(device: str, n_neg: int) -> None:
    import soundfile as sf

    import bench_audio

    man = _man()["audio"]
    a = audioseal(device)
    keys = json.loads((WM / "audio" / "keys.json").read_text())
    rec = {"pos": {}, "neg": {}}
    for s, bits in keys.items():
        for atk in attacks.AUDIO_COPY_ATTACKS:
            y, sr = sf.read(WM / "audio" / "pos" / f"{s}__{atk}.flac", dtype="float32")
            d = a.decode(y, sr)
            rec["pos"][f"{s}|{atk}"] = {"score": d["native_score"], "detect": bool(d["native_score"] >= 0.5),
                                        "key_ok": bool(d["native_score"] >= 0.5) and d["bits"].tolist() == bits}
    for s in man["sets"]["neg"][:n_neg]:
        for atk in attacks.AUDIO_COPY_ATTACKS:
            y, sr = sf.read(bench_audio.QDIR / "neg" / f"{s}__{atk}.flac", dtype="float32")
            d = a.decode(y, sr)
            rec["neg"][f"{s}|{atk}"] = {"score": d["native_score"], "present": bool(d["native_score"] >= 0.5),
                                        "bits": d["bits"].tolist()}
    write_json(WM / "audio" / "decode.json", rec)
    print("audio decode:", sum(v["key_ok"] for v in rec["pos"].values()), "/", len(rec["pos"]))


def decode_video(device: str, n_neg: int) -> None:
    import bench_video

    man = _man()["video"]
    a = videoseal(device)
    keys = json.loads((WM / "video" / "keys.json").read_text())
    k = k_threshold(VIDEOSEAL_BITS)
    rec = {"pos": {}, "neg": {}, "k_threshold": k}
    for s, bits in keys.items():
        for atk in attacks.VIDEO_COPY_ATTACKS:
            fr, _ = _read_all(WM / "video" / "pos" / f"{s}__{atk}.mp4")
            got = vs_decode(a, fr)
            agree = int((got == np.array(bits)).sum())
            rec["pos"][f"{s}|{atk}"] = {"agree": agree, "key_ok": agree >= k}
    # False keys on unmarked video: agreement with every registered key; any >= k is a false key.
    allk = np.array(list(keys.values()))
    for s in man["sets"]["neg"][:n_neg]:
        for atk in attacks.VIDEO_COPY_ATTACKS:
            fr, _ = _read_all(bench_video.QDIR / "neg" / f"{s}__{atk}.mp4")
            got = vs_decode(a, fr)
            best = int((allk == got[None]).sum(1).max())
            rec["neg"][f"{s}|{atk}"] = {"best_agree": best, "present": best >= k}
    write_json(WM / "video" / "decode.json", rec)
    print("video decode:", sum(v["key_ok"] for v in rec["pos"].values()), "/", len(rec["pos"]))


# ------------------------------------------------------------------ analyze
def _thresholds(modality: str, method: str) -> dict:
    p = RESULTS / modality / f"{method}.json"
    return json.loads(p.read_text())["thresholds"] if p.exists() else {}


def analyze_image(th_name: str) -> dict:
    import bench_image
    from fingerprints_image import IMAGE_METHODS

    man = _man()["image"]
    dec = json.loads((WM / "image" / "decode.json").read_text())
    ids_reg = json.loads((DESC / "reg" / "ids.json").read_text())
    ids_dist = json.loads((DESC / "dist" / "ids.json").read_text())
    ids_wref = json.loads((DESC / "wm_ref" / "ids.json").read_text())
    ids_wpos = json.loads((DESC / "wm_pos" / "ids.json").read_text())
    ref_ids = ids_reg + ids_dist
    rid = {r: j for j, r in enumerate(ref_ids)}
    out = {}
    for n, ctor in IMAGE_METHODS.items():
        if not all((DESC / s / f"{n}.npy").exists() for s in ("reg", "dist", "wm_ref", "wm_pos")):
            continue
        m = ctor()
        th = _thresholds("image", n).get(th_name)
        if th is None or not np.isfinite(th):
            continue
        reg, dist = np.load(DESC / "reg" / f"{n}.npy"), np.load(DESC / "dist" / f"{n}.npy")
        wref, wpos = np.load(DESC / "wm_ref" / f"{n}.npy"), np.load(DESC / "wm_pos" / f"{n}.npy")
        orig = np.concatenate([reg, dist])
        marked = orig.copy()
        widx = np.array([rid[s] for s in ids_wref])
        marked[widx] = wref if wref.ndim == 2 else wref[:, 0]
        # drift: marked asset vs its unmarked original
        drift = np.array([m.score(wref[i:i + 1], orig[widx[i]:widx[i] + 1])[0, 0] for i in range(len(widx))])
        res = {"drift_score_median": float(np.median(drift)), "drift_below_threshold": float((drift < th).mean()),
               "threshold": th}
        tidx = np.array([rid[k.split("|")[0]] for k in ids_wpos])
        for tag, index in (("registry_marked", marked), ("registry_unmarked", orig)):
            idx = bench_image._index(index, m.metric)
            S, I = bench_image._search(idx, wpos, m.metric, m.dim, 5)
            res[f"fp_ok_{tag}"] = ((I[:, 0] == tidx) & (S[:, 0] >= th)).tolist()
        res["attack"] = [k.split("|")[1] for k in ids_wpos]
        res["source"] = [k.split("|")[0] for k in ids_wpos]
        res["wm_ok"] = [dec["pos"][k]["key_ok"] for k in ids_wpos]
        res["summary"] = _complement(res["attack"], np.array(res["wm_ok"]), np.array(res["fp_ok_registry_marked"]),
                                     list(attacks.COPY_ATTACKS))
        res["summary_unmarked_registry"] = _complement(res["attack"], np.array(res["wm_ok"]),
                                                       np.array(res["fp_ok_registry_unmarked"]), list(attacks.COPY_ATTACKS))
        res["method"] = m.meta()
        out[n] = res
    wm_fp = np.array([v["present"] for v in dec["neg"].values()])
    return {"methods": out, "wm_false_key_rate": float(wm_fp.mean()), "wm_false_key_n": int(len(wm_fp)),
            "watermark": {"model": "TrustMark", "variant": TRUSTMARK[0], "strength": TRUSTMARK[1], "ecc": "BCH_5"}}


def _complement(atk: list[str], wm: np.ndarray, fp: np.ndarray, order: list[str]) -> dict:
    from analysis_stats import cp_interval

    atk = np.array(atk)
    s = {}
    for a in order + ["_all"]:
        m = np.ones(len(atk), bool) if a == "_all" else atk == a
        if not m.any():
            continue
        n = int(m.sum())
        e = int((wm[m] | fp[m]).sum())
        s[a] = {"n": n, "wm": float(wm[m].mean()), "fp": float(fp[m].mean()), "either": e / n,
                "both": float((wm[m] & fp[m]).mean()), "wm_only": float((wm[m] & ~fp[m]).mean()),
                "fp_only": float((~wm[m] & fp[m]).mean()), "either_ci": cp_interval(e, n)}
    return s


def analyze_timebased(modality: str, th_name: str) -> dict:
    """Audio / video: systems search wm_pos against a registry holding the marked references."""
    if modality == "audio":
        from fingerprints_audio import AUDIO_METHODS as METHODS

        order = list(attacks.AUDIO_COPY_ATTACKS)
    else:
        from fingerprints_video import TMK, VIDEO_METHODS as METHODS

        order = list(attacks.VIDEO_COPY_ATTACKS)
    man = _man()[modality]
    dec = json.loads((WM / modality / "decode.json").read_text())
    pre = f"{modality}_"
    ref_ids = man["sets"]["reg"] + man["sets"]["dist"]
    rid = {r: j for j, r in enumerate(ref_ids)}
    out = {}
    for n, ctor in METHODS.items():
        paths = [DESC / f"{pre}{s}" / f"{n}.pkl" for s in ("reg", "dist", "wm_ref", "wm_pos")]
        if not all(p.exists() for p in paths):
            continue
        th = _thresholds(modality, n).get(th_name)
        if th is None or not np.isfinite(th):
            continue
        reg, dist, wref, wpos = (pickle.loads(p.read_bytes()) for p in paths)
        ids_wref = json.loads((DESC / f"{pre}wm_ref" / "ids.json").read_text())
        ids_wpos = json.loads((DESC / f"{pre}wm_pos" / "ids.json").read_text())
        refs = list(reg) + list(dist)
        for s, d in zip(ids_wref, wref):
            refs[rid[s]] = d
        tidx = np.array([rid[k.split("|")[0]] for k in ids_wpos])
        if n == "TMK+PDQF":
            sc = TMK.batch_scores([Path(p) for p in wpos], [Path(p) for p in refs])
            best = {}
            for (a, b), (_, s2) in sc.items():
                if s2 > best.get(a, (-9, -1))[0]:
                    best[a] = (s2, refs.index(b))
            fp_ok = np.array([best.get(q, (-9, -1))[1] == t and best.get(q, (-9, -1))[0] >= th for q, t in zip(wpos, tidx)])
        else:
            m = ctor()
            m.load("cpu")
            m.build(refs)
            top = [m.search(q, 1) for q in wpos]
            fp_ok = np.array([I[0] == t and S[0] >= th for (I, S), t in zip(top, tidx)])
        atk = [k.split("|")[1] for k in ids_wpos]
        wm_ok = np.array([dec["pos"][k]["key_ok"] for k in ids_wpos])
        out[n] = {"summary": _complement(atk, wm_ok, fp_ok, order), "threshold": th,
                  "attack": atk, "source": [k.split("|")[0] for k in ids_wpos],
                  "wm_ok": wm_ok.tolist(), "fp_ok": [bool(x) for x in fp_ok]}
        if modality == "video":
            # agreement with the asset's own key, so exact 256-bit recovery can be reported too
            out[n]["wm_agree"] = [int(dec["pos"][k]["agree"]) for k in ids_wpos]
    fk = np.array([v["present"] for v in dec["neg"].values()])
    res = {"methods": out, "wm_false_key_rate": float(fk.mean()) if len(fk) else float("nan"), "wm_false_key_n": int(len(fk))}
    if modality == "audio":
        # AudioSeal's 16-bit message is the registry key: a false key in the identification sense is a
        # detection whose message equals any registered key (the detection-only rate above bounds it).
        keys = {tuple(b) for b in json.loads((WM / "audio" / "keys.json").read_text()).values()}
        idf = np.array([v["present"] and tuple(v.get("bits", ())) in keys for v in dec["neg"].values()])
        res["wm_false_key_registry_rate"] = float(idf.mean()) if len(idf) else float("nan")
        res["wm_n_registered_keys"] = len(keys)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "decode", "analyze"])
    ap.add_argument("modality", nargs="?", default="image", choices=["image", "audio", "video"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-neg", type=int, default=500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", default="pair@1e-07")
    ap.add_argument("--av-threshold", default="query@0.01")
    args = ap.parse_args()
    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    n = args.n or N_DEFAULT[args.modality]
    t0 = time.time()
    if args.stage == "render":
        {"image": render_image, "audio": render_audio, "video": render_video}[args.modality](n, dev)
    elif args.stage == "decode":
        {"image": decode_image, "audio": decode_audio, "video": decode_video}[args.modality](dev, args.n_neg)
    else:
        res = {"host": host_info(), "threshold": args.threshold}
        if (WM / "image" / "decode.json").exists():
            res["image"] = analyze_image(args.threshold)
        for mod in ("audio", "video"):
            if (WM / mod / "decode.json").exists():
                # the smaller audio/video calibration sets resolve query-level 1%, not pair 1e-7
                res[mod] = analyze_timebased(mod, args.av_threshold)
        res["av_threshold"] = args.av_threshold
        write_json(RESULTS / "wmcombo" / "wmcombo.json", res)
        for mod in ("image", "audio", "video"):
            for n_, r in res.get(mod, {}).get("methods", {}).items():
                a = r["summary"]["_all"]
                print(f"{mod:5s} {n_:18s} wm={a['wm']:.3f} fp={a['fp']:.3f} either={a['either']:.3f}")
    print(f"{args.stage} {args.modality}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
