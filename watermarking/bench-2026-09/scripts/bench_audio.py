"""Audio bake-off runner (one JSON per config in $BENCH_RESULTS/audio/).

Usage: python bench_audio.py [--profile full|smoke] [--only substr,...] [--device cuda]
                             [--manifest path] [--limit N] [--cpu-latency]
SilentCipher needs its own torch<=2.0 env: run with /opt/bench/.venv-sc/bin/python --only SilentCipher.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

import attacks
import metrics
from common import BENCH_DIR, CORPUS, RESULTS, Timer, bit_accuracy, device_name, host_info, payload, write_json
from models_audio import audio_configs

FPR_ATTACKS = ["none", "mp3_128", "resample_22k"]
FPR_EXTRA_KEYS = 50


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


def gallery_items(items: list[dict]) -> set[str]:
    out = set()
    for want in ("music", "speech"):
        cands = [i["path"] for i in items if i["set"] == want and i["sr"] == 44100]
        if cands:
            out.add(sorted(cands)[0])
    return out


def save_spectrogram(name: str, item: str, y: np.ndarray, yw: np.ndarray, sr: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import librosa
    import matplotlib.pyplot as plt

    out = BENCH_DIR / "thumbs" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    n = min(len(y), len(yw), sr * 6)
    S = lambda a: librosa.amplitude_to_db(np.abs(librosa.stft(a[:n], n_fft=2048, hop_length=512)), ref=np.max(np.abs(librosa.stft(y[:n], n_fft=2048, hop_length=512))))  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), dpi=110)
    for ax, data, title in ((axes[0], S(y), "original"), (axes[1], S(yw - y[: len(yw)]), "watermark residual")):
        ax.imshow(data, origin="lower", aspect="auto", vmin=-100, vmax=0, cmap="magma",
                  extent=[0, n / sr, 0, sr / 2000])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("kHz", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"{name} - {Path(item).stem[:40]}", fontsize=8)
    fig.tight_layout()
    safe = name.replace("@", "_at_")
    fig.savefig(out / f"{safe}__{Path(item).parent.name}.png")
    plt.close(fig)


def run_config(adapter, items: list[dict], device: str) -> dict:
    t0 = time.perf_counter()
    adapter.load(device)
    load_s = time.perf_counter() - t0
    gal = gallery_items(items)
    records, fpr_raw, emb_ms_per_s, dec_ms_per_s = [], [], [], []
    for idx, it in enumerate(items):
        y, sr = load_audio(CORPUS / it["path"])
        bits = payload(max(adapter.nbits, 1), idx, salt=f"audio{adapter.nbits}")[: adapter.nbits]
        rec: dict = {"item": it["path"], "set": it["set"], "sr": sr, "seconds": len(y) / sr}
        try:
            with Timer(device) as te:
                yw = adapter.embed(y, sr, bits)
            if idx > 0:
                emb_ms_per_s.append(te.elapsed * 1000 / (len(y) / sr))
            yw = np.clip(np.round(yw * 32767) / 32767, -1, 1).astype(np.float32)  # stored as PCM16
            rec["snr"] = metrics.snr(y, yw)
            rec["si_snr"] = metrics.si_snr(y, yw)
            rec["pesq"] = metrics.pesq_wb(y, yw, sr)
            rec["stoi"] = metrics.stoi(y, yw, sr)
            rec["dlufs"] = metrics.delta_lufs(y, yw, sr)
            rec["spec_diff_db"] = metrics.spectral_diff_db(y, yw, sr)
            if it["path"] in gal:
                save_spectrogram(adapter.name, it["path"], y, yw, sr)
            rec["attacks"] = {}
            for aname, fn in attacks.AUDIO_ATTACKS.items():
                x = fn(yw, sr)
                with Timer(device) as td:
                    d = adapter.decode(x, sr)
                if aname == "none" and idx > 0:
                    dec_ms_per_s.append(td.elapsed * 1000 / (len(y) / sr))
                r = {"native_detect": d.get("native_detect"), "native_score": d.get("native_score")}
                if adapter.nbits:
                    got = d.get("bits")
                    r["bit_acc"] = bit_accuracy(bits, got)
                    r["matches"] = int((np.asarray(got)[: adapter.nbits] == bits).sum()) if got is not None and len(got) >= adapter.nbits else 0
                rec["attacks"][aname] = r
            for aname in FPR_ATTACKS:
                d = adapter.decode(attacks.AUDIO_ATTACKS[aname](y, sr), sr)
                fpr_raw.append({"item": idx, "attack": aname, "native_detect": d.get("native_detect"),
                                "native_score": d.get("native_score"),
                                "bits": None if d.get("bits") is None else np.asarray(d["bits"]).astype(int).tolist()})
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            traceback.print_exc()
        records.append(rec)
        print(f"  {adapter.name} [{idx + 1}/{len(items)}] {it['path']} snr={rec.get('snr', float('nan')):.1f} "
              f"pesq={rec.get('pesq', float('nan')):.2f} mp3_128={rec.get('attacks', {}).get('mp3_128', {})}", flush=True)
    fpr = []
    for fb in fpr_raw:
        entry = {k: fb[k] for k in ("item", "attack", "native_detect", "native_score")}
        if adapter.nbits:
            exp = [payload(adapter.nbits, fb["item"], salt=f"audio{adapter.nbits}")] + \
                  [payload(adapter.nbits, 10_000 + k, salt="fpr") for k in range(FPR_EXTRA_KEYS)]
            got = None if fb["bits"] is None else np.asarray(fb["bits"], dtype=np.uint8)
            entry["matches"] = [int((got[: adapter.nbits] == e).sum()) if got is not None and len(got) >= adapter.nbits else 0 for e in exp]
        fpr.append(entry)
    return {"meta": adapter.meta(), "device": device, "load_s": load_s,
            "embed_ms_per_audio_s_median": float(np.median(emb_ms_per_s)) if emb_ms_per_s else None,
            "decode_ms_per_audio_s_median": float(np.median(dec_ms_per_s)) if dec_ms_per_s else None,
            "records": records, "fpr": fpr}


def cpu_latency(adapter, items: list[dict]) -> dict:
    import torch

    torch.set_num_threads(4)
    adapter.load("cpu")
    emb, dec = [], []
    for idx, it in enumerate(items[:4]):
        y, sr = load_audio(CORPUS / it["path"])
        bits = payload(max(adapter.nbits, 1), idx, salt=f"audio{adapter.nbits}")[: adapter.nbits]
        with Timer("cpu") as te:
            yw = adapter.embed(y, sr, bits)
        with Timer("cpu") as td:
            adapter.decode(yw, sr)
        if idx > 0:
            emb.append(te.elapsed * 1000 / (len(y) / sr))
            dec.append(td.elapsed * 1000 / (len(y) / sr))
    return {"embed_ms_per_audio_s_median": float(np.median(emb)), "decode_ms_per_audio_s_median": float(np.median(dec)),
            "threads": 4, "host": host_info()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--manifest", default=str(BENCH_DIR / "corpus_manifest.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu-latency", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: run configs with index %% n == i (parallel workers)")
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))
    device = device_name(args.device)
    items = json.loads(Path(args.manifest).read_text())["audio"]
    if args.limit:
        by: dict = {}
        for i in items:
            by.setdefault((i["set"], i["sr"]), []).append(i)
        items = [x for v in by.values() for x in v[: args.limit]]
    for ci, adapter in enumerate(audio_configs(args.profile)):
        if ci % shard_n != shard_i:
            continue
        if args.only and not any(s in adapter.name for s in args.only.split(",")):
            continue
        safe = adapter.name.replace("@", "_at_")
        if args.cpu_latency:
            out = RESULTS / "audio" / "cpu_latency" / f"{safe}.json"
            if out.exists():
                continue
            lat = [i for i in items if i["sr"] == 44100][:5]
            try:
                write_json(out, {"meta": adapter.meta(), **cpu_latency(adapter, lat)})
            except Exception as exc:
                write_json(out, {"meta": adapter.meta(), "error": repr(exc)[:500]})
            continue
        out = RESULTS / "audio" / f"{safe}.json"
        if out.exists():
            print("skip", adapter.name)
            continue
        print(f"== {adapter.name} on {device}", flush=True)
        t0 = time.time()
        try:
            res = run_config(adapter, items, device)
        except Exception as exc:
            traceback.print_exc()
            res = {"meta": adapter.meta(), "error": f"{type(exc).__name__}: {exc}"[:800], "device": device}
        res["host"] = host_info()
        res["wall_s"] = time.time() - t0
        write_json(out, res)


if __name__ == "__main__":
    main()
