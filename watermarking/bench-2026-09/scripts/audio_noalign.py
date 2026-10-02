"""Audio watermark decoding after a codec round trip, with and without alignment.

The benchmark aligns every MP3/AAC round trip to the original by cross-correlation before
decoding (attacks._align), so that encoder priming delay does not register as watermark
loss. A verifier holding only the received file has no original to align to. This script
embeds each audio configuration exactly as bench_audio.run_config does, applies the six
codec round trips, and decodes each output twice: aligned (the benchmark's protocol, which
also checks that this host reproduces the published numbers) and unaligned (the decoded
codec output cut or padded to the original length, with no offset search).

    python audio_noalign.py [--only AudioSeal,WavMark] [--device cpu]

Writes $BENCH_RESULTS/audio_noalign/<config>.json.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import subprocess
from pathlib import Path

import numpy as np

import attacks
from bench_audio import audio_configs, load_audio
from common import BENCH_DIR, CORPUS, RESULTS, bit_accuracy, device_name, host_info, payload, write_json

CODECS = {
    "mp3_320": (["-c:a", "libmp3lame", "-b:a", "320k"], "mp3"), "mp3_128": (["-c:a", "libmp3lame", "-b:a", "128k"], "mp3"),
    "mp3_64": (["-c:a", "libmp3lame", "-b:a", "64k"], "mp3"), "aac_256": (["-c:a", "aac", "-b:a", "256k"], "m4a"),
    "aac_128": (["-c:a", "aac", "-b:a", "128k"], "m4a"), "aac_64": (["-c:a", "aac", "-b:a", "64k"], "m4a"),
}


def roundtrip_raw(y: np.ndarray, sr: int, codec_args: list[str], ext: str) -> np.ndarray:
    """The codec round trip of attacks._ffmpeg_roundtrip, before alignment."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = Path(d) / "a.wav", Path(d) / f"b.{ext}", Path(d) / "c.wav"
        sf.write(src, np.clip(y, -1, 1), sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *codec_args, str(enc)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dec)], check=True)
        out, _ = sf.read(dec, dtype="float32")
    return out


def unaligned(ref: np.ndarray, out: np.ndarray) -> np.ndarray:
    out = out[: len(ref)]
    return np.pad(out, (0, len(ref) - len(out))).astype(np.float32) if len(out) < len(ref) else out.astype(np.float32)


def run(adapter, items: list[dict], device: str, excluded: set[str]) -> dict:
    adapter.load(device)
    recs = []
    for idx, it in enumerate(items):
        # Same payload and index as bench_audio.run_config, including excluded items, so the
        # payloads match the published run item for item.
        if it["path"] in excluded:
            continue
        y, sr = load_audio(CORPUS / it["path"])
        bits = payload(max(adapter.nbits, 1), idx, salt=f"audio{adapter.nbits}")[: adapter.nbits]
        yw = adapter.embed(y, sr, bits)
        yw = np.clip(np.round(yw * 32767) / 32767, -1, 1).astype(np.float32)
        rec = {"item": it["path"], "set": it["set"], "sr": sr, "codecs": {}}
        for name, (args, ext) in CODECS.items():
            raw = roundtrip_raw(yw, sr, args, ext)
            r = {"delay_samples": None}
            for mode, x in (("aligned", attacks._align(yw, raw)), ("unaligned", unaligned(yw, raw))):
                d = adapter.decode(x, sr)
                e = {"native_detect": d.get("native_detect"), "native_score": d.get("native_score")}
                if adapter.nbits:
                    got = d.get("bits")
                    e["bit_acc"] = bit_accuracy(bits, got)
                    e["matches"] = int((np.asarray(got)[: adapter.nbits] == bits).sum()) if got is not None and len(got) >= adapter.nbits else 0
                r[mode] = e
            rec["codecs"][name] = r
        recs.append(rec)
        print(f"  {adapter.name} [{idx + 1}/{len(items)}] mp3_128 aligned={rec['codecs']['mp3_128']['aligned'].get('bit_acc')} "
              f"unaligned={rec['codecs']['mp3_128']['unaligned'].get('bit_acc')}", flush=True)
    return {"meta": adapter.meta(), "device": device, "host": host_info(), "records": recs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    items = json.loads((BENCH_DIR / "corpus_manifest.json").read_text())["audio"]
    excluded = set(json.loads((BENCH_DIR / "results.json").read_text())["corpus_excluded"].get("audio", []))
    dev = device_name(args.device)
    for adapter in audio_configs("full"):
        if args.only and not any(s in adapter.name for s in args.only.split(",")):
            continue
        out = RESULTS / "audio_noalign" / f"{adapter.name.replace('@', '_at_')}.json"
        if out.exists():
            continue
        write_json(out, run(adapter, items, dev, excluded))


if __name__ == "__main__":
    main()
