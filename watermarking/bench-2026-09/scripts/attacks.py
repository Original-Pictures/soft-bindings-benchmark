"""Deterministic distortions applied to marked media before decoding.

Images: float32 HxWx3 in [0,1] -> float32 (any size; geometry attacks change size).
Audio:  float32 mono at sr -> float32 mono at the same sr (codec round trips via ffmpeg).
Video:  file -> file via ffmpeg (see bench_video).
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


def _u8(x: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.round(x * 255), 0, 255).astype(np.uint8))


def _f(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def jpeg(q: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        _u8(x).save(buf, "JPEG", quality=q, subsampling=2 if q < 90 else 0)
        return _f(Image.open(buf))
    return f


def webp(q: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        _u8(x).save(buf, "WEBP", quality=q, method=4)
        return _f(Image.open(buf))
    return f


def resize(s: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        im = _u8(x)
        return _f(im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BICUBIC))
    return f


def center_crop(keep: float) -> Callable[[np.ndarray], np.ndarray]:
    """keep = fraction of each side retained (0.5 -> 25% of the area)."""
    def f(x: np.ndarray) -> np.ndarray:
        h, w = x.shape[:2]
        ch, cw = round(h * keep), round(w * keep)
        t, l = (h - ch) // 2, (w - cw) // 2
        return x[t:t + ch, l:l + cw].copy()
    return f


def blur(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return _f(_u8(x).filter(ImageFilter.GaussianBlur(sigma)))
    return f


def brightness(k: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(x: np.ndarray) -> np.ndarray:
        return np.clip(x * k, 0, 1).astype(np.float32)
    return f


def identity(x: np.ndarray) -> np.ndarray:
    return x


IMAGE_ATTACKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": identity,
    "jpeg90": jpeg(90), "jpeg75": jpeg(75), "jpeg60": jpeg(60), "jpeg50": jpeg(50),
    "webp80": webp(80), "webp60": webp(60),
    "resize0.75": resize(0.75), "resize0.5": resize(0.5),
    "crop90": center_crop(0.9), "crop70": center_crop(0.7), "crop50": center_crop(0.5),
    "blur1": blur(1.0), "bright1.2": brightness(1.2), "bright0.8": brightness(0.8),
}


# ---------------------------------------------------------------- audio
def _ffmpeg_roundtrip(y: np.ndarray, sr: int, codec_args: list[str], ext: str) -> np.ndarray:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = Path(d) / "a.wav", Path(d) / f"b.{ext}", Path(d) / "c.wav"
        sf.write(src, np.clip(y, -1, 1), sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *codec_args, str(enc)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dec)], check=True)
        out, _ = sf.read(dec, dtype="float32")
    # Codecs add priming delay/padding; align by cross-correlation on the first 2 s.
    return _align(y, out)


def _align(ref: np.ndarray, out: np.ndarray, max_shift: int = 4096) -> np.ndarray:
    seg = min(len(ref), len(out), 32768)
    best, best_s = -np.inf, 0
    r = ref[:seg]
    for s in range(0, max_shift, 1):
        if s + seg > len(out):
            break
        v = float(np.dot(r, out[s:s + seg]))
        if v > best:
            best, best_s = v, s
    out = out[best_s:]
    if len(out) < len(ref):
        out = np.pad(out, (0, len(ref) - len(out)))
    return out[: len(ref)].astype(np.float32)


def mp3(kbps: int):
    return lambda y, sr: _ffmpeg_roundtrip(y, sr, ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"], "mp3")


def aac(kbps: int):
    return lambda y, sr: _ffmpeg_roundtrip(y, sr, ["-c:a", "aac", "-b:a", f"{kbps}k"], "m4a")


def resample_rt(mid: int):
    def f(y: np.ndarray, sr: int) -> np.ndarray:
        import soxr

        return soxr.resample(soxr.resample(y, sr, mid, quality="HQ"), mid, sr, quality="HQ")[: len(y)].astype(np.float32)
    return f


def noise(snr_db: float):
    def f(y: np.ndarray, sr: int) -> np.ndarray:
        rng = np.random.default_rng(7)
        p = np.mean(y.astype(np.float64) ** 2) / (10 ** (snr_db / 10))
        return (y + rng.normal(0, np.sqrt(p), len(y))).astype(np.float32)
    return f


def gain(db: float):
    return lambda y, sr: np.clip(y * (10 ** (db / 20)), -1, 1).astype(np.float32)


AUDIO_ATTACKS = {
    "none": lambda y, sr: y,
    "mp3_320": mp3(320), "mp3_128": mp3(128), "mp3_64": mp3(64),
    "aac_256": aac(256), "aac_128": aac(128), "aac_64": aac(64),
    "resample_22k": resample_rt(22050), "resample_8k": resample_rt(8000),
    "noise_30db": noise(30), "gain_-6db": gain(-6),
}
