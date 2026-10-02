"""Deterministic distortions applied to registered media to form queries.

Copied from the 2026-09 watermark bench (same codec round trips and alignment) and
extended for copy detection with the AugLy transform families that generated DISC21
(overlays, screenshots, memes, perspective) plus re-capture and social-media chains.

Images: float32 HxWx3 in [0,1] -> float32 (any size). Every image attack is
`fn(x, rng)`; `rng` is a per-(attack, item) generator from common.rng_for so random
parameters (crop offset, overlay position, emoji) are reproducible.
Audio:  float32 mono at sr -> float32 mono (sr may be unchanged; length may change).
Video:  see fingerprints_video / bench_video (ffmpeg filter chains).
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

# AugLy 1.0.0 (last release) still calls the removed alias np.float in its perspective
# helper; restoring the alias keeps its maths unchanged (np.float was builtin float).
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


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


# ---------------------------------------------------------------- image, copy-detection ladder
def _aug(fn_name: str, **kw):
    def f(x: np.ndarray, rng=None) -> np.ndarray:
        import augly.image as imaugs

        return _f(getattr(imaugs, fn_name)(_u8(x), **kw))
    return f


def _plain(g):
    """Wrap a watermark-bench attack (x -> x) into the (x, rng) form."""
    return lambda x, rng=None: g(x)


def crop_offset(keep: float):
    def f(x, rng):
        h, w = x.shape[:2]
        ch, cw = round(h * keep), round(w * keep)
        t, l = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        return x[t:t + ch, l:l + cw].copy()
    return f


def squash(ax: float):
    def f(x, rng=None):
        im = _u8(x)
        return _f(im.resize((max(8, round(im.width * ax)), im.height), Image.BICUBIC))
    return f


def rotate(deg: float):
    def f(x, rng=None):
        return _f(_u8(x).rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0)))
    return f


def gauss_noise(sigma: float):
    def f(x, rng):
        return np.clip(x + rng.normal(0, sigma, x.shape).astype(np.float32), 0, 1)
    return f


def gamma(g: float):
    return lambda x, rng=None: np.power(np.clip(x, 0, 1), g).astype(np.float32)


def text_overlay(x, rng):
    import augly.image as imaugs

    return _f(imaugs.overlay_text(_u8(x), text=[int(v) for v in rng.integers(0, 1000, 12)], font_size=0.12,
                                  color=tuple(int(c) for c in rng.integers(0, 256, 3)),
                                  x_pos=float(rng.uniform(0, 0.3)), y_pos=float(rng.uniform(0.1, 0.8)),
                                  font_file=_font("NotoSans-Regular.ttf")))


def _font(name: str) -> str:
    import augly
    from pathlib import Path as _P

    root = _P(augly.__file__).parent / "assets" / "fonts"
    cands = sorted(root.glob("NotoSans*Regular*.ttf")) or sorted(root.glob("*.ttf"))
    return str(root / name) if (root / name).exists() else str(cands[0])


def emoji_overlay(size: float):
    def f(x, rng):
        import augly
        import augly.image as imaugs
        from pathlib import Path as _P

        pool = sorted((_P(augly.__file__).parent / "assets" / "twemojis").rglob("*.png"))
        e = pool[int(rng.integers(0, len(pool)))]
        return _f(imaugs.overlay_emoji(_u8(x), emoji_path=str(e), emoji_size=size,
                                       x_pos=float(rng.uniform(0, 1 - size)), y_pos=float(rng.uniform(0, 1 - size))))
    return f


def meme(x, rng):
    """AugLy meme_format layout (white caption band above the image, bold centred text),
    re-implemented with Pillow>=10 text metrics because AugLy 1.0.0 calls the removed
    FreeTypeFont.getsize_multiline."""
    from PIL import ImageDraw, ImageFont

    im = _u8(x)
    band = max(40, im.height // 5)
    out = Image.new("RGB", (im.width, im.height + band), (255, 255, 255))
    out.paste(im, (0, band))
    text = "WHEN THE PHOTO IS NOT YOURS"
    size = band
    while size > 6:
        font = ImageFont.truetype(_font("Raleway-ExtraBold.ttf"), size)
        l, t, r, b = ImageDraw.Draw(out).textbbox((0, 0), text, font=font)
        if r - l <= 0.9 * im.width and b - t <= 0.7 * band:
            break
        size -= 2
    ImageDraw.Draw(out).text(((im.width - (r - l)) / 2 - l, (band - (b - t)) / 2 - t), text, font=font, fill=(0, 0, 0))
    return _f(out)


def screenshot(x, rng):
    import augly.image as imaugs

    return _f(imaugs.overlay_onto_screenshot(_u8(x)))


BACKGROUNDS: list[np.ndarray] = []  # set by the runner (distractor images never used as queries)


def onto_background(size: float):
    def f(x, rng):
        import augly.image as imaugs

        bg = BACKGROUNDS[int(rng.integers(0, len(BACKGROUNDS)))] if BACKGROUNDS else np.full((768, 1024, 3), 0.5, np.float32)
        if isinstance(bg, (str, Path)):  # paths are loaded on use (keeps worker memory small)
            from common import load_rgb

            bg = load_rgb(Path(bg), 1024)
        return _f(imaugs.overlay_onto_background_image(_u8(x), background_image=_u8(bg), overlay_size=size,
                                                       x_pos=float(rng.uniform(0, 1 - size)),
                                                       y_pos=float(rng.uniform(0, 1 - size)), scale_bg=True))
    return f


def perspective(sigma: float):
    def f(x, rng):
        import augly.image as imaugs

        return _f(imaugs.perspective_transform(_u8(x), sigma=sigma, seed=int(rng.integers(0, 2**31))))
    return f


def chain(*fs):
    def f(x, rng):
        for g in fs:
            x = g(x, rng)
        return x
    return f


def recapture(x, rng):
    """Phone photo of a screen/print: perspective, defocus, sensor noise, tone curve, JPEG."""
    return chain(perspective(25), _plain(blur(1.2)), gauss_noise(0.02), gamma(0.85), _plain(jpeg(80)))(x, rng)


# name -> (fn, family). Families group the heatmap rows and the per-family summaries.
COPY_ATTACKS: dict[str, tuple] = {
    "none": (_plain(identity), "none"),
    "jpeg75": (_plain(jpeg(75)), "compression"), "jpeg50": (_plain(jpeg(50)), "compression"),
    "jpeg30": (_plain(jpeg(30)), "compression"), "webp50": (_plain(webp(50)), "compression"),
    "resize0.5": (_plain(resize(0.5)), "scale"), "resize0.25": (_plain(resize(0.25)), "scale"),
    "squash0.75": (squash(0.75), "scale"),
    "crop90": (_plain(center_crop(0.9)), "crop"), "crop70": (_plain(center_crop(0.7)), "crop"),
    "crop50": (_plain(center_crop(0.5)), "crop"), "crop30": (_plain(center_crop(0.3)), "crop"),
    "crop50off": (crop_offset(0.5), "crop"),
    "rot5": (rotate(5), "geometric"), "rot15": (rotate(15), "geometric"), "rot90": (rotate(90), "geometric"),
    "hflip": (_aug("hflip"), "geometric"), "perspective": (perspective(40), "geometric"),
    "pad25": (_aug("pad", w_factor=0.25, h_factor=0.25, color=(255, 255, 255)), "geometric"),
    "skew": (_aug("skew", skew_factor=0.4), "geometric"),
    "bright1.4": (_plain(brightness(1.4)), "photometric"), "bright0.6": (_plain(brightness(0.6)), "photometric"),
    "contrast0.5": (_aug("contrast", factor=0.5), "photometric"), "gray": (_aug("grayscale"), "photometric"),
    "sat2": (_aug("saturation", factor=2.0), "photometric"), "gamma0.6": (gamma(0.6), "photometric"),
    "blur2": (_plain(blur(2.0)), "photometric"), "noise0.05": (gauss_noise(0.05), "photometric"),
    "pixelize0.3": (_aug("pixelization", ratio=0.3), "photometric"),
    "text": (text_overlay, "overlay"), "emoji0.3": (emoji_overlay(0.3), "overlay"), "meme": (meme, "overlay"),
    "screenshot": (screenshot, "overlay"), "background0.5": (onto_background(0.5), "overlay"),
    "social": (chain(_plain(resize(0.5)), _plain(jpeg(70)), _plain(center_crop(0.9))), "chain"),
    "recapture": (recapture, "chain"),
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


# ---------------------------------------------------------------- audio, copy-detection ladder
def _ff(y: np.ndarray, sr: int, filt: str | None = None, codec: list[str] | None = None, ext: str = "wav") -> np.ndarray:
    """ffmpeg round trip: optional filter graph, optional lossy codec, back to mono float at sr.
    Unlike the watermark bench's _ffmpeg_roundtrip, no re-alignment: fingerprints must cope
    with codec delay and time-scale changes on their own."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src, mid, dec = Path(d) / "a.wav", Path(d) / f"b.{ext}", Path(d) / "c.wav"
        sf.write(src, np.clip(y, -1, 1), sr, subtype="PCM_16")
        args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
        if filt:
            args += ["-af", filt]
        args += (codec or []) + [str(mid)]
        subprocess.run(args, check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mid), "-ar", str(sr), "-ac", "1", str(dec)], check=True)
        out, _ = sf.read(dec, dtype="float32")
    return out.astype(np.float32)


def _mix(y: np.ndarray, other: np.ndarray, snr_db: float) -> np.ndarray:
    other = np.resize(other, len(y))
    py, po = np.mean(y.astype(np.float64) ** 2), np.mean(other.astype(np.float64) ** 2) + 1e-12
    return np.clip(y + other * np.sqrt(py / po / 10 ** (snr_db / 10)), -1, 1).astype(np.float32)


def a_noise(snr_db: float):
    def f(y, sr, rng):
        return _mix(y, rng.standard_normal(len(y)).astype(np.float32), snr_db)
    return f


AUDIO_BACKGROUNDS: list[np.ndarray] = []  # set by the runner (at the corpus rate)


def a_babble(snr_db: float):
    def f(y, sr, rng):
        bg = AUDIO_BACKGROUNDS[int(rng.integers(0, len(AUDIO_BACKGROUNDS)))] if AUDIO_BACKGROUNDS else rng.standard_normal(len(y))
        start = int(rng.integers(0, max(1, len(bg) - len(y))))
        return _mix(y, bg[start:start + len(y)], snr_db)
    return f


def a_reverb(rt60: float, wet: float = 0.35):
    def f(y, sr, rng):
        from scipy.signal import fftconvolve

        n = int(rt60 * sr)
        ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * np.arange(n) / n).astype(np.float32)
        ir /= np.sqrt((ir ** 2).sum())
        w = fftconvolve(y, ir)[: len(y)].astype(np.float32)
        out = (1 - wet) * y + wet * w * (np.std(y) / (np.std(w) + 1e-9))
        return np.clip(out, -1, 1).astype(np.float32)
    return f


def a_excerpt(seconds: float):
    def f(y, sr, rng):
        n = int(seconds * sr)
        if len(y) <= n:
            return y
        s = int(rng.integers(0, len(y) - n))
        return y[s:s + n].copy()
    return f


def a_filter(filt: str, codec: list[str] | None = None, ext: str = "wav"):
    return lambda y, sr, rng: _ff(y, sr, filt, codec, ext)


def a_chain(*fs):
    def f(y, sr, rng):
        for g in fs:
            y = g(y, sr, rng)
        return y
    return f


_SEMI2 = 2 ** (2 / 12)
MP3_64 = (["-c:a", "libmp3lame", "-b:a", "64k"], "mp3")
AUDIO_COPY_ATTACKS: dict[str, tuple] = {
    "none": (lambda y, sr, rng: y, "none"),
    "mp3_64": (a_filter(None, *MP3_64), "codec"),
    "aac_64": (a_filter(None, ["-c:a", "aac", "-b:a", "64k"], "m4a"), "codec"),
    "opus_24": (a_filter(None, ["-c:a", "libopus", "-b:a", "24k"], "ogg"), "codec"),
    "resample_8k": (a_filter("aresample=8000"), "codec"),
    "noise_20db": (a_noise(20), "noise"), "noise_10db": (a_noise(10), "noise"),
    "babble_10db": (a_babble(10), "noise"), "babble_0db": (a_babble(0), "noise"),
    "phone": (a_filter("highpass=f=300,lowpass=f=3400", *MP3_64), "filter"),
    "lowpass_2k": (a_filter("lowpass=f=2000"), "filter"),
    "compressor": (a_filter("acompressor=threshold=-24dB:ratio=6:attack=5:release=50,volume=6dB"), "filter"),
    "reverb": (a_reverb(0.6), "filter"),
    "pitch+2": (lambda y, sr, rng: _ff(y, sr, f"asetrate={sr}*{_SEMI2:.6f},aresample={sr},atempo={1 / _SEMI2:.6f}"), "time-pitch"),
    "tempo_0.9": (a_filter("atempo=0.9"), "time-pitch"), "tempo_1.1": (a_filter("atempo=1.1"), "time-pitch"),
    "speed_1.05": (lambda y, sr, rng: _ff(y, sr, f"asetrate={sr}*1.05,aresample={sr}"), "time-pitch"),
    "excerpt10": (a_excerpt(10), "excerpt"), "excerpt5": (a_excerpt(5), "excerpt"),
    "excerpt10_mp3_noise20": (a_chain(a_excerpt(10), a_filter(None, *MP3_64), a_noise(20)), "chain"),
    "rerecord": (a_chain(a_excerpt(10), a_reverb(0.4), a_filter("highpass=f=150,lowpass=f=7000"), a_noise(15),
                         a_filter(None, ["-c:a", "aac", "-b:a", "64k"], "m4a")), "chain"),
}


# ---------------------------------------------------------------- video, copy-detection ladder
# Each video attack returns ffmpeg arguments for one pass: (extra inputs, filter_complex or
# -vf chain, output codec args). `ctx` supplies the source duration and a background clip
# (a distractor never used as a query source) for picture-in-picture and insertion.
H264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]


EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # yuv420p needs even frame dimensions


def v_vf(chain: str, codec=None):
    return lambda rng, ctx: ([], ["-vf", f"{chain},{EVEN}"], codec or H264)


def v_codec(args):
    return lambda rng, ctx: ([], [], args)


def v_logo(rng, ctx):
    import augly

    pool = sorted((Path(augly.__file__).parent / "assets" / "twemojis").rglob("*.png"))
    logo = pool[int(rng.integers(0, len(pool)))]
    x, y = rng.uniform(0.05, 0.7, 2)
    return (["-i", str(logo)], ["-filter_complex",
            f"[1:v]scale=iw*0.9:-1[l];[0:v][l]overlay=x=W*{x:.3f}:y=H*{y:.3f}:format=auto,format=yuv420p"], H264)


def _text_png() -> Path:
    """White caption with a black outline, rendered once with Pillow (FFmpeg builds without
    libfreetype have no drawtext; an image overlay behaves the same on every host)."""
    from PIL import ImageDraw, ImageFont

    out = Path(tempfile.gettempdir()) / "op-fp-bench-caption.png"
    if not out.exists():
        font = ImageFont.truetype(_font("NotoSans-Regular.ttf"), 64)
        text = "NOT ORIGINAL - REPOST"
        l, t, r, b = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font, stroke_width=4)
        im = Image.new("RGBA", (r - l + 8, b - t + 8), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((4 - l, 4 - t), text, font=font, fill=(255, 255, 255, 255), stroke_width=4,
                                stroke_fill=(0, 0, 0, 255))
        tmp = out.with_suffix(f".{os.getpid()}.png")
        im.save(tmp)
        tmp.replace(out)
    return out


def v_text(rng, ctx):
    y = rng.uniform(0.1, 0.8)
    # -loop 1: the caption PNG is a single frame; looped, it can follow the clip, and
    # shortest=1 ends the output with the clip.
    return (["-loop", "1", "-i", str(_text_png())], ["-filter_complex",
            f"[1:v][0:v]scale2ref=w=main_w*0.8:h=ow/mdar[t][v];[v][t]overlay=x=(W-w)/2:y=H*{y:.3f}:format=auto:shortest=1,"
            f"{EVEN},format=yuv420p"], H264)


def v_pip(rng, ctx):
    x, y = rng.uniform(0.02, 0.48, 2)
    return (["-i", str(ctx["bg"])], ["-filter_complex",
            f"[0:v]scale=iw/2:-2[s];[1:v][s]overlay=x=W*{x:.3f}:y=H*{y:.3f}:shortest=1,format=yuv420p"], H264)


def v_excerpt(seconds: float):
    def f(rng, ctx):
        s = float(rng.uniform(0, max(0.0, ctx["duration"] - seconds)))
        ctx["excerpt_start"] = s
        return (["-ss", f"{s:.3f}", "-t", f"{seconds}"], [], H264)
    return f


def render_insert(src: Path, dst: Path, rng, ctx) -> None:
    """2.5 s of the source placed between 1.25 s of a background clip before and after.
    Rendered as three H.264 pieces at the source's frame size and 25 fps, then joined
    with the concat demuxer (a single filter graph with split + concat stalls on some builds)."""
    import json as _json

    s = float(rng.uniform(0, ctx["duration"] - 2.5))
    ctx["insert_at"], ctx["insert_src_start"], ctx["insert_len"] = 1.25, s, 2.5
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "json", str(src)], capture_output=True, text=True, check=True)
    st = _json.loads(probe.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    norm = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25"
    with tempfile.TemporaryDirectory() as d:
        parts = []
        for i, (path, start, dur) in enumerate(((ctx["bg"], 0.0, 1.25), (src, s, 2.5), (ctx["bg"], 1.25, 1.25))):
            out = Path(d) / f"p{i}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{dur}",
                            "-vf", norm, *H264, "-an", str(out)], check=True)
            parts.append(out)
        lst = Path(d) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                        str(dst)], check=True)


v_insert = "custom:render_insert"


def v_screenrec(rng, ctx):
    o = rng.uniform(0.02, 0.08, 8)
    persp = (f"perspective=x0=W*{o[0]:.3f}:y0=H*{o[1]:.3f}:x1=W*(1-{o[2]:.3f}):y1=H*{o[3]:.3f}:"
             f"x2=W*{o[4]:.3f}:y2=H*(1-{o[5]:.3f}):x3=W*(1-{o[6]:.3f}):y3=H*(1-{o[7]:.3f}):sense=destination")
    return ([], ["-vf", f"{persp},gblur=sigma=1,noise=alls=12:allf=t,eq=gamma=0.9,{EVEN}"],
            ["-c:v", "libx264", "-preset", "medium", "-crf", "30", "-pix_fmt", "yuv420p"])


VIDEO_COPY_ATTACKS: dict[str, tuple] = {
    "none": (None, "none"),
    "h264_crf28": (v_codec(["-c:v", "libx264", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p"]), "codec"),
    "h264_crf35": (v_codec(["-c:v", "libx264", "-preset", "medium", "-crf", "35", "-pix_fmt", "yuv420p"]), "codec"),
    "h265_crf32": (v_codec(["-c:v", "libx265", "-preset", "medium", "-crf", "32", "-pix_fmt", "yuv420p",
                            "-x265-params", "log-level=error"]), "codec"),
    "resize0.5": (v_vf("scale=iw/2:-2"), "scale"), "resize0.25": (v_vf("scale=iw/4:-2"), "scale"),
    "crop75": (v_vf("crop=iw*0.75:ih*0.75"), "crop"), "crop50": (v_vf("crop=iw*0.5:ih*0.5"), "crop"),
    "hflip": (v_vf("hflip"), "geometric"), "rot90": (v_vf("transpose=1"), "geometric"),
    "rot5": (v_vf("rotate=5*PI/180:ow=rotw(5*PI/180):oh=roth(5*PI/180):c=black"), "geometric"),
    "letterbox": (v_vf("scale=iw:ih*0.75,pad=iw:ih/0.75:0:(oh-ih)/2:black"), "geometric"),
    "gray": (v_vf("format=gray,format=yuv420p"), "photometric"),
    "bright": (v_vf("eq=brightness=0.15"), "photometric"), "contrast": (v_vf("eq=contrast=0.6"), "photometric"),
    "blur3": (v_vf("gblur=sigma=3"), "photometric"), "noise": (v_vf("noise=alls=25:allf=t"), "photometric"),
    "text": (v_text, "overlay"), "logo": (v_logo, "overlay"), "pip": (v_pip, "overlay"),
    "fps15": (v_vf("fps=15"), "temporal"), "speed1.25": (v_vf("setpts=0.8*PTS"), "temporal"),
    "excerpt2.5": (v_excerpt(2.5), "temporal"), "insert": (v_insert, "temporal"),
    "screenrec": (v_screenrec, "chain"),
}
