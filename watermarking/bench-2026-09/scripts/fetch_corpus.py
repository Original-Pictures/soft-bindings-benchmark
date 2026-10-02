"""Build the public, licence-traceable corpus under $BENCH_WORK/corpus.

Never customer media. Every item lands in corpus_manifest.json with source URL,
licence note and sha256 of the file actually used. Licence posture per set:

  kodak      Kodak Lossless True Color suite (r0k.us), released by Eastman Kodak
             for unrestricted usage.
  clic       CLIC 2020 professional validation (Unsplash-sourced, Unsplash licence).
  div2k      DIV2K validation HR (ETH Zurich). ACADEMIC-RESEARCH USE ONLY per the
             dataset page: used only for internal measurement, never redistributed,
             never shown in the report.
  hdr16      Poly Haven HDRIs (CC0), exposure-tone-mapped to 16-bit sRGB TIFF.
  speech     LibriSpeech test-clean (CC BY 4.0).
  music      librosa/data recordings whose per-file licence text is CC0/CC-BY/public
             domain; any NC/ND licence is skipped automatically.
  video      Xiph derf 1080p: NTIA/ITS VQEG HDTV sources (US Government work) and
             the Blender Sintel trailer (CC BY 3.0). First N frames only, via HTTP
             range requests.

Usage: python fetch_corpus.py [--small]   (--small = local smoke subset)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from common import CORPUS, SCRIPTS, sha256_file, write_json

UA = {"User-Agent": "op-wm-bench"}
MANIFEST = SCRIPTS.parent / "corpus_manifest.json"

POLYHAVEN = ["venice_sunset", "studio_small_09", "kloppenheim_06", "abandoned_parking",
             "rosendal_plains_2", "moonless_golf", "brown_photostudio_02", "spruit_sunrise"]
VIDEO_CLIPS = {  # name -> (file, fps, licence)
    "aspen": ("aspen_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "controlled_burn": ("controlled_burn_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "red_kayak": ("red_kayak_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "rush_field_cuts": ("rush_field_cuts_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "snow_mnt": ("snow_mnt_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "speed_bag": ("speed_bag_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "touchdown_pass": ("touchdown_pass_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "west_wind_easy": ("west_wind_easy_1080p.y4m", 30, "NTIA/ITS VQEG HDTV source (US Government work)"),
    "sintel_trailer": ("sintel_trailer_2k_1080p24.y4m", 24, "Blender Foundation, CC BY 3.0"),
    "crowd_run": ("crowd_run_1080p50.y4m", 50, "SVT MultiFormat test sequence (research test use)"),
}
MUSIC = ["Kevin_MacLeod_-_Vibe_Ace", "admiralbob77_-_Choice_-_Drum-bass",
         "Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy",
         "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra)",
         "sorohanro_-_solo-trumpet-06", "Karissa_Hobbs_-_Lets_Go_Fishin",
         "147793__setuniman__sweet-waltz-0i-22mi",
         "442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime",
         "drese-midi", "snare-accelerate", "glacier-bay-humpback"]


def get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    headers = dict(UA)
    if rng:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=900) as r:
        return r.read()


def fetch_to(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(tmp), url], check=True)
        tmp.replace(dest)
    return dest


def item(kind: str, path: Path, source: str, licence: str, **extra: object) -> dict:
    return {"set": kind, "path": str(path.relative_to(CORPUS)), "sha256": sha256_file(path),
            "source": source, "licence": licence, **extra}


def images(small: bool) -> list[dict]:
    out = []
    n_kodak = 6 if small else 24
    for i in range(1, n_kodak + 1):
        url = f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
        p = fetch_to(url, CORPUS / "image/kodak" / f"kodim{i:02d}.png")
        out.append(item("kodak", p, url, "Kodak suite, released for unrestricted usage"))
    # CLIC 2020 professional validation (Unsplash licence)
    clic_dir = CORPUS / "image/clic"
    if not clic_dir.exists():
        url = "https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip"
        zpath = fetch_to(url, CORPUS / "_dl/clic_prof_valid.zip")
        clic_dir.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".png") and not Path(n).name.startswith("._"))
            for n in names[:30]:
                (clic_dir / Path(n).name).write_bytes(z.read(n))
    for p in sorted(q for q in clic_dir.glob("*.png") if not q.name.startswith("._"))[: 4 if small else 30]:
        out.append(item("clic", p, "CLIC 2020 professional_valid", "Unsplash licence (CLIC)"))
    if not small:
        div_dir = CORPUS / "image/div2k"
        if not div_dir.exists():
            url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
            zpath = fetch_to(url, CORPUS / "_dl/DIV2K_valid_HR.zip")
            div_dir.mkdir(parents=True)
            with zipfile.ZipFile(zpath) as z:
                for n in sorted(n for n in z.namelist() if n.endswith(".png") and not Path(n).name.startswith("._"))[:100]:
                    (div_dir / Path(n).name).write_bytes(z.read(n))
        for p in sorted(q for q in div_dir.glob("*.png") if not q.name.startswith("._")):
            out.append(item("div2k", p, "DIV2K_valid_HR", "DIV2K: academic research only; internal measurement, not redistributed"))
    # 16-bit TIFF from CC0 HDRIs
    hdr_dir = CORPUS / "image/hdr16"
    for name in POLYHAVEN[: 2 if small else len(POLYHAVEN)]:
        tif = hdr_dir / f"{name}.tif"
        url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/{name}_2k.hdr"
        if not tif.exists():
            raw = fetch_to(url, CORPUS / "_dl" / f"{name}_2k.hdr")
            import cv2

            hdr = cv2.imread(str(raw), cv2.IMREAD_UNCHANGED)[:, :, ::-1].astype(np.float32)
            # Exposure so the median maps to 18% grey, Reinhard, then sRGB OETF.
            lum = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
            x = hdr * (0.18 / max(float(np.median(lum)), 1e-6))
            x = x / (1.0 + x)
            srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055)
            u16 = np.clip(np.round(srgb * 65535), 0, 65535).astype(np.uint16)
            import tifffile

            hdr_dir.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(tif, u16, photometric="rgb", compression="zlib")
        out.append(item("hdr16", tif, url, "Poly Haven CC0", bits=16))
    return out


def audio(small: bool) -> list[dict]:
    import librosa
    import soundfile as sf

    out = []
    base = CORPUS / "audio"
    # LibriSpeech test-clean: first utterance >= 6 s from distinct speakers
    ls_dir = base / "speech16k"
    want = 6 if small else 40
    if len(list(ls_dir.glob("*.wav"))) < want:
        tgz = fetch_to("https://www.openslr.org/resources/12/test-clean.tar.gz", CORPUS / "_dl/test-clean.tar.gz")
        ls_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()
        with tarfile.open(tgz) as tar:
            for m in tar:
                if not m.name.endswith(".flac"):
                    continue
                spk = m.name.split("/")[-3]
                if spk in seen:
                    continue
                y, sr = sf.read(io.BytesIO(tar.extractfile(m).read()))
                if len(y) / sr < 6:
                    continue
                seen.add(spk)
                sf.write(ls_dir / (Path(m.name).stem + ".wav"), y[: sr * 10].astype(np.float32), sr, subtype="PCM_16")
                if len(seen) >= want:
                    break
    for p in sorted(ls_dir.glob("*.wav"))[:want]:
        out.append(item("speech", p, "LibriSpeech test-clean", "CC BY 4.0", sr=16000))
    # CC music excerpts (librosa/data); licence text parsed from the sidecar .txt
    mus_dir = base / "music44k"
    for key in MUSIC[: 3 if small else len(MUSIC)]:
        stem = key.replace("'", "")
        txt = get(f"https://raw.githubusercontent.com/librosa/data/main/audio/{urllib.request.quote(key)}.txt").decode("utf-8", "replace")
        lic = _licence_line(txt)
        if re.search(r"\bNC\b|NonCommercial|Non-Commercial|\bND\b|NoDeriv", lic, re.I):
            print(f"skip {key}: {lic}")
            continue
        wav = mus_dir / f"{stem}.wav"
        if not wav.exists():
            ogg = fetch_to(f"https://raw.githubusercontent.com/librosa/data/main/audio/{urllib.request.quote(key)}.hq.ogg",
                           CORPUS / "_dl" / f"{stem}.hq.ogg")
            mus_dir.mkdir(parents=True, exist_ok=True)
            # ffmpeg decodes every container the librosa/data files use; skip what it cannot.
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "5", "-t", "10", "-i", str(ogg), "-ac", "1",
                                "-ar", "44100", "-c:a", "pcm_s16le", str(wav)], capture_output=True, text=True)
            if r.returncode != 0 or not wav.exists() or sf.info(wav).duration < 5:
                print(f"skip {key}: undecodable ({r.stderr.strip()[:120]})")
                wav.unlink(missing_ok=True)
                continue
        out.append(item("music", wav, f"librosa/data {key}", lic, sr=44100))
    # Derived sample-rate variants: 16 k speech -> 44.1/48 k, 44.1 k music -> 16/48 k
    for sr_out in (16000, 44100, 48000):
        for src in list(out):
            if src["sr"] == sr_out or src.get("derived_from"):
                continue
            p = CORPUS / src["path"]
            dst = base / f"{src['set']}_{sr_out}" / p.name
            if not dst.exists():
                y, _ = librosa.load(p, sr=sr_out, mono=True, res_type="soxr_vhq")
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, y, sr_out, subtype="PCM_16")
            out.append(item(src["set"], dst, src["source"], src["licence"], sr=sr_out, derived_from=src["path"]))
    return out


def _licence_line(txt: str) -> str:
    for line in txt.splitlines():
        if re.search(r"licen[cs]e|creative commons|public domain|CC[- ]?(BY|0)", line, re.I):
            return line.strip()
    return "UNKNOWN: " + txt.strip().splitlines()[-1][:120] if txt.strip() else "UNKNOWN"


def video(small: bool) -> list[dict]:
    out = []
    seconds = 2 if small else 5
    clips = list(VIDEO_CLIPS.items())[: 2 if small else len(VIDEO_CLIPS)]
    for name, (fname, fps, lic) in clips:
        dst = CORPUS / "video" / f"{name}_{seconds}s.mkv"
        url = f"https://media.xiph.org/video/derf/y4m/{fname}"
        if not dst.exists():
            head = get(url, (0, 4095))
            hdr_end = head.index(b"\n") + 1
            header = head[:hdr_end].decode()
            w = int(re.search(r" W(\d+)", header).group(1))
            h = int(re.search(r" H(\d+)", header).group(1))
            c = re.search(r" C(\S+)", header)
            chroma = c.group(1) if c else "420"
            frame_bytes = w * h * (3 if chroma.startswith("444") else 2 if chroma.startswith("422") else 1.5)
            frame_bytes = int(frame_bytes) + 6  # "FRAME\n"
            nframes = fps * seconds
            blob = get(url, (0, hdr_end + frame_bytes * nframes - 1))
            raw = CORPUS / "_dl" / f"{name}.y4m"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(blob)
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Lossless FFV1 keeps the reference exact; 8-bit 4:2:0 as in the source.
            vf = "scale=1920:1080:flags=lanczos" if (w, h) != (1920, 1080) else "null"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-frames:v", str(nframes), "-vf", vf,
                            "-pix_fmt", "yuv420p", "-c:v", "ffv1", "-level", "3", str(dst)], check=True)
            raw.unlink()
        out.append(item("video", dst, url, lic, fps=fps, seconds=seconds))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--only", default="image,audio,video")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for kind in args.only.split(","):
        manifest[kind] = {"image": images, "audio": audio, "video": video}[kind](args.small)
        print(kind, len(manifest[kind]))
    if not args.small:
        write_json(MANIFEST, manifest)
    else:
        write_json(CORPUS / "corpus_manifest.small.json", manifest)


if __name__ == "__main__":
    main()
