"""Single source of every number in the manuscript.

Reads the per-method records the bench wrote (results-gpu/ from the GPU host, or the
directory in $PAPER_RESULTS for a local smoke run) and derives what the paper reports.
Nothing here re-measures; figures.py and build_numbers.py only call load().

Conventions
-----------
* Operating point. A registry must bound false bindings, so the headline robustness
  number is TPR at a *pair-level* false-positive rate of 1e-7: the threshold at which one
  in ten million (never-registered query, reference) pairs passes, fixed on calibration
  negatives and applied to held-out ones. At registry size N the query-level FPR is about
  N x 1e-7, i.e. ~1% at the benchmark's ~100k references (checked out of sample on the
  test negatives). TPR counts a positive only if its top-1 is the true asset and the
  score passes.
* R@1 and muAP are threshold-free and reported alongside.
* Product-eligible (tier P) = code and weights under MIT/BSD/Apache/CC0; reference tier
  (R) = weights unlicensed, trained on non-commercial data, or copyleft.
* Latency: GPU methods on one A10G (batch 64); CPU methods single-threaded on the same
  host's EPYC 7R32. Per image at the benchmark's 1024 px working size.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

PAPER = Path(__file__).resolve().parents[1]
BENCH = PAPER.parent
RAW = Path(os.environ.get("PAPER_RESULTS", BENCH / "results-gpu"))

OP = "pair@1e-07"          # headline operating point
OP_Q = "query@0.01"        # secondary (query-level at the benchmark registry size)
PAIR_TARGETS = ("pair@0.0001", "pair@1e-05", "pair@1e-06", "pair@1e-07", "pair@1e-08")

# On-demand Linux prices, us-east-1 (AWS Price List API), retrieved 2026-09-26.
PRICE_GPU_USD_H = 1.212    # g5.2xlarge (A10G, 8 vCPU EPYC 7R32)
PRICE_CPU_USD_H = 0.0864   # m6a.large-equivalent per vCPU-hour pair; used per single thread
PRICE_SOURCE = "AWS Price List API, us-east-1, on-demand Linux, retrieved 2026-09-26"

# Display names and class of each method. Class orders tables and sets the colour.
LABEL = {
    "PDQ": "PDQ", "PDQ-dihedral": "PDQ (dihedral)", "aHash-64": "aHash", "dHash-64": "dHash",
    "pHash-64": "pHash", "pHash-256": "pHash-256", "wHash-64": "wHash", "BlockMean": "BlockMean",
    "MarrHildreth": "Marr–Hildreth", "ColorMoment": "ColorMoment", "Blockhash-256": "Blockhash",
    "ISCC-Image-64": "ISCC Image-64", "ISCC-Image-256": "ISCC Image-256", "ISCC-SCI-256": "ISCC SCI-256",
    "DINOv2-S": "DINOv2-S", "DINOv2-B": "DINOv2-B", "DINOv2-S-LSH256": "DINOv2-S LSH-256",
    "OpenCLIP-B32": "OpenCLIP B/32", "SSCD-mixup": "SSCD R50", "SSCD-large": "SSCD RX101",
    "ISC21-1st": "ISC21 1st", "DINOHash-96": "DINOHash-96",
    "Chromaprint": "Chromaprint", "ISCC-Audio-64": "ISCC Audio-64", "ISCC-Audio-256": "ISCC Audio-256",
    "audfprint": "audfprint", "CLAP": "CLAP", "NMFP-triplet": "NMFP",
    "vPDQ": "vPDQ", "TMK+PDQF": "TMK+PDQF", "ISCC-Video-64": "ISCC Video-64", "ISCC-Video-256": "ISCC Video-256",
    "DINOv2-S-seq": "DINOv2-S frames", "DINOv2-S-mean": "DINOv2-S mean", "SSCD-mixup-seq": "SSCD frames",
    "SSCD-mixup-mean": "SSCD mean",
}
CLASS = {  # hash | standard | general | copydet
    **{m: "hash" for m in ("PDQ", "PDQ-dihedral", "aHash-64", "dHash-64", "pHash-64", "pHash-256", "wHash-64",
                           "BlockMean", "MarrHildreth", "ColorMoment", "Blockhash-256", "vPDQ", "TMK+PDQF",
                           "Chromaprint", "audfprint")},
    **{m: "standard" for m in ("ISCC-Image-64", "ISCC-Image-256", "ISCC-Audio-64", "ISCC-Audio-256",
                               "ISCC-Video-64", "ISCC-Video-256")},
    **{m: "general" for m in ("DINOv2-S", "DINOv2-B", "DINOv2-S-LSH256", "OpenCLIP-B32", "CLAP", "DINOv2-S-seq",
                              "DINOv2-S-mean")},
    **{m: "copydet" for m in ("SSCD-mixup", "SSCD-large", "ISC21-1st", "ISCC-SCI-256", "DINOHash-96", "NMFP-triplet",
                              "SSCD-mixup-seq", "SSCD-mixup-mean")},
}
CLASS_ORDER = ["hash", "standard", "general", "copydet"]
CLASS_LABEL = {"hash": "Perceptual hash", "standard": "ISCC (ISO 24138)", "general": "General embedding",
               "copydet": "Copy-detection trained"}
ORDER = list(LABEL)

ATTACK_LABEL = {
    # image
    "none": "None", "jpeg75": "JPEG 75", "jpeg50": "JPEG 50", "jpeg30": "JPEG 30", "webp50": "WebP 50",
    "resize0.5": "Resize 0.5", "resize0.25": "Resize 0.25", "squash0.75": "Aspect 0.75",
    "crop90": "Crop 90%", "crop70": "Crop 70%", "crop50": "Crop 50%", "crop30": "Crop 30%", "crop50off": "Crop 50% off-ctr",
    "rot5": "Rotate 5°", "rot15": "Rotate 15°", "rot90": "Rotate 90°", "hflip": "Flip", "perspective": "Perspective",
    "pad25": "Pad 25%", "skew": "Skew", "bright1.4": "Bright 1.4", "bright0.6": "Bright 0.6", "contrast0.5": "Contrast 0.5",
    "gray": "Grayscale", "sat2": "Saturation 2", "gamma0.6": "Gamma 0.6", "blur2": "Blur σ=2", "noise0.05": "Noise σ=.05",
    "pixelize0.3": "Pixelate", "text": "Text overlay", "emoji0.3": "Emoji 30%", "meme": "Meme", "screenshot": "Screenshot",
    "background0.5": "On background", "social": "Social chain", "recapture": "Re-capture",
    # audio
    "mp3_64": "MP3 64k", "aac_64": "AAC 64k", "opus_24": "Opus 24k", "resample_8k": "Resample 8k",
    "noise_20db": "Noise 20 dB", "noise_10db": "Noise 10 dB", "babble_10db": "Babble 10 dB", "babble_0db": "Babble 0 dB",
    "phone": "Telephone", "lowpass_2k": "Low-pass 2k", "compressor": "Compressor", "reverb": "Reverb",
    "pitch+2": "Pitch +2 st", "tempo_0.9": "Tempo 0.9", "tempo_1.1": "Tempo 1.1", "speed_1.05": "Speed 1.05",
    "excerpt10": "Excerpt 10 s", "excerpt5": "Excerpt 5 s", "excerpt10_mp3_noise20": "Excerpt+MP3+noise",
    "rerecord": "Re-record",
    # video
    "h264_crf28": "H.264 CRF28", "h264_crf35": "H.264 CRF35", "h265_crf32": "H.265 CRF32", "crop75": "Crop 75%",
    "letterbox": "Letterbox", "bright": "Brightness", "contrast": "Contrast", "blur3": "Blur σ=3", "noise": "Noise",
    "logo": "Logo", "pip": "Picture-in-picture", "fps15": "15 fps", "speed1.25": "Speed 1.25",
    "excerpt2.5": "Excerpt 2.5 s", "insert": "Inserted 2.5 s", "screenrec": "Screen recording",
}


@dataclass
class Method:
    modality: str
    name: str
    rec: dict[str, Any]
    label: str = ""
    cls: str = ""
    tier: str = "P"

    def __post_init__(self):
        self.label = LABEL.get(self.name, self.name)
        self.cls = CLASS.get(self.name, "general")
        self.tier = self.rec["method"].get("tier", "P")

    @property
    def pooled(self) -> dict:
        return self.rec["per_attack"]["_all"]

    def at(self, attack: str, key: str = "R@1") -> float:
        v = self.rec["per_attack"].get(attack, {}).get(key)
        return float("nan") if v is None else float(v)

    @property
    def bits(self) -> float:
        m = self.rec["method"]
        if "bits" in m:
            return float(m["bits"])
        return float(self.rec.get("ref_bits_mean", float("nan")))

    def _lat(self) -> dict:
        lat = _j(RAW / "latency.json").get(self.modality, {})
        return lat.get(self.name) or lat.get(self.name.replace("-mean", "-seq")) or {}

    def ms_per_item(self) -> float:
        """Clean latency from latency.json (idle host): ms per image, or ms per second of
        audio/video. Extraction-time timings (contended) are not used."""
        lat = self._lat()
        if lat:
            return float(lat["ms_per_item"] if self.modality == "image" else lat["ms_per_media_second"])
        return float("nan")

    @property
    def device(self) -> str:
        return self._lat().get("device", "")


MIN_PAIRS = 5  # bench rule (scripts/retrieval_metrics.py): a target needs >= 5 false pairs/queries


def check_resolvable(mod: str, name: str, rec: dict) -> None:
    """Refuse a record that reports an operating point its calibration set cannot resolve.

    With n calibration queries and N references, a pair-level target t is resolved only if
    t * n * N >= MIN_PAIRS (a query-level one if t * n >= MIN_PAIRS); below that the threshold
    just sits above every observed negative. Records written before the bench enforced this
    (the first video pass) reported such thresholds as if they were the stated rate."""
    n, N = rec.get("n_cal_neg"), rec.get("n_ref")
    if not n or not N:
        return
    for key, th in rec.get("thresholds", {}).items():
        kind, t = key.split("@")
        need = float(t) * n * (N if kind == "pair" else 1)
        if need < MIN_PAIRS and th is not None and np.isfinite(th):
            raise SystemExit(f"{mod}/{name}: {key} is not resolvable with {n} calibration queries x {N} refs "
                             f"but the record gives threshold {th}; re-run the metrics stage")


def _load_dir(mod: str) -> list[Method]:
    d = RAW / mod
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        if f.name.startswith("_"):
            continue
        rec = _j(f)
        check_resolvable(mod, f.stem, rec)
        out.append(Method(mod, f.stem, rec))
    return sorted(out, key=lambda m: (CLASS_ORDER.index(m.cls), ORDER.index(m.name) if m.name in ORDER else 99))


def _j(path: Path) -> dict:
    """Read JSON, falling back to a gzipped copy: large raw files are committed compressed."""
    if path.exists():
        return json.loads(path.read_text())
    gz = path.with_name(path.name + ".gz")
    return json.loads(gzip.decompress(gz.read_bytes())) if gz.exists() else {}


@lru_cache(maxsize=1)
def load() -> dict:
    data = {mod: _load_dir(mod) for mod in ("image", "audio", "video")}
    data["dedup"] = _j(RAW / "image" / "_dedup.json")
    # duplicate-rule sensitivity (same queries, different cleanup of catalogue variants)
    data["image_sens"] = {}
    for rule in ("two", "none"):
        d = RAW / f"image_dedup-{rule}"
        data["image_sens"][rule] = {m.stem: _j(m) for m in sorted(d.glob("*.json")) if not m.name.startswith("_")} \
            if d.exists() else {}
    loc = RAW / "localize"
    data["localize"] = {f.name.split(".")[0]: _j(f.with_name(f.name.split(".")[0] + ".json")) for f in sorted(loc.glob("*.json*"))} if loc.exists() else {}
    data["localize_retrieval"] = {f.stem: _j(f) for f in sorted((loc / "retrieval").glob("*.json"))} \
        if (loc / "retrieval").exists() else {}
    sec = RAW / "security"
    data["security"] = {f.name.split(".")[0]: _j(f.with_name(f.name.split(".")[0] + ".json")) for f in sorted(sec.glob("*.json*"))} if sec.exists() else {}
    data["wmcombo"] = _j(RAW / "wmcombo" / "wmcombo.json")
    data["stability"] = _j(RAW / "stability" / "compare.json")
    data["manifest"] = _j(RAW / "locks" / "corpus_manifest.json") or _j(BENCH / "corpus_manifest.json")
    # Source-level bootstrap intervals and paired tests (paper/scripts/cluster_ci.py).
    data["cluster"] = _j(RAW / "_cluster_ci.json")
    data["verify"] = _j(RAW / "image" / "_verify.json")
    return data


def by_name(methods: list[Method]) -> dict[str, Method]:
    return {m.name: m for m in methods}


def best(methods: list[Method], key=lambda m: m.pooled.get(f"TPR@{OP}", float("nan")), tier: str | None = None) -> Method:
    cand = [m for m in methods if tier is None or m.tier == tier]
    return max(cand, key=lambda m: np.nan_to_num(key(m), nan=-1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def family_mean(m: Method, family: str, key: str = "R@1") -> float:
    fam = m.rec["attack_family"]
    vals = [m.at(a, key) for a, f in fam.items() if f == family]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# Attack ladders in the order attacks.py defines them (JSON records store them sorted).
ORDER_BY_MODALITY = {
    "image": ["none", "jpeg75", "jpeg50", "jpeg30", "webp50", "resize0.5", "resize0.25", "squash0.75", "crop90", "crop70",
              "crop50", "crop30", "crop50off", "rot5", "rot15", "rot90", "hflip", "perspective", "pad25", "skew",
              "bright1.4", "bright0.6", "contrast0.5", "gray", "sat2", "gamma0.6", "blur2", "noise0.05", "pixelize0.3",
              "text", "emoji0.3", "meme", "screenshot", "background0.5", "social", "recapture"],
    "audio": ["none", "mp3_64", "aac_64", "opus_24", "resample_8k", "noise_20db", "noise_10db", "babble_10db",
              "babble_0db", "phone", "lowpass_2k", "compressor", "reverb", "pitch+2", "tempo_0.9", "tempo_1.1",
              "speed_1.05", "excerpt10", "excerpt5", "excerpt10_mp3_noise20", "rerecord"],
    "video": ["none", "h264_crf28", "h264_crf35", "h265_crf32", "resize0.5", "resize0.25", "crop75", "crop50", "hflip",
              "rot90", "rot5", "letterbox", "gray", "bright", "contrast", "blur3", "noise", "text", "logo", "pip",
              "fps15", "speed1.25", "excerpt2.5", "insert", "screenrec"],
}
OP_BY_MODALITY = {"image": OP, "audio": OP_Q, "video": OP_Q}


def attack_order(m: Method) -> list[str]:
    fam = m.rec["attack_family"]
    order = ORDER_BY_MODALITY[m.modality]
    return [a for a in order if a in fam] + [a for a in fam if a not in order]


def attack_families(m: Method) -> list[str]:
    seen = []
    for f in (m.rec["attack_family"][a] for a in attack_order(m)):
        if f not in seen:
            seen.append(f)
    return seen


def pct(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"
