"""Single source of every number in the manuscript.

Reads the bench's aggregated `results.json` and the raw per-configuration records in
`results-gpu/`, and derives the quantities the paper reports: bootstrap confidence
intervals, exact binomial intervals for false-positive rates, latency, an estimated
cloud cost per unit of media, and a combined score. Nothing here re-measures.

Conventions
-----------
* A configuration *qualifies* when (i) its code and weights carry an open licence,
  (ii) mean bit accuracy after the modality's gate attack is >= 0.95 both pooled and
  on every content subset, and (iii) its false-positive rate is supported below the
  target, see Config.fpr_evidence. Best-cost, best-speed and best-combined are chosen among qualifying
  configurations only, because a fast watermark that does not survive the gate
  attack is not a usable operating point.
* Speed is GPU embedding latency (A10G). Cost is the cheaper of running on the GPU
  instance or on a 4-vCPU CPU instance of the same processor family, priced at
  on-demand rates, and excludes I/O, storage and idle time.
* The combined score is the weighted geometric mean of four ratios in (0, 1], each
  1 for the best qualifying configuration: quality, robustness, speed and cost.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

PAPER = Path(__file__).resolve().parents[1]
BENCH = PAPER.parent
RESULTS = BENCH / "results.json"
RAW = BENCH / "results-gpu"

# On-demand Linux prices, us-east-1, AWS Price List API, effective 2026-09-01,
# retrieved 2026-09-23. g5.xlarge is the host the GPU numbers were measured on
# (A10G, AMD EPYC 7R32, 4 vCPU); m6a.xlarge is the 4-vCPU EPYC 7R13 instance that
# matches the 4-thread CPU latency measurement.
PRICE_GPU_USD_H = 1.006
PRICE_CPU_USD_H = 0.1728
PRICE_DATE = "2026-09-01"
PRICE_SOURCE = "AWS Price List API, us-east-1, on-demand Linux, retrieved 2026-09-23"

VIDEO_MP_PER_FRAME = 1920 * 1080 / 1e6
VIDEO_FPS = 30.0
BOOT = 10_000
RNG_SEED = 20260923

WEIGHTS_EQUAL = (0.25, 0.25, 0.25, 0.25)
WEIGHTS_QUALITY = (0.5, 0.2, 0.15, 0.15)

GATE_ATTACK = {"image": "jpeg75", "audio": "mp3_128", "video": "h264_crf23"}

# Display names. Canonical capitalisation of each model, strength in parentheses.
FAMILY_LABEL = {
    "TrustMark": "TrustMark", "PixelSeal": "PixelSeal", "VideoSeal-1.0": "Video Seal", "ChunkySeal": "ChunkySeal",
    "WAM": "WAM", "invisible-watermark": "DWT-DCT", "RivaGAN": "RivaGAN", "InvisMark": "InvisMark",
    "AudioSeal": "AudioSeal", "WavMark": "WavMark", "SilentCipher": "SilentCipher", "Perth": "Perth",
}


@dataclass
class Config:
    modality: str
    name: str
    family: str
    label: str
    strength: float | None
    open_licence: bool
    n: int
    quality: dict[str, float]
    quality_ci: dict[str, tuple[float, float]]
    robustness: dict[str, float]
    gate_pooled: float
    gate_worst: float
    gate_worst_set: str
    blind_fpr: float | None
    blind_trials: int | None
    blind_fpr_ci: tuple[float, float] | None
    blind_fpr_by_set: dict[str, float]
    key_fpr: float | None
    key_trials: int | None
    key_fpr_upper: float | None
    gpu_ms: float | None
    cpu_ms: float | None
    cpu_imputed: bool
    cost_gpu: float | None
    cost_cpu: float | None
    per_item: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    per_item_gate: np.ndarray | None = field(repr=False, default=None)
    items: list[str] = field(repr=False, default_factory=list)
    # Source of each item: audio items are one clip at three sample rates, so the clip, not
    # the rendered file, is the independent unit for intervals and paired tests.
    groups: list[str] = field(repr=False, default_factory=list)
    sets: list[str] = field(repr=False, default_factory=list)
    # Exact payload recovery per transformation: the decoded payload equals the embedded one
    # (after BCH decoding for TrustMark; all n bits otherwise). None for Perth (no payload).
    exact: dict[str, float] = field(default_factory=dict)
    # Expected-payload match per transformation: at least k of n bits equal the embedded
    # payload (the verifier's decision rule when k exists).
    keymatch: dict[str, float] = field(default_factory=dict)
    gate_exact_items: np.ndarray | None = field(repr=False, default=None)
    scores: dict[str, float] = field(default_factory=dict)
    gpu_ms_raw: float | None = None
    cpu_ms_raw: float | None = None
    nbits: int = 0
    k_threshold: int | None = None

    @property
    def network(self) -> str:
        """The forward-pass identity. Strength only scales the residual, so latency is
        a property of the network, not of the strength setting."""
        if self.family == "TrustMark":
            return "TrustMark-" + self.label.split("-")[1][0]
        if self.family == "AudioSeal":
            return "AudioSeal " + ("streaming" if "streaming" in self.name else "base")
        return self.family

    @property
    def cost(self) -> float | None:
        c = [v for v in (self.cost_gpu, self.cost_cpu) if v is not None]
        return min(c) if c else None

    @property
    def cost_device(self) -> str:
        if self.cost_cpu is not None and self.cost_gpu is not None and self.cost_cpu < self.cost_gpu:
            return "CPU"
        return "GPU"

    @property
    def key_reachable(self) -> bool:
        """Whether an expected-payload test at per-key p <= 1e-6 exists for this payload size.

        For n = 16 bits no k <= n reaches 1e-6 (2^-16 = 1.5e-5), so the test is vacuous
        and its zero false matches are not evidence.
        """
        return self.k_threshold is not None and self.k_threshold <= self.nbits

    @property
    def fpr_evidence(self) -> str:
        """How the false-positive criterion is supported.

        'analytic': expected-payload test with per-key bound <= 1e-6 under fair independent
        bits, and zero exceedances on unmarked content. 'observed': no analytic test is
        possible; the blind detector produced zero false detections (bound = CP upper).
        'fails': neither.
        """
        if self.key_reachable and self.key_trials and round((self.key_fpr or 0) * self.key_trials) == 0:
            return "analytic"
        if not self.key_reachable and self.blind_trials and self.blind_fpr == 0:
            return "observed"
        return "fails"

    @property
    def qualifies(self) -> bool:
        return (self.open_licence and self.fpr_evidence != "fails" and self.gate_pooled >= 0.95
                and self.gate_worst >= 0.95)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def bootstrap_ci(x: np.ndarray, groups: list[str] | None = None, seed: int = RNG_SEED) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI of the item mean, resampling sources.

    `groups` names the source of each item. Items of one source (an audio clip rendered at
    three sample rates) are resampled together, so the interval reflects the number of
    independent sources rather than the number of rendered files. Without groups every
    item is its own source."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(groups if groups is not None and len(groups) == len(x) else np.arange(len(x)).astype(str))
    ok = np.isfinite(x)
    x, g = x[ok], g[ok]
    _, gi = np.unique(g, return_inverse=True)
    n_src = int(gi.max()) + 1 if len(gi) else 0
    if n_src < 2:
        return (float("nan"), float("nan"))
    sums, counts = np.bincount(gi, weights=x, minlength=n_src), np.bincount(gi, minlength=n_src).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_src, size=(BOOT, n_src))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def source_of(modality: str, item: str) -> str:
    """Independent unit behind an item: the clip for audio (one file per sample rate), the
    item itself for images and video."""
    return Path(item).name if modality == "audio" else item


def _raw_path(modality: str, name: str) -> Path:
    return RAW / modality / (name.replace("@", "_at_") + ".json.gz")


@lru_cache(maxsize=None)
def _raw(modality: str, name: str) -> dict[str, Any] | None:
    p = _raw_path(modality, name)
    return json.load(gzip.open(p)) if p.exists() else None


def _strength(meta: dict[str, Any]) -> float | None:
    p = meta.get("params", {})
    for k in ("scaling_w_effective", "scaling_w", "WM_STRENGTH", "alpha", "strength"):
        if isinstance(p.get(k), (int, float)):
            return float(p[k])
    return None


def _label(meta: dict[str, Any]) -> str:
    fam = FAMILY_LABEL.get(meta["family"], meta["family"])
    name = meta["name"]
    if meta["family"] == "TrustMark":
        variant = meta["params"]["variant"]
        return f"TrustMark-{variant} ({meta['params']['WM_STRENGTH']:g})"
    if meta["family"] == "invisible-watermark":
        return name
    if meta["family"] == "AudioSeal":
        kind = "streaming" if "streaming" in name else "base"
        return f"AudioSeal {kind} ({name.split('@')[1]})"
    if meta["family"] == "SilentCipher":
        return "SilentCipher 44.1k"
    s = _strength(meta)
    return f"{fam} ({s:g})" if s is not None and "@" in name else fam


_QUALITY_KEYS = {
    "image": ("psnr", "ssim", "ms_ssim", "lpips", "flip", "cvvdp"),
    "audio": ("snr", "si_snr", "pesq", "stoi", "dlufs", "spec_diff_db"),
    "video": ("psnr", "ssim", "lpips", "flip", "vmaf", "cvvdp"),
}


def _per_item(modality: str, raw: dict[str, Any] | None) -> tuple[dict[str, np.ndarray], np.ndarray | None, list[str]]:
    if raw is None:
        return {}, None, []
    # Items the bench excluded from aggregation stay excluded here, so per-item statistics
    # use the same population as the aggregated means.
    excluded = set(json.loads(RESULTS.read_text())["corpus_excluded"].get(modality, []))
    recs = [r for r in raw["records"] if r["item"] not in excluded]
    out = {k: np.array([r.get(k, np.nan) if r.get(k) is not None else np.nan for r in recs], dtype=float)
           for k in _QUALITY_KEYS[modality]}
    if modality == "audio":
        out["abs_dlufs"] = np.abs(out["dlufs"])
        # PESQ models speech; it is reported on speech items and, separately, on music.
        speech = np.array([r.get("set") == "speech" for r in recs])
        out["pesq_speech"] = np.where(speech, out["pesq"], np.nan)
        out["pesq_music"] = np.where(~speech, out["pesq"], np.nan)
    g = GATE_ATTACK[modality]
    gate = np.array([(r.get("attacks", {}).get(g) or {}).get("bit_acc") for r in recs], dtype=float)
    return out, gate, [r["item"] for r in recs]


def _included(modality: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = set(json.loads(RESULTS.read_text())["corpus_excluded"].get(modality, []))
    return [r for r in raw["records"] if r["item"] not in excluded]


def _sets(raw: dict[str, Any] | None, items: list[str]) -> list[str]:
    if raw is None:
        return []
    by_item = {r["item"]: r.get("set", "") for r in raw["records"]}
    return [by_item.get(i, "") for i in items]


def _exact_recovery(modality: str, raw: dict[str, Any] | None, meta: dict[str, Any],
                    k: int | None = None) -> tuple[dict[str, float], np.ndarray | None, dict[str, float]]:
    """Share of items whose decoded payload equals the embedded one, per transformation.

    TrustMark records `payload_exact`: its BCH decoder succeeded and returned the embedded
    payload. For the other payload-carrying methods, exact recovery means all payload bits
    match (`matches == nbits`); none of them has an error-correcting layer."""
    nbits = int(meta.get("nbits") or 0)
    if raw is None or nbits == 0:
        return {}, None, {}
    recs = _included(modality, raw)

    def ok(a: dict[str, Any] | None) -> float:
        if not a:
            return float("nan")
        if "payload_exact" in a:
            return float(bool(a["payload_exact"]))
        m = a.get("matches")
        return float("nan") if m is None else float(m == nbits)

    attacks = sorted({k for r in recs for k in r.get("attacks", {})})
    out = {}
    for atk in attacks:
        v = np.array([ok(r.get("attacks", {}).get(atk)) for r in recs])
        if np.isfinite(v).any():
            out[atk] = float(np.nanmean(v))
    gate = np.array([ok(r.get("attacks", {}).get(GATE_ATTACK[modality])) for r in recs])
    keym = {}
    if k is not None and k <= nbits:
        for atk in attacks:
            m = [(r.get("attacks", {}).get(atk) or {}).get("matches") for r in recs]
            m = [x for x in m if x is not None]
            if m:
                keym[atk] = float(np.mean([x >= k for x in m]))
    return out, gate, keym


def _latency(modality: str, c: dict[str, Any]) -> tuple[float | None, float | None]:
    lat, cpu = c.get("latency") or {}, c.get("latency_cpu") or {}
    if modality == "image":
        return lat.get("embed_ms_per_mp_median"), cpu.get("embed_ms_per_mp_median")
    if modality == "audio":
        return lat.get("embed_ms_per_audio_s_median"), cpu.get("embed_ms_per_audio_s_median")
    fps = lat.get("embed_fps") or c.get("quality", {}).get("embed_fps")
    # Video: ms per megapixel of 1080p frames, so image and video speed share a unit.
    return (1000.0 / fps / VIDEO_MP_PER_FRAME if fps else None), None


def _cost(modality: str, ms: float | None, price: float) -> float | None:
    """USD per 1,000 MP (image, video) or per 1,000 h of audio."""
    if ms is None:
        return None
    if modality == "audio":
        seconds_compute = ms / 1000.0 * 3600.0 * 1000.0
    else:
        seconds_compute = ms / 1000.0 * 1000.0
    return seconds_compute / 3600.0 * price


def load() -> dict[str, list[Config]]:
    res = json.loads(RESULTS.read_text())
    out: dict[str, list[Config]] = {}
    for modality in ("image", "audio", "video"):
        cfgs = []
        for c in res[modality]["configs"]:
            if "quality" not in c:
                continue
            meta, fpr = c["meta"], c.get("fpr", {})
            raw = _raw(modality, meta["name"])
            per_item, gate_items, items = _per_item(modality, raw)
            q = dict(c["quality"])
            groups = [source_of(modality, i) for i in items]
            if modality == "audio":
                q["abs_dlufs"] = q.get("abs_dlufs", abs(q.get("dlufs", 0.0)))
                # The pooled PESQ in results.json averages speech and music; PESQ is a speech
                # model, so the primary audio quality is PESQ on speech, with music reported apart.
                q["pesq_pooled"] = q.get("pesq")
                q["pesq"] = float(np.nanmean(per_item["pesq_speech"]))
                q["pesq_music"] = float(np.nanmean(per_item["pesq_music"]))
            qci = {k: bootstrap_ci(v, groups) for k, v in per_item.items()}
            if modality == "audio":
                qci["pesq"] = qci["pesq_speech"]
            exact, gate_exact, keymatch = _exact_recovery(modality, raw, meta, fpr.get("k_threshold"))
            gpu, cpu = _latency(modality, c)
            native = fpr.get("native_fpr")
            n_native = fpr.get("native_trials")
            if native is None and meta["family"] == "TrustMark":
                native = c.get("gate", {}).get("fpr_used") if c.get("gate", {}).get("fpr_source") == "native" else None
            blind_by_set = c.get("fpr_by_set") or {}
            if native is None and blind_by_set and meta["family"] == "TrustMark":
                native = None  # per-set only; pooled computed below
            k_trials = fpr.get("key_trials")
            key_upper = fpr.get("key_fpr_cp95")
            cfg = Config(
                modality=modality, name=meta["name"], family=meta["family"], label=_label(meta),
                strength=_strength(meta), open_licence=bool(meta["deployable"]), n=c["n_items"],
                quality=q, quality_ci=qci,
                robustness={a: v.get("bit_acc") if isinstance(v, dict) else v for a, v in c["robustness"].items()},
                gate_pooled=c["gate"]["value"], gate_worst=c["gate"]["worst_set_value"],
                gate_worst_set=c["gate"]["worst_set"],
                blind_fpr=native, blind_trials=n_native, blind_fpr_ci=None, blind_fpr_by_set=blind_by_set,
                key_fpr=fpr.get("key_fpr"), key_trials=k_trials, key_fpr_upper=key_upper,
                gpu_ms=gpu, cpu_ms=cpu, cpu_imputed=False, cost_gpu=None, cost_cpu=None,
                per_item=per_item, per_item_gate=gate_items, items=items, groups=groups,
                sets=_sets(raw, items), exact=exact, gate_exact_items=gate_exact, keymatch=keymatch,
                nbits=int(meta.get("nbits") or 0), k_threshold=fpr.get("k_threshold"),
            )
            cfgs.append(cfg)
        _blind_fpr_from_raw(modality, cfgs)
        _impute_cpu(cfgs)
        _pool_latency_by_network(cfgs)
        for cfg in cfgs:
            cfg.cost_gpu = _cost(modality, cfg.gpu_ms, PRICE_GPU_USD_H)
            cfg.cost_cpu = _cost(modality, cfg.cpu_ms, PRICE_CPU_USD_H)
        _score(modality, cfgs)
        out[modality] = cfgs
    return out


def _blind_fpr_from_raw(modality: str, cfgs: list[Config]) -> None:
    """Pooled blind false-positive rate and its exact interval from the raw trials.

    A blind detector decides "watermarked" without knowing the expected payload (the
    model's own detection head or ECC check). Trials are unmarked items under each
    FPR attack; the rate is detections / trials.
    """
    for cfg in cfgs:
        raw = _raw(modality, cfg.name)
        if raw is None:
            continue
        trials = [t for t in raw.get("fpr", []) if t.get("native_detect") is not None]
        if not trials:
            continue
        k = sum(bool(t["native_detect"]) for t in trials)
        n = len(trials)
        cfg.blind_fpr, cfg.blind_trials = k / n, n
        cfg.blind_fpr_ci = clopper_pearson(k, n)


def _impute_cpu(cfgs: list[Config]) -> None:
    """Fill a missing CPU latency only from another strength of the same network.

    Strength is a scalar on the residual, so the forward pass (and its cost) is the same
    across strengths of one model. Nothing is imputed across different models.
    """
    by_family: dict[str, list[float]] = {}
    for c in cfgs:
        if c.cpu_ms is not None:
            by_family.setdefault(c.family + ("-streaming" if "streaming" in c.name else ""), []).append(c.cpu_ms)
    for c in cfgs:
        key = c.family + ("-streaming" if "streaming" in c.name else "")
        if c.cpu_ms is None and c.family not in ("RivaGAN", "InvisMark", "WavMark") and by_family.get(key):
            c.cpu_ms = float(np.median(by_family[key]))
            c.cpu_imputed = True


def _pool_latency_by_network(cfgs: list[Config]) -> None:
    """Replace per-configuration latency with the median over the network's strengths.

    Repeated strengths of one network differ by up to ~20 % in measured latency with
    identical compute, which is timing noise. Pooling keeps that noise from deciding
    the speed and cost rankings. The raw per-configuration medians stay in
    `gpu_ms_raw` / `cpu_ms_raw` for the supplementary table.
    """
    groups: dict[str, list[Config]] = {}
    for c in cfgs:
        groups.setdefault(c.network, []).append(c)
    for members in groups.values():
        g = [c.gpu_ms for c in members if c.gpu_ms is not None]
        m = [c.cpu_ms for c in members if c.cpu_ms is not None and not c.cpu_imputed]
        for c in members:
            c.gpu_ms_raw, c.cpu_ms_raw = c.gpu_ms, (None if c.cpu_imputed else c.cpu_ms)
            if g:
                c.gpu_ms = float(np.median(g))
            if m:
                c.cpu_ms = float(np.median(m))
                c.cpu_imputed = c.cpu_ms_raw is None


def quality_ratio(modality: str, c: Config, best: Config) -> float:
    if modality == "image":
        return best.quality["flip"] / c.quality["flip"]
    if modality == "audio":
        return (c.quality["pesq"] - 1.0) / (best.quality["pesq"] - 1.0)
    return c.quality["vmaf"] / best.quality["vmaf"]


def primary_quality(modality: str) -> tuple[str, bool]:
    """(metric key, higher is better) used for the quality ratio."""
    return {"image": ("flip", False), "audio": ("pesq", True), "video": ("vmaf", True)}[modality]


def _score(modality: str, cfgs: list[Config]) -> None:
    q_key, higher = primary_quality(modality)
    pool = [c for c in cfgs if c.qualifies and c.gpu_ms is not None]
    if not pool:
        return
    best_q = (max if higher else min)(pool, key=lambda c: c.quality[q_key])
    best_r = max(c.gate_worst for c in pool)
    best_t = min(c.gpu_ms for c in pool)
    best_c = min(c.cost for c in pool)
    for c in pool:
        ratios = (quality_ratio(modality, c, best_q), (c.gate_worst - 0.5) / (best_r - 0.5), best_t / c.gpu_ms,
                  best_c / c.cost)
        c.scores = {"q": ratios[0], "r": ratios[1], "t": ratios[2], "c": ratios[3],
                    "combined": _gmean(ratios, WEIGHTS_EQUAL), "combined_q": _gmean(ratios, WEIGHTS_QUALITY)}


def _gmean(x: tuple[float, ...], w: tuple[float, ...]) -> float:
    return float(math.exp(sum(wi * math.log(max(xi, 1e-9)) for xi, wi in zip(x, w)) / sum(w)))


def winners(cfgs: list[Config]) -> dict[str, Config | None]:
    pool = [c for c in cfgs if c.scores]
    if not pool:
        return {"cost": None, "speed": None, "combined": None, "combined_q": None}
    # Strengths of one network tie on speed and cost; break ties by quality.
    q = {c.name: c.scores["q"] for c in pool}
    return {
        "cost": min(pool, key=lambda c: (round(c.cost, 12), -q[c.name])),
        "speed": min(pool, key=lambda c: (round(c.gpu_ms, 9), -q[c.name])),
        "combined": max(pool, key=lambda c: c.scores["combined"]),
        "combined_q": max(pool, key=lambda c: c.scores["combined_q"]),
    }


def paired_wilcoxon(a: Config, b: Config, metric: str) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank on per-source values, paired by source.

    Items of one source (an audio clip at three sample rates) are averaged first, so each
    independent source contributes one pair. Returns (median paired difference a - b,
    p value, n sources)."""
    def by_source(c: Config) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for g, v in zip(c.groups or c.items, c.per_item[metric]):
            if np.isfinite(v):
                acc.setdefault(g, []).append(float(v))
        return {g: float(np.mean(v)) for g, v in acc.items()}

    ia, ib = by_source(a), by_source(b)
    common = [k for k in ia if k in ib]
    da = np.array([ia[k] for k in common])
    db = np.array([ib[k] for k in common])
    r = stats.wilcoxon(da, db)
    return float(np.median(da - db)), float(r.pvalue), len(common)


def payload_null(n: int, k: int) -> float:
    """Per-key false-match probability P(Binomial(n, 1/2) >= k): the theoretical rate at which
    unmarked content passes an expected-payload test, if decoded bits are fair and independent."""
    return float(stats.binom.sf(k - 1, n, 0.5))


def key_trial_split(c: Config) -> dict[str, int] | None:
    """Expected-payload false matches on unmarked content, split by payload.

    Each unmarked trial is compared with the item's own payload (`matches[0]`) and with
    unrelated random payloads (`matches[1:]`). A verifier checks one fixed expected payload;
    the random payloads are additional draws of that same check against payloads the content
    was never marked with, which is what multiplies the trial count. Both follow the same
    null if decoded bits are independent of the payload."""
    raw = _raw(c.modality, c.name)
    if raw is None or not c.key_reachable:
        return None
    trials = [t["matches"] for t in raw.get("fpr", []) if t.get("matches")]
    k = int(c.k_threshold)
    own = [m[0] for m in trials]
    rnd = [x for m in trials for x in m[1:]]
    return {"own_n": len(own), "own_fm": sum(x >= k for x in own), "rnd_n": len(rnd), "rnd_fm": sum(x >= k for x in rnd)}


def max_matched_bits(c: Config) -> int | None:
    """Largest number of matching bits in any own- or random-payload comparison on unmarked
    content: the observed margin below the decision threshold k."""
    raw = _raw(c.modality, c.name)
    if raw is None:
        return None
    return max((x for t in raw.get("fpr", []) for x in t.get("matches") or []), default=None)


def fpr_by_source(c: Config) -> dict[str, Any] | None:
    """False positives with the unmarked source, not the trial, as the independent unit.

    The transformations of one item and the renderings of one audio clip share the source,
    and the random payloads are the same fixed list for every item, so neither adds
    independent media. A source counts as a false positive if any of its trials did: the
    own-payload test reached k, or the blind detector fired. Intervals are two-sided exact
    95 % Clopper-Pearson over sources."""
    raw = _raw(c.modality, c.name)
    if raw is None or not raw.get("fpr"):
        return None
    recs = raw["records"]
    k = int(c.k_threshold) if c.key_reachable and c.k_threshold else None
    own: dict[str, bool] = {}
    blind: dict[str, bool] = {}
    for t in raw["fpr"]:
        s = source_of(c.modality, recs[t["item"]]["item"])
        if k is not None and t.get("matches"):
            own[s] = own.get(s, False) or t["matches"][0] >= k
        if t.get("native_detect") is not None:
            blind[s] = blind.get(s, False) or bool(t["native_detect"])
    out: dict[str, Any] = {}
    if own:
        fm, n = sum(own.values()), len(own)
        out.update(own_src_fm=fm, own_src_n=n, own_src_upper=clopper_pearson(fm, n)[1])
    if blind:
        fm, n = sum(blind.values()), len(blind)
        out.update(blind_src_fm=fm, blind_src_n=n, blind_src_ci=clopper_pearson(fm, n))
    return out


def stability() -> dict[str, Any]:
    return json.loads(RESULTS.read_text())["stability"]


def corpus() -> dict[str, Any]:
    return json.loads((BENCH / "corpus_manifest.json").read_text())


def corpus_excluded() -> dict[str, Any]:
    return json.loads(RESULTS.read_text())["corpus_excluded"]
