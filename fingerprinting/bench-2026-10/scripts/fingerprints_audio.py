"""Audio fingerprint systems behind one interface.

Audio identification is not uniformly "one vector per file": landmark systems store
hash tables and vote on time offsets, Chromaprint compares frame sequences at the best
alignment, neural fingerprinters index 1 s segments. So each method is a *system*:

    s = AUDIO_METHODS[name](); s.load(device)
    d = s.extract(y, sr)              # y float32 mono; descriptor is method-specific
    s.build([d_ref, ...])             # index the references
    ids, scores = s.search(d_q, k)    # top-k reference indices, higher score = better
    s.pair(d_q, d_ref)                # score of one query against one reference

Scores are on each method's own scale (BER-based similarity, aligned-hash counts,
cosine); thresholds are always calibrated per method on held-out negatives.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import BENCH_WORK

SRC = BENCH_WORK / "src"


def _resample(y: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return y.astype(np.float32)
    import soxr

    return soxr.resample(y.astype(np.float32), sr, to, quality="HQ").astype(np.float32)


@dataclass
class AudioSystem:
    name: str
    family: str
    tier: str
    code_licence: str
    weights_licence: str
    gpu: bool = False
    note: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def load(self, device: str = "cpu") -> None:
        self.device = device

    def extract(self, y: np.ndarray, sr: int) -> Any:
        raise NotImplementedError

    def build(self, refs: list[Any]) -> None:
        raise NotImplementedError

    def search(self, q: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def pair(self, q: Any, r: Any) -> float:
        raise NotImplementedError

    def size_bits(self, d: Any) -> int:
        """Stored bits per reference descriptor."""
        raise NotImplementedError

    def meta(self) -> dict[str, Any]:
        keys = ("name", "family", "tier", "code_licence", "weights_licence", "gpu", "note", "params")
        return {k: getattr(self, k) for k in keys}


def _topk(scores: dict[int, float], k: int) -> tuple[np.ndarray, np.ndarray]:
    top = sorted(scores.items(), key=lambda t: -t[1])[:k]
    ids = np.full(k, -1, np.int64)
    sc = np.full(k, -np.inf, np.float32)
    for j, (i, s) in enumerate(top):
        ids[j], sc[j] = i, s
    return ids, sc


# ------------------------------------------------------------------ Chromaprint
def fpcalc(y: np.ndarray, sr: int) -> np.ndarray:
    """Chromaprint raw fingerprint (algorithm 2, the AcoustID default) as uint32."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, np.clip(y, -1, 1), sr, subtype="PCM_16")
        out = subprocess.run(["fpcalc", "-raw", "-signed", "-json", "-length", "0", f.name],
                             capture_output=True, text=True, check=True).stdout
    return np.array(json.loads(out)["fingerprint"], dtype=np.int64).astype(np.uint32)


_POP = np.array([bin(i).count("1") for i in range(1 << 16)], np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    return _POP[x & 0xFFFF].astype(np.int32) + _POP[x >> 16].astype(np.int32)


def chroma_similarity(q: np.ndarray, r: np.ndarray, offsets=None, min_overlap: int = 20) -> float:
    """1 - bit error rate at the best alignment (AcoustID-style comparison)."""
    if len(q) == 0 or len(r) == 0:
        return 0.0
    best = 0.0
    offs = range(-(len(q) - 1), len(r)) if offsets is None else offsets
    for o in offs:
        qa, ra = max(0, -o), max(0, o)
        n = min(len(q) - qa, len(r) - ra)
        if n < min(min_overlap, len(q)):
            continue
        ber = _popcount(q[qa:qa + n] ^ r[ra:ra + n]).sum() / (32.0 * n)
        best = max(best, 1.0 - ber)
    return float(best)


class Chromaprint(AudioSystem):
    """Chromaprint 1.6 (fpcalc). Candidates: exact matches on the 20 most significant bits
    of each 32-bit sub-fingerprint voted by time offset (the AcoustID index design); the
    top candidates are then verified by bit error rate at the voted offset +-2."""

    MASK = np.uint32(0xFFFFF000)

    def __init__(self) -> None:
        super().__init__(name="Chromaprint", family="Chromaprint", tier="P",
                         code_licence="MIT (chromaprint core; fpcalc binary links FFmpeg, LGPL-2.1)",
                         weights_licence="n/a", params={"algorithm": 2, "index_bits": 20, "verify": 20})

    def extract(self, y, sr):
        return fpcalc(y, sr)

    def build(self, refs):
        self.refs = refs
        self.inv: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, r in enumerate(refs):
            for t, v in enumerate(r & self.MASK):
                self.inv[int(v)].append((i, t))

    def search(self, q, k):
        votes: Counter = Counter()
        for t, v in enumerate(q & self.MASK):
            for i, rt in self.inv.get(int(v), ()):
                votes[(i, rt - t)] += 1
        cand: dict[int, int] = {}
        for (i, off), c in votes.most_common(200):
            if i not in cand:
                cand[i] = off
            if len(cand) >= self.params["verify"]:
                break
        scores = {i: chroma_similarity(q, self.refs[i], range(off - 2, off + 3)) for i, off in cand.items()}
        return _topk(scores, k)

    def pair(self, q, r):
        return chroma_similarity(q, r)

    def size_bits(self, d):
        return 32 * len(d)


class ISCCAudio(AudioSystem):
    """ISCC Audio-Code v0 (ISO 24138, iscc-core Apache-2.0): SimHash over the whole-file
    Chromaprint vector (plus quarter/third buckets for 128+ bits). A single code per file,
    so it is not designed for excerpt queries."""

    def __init__(self, bits: int = 64) -> None:
        super().__init__(name=f"ISCC-Audio-{bits}", family="ISCC", tier="P", code_licence="Apache-2.0",
                         weights_licence="n/a", params={"bits": bits}, note="C2PA-registered io.iscc.v0")
        self.bits = bits

    def extract(self, y, sr):
        import iscc_core as ic

        cv = fpcalc(y, sr).astype(np.int64)
        cv = np.where(cv >= 2**31, cv - 2**32, cv).tolist()
        body = ic.Code(ic.gen_audio_code_v0(cv, bits=self.bits)["iscc"]).hash_bytes
        return np.unpackbits(np.frombuffer(body, np.uint8))[: self.bits]

    def build(self, refs):
        self.R = np.stack(refs).astype(np.float32)

    def search(self, q, k):
        s = 1 - np.abs(self.R - q[None].astype(np.float32)).mean(1)
        o = np.argsort(-s)[:k]
        return o.astype(np.int64), s[o].astype(np.float32)

    def pair(self, q, r):
        return float(1 - (q != r).mean())

    def size_bits(self, d):
        return self.bits


# ------------------------------------------------------------------ audfprint (landmarks)
class Audfprint(AudioSystem):
    """audfprint (D. Ellis, MIT): Shazam-style spectral-peak landmark hashes, score = number
    of hashes agreeing on one time offset (the tool's filtered match count)."""

    def __init__(self, density: float = 20.0) -> None:
        super().__init__(name="audfprint", family="Landmark", tier="P", code_licence="MIT", weights_licence="n/a",
                         params={"density": density, "sr": 11025, "hashbits": 20, "depth": 100})

    def load(self, device="cpu"):
        super().load(device)
        sys.path.insert(0, str(SRC / "audfprint"))
        import audfprint_analyze
        import audfprint_match
        import hash_table

        self.ana_mod, self.ht_mod = audfprint_analyze, hash_table
        self.an = audfprint_analyze.Analyzer(density=self.params["density"])
        self.matcher = audfprint_match.Matcher()
        self.matcher.threshcount = 1
        self.matcher.max_returns = 100

    def extract(self, y, sr):
        d = _resample(y, sr, 11025)
        peaks = self.an.find_peaks(d, 11025)
        if len(peaks) == 0:
            return np.zeros((0, 2), np.int32)
        h = self.ana_mod.landmarks2hashes(self.an.peaks2landmarks(peaks))
        return np.unique(h, axis=0).astype(np.int32)

    def build(self, refs):
        self.ht = self.ht_mod.HashTable(hashbits=20, depth=100, maxtime=16384)
        for i, r in enumerate(refs):
            self.ht.store(str(i), r)

    def search(self, q, k):
        if len(q) == 0:
            return _topk({}, k)
        res = self.matcher.match_hashes(self.ht, q)
        # One row per (reference, time offset), sorted by count: keep each reference's best row.
        scores: dict[int, float] = {}
        for row in res:
            ref = int(self.ht.names[int(row[0])])
            scores[ref] = max(scores.get(ref, 0.0), float(row[1]))
        return _topk(scores, k)

    def pair(self, q, r):
        if len(q) == 0 or len(r) == 0:
            return 0.0
        # aligned-count between two hash lists: histogram of time differences for equal hashes
        rd = defaultdict(list)
        for t, h in r:
            rd[int(h)].append(int(t))
        diffs = Counter(rt - int(t) for t, h in q for rt in rd.get(int(h), ()))
        if not diffs:
            return 0.0
        best = max(diffs, key=diffs.get)
        return float(sum(diffs.get(best + w, 0) for w in (-1, 0, 1)))

    def size_bits(self, d):
        return 64 * len(d)


# ------------------------------------------------------------------ CLAP (semantic embedding)
class CLAP(AudioSystem):
    """LAION-CLAP (code CC0-1.0; checkpoint 630k-audioset-best). A semantic audio embedding,
    the audio counterpart of CLIP: 10 s windows at 48 kHz, mean-pooled, L2-normalised."""

    def __init__(self) -> None:
        super().__init__(name="CLAP", family="CLAP", tier="P", gpu=True, code_licence="CC0-1.0",
                         weights_licence="CC0-1.0 (LAION-AI/CLAP release)", params={"ckpt": "630k-audioset-best"})

    def load(self, device="cpu"):
        import laion_clap

        super().load(device)
        self.m = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        self.m.load_ckpt(str(BENCH_WORK / "weights" / "clap" / "630k-audioset-best.pt"), verbose=False)

    def extract(self, y, sr):
        import torch

        x = _resample(y, sr, 48000)
        win = 480000
        segs = [x[i:i + win] for i in range(0, max(1, len(x) - win // 2), win)] or [x]
        segs = [np.pad(s, (0, win - len(s))) if len(s) < win else s for s in segs]
        with torch.no_grad():
            e = self.m.get_audio_embedding_from_data(x=np.stack(segs).astype(np.float32), use_tensor=False)
        v = e.mean(0)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def build(self, refs):
        self.R = np.stack(refs)

    def search(self, q, k):
        s = self.R @ q
        o = np.argsort(-s)[:k]
        return o.astype(np.int64), s[o].astype(np.float32)

    def pair(self, q, r):
        return float(q @ r)

    def size_bits(self, d):
        return 32 * len(d)


# ------------------------------------------------------------------ segment-embedding systems
class SegmentSystem(AudioSystem):
    """Matching for fingerprinters that emit one embedding per short segment (NAFP-style):
    every query segment retrieves its nearest reference segments; hits vote for (reference,
    offset); the score of a candidate is the mean best similarity of the query segments
    aligned at its winning offset."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)

    def build(self, refs):
        import faiss

        self.refs = refs
        segs = np.concatenate(refs).astype(np.float32)
        self.owner = np.concatenate([np.full(len(r), i) for i, r in enumerate(refs)])
        self.pos = np.concatenate([np.arange(len(r)) for r in refs])
        self.index = faiss.IndexFlatIP(segs.shape[1])
        self.index.add(segs)

    def search(self, q, k):
        if len(q) == 0:
            return _topk({}, k)
        S, I = self.index.search(np.ascontiguousarray(q, np.float32), 20)
        votes: Counter = Counter()
        for t in range(len(q)):
            for s, j in zip(S[t], I[t]):
                votes[(int(self.owner[j]), int(self.pos[j]) - t)] += 1
        cand: dict[int, int] = {}
        for (i, off), _ in votes.most_common(100):
            cand.setdefault(i, off)
            if len(cand) >= 20:
                break
        return _topk({i: self._aligned(q, self.refs[i], off) for i, off in cand.items()}, k)

    @staticmethod
    def _aligned(q, r, off):
        best = -1.0
        for o in (off - 1, off, off + 1):
            idx = np.arange(len(q)) + o
            ok = (idx >= 0) & (idx < len(r))
            if ok.sum() == 0:
                continue
            best = max(best, float((q[ok] * r[idx[ok]]).sum(1).mean()))
        return best

    def pair(self, q, r):
        if len(q) == 0 or len(r) == 0:
            return -1.0
        sim = q @ r.T
        return max(self._aligned(q, r, off) for off in range(-len(q) + 1, len(r)))

    def size_bits(self, d):
        return 32 * d.size


class NMFP(SegmentSystem):
    """Neural Music Fingerprint (Araz et al., 2025; raraz15/neural-music-fp, GPL-3.0), triplet
    model from Zenodo 15719945, 8 kHz, 1 s segments, 0.5 s hop, 128-d. Runs in its own
    TensorFlow environment (nmfp_extract.py); this class only matches the embeddings."""

    def __init__(self) -> None:
        super().__init__(name="NMFP-triplet", family="NeuralFP", tier="R", gpu=True, code_licence="GPL-3.0",
                         weights_licence="Zenodo 15719945 (licence as repo, GPL-3.0); trained on FMA",
                         params={"segment_s": 1.0, "hop_s": 0.5, "dim": 128})

    def extract(self, y, sr):
        raise RuntimeError("NMFP descriptors come from nmfp_extract.py (TensorFlow env)")


AUDIO_METHODS: dict[str, Callable[[], AudioSystem]] = {
    "Chromaprint": Chromaprint,
    "ISCC-Audio-64": lambda: ISCCAudio(64),
    "ISCC-Audio-256": lambda: ISCCAudio(256),
    "audfprint": Audfprint,
    "CLAP": CLAP,
    "NMFP-triplet": NMFP,
}
