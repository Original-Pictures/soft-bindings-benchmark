"""Video fingerprint systems (same system interface as fingerprints_audio).

    s = VIDEO_METHODS[name](); s.load(device)
    d = s.extract(path)               # descriptor from a video file
    s.build([d_ref, ...]); ids, scores = s.search(d_q, k); s.pair(d_q, d_ref)

TMK+PDQF is scored by its own C++ tool over (needle, haystack) lists, so it exposes
batch_scores() instead of search(); bench_video handles both shapes.

Frames for the per-frame methods are sampled at 1 fps (t = 0.5 s, 1.5 s, ...), the
rate vPDQ is usually run at, decoded with PyAV in RGB.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common import BENCH_WORK
from fingerprints_audio import AudioSystem, SegmentSystem, _topk
from fingerprints_image import DINOv2, SSCD

SRC = BENCH_WORK / "src"
TMK_BIN = Path(os.environ.get("TMK_BIN", SRC / "ThreatExchange" / "tmk" / "cpp"))
FPS = 1.0


def sample_frames(path: Path, fps: float = FPS) -> np.ndarray:
    """uint8 (N, H, W, 3) frames nearest to t = (k + 0.5) / fps."""
    import av

    out, want = [], 0.5 / fps
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base) if fr.pts is not None else 0.0
            if t + 1e-6 >= want:
                out.append(fr.to_ndarray(format="rgb24"))
                want += 1.0 / fps
    return np.stack(out) if out else np.zeros((0, 8, 8, 3), np.uint8)


VideoSystem = AudioSystem  # same dataclass fields and interface


# ------------------------------------------------------------------ vPDQ
class VPDQ(VideoSystem):
    """vPDQ (ThreatExchange, BSD): PDQ per sampled frame; a query matches a reference by the
    share of its frames (quality >= 50) that have a reference frame within Hamming 31.
    Score = query-frame match share (+1e-3 x reference share as tie-break), so sub-clips
    score like whole clips."""

    def __init__(self) -> None:
        super().__init__(name="vPDQ", family="PDQ", tier="P", code_licence="BSD (ThreatExchange vpdq)",
                         weights_licence="n/a", params={"seconds_per_hash": 1, "quality_tol": 50, "distance_tol": 31})

    def extract(self, path):
        import vpdq

        feats = vpdq.computeHash(str(path), seconds_per_hash=1)
        bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(f.hex), np.uint8)) for f in feats], np.uint8) \
            if feats else np.zeros((0, 256), np.uint8)
        q = np.array([f.quality for f in feats], np.int32)
        return bits[q >= self.params["quality_tol"]]

    def build(self, refs):
        import faiss

        self.refs = refs
        self.owner = np.concatenate([np.full(len(r), i) for i, r in enumerate(refs)])
        self.idx = faiss.IndexBinaryFlat(256)
        self.idx.add(np.packbits(np.concatenate(refs), axis=1))

    def _shares(self, q, r):
        if len(q) == 0 or len(r) == 0:
            return 0.0, 0.0
        d = (q[:, None, :] != r[None, :, :]).sum(-1)
        ok = d <= self.params["distance_tol"]
        return float(ok.any(1).mean()), float(ok.any(0).mean())

    def search(self, q, k):
        if len(q) == 0:
            return _topk({}, k)
        lims, D, I = self.idx.range_search(np.packbits(q, axis=1), self.params["distance_tol"] + 1)
        cands = set(int(self.owner[i]) for i in I)
        scores = {}
        for c in cands:
            a, b = self._shares(q, self.refs[c])
            scores[c] = a + 1e-3 * b
        return _topk(scores, k)

    def pair(self, q, r):
        a, b = self._shares(q, r)
        return a + 1e-3 * b

    def size_bits(self, d):
        return 256 * len(d)


# ------------------------------------------------------------------ TMK+PDQF
class TMK(VideoSystem):
    """TMK+PDQF (ThreatExchange, BSD): temporal match kernel over per-frame PDQ float
    features; fixed-size (~256 KB) signature. Level-1 = cosine of the time-averaged
    feature, level-2 = TMK time-kernel score; score = level-2 (Meta's defaults c1 = c2 = 0.7)."""

    def __init__(self) -> None:
        super().__init__(name="TMK+PDQF", family="PDQ", tier="P", code_licence="BSD (ThreatExchange tmk)",
                         weights_licence="n/a", params={"score": "level-2", "c1": 0.7, "c2": 0.7})

    def extract(self, path):
        out = Path(tempfile.mkdtemp()) / "x.tmk"
        subprocess.run([str(TMK_BIN / "tmk-hash-video"), "-f", os.environ.get("FFMPEG", "ffmpeg"), "-i", str(path),
                        "-o", str(out)], check=True, capture_output=True)
        return out.read_bytes()

    @staticmethod
    def batch_scores(needles: list[Path], haystack: list[Path]) -> dict[tuple[str, str], tuple[float, float]]:
        with tempfile.TemporaryDirectory() as d:
            nf, hf = Path(d) / "n.txt", Path(d) / "h.txt"
            nf.write_text("\n".join(map(str, needles)) + "\n")
            hf.write_text("\n".join(map(str, haystack)) + "\n")
            # the OpenMP build of Meta's tool; all cores (the bench caps OMP threads for its workers)
            exe = TMK_BIN / "tmk-query-parallel"
            exe = exe if exe.exists() else TMK_BIN / "tmk-query"
            env = {**os.environ, "OMP_NUM_THREADS": str(os.cpu_count() or 1)}
            out = subprocess.run([str(exe), "--c1", "-1", "--c2", "-1", str(nf), str(hf)],
                                 capture_output=True, text=True, check=True, env=env).stdout
        res = {}
        for line in out.splitlines():
            p = line.split()
            if len(p) == 4:
                res[(p[2], p[3])] = (float(p[0]), float(p[1]))
        return res

    def size_bits(self, d):
        return 8 * len(d)


# ------------------------------------------------------------------ ISCC Video-Code
class ISCCVideo(VideoSystem):
    """ISCC Video-Code v0 (ISO 24138, Apache-2.0): WTA-hash over MPEG-7 frame signatures
    (ffmpeg `signature` filter). C2PA-registered as part of io.iscc.v0."""

    def __init__(self, bits: int = 64) -> None:
        super().__init__(name=f"ISCC-Video-{bits}", family="ISCC", tier="P", code_licence="Apache-2.0",
                         weights_licence="n/a", params={"bits": bits},
                         note="MPEG-7 signatures via FFmpeg (LGPL build)")
        self.bits = bits

    def extract(self, path):
        import iscc_core as ic
        import iscc_sdk as idk

        sigs = idk.video_mp7sig_extract(str(path))
        frames = idk.read_mp7_signature(sigs)
        vecs = [tuple(f.vector.tolist()) for f in frames]
        body = ic.Code(ic.gen_video_code_v0(vecs, bits=self.bits)["iscc"]).hash_bytes
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


# ------------------------------------------------------------------ neural, per frame
class FrameNeural(SegmentSystem):
    """Image copy-detection descriptor on 1 fps frames. mode="seq": frame-sequence matching
    with temporal voting (the VSC2022 baseline recipe); mode="mean": one clip descriptor
    (mean of frame descriptors, re-normalised) compared by cosine."""

    def __init__(self, base: str, mode: str) -> None:
        mk = {"SSCD-mixup": lambda: SSCD("mixup"), "DINOv2-S": lambda: DINOv2("S")}[base]
        b = mk()
        super().__init__(name=f"{base}-{mode}", family=b.family, tier=b.tier, gpu=True, code_licence=b.code_licence,
                         weights_licence=b.weights_licence, params={"frames_fps": FPS, "mode": mode})
        self.base, self.mode = mk, mode

    def load(self, device="cpu"):
        super().load(device)
        self.b = self.base()
        self.b.load(device)

    def extract(self, path):
        fr = sample_frames(Path(path))
        if len(fr) == 0:
            return np.zeros((0, self.b.dim), np.float32)
        return self.b.extract([f.astype(np.float32) / 255.0 for f in fr])

    def _mean(self, d):
        v = d.mean(0)
        return v / max(np.linalg.norm(v), 1e-12)

    def build(self, refs):
        if self.mode == "seq":
            return super().build(refs)
        self.R = np.stack([self._mean(r) for r in refs]).astype(np.float32)

    def search(self, q, k):
        if self.mode == "seq":
            return super().search(q, k)
        if len(q) == 0:
            return _topk({}, k)
        s = self.R @ self._mean(q)
        o = np.argsort(-s)[:k]
        return o.astype(np.int64), s[o].astype(np.float32)

    def pair(self, q, r):
        if self.mode == "seq":
            return super().pair(q, r)
        if len(q) == 0 or len(r) == 0:
            return -1.0
        return float(self._mean(q) @ self._mean(r))

    def size_bits(self, d):
        return 32 * (d.size if self.mode == "seq" else d.shape[1])


VIDEO_METHODS: dict[str, Callable[[], VideoSystem]] = {
    "vPDQ": VPDQ,
    "TMK+PDQF": TMK,
    "ISCC-Video-64": lambda: ISCCVideo(64),
    "ISCC-Video-256": lambda: ISCCVideo(256),
    "DINOv2-S-seq": lambda: FrameNeural("DINOv2-S", "seq"),
    "DINOv2-S-mean": lambda: FrameNeural("DINOv2-S", "mean"),
    "SSCD-mixup-seq": lambda: FrameNeural("SSCD-mixup", "seq"),
    "SSCD-mixup-mean": lambda: FrameNeural("SSCD-mixup", "mean"),
}
