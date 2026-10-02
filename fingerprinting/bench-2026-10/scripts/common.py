"""Shared paths, seeding, hashing and result I/O for the fingerprint bench.

Copied from the 2026-09 watermark bench (same Timer/host_info/write_json), with the
work directory and seed changed. Media never lives in the repo: corpus, attacked
queries and descriptor caches go to $BENCH_WORK (default ~/op-fp-bench-work). Only JSON results and small thumbnails
are written under the bench directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

try:  # iscc-sdk / iscc-sci log every call at DEBUG through loguru
    from loguru import logger as _loguru

    _loguru.remove()
    _loguru.add(lambda m: None, level="WARNING")
except ImportError:
    pass

SCRIPTS = Path(__file__).resolve().parent
BENCH_DIR = SCRIPTS.parent
BENCH_WORK = Path(os.environ.get("BENCH_WORK", Path.home() / "op-fp-bench-work")).resolve()
CORPUS = BENCH_WORK / "corpus"
WEIGHTS = BENCH_WORK / "weights"
OUT = BENCH_WORK / "out"
RESULTS = Path(os.environ.get("BENCH_RESULTS", BENCH_DIR / "results"))

# One fixed seed for every random choice (corpus sampling, attack parameters,
# payloads for the watermark-combination track), so every method sees the same queries.
SEED = 20260926
DESC = BENCH_WORK / "desc"  # descriptor caches: DESC/<set>/<method>.npy


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def payload(nbits: int, index: int, salt: str = "") -> np.ndarray:
    """Deterministic pseudo-random payload for item `index`."""
    digest = hashlib.sha256(f"{SEED}:{salt}:{index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.integers(0, 2, nbits, dtype=np.uint8)


def bit_accuracy(sent: np.ndarray, got: np.ndarray | None) -> float:
    if got is None:
        return 0.0
    got = np.asarray(got).astype(np.uint8).reshape(-1)[: len(sent)]
    if len(got) < len(sent):
        return 0.0
    return float((got == sent).mean())


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        pass
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("model name"))
    except Exception:
        import subprocess

        try:
            info["cpu_model"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        except Exception:
            pass
    return info


def device_name(requested: str | None = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class Timer:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        sync(self.device)
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        sync(self.device)
        self.elapsed = time.perf_counter() - self._t


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default) + "\n")
    tmp.replace(path)


def _default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def rng_for(*keys: object) -> np.random.Generator:
    """Deterministic generator for a named decision (e.g. rng_for("attack", "rot", item_id))."""
    digest = hashlib.sha256(":".join(map(str, (SEED, *keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def load_rgb(path: Path, max_side: int | None = None) -> np.ndarray:
    """float32 HxWx3 in [0,1]; EXIF orientation applied; optional downscale (bicubic)."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
        return np.asarray(im, dtype=np.float32) / 255.0


def to_u8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def extra_sets(modality: str) -> dict[str, list[tuple[str, Path, str]]]:
    """Additional descriptor sets registered by other tracks (e.g. watermarked copies for the
    watermark-combination track): OUT/sets/<modality>/<set>.json = [[key, path, attack], ...].
    For images `path` is the source and `attack` is applied at load time; for audio and
    video `path` is the already-rendered query and `attack` is only a label."""
    d = OUT / "sets" / modality
    out = {}
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        out[f.stem] = [(k, Path(p), a) for k, p, a in json.loads(f.read_text())]
    return out


def register_set(modality: str, name: str, items: list[tuple[str, Path, str]]) -> None:
    write_json(OUT / "sets" / modality / f"{name}.json", [[k, str(p), a] for k, p, a in items])
