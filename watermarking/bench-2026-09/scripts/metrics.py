"""Perceptual / fidelity metrics.

Images are float32 HxWx3 in [0,1] (display-referred sRGB). Audio is float32 mono.
Lower-is-better: LPIPS, FLIP, spectral_diff_db, abs dLUFS. Higher-is-better: the rest.
CVVDP is reported in JOD (10 = indistinguishable).
"""

from __future__ import annotations

import math
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(ref, test, channel_axis=2, data_range=1.0))


@lru_cache(maxsize=None)
def _lpips(device: str):
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def _to_t(x: np.ndarray, device: str):
    import torch

    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().to(device)


def lpips_score(ref: np.ndarray, test: np.ndarray, device: str) -> float:
    import torch

    with torch.no_grad():
        return float(_lpips(device)(_to_t(ref, device) * 2 - 1, _to_t(test, device) * 2 - 1).item())


def ms_ssim(ref: np.ndarray, test: np.ndarray, device: str) -> float:
    import torch
    from pytorch_msssim import ms_ssim as _ms

    if min(ref.shape[:2]) < 161:
        return float("nan")
    with torch.no_grad():
        return float(_ms(_to_t(ref, device), _to_t(test, device), data_range=1.0).item())


def flip(ref: np.ndarray, test: np.ndarray, want_map: bool = False):
    import flip_evaluator

    err_map, mean, _ = flip_evaluator.evaluate(np.ascontiguousarray(ref, dtype=np.float32),
                                              np.ascontiguousarray(test, dtype=np.float32),
                                              "LDR", applyMagma=want_map)
    return (float(mean), err_map) if want_map else float(mean)


@lru_cache(maxsize=None)
def _cvvdp(device: str):
    import pycvvdp
    import torch

    return pycvvdp.cvvdp(display_name="standard_4k", quiet=True, device=torch.device(device))


def cvvdp_image(ref: np.ndarray, test: np.ndarray, device: str) -> float:
    import torch

    if device == "mps":  # pycvvdp uses float64 ops unsupported on MPS
        device = "cpu"
    with torch.no_grad():
        jod, _ = _cvvdp(device).predict(torch.from_numpy(test).float(), torch.from_numpy(ref).float(), dim_order="HWC")
    return float(jod)


def cvvdp_video(ref_frames: np.ndarray, test_frames: np.ndarray, fps: float, device: str) -> float:
    """ref/test: F x H x W x 3 float [0,1]."""
    import torch

    if device == "mps":
        device = "cpu"
    with torch.no_grad():
        jod, _ = _cvvdp(device).predict(torch.from_numpy(test_frames).float(), torch.from_numpy(ref_frames).float(),
                                        dim_order="FHWC", frames_per_second=fps)
    return float(jod)


def vmaf(ref_path: Path, test_path: Path, threads: int = 4) -> float:
    """VMAF (vmaf_v0.6.1) via ffmpeg libvmaf. Returns pooled mean."""
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as log:
        cmd = ["ffmpeg", "-v", "error", "-i", str(test_path), "-i", str(ref_path), "-lavfi",
               f"[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];[d][r]libvmaf=log_fmt=json:log_path={log.name}:n_threads={threads}",
               "-f", "null", "-"]
        subprocess.run(cmd, check=True)
        data = json.loads(Path(log.name).read_text())
    return float(data["pooled_metrics"]["vmaf"]["mean"])


# ---------------------------------------------------------------- audio
def snr(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    noise = np.sum((ref - test) ** 2)
    return 100.0 if noise == 0 else 10 * math.log10(np.sum(ref ** 2) / noise)


def si_snr(ref: np.ndarray, test: np.ndarray) -> float:
    n = min(len(ref), len(test))
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    ref = ref - ref.mean()
    test = test - test.mean()
    s = np.dot(test, ref) / (np.dot(ref, ref) + 1e-12) * ref
    e = test - s
    return 100.0 if np.sum(e ** 2) == 0 else 10 * math.log10(np.sum(s ** 2) / np.sum(e ** 2))


def _to16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return x
    import soxr

    return soxr.resample(x, sr, 16000, quality="VHQ")


def pesq_wb(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
    from pesq import pesq

    r, t = _to16k(ref, sr), _to16k(test, sr)
    n = min(len(r), len(t))
    try:
        return float(pesq(16000, r[:n], t[:n], "wb"))
    except Exception:
        return float("nan")


def stoi(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
    from pystoi import stoi as _stoi

    n = min(len(ref), len(test))
    return float(_stoi(ref[:n], test[:n], sr, extended=False))


def delta_lufs(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    n = min(len(ref), len(test))
    return float(meter.integrated_loudness(test[:n].astype(np.float64)) - meter.integrated_loudness(ref[:n].astype(np.float64)))


def spectral_diff_db(ref: np.ndarray, test: np.ndarray, sr: int) -> float:
    """Mean absolute log-magnitude STFT difference (dB) over bins above -80 dBFS."""
    import librosa

    n = min(len(ref), len(test))
    R = np.abs(librosa.stft(ref[:n], n_fft=2048, hop_length=512))
    T = np.abs(librosa.stft(test[:n], n_fft=2048, hop_length=512))
    Rd = 20 * np.log10(R + 1e-8)
    Td = 20 * np.log10(T + 1e-8)
    mask = Rd > (Rd.max() - 80)
    return float(np.mean(np.abs(Rd - Td)[mask]))
