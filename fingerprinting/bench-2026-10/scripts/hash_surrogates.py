"""Differentiable surrogates of the DCT/mean-family perceptual hashes.

An attacker who cannot differentiate PDQ or pHash can still differentiate a faithful
re-implementation of their pipeline: grey conversion, low-pass resize, DCT, median
threshold. Bits become soft through tanh(k * (v - median) / scale). The surrogate is
only a gradient source: every attack result is re-scored with the REAL extractor on
the uint8-quantised image (bench_security), so a poor surrogate can only make the
attack weaker, never inflate its success.

Surrogate differences from the originals (bit agreement is checked in
tests/test_security.py):
  * PDQ: the two Jarosz box passes + centre decimation to 64x64 are replaced by a
    single antialiased (triangle) resize; the DCT is PDQ's 16x64 matrix with rows 1..16
    (no DC), threshold at the median of the 256 coefficients.
  * pHash/aHash/dHash: PIL's Lanczos resize is replaced by an antialiased bicubic resize.
  * ISCC-Image-64: the SDK's uniform-border trim box is passed in (see trim_box).
Bit order follows each library's output order (row-major over the coefficient grid).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)


def grey(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [0,1] -> (B,1,H,W) luma in [0,255]."""
    w = torch.tensor(LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True) * 255.0


def resize(x: torch.Tensor, h: int, w: int, mode: str = "bicubic") -> torch.Tensor:
    # antialias backward is not implemented for every backend; fall back to area pooling
    try:
        return F.interpolate(x, size=(h, w), mode=mode, align_corners=False, antialias=True)
    except (NotImplementedError, RuntimeError):
        return F.interpolate(x, size=(h, w), mode="area")


def dct_matrix(n_out: int, n_in: int, start: int = 0, device=None, dtype=torch.float32) -> torch.Tensor:
    """Unnormalised DCT-II rows start..start+n_out-1 (a uniform scale does not move a median)."""
    i = torch.arange(start, start + n_out, device=device, dtype=dtype)[:, None]
    j = torch.arange(n_in, device=device, dtype=dtype)[None, :]
    return torch.cos(math.pi / n_in * (j + 0.5) * i)


def _soft(v: torch.Tensor, k: float) -> torch.Tensor:
    """v: (B, n) coefficients -> soft bits in (-1, 1), median-centred and scale-normalised."""
    med = v.median(dim=1, keepdim=True).values
    scale = (v - med).abs().mean(dim=1, keepdim=True) + 1e-6
    return torch.tanh(k * (v - med) / scale)


def pdq(x: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    g = resize(grey(x), 64, 64, "bilinear")[:, 0]
    D = dct_matrix(16, 64, start=1, device=x.device, dtype=x.dtype)
    B = D @ g @ D.T
    # pdqhash returns the 256 bits most-significant word first: reversed row-major order
    return _soft(B.reshape(len(x), -1), k).flip(1)


def phash(x: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    g = resize(grey(x), 32, 32)[:, 0]
    D = dct_matrix(8, 32, device=x.device, dtype=x.dtype)
    return _soft((D @ g @ D.T).reshape(len(x), -1), k)


def ahash(x: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    g = resize(grey(x), 8, 8)[:, 0].reshape(len(x), -1)
    m = g.mean(1, keepdim=True)
    return torch.tanh(k * (g - m) / ((g - m).abs().mean(1, keepdim=True) + 1e-6))


def dhash(x: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    g = resize(grey(x), 8, 9)[:, 0]
    # imagehash compares uint8 pixels (equal neighbours -> 0), i.e. a float difference of
    # at least half a grey level after rounding; flat product backgrounds depend on this.
    d = (g[:, :, 1:] - g[:, :, :-1]).reshape(len(x), -1) - 0.5
    return torch.tanh(k * d / (d.abs().mean(1, keepdim=True) + 1e-6))


def trim_box(x) -> tuple[int, int, int, int] | None:
    """ISCC SDK uniform-border trim box (left, top, right, bottom) of one image, or None.
    Not differentiable; recomputed on the current image at each attack step, because a
    perturbation that touches the border switches trimming off in the real extractor."""
    import numpy as np
    from PIL import Image, ImageChops

    a = x.detach()[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy() if torch.is_tensor(x) else x
    img = Image.fromarray(np.asarray(a))
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    d = ImageChops.difference(img, bg)
    bbox = ImageChops.add(d, d).getbbox()
    return None if bbox is None or bbox == (0, 0) + img.size else bbox


def iscc64(x: torch.Tensor, k: float = 4.0, box=None) -> torch.Tensor:
    if box is not None:
        x = x[:, :, box[1]:box[3], box[0]:box[2]]
    g = resize(grey(x), 32, 32)[:, 0]
    D = dct_matrix(8, 32, device=x.device, dtype=x.dtype)
    return _soft((D @ g @ D.T).reshape(len(x), -1), k)


# method name -> surrogate; outputs are ordered like the real extractor's bits
SURROGATES = {"PDQ": pdq, "pHash-64": phash, "aHash-64": ahash, "dHash-64": dhash, "ISCC-Image-64": iscc64}
