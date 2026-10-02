"""Surrogate hashes must agree with the real extractors on clean images (>= 90% of bits),
otherwise their gradients do not point where the real hash moves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import CORPUS, load_rgb  # noqa: E402
from fingerprints_image import IMAGE_METHODS  # noqa: E402
import hash_surrogates as hs  # noqa: E402


def _images(n: int = 12) -> list[np.ndarray]:
    man = CORPUS / "corpus_manifest.json"
    if man.exists():
        a = json.loads(man.read_text())["image"]
        return [load_rgb(CORPUS / a["items"][i]["path"], 1024) for i in a["sets"]["reg"][:n]]
    rng = np.random.default_rng(0)
    base = rng.random((n, 6, 8, 3)).astype(np.float32)
    return [np.asarray(torch.nn.functional.interpolate(torch.from_numpy(b.transpose(2, 0, 1))[None], size=(300, 400),
                                                        mode="bicubic")[0].clamp(0, 1).permute(1, 2, 0)) for b in base]


@pytest.mark.parametrize("name", list(hs.SURROGATES))
def test_surrogate_bit_agreement(name: str) -> None:
    imgs = _images()
    m = IMAGE_METHODS[name]()
    m.load("cpu")
    real = m.extract(imgs)
    agree = []
    for x, r in zip(imgs, real):
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        kw = {"box": hs.trim_box(t)} if name == "ISCC-Image-64" else {}
        s = (hs.SURROGATES[name](t, **kw)[0] > 0).numpy().astype(np.uint8)
        agree.append((s == r).mean())
    assert np.mean(agree) >= 0.90, (name, float(np.mean(agree)))
