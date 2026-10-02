"""Adapter sanity: identity scores, determinism, and the vectorised Blockhash against the
pinned reference implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import BENCH_WORK  # noqa: E402
from fingerprints_image import IMAGE_METHODS, Blockhash  # noqa: E402

RNG = np.random.default_rng(0)


def _img(h, w):
    # smooth random image (structure at several scales) rather than white noise
    base = RNG.random((h // 8 + 2, w // 8 + 2, 3))
    from PIL import Image

    im = Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255


@pytest.mark.parametrize("hw", [(256, 256), (300, 457), (1024, 683), (17, 33), (768, 512)])
def test_blockhash_matches_reference(hw):
    sys.path.insert(0, str(BENCH_WORK / "src" / "blockhash-python"))
    import blockhash
    from PIL import Image

    x = _img(*hw)
    u8 = (np.clip(np.round(x * 255), 0, 255)).astype(np.uint8)
    ref = np.unpackbits(np.frombuffer(bytes.fromhex(blockhash.blockhash(Image.fromarray(u8), 16)), np.uint8))
    assert (Blockhash().hash_u8(u8) == ref).all()


CHEAP = ["PDQ", "aHash-64", "dHash-64", "pHash-64", "wHash-64", "BlockMean", "MarrHildreth", "Blockhash-256",
         "ISCC-Image-64", "ISCC-Image-256"]


@pytest.mark.parametrize("name", CHEAP)
def test_identity_and_separation(name):
    m = IMAGE_METHODS[name]()
    m.load("cpu")
    a, b = _img(320, 480), _img(320, 480)
    da, db = m.extract([a, b]), m.extract([a, b])
    assert (da == db).all(), "not deterministic"
    s = m.score(da, da)
    assert s[0, 0] == pytest.approx(1.0) and s[0, 1] < 0.9
