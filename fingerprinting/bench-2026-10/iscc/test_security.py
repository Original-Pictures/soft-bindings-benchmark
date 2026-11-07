"""Tests of the ISCC surrogate's trim box and of the security rerun's record handling (need torch and iscc-sdk)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
idk = pytest.importorskip("iscc_sdk")
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hash_surrogates as hs  # noqa: E402

tolerant = pytest.mark.skipif(not hasattr(idk, "image_trim_tolerant"), reason="iscc-sdk without tolerant trim")


def framed(seed: int, frames: list[tuple[int, int]], size=(320, 240)) -> Image.Image:
    """Textured content inside nested frames of (width, grey level), outermost first, with +-6 grey noise."""
    rng = np.random.default_rng(seed)
    w, h = size
    a = np.zeros((h, w, 3), np.float64)
    inset = 0
    for width, level in frames:
        a[inset:h - inset, inset:w - inset] = level
        inset += width
    a[inset:h - inset, inset:w - inset] = rng.integers(0, 256, (h - 2 * inset, w - 2 * inset, 3))
    a += rng.integers(-6, 7, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def gradient(size=(320, 240)) -> Image.Image:
    """A smooth vertical gradient (a sky), which a one-sided cut must keep."""
    col = np.linspace(255, 120, size[1])[:, None, None]
    return Image.fromarray(np.broadcast_to(col, (size[1], size[0], 3)).astype(np.uint8))


@tolerant
@pytest.mark.parametrize("img", [framed(1, [(20, 250)]), framed(2, [(12, 200), (18, 250)]), framed(3, []), gradient()])
def test_tolerant_trim_box_crops_like_the_sdk(img):
    box = hs.trim_box(np.asarray(img))
    got = img.crop(box) if box else img
    assert np.array_equal(np.asarray(got), np.asarray(idk.image_trim_tolerant(img)))


@tolerant
def test_trim_box_follows_the_sdk_option(monkeypatch):
    img = framed(4, [(16, 245)])
    monkeypatch.setattr(idk.sdk_opts, "image_trim_tolerant", False)
    box = hs.trim_box(np.asarray(img))
    got = img.crop(box) if box else img
    assert np.array_equal(np.asarray(got), np.asarray(idk.image_trim_border(img)))


def test_trim_box_of_an_unframed_image_is_none():
    assert hs.trim_box(np.asarray(framed(5, []))) is None


def test_update_record_merges_stages(tmp_path, monkeypatch):
    security = pytest.importorskip("security")
    monkeypatch.setattr(security, "RECORD", tmp_path / "evasion.json")
    security.update_record({"surrogate": {"ISCC-Image-64": {}}, "thresholds": {"ISCC-Image-64": (0.9, "pair@1e-06")}}, 3)
    security.update_record({"transfer": {"2": {}}, "thresholds": {"PDQ": (0.8, "pair@1e-06 (paper)")}}, 3)
    rec = json.loads((tmp_path / "evasion.json").read_text())
    assert set(rec["thresholds"]) == {"ISCC-Image-64", "PDQ"}
    assert rec["surrogate"] == {"ISCC-Image-64": {}} and rec["transfer"] == {"2": {}} and rec["whitebox"] == {}
    assert rec["params"]["n"] == 3 and rec["experiment"] == "evasion"
