"""The adapter's detect-first path gives the same verdict and payload as trustmark's own decode(DETECTFIRST=True),
and its decoder-only path the same as decode(DETECTFIRST=False), on marked, attacked and unmarked images.

    BENCH_WORK=~/op-v11-work/wm python -m pytest -q test_detect_first.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import CORPUS, payload  # noqa: E402
from models_image import TrustMarkAdapter, _pil  # noqa: E402

import attacks  # noqa: E402
from bench_image import load_image  # noqa: E402

IMAGES = sorted((CORPUS / "image" / "kodak").glob("kodim0[1-4].png"))


@pytest.fixture(scope="module")
def adapter():
    if not IMAGES:
        pytest.skip("no Kodak images under $BENCH_WORK/corpus")
    a = TrustMarkAdapter("Q", 1.0, detect_first=True)
    a.load("cpu")
    return a


def test_no_detector_for_b_and_c():
    with pytest.raises(ValueError):
        TrustMarkAdapter("B", 1.0, detect_first=True)


def test_names_keep_v10_configs():
    assert TrustMarkAdapter("Q", 1.2).name == "trustmark-Q@1.2"
    assert TrustMarkAdapter("Q", 1.2, detect_first=True).name == "trustmark-Q@1.2+detect"


@pytest.mark.parametrize("attack", ["none", "jpeg75", "crop70", "resize0.5"])
def test_matches_library(adapter, attack):
    for i, p in enumerate(IMAGES):
        ref = load_image(p)
        marked = adapter.embed(ref, payload(adapter.nbits, i, salt="100"))
        for x in (attacks.IMAGE_ATTACKS[attack](marked), attacks.IMAGE_ATTACKS[attack](ref)):
            for flag in (False, True):
                adapter.detect_first = flag
                got = adapter.decode(x)
                secret, present, _ = adapter.tm.decode(_pil(x), MODE="binary", DETECTFIRST=flag)
                assert got["native_detect"] == present
                if present:
                    assert got["native_payload"] == secret
