"""Tests for bench.py: the added methods and the descriptors made from stored ones.

    uv run --no-project --with pytest --with numpy==2.4.6 --with scipy==1.17.1 --with pillow==12.3.0 pytest iscc
"""

import json

import numpy as np

import bench
import fingerprints_image as fi


def codes(rows: int, bits: int, seed: int) -> np.ndarray:
    """Random 0/1 descriptors shaped like the bench's binary codes."""
    return np.random.default_rng(seed).integers(0, 2, (rows, bits), dtype=np.uint8)


def test_combine_concatenates_leading_bits():
    a = np.array([[1, 0, 1, 1], [0, 0, 1, 0]], np.uint8)
    b = np.array([[1, 1, 0, 0], [0, 1, 0, 1]], np.uint8)
    assert bench.combine([a, b], [2, 4]).tolist() == [[1, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 1]]


def test_derive_writes_prefix_and_combination(tmp_path):
    s = tmp_path / "pos"
    s.mkdir()
    (s / "ids.json").write_text(json.dumps(["a", "b", "c"]))
    image, semantic = codes(3, 256, 1), codes(3, 256, 2)
    np.save(s / "ISCC-Image-256.npy", image)
    np.save(s / "ISCC-SCI-256.npy", semantic)
    bench.derive(tmp_path)
    assert (np.load(s / "ISCC-SCI-64.npy") == semantic[:, :64]).all()
    assert (np.load(s / "ISCC-Combined-512.npy") == np.concatenate([semantic, image], axis=1)).all()


def test_derive_skips_sets_without_semantic_codes(tmp_path):
    s = tmp_path / "reg"
    s.mkdir()
    (s / "ids.json").write_text("[]")
    np.save(s / "ISCC-Image-256.npy", codes(2, 256, 3))
    bench.derive(tmp_path)
    assert sorted(p.name for p in s.glob("*.npy")) == ["ISCC-Image-256.npy"]


def test_registered_methods():
    assert "ISCC-SCI-256" in fi.IMAGE_METHODS
    combined = fi.IMAGE_METHODS["ISCC-Combined-512"]()
    assert (combined.dim, combined.metric, combined.tier) == (512, "hamming", "R")
    assert combined.params == {"parts": ["ISCC-SCI-256[:256]", "ISCC-Image-256[:256]"]}
    assert fi.IMAGE_METHODS["ISCC-SCI-64"]().dim == 64
