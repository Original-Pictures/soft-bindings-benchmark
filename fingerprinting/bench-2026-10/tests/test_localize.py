"""Checks for the partial-edit generator and localization helpers (no corpus needed)."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_localize as bl  # noqa: E402
import edits  # noqa: E402
from common import rng_for  # noqa: E402


def test_edit_paths_keep_dotted_spec_names():
    # Regression: Path.with_suffix treated ".03_none" as an extension and made specs collide.
    a = edits._paths("X", "splice", 0.03, "none")[0]
    b = edits._paths("X", "splice", 0.03, "jpeg75")[0]
    assert a != b and a.name == "X__splice_0.03_none.png"


def test_design_is_balanced():
    plan = edits.design([f"s{i}" for i in range(96)], 12)
    for j in (1, 2, 3):
        c = Counter(p[j] for p in plan)
        assert max(c.values()) - min(c.values()) <= 1


def test_blob_area_close_to_target():
    content = np.ones((400, 600), bool)
    for a in edits.AREAS:
        m = edits.blob_mask((400, 600), a, rng_for("t", a), content)
        assert abs((m > 0.5).mean() - a) / a < 0.35


def test_edit_changes_only_near_mask():
    x = np.random.default_rng(0).random((200, 300, 3)).astype(np.float32)
    soft = edits.blob_mask((200, 300), 0.1, rng_for("t2"), np.ones((200, 300), bool))
    for kind in ("splice", "copymove", "inpaint-telea", "recolor"):
        y, gt = edits.apply_edit(x, soft, kind, rng_for("k", kind), x[::-1].copy(), "cpu")
        outside = gt < 1e-3
        assert np.abs(y - x)[outside].max() < 0.5 / 255, kind  # below 8-bit quantisation


def test_midrank_cdf_handles_ties():
    null = np.sort(np.r_[np.zeros(90), np.linspace(0.1, 1, 10)])
    assert abs(bl._cdf(null, np.array([0.0]))[0] - 0.45) < 1e-9


def test_identity_warp_when_unaligned():
    q = np.random.default_rng(1).random((50, 80, 3)).astype(np.float32)
    w, valid = bl.warp_to(q, (100, 160), None)
    assert w.shape == (100, 160, 3) and valid.all()


def test_f1_iou():
    g = np.array([1, 1, 0, 0], bool)
    assert bl._f1_iou(g.copy(), g) == (1.0, 1.0)
    assert bl._best_f1(np.array([0.9, 0.8, 0.1, 0.0]), g) == 1.0
