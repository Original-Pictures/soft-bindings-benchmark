"""NMFP (reference row): run the authors' extraction.py over every audio set and collect
the per-file segment embeddings into $BENCH_WORK/desc/audio_<set>/NMFP-triplet.pkl.

Runs in two environments: `list` (main venv) writes one path list per set; the
authors' `extraction.py` (TensorFlow venv, invoked by gpu/stage.sh) writes one .npy
per audio file under $BENCH_WORK/nmfp_raw mirroring the path below the common root;
`collect` (main venv) gathers them in the set's id order.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from common import BENCH_WORK, DESC

import bench_audio

RAW = BENCH_WORK / "nmfp_raw"
LISTS = BENCH_WORK / "nmfp_lists"


def sets() -> dict:
    return bench_audio.item_lists(bench_audio.manifest())


def cmd_list() -> None:
    LISTS.mkdir(parents=True, exist_ok=True)
    allp = []
    for name, items in sets().items():
        allp += [str(p) for _, p, _ in items]
    # One list for everything: extraction.py preserves paths relative to the common root,
    # so a single run keeps the mapping unambiguous.
    (LISTS / "all.txt").write_text("\n".join(allp) + "\n")
    (LISTS / "root.txt").write_text(os.path.commonpath(allp))
    print(len(allp), "files; root", os.path.commonpath(allp))


def cmd_collect() -> None:
    root = Path((LISTS / "root.txt").read_text().strip())
    for name, items in sets().items():
        out, missing = [], 0
        for _, p, _ in items:
            f = (RAW / Path(p).relative_to(root)).with_suffix(".npy")
            if f.exists():
                e = np.load(f).astype(np.float32)
                e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            else:
                e, missing = np.zeros((0, 128), np.float32), missing + 1
            out.append(e)
        d = DESC / f"audio_{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "NMFP-triplet.pkl").write_bytes(pickle.dumps(out))
        if not (d / "ids.json").exists():
            (d / "ids.json").write_text(json.dumps([k for k, _, _ in items]))
        print(f"{name}: {len(out)} items, {missing} missing")


if __name__ == "__main__":
    {"list": cmd_list, "collect": cmd_collect}[sys.argv[1]]()
