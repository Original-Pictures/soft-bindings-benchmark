"""Clean extraction latency, measured on an otherwise idle host.

The timings recorded during extraction include contention (dozens of worker processes
sharing the machine, attack rendering in the same workers), so they are not reported as
latency. Here each image method fingerprints the same 256 registered originals (1024 px
long side, decoded once) alone on the host: CPU methods on one thread, learned methods
on the GPU in batches of 64 (host-to-device copy included). Audio systems fingerprint
64 registered 30 s items and video systems 32 registered 5 s clips, one at a time.

    python latency.py [--device cuda]   -> $BENCH_RESULTS/latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from common import CORPUS, RESULTS, Timer, host_info, load_rgb, to_u8, write_json


def image(device: str) -> dict:
    import cv2
    import torch

    from fingerprints_image import IMAGE_METHODS

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    man = json.loads((CORPUS / "corpus_manifest.json").read_text())["image"]
    imgs = [to_u8(load_rgb(CORPUS / man["items"][i]["path"], 1024)) for i in man["sets"]["reg"][:256]]
    out = {}
    for n, ctor in IMAGE_METHODS.items():
        m = ctor()
        dev = device if m.gpu else "cpu"
        m.load(dev)
        m.extract(imgs[:8])  # warm-up (model load, CUDA kernels)
        with Timer(dev) as t:
            for i in range(0, len(imgs), 64):
                m.extract(imgs[i:i + 64])
        out[n] = {"ms_per_item": 1000 * t.elapsed / len(imgs), "device": dev, "items": len(imgs)}
        print(f"image {n:16s} {out[n]['ms_per_item']:8.2f} ms  {dev}", flush=True)
    return out


def timebased(mod: str, device: str) -> dict:
    if mod == "audio":
        from bench_audio import _read
        from fingerprints_audio import AUDIO_METHODS as M

        man = json.loads((CORPUS / "corpus_manifest.json").read_text())["audio"]
        items = [CORPUS / man["items"][i]["path"] for i in man["sets"]["reg"][:64]]
        load = _read
        run = lambda m, x: m.extract(*x)
        dur = lambda x: len(x[0]) / x[1]
    else:
        from fingerprints_video import VIDEO_METHODS as M

        man = json.loads((CORPUS / "corpus_manifest.json").read_text())["video"]
        items = [CORPUS / man["items"][i]["path"] for i in man["sets"]["reg"][:32]]
        load = lambda p: p
        run = lambda m, x: m.extract(x)
        dur = lambda x: 5.0
    data = [load(p) for p in items]
    out = {}
    for n, ctor in M.items():
        if n == "NMFP-triplet" or n.endswith("-mean"):
            continue  # NMFP runs in its own TF environment; "-mean" shares "-seq" descriptors
        m = ctor()
        dev = device if m.gpu else "cpu"
        m.load(dev)
        run(m, data[0])
        t0 = time.perf_counter()
        for x in data:
            run(m, x)
        el = time.perf_counter() - t0
        secs = sum(dur(x) for x in data)
        out[n] = {"ms_per_item": 1000 * el / len(data), "ms_per_media_second": 1000 * el / secs, "device": dev}
        print(f"{mod} {n:16s} {out[n]['ms_per_media_second']:8.2f} ms per media second  {dev}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = {"host": host_info(), "image": image(dev), "audio": timebased("audio", dev), "video": timebased("video", dev)}
    write_json(RESULTS / "latency.json", res)


if __name__ == "__main__":
    main()
