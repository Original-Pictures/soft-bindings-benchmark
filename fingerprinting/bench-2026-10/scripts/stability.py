"""Cross-host determinism of fingerprints.

A registry lookup assumes the fingerprint computed at signing time equals the one
computed at query time for the same pixels, possibly on another machine. The 2026-09
watermark bench found host-dependent outcomes (AVX2 vs AVX-512 FFmpeg/x264 paths), so
here every image method fingerprints the same 200 images (100 registered originals and
100 JPEG-75 copies, decoded once to PNG so decoding is not part of the comparison) on
each host, and `compare` reports per method: share of bit-identical descriptors, mean
Hamming bits differing (binary) or 1 - cosine (float), and whether any difference would
cross the method's calibrated threshold.

    python stability.py make [--device cuda|mps|cpu]   -> $BENCH_RESULTS/stability/<host>.npz
    python stability.py compare a.npz b.npz            -> $BENCH_RESULTS/stability/compare.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from common import CORPUS, OUT, RESULTS, host_info, load_rgb, sha256_file, to_u8, write_json

N = 100


def stable_inputs() -> list[Path]:
    from PIL import Image

    import attacks

    man = json.loads((CORPUS / "corpus_manifest.json").read_text())["image"]
    d = OUT / "stability"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in man["sets"]["reg"][:N]:
        for tag, fn in (("orig", None), ("jpeg75", attacks.IMAGE_ATTACKS["jpeg75"])):
            p = d / f"{s}_{tag}.png"
            if not p.exists():
                x = load_rgb(CORPUS / man["items"][s]["path"], 1024)
                Image.fromarray(to_u8(fn(x) if fn else x)).save(p)
            paths.append(p)
    return paths


def make(device: str) -> None:
    from fingerprints_image import IMAGE_METHODS

    paths = stable_inputs()
    imgs = [load_rgb(p) for p in paths]
    out = {}
    for n, ctor in IMAGE_METHODS.items():
        m = ctor()
        m.load(device if m.gpu else "cpu")
        out[n] = m.extract(imgs)
        print(n, out[n].shape, flush=True)
    tag = f"{platform.machine()}-{device}"
    RESULTS.joinpath("stability").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "stability" / f"{tag}.npz", **{k.replace("+", "_"): v for k, v in out.items()},
                        _host=json.dumps(host_info()), _paths=json.dumps([p.name for p in paths]),
                        _sha=json.dumps([sha256_file(p) for p in paths]))
    print("wrote", tag)


def compare(a: Path, b: Path) -> None:
    from fingerprints_image import IMAGE_METHODS

    A, B = np.load(a), np.load(b)
    if json.loads(str(A["_sha"])) != json.loads(str(B["_sha"])):
        raise SystemExit("inputs differ between hosts: copy OUT/stability/*.png from one host to the other first")
    res = {"a": a.name, "b": b.name, "host_a": json.loads(str(A["_host"])), "host_b": json.loads(str(B["_host"])), "methods": {}}
    for n, ctor in IMAGE_METHODS.items():
        k = n.replace("+", "_")
        if k not in A or k not in B:
            continue
        m = ctor()
        x, y = A[k], B[k]
        same = (x == y).all(1)
        r = {"identical_share": float(same.mean()), "n": int(len(x))}
        if m.metric == "hamming":
            r["bits_differing_mean"] = float((x != y).sum(1).mean())
            r["bits_differing_max"] = int((x != y).sum(1).max())
        elif m.metric == "ip":
            cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
            r["one_minus_cos_max"] = float((1 - cos).max())
            r["one_minus_cos_median"] = float(np.median(1 - cos))
        else:
            r["l2_max"] = float(np.linalg.norm(x - y, axis=1).max())
        th_file = RESULTS / "image" / f"{n}.json"
        if th_file.exists():
            th = json.loads(th_file.read_text())["thresholds"].get("query@0.01")
            if th is not None and np.isfinite(th):
                s = np.array([m.score(x[i:i + 1], y[i:i + 1])[0, 0] for i in range(len(x))])
                r["self_score_min"] = float(s.min())
                r["self_below_threshold"] = int((s < th).sum())
        res["methods"][n] = r
        print(f"{n:16s} identical={r['identical_share']:.3f} "
              + " ".join(f"{k}={v:.3g}" for k, v in r.items() if k not in ("identical_share", "n")))
    write_json(RESULTS / "stability" / "compare.json", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["make", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.stage == "make":
        import torch

        dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        make(dev)
    else:
        compare(Path(args.files[0]), Path(args.files[1]))


if __name__ == "__main__":
    sys.exit(main())
