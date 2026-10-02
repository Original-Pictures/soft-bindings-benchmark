"""Second-stage verification of image retrieval: geometric agreement with the stored original.

For each positive and negative query of the ABO track, the top-1 candidate of a retrieval
method is checked against the registered original it points to: SIFT on both images at the
localization track's working size, ratio test, RANSAC homography, and the number of inliers.
A registry that accepts a match only when both the retrieval score and the inlier count pass
their thresholds can run the retrieval stage at a looser operating point; this measures
what that buys and what it costs, on the same queries and the same calibration split as
bench_image.metrics.

Writes results/image/_verify/<method>.npz, with rows in the same order as
results/image/_perquery/<method>.npz (positives: pos ids; negatives: neg ids after the
duplicate rule), so the analysis can join them.

    python bench_verify.py --only DINOv2-S,PDQ --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import attacks
import bench_image
from bench_localize import WORK, _work_size
from common import CORPUS, DESC, RESULTS, load_rgb, to_u8

_W: dict = {}


def _init(bg: list[str]) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2

    cv2.setNumThreads(1)
    attacks.BACKGROUNDS = list(bg)
    _W["sift"] = cv2.SIFT_create(4000)


def _gray(x: np.ndarray) -> np.ndarray:
    import cv2

    u8 = to_u8(x)
    h, w = _work_size(u8.shape[:2])
    return cv2.cvtColor(cv2.resize(u8, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def _inliers(job: tuple[str, str, str, str]) -> int:
    """job = (query key, query source path, attack, candidate path) -> RANSAC inliers."""
    import cv2

    key, src, atk, cand = job
    q = _gray(bench_image.make_image(src, atk, key))
    o = _gray(load_rgb(cand, bench_image.MAX_SIDE))
    k1, d1 = _W["sift"].detectAndCompute(q, None)
    k2, d2 = _W["sift"].detectAndCompute(o, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 4:
        return 0  # a homography needs four correspondences
    s = np.float32([k1[m.queryIdx].pt for m in good])
    d = np.float32([k2[m.trainIdx].pt for m in good])
    _, inl = cv2.findHomography(s, d, cv2.RANSAC, 3.0)
    return int(inl.sum()) if inl is not None else 0


def run(names: list[str], workers: int) -> None:
    man = bench_image.manifest()
    a = man["image"]
    path = lambda i: str(CORPUS / a["items"][i]["path"])
    bg = [path(i) for i in a["sets"]["bg"]]
    pos_ids = json.loads((DESC / "pos" / "ids.json").read_text())
    neg_ids = json.loads((DESC / "neg" / "ids.json").read_text())
    flagged = set(bench_image._dedup()["neg_flagged"])
    neg_keep = np.array([k.split("|")[0] not in flagged for k in neg_ids])
    neg_ids = [k for k, ok in zip(neg_ids, neg_keep) if ok]
    for n in names:
        z = np.load(DESC / "search_abo" / f"{n}.npz")
        ref_ids = list(z["ref_ids"])
        jobs = []
        for ids, I in ((pos_ids, z["pos_I"]), (neg_ids, z["neg_I"][neg_keep])):
            for k, top in zip(ids, I[:, 0]):
                s, atk = k.split("|")
                jobs.append((k, path(s), atk, path(ref_ids[int(top)])))
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(bg,)) as pool:
            inl = np.array(pool.map(_inliers, jobs, chunksize=64), np.int32)
        out = RESULTS / "image" / "_verify" / f"{n}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, pos_inliers=inl[:len(pos_ids)], neg_inliers=inl[len(pos_ids):],
                            work_px=np.int32(WORK))
        print(f"verify {n}: {len(jobs)} pairs, {time.time() - t0:.0f}s, median inliers pos "
              f"{np.median(inl[:len(pos_ids)]):.0f} neg {np.median(inl[len(pos_ids):]):.0f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="DINOv2-S,PDQ")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    run(args.only.split(","), args.workers)


if __name__ == "__main__":
    main()
