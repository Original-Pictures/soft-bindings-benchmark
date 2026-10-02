"""Video retrieval-robustness track (same stages as bench_audio).

    python bench_video.py render | extract [--only] | search [--only] | metrics

Index = reg + dist 5 s clips. Queries: pos = reg clips under every attack in
attacks.VIDEO_COPY_ATTACKS, neg = never-registered clips under the same attacks.
Queries are rendered once to MP4 ($BENCH_WORK/out/video/<set>/<key>.mp4) with a JSON
sidecar holding the attack's ground truth (excerpt start, insertion offset) for the
temporal-localization analysis.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import attacks
from common import CORPUS, DESC, OUT, RESULTS, extra_sets, host_info, rng_for, write_json
from fingerprints_video import TMK, VIDEO_METHODS

QDIR = OUT / "video"
TOPK = 50


def manifest() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())["video"]


def item_lists(man: dict) -> dict[str, list[tuple[str, Path, str]]]:
    p = lambda i: CORPUS / man["items"][i]["path"]
    atk = list(attacks.VIDEO_COPY_ATTACKS)
    return {
        **extra_sets("video"),
        "reg": [(i, p(i), "ref") for i in man["sets"]["reg"]],
        "dist": [(i, p(i), "ref") for i in man["sets"]["dist"]],
        "pos": [(f"{i}|{a}", QDIR / "pos" / f"{i}__{a}.mp4", a) for a in atk for i in man["sets"]["reg"]],
        "neg": [(f"{i}|{a}", QDIR / "neg" / f"{i}__{a}.mp4", a) for a in atk for i in man["sets"]["neg"]],
    }


def _render(job):
    key, src, atk, dst, bg, dur = job
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fn = attacks.VIDEO_COPY_ATTACKS[atk][0]
    ctx = {"duration": dur, "bg": bg}
    tmp = dst.with_suffix(".tmp.mp4")
    if fn is None:
        shutil.copyfile(src, tmp)
    elif isinstance(fn, str) and fn.startswith("custom:"):
        getattr(attacks, fn.split(":", 1)[1])(src, tmp, rng_for("video-attack", atk, key.split("|")[0]), ctx)
    else:
        extra, filt, codec = fn(rng_for("video-attack", atk, key.split("|")[0]), ctx)
        pre = []
        if extra and extra[0] == "-ss":
            pre, extra = extra, []
        subprocess.run(["ffmpeg", "-v", "error", "-y", *pre, "-i", str(src), *extra, *filt, *codec, "-an", str(tmp)],
                       check=True)
    meta = {k: v for k, v in ctx.items() if k not in ("bg",)}
    dst.with_suffix(".json").write_text(json.dumps(meta))
    tmp.replace(dst)


def render(workers: int) -> None:
    man = manifest()
    lists = item_lists(man)
    dist = man["sets"]["dist"]
    dur = man["clip_seconds"]
    jobs = []
    for s in ("pos", "neg"):
        for key, path, atk in lists[s]:
            src_id = key.split("|")[0]
            # background for pip/insert: a distractor chosen per item (never the item itself)
            bg = CORPUS / man["items"][dist[int(rng_for("video-bg", src_id).integers(0, len(dist)))]]["path"]
            jobs.append((key, CORPUS / man["items"][src_id]["path"], atk, path, bg, dur))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render, jobs, chunksize=2)):
            if i % 500 == 0:
                print(f"  render {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"rendered {len(jobs)} video queries in {time.time() - t0:.0f}s")


_W: dict = {}


def _init(name, device):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["m"] = VIDEO_METHODS[name]()
    _W["m"].load(device)


def _ext(path):
    t = time.perf_counter()
    d = _W["m"].extract(Path(path))
    return d, time.perf_counter() - t


def extract(names: list[str], workers: int, device: str) -> None:
    import iscc_sdk

    iscc_sdk.install()  # its bundled ffmpeg (MPEG-7 signatures) is fetched once, before the workers start
    man = manifest()
    for set_name, items in item_lists(man).items():
        out = DESC / f"video_{set_name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "ids.json").write_text(json.dumps([k for k, _, _ in items]))
        timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
        for n in names:
            if (out / f"{n}.pkl").exists():
                continue
            m = VIDEO_METHODS[n]()
            paths = [str(p) for _, p, _ in items]
            t0 = time.time()
            twin = out / f"{n.removesuffix('-mean')}-seq.pkl"
            if n.endswith("-mean") and twin.exists():
                # "-mean" and "-seq" store the same per-frame descriptors and differ only in
                # matching; reuse them rather than decoding and embedding every clip twice.
                (out / f"{n}.pkl").write_bytes(twin.read_bytes())
                timing[n] = {**timing[n.removesuffix("-mean") + "-seq"], "reused_from": twin.name}
                (out / "timing.json").write_text(json.dumps(timing, indent=1))
                continue
            if n == "TMK+PDQF":
                tdir = OUT / "video_tmk" / set_name
                tdir.mkdir(parents=True, exist_ok=True)
                with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(n, "cpu")) as pool:
                    res = pool.map(_ext, paths, chunksize=2)
                files = []
                for (k, _, _), (blob, _) in zip(items, res):
                    f = tdir / (k.replace("|", "__") + ".tmk")
                    f.write_bytes(blob)
                    files.append(str(f))
                res = [(f, s) for f, (_, s) in zip(files, res)]
            elif m.gpu:
                _init(n, device)
                res = [_ext(p) for p in paths]
            else:
                with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(n, "cpu")) as pool:
                    res = pool.map(_ext, paths, chunksize=2)
            with open(out / f"{n}.pkl", "wb") as f:
                pickle.dump([r[0] for r in res], f)
            timing[n] = {"seconds": sum(r[1] for r in res), "items": len(res), "device": device if m.gpu else "cpu:1thread"}
            (out / "timing.json").write_text(json.dumps(timing, indent=1))
            print(f"video {set_name} {n}: {len(res)} in {time.time() - t0:.0f}s", flush=True)


def _load(set_name, n):
    with open(DESC / f"video_{set_name}" / f"{n}.pkl", "rb") as f:
        return pickle.load(f)


def search(names: list[str]) -> None:
    import faiss

    faiss.omp_set_num_threads(os.cpu_count() or 1)  # workers are single-threaded; search is not
    man = manifest()
    ref_ids = man["sets"]["reg"] + man["sets"]["dist"]
    rid = {r: j for j, r in enumerate(ref_ids)}
    for n in names:
        out = DESC / "search_video" / f"{n}.npz"
        if out.exists() or not all((DESC / f"video_{s}" / f"{n}.pkl").exists() for s in ("reg", "dist", "pos", "neg")):
            continue
        t0 = time.time()
        refs = _load("reg", n) + _load("dist", n)
        res = {}
        if n == "TMK+PDQF":
            for qs in ("pos", "neg"):
                qf = _load(qs, n)
                ts = time.perf_counter()
                sc = TMK.batch_scores([Path(p) for p in qf], [Path(p) for p in refs])
                res[f"{qs}_search_s"] = time.perf_counter() - ts
                S = np.full((len(qf), TOPK), -np.inf, np.float32)
                I = np.full((len(qf), TOPK), -1, np.int64)
                ref_pos = {p: j for j, p in enumerate(refs)}
                per_q: dict[str, list] = {}
                for (a, b), (s1, s2) in sc.items():
                    per_q.setdefault(a, []).append((s2, ref_pos[b], s1))
                for j, p in enumerate(qf):
                    top = sorted(per_q.get(p, []), key=lambda t: -t[0])[:TOPK]
                    for r, (s2, idx, _) in enumerate(top):
                        S[j, r], I[j, r] = s2, idx
                res[f"{qs}_S"], res[f"{qs}_I"] = S, I
                if qs == "pos":
                    qids = json.loads((DESC / "video_pos" / "ids.json").read_text())
                    tidx = np.array([rid[k.split("|")[0]] for k in qids])
                    res["pos_trueidx"] = tidx
                    res["pos_true"] = np.array([sc.get((p, refs[t]), (np.nan, np.nan))[1] for p, t in zip(qf, tidx)],
                                               np.float32)
            res["ref_bits_mean"] = float(np.mean([8 * Path(p).stat().st_size for p in refs[:200]]))
        else:
            m = VIDEO_METHODS[n]()
            m.build(refs)
            for qs in ("pos", "neg"):
                qd = _load(qs, n)
                S = np.full((len(qd), TOPK), -np.inf, np.float32)
                I = np.full((len(qd), TOPK), -1, np.int64)
                ts = time.perf_counter()
                for j, q in enumerate(qd):
                    I[j], S[j] = m.search(q, TOPK)
                res[f"{qs}_search_s"] = time.perf_counter() - ts
                res[f"{qs}_S"], res[f"{qs}_I"] = S, I
                if qs == "pos":
                    qids = json.loads((DESC / "video_pos" / "ids.json").read_text())
                    tidx = np.array([rid[k.split("|")[0]] for k in qids])
                    res["pos_trueidx"] = tidx
                    res["pos_true"] = np.array([m.pair(q, refs[t]) for q, t in zip(qd, tidx)], np.float32)
            res["ref_bits_mean"] = float(np.mean([m.size_bits(r) for r in refs[:200]]))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res, ref_ids=np.array(ref_ids))
        print(f"search video {n}: {time.time() - t0:.0f}s", flush=True)


def metrics() -> None:
    from retrieval_metrics import dump_perquery, summarize

    man = manifest()
    calib = set(sorted(man["sets"]["neg"])[::2])
    pos_ids = json.loads((DESC / "video_pos" / "ids.json").read_text())
    neg_ids = json.loads((DESC / "video_neg" / "ids.json").read_text())
    neg_cal = np.array([k.split("|")[0] in calib for k in neg_ids])
    order = list(attacks.VIDEO_COPY_ATTACKS)
    for f in sorted((DESC / "search_video").glob("*.npz")):
        n = f.stem
        z = np.load(f)
        rec = summarize(z["pos_S"], z["pos_I"], z["pos_trueidx"], z["pos_true"], [k.split("|")[1] for k in pos_ids],
                        z["neg_S"], [k.split("|")[1] for k in neg_ids], neg_cal, len(z["ref_ids"]), order)
        tj = DESC / "video_reg" / "timing.json"
        rec.update({"method": VIDEO_METHODS[n]().meta(), "timing_reg": json.loads(tj.read_text()).get(n, {}) if tj.exists() else {},
                    "search_s": {"pos": float(z["pos_search_s"]), "neg": float(z["neg_search_s"])},
                    "ref_bits_mean": float(z["ref_bits_mean"]),
                    "attack_family": {k: fam for k, (_, fam) in attacks.VIDEO_COPY_ATTACKS.items()}, "host": host_info()})
        write_json(RESULTS / "video" / f"{n}.json", rec)
        dump_perquery(RESULTS / "video" / "_perquery" / f"{n}.npz", pos_ids, z["pos_S"], z["pos_I"],
                      z["pos_trueidx"], neg_ids, z["neg_S"], neg_cal)
        pa = rec["per_attack"]["_all"]
        print(f"{n:18s} R@1={pa['R@1']:.3f} muAP={pa['muAP']:.3f} TPR@q1e-2={pa.get('TPR@query@0.01', float('nan')):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "extract", "search", "metrics"])
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    names = [n for n in VIDEO_METHODS if not args.only or n in args.only.split(",")]
    if args.stage == "render":
        render(args.workers)
    elif args.stage == "extract":
        import torch

        extract(names, args.workers, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    elif args.stage == "search":
        search(names)
    else:
        metrics()


if __name__ == "__main__":
    main()
