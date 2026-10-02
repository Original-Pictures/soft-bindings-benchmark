"""Audio retrieval-robustness track.

    python bench_audio.py render            # attacked queries -> $BENCH_WORK/out/audio/<set>/<key>.flac
    python bench_audio.py extract [--only]  # descriptors -> $BENCH_WORK/desc/audio_<set>/<method>.pkl
    python bench_audio.py search  [--only]
    python bench_audio.py metrics

Index = reg + dist (music and speech). Queries: pos = every reg item under every attack
in attacks.AUDIO_COPY_ATTACKS; neg = never-registered items under the same attacks.
Rendering once to FLAC (lossless, 22.05 kHz mono) means every method, including the
NMFP TensorFlow environment, reads byte-identical queries.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

import attacks
from common import CORPUS, DESC, OUT, RESULTS, SCRIPTS, Timer, extra_sets, host_info, rng_for, sha256_file, write_json
from fingerprints_audio import AUDIO_METHODS

QDIR = OUT / "audio"
TOPK = 50


def manifest() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())["audio"]


def item_lists(man: dict) -> dict[str, list[tuple[str, Path, str]]]:
    """set -> [(key, audio path, attack)]; reference sets carry attack 'ref'."""
    p = lambda i: CORPUS / man["items"][i]["path"]
    atk = list(attacks.AUDIO_COPY_ATTACKS)
    return {
        **extra_sets("audio"),
        "reg": [(i, p(i), "ref") for i in man["sets"]["reg"]],
        "dist": [(i, p(i), "ref") for i in man["sets"]["dist"]],
        "pos": [(f"{i}|{a}", QDIR / "pos" / f"{i}__{a}.flac", a) for a in atk for i in man["sets"]["reg"]],
        "neg": [(f"{i}|{a}", QDIR / "neg" / f"{i}__{a}.flac", a) for a in atk for i in man["sets"]["neg"]],
    }


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    return (y if y.ndim == 1 else y.mean(1)), sr


# ------------------------------------------------------------------ render
def _render_init(bg_paths):
    attacks.AUDIO_BACKGROUNDS = [_read(Path(p))[0] for p in bg_paths]


def _render(job):
    import soundfile as sf

    key, src, atk, dst = job
    if dst.exists():
        return
    y, sr = _read(src)
    z = attacks.AUDIO_COPY_ATTACKS[atk][0](y, sr, rng_for("audio-attack", atk, key.split("|")[0]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.flac")
    sf.write(tmp, np.clip(z, -1, 1), sr, subtype="PCM_16")
    tmp.replace(dst)


def babble_pool(man: dict, n: int = 50) -> list[str]:
    """Background music for the babble attacks: FMA tracks that were downloaded but belong to
    no set (fetch_corpus over-fetches 15% for unreadable members). They are never indexed
    and never queried, so a babble query contains no second registered or distractor item."""
    used = {Path(man["items"][i]["path"]).name for s in ("reg", "dist", "neg") for i in man["sets"][s]}
    # Which surplus tracks exist depends on which downloads happened to succeed, so the 2026-09-26
    # and 2026-09-28 runs drew different backgrounds. The pool of the 2026-09-28 run is pinned
    # (name, sha256) in babble_pool.lock.json and used whenever present.
    lock = SCRIPTS / "babble_pool.lock.json"
    if lock.exists():
        pinned = json.loads(lock.read_text())[:n]
        for name, digest in pinned:
            if sha256_file(CORPUS / "audio" / "music" / name) != digest:
                raise SystemExit(f"babble background {name} does not match babble_pool.lock.json")
        return [str(CORPUS / "audio" / "music" / name) for name, _ in pinned]
    spare = sorted(p for p in (CORPUS / "audio" / "music").glob("*.wav") if p.name not in used)
    if len(spare) < n:
        raise SystemExit(f"only {len(spare)} spare FMA tracks for babble backgrounds")
    return [str(p) for p in spare[:n]]


def render(workers: int) -> None:
    man = manifest()
    lists = item_lists(man)
    bg = babble_pool(man)
    jobs = []
    for s in ("pos", "neg"):
        src = {k: CORPUS / man["items"][k]["path"] for k in man["sets"]["reg" if s == "pos" else "neg"]}
        jobs += [(key, src[key.split("|")[0]], atk, path) for key, path, atk in lists[s]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_render_init, initargs=(bg,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render, jobs, chunksize=8)):
            if i % 2000 == 0:
                print(f"  render {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"rendered {len(jobs)} queries in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ extract
_W: dict = {}


def _ext_init(name, device):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["m"] = AUDIO_METHODS[name]()
    _W["m"].load(device)


def _ext(path):
    y, sr = _read(Path(path))
    t = time.perf_counter()
    d = _W["m"].extract(y, sr)
    return d, time.perf_counter() - t, len(y) / sr


def extract(names: list[str], workers: int, device: str) -> None:
    man = manifest()
    for set_name, items in item_lists(man).items():
        out = DESC / f"audio_{set_name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "ids.json").write_text(json.dumps([k for k, _, _ in items]))
        timing = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
        for n in names:
            if n == "NMFP-triplet" or (out / f"{n}.pkl").exists():
                continue
            m = AUDIO_METHODS[n]()
            t0 = time.time()
            paths = [str(p) for _, p, _ in items]
            if m.gpu:
                _ext_init(n, device)
                res = [_ext(p) for p in paths]
            else:
                with mp.get_context("spawn").Pool(workers, initializer=_ext_init, initargs=(n, "cpu")) as pool:
                    res = pool.map(_ext, paths, chunksize=4)
            with open(out / f"{n}.pkl", "wb") as f:
                pickle.dump([r[0] for r in res], f)
            timing[n] = {"seconds": sum(r[1] for r in res), "audio_seconds": sum(r[2] for r in res), "items": len(res),
                         "device": device if m.gpu else "cpu:1thread"}
            (out / "timing.json").write_text(json.dumps(timing, indent=1))
            print(f"audio {set_name} {n}: {len(res)} in {time.time() - t0:.0f}s")


def load_desc(set_name: str, n: str) -> list:
    p = DESC / f"audio_{set_name}" / f"{n}.pkl"
    if n == "NMFP-triplet":
        p = DESC / f"audio_{set_name}" / "NMFP-triplet.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ search / metrics
def search(names: list[str]) -> None:
    import faiss

    faiss.omp_set_num_threads(os.cpu_count() or 1)  # workers are single-threaded; search is not
    man = manifest()
    for n in names:
        out = DESC / "search_audio" / f"{n}.npz"
        if out.exists() or not all((DESC / f"audio_{s}" / f"{n}.pkl").exists() for s in ("reg", "dist", "pos", "neg")):
            continue
        m = AUDIO_METHODS[n]()
        m.load("cpu")
        refs = load_desc("reg", n) + load_desc("dist", n)
        ref_ids = man["sets"]["reg"] + man["sets"]["dist"]
        t0 = time.time()
        m.build(refs)
        res = {}
        rid = {r: j for j, r in enumerate(ref_ids)}
        for qs in ("pos", "neg"):
            qd = load_desc(qs, n)
            qids = json.loads((DESC / f"audio_{qs}" / "ids.json").read_text())
            S = np.full((len(qd), TOPK), -np.inf, np.float32)
            I = np.full((len(qd), TOPK), -1, np.int64)
            ts = time.perf_counter()
            for j, q in enumerate(qd):
                I[j], S[j] = m.search(q, TOPK)
            res[f"{qs}_search_s"] = time.perf_counter() - ts
            res[f"{qs}_S"], res[f"{qs}_I"] = S, I
            if qs == "pos":
                tidx = np.array([rid[k.split("|")[0]] for k in qids])
                res["pos_trueidx"] = tidx
                res["pos_true"] = np.array([m.pair(q, refs[t]) for q, t in zip(qd, tidx)], np.float32)
        res["ref_bits_mean"] = float(np.mean([m.size_bits(r) for r in refs[:500]]))
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res, ref_ids=np.array(ref_ids))
        print(f"search audio {n}: {time.time() - t0:.0f}s")


# Duplicate recordings (FMA carries some tracks under two IDs). Same principle as the image
# track: on unattacked items, a pair is a duplicate if ANY of three independent systems says
# the two are the same recording. Thresholds are near-identity levels for each system.
AUDIO_DUP_RULE = {"Chromaprint": 0.95, "NMFP-triplet": 0.95, "audfprint": 200.0}
AUDIO_DEDUP = os.environ.get("BENCH_DEDUP", "any")


def audio_dedup() -> dict:
    """Flags from the unattacked queries: a negative source whose best match passes a rule is
    a copy of an indexed track; a registered item whose runner-up passes it has a duplicate
    distractor, which then counts as the same asset (it is the same recording)."""
    pos_ids = json.loads((DESC / "audio_pos" / "ids.json").read_text())
    neg_ids = json.loads((DESC / "audio_neg" / "ids.json").read_text())
    neg_flag, same = set(), {}
    for m, thr in AUDIO_DUP_RULE.items():
        f = DESC / "search_audio" / f"{m}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for j, k in enumerate(neg_ids):
            if k.endswith("|none") and z["neg_S"][j, 0] >= thr:
                neg_flag.add(k.split("|")[0])
        for j, k in enumerate(pos_ids):
            if not k.endswith("|none"):
                continue
            t = int(z["pos_trueidx"][j])
            for s_, i in zip(z["pos_S"][j], z["pos_I"][j]):
                if i >= 0 and i != t and s_ >= thr:
                    same.setdefault(t, set()).add(int(i))
    rec = {"rule": AUDIO_DUP_RULE, "neg_flagged": sorted(neg_flag),
           "duplicates": {str(k): sorted(v) for k, v in same.items()}}
    write_json(DESC / "audio_dedup.json", rec)
    return rec


def metrics() -> None:
    from retrieval_metrics import dump_perquery, summarize

    man = manifest()
    calib = set(sorted(man["sets"]["neg"])[::2])
    pos_ids = json.loads((DESC / "audio_pos" / "ids.json").read_text())
    neg_ids = json.loads((DESC / "audio_neg" / "ids.json").read_text())
    dd = audio_dedup() if AUDIO_DEDUP != "none" else {"neg_flagged": [], "duplicates": {}}
    flagged = set(dd["neg_flagged"])
    keep = np.array([k.split("|")[0] not in flagged for k in neg_ids])
    neg_ids = [k for k, ok in zip(neg_ids, keep) if ok]
    dup = {int(k): set(v) for k, v in dd["duplicates"].items()}
    neg_cal = np.array([k.split("|")[0] in calib for k in neg_ids])
    kind = lambda k: "music" if k.startswith("m") else "speech"
    suffix = "" if AUDIO_DEDUP == "any" else f"_dedup-{AUDIO_DEDUP}"
    for f in sorted((DESC / "search_audio").glob("*.npz")):
        n = f.stem
        z = {k: np.load(f)[k] for k in np.load(f).files}
        z["neg_S"] = z["neg_S"][keep]
        # a duplicate recording of the registered item is the same asset
        I = z["pos_I"].copy()
        for j in range(len(I)):
            d = dup.get(int(z["pos_trueidx"][j]))
            if d:
                I[j][np.isin(I[j], list(d))] = z["pos_trueidx"][j]
        z["pos_I"] = I
        order = list(attacks.AUDIO_COPY_ATTACKS)
        rec = summarize(z["pos_S"], z["pos_I"], z["pos_trueidx"], z["pos_true"], [k.split("|")[1] for k in pos_ids],
                        z["neg_S"], [k.split("|")[1] for k in neg_ids], neg_cal, len(z["ref_ids"]), order)
        # per content type (music vs speech) at the pooled thresholds
        for ct in ("music", "speech"):
            pm = np.array([kind(k) == ct for k in pos_ids])
            nm = np.array([kind(k) == ct for k in neg_ids])
            sub = summarize(z["pos_S"][pm], z["pos_I"][pm], z["pos_trueidx"][pm], z["pos_true"][pm],
                            [k.split("|")[1] for k in np.array(pos_ids)[pm]], z["neg_S"][nm],
                            [k.split("|")[1] for k in np.array(neg_ids)[nm]], neg_cal[nm], len(z["ref_ids"]), order)
            rec[f"per_attack_{ct}"] = sub["per_attack"]
        m = AUDIO_METHODS[n]()
        tj = json.loads((DESC / "audio_reg" / "timing.json").read_text()).get(n, {}) if (DESC / "audio_reg" / "timing.json").exists() else {}
        rec.update({"method": m.meta(), "timing_reg": tj, "search_s": {"pos": float(z["pos_search_s"]), "neg": float(z["neg_search_s"])},
                    "ref_bits_mean": float(z["ref_bits_mean"]),
                    "attack_family": {k: fam for k, (_, fam) in attacks.AUDIO_COPY_ATTACKS.items()}, "host": host_info()})
        rec["dedup"] = {"n_neg_flagged": len(flagged), "n_items_with_duplicate": len(dup), "rule": AUDIO_DUP_RULE,
                        "mode": AUDIO_DEDUP}
        write_json(RESULTS / f"audio{suffix}" / f"{n}.json", rec)
        dump_perquery(RESULTS / f"audio{suffix}" / "_perquery" / f"{n}.npz", pos_ids, z["pos_S"], z["pos_I"],
                      z["pos_trueidx"], neg_ids, z["neg_S"], neg_cal)
        pa = rec["per_attack"]["_all"]
        print(f"{n:16s} R@1={pa['R@1']:.3f} muAP={pa['muAP']:.3f} TPR@q1e-2={pa.get('TPR@query@0.01', float('nan')):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["render", "extract", "search", "metrics"])
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    names = [n for n in AUDIO_METHODS if not args.only or n in args.only.split(",")]
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
