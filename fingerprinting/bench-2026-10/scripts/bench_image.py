"""Image retrieval-robustness track: extract descriptors, search, score.

    python bench_image.py extract [--only m1,m2] [--sets reg,dist,pos,neg,hard,disc]
    python bench_image.py search  [--only ...]
    python bench_image.py metrics

extract  Every descriptor is computed from the same pixels: an item is (source image,
         attack); the attacked image is generated once per item and handed to every
         method. CPU methods run in a process pool over item chunks; GPU methods run in
         the main process on the images the pool returns. Output per (set, method):
         $BENCH_WORK/desc/<set>/<method>.npy (row order = <set>/ids.json) and timing.
search   faiss exact search of every query set against the index reg+dist (ABO) or
         refs (DISC21): top-K ids and scores, plus the score of the true reference.
metrics  per method x attack: R@1, muAP (ISC protocol), TPR at calibrated FPR (query-
         and pair-level), ABO hard-negative FPR; written to $BENCH_RESULTS/image/*.json.

Query sets (ABO): pos = registered sources under every attack in attacks.COPY_ATTACKS;
neg = never-registered sources under the same attacks (split by source into calib/test
halves; thresholds are fixed on calib only); hard = another photo of a registered
product (none, jpeg75, resize0.5, crop90) that must not match.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import attacks
from common import BENCH_WORK, CORPUS, DESC, OUT, RESULTS, Timer, extra_sets, host_info, load_rgb, rng_for, to_u8, write_json
from fingerprints_image import IMAGE_METHODS, Fingerprinter, pairwise_score

HARD_ATTACKS = ["none", "jpeg75", "resize0.5", "crop90"]
CHUNK = 32
TOPK = 50
MAX_SIDE = 1024


def manifest() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())


def item_lists(man: dict, sets: list[str]) -> dict[str, list[tuple[str, str, str]]]:
    """set -> [(item key, source path, attack)]"""
    a = man["image"]
    it = a["items"]
    p = lambda i: str(CORPUS / it[i]["path"])
    out: dict[str, list] = {}
    atk = list(attacks.COPY_ATTACKS)
    if "reg" in sets:
        out["reg"] = [(i, p(i), "none") for i in a["sets"]["reg"]]
    if "dist" in sets:
        out["dist"] = [(i, p(i), "none") for i in a["sets"]["dist"]]
    if "pos" in sets:
        out["pos"] = [(f"{i}|{k}", p(i), k) for k in atk for i in a["sets"]["pos"]]
    if "neg" in sets:
        out["neg"] = [(f"{i}|{k}", p(i), k) for k in atk for i in a["sets"]["neg"]]
    if "hard" in sets:
        out["hard"] = [(f"{h}|{k}", p(h), k) for k in HARD_ATTACKS for _, h in a["hard_pairs"]]
    if "disc" in sets and "disc21" in man:
        d = man["disc21"]
        dp = lambda i: str(CORPUS / d["items"][i]["path"])
        out["disc_ref"] = [(r, dp(r), "none") for r in d["refs"]]
        qs = sorted({q for q, _ in d["pairs"]} | set(d["negq"]))
        out["disc_q"] = [(q, dp(q), "none") for q in qs]
    if "edit" in sets and (OUT / "edits" / "index.json").exists():
        edits = json.loads((OUT / "edits" / "index.json").read_text())["edits"]
        out["edit"] = [(r["key"], str(OUT / r["path"]), "none") for r in edits]
    for name, items in extra_sets("image").items():
        if name in sets or "extra" in sets:
            out[name] = [(k, str(p), a) for k, p, a in items]
    return out


def make_image(path: str, attack: str, key: str) -> np.ndarray:
    x = load_rgb(Path(path), MAX_SIDE)
    fn, _ = attacks.COPY_ATTACKS[attack]
    return fn(x, rng_for("attack", attack, key.split("|")[0]))


# ------------------------------------------------------------------ worker pool
_W: dict = {}


def _init(cpu_names: list[str], query: bool, bg_paths: list[str], send_images: bool = True) -> None:
    # One thread per worker. BLAS pools are sized when numpy is first imported, so the
    # launcher must also export OMP/OPENBLAS/MKL_NUM_THREADS=1 (gpu/stage.sh does).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    import torch

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    attacks.BACKGROUNDS = list(bg_paths)  # loaded on use by attacks.onto_background
    _W["methods"] = []
    for n in cpu_names:
        m = IMAGE_METHODS[n]()
        m.load("cpu")
        _W["methods"].append(m)
    _W["query"] = query
    _W["send_images"] = send_images


def _work(chunk: list[tuple[str, str, str]]):
    imgs = [make_image(p, a, k) for k, p, a in chunk]
    desc, secs = {}, {}
    for m in _W["methods"]:
        t = time.perf_counter()
        desc[m.name] = m.extract_query(imgs) if _W["query"] else m.extract(imgs)
        secs[m.name] = time.perf_counter() - t
    # Pixels go back to the parent only when GPU methods there need them (each chunk is ~100 MB).
    return ([to_u8(x) for x in imgs] if _W["send_images"] else []), desc, secs


def _bounded_imap(pool, fn, items, max_in_flight: int):
    """Ordered imap with back-pressure: at most `max_in_flight` chunks submitted but not yet
    consumed, so fast CPU workers cannot fill RAM with decoded images while the GPU lags."""
    from collections import deque

    q: deque = deque()
    it = iter(items)
    for x in it:
        q.append(pool.apply_async(fn, (x,)))
        if len(q) >= max_in_flight:
            break
    while q:
        r = q.popleft().get()
        nxt = next(it, None)
        if nxt is not None:
            q.append(pool.apply_async(fn, (nxt,)))
        yield r


def extract(names: list[str], sets: list[str], workers: int, device: str) -> None:
    man = manifest()
    lists = item_lists(man, sets)
    bg = [str(CORPUS / man["image"]["items"][i]["path"]) for i in man["image"]["sets"]["bg"]]
    methods = {n: IMAGE_METHODS[n]() for n in names}
    for set_name, items in lists.items():
        query = set_name in ("pos", "neg", "hard", "disc_q", "edit") or set_name.endswith("_pos")
        out_dir = DESC / set_name
        todo = [n for n in names if not (out_dir / f"{n}.npy").exists()]
        if not todo:
            print(f"{set_name}: all present")
            continue
        cpu = [n for n in todo if not methods[n].gpu]
        if any(n.startswith("ISCC-SCI") for n in todo):
            import iscc_sci

            iscc_sci.get_model()  # download and verify once, before workers race for it
        gpu = [n for n in todo if methods[n].gpu]
        loaded = {}
        for n in gpu:
            loaded[n] = IMAGE_METHODS[n]()
            loaded[n].load(device)
        acc: dict[str, list] = {n: [] for n in todo}
        secs = {n: 0.0 for n in todo}
        chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cpu, query, bg, bool(gpu))) as pool:
            for ci, (imgs_u8, desc, s) in enumerate(_bounded_imap(pool, _work, chunks, 2 * workers)):
                for n in cpu:
                    acc[n].append(desc[n])
                    secs[n] += s[n]
                if gpu and device.startswith("cuda"):
                    import torch

                    # one host-to-device copy per chunk, shared by every GPU method (the per-method
                    # timings below therefore exclude this upload)
                    imgs_dev = [torch.from_numpy(x).to(device, non_blocking=True) for x in imgs_u8]
                else:
                    imgs_dev = imgs_u8
                for n in gpu:  # uint8 pixels; adapters resize on the device
                    with Timer(device) as tm:
                        acc[n].append(loaded[n].extract_query(imgs_dev) if query else loaded[n].extract(imgs_dev))
                    secs[n] += tm.elapsed
                if ci % 20 == 0:
                    done = (ci + 1) * CHUNK
                    print(f"  {set_name} {min(done, len(items))}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ids.json").write_text(json.dumps([k for k, _, _ in items]))
        # One timing file per method: several extractor processes (and hosts) may fill one set.
        for n in todo:
            np.save(out_dir / f"{n}.npy", np.concatenate(acc[n]))
            write_json(out_dir / f"{n}.timing.json",
                       {"seconds": secs[n], "items": len(items), "device": device if n in gpu else "cpu:1thread",
                        "host": host_info().get("cpu_model", "")})
        print(f"{set_name}: {len(items)} items x {len(todo)} methods in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ near-duplicate cleanup
DUP_RULE = {"PDQ": ("hamming_bits_le", 16), "SSCD-mixup": ("cos_ge", 0.80), "DINOv2-B": ("cos_ge", 0.95)}


def dedup() -> None:
    """Flag ground-truth ambiguities before scoring: ABO reuses photos of one product across
    listings and colour variants, so a distractor (or a negative source) can be a genuine
    copy of a registered image. Such a pair is not a false positive of any method.

    Rule (fixed before any results were looked at, on unattacked originals): a pair is a
    duplicate if ANY of PDQ <= 16 bits, SSCD cosine >= 0.80 or DINOv2-B cosine >= 0.95.
    Using three different families keeps the rule from favouring one of them. Flagged
    distractors leave the index; flagged negative sources leave the negative set.
    """
    ids = {s: json.loads((DESC / s / "ids.json").read_text()) for s in ("reg", "dist")}
    man = manifest()["image"]
    neg_src = man["sets"]["neg"]
    # negative sources, unattacked: rows of the neg set with attack "none"
    neg_ids = json.loads((DESC / "neg" / "ids.json").read_text())
    neg_rows = [j for j, k in enumerate(neg_ids) if k.endswith("|none")]
    flagged_dist, flagged_neg, evidence = set(), set(), {}
    for m, (rule, t) in DUP_RULE.items():
        reg = np.load(DESC / "reg" / f"{m}.npy")
        dist = np.load(DESC / "dist" / f"{m}.npy")
        negd = np.load(DESC / "neg" / f"{m}.npy")[neg_rows]
        if negd.ndim == 3:
            negd = negd[:, 0]
        anchors = np.concatenate([reg, negd])
        anchor_ids = ids["reg"] + [neg_ids[j].split("|")[0] for j in neg_rows]
        for name, cand, cand_ids in (("dist", dist, ids["dist"]), ("neg", negd, [neg_ids[j].split("|")[0] for j in neg_rows])):
            for i0 in range(0, len(cand), 4096):
                blk = cand[i0:i0 + 4096]
                if rule == "hamming_bits_le":
                    sim = pairwise_score(blk, anchors, "hamming")
                    hit = sim >= 1 - t / blk.shape[1]
                else:
                    sim = pairwise_score(blk, anchors, "ip")
                    hit = sim >= t
                for r, c in zip(*np.nonzero(hit)):
                    a, b = cand_ids[i0 + r], anchor_ids[c]
                    if a == b:
                        continue
                    (flagged_dist if name == "dist" else flagged_neg).add(a)
                    evidence.setdefault(a, []).append([m, b, float(sim[r, c])])
    # Hard negatives (another photo of a registered product): colour variants are often the
    # same photograph recoloured. Those pairs are reported separately, not as false bindings.
    hard_ids = json.loads((DESC / "hard" / "ids.json").read_text())
    pairs = {h: k for k, h in man["hard_pairs"]}
    rid = {r: j for j, r in enumerate(ids["reg"])}
    hard_rows = [j for j, k in enumerate(hard_ids) if k.endswith("|none")]
    flagged_hard = set()
    for m, (rule, t) in DUP_RULE.items():
        reg = np.load(DESC / "reg" / f"{m}.npy")
        hd = np.load(DESC / "hard" / f"{m}.npy")[hard_rows]
        hd = hd[:, 0] if hd.ndim == 3 else hd
        for r, j in enumerate(hard_rows):
            h = hard_ids[j].split("|")[0]
            own = reg[rid[pairs[h]]][None]
            sim = pairwise_score(hd[r:r + 1], own, "hamming" if rule == "hamming_bits_le" else "ip")[0, 0]
            if (rule == "hamming_bits_le" and sim >= 1 - t / hd.shape[1]) or (rule == "cos_ge" and sim >= t):
                flagged_hard.add(h)
    rec = {"rule": {k: list(v) for k, v in DUP_RULE.items()}, "dist_flagged": sorted(flagged_dist),
           "hard_near_duplicate": sorted(flagged_hard),
           "neg_flagged": sorted(flagged_neg), "n_dist": len(ids["dist"]), "n_neg_sources": len(neg_src),
           "evidence": evidence}
    write_json(DESC / "dedup.json", rec)
    write_json(RESULTS / "image" / "_dedup.json", {k: v for k, v in rec.items() if k != "evidence"} |
               {"n_dist_flagged": len(flagged_dist), "n_neg_flagged": len(flagged_neg),
                "n_hard_near_duplicate": len(flagged_hard), "n_hard": len(hard_rows)})
    print(f"dedup: {len(flagged_dist)} of {len(ids['dist'])} distractors and {len(flagged_neg)} of {len(neg_src)} "
          f"negative sources flagged")


# Sensitivity of the results to the duplicate rule (paper Sec. 3): "any" is the rule fixed in
# advance; "two" needs two of the three families to agree on the same pair; "none" skips the
# cleanup. Selected with BENCH_DEDUP; non-default rules write to suffixed search/result dirs.
DEDUP_MODE = os.environ.get("BENCH_DEDUP", "any")
SUFFIX = "" if DEDUP_MODE == "any" else f"_dedup-{DEDUP_MODE}"


def _dedup() -> dict:
    p = DESC / "dedup.json"
    if not p.exists() or DEDUP_MODE == "none":
        return {"dist_flagged": [], "neg_flagged": [], "hard_near_duplicate": json.loads(p.read_text()).get(
            "hard_near_duplicate", []) if p.exists() else []}
    d = json.loads(p.read_text())
    if DEDUP_MODE == "two":
        from collections import defaultdict

        def two(k):
            by = defaultdict(set)
            for m, anchor, _ in d["evidence"].get(k, []):
                by[anchor].add(m)
            return any(len(v) >= 2 for v in by.values())

        d = {**d, "dist_flagged": [k for k in d["dist_flagged"] if two(k)],
             "neg_flagged": [k for k in d["neg_flagged"] if two(k)]}
    return d


# ------------------------------------------------------------------ search
def _index(desc: np.ndarray, metric: str):
    import faiss

    if metric == "hamming":
        idx = faiss.IndexBinaryFlat(desc.shape[1] if desc.shape[1] % 8 == 0 else (desc.shape[1] + 7) // 8 * 8)
        idx.add(_pack(desc))
        return idx
    idx = faiss.IndexFlatIP(desc.shape[1]) if metric == "ip" else faiss.IndexFlatL2(desc.shape[1])
    idx.add(np.ascontiguousarray(desc, dtype=np.float32))
    return idx


def _pack(bits: np.ndarray) -> np.ndarray:
    n = bits.shape[1]
    if n % 8:
        bits = np.pad(bits, ((0, 0), (0, 8 - n % 8)))
    return np.packbits(bits.astype(np.uint8), axis=1)


def _search(idx, q: np.ndarray, metric: str, dim: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scores, ids), scores in the method's similarity convention."""
    if q.ndim == 3:  # query-side variants: best score over variants per reference
        allS, allI = zip(*(_search(idx, q[:, v], metric, dim, k) for v in range(q.shape[1])))
        S, I = np.concatenate(allS, 1), np.concatenate(allI, 1)
        outS, outI = np.empty((len(q), k), np.float32), np.empty((len(q), k), np.int64)
        for r in range(len(q)):
            best: dict[int, float] = {}
            for s, i in zip(S[r], I[r]):
                if i >= 0 and s > best.get(i, -1e9):
                    best[i] = s
            top = sorted(best.items(), key=lambda t: -t[1])[:k]
            outI[r], outS[r] = [t[0] for t in top], [t[1] for t in top]
        return outS, outI
    if metric == "hamming":
        D, I = idx.search(_pack(q), k)
        return (1 - D / dim).astype(np.float32), I
    D, I = idx.search(np.ascontiguousarray(q, dtype=np.float32), k)
    return (D if metric == "ip" else -np.sqrt(np.maximum(D, 0))).astype(np.float32), I


def search(names: list[str]) -> None:
    import faiss

    # Workers run single-threaded (OMP_NUM_THREADS=1 in gpu/stage.sh); exact search should not.
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    man = manifest()
    for n in names:
        m = IMAGE_METHODS[n]()
        edit = ["edit"] if (DESC / "edit" / f"{n}.npy").exists() else []
        for index_sets, query_sets, tag in ((["reg", "dist"], ["pos", "neg", "hard"] + edit, "abo"),
                                            (["disc_ref"], ["disc_q"], "disc")):
            if not all((DESC / s / f"{n}.npy").exists() for s in index_sets + query_sets):
                continue
            out = DESC / f"search_{tag}{SUFFIX if tag == 'abo' else ''}" / f"{n}.npz"
            prev = dict(np.load(out)) if out.exists() else {}
            missing = [qs for qs in query_sets if f"{qs}_S" not in prev]
            if not missing:
                continue
            query_sets = missing  # incremental: only query sets not searched yet (e.g. edits added later)
            ref = np.concatenate([np.load(DESC / s / f"{n}.npy") for s in index_sets])
            ref_ids = sum((json.loads((DESC / s / "ids.json").read_text()) for s in index_sets), [])
            if tag == "abo":
                drop = set(_dedup()["dist_flagged"])
                keep = np.array([r not in drop for r in ref_ids])
                ref, ref_ids = ref[keep], [r for r, k in zip(ref_ids, keep) if k]
            idx = _index(ref, m.metric)
            res = {}
            t0 = time.time()
            for qs in query_sets:
                q = np.load(DESC / qs / f"{n}.npy")
                S, I = _search(idx, q, m.metric, m.dim, TOPK)
                res[f"{qs}_S"], res[f"{qs}_I"] = S, I
                if qs in ("pos", "disc_q", "edit"):
                    # score against the true reference (whether or not it is in the top-K)
                    qids = json.loads((DESC / qs / "ids.json").read_text())
                    pos = {r: j for j, r in enumerate(ref_ids)}
                    if qs in ("pos", "edit"):
                        true = [pos[k.split("|")[0]] for k in qids]
                    else:
                        gt = dict(map(tuple, man["disc21"]["pairs"]))
                        true = [pos.get(gt.get(k, ""), -1) for k in qids]
                    true = np.array(true)
                    ts = np.full(len(q), np.nan, np.float32)
                    ok = true >= 0
                    qq = q[ok]
                    rr = ref[true[ok]]
                    ts[ok] = np.array([m.score(qq[i:i + 1], rr[i:i + 1])[0, 0] for i in range(len(qq))]) \
                        if q.ndim == 3 else _rowwise(qq, rr, m.metric)
                    res[f"{qs}_true"], res[f"{qs}_trueidx"] = ts, true
            if prev and list(prev["ref_ids"]) != ref_ids:
                raise SystemExit(f"{out}: index changed since the earlier search; delete it and search again")
            out.parent.mkdir(parents=True, exist_ok=True)
            prev.pop("ref_ids", None)
            np.savez_compressed(out, **{**prev, **res}, ref_ids=np.array(ref_ids))
            print(f"search {tag} {n}: {len(ref)} refs, {time.time() - t0:.0f}s")


def _rowwise(q: np.ndarray, r: np.ndarray, metric: str) -> np.ndarray:
    if metric == "hamming":
        return 1 - (q != r).mean(1)
    if metric == "ip":
        return (q.astype(np.float32) * r.astype(np.float32)).sum(1)
    return -np.linalg.norm(q - r, axis=1)


# ------------------------------------------------------------------ metrics
def metrics() -> None:
    from analysis_stats import cp_interval
    from retrieval_metrics import dump_perquery, summarize

    man = manifest()
    a = man["image"]
    calib_src = set(sorted(a["sets"]["neg"])[::2])  # half the negative sources calibrate thresholds
    atk_family = {k: fam for k, (_, fam) in attacks.COPY_ATTACKS.items()}
    pos_ids = json.loads((DESC / "pos" / "ids.json").read_text())
    neg_ids = json.loads((DESC / "neg" / "ids.json").read_text())
    hard_ids = json.loads((DESC / "hard" / "ids.json").read_text())
    neg_cal = np.array([k.split("|")[0] in calib_src for k in neg_ids])
    dd = _dedup()
    neg_flagged = set(dd["neg_flagged"])
    hard_nd = set(dd.get("hard_near_duplicate", []))
    neg_keep = np.array([k.split("|")[0] not in neg_flagged for k in neg_ids])
    for f in sorted((DESC / f"search_abo{SUFFIX}").glob("*.npz")):
        n = f.stem
        m = IMAGE_METHODS[n]()
        z = np.load(f)
        ref_ids = list(z["ref_ids"])
        rec = summarize(z["pos_S"], z["pos_I"], z["pos_trueidx"], z["pos_true"], [k.split("|")[1] for k in pos_ids],
                        z["neg_S"][neg_keep], [k.split("|")[1] for k, ok in zip(neg_ids, neg_keep) if ok],
                        neg_cal[neg_keep], len(ref_ids), list(attacks.COPY_ATTACKS))
        # Hard negatives: another photo of a registered product. A top-1 on the product's
        # registered image that passes the threshold is a false binding.
        hS, hI = z["hard_S"], z["hard_I"]
        pairs = {h: k for k, h in a["hard_pairs"]}
        rid = {r: j for j, r in enumerate(ref_ids)}
        own = np.array([rid.get(pairs[k.split("|")[0]], -1) for k in hard_ids])
        hard = {}
        hdup = np.array([k.split("|")[0] in hard_nd for k in hard_ids])
        for name, t in rec["thresholds"].items():
            if not np.isfinite(t):
                continue
            fb = (hI[:, 0] == own) & (hS[:, 0] >= t)
            # primary: distinct photographs of the product; variants (same photo recoloured) apart
            d, v = fb[~hdup], fb[hdup]
            hard[name] = {"false_bind": int(d.sum()), "n": int(len(d)), "rate": float(d.mean()) if len(d) else float("nan"),
                          "ci": cp_interval(int(d.sum()), int(len(d))),
                          "variant_bind": int(v.sum()), "n_variant": int(len(v))}
        rec["hard_negative"] = hard
        rec["hard_top1_is_own"] = float((hI[:, 0] == own).mean())
        rec["timing"] = {}
        for s in ("reg", "dist", "pos"):
            tj = DESC / s / f"{n}.timing.json"
            if tj.exists():
                rec["timing"][s] = json.loads(tj.read_text())
        rec.update({"method": m.meta(), "attack_family": atk_family, "host": host_info()})
        dz = DESC / "search_disc" / f"{n}.npz"
        if dz.exists():
            rec["disc21"] = disc_metrics(dz, man)
        write_json(RESULTS / f"image{SUFFIX}" / f"{n}.json", rec)
        dump_perquery(RESULTS / f"image{SUFFIX}" / "_perquery" / f"{n}.npz", pos_ids, z["pos_S"], z["pos_I"],
                      z["pos_trueidx"], [k for k, ok in zip(neg_ids, neg_keep) if ok], z["neg_S"][neg_keep],
                      neg_cal[neg_keep])
        pa = rec["per_attack"]["_all"]
        print(f"{n:16s} R@1={pa['R@1']:.3f} muAP={pa['muAP']:.3f} TPR@q1e-2={pa.get('TPR@query@0.01', float('nan')):.3f} "
              f"hardFB@q1e-2={hard.get('query@0.01', {}).get('rate', float('nan')):.3f} disc muAP={rec.get('disc21', {}).get('muAP', float('nan')):.3f}")


PAIRED = [("SSCD-mixup", "DINOv2-S"), ("DINOv2-S", "DINOv2-S-LSH256"), ("PDQ", "ISCC-Image-64"),
          ("PDQ", "PDQ-dihedral"), ("ISCC-Image-64", "ISCC-Image-256"), ("SSCD-mixup", "ISC21-1st"),
          ("DINOv2-S", "OpenCLIP-B32"), ("pHash-64", "PDQ")]


def paired_tests() -> None:
    """Planned paired comparisons on the same positive queries: exact McNemar test on top-1
    correctness at the pair@1e-7 operating point (correct top-1 AND score >= threshold),
    Holm-adjusted over the planned pairs. Written to results/image/_paired.json."""
    from scipy.stats import binomtest

    pos_ids = json.loads((DESC / "pos" / "ids.json").read_text())
    ok = {}
    for n in {m for pair in PAIRED for m in pair}:
        f, r = DESC / "search_abo" / f"{n}.npz", RESULTS / "image" / f"{n}.json"
        if not (f.exists() and r.exists()):
            continue
        z = np.load(f)
        t = json.loads(r.read_text())["thresholds"].get("pair@1e-07", float("nan"))
        ok[n] = (z["pos_I"][:, 0] == z["pos_trueidx"]) & (z["pos_S"][:, 0] >= t)
    rows = []
    for a, b in PAIRED:
        if a not in ok or b not in ok:
            continue
        only_a, only_b = int((ok[a] & ~ok[b]).sum()), int((~ok[a] & ok[b]).sum())
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append({"a": a, "b": b, "n": len(pos_ids), "tpr_a": float(ok[a].mean()), "tpr_b": float(ok[b].mean()),
                     "only_a": only_a, "only_b": only_b, "p": float(p)})
    order = np.argsort([r["p"] for r in rows])
    m = len(rows)
    running = 0.0
    for rank, i in enumerate(order):  # Holm step-down
        running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    write_json(RESULTS / "image" / "_paired.json", {"operating_point": "pair@1e-07", "test": "exact McNemar", "rows": rows})


def disc_metrics(path: Path, man: dict) -> dict:
    from retrieval_metrics import micro_ap

    z = np.load(path)
    q_ids = json.loads((DESC / "disc_q" / "ids.json").read_text())
    gt = dict(map(tuple, man["disc21"]["pairs"]))
    ref_ids = list(z["ref_ids"])
    S, I, tidx = z["disc_q_S"], z["disc_q_I"], z["disc_q_trueidx"]
    has = np.array([q in gt for q in q_ids])
    C = (I[:, :10] == tidx[:, None]) & has[:, None]
    return {"muAP": micro_ap(S[:, :10].ravel(), C.ravel(), int(has.sum())),
            "R@1": float((I[has, 0] == tidx[has]).mean()), "n_pos": int(has.sum()), "n_neg": int((~has).sum()),
            "n_ref": len(ref_ids)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "dedup", "search", "metrics"])
    ap.add_argument("--only", default="")
    ap.add_argument("--sets", default="reg,dist,pos,neg,hard,disc,edit")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    names = [n for n in IMAGE_METHODS if not args.only or n in args.only.split(",")]
    if args.stage == "extract":
        import torch

        dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        extract(names, args.sets.split(","), args.workers, dev)
    elif args.stage == "dedup":
        dedup()
    elif args.stage == "search":
        search(names)
    else:
        metrics()
        if not SUFFIX:
            paired_tests()


if __name__ == "__main__":
    main()
