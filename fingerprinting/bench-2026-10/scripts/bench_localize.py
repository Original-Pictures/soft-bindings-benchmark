"""Partial-edit track: retrieval under edit, spatial localization, temporal localization.

    python bench_localize.py retrieval [--direct]   # R@1 / TPR of edited queries, stratified
    python bench_localize.py localize  [--device]   # where was the image changed?
    python bench_localize.py temporal  [--device]   # where in the reference does the query sit?
    python bench_localize.py all

Retrieval. Edited images (edits.py) are a query set like `pos`: bench_image extracts
their descriptors as set "edit" and search writes edit_S / edit_I / edit_trueidx into
search_abo/<method>.npz. Here they are scored at each method's thresholds from
$BENCH_RESULTS/image/<method>.json (fixed on negatives, never on edits) and stratified by
edit type, area and post-processing. `--direct` computes a few methods in-process
(smoke/tests only), searching reg+dist descriptors already in $BENCH_WORK/desc.

Localization assumes the registered original was found (oracle retrieval; results are
also reported on the subset each retrieval method actually found). The query is aligned
to the original (SIFT + RANSAC homography; DISK + LightGlue, kornia, as the alternative;
identity + resize when fewer than MIN_INLIERS agree) and compared in the original's
frame at WORK px on the long side. Maps (higher = more likely edited):
  pixel  blurred mean absolute difference       ssim   1 - SSIM map (7 px window)
  lpips  LPIPS-alex spatial map                 dino   DINOv2-S patch-token cosine distance
  pdq    PDQ Hamming per tile of an 8x8 grid    fused  mean of the null-CDF values of the others
Calibration uses UNEDITED originals put through the same post-processing on half of the
sources: each map's null distribution, pooled over the post-processing levels (the
verifier does not know which was applied), fixes the threshold at pixel FPR 1 % and the
null CDF used by `fused`. Evaluation uses the other half. Metrics: pixel ROC AUC,
F1 and IoU at the calibrated threshold, oracle-best F1 (upper bound on thresholding).

Temporal localization (audio, video) needs no calibration for offsets: the ground truth
offset is reproduced from the attack's seeded generator (audio) or read from the sidecar
JSON written when the query was rendered (video).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import CORPUS, DESC, OUT, RESULTS, load_rgb, rng_for, to_u8, write_json

EDIT_DIR = OUT / "edits"
LOC_OUT = OUT / "localize"
WORK = 512
MIN_INLIERS = 12
MAPS = ["pixel", "ssim", "lpips", "dino", "pdq"]
FPR = 0.01


def edit_index() -> list[dict]:
    return json.loads((EDIT_DIR / "index.json").read_text())["edits"]


def image_manifest() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())["image"]


def split(sources: list[str]) -> set[str]:
    """Calibration half of the sources (the other half is evaluated)."""
    return set(sorted(set(sources))[::2])


# ================================================================== retrieval under edit
def _strata(rows: list[dict], hit: np.ndarray, key: str) -> dict:
    out = {}
    for v in sorted({r[key] for r in rows}, key=str):
        m = np.array([r[key] == v for r in rows])
        out[str(v)] = {"n": int(m.sum()), "rate": float(hit[m].mean())}
    return out


def score_retrieval(name: str, rows: list[dict], S1: np.ndarray, I1: np.ndarray, tidx: np.ndarray, th: dict) -> dict:
    from analysis_stats import cp_interval

    ok = I1 == tidx
    rec = {"method": name, "n": len(rows), "R@1": float(ok.mean()), "R@1_ci": cp_interval(int(ok.sum()), len(rows)),
           "R@1_by": {k: _strata(rows, ok, k) for k in ("type", "area", "post")}, "at_threshold": {}}
    for tname, t in th.items():
        if t is None or not np.isfinite(t):
            continue
        hit = ok & (S1 >= t)
        rec["at_threshold"][tname] = {"TPR": float(hit.mean()), "ci": cp_interval(int(hit.sum()), len(rows)),
                                      "by": {k: _strata(rows, hit, k) for k in ("type", "area", "post")}}
    rec["per_edit_hit"] = {r["key"]: bool(o) for r, o in zip(rows, ok)}
    return rec


def retrieval(direct: bool, device: str) -> None:
    rows = edit_index()
    key_row = {r["key"]: r for r in rows}
    if direct:
        return _retrieval_direct(rows, device)
    for f in sorted((DESC / "search_abo").glob("*.npz")):
        z = np.load(f)
        if "edit_S" not in z:
            continue
        ids = json.loads((DESC / "edit" / "ids.json").read_text())
        r = [key_row[k] for k in ids]
        th = json.loads((RESULTS / "image" / f"{f.stem}.json").read_text())["thresholds"]
        rec = score_retrieval(f.stem, r, z["edit_S"][:, 0], z["edit_I"][:, 0], z["edit_trueidx"], th)
        write_json(RESULTS / "localize" / "retrieval" / f"{f.stem}.json", rec)
        print(f"edit-retrieval {f.stem:16s} R@1={rec['R@1']:.3f}")


def _retrieval_direct(rows: list[dict], device: str) -> None:
    """Smoke path: descriptors computed here for a few methods against desc/{reg,dist}."""
    from fingerprints_image import IMAGE_METHODS

    names = [n for n in ("PDQ", "pHash-64", "ISCC-Image-64", "DINOv2-S", "SSCD-mixup")
             if all((DESC / s / f"{n}.npy").exists() for s in ("reg", "dist"))]
    ref_ids = sum((json.loads((DESC / s / "ids.json").read_text()) for s in ("reg", "dist")), [])
    rid = {r: j for j, r in enumerate(ref_ids)}
    imgs = [load_rgb(OUT / r["path"]) for r in rows]
    tidx = np.array([rid[r["src"]] for r in rows])
    for n in names:
        m = IMAGE_METHODS[n]()
        m.load(device if m.gpu else "cpu")
        ref = np.concatenate([np.load(DESC / s / f"{n}.npy") for s in ("reg", "dist")])
        q = m.extract_query(imgs)
        S = m.score(q, ref)
        I1, S1 = S.argmax(1), S.max(1)
        rj = RESULTS / "image" / f"{n}.json"
        th = json.loads(rj.read_text())["thresholds"] if rj.exists() else {}
        rec = score_retrieval(n, rows, S1, I1, tidx, th)
        rec["mode"] = "direct (smoke)"
        write_json(RESULTS / "localize" / "retrieval" / f"{n}.json", rec)
        by = rec["R@1_by"]["area"]
        print(f"edit-retrieval {n:14s} R@1={rec['R@1']:.3f}  by area " + " ".join(f"{a}:{v['rate']:.2f}" for a, v in by.items()))


# ================================================================== alignment
_KF: dict = {}


def align_sift(q: np.ndarray, o: np.ndarray) -> tuple[np.ndarray | None, int]:
    import cv2

    g1 = cv2.cvtColor(to_u8(q), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(to_u8(o), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(4000)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H if n >= MIN_INLIERS else None), n


def align_lightglue(q: np.ndarray, o: np.ndarray, device: str) -> tuple[np.ndarray | None, int]:
    import cv2
    import kornia.feature as KF
    import torch

    if "disk" not in _KF:
        _KF["disk"] = KF.DISK.from_pretrained("depth").eval().to(device)
        _KF["lg"] = KF.LightGlue("disk").eval().to(device)

    def feats(x):
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().to(device)
        h, w = t.shape[-2:]
        t = torch.nn.functional.pad(t, (0, (-w) % 16, 0, (-h) % 16))
        f = _KF["disk"](t, n=2048, pad_if_not_divisible=True)[0]
        return {"keypoints": f.keypoints[None], "descriptors": f.descriptors[None],
                "image_size": torch.tensor([[w, h]], device=device).float()}

    with torch.no_grad():
        f0, f1 = feats(q), feats(o)
        out = _KF["lg"]({"image0": f0, "image1": f1})
    m = out["matches"][0].cpu().numpy()
    if len(m) < MIN_INLIERS:
        return None, len(m)
    src = f0["keypoints"][0].cpu().numpy()[m[:, 0]]
    dst = f1["keypoints"][0].cpu().numpy()[m[:, 1]]
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H if n >= MIN_INLIERS else None), n


def warp_to(q: np.ndarray, o_shape: tuple[int, int], H: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Query in the original's frame + validity mask. H=None: identity up to scale."""
    import cv2

    h, w = o_shape
    if H is None:
        return cv2.resize(q, (w, h), interpolation=cv2.INTER_AREA), np.ones((h, w), bool)
    wq = cv2.warpPerspective(q, H, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.ones(q.shape[:2], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return wq, valid


# ================================================================== difference maps
_M: dict = {}


def _lpips(device):
    if "lpips" not in _M:
        import lpips

        _M["lpips"] = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval().to(device)
    return _M["lpips"]


def _dino(device):
    if "dino" not in _M:
        from transformers import AutoModel

        _M["dino"] = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(device)
    return _M["dino"]


def diff_maps(a: np.ndarray, b: np.ndarray, device: str) -> dict[str, np.ndarray]:
    """a = original, b = aligned query, both float HxWx3 at working size."""
    import cv2
    import pdqhash
    import torch
    import torch.nn.functional as F
    from skimage.metrics import structural_similarity

    h, w = a.shape[:2]
    out = {}
    ab, bb = cv2.GaussianBlur(a, (0, 0), 1.0), cv2.GaussianBlur(b, (0, 0), 1.0)
    out["pixel"] = cv2.GaussianBlur(np.abs(ab - bb).mean(-1), (0, 0), 2.0)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    _, s = structural_similarity(ga, gb, win_size=7, data_range=1.0, full=True)
    out["ssim"] = cv2.GaussianBlur((1 - s).astype(np.float32), (0, 0), 2.0)
    with torch.no_grad():
        ta = torch.from_numpy(a.transpose(2, 0, 1))[None].float().to(device) * 2 - 1
        tb = torch.from_numpy(b.transpose(2, 0, 1))[None].float().to(device) * 2 - 1
        out["lpips"] = _lpips(device)(ta, tb)[0, 0].float().cpu().numpy()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = torch.cat([ta, tb]) * 0.5 + 0.5
        x = F.interpolate(x, size=(448, 448), mode="bicubic", align_corners=False, antialias=True)
        tok = _dino(device)(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
        tok = F.normalize(tok, dim=-1)
        d = (1 - (tok[0] * tok[1]).sum(-1)).reshape(1, 1, 32, 32)
        out["dino"] = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    tiles = np.zeros((8, 8), np.float32)
    ua, ub = to_u8(a), to_u8(b)
    for i in range(8):
        for j in range(8):
            ys, xs = slice(i * h // 8, (i + 1) * h // 8), slice(j * w // 8, (j + 1) * w // 8)
            ha = pdqhash.compute(np.ascontiguousarray(ua[ys, xs]))[0]
            hb = pdqhash.compute(np.ascontiguousarray(ub[ys, xs]))[0]
            tiles[i, j] = (ha != hb).mean()
    out["pdq"] = cv2.resize(tiles, (w, h), interpolation=cv2.INTER_NEAREST)
    if out["lpips"].shape != (h, w):
        out["lpips"] = cv2.resize(out["lpips"], (w, h), interpolation=cv2.INTER_LINEAR)
    return {k: v.astype(np.float32) for k, v in out.items()}


def _work_size(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    s = WORK / max(h, w)
    return max(16, round(h * s)), max(16, round(w * s))


def prepare(orig: np.ndarray, query: np.ndarray, aligner: str, device: str) -> dict:
    """Align at working resolution; returns aligned pair, validity and alignment record."""
    import cv2

    hw = _work_size(orig.shape[:2])
    o = cv2.resize(orig, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    s = WORK / max(query.shape[:2])
    q = cv2.resize(query, (max(16, round(query.shape[1] * min(1, s))), max(16, round(query.shape[0] * min(1, s)))),
                   interpolation=cv2.INTER_AREA)
    H, n = align_sift(q, o) if aligner == "sift" else align_lightglue(q, o, device)
    wq, valid = warp_to(q, hw, H)
    # Pixels the query does not cover (crop, rotation) are filled from the original so the
    # window-based maps (SSIM, LPIPS, DINOv2 patches, PDQ tiles) do not fire on the black
    # border; they are also excluded from scoring via `valid`.
    wq = np.where(valid[..., None], wq, o).astype(np.float32)
    return {"o": o, "q": wq, "valid": valid, "aligned": H is not None, "inliers": n}


# ================================================================== localization
def _pixels(m: np.ndarray, valid: np.ndarray, rng: np.random.Generator, k: int = 4000) -> np.ndarray:
    v = m[valid]
    return v if len(v) <= k else v[rng.choice(len(v), k, replace=False)]


def calibrate(rows: list[dict], cal_src: set[str], aligner: str, device: str, max_sources: int) -> dict:
    """Null distribution of every map on unedited originals put through each post.

    The same sampled pixels are kept for all maps, so the fused map's own null (mean of
    null-CDF values) comes from the same pass."""
    from edits import post_process

    man = image_manifest()
    joint: dict[str, list[np.ndarray]] = {}
    srcs = sorted({r["src"] for r in rows if r["src"] in cal_src})[:max_sources]
    posts = sorted({r["post"] for r in rows})
    rng = rng_for("loc-calib")
    for s in srcs:
        x = load_rgb(CORPUS / man["items"][s]["path"], 1024)
        for p in posts:
            q, _ = post_process(x, p, rng_for("loc-calib-post", s, p))
            pr = prepare(x, q, aligner, device)
            maps = diff_maps(pr["o"], pr["q"], device)
            idx = np.flatnonzero(pr["valid"].ravel())
            idx = idx if len(idx) <= 4000 else rng.choice(idx, 4000, replace=False)
            joint.setdefault(p, []).append(np.stack([maps[k].ravel()[idx] for k in MAPS], 1))
    # One null pooled over the post-processing levels: a verifier does not know how the
    # query was re-encoded, and a per-post null for lossless queries is degenerate (all 0).
    J = np.concatenate([x for parts in joint.values() for x in parts])
    pooled = {k: {"sorted": np.sort(J[:, i]), "thr": float(np.quantile(J[:, i], 1 - FPR))} for i, k in enumerate(MAPS)}
    fused = np.mean([_cdf(pooled[k]["sorted"], J[:, i]) for i, k in enumerate(MAPS)], axis=0)
    pooled["fused"] = {"thr": float(np.quantile(fused, 1 - FPR))}
    cal = {p: pooled for p in joint}
    cal["_n_sources"] = len(srcs)
    return cal


def _cdf(sorted_null: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF: flat backgrounds give many exact ties (often 0) in the null,
    and a right-continuous CDF would score every tied pixel as extreme."""
    lo = np.searchsorted(sorted_null, v, side="left")
    hi = np.searchsorted(sorted_null, v, side="right")
    return (lo + hi) / (2.0 * len(sorted_null))


def fused_map(maps: dict[str, np.ndarray], cal_p: dict) -> np.ndarray:
    return np.mean([_cdf(cal_p[k]["sorted"], maps[k]) for k in MAPS], axis=0).astype(np.float32)


def _f1_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    f1 = 2 * tp / max(1.0, 2 * tp + fp + fn)
    iou = tp / max(1.0, tp + fp + fn)
    return f1, iou


def _best_f1(score: np.ndarray, gt: np.ndarray) -> float:
    o = np.argsort(-score)
    g = gt[o]
    tp = np.cumsum(g)
    k = np.arange(1, len(g) + 1)
    f1 = 2 * tp / (k + g.sum())
    return float(f1.max()) if len(f1) else 0.0


def _auc(score: np.ndarray, gt: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if gt.all() or (~gt).all():
        return float("nan")
    return float(roc_auc_score(gt, score))


def evaluate_edit(r: dict, cal: dict, aligner: str, device: str, keep: bool = False) -> dict:
    import cv2

    man = image_manifest()
    x = load_rgb(CORPUS / man["items"][r["src"]]["path"], 1024)
    q = load_rgb(OUT / r["path"])
    gt_full = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(OUT / r["mask"]), np.float32) / 255.0
    pr = prepare(x, q, aligner, device)
    hw = pr["o"].shape[:2]
    gt = cv2.resize(gt_full, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) > 0.5
    valid = pr["valid"]
    maps = diff_maps(pr["o"], pr["q"], device)
    cp = cal[r["post"]]
    maps["fused"] = fused_map(maps, cp)
    res = {"key": r["key"], "type": r["type"], "area": r["area"], "post": r["post"], "aligned": pr["aligned"],
           "inliers": pr["inliers"], "gt_in_valid": float(gt[valid].mean()) if valid.any() else 0.0}
    g = gt[valid]
    for k, m in maps.items():
        thr = cp[k]["thr"]
        s = m[valid]
        # strict: with no post-processing the null is all zeros and the threshold is 0
        f1, iou = _f1_iou(s > thr, g)
        res[k] = {"auc": _auc(s, g), "f1": f1, "iou": iou, "best_f1": _best_f1(s, g)}
    if keep:
        res["_maps"], res["_pair"], res["_gt"] = maps, (pr["o"], pr["q"]), gt
    return res


def summarize_loc(res: list[dict]) -> dict:
    out = {"n": len(res), "aligned_rate": float(np.mean([r["aligned"] for r in res]))}
    for k in MAPS + ["fused"]:
        out[k] = {m: float(np.nanmean([r[k][m] for r in res])) for m in ("auc", "f1", "iou", "best_f1")}
        out[k]["by"] = {}
        for f in ("type", "area", "post"):
            out[k]["by"][f] = {str(v): {m: float(np.nanmean([r[k][m] for r in res if r[f] == v])) for m in ("auc", "f1", "iou")}
                               for v in sorted({r[f] for r in res}, key=str)}
    return out


def localize(device: str, aligners: tuple[str, ...] = ("sift", "lightglue"), limit: int | None = None,
             cal_sources: int = 100) -> None:
    rows = edit_index()
    cal_src = split([r["src"] for r in rows])
    ev = [r for r in rows if r["src"] not in cal_src][:limit]
    hits = {}
    for f in sorted((RESULTS / "localize" / "retrieval").glob("*.json")):
        hits[f.stem] = json.loads(f.read_text()).get("per_edit_hit", {})
    for al in aligners:
        t0 = time.time()
        cal = calibrate(rows, cal_src, al, device, cal_sources)
        n_cal = cal.pop("_n_sources")
        t_cal = time.time() - t0
        res = []
        for i, r in enumerate(ev):
            res.append(evaluate_edit(r, cal, al, device))
        t_ev = time.time() - t0 - t_cal
        rec = {"aligner": al, "work_px": WORK, "fpr_target": FPR, "n_cal_sources": n_cal,
               "thresholds": {k: v["thr"] for k, v in next(iter(cal.values())).items()},
               "oracle": summarize_loc(res),
               "conditional_on_retrieval": {m: summarize_loc([x for x in res if h.get(x["key"])])
                                            for m, h in hits.items() if any(h.get(x["key"]) for x in res)},
               "timing_s": {"calibrate": t_cal, "evaluate": t_ev, "per_edit": t_ev / max(1, len(res))},
               "per_edit": res}
        write_json(RESULTS / "localize" / f"spatial_{al}.json", rec)
        o = rec["oracle"]
        print(f"localize[{al}] n={len(res)} aligned={o['aligned_rate']:.2f} " +
              " ".join(f"{k}:AUC={o[k]['auc']:.2f}/F1={o[k]['f1']:.2f}" for k in MAPS + ["fused"]) +
              f"  {rec['timing_s']['per_edit']:.2f}s/edit")
        if al == "sift":
            gallery(ev, cal, al, device)


def gallery(ev: list[dict], cal: dict, al: str, device: str, n: int = 8) -> None:
    """Paper figure material: original / edited / GT / pixel / dino / fused for n edits."""
    from PIL import Image

    LOC_OUT.mkdir(parents=True, exist_ok=True)
    picks, seen, used = [], set(), set()
    for r in ev:  # one per edit type, each from a different source, then larger edits
        if r["type"] not in seen and r["src"] not in used and r["area"] >= 0.03 and r["post"] != "crop90":
            picks.append(r)
            seen.add(r["type"])
            used.add(r["src"])
    picks += [r for r in ev if r not in picks and r["area"] >= 0.10][: max(0, n - len(picks))]
    meta = []
    for r in picks[:n]:
        e = evaluate_edit(r, cal, al, device, keep=True)
        o, q = e["_pair"]
        tiles = [o, q, np.repeat(e["_gt"][..., None].astype(np.float32), 3, -1)]
        for k in ("pixel", "dino", "fused"):
            m = e["_maps"][k]
            m = (m - m.min()) / max(1e-9, np.ptp(m))
            tiles.append(np.repeat(m[..., None], 3, -1))
        row = np.concatenate([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=1.0) for t in tiles], axis=1)
        stem = LOC_OUT / r["key"].replace("|", "__")
        Image.fromarray(to_u8(row)).save(str(stem) + ".png")
        np.savez_compressed(str(stem) + ".npz", orig=to_u8(o), query=to_u8(q), gt=e["_gt"],
                            **{k: v.astype(np.float16) for k, v in e["_maps"].items()})
        meta.append({"key": r["key"], "type": r["type"], "area": r["area"], "post": r["post"],
                     "fused_f1": e["fused"]["f1"], "columns": ["original", "edited", "gt", "pixel", "dino", "fused"]})
    (LOC_OUT / "gallery.json").write_text(json.dumps(meta, indent=1))


# ================================================================== temporal: audio
EXCERPT_ATTACKS = {"excerpt10": 10.0, "excerpt5": 5.0, "excerpt10_mp3_noise20": 10.0, "rerecord": 10.0}
CHROMA_HOP_S = 4096 / 3 / 11025  # Chromaprint: 11025 Hz, 4096-sample frame, 1/3 overlap
AUDFPRINT_HOP_S = 256 / 11025


def audio_gt_offset(src: str, attack: str, n_samples: int, sr: int) -> float:
    """Replicates attacks.a_excerpt's draw (it is the first draw in every chain that uses it)."""
    n = int(EXCERPT_ATTACKS[attack] * sr)
    if n_samples <= n:
        return 0.0
    rng = rng_for("audio-attack", attack, src)
    return int(rng.integers(0, n_samples - n)) / sr


def _chroma_offset(q: np.ndarray, r: np.ndarray) -> float:
    from fingerprints_audio import _popcount

    best, bo = -1.0, 0
    for o in range(0, max(1, len(r) - len(q) + 1)):
        n = min(len(q), len(r) - o)
        if n < min(20, len(q)):
            continue
        s = 1 - _popcount(q[:n] ^ r[o:o + n]).sum() / (32.0 * n)
        if s > best:
            best, bo = s, o
    return bo * CHROMA_HOP_S


def temporal_audio() -> dict:
    import soundfile as sf

    from fingerprints_audio import Audfprint, fpcalc

    man = json.loads((CORPUS / "corpus_manifest.json").read_text())["audio"]
    af = Audfprint()
    af.load("cpu")
    from hash_table import HashTable

    rows = []
    for src in man["sets"]["reg"]:
        rp = CORPUS / man["items"][src]["path"]
        y, sr = sf.read(rp, dtype="float32")
        y = y if y.ndim == 1 else y.mean(1)
        rc = fpcalc(y, sr)
        ra = af.extract(y, sr)
        for atk in EXCERPT_ATTACKS:
            qp = OUT / "audio" / "pos" / f"{src}__{atk}.flac"
            if not qp.exists():
                continue
            q, qsr = sf.read(qp, dtype="float32")
            gt = audio_gt_offset(src, atk, len(y), sr)
            rec = {"src": src, "attack": atk, "gt_s": gt, "kind": "music" if src.startswith("m") else "speech"}
            rec["chromaprint_s"] = _chroma_offset(fpcalc(q, qsr), rc)
            ht = HashTable(hashbits=20, depth=100, maxtime=16384)
            ht.store("ref", ra)
            res = af.matcher.match_hashes(ht, af.extract(q, qsr))
            rec["audfprint_s"] = float(res[0][2]) * AUDFPRINT_HOP_S if len(res) else float("nan")
            rows.append(rec)
    out = {}
    for m in ("chromaprint", "audfprint"):
        for atk in list(EXCERPT_ATTACKS) + ["_all"]:
            sel = [r for r in rows if atk == "_all" or r["attack"] == atk]
            if not sel:
                continue
            err = np.array([abs(r[f"{m}_s"] - r["gt_s"]) for r in sel])
            out.setdefault(m, {})[atk] = {"n": len(sel), "median_abs_err_s": float(np.nanmedian(err)),
                                          "within_0.5s": float(np.mean(err <= 0.5)), "within_2s": float(np.mean(err <= 2.0))}
    rec = {"summary": out, "rows": rows, "note": "NMFP offsets added when its segment descriptors exist"}
    nm = DESC / "audio_pos" / "NMFP-triplet.pkl"
    rec["nmfp"] = "descriptors present; not yet wired" if nm.exists() else "skipped: no NMFP descriptors"
    write_json(RESULTS / "localize" / "temporal_audio.json", rec)
    for m, d in out.items():
        a = d.get("_all", {})
        print(f"temporal audio {m:12s} n={a.get('n')} median|err|={a.get('median_abs_err_s', float('nan')):.2f}s "
              f"within0.5s={a.get('within_0.5s', float('nan')):.2f}")
    return rec


# ================================================================== temporal: video
VFPS = 4.0


def _diag_offset(sim: np.ndarray) -> tuple[int, float]:
    """Offset d (reference frame = query frame + d) maximising the mean diagonal similarity."""
    nq, nr = sim.shape
    best, bd = -np.inf, 0
    for d in range(-nq + 1, nr):
        i = np.arange(nq)
        j = i + d
        ok = (j >= 0) & (j < nr)
        if ok.sum() < max(2, nq // 3):
            continue
        s = sim[i[ok], j[ok]].mean()
        if s > best:
            best, bd = s, d
    return bd, float(best)


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    best, cur, bs, s = (0, 0), 0, 0, 0
    for i, v in enumerate(mask):
        if v:
            if cur == 0:
                s = i
            cur += 1
            if cur > best[1] - best[0]:
                best = (s, i + 1)
        else:
            cur = 0
    return best


def temporal_video(device: str) -> dict:
    from fingerprints_image import DINOv2, SSCD
    from fingerprints_video import sample_frames

    man = json.loads((CORPUS / "corpus_manifest.json").read_text())["video"]
    backbones = {"DINOv2-S": DINOv2("S"), "SSCD-mixup": SSCD("mixup")}
    for b in backbones.values():
        b.load(device)
    emb = lambda b, p: b.extract([f.astype(np.float32) / 255.0 for f in sample_frames(p, VFPS)])
    rows, null = [], {k: [] for k in backbones}
    # null for the insert threshold: per-frame best similarity of never-registered clips
    for src in man["sets"]["neg"][:10]:
        qp = OUT / "video" / "neg" / f"{src}__none.mp4"
        refs = man["sets"]["reg"][:5]
        if not qp.exists():
            continue
        for k, b in backbones.items():
            qe = emb(b, qp)
            for r in refs:
                null[k].append((qe @ emb(b, CORPUS / man["items"][r]["path"]).T).max(1))
    tau = {k: float(np.quantile(np.concatenate(v), 0.99)) if v else 0.8 for k, v in null.items()}
    for src in man["sets"]["reg"]:
        ref_path = CORPUS / man["items"][src]["path"]
        ref_e = {k: emb(b, ref_path) for k, b in backbones.items()}
        for atk in ("excerpt2.5", "insert"):
            qp = OUT / "video" / "pos" / f"{src}__{atk}.mp4"
            if not qp.exists():
                continue
            side = json.loads(qp.with_suffix(".json").read_text())
            rec = {"src": src, "attack": atk, **{k: v for k, v in side.items() if k != "duration"}}
            for k, b in backbones.items():
                sim = emb(b, qp) @ ref_e[k].T
                if atk == "excerpt2.5":
                    d, _ = _diag_offset(sim)
                    rec[f"{k}_start_s"] = d / VFPS
                    rec[f"{k}_err_s"] = abs(d / VFPS - side["excerpt_start"])
                else:
                    best = sim.max(1)
                    a, e = _longest_run(best >= tau[k])
                    t0, t1 = a / VFPS, e / VFPS
                    inter = max(0.0, min(t1, 3.75) - max(t0, 1.25))
                    rec[f"{k}_interval"] = [t0, t1]
                    rec[f"{k}_iou"] = inter / max(1e-9, (t1 - t0) + 2.5 - inter)
                    sub = sim[int(1.25 * VFPS):int(3.75 * VFPS)]
                    d, _ = _diag_offset(sub)
                    rec[f"{k}_err_s"] = abs(d / VFPS - side["insert_src_start"])
            rows.append(rec)
    summ = {}
    for k in backbones:
        for atk in ("excerpt2.5", "insert"):
            sel = [r for r in rows if r["attack"] == atk]
            if not sel:
                continue
            err = np.array([r[f"{k}_err_s"] for r in sel])
            s = {"n": len(sel), "median_abs_err_s": float(np.median(err)), "within_1s": float(np.mean(err <= 1.0))}
            if atk == "insert":
                s["interval_iou_mean"] = float(np.mean([r[f"{k}_iou"] for r in sel]))
            summ.setdefault(k, {})[atk] = s
    try:
        import vpdq  # noqa: F401

        vp = "available (not used for alignment: 1 hash per second)"
    except Exception as exc:  # the macOS wheel cannot load against Homebrew FFmpeg 9
        vp = f"skipped: {type(exc).__name__}"
    rec = {"fps": VFPS, "insert_tau": tau, "summary": summ, "rows": rows, "vpdq": vp}
    write_json(RESULTS / "localize" / "temporal_video.json", rec)
    for k, d in summ.items():
        print(f"temporal video {k:10s} " + " ".join(f"{a}: within1s={v['within_1s']:.2f}" +
              (f" IoU={v['interval_iou_mean']:.2f}" if "interval_iou_mean" in v else "") for a, v in d.items()))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["retrieval", "localize", "temporal", "all"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--aligners", default="sift,lightglue")
    ap.add_argument("--cal-sources", type=int, default=100)
    args = ap.parse_args()
    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.stage in ("retrieval", "all"):
        retrieval(args.direct, dev)
    if args.stage in ("localize", "all"):
        localize(dev, tuple(args.aligners.split(",")), args.limit, args.cal_sources)
    if args.stage in ("temporal", "all"):
        if (OUT / "audio" / "pos").exists():
            temporal_audio()
        if (OUT / "video" / "pos").exists():
            temporal_video(dev)


if __name__ == "__main__":
    main()
