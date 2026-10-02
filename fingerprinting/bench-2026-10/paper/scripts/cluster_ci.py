"""Source-level bootstrap for the retrieval operating points and paired comparisons.

Every source contributes one query per transformation, so queries are not independent:
the 36 transformed copies of one ABO product succeed or fail together far more often than
36 unrelated queries would. This script resamples *sources*, never single queries:

* positives: registered sources are drawn with replacement, carrying all their queries;
* calibration negatives: negative sources of the calibration half are drawn with
  replacement, and the threshold is recomputed from the drawn set in every replicate, so
  the interval includes the uncertainty of the threshold itself;
* held-out negatives: test-half sources are drawn with replacement for the out-of-sample
  false-match rate.

Paired comparisons (the planned image pairs) use a sign-flip test on per-source
differences in success at each method's full-sample threshold, Holm-adjusted, plus a
bootstrap interval of the TPR difference under shared resampling.

Inputs: results-gpu/<modality>/_perquery/<method>.npz written by the bench metrics stage.
Output: results-gpu/_cluster_ci.json, read by build_numbers.py. Deterministic (fixed seed).

    uv run --with numpy python cluster_ci.py [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import analysis as A

SEED = 20260928
MIN_PAIRS = 5  # same resolvability rule as the bench (scripts/retrieval_metrics.py)
TARGETS = {"image": ["query@0.01", "pair@0.0001", "pair@1e-07"], "audio": ["query@0.01", "pair@1e-06"],
           "video": ["query@0.01", "pair@0.0001"]}
PAIRED = [("SSCD-mixup", "DINOv2-S"), ("DINOv2-S", "DINOv2-S-LSH256"), ("PDQ", "ISCC-Image-64"),
          ("PDQ", "PDQ-dihedral"), ("ISCC-Image-64", "ISCC-Image-256"), ("SSCD-mixup", "ISC21-1st"),
          ("DINOv2-S", "OpenCLIP-B32"), ("pHash-64", "PDQ")]


class PerQuery:
    """Per-query arrays of one method, indexed by source for fast weighted resampling."""

    def __init__(self, path: Path, n_ref: int):
        z = np.load(path)
        self.n_ref = n_ref
        self.pos_src, self.pos_hit, self.pos_top1 = z["pos_src"], z["pos_hit"], z["pos_top1"]
        self.pos_atk = z["pos_atk"]
        cal = z["neg_cal"]
        self.p_src_names, self.p_gi = np.unique(self.pos_src, return_inverse=True)
        # calibration negatives: query-level (top-1) and pair-level (all top-K scores) pools
        cs = z["neg_src"][cal]
        self.c_src_names, c_gi = np.unique(cs, return_inverse=True)
        S = z["neg_S"][cal]
        top1 = S[:, 0]
        o = np.argsort(-top1, kind="stable")
        self.q_scores, self.q_gi = top1[o], c_gi[o]
        fin = np.isfinite(S)
        ps, pg = S[fin], np.broadcast_to(c_gi[:, None], S.shape)[fin]
        o = np.argsort(-ps, kind="stable")
        self.p_scores, self.p_gi_pairs = ps[o], pg[o]
        self.c_queries = np.bincount(c_gi, minlength=len(self.c_src_names)).astype(float)
        # held-out negatives
        ts = z["neg_src"][~cal]
        self.t_src_names, self.t_gi = np.unique(ts, return_inverse=True)
        self.t_top1 = z["neg_S"][~cal][:, 0]

    def threshold(self, target: str, w_cal: np.ndarray) -> float:
        """Threshold from calibration negatives weighted by source draw counts (the bench's
        rule: the score of the (k+1)-th highest false pair or query, k = floor(t * N))."""
        kind, t = target.split("@")
        t = float(t)
        if kind == "query":
            w = w_cal[self.q_gi]
            n = float(w.sum())
            scores = self.q_scores
        else:
            w = w_cal[self.p_gi_pairs]
            n = float((w_cal * self.c_queries).sum()) * self.n_ref
            scores = self.p_scores
        k = int(np.floor(t * n))
        if k < MIN_PAIRS:
            return float("nan")
        cw = np.cumsum(w)
        i = int(np.searchsorted(cw, k, side="right"))  # first position with more than k weight above
        return float(scores[i]) + 1e-6 if i < len(scores) else float("nan")

    def success(self, th: float) -> np.ndarray:
        return self.pos_hit & (self.pos_top1 >= th)


def _weights(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def boot_method(pq: PerQuery, targets: list[str], B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_p, n_c, n_t = len(pq.p_src_names), len(pq.c_src_names), len(pq.t_src_names)
    ones_c = np.ones(n_c)
    full = {}
    for tg in targets:
        th = pq.threshold(tg, ones_c)
        full[tg] = {"threshold": th, "resolved": bool(np.isfinite(th))}
        if np.isfinite(th):
            full[tg]["tpr"] = float(pq.success(th).mean())
            full[tg]["fpr_q_test"] = float((pq.t_top1 >= th).mean())
    draws = {tg: {"tpr": [], "fpr": []} for tg in targets}
    pos_per_src = np.bincount(pq.p_gi, minlength=n_p).astype(float)
    t_per_src = np.bincount(pq.t_gi, minlength=n_t).astype(float)
    unresolved = {tg: 0 for tg in targets}
    hits = np.bincount(pq.p_gi, weights=pq.pos_hit, minlength=n_p)
    r1 = []
    for _ in range(B):
        wp, wc, wt = _weights(rng, n_p), _weights(rng, n_c), _weights(rng, n_t)
        r1.append(float((wp * hits).sum() / (wp * pos_per_src).sum()))
        for tg in targets:
            th = pq.threshold(tg, wc)
            if not np.isfinite(th):
                unresolved[tg] += 1
                continue
            succ = np.bincount(pq.p_gi, weights=pq.success(th), minlength=n_p)
            draws[tg]["tpr"].append(float((wp * succ).sum() / (wp * pos_per_src).sum()))
            fp = np.bincount(pq.t_gi, weights=(pq.t_top1 >= th), minlength=n_t)
            draws[tg]["fpr"].append(float((wt * fp).sum() / (wt * t_per_src).sum()))
    out = {"n_pos_sources": n_p, "n_cal_sources": n_c, "n_test_sources": n_t,
           "n_pos": int(len(pq.pos_hit)), "n_test": int(len(pq.t_top1)), "boot": B,
           "r1": float(pq.pos_hit.mean()), "r1_ci": [float(np.percentile(r1, 2.5)), float(np.percentile(r1, 97.5))]}
    for tg in targets:
        r = dict(full[tg])
        r["unresolved_share"] = unresolved[tg] / B
        for key in ("tpr", "fpr"):
            v = np.array(draws[tg][key])
            if len(v) and r["resolved"]:
                r[f"{key}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[tg] = r
    return out


def sign_flip(d: np.ndarray, rng: np.random.Generator, n: int = 20_000) -> float:
    """Two-sided sign-flip test of mean per-source difference = 0 (exchangeable signs
    under H0). Monte Carlo with add-one correction."""
    obs = abs(d.sum())
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    stat = np.abs(flips @ d)
    return float((1 + (stat >= obs - 1e-12).sum()) / (n + 1))


def paired(pqs: dict[str, PerQuery], target: str, B: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in PAIRED:
        if a not in pqs or b not in pqs:
            continue
        A_, B_ = pqs[a], pqs[b]
        assert (A_.pos_src == B_.pos_src).all(), "paired methods must share query order"
        ones = np.ones(len(A_.c_src_names))
        ta, tb = A_.threshold(target, ones), B_.threshold(target, np.ones(len(B_.c_src_names)))
        sa, sb = A_.success(ta).astype(float), B_.success(tb).astype(float)
        n_p = len(A_.p_src_names)
        d = np.bincount(A_.p_gi, weights=sa - sb, minlength=n_p)
        p = sign_flip(d, rng)
        # interval of the TPR difference: shared resampling of positive and calibration sources
        per = np.bincount(A_.p_gi, minlength=n_p).astype(float)
        diffs = []
        for _ in range(B):
            wp = _weights(rng, n_p)
            wa, wb = _weights(rng, len(A_.c_src_names)), None
            wb = wa if len(B_.c_src_names) == len(A_.c_src_names) else _weights(rng, len(B_.c_src_names))
            ta_, tb_ = A_.threshold(target, wa), B_.threshold(target, wb)
            if not (np.isfinite(ta_) and np.isfinite(tb_)):
                continue
            da = np.bincount(A_.p_gi, weights=A_.success(ta_), minlength=n_p)
            db = np.bincount(A_.p_gi, weights=B_.success(tb_), minlength=n_p)
            diffs.append(float((wp * (da - db)).sum() / (wp * per).sum()))
        rows.append({"a": a, "b": b, "tpr_a": float(sa.mean()), "tpr_b": float(sb.mean()),
                     "only_a": int(((sa == 1) & (sb == 0)).sum()), "only_b": int(((sa == 0) & (sb == 1)).sum()),
                     "diff": float(sa.mean() - sb.mean()),
                     "diff_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None,
                     "p_signflip": p, "n_sources": n_p})
    # Holm step-down over the planned family
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_signflip"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[i]["p_signflip"]))
        rows[i]["p_holm"] = running
    return rows


def verification(pq: PerQuery, vz: dict, B: int, seed: int) -> dict:
    """Two-stage decision: retrieval score >= t_ret AND geometric inliers >= k.

    t_ret is a looser retrieval threshold (pair-level 1e-4 or query-level 1%, both fixed on the
    calibration negatives as usual). k is fixed on the same calibration negatives, three ways:
    k_match, the smallest k whose calibration false bindings do not exceed those of the one-stage
    headline threshold (detection compared at an equal calibrated rate); k_zero, the smallest k
    no calibration negative reaches; and the localization track's fixed k = 12. The held-out
    negatives then give the out-of-sample false-binding rate. Intervals: bootstrap over
    sources with both thresholds held at their full-sample values."""
    z = np.load(A.RAW / "image" / "_perquery" / f"{vz['name']}.npz")
    pos_inl, neg_inl = vz["pos_inliers"], vz["neg_inliers"]
    cal = z["neg_cal"]
    top1 = z["neg_S"][:, 0]
    rng = np.random.default_rng(seed)
    out = {}
    one_stage = pq.threshold(A.OP, np.ones(len(pq.c_src_names)))
    base = {"tpr": float(pq.success(one_stage).mean()), "fp_test": int((pq.t_top1 >= one_stage).sum()),
            "fp_cal": int((cal & (top1 >= one_stage)).sum()), "n_test": int((~cal).sum()), "n_cal": int(cal.sum())}
    for ret in ("pair@0.0001", "query@0.01"):
        t = pq.threshold(ret, np.ones(len(pq.c_src_names)))
        if not np.isfinite(t):
            continue
        cal_pass = cal & (top1 >= t)
        k_zero = int(neg_inl[cal_pass].max()) + 1 if cal_pass.any() else 0
        # matched: the smallest k whose calibration false bindings do not exceed those of the
        # one-stage headline threshold, so detection is compared at an equal calibrated rate
        k_match = next(k for k in range(0, k_zero + 1) if int((cal_pass & (neg_inl >= k)).sum()) <= base["fp_cal"])
        for kname, k in (("k_match", k_match), ("k_zero", k_zero), ("k12", 12)):
            succ = pq.success(t) & (pos_inl >= k)
            fp = (~cal) & (top1 >= t) & (neg_inl >= k)
            n_p = len(pq.p_src_names)
            per = np.bincount(pq.p_gi, minlength=n_p).astype(float)
            sp = np.bincount(pq.p_gi, weights=succ, minlength=n_p)
            draws = []
            for _ in range(B):
                w = _weights(rng, n_p)
                draws.append(float((w * sp).sum() / (w * per).sum()))
            n_test = int((~cal).sum())
            out[f"{ret}/{kname}"] = {
                "t_ret": t, "k": k, "tpr": float(succ.mean()),
                "tpr_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "retrieval_only_tpr": float(pq.success(t).mean()),
                "retrieval_only_fp_test": int(((~cal) & (top1 >= t)).sum()),
                "fp_test": int(fp.sum()), "n_test": n_test,
                "fp_test_upper": float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.975, fp.sum() + 1, n_test - fp.sum()))
                if fp.sum() < n_test else 1.0,
                "cal_fp": int((cal_pass & (neg_inl >= k)).sum()), "n_cal": int(cal.sum()),
            }
    out["one_stage_" + A.OP] = base
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    out: dict = {"seed": SEED, "boot": args.boot, "unit": "source", "methods": {}}
    for mod in ("image", "audio", "video"):
        d = A.RAW / mod / "_perquery"
        if not d.exists():
            print(f"{mod}: no per-query arrays", file=sys.stderr)
            continue
        recs = {m.name: m for m in A.load()[mod]}
        pqs = {}
        for i, f in enumerate(sorted(d.glob("*.npz"))):
            n_ref = recs[f.stem].rec["n_ref"] if f.stem in recs else None
            if n_ref is None:
                continue
            pqs[f.stem] = PerQuery(f, n_ref)
            out["methods"][f"{mod}/{f.stem}"] = boot_method(pqs[f.stem], TARGETS[mod], args.boot, SEED + i)
            print(f"{mod} {f.stem}: done", flush=True)
        if mod == "image":
            out["paired"] = paired(pqs, A.OP, args.boot, SEED)
            vdir = A.RAW / "image" / "_verify"
            out["verify"] = {}
            for f in sorted(vdir.glob("*.npz")) if vdir.exists() else []:
                if f.stem in pqs:
                    vz = dict(np.load(f))
                    vz["name"] = f.stem
                    out["verify"][f.stem] = verification(pqs[f.stem], vz, args.boot, SEED + 99)
                    print(f"verify {f.stem}: done", flush=True)
    (A.RAW / "_cluster_ci.json").write_text(json.dumps(out, indent=1))
    print("wrote", A.RAW / "_cluster_ci.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
