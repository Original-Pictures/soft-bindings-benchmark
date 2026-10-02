"""Retrieval metrics shared by the image, audio and video tracks.

Inputs are the top-K search results of positive queries (their registered source is in
the index) and negative queries (never registered). Conventions:

* Thresholds are fixed on the *calibration* negatives only (half of the negative
  sources) and evaluated on the other half, so reported FPRs are out-of-sample.
* query-level FPR: fraction of negative queries whose best score passes the threshold,
  at the benchmark's registry size n_ref.
* pair-level FPR: fraction of (negative query, reference) pairs above threshold. It is
  registry-size free: at registry size N the query-level FPR is about N x pair-FPR.
  Pairs are observed through each query's top-K; `topk_saturated` flags a threshold
  whose count may be truncated by K.
* A positive query counts at a threshold only if its top-1 is the true reference AND the
  score passes (identification, not mere similarity).
* muAP follows the ISC2021 protocol over the pooled (query, reference, score) list.
"""

from __future__ import annotations

import numpy as np

from analysis_stats import cp_interval

QUERY_TARGETS = (1e-2, 1e-3)
PAIR_TARGETS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
MIN_PAIRS = 5


def micro_ap(scores: np.ndarray, correct: np.ndarray, n_pos: int) -> float:
    if n_pos == 0:
        return float("nan")
    ok = np.isfinite(scores)
    scores, correct = scores[ok], correct[ok]
    o = np.argsort(-scores, kind="stable")
    c = correct[o].astype(np.float64)
    prec = np.cumsum(c) / np.arange(1, len(c) + 1)
    return float((prec * c).sum() / n_pos)


def thresholds(neg_S: np.ndarray, n_ref: int) -> tuple[dict[str, float], dict[str, bool]]:
    top1 = np.sort(neg_S[:, 0])[::-1]
    pairs = np.sort(neg_S[np.isfinite(neg_S)])[::-1]
    nq, npairs = len(neg_S), len(neg_S) * n_ref
    th, sat = {}, {}
    for t in QUERY_TARGETS:
        k = int(np.floor(t * nq))
        th[f"query@{t:g}"] = float(top1[k]) + 1e-6 if MIN_PAIRS <= k < len(top1) else float("nan")
    for t in PAIR_TARGETS:
        k = int(np.floor(t * npairs))
        # A target implying fewer than MIN_PAIRS false pairs on the calibration set is not
        # resolved by it: the threshold would just sit above every negative observed.
        if k < MIN_PAIRS:
            th[f"pair@{t:g}"] = float("nan")
            continue
        if k < len(pairs):
            th[f"pair@{t:g}"] = float(pairs[k]) + 1e-6
            sat[f"pair@{t:g}"] = bool((neg_S[:, -1] >= th[f"pair@{t:g}"]).any())
        else:
            th[f"pair@{t:g}"] = float("nan")
    for k, v in list(th.items()):
        if not np.isfinite(v):
            th[k] = float("nan")
    return th, sat


def summarize(pos_S, pos_I, pos_tidx, pos_true, pos_atk, neg_S, neg_atk, neg_cal, n_ref, attack_order) -> dict:
    th, sat = thresholds(neg_S[neg_cal], n_ref)
    test = ~neg_cal
    per = {}
    for atk in list(attack_order) + ["_all"]:
        pm = np.ones(len(pos_atk), bool) if atk == "_all" else (np.asarray(pos_atk) == atk)
        nm = test & (np.ones(len(neg_atk), bool) if atk == "_all" else (np.asarray(neg_atk) == atk))
        if not pm.any():
            continue
        top1 = pos_I[pm, 0] == pos_tidx[pm]
        npos = int(pm.sum())
        r = {"n_pos": npos, "n_neg": int(nm.sum()), "R@1": float(top1.mean()), "R@1_ci": cp_interval(int(top1.sum()), npos),
             "R@10": float((pos_I[pm, :10] == pos_tidx[pm, None]).any(1).mean()),
             "true_score_median": float(np.nanmedian(pos_true[pm])) if np.isfinite(pos_true[pm]).any() else float("nan")}
        S = np.concatenate([pos_S[pm, :10], neg_S[nm, :10]]).ravel()
        C = np.concatenate([pos_I[pm, :10] == pos_tidx[pm, None], np.zeros(neg_S[nm, :10].shape, bool)]).ravel()
        r["muAP"] = micro_ap(S, C, npos)
        for name, t in th.items():
            if not np.isfinite(t):
                continue
            tp = int((top1 & (pos_S[pm, 0] >= t)).sum())
            fp = int((neg_S[nm, 0] >= t).sum())
            r[f"TPR@{name}"] = tp / npos
            r[f"TPR@{name}_ci"] = cp_interval(tp, npos)
            r[f"FPRq@{name}"] = fp / max(1, int(nm.sum()))
            r[f"FPRq@{name}_ci"] = cp_interval(fp, max(1, int(nm.sum())))
        per[atk] = r
    # score distributions for figures (untouched positives vs pooled test negatives)
    dist = {"pos_true": _hist(pos_true), "neg_top1": _hist(neg_S[test, 0])}
    return {"thresholds": th, "topk_saturated": sat, "n_cal_neg": int(neg_cal.sum()), "n_test_neg": int(test.sum()),
            "n_ref": n_ref, "per_attack": per, "score_hist": dist}


def _hist(v: np.ndarray, bins: int = 60) -> dict:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    h, e = np.histogram(v, bins=bins)
    return {"counts": h.tolist(), "edges": e.tolist(), "q": np.quantile(v, [0.001, 0.01, 0.5, 0.99, 0.999]).tolist()}


def dump_perquery(path, pos_ids, pos_S, pos_I, pos_tidx, neg_ids, neg_S, neg_cal) -> None:
    """Write what a source-level bootstrap needs, compactly: each query's source and attack,
    whether its top-1 is the true asset with that score, and every negative query's top-K
    scores (pair-level thresholds are recomputed from them in each replicate).

    Sources, not queries, are the independent units: every source contributes one query per
    transformation, so resampling queries would understate the uncertainty."""
    import pathlib

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pos_src=np.array([k.split("|")[0] for k in pos_ids]),
        pos_atk=np.array([k.split("|")[1] for k in pos_ids]),
        pos_hit=np.asarray(pos_I)[:, 0] == np.asarray(pos_tidx),
        pos_top1=np.asarray(pos_S)[:, 0].astype(np.float32),
        neg_src=np.array([k.split("|")[0] for k in neg_ids]),
        neg_atk=np.array([k.split("|")[1] for k in neg_ids]),
        neg_S=np.asarray(neg_S).astype(np.float32),
        neg_cal=np.asarray(neg_cal, bool),
    )
