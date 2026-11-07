"""Rerun the image evasion attacks of the paper's security track that involve the ISCC Image-Code.

The paper's bench_security.py runs every image method in one process. This wrapper runs its ISCC parts with the
same items, seeds, budgets and steps, built from bench_security's own functions:

  craft      transfer perturbations crafted on the paper's ensemble (DINOv2-S, SSCD-mixup, OpenCLIP-B32), as
             bench_security.evasion makes them. They do not depend on ISCC, so they are made once and saved
             losslessly with their PSNR and LPIPS, for both ISCC versions to score.
  transfer   scores the saved perturbations with the ISCC methods of this environment and, as reference points,
             with PDQ and the ensemble methods at the paper's thresholds.
  surrogate  the ISCC-Image-64 surrogate attack of bench_security.evasion (gradients from hash_surrogates).

Both scoring stages write their part of $BENCH_RESULTS/security/evasion.json in the layout of the paper's record.
The collision attack runs unchanged: bench_security.py collision.

    python security.py craft [--n 100] [--shard 0/8]
    python security.py transfer [--n 100]
    python security.py surrogate [--n 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import bench  # noqa: F401  registers the Semantic-Code adapter (sets up sys.path for scripts/)
import bench_security as bs  # noqa: E402
from common import BENCH_DIR, CORPUS, RESULTS, host_info, load_rgb, to_u8, write_json  # noqa: E402

DEVICE = "cpu"
STEPS = 40  # bench_security's default
ISCC = ["ISCC-Image-64", "ISCC-Image-256", "ISCC-SCI-256"]
ANCHORS = ["PDQ", *bs.ENSEMBLE]  # reference points, scored at the paper's thresholds
TRANSFER = bs.SEC_OUT / "transfer"
RECORD = RESULTS / "security" / "evasion.json"


def sources(n: int) -> list[str]:
    """The registered originals the paper attacks: the first n of the positive set."""
    return bs.manifest()["image"]["sets"]["pos"][:n]


def original(i: str) -> np.ndarray:
    """One registered original at the paper's attack resolution."""
    return load_rgb(CORPUS / bs.manifest()["image"]["items"][i]["path"], 1024)


def adversarial(eps: int, i: str) -> Path:
    """Path of the saved transfer perturbation of item i at budget eps."""
    return TRANSFER / str(eps) / f"{i}.png"


def craft(n: int, shard: int, shards: int) -> None:
    """Make the missing transfer perturbations of this shard of items."""
    todo = [(eps, i) for eps in bs.EPS_EVASION for i in sources(n)[shard::shards] if not adversarial(eps, i).exists()]
    if not todo:
        return
    views = [bs.TorchView(m, DEVICE) for m in bs.ENSEMBLE]
    lp = bs.Lpips(DEVICE)
    t0 = time.time()
    for k, (eps, i) in enumerate(todo, 1):
        x = original(i)
        x0 = bs.to_t(x, DEVICE)
        with torch.no_grad():
            refs = [v(x0)[0] for v in views]
        loss = lambda xx: sum((v(xx) * r).sum() for v, r in zip(views, refs)) / len(views)  # noqa: E731, B023
        xa, _ = bs.pgd(x0, loss, eps / 255, STEPS, f"tr:{eps}:{i}")
        dst = adversarial(eps, i)
        write_json(dst.with_suffix(".json"), {"psnr": bs.psnr(xa, x), "lpips": lp(xa, x)})
        tmp = dst.with_suffix(".tmp.png")
        Image.fromarray(to_u8(xa)).save(tmp)
        tmp.replace(dst)
        print(f"craft {shard}/{shards}: {k}/{len(todo)} eps={eps} {i} {time.time() - t0:.0f}s", flush=True)


def thresholds() -> dict[str, tuple[float, str]]:
    """pair@1e-6 thresholds (bench_security's rule): this run's ISCC records, the paper's for the anchors."""
    ths = {m: bs.threshold("image", m) for m in ISCC if (RESULTS / "image" / f"{m}.json").exists()}
    for m in ANCHORS:
        rec = json.loads((BENCH_DIR / "results-gpu" / "image" / f"{m}.json").read_text())
        ths[m] = (rec["thresholds"]["pair@1e-06"], "pair@1e-06 (paper)")
    return ths


def transfer(n: int) -> dict:
    """Score every saved transfer perturbation, as bench_security.evasion scores its transfer attack."""
    real, ths = bs.Real(DEVICE), thresholds()
    refs, clean, out = {}, {}, {}
    for eps in bs.EPS_EVASION:
        per: dict[str, list] = {m: [] for m in ths}
        for i in sources(n):
            x, xa = original(i), load_rgb(adversarial(eps, i))
            q = json.loads(adversarial(eps, i).with_suffix(".json").read_text())
            for name, (t, _) in ths.items():
                if (name, i) not in refs:
                    refs[(name, i)] = real(name).extract([x])[0]
                    clean[(name, i)] = real.score(name, x, refs[(name, i)])
                s0, s1 = clean[(name, i)], real.score(name, xa, refs[(name, i)])
                per[name].append({"item": i, "success": s1 < t, "unmatched_clean": s0 < t, "score_clean": s0,
                                  "score_adv": s1, **q})
        out[str(eps)] = {m: {"summary": bs.summarise(rows), "errors": 0, "rows": rows} for m, rows in per.items()}
        print(f"transfer eps={eps}: " + ", ".join(f"{m}={v['summary']['rate']:.2f}" for m, v in out[str(eps)].items()),
              flush=True)
    return {"transfer": out, "thresholds": ths}


def surrogate(n: int) -> dict:
    """The ISCC-Image-64 surrogate attack, as in bench_security.evasion."""
    name = "ISCC-Image-64"
    real, lp = bs.Real(DEVICE), bs.Lpips(DEVICE)
    t, key = bs.threshold("image", name)
    res = {}
    for eps in bs.EPS_EVASION:
        rows = []
        for i in sources(n):
            x = original(i)
            ref = real(name).extract([x])[0]
            done = lambda xa: real.score(name, xa, ref) < t  # noqa: E731, B023
            xa, used = bs.pgd(bs.to_t(x, DEVICE), bs.hash_loss(name, ref, +1, DEVICE), eps / 255, STEPS,
                              f"ev:{name}:{eps}:{i}", done)
            s_adv, s_clean = real.score(name, xa, ref), real.score(name, x, ref)
            rows.append({"item": i, "success": s_adv < t, "unmatched_clean": s_clean < t, "score_clean": s_clean,
                         "score_adv": s_adv, "steps": used, "psnr": bs.psnr(xa, x), "lpips": lp(xa, x)})
        res[str(eps)] = {"summary": bs.summarise(rows), "rows": rows}
        print(f"surrogate {name} eps={eps}: {res[str(eps)]['summary']['rate']:.2f}", flush=True)
    return {"surrogate": {name: res}, "thresholds": {name: (t, key)}}


def update_record(part: dict, n: int) -> None:
    """Merge one stage's results into the evasion record, in the layout of the paper's."""
    rec = json.loads(RECORD.read_text()) if RECORD.exists() else {"whitebox": {}, "surrogate": {}, "transfer": {}}
    rec["thresholds"] = {**rec.get("thresholds", {}), **part.pop("thresholds")}
    rec.update(part)
    rec.update({"params": {"n": n, "eps_255": bs.EPS_EVASION, "steps": STEPS, "step": "eps/8"},
                "experiment": "evasion", "device": DEVICE, "host": host_info()})
    write_json(RECORD, rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["craft", "transfer", "surrogate"])
    ap.add_argument("--n", type=int, default=bs.DEFAULT_N["evasion"])
    ap.add_argument("--shard", default="0/1", help="k/K: craft every K-th item, starting at item k")
    args = ap.parse_args()
    if args.stage == "craft":
        k, shards = map(int, args.shard.split("/"))
        craft(args.n, k, shards)
    else:
        update_record({"transfer": transfer, "surrogate": surrogate}[args.stage](args.n), args.n)


if __name__ == "__main__":
    sys.exit(main())
