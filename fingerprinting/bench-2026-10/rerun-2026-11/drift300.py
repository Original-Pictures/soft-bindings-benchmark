"""v1.1 reproduction of the watermark-drift result on the 300 product photos of the watermark-combination track.

The paper (Table wm, `wm/image/ISCC-Image-64/drift/pct`) reports that embedding TrustMark Q at strength 1 moved the
ISCC Image-Code of 62.0 % of these photos past the pair-level 1e-7 match threshold against their unmarked original;
the ISCC Foundation's rerun with iscc-sdk 0.9.7 / iscc-core 1.5.0 / iscc-lib 0.7.0 reports 3.7 % (256-bit: 56.7 ->
1.3 %). This script recomputes that number on this machine, independently of both earlier runs:

  1. stage   the first 300 sources of the image track's `pos` set (full manifest, results-gpu/locks), from the local
             corpus copy or the ABO public bucket, each checked against the manifest's SHA-256;
  2. render  each photo marked with the harness's TrustMarkAdapter (Q, strength 1, BCH_5, payload as
             bench_wmcombo.render_image), once, on CPU;
  3. drift   the harness's ISCCImage adapter on original and marked photo (loaded as bench_image.make_image loads
             them), score as the harness scores, and the share below the run's pair@1e-07 threshold.

Thresholds come from the image-track record of the matching ISCC version (the paper's results-gpu/ for 0.9.5, the
fork's results-iscc/iscc-sdk-0.9.7/ for 0.9.7): the slice cannot calibrate a 1e-7 threshold itself, so this checks
the drift given those calibrations, not the calibrations. Run stage/render with either venv, drift with each:

    python drift300.py stage|render
    python drift300.py drift            # writes results-drift/drift-iscc-sdk-<installed version>.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
WORK = Path(os.environ.get("V11_WORK", Path.home() / "op-v11-work")) / "wm300"
os.environ.setdefault("BENCH_WORK", str(WORK))
LOCAL = Path(os.environ.get("FP_WORK", Path.home() / "op-fp-bench-work")) / "corpus"
sys.path.insert(0, str(BENCH / "scripts"))

import numpy as np  # noqa: E402

N = 300
OP = "pair@1e-07"


def manifest() -> dict:
    return json.loads(gzip.decompress((BENCH / "results-gpu/locks/corpus_manifest.json.gz").read_bytes()))["image"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sources() -> list[tuple[str, dict]]:
    man = manifest()
    return [(s, man["items"][s]) for s in man["sets"]["pos"][:N]]


def stage() -> None:
    from fetch_corpus import _get, _save_jpeg

    fetched = linked = 0
    for s, it in sources():
        dst = WORK / "corpus" / it["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            src = LOCAL / it["path"]
            if src.exists() and sha256(src) == it["sha256"]:
                os.symlink(src, dst)
                linked += 1
            else:
                _save_jpeg(_get(it["url"]), dst, 1024)
                fetched += 1
        if sha256(dst) != it["sha256"]:
            raise SystemExit(f"{s}: sha256 differs from the manifest")
    print(f"staged {N}: {linked} linked from the local corpus, {fetched} fetched, all sha256 match")


def render() -> None:
    from PIL import Image

    from bench_wmcombo import TRUSTMARK
    from common import load_rgb, payload, to_u8
    from models_image import TrustMarkAdapter

    tm = TrustMarkAdapter(*TRUSTMARK)
    tm.load("cpu")
    out = WORK / "marked"
    out.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    for i, (s, it) in enumerate(sources()):
        dst = out / f"{s}.png"
        if dst.exists():
            continue
        x = load_rgb(WORK / "corpus" / it["path"], 1024)
        Image.fromarray(to_u8(tm.embed(x, payload(tm.nbits, i, salt="wm-image")))).save(dst)
        n += 1
    print(f"rendered {n} marked photos in {time.time() - t0:.0f}s")


def drift() -> None:
    import iscc_core
    import iscc_sdk

    from bench_image import MAX_SIDE
    from common import load_rgb
    from fingerprints_image import IMAGE_METHODS

    ver = iscc_sdk.__version__
    records = {"0.9.5": BENCH / "results-gpu/image", "0.9.7": BENCH / "results-iscc/iscc-sdk-0.9.7/image"}
    src = sources()
    res = {"iscc_sdk": ver, "iscc_core": iscc_core.__version__, "n": len(src), "threshold_name": OP, "methods": {}}
    for name in ("ISCC-Image-64", "ISCC-Image-256"):
        p = records[ver] / f"{name}.json"
        rec = json.loads(p.read_text() if p.exists() else gzip.decompress(p.with_name(p.name + ".gz").read_bytes()))
        th = rec["thresholds"][OP]
        m = IMAGE_METHODS[name]()
        m.load("cpu")
        t0 = time.perf_counter()
        orig = m.extract([load_rgb(WORK / "corpus" / it["path"], MAX_SIDE) for _, it in src])
        marked = m.extract([load_rgb(WORK / "marked" / f"{s}.png", MAX_SIDE) for s, _ in src])
        dt = time.perf_counter() - t0
        score = np.array([m.score(marked[i:i + 1], orig[i:i + 1])[0, 0] for i in range(len(src))])
        res["methods"][name] = {"threshold": th, "threshold_source": str(p.relative_to(BENCH)),
                                "drift_below_threshold": float((score < th).mean()),
                                "n_below": int((score < th).sum()),
                                "drift_score_median": float(np.median(score)),
                                "extract_ms_per_image": 1000 * dt / (2 * len(src))}
        print(f"{name} iscc-sdk {ver}: {100 * (score < th).mean():.1f} % below {OP} ({th:.4f}); "
              f"median score {np.median(score):.4f}; {1000 * dt / (2 * len(src)):.1f} ms/image")
    out = HERE / "results-drift" / f"drift-iscc-sdk-{ver}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"stage": stage, "render": render, "drift": drift}[sys.argv[1]]()
