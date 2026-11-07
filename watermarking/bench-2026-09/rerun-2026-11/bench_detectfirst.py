"""v1.1: TrustMark decoder-only versus detect-first on the watermark track's image corpus, paired.

John Collomosse (Adobe) pointed out that v1.0 decoded TrustMark with the decoder alone, never through the
watermark-region detector that trustmark ships for Q and P (TrustMark(loadBBoxDetector=True),
decode(DETECTFIRST=True)). This runner embeds each image once per config, exactly as bench_image.run_config does
(same items, order, payloads, quantisation and attacks), and decodes every attacked marked copy and every
unmarked false-positive probe with BOTH paths of models_image.TrustMarkAdapter. Quality metrics are not
recomputed: embedding does not depend on the decode path, and they are in the v1.0 records.

    BENCH_WORK=~/op-v11-work/wm python bench_detectfirst.py [--only Q@1,P@1] [--device cpu|mps]

Writes rerun-2026-11/results-detect/<config>.json (resumable: existing files are skipped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import attacks  # noqa: E402
from bench_image import FPR_ATTACKS, load_image, quantize  # noqa: E402
from common import BENCH_DIR, CORPUS, bit_accuracy, host_info, payload, write_json  # noqa: E402
from models_image import TrustMarkAdapter  # noqa: E402

CONFIGS = [("Q", s) for s in (1.0, 1.2, 1.3, 1.4, 1.5)] + [("P", s) for s in (0.8, 1.0, 1.5)]
MODES = {"decoder": False, "detect": True}


def items(div2k: int = 25) -> list[dict]:
    """bench_image.main's item list: every non-DIV2K item, then the first 25 DIV2K items."""
    man = json.loads((BENCH_DIR / "corpus_manifest.json").read_text())["image"]
    return [i for i in man if i["set"] != "div2k"] + [i for i in man if i["set"] == "div2k"][:div2k]


def decode_both(a: TrustMarkAdapter, x: np.ndarray, timing: dict) -> dict:
    out = {}
    for mode, flag in MODES.items():
        a.detect_first = flag
        t0 = time.perf_counter()
        out[mode] = a.decode(x)
        timing[mode].append(time.perf_counter() - t0)
    return out


def run(variant: str, strength: float, device: str) -> dict:
    a = TrustMarkAdapter(variant, strength, detect_first=True)
    a.load(device)
    expect_payload = lambda bits: "".join(str(int(b)) for b in bits[: a.capacity])  # noqa: E731
    timing = {m: [] for m in MODES}
    recs, fpr = [], []
    its = items()
    for idx, it in enumerate(its):
        bits = payload(a.nbits, idx, salt=str(a.nbits))
        ref = load_image(CORPUS / it["path"])
        marked = quantize(a.embed(ref, bits), it.get("bits") == 16)
        expect = a.physical_bits(bits)
        rec = {"item": it["path"], "set": it["set"], "attacks": {}}
        for aname, fn in attacks.IMAGE_ATTACKS.items():
            d = decode_both(a, fn(marked), timing)
            rec["attacks"][aname] = {m: {"bit_acc": bit_accuracy(expect, v.get("bits")), "native_detect": v["native_detect"],
                                         "payload_exact": bool(v["native_detect"]) and v["native_payload"] == expect_payload(bits),
                                         "n_regions": v.get("n_regions")} for m, v in d.items()}
        for aname in FPR_ATTACKS:
            d = decode_both(a, attacks.IMAGE_ATTACKS[aname](ref), timing)
            fpr.append({"item": idx, "set": it["set"], "attack": aname,
                        **{m: {"native_detect": v["native_detect"], "n_regions": v.get("n_regions")} for m, v in d.items()}})
        recs.append(rec)
        j = rec["attacks"]["jpeg75"]
        print(f"  {a.name} [{idx + 1}/{len(its)}] {it['path']} jpeg75 decoder={j['decoder']['payload_exact']} "
              f"detect={j['detect']['payload_exact']}", flush=True)
    return {"config": f"trustmark-{variant}@{strength:g}", "variant": variant, "strength": strength, "device": device,
            "decode_s_median": {m: float(np.median(v)) for m, v in timing.items()},
            "decode_s_mean": {m: float(np.mean(v)) for m, v in timing.items()}, "n_decodes": len(timing["decoder"]),
            "records": recs, "fpr": fpr, "host": host_info()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch default)")
    args = ap.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    out = HERE / "results-detect"
    for v, s in CONFIGS:
        name = f"{v}@{s:g}"
        if args.only and name not in args.only.split(","):
            continue
        dst = out / f"trustmark-{v}_at_{s:g}.json"
        if dst.exists():
            print("skip", name)
            continue
        t0 = time.time()
        res = run(v, s, args.device)
        res["wall_s"] = time.time() - t0
        write_json(dst, res)
        print(f"== {name} done in {res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
