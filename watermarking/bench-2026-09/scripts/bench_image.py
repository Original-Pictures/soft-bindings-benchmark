"""Image bake-off runner: quality, robustness, FPR and latency per model config.

One JSON per config in $BENCH_RESULTS/image/<config>.json (skipped if present, so
the run is resumable). Usage:
  python bench_image.py [--profile full|smoke] [--only substr,...] [--device cuda]
                        [--manifest path] [--limit N] [--cpu-latency]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import attacks
import metrics
from common import BENCH_DIR, CORPUS, RESULTS, Timer, bit_accuracy, device_name, host_info, payload, write_json
from models_image import image_configs

PERCEPTUAL_SETS = {"kodak", "clic", "hdr16"}   # FLIP computed here (DIV2K: PSNR/SSIM/MS-SSIM/LPIPS only)
CVVDP_SETS = {"kodak", "hdr16"}              # CVVDP is the most expensive metric; Kodak + HDR only
GALLERY = {"kodim23.png", "kodim05.png"}       # public-domain-equivalent Kodak only in the PDF
FPR_ATTACKS = ["none", "jpeg75", "resize0.5", "crop70"]
FPR_EXTRA_KEYS = 50


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(path)
        return (a.astype(np.float32) / 65535.0)[..., :3]
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def quantize(x: np.ndarray, bits16: bool) -> np.ndarray:
    q = 65535.0 if bits16 else 255.0
    return (np.clip(np.round(x * q), 0, q) / q).astype(np.float32)


def save_gallery(name: str, fname: str, ref: np.ndarray, marked: np.ndarray, flip_map: np.ndarray | None) -> None:
    out = BENCH_DIR / "thumbs" / "image"
    out.mkdir(parents=True, exist_ok=True)
    h, w = ref.shape[:2]
    s = 256
    t, l = (h - s) // 2, (w - s) // 2
    crop = lambda a: a[t:t + s, l:l + s]  # noqa: E731
    diff = np.clip(0.5 + (marked - ref) * 20, 0, 1)
    tiles = [crop(ref), crop(marked), crop(diff)]
    if flip_map is not None:
        tiles.append(crop(np.asarray(flip_map, dtype=np.float32)))
    strip = np.concatenate(tiles, axis=1)
    safe = name.replace("@", "_at_").replace("/", "_")
    Image.fromarray((strip * 255).round().astype(np.uint8)).save(out / f"{safe}__{Path(fname).stem}.jpg", quality=88)


def run_config(adapter, items: list[dict], device: str, args) -> dict:
    t_load = time.perf_counter()
    adapter.load(device)
    load_s = time.perf_counter() - t_load
    records, fpr_bits = [], []
    embed_ms_per_mp, decode_ms = [], []
    for idx, it in enumerate(items):
        path = CORPUS / it["path"]
        bits16 = it.get("bits") == 16
        bits = payload(adapter.nbits, idx, salt=str(adapter.nbits))
        rec: dict = {"item": it["path"], "set": it["set"]}
        try:
            ref = load_image(path)
            rec["h"], rec["w"] = ref.shape[0], ref.shape[1]
            with Timer(device) as te:
                marked = adapter.embed(ref, bits)
            if idx > 0:  # first call includes lazy init / cudnn autotune
                embed_ms_per_mp.append(te.elapsed * 1000 / (ref.shape[0] * ref.shape[1] / 1e6))
            marked = quantize(marked, bits16)
            if marked.shape != ref.shape:
                raise RuntimeError(f"shape changed {ref.shape}->{marked.shape}")
            expect = adapter.physical_bits(bits) if hasattr(adapter, "physical_bits") else bits
            rec["psnr"] = metrics.psnr(ref, marked)
            rec["ssim"] = metrics.ssim(ref, marked)
            rec["ms_ssim"] = metrics.ms_ssim(ref, marked, device)
            rec["lpips"] = metrics.lpips_score(ref, marked, device)
            if it["set"] in PERCEPTUAL_SETS:
                want_map = Path(it["path"]).name in GALLERY
                f = metrics.flip(ref, marked, want_map=want_map)
                rec["flip"], fmap = (f if want_map else (f, None))
                if it["set"] in CVVDP_SETS:
                    try:
                        rec["cvvdp"] = metrics.cvvdp_image(ref, marked, device)
                    except Exception as exc:
                        rec["cvvdp_error"] = repr(exc)[:200]
                if want_map:
                    save_gallery(adapter.name, it["path"], ref, marked, fmap)
            rec["attacks"] = {}
            for aname, fn in attacks.IMAGE_ATTACKS.items():
                x = fn(marked)
                with Timer(device) as td:
                    d = adapter.decode(x)
                if aname == "none" and idx > 0:
                    decode_ms.append(td.elapsed * 1000)
                r = {"bit_acc": bit_accuracy(expect, d.get("bits")),
                     "matches": int((np.asarray(d["bits"]).reshape(-1)[: len(expect)] == expect).sum()) if d.get("bits") is not None and len(np.asarray(d["bits"]).reshape(-1)) >= len(expect) else 0}
                if "native_detect" in d:
                    r["native_detect"] = d["native_detect"]
                    if "native_payload" in d:
                        r["payload_exact"] = bool(d["native_detect"]) and d["native_payload"] == "".join(str(int(b)) for b in bits[: adapter.capacity])
                if "native_score" in d:
                    r["native_score"] = d["native_score"]
                rec["attacks"][aname] = r
            # False-positive material: decode the UNMARKED reference (and attacked copies)
            for aname in FPR_ATTACKS:
                d = adapter.decode(attacks.IMAGE_ATTACKS[aname](ref))
                fpr_bits.append({"item": idx, "attack": aname,
                                 "bits": None if d.get("bits") is None else np.asarray(d["bits"]).reshape(-1).astype(int).tolist(),
                                 "native_detect": d.get("native_detect"), "native_score": d.get("native_score")})
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            traceback.print_exc()
        records.append(rec)
        print(f"  {adapter.name} [{idx + 1}/{len(items)}] {it['path']} psnr={rec.get('psnr', float('nan')):.2f} "
              f"jpeg75={rec.get('attacks', {}).get('jpeg75', {}).get('bit_acc', float('nan')):.3f}", flush=True)
    # FPR keys: each unmarked decode is compared with the payload item idx would have carried,
    # plus FPR_EXTRA_KEYS independent random keys (more trials, same decodes).
    keys = []
    for fb in fpr_bits:
        exp = [payload(adapter.nbits, fb["item"], salt=str(adapter.nbits))] + \
              [payload(adapter.nbits, 10_000 + k, salt="fpr") for k in range(FPR_EXTRA_KEYS)]
        if hasattr(adapter, "physical_bits"):
            exp = [adapter.physical_bits(e) for e in exp]
        got = None if fb["bits"] is None else np.asarray(fb["bits"], dtype=np.uint8)
        keys.append({"item": fb["item"], "attack": fb["attack"], "native_detect": fb["native_detect"],
                     "native_score": fb["native_score"],
                     "matches": [int((got[: len(e)] == e).sum()) if got is not None and len(got) >= len(e) else 0 for e in exp]})
    return {"meta": adapter.meta(), "device": device, "load_s": load_s,
            "embed_ms_per_mp_median": float(np.median(embed_ms_per_mp)) if embed_ms_per_mp else None,
            "decode_ms_median": float(np.median(decode_ms)) if decode_ms else None,
            "records": records, "fpr": keys}


def cpu_latency(adapter, items: list[dict], n: int = 4) -> dict:
    import torch

    torch.set_num_threads(4)
    adapter.load("cpu")
    emb, dec = [], []
    for idx, it in enumerate(items[: n + 1]):
        ref = load_image(CORPUS / it["path"])
        bits = payload(adapter.nbits, idx, salt=str(adapter.nbits))
        with Timer("cpu") as te:
            m = adapter.embed(ref, bits)
        with Timer("cpu") as td:
            adapter.decode(quantize(m, False))
        if idx > 0:
            emb.append(te.elapsed * 1000 / (ref.shape[0] * ref.shape[1] / 1e6))
            dec.append(td.elapsed * 1000)
    return {"embed_ms_per_mp_median": float(np.median(emb)), "decode_ms_median": float(np.median(dec)),
            "threads": 4, "host": host_info()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full")
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--manifest", default=str(BENCH_DIR / "corpus_manifest.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu-latency", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--div2k", type=int, default=25, help="DIV2K items used (the run is CPU-bound)")
    ap.add_argument("--shard", default="0/1", help="i/n: run configs with index %% n == i (parallel workers)")
    args = ap.parse_args()
    shard_i, shard_n = map(int, args.shard.split("/"))
    device = device_name(args.device)
    manifest = json.loads(Path(args.manifest).read_text())
    items = manifest["image"]
    items = [i for i in items if i["set"] != "div2k"] + [i for i in items if i["set"] == "div2k"][: args.div2k]
    if args.limit:
        items = items[: args.limit]
    outdir = RESULTS / ("image" + args.tag)
    for ci, adapter in enumerate(image_configs(args.profile)):
        if ci % shard_n != shard_i:
            continue
        if args.only and not any(s in adapter.name for s in args.only.split(",")):
            continue
        safe = adapter.name.replace("@", "_at_")
        if args.cpu_latency:
            out = outdir / "cpu_latency" / f"{safe}.json"
            if out.exists():
                continue
            lat_items = [i for i in items if i["set"] == "kodak"][:3] + [i for i in items if i["set"] == "div2k"][:2]
            try:
                write_json(out, {"meta": adapter.meta(), **cpu_latency(adapter, lat_items or items)})
            except Exception as exc:
                write_json(out, {"meta": adapter.meta(), "error": repr(exc)[:500]})
            print("cpu latency", adapter.name, flush=True)
            continue
        out = outdir / f"{safe}.json"
        if out.exists():
            print("skip", adapter.name)
            continue
        print(f"== {adapter.name} on {device}", flush=True)
        t0 = time.time()
        try:
            res = run_config(adapter, items, device, args)
        except Exception as exc:
            traceback.print_exc()
            res = {"meta": adapter.meta(), "error": f"{type(exc).__name__}: {exc}"[:800], "device": device}
        res["host"] = host_info()
        res["wall_s"] = time.time() - t0
        write_json(out, res)
        del adapter
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
