"""Download every benchmarked checkpoint to $BENCH_WORK/weights and pin its sha256.

First download records sha256 + bytes in weights.lock.json (our own pin: several
upstreams, e.g. Meta's dl.fbaipublicfiles, publish no digest). Later runs verify
against the lock and refuse on mismatch, so GPU and local runs use byte-identical
weights. Licence per checkpoint is carried in the table for the report.

Usage: python fetch_weights.py [--only name,name] [--skip-large]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from common import SCRIPTS, WEIGHTS, sha256_file

TM = "https://cai-watermark.adobe.net/watermarking/trustmark-models/"
FB = "https://dl.fbaipublicfiles.com/"
HF = "https://huggingface.co/"

# name -> (url, relative path, licence of this checkpoint, licence source)
WEIGHTS_TABLE: dict[str, tuple[str, str, str, str]] = {
    **{
        f"trustmark_{p}_{v}": (f"{TM}{p}_{v}.ckpt", f"trustmark/{p}_{v}.ckpt", "MIT",
                               "adobe/trustmark README: MIT covers code and downloaded model files")
        for v in "BCQP" for p in ("encoder", "decoder")
    },
    **{
        f"trustmark_yaml_{v}": (f"{TM}trustmark_{v}.yaml", f"trustmark/trustmark_{v}.yaml", "MIT", "same")
        for v in "BCQP"
    },
    "videoseal_1.0_pth": (f"{FB}videoseal/y_256b_img.pth", "videoseal/y_256b_img.pth", "MIT",
                          "facebookresearch/videoseal README: all models MIT"),
    "videoseal_1.0_jit": (f"{FB}videoseal/y_256b_img.jit", "videoseal/y_256b_img.jit", "MIT",
                          "same; TorchScript artefact (sha pinned here)"),
    "pixelseal": (f"{FB}videoseal/pixelseal/checkpoint.pth", "videoseal/pixelseal.pth", "MIT", "same"),
    "chunkyseal": (f"{FB}videoseal/chunkyseal/checkpoint.pth", "videoseal/chunkyseal.pth", "MIT", "same (13.4 GB)"),
    "wam_mit": (f"{FB}watermark_anything/wam_mit.pth", "wam/wam_mit.pth", "MIT",
                "facebookresearch/watermark-anything README + HF card (SA-1B weights)"),
    "audioseal_gen_streaming": (f"{HF}facebook/audioseal/resolve/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/generator_streaming.pth",
                                "audioseal/generator_streaming.pth", "MIT", "HF facebook/audioseal license: mit (since 2024-04-02)"),
    "audioseal_det_streaming": (f"{HF}facebook/audioseal/resolve/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/detector_streaming.pth",
                                "audioseal/detector_streaming.pth", "MIT", "same"),
    "audioseal_gen_base": (f"{HF}facebook/audioseal/resolve/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/generator_base.pth",
                           "audioseal/generator_base.pth", "MIT", "same"),
    "audioseal_det_base": (f"{HF}facebook/audioseal/resolve/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/detector_base.pth",
                           "audioseal/detector_base.pth", "MIT", "same"),
    "wavmark": (f"{HF}M4869/WavMark/resolve/main/step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl",
                "wavmark/wavmark.model.pkl", "MIT", "HF M4869/WavMark license: mit"),
    "silentcipher_models": ("https://github.com/sony/silentcipher/releases/download/release/Models.tar.gz",
                            "silentcipher/Models.tar.gz", "UNSTATED",
                            "GitHub release asset of MIT repo; HF Sony/SilentCipher card has no licence field"),
}
LARGE = {"chunkyseal"}
LOCK = SCRIPTS / "weights.lock.json"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "op-wm-bench"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-large", action="store_true")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    failed = []
    for name, (url, rel, lic, src) in WEIGHTS_TABLE.items():
        if only and name not in only:
            continue
        if args.skip_large and name in LARGE:
            continue
        dest = WEIGHTS / rel
        try:
            if not dest.exists():
                print(f"downloading {name} ...", flush=True)
                download(url, dest)
            digest = sha256_file(dest)
        except Exception as exc:  # recorded, not fatal: the report lists unavailable weights
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            failed.append(name)
            continue
        pinned = lock.get(name, {}).get("sha256")
        if pinned and pinned != digest:
            raise SystemExit(f"{name}: sha256 mismatch pinned={pinned} got={digest}")
        lock[name] = {"url": url, "path": rel, "sha256": digest, "bytes": dest.stat().st_size,
                      "licence": lic, "licence_source": src}
        print(f"{name}: {digest} {dest.stat().st_size}")
        if dest.name.endswith(".tar.gz") and not (dest.parent / "Models").exists():
            import tarfile

            with tarfile.open(dest) as tar:
                tar.extractall(dest.parent, filter="data")
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    if failed:
        print("unavailable:", ",".join(failed))


if __name__ == "__main__":
    main()
