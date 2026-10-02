"""Download every fingerprint checkpoint (and the watermarks of the combination track) to $BENCH_WORK/weights and pin its sha256.

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
    # --- fingerprint methods
    "sscd_disc_mixup": (f"{FB}sscd-copy-detection/sscd_disc_mixup.torchscript.pt", "sscd/sscd_disc_mixup.torchscript.pt",
                        "NONE STATED (code MIT)", "facebookresearch/sscd-copy-detection: weights trained on DISC21 (CC BY-NC 4.0)"),
    "sscd_disc_large": (f"{FB}sscd-copy-detection/sscd_disc_large.torchscript.pt", "sscd/sscd_disc_large.torchscript.pt",
                        "NONE STATED (code MIT)", "same"),
    "isc21_ft_v107": ("https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar",
                      "isc21/isc_ft_v107.pth.tar", "MIT (repo release asset)", "lyakaap/ISC21-Descriptor-Track-1st; fine-tuned on DISC21"),
    "clap_630k_audioset_best": (f"{HF}lukewys/laion_clap/resolve/main/630k-audioset-best.pt", "clap/630k-audioset-best.pt",
                                "CC0-1.0", "HF lukewys/laion_clap license: cc0-1.0"),
    "nmfp_triplet": ("https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1", "nmfp/nmfp-triplet.zip",
                     "GPL-3.0 (repo)", "raraz15/neural-music-fp; Zenodo 15719945"),
    "lama_big": ("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
                 "lama/big-lama.pt", "Apache-2.0", "advimman/lama (Apache-2.0) weights, TorchScript by simple-lama-inpainting (Apache-2.0)"),
    # --- watermarks for the combination track (same URLs and pins as the 2026-09 watermark bench)
    **{f"trustmark_{p}_Q": (f"{TM}{p}_Q.ckpt", f"trustmark/{p}_Q.ckpt", "MIT",
                            "adobe/trustmark README: MIT covers code and downloaded model files") for p in ("encoder", "decoder")},
    "trustmark_yaml_Q": (f"{TM}trustmark_Q.yaml", "trustmark/trustmark_Q.yaml", "MIT", "same"),
    "videoseal_1.0_pth": (f"{FB}videoseal/y_256b_img.pth", "videoseal/y_256b_img.pth", "MIT",
                          "facebookresearch/videoseal README: all models MIT"),
    "audioseal_gen_base": (f"{HF}facebook/audioseal/resolve/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/generator_base.pth",
                           "audioseal/generator_base.pth", "MIT", "HF facebook/audioseal license: mit"),
    "audioseal_det_base": (f"{HF}facebook/audioseal/resolve/3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba/detector_base.pth",
                           "audioseal/detector_base.pth", "MIT", "same"),
}
LARGE: set[str] = set()
LOCK = SCRIPTS / "weights.lock.json"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "op-fp-bench"})
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
