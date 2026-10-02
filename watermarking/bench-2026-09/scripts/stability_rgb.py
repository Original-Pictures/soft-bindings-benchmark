"""Separate colour conversion from model arithmetic in the cross-host decode difference.

stability.py decodes the same marked MP4 files on two hosts and finds different Video Seal
logits; frame_hash.py shows the YUV planes agree and PyAV's RGB conversion does not. This
script closes the remaining gap with a 2x2 design: each host converts the fixtures to RGB
once and saves the arrays, then every host decodes every host's arrays with the same
Video Seal TorchScript artefact.

  python stability_rgb.py dump      # on each host: $BENCH_WORK/stability/rgb/<host>/<clip>.npy
  python stability_rgb.py decode    # on each host, after copying the other host's rgb/<host> dir

If logits agree across decode hosts for the same input arrays, the model is not the source
of the difference; if they differ between input hosts on one decode host, the RGB input is.
Only the first N_FRAMES frames are used, as in frame_hash.py, to keep the arrays portable.
"""

from __future__ import annotations

import argparse
import hashlib
import platform

import numpy as np

from common import RESULTS, host_info, sha256_file, write_json
from stability import FIX, JIT, PROD_JIT_SHA, backends, read_frames

N_FRAMES = 16
RGB = FIX / "rgb"


def host_tag() -> str:
    return f"{platform.machine()}-{host_info().get('cpu_model', 'cpu')}".replace(" ", "_")[:80]


def dump() -> None:
    out = RGB / host_tag()
    out.mkdir(parents=True, exist_ok=True)
    for fx in sorted(FIX.glob("*.mp4")):
        np.save(out / f"{fx.stem}.npy", read_frames(fx, N_FRAMES))
    print("dumped", out)


def _logits(model, frames: np.ndarray, dev: str) -> np.ndarray:
    import torch

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255)
    with torch.no_grad():
        return model.detect(x.to(dev), is_video=True)[:, 1:].float().cpu().numpy()


def decode() -> None:
    import torch

    assert sha256_file(JIT) == PROD_JIT_SHA, "jit artefact differs from the pinned sha"
    res: dict = {"host": host_info(), "decode_host": host_tag(), "n_frames": N_FRAMES, "inputs": {}}
    for d in sorted(p for p in RGB.iterdir() if p.is_dir()):
        per_clip = {}
        for f in sorted(d.glob("*.npy")):
            frames = np.load(f)
            per = {"rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest()}
            for name, dev, threads in backends():
                if threads:
                    torch.set_num_threads(threads)
                model = torch.jit.load(str(JIT), map_location=dev).eval()
                L = _logits(model, frames, dev)
                per[f"{name}{threads or ''}"] = {"avg_logits": L.mean(0).tolist(), "frame0_logits": L[0].tolist()}
            per_clip[f.stem] = per
        res["inputs"][d.name] = per_clip
    write_json(RESULTS / "stability" / f"rgb-decode-{host_tag()}.json", res)
    print("decoded inputs from", list(res["inputs"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "decode"])
    args = ap.parse_args()
    dump() if args.mode == "dump" else decode()


if __name__ == "__main__":
    main()
