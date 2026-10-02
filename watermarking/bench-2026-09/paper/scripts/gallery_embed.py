"""Re-embed one Kodak image with five configurations for the visual gallery (Fig. 9).

The bench kept only JPEG crops of marked outputs, and a residual computed against a
JPEG crop shows compression noise rather than the watermark. The marked PNGs written
here go to $BENCH_WORK/out/gallery, never to git. Run with the bench venv:

    ~/op-wm-bench-work/.venv/bin/python paper/scripts/gallery_embed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench_image import load_image  # noqa: E402
from common import CORPUS, OUT, payload  # noqa: E402
from metrics import psnr  # noqa: E402
from models_image import ImWatermarkAdapter, MetaSealAdapter, TrustMarkAdapter, WAMAdapter  # noqa: E402

ITEM = CORPUS / "image" / "kodak" / "kodim05.png"
DEST = OUT / "gallery"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    ref = load_image(ITEM)
    Image.fromarray(np.round(ref * 255).astype(np.uint8)).save(DEST / "original.png")
    configs = [MetaSealAdapter("pixelseal", 0.15), MetaSealAdapter("videoseal", None), TrustMarkAdapter("Q", 1.2),
               WAMAdapter(None), ImWatermarkAdapter("dwtDctSvd")]
    summary = {}
    for a in configs:
        a.load("cpu")
        marked = a.embed(ref, payload(a.nbits, 0))
        u8 = np.clip(np.round(marked * 255), 0, 255).astype(np.uint8)
        Image.fromarray(u8).save(DEST / f"{a.name}.png")
        m = u8.astype(np.float32) / 255.0
        # Scalar FLIP error map (no colour map applied), so the figure owns the colour scale.
        import flip_evaluator

        err, _, _ = flip_evaluator.evaluate(np.ascontiguousarray(ref), np.ascontiguousarray(m), "LDR", applyMagma=False)
        np.save(DEST / f"{a.name}.flipmap.npy", np.asarray(err, dtype=np.float32).squeeze())
        summary[a.name] = {"psnr": psnr(ref, m)}
        print(a.name, summary[a.name], flush=True)
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
