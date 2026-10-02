"""Cross-device / cross-run decode stability of the SAME marked files.

Tests whether the Video Seal detector is bit-stable across CPUs: a clip is marked once
with the Video Seal TorchScript artefact (y_256b_img.jit, sha pinned below) at scaling_w
0.4 on CPU, delivered as H.264 CRF 18 and probed at CRF 22. The resulting
files are then decoded on every available backend; raw per-bit logits are
recorded so hosts can be compared bit-for-bit.

  python stability.py make            # on the GPU host: create the fixtures + decode there
  python stability.py decode          # on any other host (e.g. the Mac): decode the same fixtures

Fixtures live in $BENCH_WORK/stability (never in git); results in $BENCH_RESULTS/stability/.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from common import BENCH_WORK, CORPUS, WEIGHTS, host_info, payload, sha256_file, write_json, RESULTS

FIX = BENCH_WORK / "stability"
JIT = WEIGHTS / "videoseal" / "y_256b_img.jit"
PROD_JIT_SHA = "5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b"


def read_frames(path: Path, n: int | None = None) -> np.ndarray:
    import av

    out = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            out.append(f.to_ndarray(format="rgb24"))
            if n and len(out) >= n:
                break
    return np.stack(out)


def encode(frames: np.ndarray, fps: float, dst: Path, crf: int) -> None:
    h, w = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "high",
                          "-pix_fmt", "yuv420p", "-x264-params", "threads=1", str(dst)], stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    assert p.wait() == 0


def backends() -> list[tuple[str, str, int]]:
    import torch

    out = [("cpu", "cpu", 4), ("cpu", "cpu", 1)]
    if torch.cuda.is_available():
        out.insert(0, ("cuda", "cuda", 0))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        out.append(("mps", "mps", 0))
    return out


def decode_all(tag: str) -> dict:
    import torch

    res: dict = {"host": host_info(), "tag": tag, "video": {}, "image": {}}
    assert sha256_file(JIT) == PROD_JIT_SHA, "jit artefact differs from the pinned sha"
    for fx in sorted(FIX.glob("*.mp4")):
        frames = read_frames(fx)
        per = {}
        for name, dev, threads in backends():
            if threads:
                torch.set_num_threads(threads)
            model = torch.jit.load(str(JIT), map_location=dev).eval()
            x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255)
            logits = []
            with torch.no_grad():
                for s in range(0, len(x), 16):
                    logits.append(model.detect(x[s:s + 16].to(dev), is_video=True)[:, 1:].float().cpu())
            L = torch.cat(logits).numpy()
            per[f"{name}{threads or ''}"] = {"avg_logits": L.mean(0).tolist(), "frame0_logits": L[0].tolist()}
        res["video"][fx.name] = {"sha256": sha256_file(fx), "decodes": per}
    # TrustMark Q: same JPEG decoded on every backend
    tmf = FIX / "trustmark_q_jpeg75.jpg"
    if tmf.exists():
        from PIL import Image
        from models_image import TrustMarkAdapter
        from torchvision import transforms

        img = Image.open(tmf).convert("RGB")
        per = {}
        for name, dev, threads in backends():
            if threads:
                torch.set_num_threads(threads)
            a = TrustMarkAdapter("Q", 1.2)
            a.load(dev)
            sub = a.tm.get_the_image_for_processing(img).resize((a.tm.model_resolution_dec,) * 2, Image.BILINEAR)
            t = transforms.ToTensor()(sub).unsqueeze(0).to(dev) * 2 - 1
            with torch.no_grad():
                per[f"{name}{threads or ''}"] = a.tm.decoder.decoder(t)[0].float().cpu().numpy().tolist()
        res["image"][tmf.name] = {"sha256": sha256_file(tmf), "decodes": per}
    return res


def make() -> None:
    import torch

    FIX.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).resolve().parent.parent / "corpus_manifest.json").read_text())
    clips = manifest["video"][:3]
    torch.set_num_threads(4)
    model = torch.jit.load(str(JIT), map_location="cpu").eval()
    for i, it in enumerate(clips):
        frames = read_frames(CORPUS / it["path"], 60)
        bits = torch.from_numpy(payload(256, i, salt="stability").astype(np.float32))[None]
        for sw in (0.4, 1.1):
            model.blender.scaling_w = sw
            out = []
            with torch.no_grad():
                for s in range(0, len(frames), 16):
                    x = torch.from_numpy(frames[s:s + 16]).permute(0, 3, 1, 2).float().div(255)
                    out.append(model.embed(x, bits, is_video=True).clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).numpy())
            marked = np.concatenate(out)
            stem = f"{Path(it['path']).stem}_sw{sw}"
            encode(marked, it.get("fps", 30), FIX / f"{stem}_crf18.mp4", 18)
            encode(read_frames(FIX / f"{stem}_crf18.mp4"), it.get("fps", 30), FIX / f"{stem}_crf22.mp4", 22)
        (FIX / "payloads.json").write_text(json.dumps({Path(c["path"]).stem: payload(256, i, salt="stability").tolist() for i, c in enumerate(clips)}))
    # TrustMark Q fixture
    from PIL import Image
    from models_image import TrustMarkAdapter
    import io

    a = TrustMarkAdapter("Q", 1.2)
    a.load("cpu")
    ref = np.asarray(Image.open(CORPUS / "image/kodak/kodim23.png").convert("RGB"), dtype=np.float32) / 255
    m = a.embed(ref, payload(100, 0, salt="stability"))
    buf = io.BytesIO()
    Image.fromarray((m * 255).round().astype(np.uint8)).save(buf, "JPEG", quality=75)
    (FIX / "trustmark_q_jpeg75.jpg").write_bytes(buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "decode"])
    args = ap.parse_args()
    if args.mode == "make" and not any(FIX.glob("*.mp4")):
        make()
    tag = f"{platform.machine()}-{host_info().get('cpu_model', 'cpu')}".replace(" ", "_")[:80]
    write_json(RESULTS / "stability" / f"{tag}.json", decode_all(tag))
    print("wrote", tag)


if __name__ == "__main__":
    main()
