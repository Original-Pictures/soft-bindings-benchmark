"""Video bake-off runner (one JSON per config in $BENCH_RESULTS/video/).

Pipeline per clip: FFV1 reference -> model embed (chunked, video mode) -> marked
lossless FFV1 (quality is measured here, watermark cost only) -> delivered H.264
CRF 18 High yuv420p (the delivery profile) -> transcode attacks -> decode.

Usage: python bench_video.py [--only substr,...] [--device cuda] [--manifest p] [--limit N] [--seconds S]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

import metrics
from common import BENCH_DIR, CORPUS, OUT, RESULTS, Timer, bit_accuracy, device_name, host_info, payload, write_json
from models_image import ImWatermarkAdapter, MetaSealAdapter

ENC = {
    "h264_crf18": ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p"],
    "h264_crf22": ["-c:v", "libx264", "-preset", "medium", "-crf", "22", "-profile:v", "high", "-pix_fmt", "yuv420p"],
    "h264_crf23": ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"],
    "h264_crf28": ["-c:v", "libx264", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p"],
    "h265_crf23": ["-c:v", "libx265", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error"],
    "h265_crf28": ["-c:v", "libx265", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p", "-x265-params", "log-level=error"],
    "resize0.5_h264_crf23": ["-vf", "scale=iw/2:ih/2:flags=bicubic", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"],
    "crop75_h264_crf23": ["-vf", "crop=iw*0.75:ih*0.75", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"],
}
ATTACKS = ["h264_crf18", "h264_crf22", "h264_crf23", "h264_crf28", "h265_crf23", "h265_crf28", "resize0.5_h264_crf23", "crop75_h264_crf23"]


def read_frames(path: Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        frames = []
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break
    return np.stack(frames), fps


def write_ffv1(frames_u8: np.ndarray, fps: float, path: Path) -> None:
    h, w = frames_u8.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
                          "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p", str(path)], stdin=subprocess.PIPE)
    p.stdin.write(frames_u8.tobytes())
    p.stdin.close()
    assert p.wait() == 0


def transcode(src: Path, dst: Path, args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, "-an", str(dst)], check=True)


class VideoModel:
    """Wraps an image adapter; Meta Seal models run in native video mode."""

    def __init__(self, base, video_mode: bool) -> None:
        self.base, self.video_mode = base, video_mode
        self.name = base.name + ("" if video_mode else "(per-frame)")

    def load(self, device):
        self.base.load(device)
        self.device = device

    def embed(self, frames: np.ndarray, bits: np.ndarray) -> np.ndarray:
        """frames: F H W 3 uint8 -> uint8"""
        import torch

        if not self.video_mode:
            return np.stack([np.clip(np.round(self.base.embed(f.astype(np.float32) / 255, bits) * 255), 0, 255).astype(np.uint8) for f in frames])
        m = self.base.m
        step = int(getattr(m, "step_size", 1) or 1)
        win = step * 8
        out = np.empty_like(frames)
        msg = torch.from_numpy(bits.astype(np.float32))[None].to(self.device)
        for s in range(0, len(frames), win):
            x = torch.from_numpy(frames[s:s + win]).permute(0, 3, 1, 2).float().div(255).to(self.device)
            with torch.no_grad():
                y = m.embed(x, msgs=msg, is_video=True, lowres_attenuation=True)["imgs_w"]
            out[s:s + win] = y.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            del x, y
        return out

    def decode(self, frames: np.ndarray) -> dict:
        import torch

        if not self.video_mode:
            per = [self.base.decode(f.astype(np.float32) / 255).get("bits") for f in frames[::3]]
            per = [p for p in per if p is not None]
            if not per:
                return {"bits": None}
            return {"bits": (np.mean(per, axis=0) > 0.5).astype(np.uint8)}
        m = self.base.m
        logits = []
        for s in range(0, len(frames), 16):
            x = torch.from_numpy(frames[s:s + 16]).permute(0, 3, 1, 2).float().div(255).to(self.device)
            with torch.no_grad():
                p = m.detect(x, is_video=True)["preds"][:, 1:]
            logits.append(p.float().cpu())
            del x
        L = torch.cat(logits).numpy()
        avg = L.mean(axis=0)
        return {"bits": (avg > 0).astype(np.uint8), "logits_avg": avg, "frame_bits": (L > 0).astype(np.uint8)}


def video_configs() -> list[VideoModel]:
    cfgs = [VideoModel(MetaSealAdapter("videoseal", sw), True) for sw in (None, 0.4, 0.7, 1.1)]
    cfgs += [VideoModel(MetaSealAdapter("pixelseal", sw), True) for sw in (None, 0.4)]
    cfgs.append(VideoModel(MetaSealAdapter("chunkyseal", None), True))
    cfgs.append(VideoModel(ImWatermarkAdapter("rivaGan"), False))
    return cfgs


def run(model: VideoModel, items: list[dict], device: str, seconds: float) -> dict:
    model.load(device)
    records, fpr = [], []
    work = OUT / "video" / model.name.replace("@", "_at_")
    work.mkdir(parents=True, exist_ok=True)
    for idx, it in enumerate(items):
        ref_path = CORPUS / it["path"]
        rec: dict = {"item": it["path"]}
        try:
            frames, fps = read_frames(ref_path, int(round(it.get("fps", 30) * seconds)))
            bits = payload(model.base.nbits, idx, salt=f"video{model.base.nbits}")
            with Timer(device) as te:
                marked = model.embed(frames, bits)
            rec["frames"], rec["fps"] = len(frames), fps
            rec["embed_fps"] = len(frames) / te.elapsed
            ref_ll, marked_ll = work / f"{idx}_ref.mkv", work / f"{idx}_marked.mkv"
            write_ffv1(frames, fps, ref_ll)
            write_ffv1(marked, fps, marked_ll)
            f32 = lambda a: a.astype(np.float32) / 255  # noqa: E731
            sub = range(0, len(frames), 10)
            rec["psnr"] = float(np.mean([metrics.psnr(f32(frames[i]), f32(marked[i])) for i in range(len(frames))]))
            rec["ssim"] = float(np.mean([metrics.ssim(f32(frames[i]), f32(marked[i])) for i in sub]))
            rec["lpips"] = float(np.mean([metrics.lpips_score(f32(frames[i]), f32(marked[i]), device) for i in sub]))
            rec["flip"] = float(np.mean([metrics.flip(f32(frames[i]), f32(marked[i])) for i in list(sub)[:4]]))
            rec["vmaf"] = metrics.vmaf(ref_ll, marked_ll)
            rec["vmaf_identical"] = metrics.vmaf(ref_ll, ref_ll)  # per-clip ceiling for the same pipeline
            try:
                n = min(len(frames), 24)  # float copies of 1080p frames: 5 s at 50 fps OOM-killed the 16 GB host
                rec["cvvdp"] = metrics.cvvdp_video(f32(frames[:n]), f32(marked[:n]), fps, device)
            except Exception as exc:
                rec["cvvdp_error"] = repr(exc)[:200]
            delivered = work / f"{idx}_h264_crf18.mp4"
            transcode(marked_ll, delivered, ENC["h264_crf18"])
            rec["attacks"] = {}
            for aname in ATTACKS:
                att = delivered if aname == "h264_crf18" else work / f"{idx}_{aname}.mp4"
                if aname != "h264_crf18":
                    transcode(delivered, att, ENC[aname])
                af, _ = read_frames(att)
                with Timer(device) as td:
                    d = model.decode(af)
                r = {"bit_acc": bit_accuracy(bits, d.get("bits")),
                     "matches": int((d["bits"][: len(bits)] == bits).sum()) if d.get("bits") is not None else 0,
                     "decode_fps": len(af) / td.elapsed}
                if "logits_avg" in d:
                    r["mean_abs_logit"] = float(np.mean(np.abs(d["logits_avg"])))
                    r["min_abs_logit"] = float(np.min(np.abs(d["logits_avg"])))
                    r["frame_bit_acc"] = float(np.mean([bit_accuracy(bits, fb) for fb in d["frame_bits"]]))
                rec["attacks"][aname] = r
                if att != delivered:
                    att.unlink()
            # FPR material: unmarked reference through the ship encode
            ref_enc = work / f"{idx}_ref_crf18.mp4"
            transcode(ref_ll, ref_enc, ENC["h264_crf18"])
            rf, _ = read_frames(ref_enc)
            d = model.decode(rf)
            got = d.get("bits")
            exp = [payload(model.base.nbits, idx, salt=f"video{model.base.nbits}")] + \
                  [payload(model.base.nbits, 10_000 + k, salt="fpr") for k in range(200)]
            fpr.append({"item": idx, "matches": [int((got[: len(e)] == e).sum()) if got is not None else 0 for e in exp]})
            for p in (ref_ll, marked_ll, ref_enc):
                p.unlink(missing_ok=True)
            if idx > 0:
                delivered.unlink(missing_ok=True)  # keep clip 0 for the stability test
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
            traceback.print_exc()
        records.append(rec)
        print(f"  {model.name} [{idx + 1}/{len(items)}] {it['path']} psnr={rec.get('psnr', float('nan')):.2f} "
              f"vmaf={rec.get('vmaf', float('nan')):.2f} crf23={rec.get('attacks', {}).get('h264_crf23', {}).get('bit_acc')}", flush=True)
    return {"meta": {**model.base.meta(), "name": model.name, "video_mode": model.video_mode}, "device": device,
            "records": records, "fpr": fpr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--manifest", default=str(BENCH_DIR / "corpus_manifest.json"))
    # 6 clips x 5 s x 8 transcodes per config keeps the single-GPU run within a working day.
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=2.5)  # 5 s: host-RAM OOM on 1080p50 crowd_run (2026-09-23)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    device = device_name(args.device)
    items = json.loads(Path(args.manifest).read_text())["video"]
    # Content spread first: static/pan, fire, water, fast motion, animation, dense crowd.
    pref = ["aspen", "controlled_burn", "red_kayak", "speed_bag", "sintel_trailer", "crowd_run"]
    items = sorted(items, key=lambda i: next((k for k, p in enumerate(pref) if Path(i["path"]).name.startswith(p)), 99))
    if args.limit:
        items = items[: args.limit]
    shard_i, shard_n = map(int, args.shard.split("/"))
    for ci, model in enumerate(video_configs()):
        if ci % shard_n != shard_i:
            continue
        if args.only and not any(s in model.name for s in args.only.split(",")):
            continue
        out = RESULTS / "video" / f"{model.name.replace('@', '_at_')}.json"
        if out.exists():
            print("skip", model.name)
            continue
        print(f"== {model.name} on {device}", flush=True)
        t0 = time.time()
        secs = min(args.seconds, 2.0) if not model.video_mode else args.seconds
        try:
            res = run(model, items, device, secs)
        except Exception as exc:
            traceback.print_exc()
            res = {"meta": {"name": model.name}, "error": f"{type(exc).__name__}: {exc}"[:800]}
        res["host"], res["wall_s"], res["seconds"] = host_info(), time.time() - t0, secs
        write_json(out, res)
        try:
            import torch

            del model
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
