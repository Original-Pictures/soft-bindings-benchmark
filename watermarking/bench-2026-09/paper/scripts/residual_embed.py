"""Re-embed one speech clip, one music clip and one video clip for the residual figures.

Companion to gallery_embed.py (images). The bench kept only rendered PNG spectrograms
and no marked video, so the figures need the marked signals themselves. Outputs go to
$BENCH_WORK/out/residuals, never to git:

    audio/<config>__<clip>.npz   original y, marked yw, sample rate sr (first 6 s)
    video/<config>.npz           original frame, marked frame, FLIP map (frame 12)

Run with the bench venv, from bench-2026-09/scripts:

    ~/op-wm-bench-work/.venv/bin/python ../paper/scripts/residual_embed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bench_audio import load_audio  # noqa: E402
from bench_video import VideoModel, read_frames  # noqa: E402
from common import CORPUS, OUT, payload  # noqa: E402
from models_audio import AudioSealAdapter, PerthAdapter, SilentCipherAdapter, WavMarkAdapter  # noqa: E402
from models_image import MetaSealAdapter  # noqa: E402

DEST = OUT / "residuals"
SECONDS = 6.0
AUDIO_CLIPS = {
    "speech": sorted((CORPUS / "audio" / "speech_44100").glob("*.wav"))[0],
    "music": sorted((CORPUS / "audio" / "music44k").glob("*.wav"))[0],
}
VIDEO_CLIP = CORPUS / "video" / "aspen_2s.mkv"
VIDEO_FRAMES = 24
VIDEO_FRAME = 12


def audio() -> None:
    (DEST / "audio").mkdir(parents=True, exist_ok=True)
    adapters = [AudioSealAdapter("base", 1.0), AudioSealAdapter("streaming", 0.5), WavMarkAdapter(),
                SilentCipherAdapter("44.1k"), PerthAdapter()]
    for a in adapters:
        try:
            a.load("cpu")
        except Exception as exc:  # a reference model may be missing locally; record, do not guess
            print(f"skip {a.name}: {exc!r}", flush=True)
            continue
        for clip, path in AUDIO_CLIPS.items():
            y, sr = load_audio(path)
            y = y[: int(SECONDS * sr)]
            bits = payload(max(a.nbits, 1), 0, salt=f"audio{a.nbits}")[: a.nbits]
            yw = a.embed(y, sr, bits)
            n = min(len(y), len(yw))
            np.savez_compressed(DEST / "audio" / f"{a.name}__{clip}.npz", y=y[:n], yw=yw[:n], sr=sr)
            print(a.name, clip, "SNR dB", round(10 * np.log10(np.sum(y[:n] ** 2) / np.sum((yw[:n] - y[:n]) ** 2)), 2),
                  flush=True)


def video() -> None:
    import flip_evaluator

    (DEST / "video").mkdir(parents=True, exist_ok=True)
    frames, _ = read_frames(VIDEO_CLIP, VIDEO_FRAMES)
    for model, sw in (("pixelseal", 0.2), ("pixelseal", 0.4), ("videoseal", 0.2), ("videoseal", 0.4)):
        base = MetaSealAdapter(model, sw)
        vm = VideoModel(base, video_mode=True)
        vm.load("cpu")
        marked = vm.embed(frames, payload(base.nbits, 0, salt=f"video{base.nbits}"))
        ref = frames[VIDEO_FRAME].astype(np.float32) / 255
        mk = marked[VIDEO_FRAME].astype(np.float32) / 255
        err, _, _ = flip_evaluator.evaluate(np.ascontiguousarray(ref), np.ascontiguousarray(mk), "LDR", applyMagma=False)
        np.savez_compressed(DEST / "video" / f"{base.name}.npz", ref=frames[VIDEO_FRAME], marked=marked[VIDEO_FRAME],
                            flip=np.asarray(err, dtype=np.float32).squeeze())
        print(base.name, "done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["audio", "video"]
    if "audio" in which:
        audio()
    if "video" in which:
        video()
