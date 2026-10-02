"""Audio watermark adapters.

    a.load(device); y_w = a.embed(y, sr, bits); d = a.decode(y_w, sr)
    d = {"bits": uint8[n] | None, "native_detect": bool | None, "native_score": float | None}

16 kHz-native models (AudioSeal, WavMark) are applied the way a >16 kHz pipeline
must use them without discarding the source band: the watermark RESIDUAL is
computed at 16 kHz and upsampled onto the untouched full-rate signal. Perth and
SilentCipher resample the whole signal internally (their own API); that band
loss is part of what their quality numbers show.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common import BENCH_WORK, WEIGHTS


def _rs(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32)
    import soxr

    return soxr.resample(y.astype(np.float32), sr_in, sr_out, quality="VHQ").astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


@dataclass
class AudioAdapter:
    name: str
    family: str
    nbits: int
    code_licence: str
    weights_licence: str
    deployable: bool
    params: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def load(self, device: str) -> None:
        self.device = device

    def meta(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", "family", "nbits", "code_licence", "weights_licence",
                                              "deployable", "params", "note")}


class AudioSealAdapter(AudioAdapter):
    def __init__(self, variant: str, alpha: float) -> None:
        super().__init__(name=f"AudioSeal-{variant}@{alpha:g}", family="AudioSeal", nbits=16, code_licence="MIT",
                         weights_licence="MIT (HF facebook/audioseal @3c19eba)", deployable=True,
                         params={"variant": variant, "alpha": alpha, "detect_frames_threshold": 0.8},
                         note="streaming variant; blind detection at detected-frames >= 0.80" if variant == "streaming" else "")
        self.variant, self.alpha = variant, alpha

    def load(self, device: str) -> None:
        super().load(device)
        import torch
        from audioseal import AudioSeal

        g = WEIGHTS / "audioseal" / f"generator_{self.variant}.pth"
        d = WEIGHTS / "audioseal" / f"detector_{self.variant}.pth"
        self.gen = AudioSeal.load_generator(str(g), nbits=16, device=torch.device(device)).eval()
        self.det = AudioSeal.load_detector(str(d), nbits=16, device=torch.device(device)).eval()

    def embed(self, y: np.ndarray, sr: int, bits: np.ndarray) -> np.ndarray:
        import torch

        x16 = _rs(y, sr, 16000)
        t = torch.from_numpy(x16)[None, None].to(self.device)
        msg = torch.from_numpy(bits.astype(np.int64))[None].to(self.device)
        with torch.no_grad():
            wm = self.gen.get_watermark(t, 16000, message=msg)[0, 0].cpu().numpy()
        return (y + self.alpha * _fit(_rs(wm, 16000, sr), len(y))).astype(np.float32)

    def decode(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        import torch

        t = torch.from_numpy(_rs(y, sr, 16000))[None, None].to(self.device)
        with torch.no_grad():
            prob, msg = self.det.detect_watermark(t, 16000, message_threshold=0.5, detection_threshold=0.5)
        p = float(prob.reshape(-1)[0].item())
        return {"bits": msg.reshape(-1).cpu().numpy().astype(np.uint8), "native_detect": p >= 0.8, "native_score": p}


class WavMarkAdapter(AudioAdapter):
    def __init__(self) -> None:
        super().__init__(name="WavMark", family="WavMark", nbits=16, code_licence="MIT",
                         weights_licence="MIT (HF M4869/WavMark)", deployable=True,
                         params={"pattern_bits": 16, "payload_bits": 16}, note="16 kHz model; 1 s segments")

    def load(self, device: str) -> None:
        super().load(device)
        import wavmark

        self.wm = wavmark
        self.model = wavmark.load_model(str(WEIGHTS / "wavmark" / "wavmark.model.pkl")).to(device).eval()

    def embed(self, y: np.ndarray, sr: int, bits: np.ndarray) -> np.ndarray:
        x16 = _rs(y, sr, 16000)
        marked, _ = self.wm.encode_watermark(self.model, x16, bits.astype(int), show_progress=False)
        resid = np.asarray(marked, dtype=np.float32) - x16[: len(marked)]
        return (y + _fit(_rs(resid, 16000, sr), len(y))).astype(np.float32)

    def decode(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        payload, info = self.wm.decode_watermark(self.model, _rs(y, sr, 16000), show_progress=False)
        if payload is None:
            return {"bits": None, "native_detect": False}
        return {"bits": np.asarray(payload, dtype=np.uint8), "native_detect": True}


class PerthAdapter(AudioAdapter):
    def __init__(self) -> None:
        super().__init__(name="Perth", family="Perth", nbits=0, code_licence="MIT",
                         weights_licence="MIT (bundled in resemble-ai/Perth; no separate weight licence)",
                         deployable=True, params={"model": "perth_net_250000 implicit", "internal_sr": 32000},
                         note="presence-only (no payload); resamples the whole signal to 32 kHz internally")

    def load(self, device: str) -> None:
        super().load(device)
        import perth

        self.w = perth.PerthImplicitWatermarker(device=device)

    def embed(self, y: np.ndarray, sr: int, bits: np.ndarray) -> np.ndarray:
        return _fit(np.asarray(self.w.apply_watermark(y, sample_rate=sr), dtype=np.float32), len(y))

    def decode(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        v = np.asarray(self.w.get_watermark(y, sample_rate=sr, round=False), dtype=np.float32)
        score = float(v.mean())
        return {"bits": None, "native_detect": score >= 0.5, "native_score": score}


class SilentCipherAdapter(AudioAdapter):
    def __init__(self, model_type: str) -> None:
        super().__init__(name=f"SilentCipher-{model_type}", family="SilentCipher", nbits=40, code_licence="MIT",
                         weights_licence="UNSTATED (GitHub release asset / HF card has no licence field)",
                         deployable=False, params={"model_type": model_type},
                         note="reference only - not deployable until Sony states a weight licence")
        self.model_type = model_type

    def load(self, device: str) -> None:
        super().load(device)
        src = BENCH_WORK / "src" / "silentcipher" / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import silentcipher

        d = WEIGHTS / "silentcipher" / "Models" / ("44_1_khz/73999_iteration" if self.model_type == "44.1k" else "16_khz/97561_iteration")
        self.m = silentcipher.get_model(model_type=self.model_type, ckpt_path=str(d), config_path=str(d / "hparams.yaml"), device=device)

    def embed(self, y: np.ndarray, sr: int, bits: np.ndarray) -> np.ndarray:
        msg = [int("".join(str(int(b)) for b in bits[i * 8:(i + 1) * 8]), 2) for i in range(5)]
        out, _ = self.m.encode_wav(y.astype(np.float32), sr, msg, calc_sdr=False)
        return _fit(np.asarray(out, dtype=np.float32).reshape(-1), len(y))

    def decode(self, y: np.ndarray, sr: int) -> dict[str, Any]:
        r = self.m.decode_wav(y.astype(np.float32), sr, phase_shift_decoding=False)
        if not r.get("status") or not r.get("messages"):
            return {"bits": None, "native_detect": False, "native_score": 0.0}
        msg = r["messages"][0]
        bits = np.array([int(c) for v in msg[:5] for c in f"{int(v) & 0xFF:08b}"], dtype=np.uint8)
        conf = float(r["confidences"][0]) if r.get("confidences") else float("nan")
        return {"bits": bits if len(bits) == 40 else None, "native_detect": conf >= 0.5, "native_score": conf}


def audio_configs(profile: str = "full") -> list[AudioAdapter]:
    cfgs: list[AudioAdapter] = [AudioSealAdapter("streaming", a) for a in (0.5, 1.0, 1.1)]
    cfgs.append(AudioSealAdapter("base", 1.0))
    cfgs += [WavMarkAdapter(), PerthAdapter(), SilentCipherAdapter("44.1k"), SilentCipherAdapter("16k")]
    if profile == "smoke":
        cfgs = [c for c in cfgs if c.name in {"AudioSeal-streaming@1", "WavMark", "Perth", "SilentCipher-44.1k"}]
    return cfgs
