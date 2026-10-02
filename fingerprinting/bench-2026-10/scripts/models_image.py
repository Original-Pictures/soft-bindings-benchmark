"""Image watermark adapters with one interface.

    a = ADAPTERS[name](**params); a.load(device)
    marked = a.embed(img, bits)          # img: float32 HxWx3 in [0,1]
    out = a.decode(img)                  # {"bits": uint8[n] | None, "native_detect": bool | None}

`bits` has length a.nbits (physical payload the model carries). TrustMark is run
as a deployed verifier would (BCH_5 ECC over its 100 physical bits): bit accuracy is
reported over the 100 raw decoder bits, and `native_detect` is the ECC decode.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import BENCH_WORK, WEIGHTS

SRC = BENCH_WORK / "src"


@dataclass
class Adapter:
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

    def embed(self, img: np.ndarray, bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, img: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def meta(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("name", "family", "nbits", "code_licence", "weights_licence",
                                              "deployable", "params", "note")}


def _pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.round(x * 255), 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ TrustMark
class TrustMarkAdapter(Adapter):
    def __init__(self, variant: str, strength: float) -> None:
        super().__init__(name=f"trustmark-{variant}@{strength:g}", family="TrustMark", nbits=100,
                         code_licence="MIT", weights_licence="MIT", deployable=True,
                         params={"variant": variant, "WM_STRENGTH": strength, "encoding": "BCH_5"},
                         note="TrustMark Q, strengths 1.0-1.5" if variant == "Q" else "")
        self.variant, self.strength = variant, strength

    def load(self, device: str) -> None:
        super().load(device)
        import trustmark
        from trustmark import TrustMark

        # Stage our sha256-pinned weights where the package expects them (it md5-checks them).
        pkg_models = Path(trustmark.__file__).parent / "models"
        pkg_models.mkdir(exist_ok=True)
        for f in (WEIGHTS / "trustmark").glob(f"*_{self.variant}.*"):
            if not (pkg_models / f.name).exists():
                shutil.copy(f, pkg_models / f.name)
        self.tm = TrustMark(verbose=False, model_type=self.variant, encoding_type=TrustMark.Encoding.BCH_5,
                            device=device, loadRemover=False)
        self.capacity = self.tm.schemaCapacity()

    def _codeword(self, bits: np.ndarray) -> tuple[str, np.ndarray]:
        data = "".join(str(int(b)) for b in bits[: self.capacity])
        return data, np.asarray(self.tm.ecc.encode_binary([data])).reshape(-1)[: self.nbits].astype(np.uint8)

    def physical_bits(self, bits: np.ndarray) -> np.ndarray:
        return self._codeword(bits)[1]

    def embed(self, img: np.ndarray, bits: np.ndarray) -> np.ndarray:
        data, _ = self._codeword(bits)
        out = self.tm.encode(_pil(img), data, MODE="binary", WM_STRENGTH=self.strength)
        return np.asarray(out, dtype=np.float32) / 255.0

    def decode(self, img: np.ndarray) -> dict[str, Any]:
        import torch
        from torchvision import transforms

        pil = _pil(img)
        sub = self.tm.get_the_image_for_processing(pil).resize(
            (self.tm.model_resolution_dec, self.tm.model_resolution_dec), Image.BILINEAR)
        t = transforms.ToTensor()(sub).unsqueeze(0).to(self.tm.decoder.device) * 2.0 - 1.0
        with torch.no_grad():
            raw = (self.tm.decoder.decoder(t) > 0).cpu().numpy().astype(np.uint8)
        secret, present, _ = self.tm.ecc.decode_bitstream(raw.astype(bool), "binary")[0]
        return {"bits": raw.reshape(-1), "native_detect": bool(present), "native_payload": secret}


# ------------------------------------------------------------------ Meta Seal family
class MetaSealAdapter(Adapter):
    CARDS = {"videoseal": ("videoseal_1.0", "videoseal/y_256b_img.pth", 256),
             "pixelseal": ("pixelseal", "videoseal/pixelseal.pth", 256),
             "chunkyseal": ("chunkyseal", "videoseal/chunkyseal.pth", 1024)}

    def __init__(self, model: str, scaling_w: float | None = None) -> None:
        card, rel, nbits = self.CARDS[model]
        label = {"videoseal": "VideoSeal-1.0", "pixelseal": "PixelSeal", "chunkyseal": "ChunkySeal"}[model]
        super().__init__(name=f"{label}@{scaling_w:g}" if scaling_w is not None else f"{label}@default",
                         family=label, nbits=nbits, code_licence="MIT", weights_licence="MIT", deployable=True,
                         params={"card": card, "scaling_w": scaling_w, "lowres_attenuation": True},
                         note="Video Seal y_256b_img; image mode here" if model == "videoseal" else "")
        self.model, self.card, self.rel, self.scaling_w = model, card, rel, scaling_w

    def load(self, device: str) -> None:
        super().load(device)
        from omegaconf import OmegaConf
        import videoseal
        from videoseal.utils.cfg import setup_model_from_model_card

        cards = Path(videoseal.__file__).parent / "cards"
        cfg = OmegaConf.load(cards / f"{self.card}.yaml")
        cfg.checkpoint_path = str(WEIGHTS / self.rel)
        tmp = Path(tempfile.mkdtemp()) / f"{self.card}.yaml"
        OmegaConf.save(cfg, tmp)
        import torch

        cwd = os.getcwd()
        os.chdir(Path(videoseal.__file__).parent.parent)  # card-relative attenuation config paths
        # ChunkySeal's 13 GB checkpoint also carries optimizer state; mmap keeps host RAM
        # to the 7 GB of model weights actually used (a plain load was OOM-killed on 16 GB).
        real_load = torch.load
        torch.load = lambda *a, **k: real_load(*a, **{**k, "mmap": True})  # type: ignore[assignment]
        try:
            self.m = setup_model_from_model_card(tmp).eval().to(device)
        finally:
            torch.load = real_load  # type: ignore[assignment]
            os.chdir(cwd)
        if self.scaling_w is not None:
            self.m.blender.scaling_w = self.scaling_w
        self.params["scaling_w_effective"] = float(self.m.blender.scaling_w)

    def embed(self, img: np.ndarray, bits: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(img.transpose(2, 0, 1)).float()[None].to(self.device)
        msg = torch.from_numpy(bits.astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            out = self.m.embed(x, msgs=msg, is_video=False, lowres_attenuation=True)["imgs_w"]
        return out[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    def decode(self, img: np.ndarray) -> dict[str, Any]:
        import torch

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()[None].to(self.device)
        with torch.no_grad():
            preds = self.m.detect(x, is_video=False)["preds"]
        logits = preds[0, 1:].float().cpu().numpy()
        return {"bits": (logits > 0).astype(np.uint8), "logits": logits}


# ------------------------------------------------------------------ WAM (MIT SA-1B weights)
class WAMAdapter(Adapter):
    def __init__(self, scaling_w: float | None = None) -> None:
        super().__init__(name=f"WAM-MIT@{scaling_w:g}" if scaling_w else "WAM-MIT@default", family="WAM", nbits=32,
                         code_licence="MIT", weights_licence="MIT (wam_mit.pth, SA-1B); COCO weights CC-BY-NC excluded",
                         deployable=True, params={"scaling_w": scaling_w})
        self.scaling_w = scaling_w

    def load(self, device: str) -> None:
        super().load(device)
        root = SRC / "watermark-anything"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        cwd = os.getcwd()
        os.chdir(root)
        try:
            from notebooks.inference_utils import load_model_from_checkpoint  # type: ignore

            self.m = load_model_from_checkpoint(str(root / "checkpoints/params.json"), str(WEIGHTS / "wam/wam_mit.pth")).to(device).eval()
        finally:
            os.chdir(cwd)
        if self.scaling_w is not None:
            self.m.scaling_w = self.scaling_w
        self.params["scaling_w_effective"] = float(self.m.scaling_w)

    def _norm(self, img: np.ndarray):
        import torch
        from watermark_anything.data.transforms import normalize_img  # type: ignore

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()[None].to(self.device)
        return normalize_img(x)

    def embed(self, img: np.ndarray, bits: np.ndarray) -> np.ndarray:
        import torch
        from watermark_anything.data.transforms import unnormalize_img  # type: ignore

        msg = torch.from_numpy(bits.astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            out = self.m.embed(self._norm(img), msg)["imgs_w"]
        return unnormalize_img(out)[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    def decode(self, img: np.ndarray) -> dict[str, Any]:
        import torch
        from watermark_anything.data.metrics import msg_predict_inference  # type: ignore

        with torch.no_grad():
            preds = self.m.detect(self._norm(img))["preds"]
            mask = torch.sigmoid(preds[:, 0])
            msg = msg_predict_inference(preds[:, 1:], mask)
        return {"bits": msg[0].cpu().numpy().astype(np.uint8), "native_score": float(mask.mean().item())}


# ------------------------------------------------------------------ invisible-watermark
class ImWatermarkAdapter(Adapter):
    def __init__(self, method: str) -> None:
        nbits = 32 if method == "rivaGan" else 64
        riva = method == "rivaGan"
        super().__init__(name={"dwtDct": "DWT-DCT", "dwtDctSvd": "DWT-DCT-SVD", "rivaGan": "RivaGAN"}[method],
                         family="invisible-watermark" if not riva else "RivaGAN", nbits=nbits, code_licence="MIT",
                         weights_licence="n/a (classical)" if not riva else "UNCLEAR: bundled ONNX, provenance undocumented",
                         deployable=not riva, params={"method": method},
                         note="reference only - not deployable (weights licence unclear)" if riva else "")
        self.method = method

    def load(self, device: str) -> None:
        super().load(device)
        from imwatermark import WatermarkDecoder, WatermarkEncoder

        if self.method == "rivaGan":
            WatermarkEncoder.loadModel()
        self._E, self._D = WatermarkEncoder, WatermarkDecoder

    def embed(self, img: np.ndarray, bits: np.ndarray) -> np.ndarray:
        enc = self._E()
        enc.set_watermark("bits", [int(b) for b in bits])
        bgr = np.clip(np.round(img[:, :, ::-1] * 255), 0, 255).astype(np.uint8)
        out = enc.encode(np.ascontiguousarray(bgr), self.method)
        return out[:, :, ::-1].astype(np.float32) / 255.0

    def decode(self, img: np.ndarray) -> dict[str, Any]:
        dec = self._D("bits", self.nbits)
        bgr = np.ascontiguousarray(np.clip(np.round(img[:, :, ::-1] * 255), 0, 255).astype(np.uint8))
        try:
            bits = np.asarray(dec.decode(bgr, self.method), dtype=np.uint8).reshape(-1)
        except Exception:
            bits = None
        return {"bits": bits}


# ------------------------------------------------------------------ InvisMark (reference only)
class InvisMarkAdapter(Adapter):
    """microsoft/InvisMark 'paper.ckpt' (100 bits, no ECC). Mirrors train.Watermark._encode/_decode
    without its training-only members (LPIPS-VGG, discriminator, SummaryWriter)."""

    def __init__(self) -> None:
        super().__init__(name="InvisMark", family="InvisMark", nbits=100, code_licence="MIT",
                         weights_licence="NONE STATED (OneDrive ckpt_paper.zip; no licence file)", deployable=False,
                         params={"checkpoint": "paper.ckpt", "enc_mode": "uuid (no ECC)"},
                         note="reference only - not deployable (no weight licence)")

    def load(self, device: str) -> None:
        super().load(device)
        import torch

        root = SRC / "InvisMark"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import model as im_model  # type: ignore

        ckpt = WEIGHTS / "invismark" / "paper.ckpt"
        if not ckpt.exists():
            import zipfile

            with zipfile.ZipFile(WEIGHTS / "invismark" / "ckpt_paper.zip") as z:
                name = next(n for n in z.namelist() if n.endswith("paper.ckpt"))
                ckpt.write_bytes(z.read(name))
        # The checkpoint pickles its own config object (trusted-by-hash: sha256 pinned in weights.lock.json).
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = sd["config"]
        self.enc = im_model.Encoder(self.cfg).to(device).eval()
        self.dec = im_model.Extractor(self.cfg).to(device).eval()
        self.enc.load_state_dict(sd["encoder_state_dict"])
        self.dec.load_state_dict(sd["decoder_state_dict"])
        self.shape = tuple(self.cfg.image_shape)

    def embed(self, img: np.ndarray, bits: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(self.device) * 2 - 1
        s = torch.from_numpy(bits.astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            small = F.interpolate(x, size=self.shape, mode="bilinear", antialias=True, align_corners=False)
            diff = self.enc(small, s) - small
            out = torch.clamp(x + F.interpolate(diff, size=x.shape[-2:], mode="bilinear", align_corners=False), -1, 1)
        return ((out[0] + 1) / 2).clamp(0, 1).cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    def decode(self, img: np.ndarray) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float().to(self.device) * 2 - 1
        with torch.no_grad():
            p = self.dec(F.interpolate(x, size=self.shape, mode="bilinear", antialias=True, align_corners=False))
        return {"bits": (p[0] > 0.5).cpu().numpy().astype(np.uint8)}


# ------------------------------------------------------------------ registry
def image_configs(profile: str = "full") -> list[Adapter]:
    cfgs: list[Adapter] = []
    for s in (1.0, 1.2, 1.3, 1.4, 1.5):
        cfgs.append(TrustMarkAdapter("Q", s))
    for v in ("P",):
        for s in (0.8, 1.0, 1.5):
            cfgs.append(TrustMarkAdapter(v, s))
    for v in ("B", "C"):
        cfgs.append(TrustMarkAdapter(v, 1.0))
    for sw in (None, 0.1, 0.4):
        cfgs.append(MetaSealAdapter("pixelseal", sw))
    for sw in (None, 0.1, 0.4):
        cfgs.append(MetaSealAdapter("videoseal", sw))
    cfgs.append(MetaSealAdapter("chunkyseal", None))
    cfgs.append(WAMAdapter(None))
    cfgs.append(WAMAdapter(1.0))
    cfgs += [ImWatermarkAdapter("dwtDct"), ImWatermarkAdapter("dwtDctSvd"), ImWatermarkAdapter("rivaGan")]
    if (WEIGHTS / "invismark").exists():
        cfgs.append(InvisMarkAdapter())
    # Added after the first pass: PixelSeal 0.1 missed the JPEG-75 gate (0.907) and the
    # default 0.2 passed with margin, so the knee lies between them.
    cfgs += [MetaSealAdapter("pixelseal", 0.15), MetaSealAdapter("pixelseal", 0.125)]
    if profile == "smoke":
        keep = {"trustmark-Q@1.2", "trustmark-P@1", "PixelSeal@default", "VideoSeal-1.0@default", "WAM-MIT@default",
                "DWT-DCT", "DWT-DCT-SVD", "RivaGAN", "trustmark-B@1", "trustmark-C@1"}
        cfgs = [c for c in cfgs if c.name in keep]
    return cfgs
