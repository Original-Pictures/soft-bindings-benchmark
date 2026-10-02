"""Image fingerprint adapters behind one interface.

    fp = IMAGE_METHODS[name](); fp.load(device)
    d = fp.extract([img, ...])      # img: float32 HxWx3 in [0,1]  ->  (N, dim) array
    s = fp.score(q, r)              # (Nq, dim) x (Nr, dim) -> (Nq, Nr), higher = more similar

Descriptors are either binary (`metric="hamming"`, uint8 0/1 of length `dim`; score is
1 - Hamming/dim), L2-normalised float (`metric="ip"`, score is cosine) or unnormalised
float (`metric="l2"`, score is -L2). Retrieval (faiss) and every track use only these
three forms, so a method's behaviour is defined entirely by its extractor.

Neural extractors resize the WHOLE image to the model's square input (no centre crop):
copy detection must see the borders a crop attack keeps, and a crop would hide
edits near the edge from the localization track. SSCD's own evaluation uses the same
square resize for its "disc" models.

Tier "P" = code and weights under MIT/BSD/Apache (product-eligible after legal review);
tier "R" = reference only (weights unlicensed or trained on non-commercial data).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

from common import BENCH_WORK, SEED, to_u8

SRC = BENCH_WORK / "src"


@dataclass
class Fingerprinter:
    name: str
    family: str
    metric: str  # hamming | ip | l2
    dim: int
    tier: str  # P | R
    code_licence: str
    weights_licence: str
    gpu: bool = False
    note: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    batch: int = 64

    def load(self, device: str = "cpu") -> None:
        self.device = device

    def extract(self, imgs: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    # Query-side variants (e.g. PDQ dihedral). Default: the descriptor itself.
    def extract_query(self, imgs: list[np.ndarray]) -> np.ndarray:
        return self.extract(imgs)

    @property
    def bits(self) -> int:
        """Storage per asset in bits (float32 descriptors count 32 bits per dimension)."""
        return self.dim if self.metric == "hamming" else 32 * self.dim

    def score(self, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return pairwise_score(q, r, self.metric)

    def meta(self) -> dict[str, Any]:
        keys = ("name", "family", "metric", "dim", "tier", "code_licence", "weights_licence", "gpu", "note", "params")
        return {**{k: getattr(self, k) for k in keys}, "bits": self.bits}


def pairwise_score(q: np.ndarray, r: np.ndarray, metric: str) -> np.ndarray:
    if metric == "hamming":
        qa, ra = q.astype(np.float32), r.astype(np.float32)
        # matches = q.r + (1-q).(1-r)
        same = qa @ ra.T + (1 - qa) @ (1 - ra).T
        return same / q.shape[1]
    if metric == "ip":
        return q.astype(np.float32) @ r.astype(np.float32).T
    if metric == "l2":
        d = (q ** 2).sum(1)[:, None] + (r ** 2).sum(1)[None] - 2 * q @ r.T
        return -np.sqrt(np.maximum(d, 0))
    raise ValueError(metric)


def _pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(to_u8(x))


def _l2n(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


# ------------------------------------------------------------------ classical hashes
class PDQ(Fingerprinter):
    """Meta PDQ (ThreatExchange, BSD; pdqhash binding MIT). 256 bits.

    `dihedral=True` stores the plain hash for references but lets a query match through
    any of its 8 dihedral variants (the ThreatExchange-recommended way to catch flips
    and 90-degree rotations); score is the best of the 8.
    """

    def __init__(self, dihedral: bool = False) -> None:
        super().__init__(name="PDQ-dihedral" if dihedral else "PDQ", family="PDQ", metric="hamming", dim=256,
                         tier="P", code_licence="BSD (ThreatExchange); pdqhash binding MIT", weights_licence="n/a",
                         params={"dihedral": dihedral}, note="query matched via 8 dihedral variants" if dihedral else "")
        self.dihedral = dihedral

    def extract(self, imgs):
        import pdqhash

        return np.stack([pdqhash.compute(to_u8(x))[0].astype(np.uint8) for x in imgs])

    def extract_query(self, imgs):
        if not self.dihedral:
            return self.extract(imgs)
        import pdqhash

        # (N, 8, 256): dihedral variants per query
        return np.stack([np.stack(pdqhash.compute_dihedral(to_u8(x))[0]).astype(np.uint8) for x in imgs])

    def score(self, q, r):
        if q.ndim == 3:
            return np.max(np.stack([pairwise_score(q[:, k], r, "hamming") for k in range(q.shape[1])]), axis=0)
        return pairwise_score(q, r, "hamming")


class ImageHash(Fingerprinter):
    """imagehash (BSD-2): aHash, dHash, pHash, wHash."""

    FN = {"aHash": "average_hash", "dHash": "dhash", "pHash": "phash", "wHash": "whash"}

    def __init__(self, kind: str, hash_size: int = 8) -> None:
        n = hash_size * hash_size
        super().__init__(name=f"{kind}-{n}", family=kind, metric="hamming", dim=n, tier="P",
                         code_licence="BSD-2-Clause (imagehash)", weights_licence="n/a", params={"hash_size": hash_size})
        self.kind, self.hash_size = kind, hash_size

    def extract(self, imgs):
        import imagehash

        fn = getattr(imagehash, self.FN[self.kind])
        return np.stack([fn(_pil(x), hash_size=self.hash_size).hash.reshape(-1).astype(np.uint8) for x in imgs])


class CvImgHash(Fingerprinter):
    """OpenCV contrib img_hash (Apache-2.0)."""

    SPEC = {  # name: (factory, metric, dim)
        "BlockMean": ("BlockMeanHash_create", "hamming", 256),
        "MarrHildreth": ("MarrHildrethHash_create", "hamming", 576),
        "ColorMoment": ("ColorMomentHash_create", "l2", 42),
    }

    def __init__(self, kind: str) -> None:
        _, metric, dim = self.SPEC[kind]
        super().__init__(name=kind, family="OpenCV img_hash", metric=metric, dim=dim, tier="P",
                         code_licence="Apache-2.0 (opencv_contrib)", weights_licence="n/a")
        self.kind = kind

    def load(self, device="cpu"):
        import cv2

        super().load(device)
        self.h = getattr(cv2.img_hash, self.SPEC[self.kind][0])()

    def extract(self, imgs):
        import cv2

        out = []
        for x in imgs:
            bgr = cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2BGR)
            v = self.h.compute(bgr).reshape(-1)
            out.append(np.unpackbits(v)[: self.dim] if self.metric == "hamming" else v.astype(np.float32))
        return np.stack(out)


class Blockhash(Fingerprinter):
    """Blockhash (commonsmachinery/blockhash-python, MIT), 16x16 = 256 bits, precise method.

    The reference implementation loops over every pixel in Python (~1 s per 1 MP image);
    `_blocks` computes the same fractional-pixel block sums as two weight matrices
    (rows and columns), and tests/test_fingerprints.py checks it is bit-identical to the
    pinned reference on odd and even image sizes."""

    def __init__(self) -> None:
        super().__init__(name="Blockhash-256", family="Blockhash", metric="hamming", dim=256, tier="P",
                         code_licence="MIT", weights_licence="n/a", params={"bits": 16, "method": "precise"})

    @staticmethod
    def _weights(n: int, bits: int) -> np.ndarray:
        """(n, bits) matrix: pixel i's share of each block, exactly as blockhash() assigns it."""
        import math

        w = np.zeros((n, bits))
        size = n / bits
        if n % bits == 0:
            w[np.arange(n), np.arange(n) // (n // bits)] = 1.0
            return w
        for i in range(n):
            frac, whole = math.modf((i + 1) % size)
            first = int(i // size)
            if whole > 0 or (i + 1) == n:
                w[i, first] += 1.0
            else:
                w[i, first] += 1 - frac
                w[i, int(-(-i // size))] += frac
        return w

    @staticmethod
    def _bits(blocks: np.ndarray, pixels_per_block: float) -> np.ndarray:
        half = pixels_per_block * 256 * 3 / 2
        v = blocks.reshape(-1)
        out = np.zeros(len(v), np.uint8)
        band = len(v) // 4
        for i in range(4):
            seg = v[i * band:(i + 1) * band]
            m = float(np.median(seg))
            out[i * band:(i + 1) * band] = (seg > m) | ((np.abs(seg - m) < 1) & (m > half))
        return out

    def hash_u8(self, u8: np.ndarray, bits: int = 16) -> np.ndarray:
        h, w = u8.shape[:2]
        value = u8.astype(np.float64).sum(2)
        blocks = self._weights(h, bits).T @ value @ self._weights(w, bits)
        return self._bits(blocks, (w / bits) * (h / bits))

    def extract(self, imgs):
        return np.stack([self.hash_u8(to_u8(x)) for x in imgs])


class ISCCImage(Fingerprinter):
    """ISCC Image-Code v0 (ISO 24138; iscc-core/iscc-sdk, Apache-2.0). C2PA soft-binding `io.iscc.v0`.

    The pixels go through the SDK's own normalisation (EXIF transpose, alpha fill,
    uniform-border trim, grey, 32x32 bicubic), then gen_image_code_v0. The code body
    (after the ISCC header) is the similarity-preserving hash compared by Hamming.
    """

    def __init__(self, bits: int = 64) -> None:
        super().__init__(name=f"ISCC-Image-{bits}", family="ISCC", metric="hamming", dim=bits, tier="P",
                         code_licence="Apache-2.0", weights_licence="n/a", params={"bits": bits},
                         note="C2PA-registered io.iscc.v0" + ("" if bits == 64 else "; non-default length"))
        self.nbits = bits

    def extract(self, imgs):
        import iscc_core as ic
        import iscc_sdk as idk

        out = []
        for x in imgs:
            code = ic.gen_image_code_v0(list(idk.image_normalize(_pil(x))), bits=self.nbits)["iscc"]
            body = ic.Code(code).hash_bytes
            out.append(np.unpackbits(np.frombuffer(body, np.uint8))[: self.nbits])
        return np.stack(out)


# ------------------------------------------------------------------ neural global descriptors
class _Torch(Fingerprinter):
    size: int = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def _tensor(self, imgs):
        """Images (uint8 or float in [0,1], any size) -> normalised batch on self.device.
        Pixels go to the device first, so the antialiased resize runs there, not on the CPU."""
        import torch
        import torch.nn.functional as F

        xs = []
        for x in imgs:
            # numpy arrays or torch tensors already on the device (bench_image uploads each
            # chunk once and shares it across methods)
            t = x.to(self.device) if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            t = (t.float() / 255.0 if t.dtype == torch.uint8 else t.float()).permute(2, 0, 1)[None]
            xs.append(F.interpolate(t, size=(self.size, self.size), mode="bicubic", align_corners=False, antialias=True))
        t = torch.cat(xs).clamp(0, 1)
        m = torch.tensor(self.mean, device=t.device).view(1, 3, 1, 1)
        s = torch.tensor(self.std, device=t.device).view(1, 3, 1, 1)
        return (t - m) / s

    def forward(self, t):
        raise NotImplementedError

    def extract(self, imgs):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), self.batch):
                out.append(self.forward(self._tensor(imgs[i:i + self.batch])).float().cpu().numpy())
        return self.post(np.concatenate(out))

    def post(self, v: np.ndarray) -> np.ndarray:
        return _l2n(v)


class DINOv2(_Torch):
    """DINOv2 CLS token (facebookresearch/dinov2, Apache-2.0 code and weights)."""

    HF = {"S": ("facebook/dinov2-small", 384), "B": ("facebook/dinov2-base", 768)}

    def __init__(self, size_code: str = "S") -> None:
        repo, dim = self.HF[size_code]
        super().__init__(name=f"DINOv2-{size_code}", family="DINOv2", metric="ip", dim=dim, tier="P", gpu=True,
                         code_licence="Apache-2.0", weights_licence="Apache-2.0 (HF model card)", params={"hf": repo})
        self.repo = repo

    def load(self, device="cpu"):
        from transformers import AutoModel

        super().load(device)
        self.m = AutoModel.from_pretrained(self.repo).eval().to(device)

    def forward(self, t):
        return self.m(pixel_values=t).last_hidden_state[:, 0]


class KeyedLSH(DINOv2):
    """DINOv2-S projected by a secret Gaussian matrix and binarised (SimHash / sign-LSH).

    Key = seed of the projection. Product-eligible, compact (256 bits) and, unlike the
    raw float descriptor, not invertible by someone who does not hold the key; the
    security track tests exactly that claim.
    """

    def __init__(self, bits: int = 256, key: int = SEED) -> None:
        super().__init__("S")
        self.name, self.family, self.metric, self.dim = f"DINOv2-S-LSH{bits}", "DINOv2", "hamming", bits
        self.params = {**self.params, "bits": bits, "key": "secret seed (bench: SEED)"}
        self.note = "keyed sign random projection of DINOv2-S CLS"
        rng = np.random.default_rng(key)
        self.P = rng.standard_normal((384, bits)).astype(np.float32)

    def post(self, v):
        return (_l2n(v) @ self.P > 0).astype(np.uint8)


class OpenCLIP(_Torch):
    """OpenCLIP ViT-B/32 LAION-2B (open_clip MIT; checkpoint MIT per model card)."""

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self) -> None:
        super().__init__(name="OpenCLIP-B32", family="CLIP", metric="ip", dim=512, tier="P", gpu=True,
                         code_licence="MIT (open_clip)", weights_licence="MIT (laion/CLIP-ViT-B-32-laion2B-s34B-b79K)",
                         params={"arch": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"})

    def load(self, device="cpu"):
        import open_clip

        super().load(device)
        self.m = open_clip.create_model("ViT-B-32", pretrained="laion2b_s34b_b79k").eval().to(device)

    def forward(self, t):
        return self.m.encode_image(t)


class SSCD(_Torch):
    """SSCD (facebookresearch/sscd-copy-detection). Code MIT; the torchscript weights carry
    no separate licence and were trained on DISC21 (CC BY-NC 4.0 repo) -> reference tier."""

    FILES = {"mixup": ("sscd_disc_mixup.torchscript.pt", 512), "large": ("sscd_disc_large.torchscript.pt", 1024)}

    def __init__(self, variant: str = "mixup") -> None:
        fname, dim = self.FILES[variant]
        super().__init__(name=f"SSCD-{variant}", family="SSCD", metric="ip", dim=dim, tier="R", gpu=True,
                         code_licence="MIT", weights_licence="none stated; trained on DISC21 (CC BY-NC 4.0)",
                         params={"weights": fname, "input": 320})
        self.fname, self.size = fname, 320

    def load(self, device="cpu"):
        import torch

        super().load(device)
        self.m = torch.jit.load(str(BENCH_WORK / "weights" / "sscd" / self.fname), map_location="cpu").eval().to(device)

    def forward(self, t):
        return self.m(t)


class ISC21(_Torch):
    """ISC21 descriptor-track 1st place (lyakaap, isc_ft_v107). Code MIT; weights fine-tuned on DISC21."""

    def __init__(self) -> None:
        super().__init__(name="ISC21-1st", family="ISC21", metric="ip", dim=256, tier="R", gpu=True,
                         code_licence="MIT", weights_licence="MIT repo release; trained on DISC21 (CC BY-NC 4.0)",
                         params={"weights": "isc_ft_v107", "input": 512})
        self.batch = 32

    def load(self, device="cpu"):
        import timm
        from isc_feature_extractor import create_model

        super().load(device)
        # The package names the backbone "timm/<arch>", which timm>=1.0.2x reads as a Hub
        # id; the backbone is built without pretrained weights (the checkpoint overwrites
        # them), so the bare architecture name is equivalent.
        real = timm.create_model
        timm.create_model = lambda arch, *a, **k: real(arch.removeprefix("timm/"), *a, **k)
        try:
            self.m, pre = create_model(weight_name="isc_ft_v107", device=device,
                                       model_dir=str(BENCH_WORK / "weights" / "isc21"))
        finally:
            timm.create_model = real
        # The package preprocessor is Resize(512)+ToTensor+Normalize; mirror it on tensors.
        self.size = 512
        norm = [t for t in pre.transforms if t.__class__.__name__ == "Normalize"][0]
        self.mean, self.std = tuple(norm.mean), tuple(norm.std)

    def forward(self, t):
        return self.m(t)


class ISCCSemantic(Fingerprinter):
    """ISCC Semantic-Code Image (iscc-sci, Apache-2.0 code). ONNX model derived from the ISC21
    1st-place solution (weights hosted in iscc-binaries without a licence file) -> reference tier.
    Not part of ISO 24138 and not what the C2PA `io.iscc.v0` entry covers."""

    def __init__(self, bits: int = 256) -> None:
        # gpu=False: its PIL preprocessing dominates, so it runs in the CPU worker pool; the
        # ONNX session inside each worker still uses CUDA when available.
        super().__init__(name=f"ISCC-SCI-{bits}", family="ISCC", metric="hamming", dim=bits, tier="R", gpu=False,
                         code_licence="Apache-2.0", weights_licence="none stated; ISC21-derived (DISC21 CC BY-NC)",
                         params={"bits": bits})
        self.nbits = bits

    def load(self, device="cpu"):
        super().load(device)
        import onnxruntime as rt

        import iscc_sci as sci
        import iscc_sci.code_semantic_image as c

        # One intra-op thread per session when on CPU: the bench runs one session per worker
        # process, and ONNX Runtime's default (all cores per session) oversubscribes the host.
        if "CUDAExecutionProvider" not in rt.get_available_providers() or not self.device.startswith("cuda"):
            so = rt.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            c._model = rt.InferenceSession(sci.get_model(), sess_options=so, providers=["CPUExecutionProvider"])

    def extract(self, imgs):
        import iscc_sci as sci
        from loguru import logger

        logger.remove()  # iscc-sci logs every inference at DEBUG
        out = []
        for x in imgs:
            digest, _ = sci.soft_hash_image_semantic(sci.preprocess_image(_pil(x)), bits=self.nbits)
            out.append(np.unpackbits(np.frombuffer(digest, np.uint8))[: self.nbits])
        return np.stack(out)


class DINOHash(Fingerprinter):
    """DINOHash (proteus-photos, ICML 2025): adversarially fine-tuned DINOv2-S/14-reg, 96 bits.
    The repository has no licence file (all rights reserved) -> reference tier, research use only."""

    def __init__(self) -> None:
        super().__init__(name="DINOHash-96", family="DINOHash", metric="hamming", dim=96, tier="R", gpu=True,
                         code_licence="none (no LICENSE file)", weights_licence="none (no LICENSE file)",
                         params={"onnx": "dinov2_vits14_reg_96bit.onnx", "input": 224})

    def load(self, device="cpu"):
        import onnxruntime as ort

        super().load(device)
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(SRC / "dinohash" / "dinov2_vits14_reg_96bit.onnx"), providers=prov)
        self.inp = self.sess.get_inputs()[0].name

    def extract(self, imgs):
        # Preprocessing as in the repo's main.py: 224 bicubic, ImageNet normalisation.
        dev = self.device if self.device.startswith("cuda") else "cpu"
        t = _Torch._tensor(type("T", (), {"size": 224, "mean": _Torch.mean, "std": _Torch.std, "device": dev})(), imgs).cpu()
        out = []
        for i in range(len(t)):  # the exported graph has a fixed batch of 1
            v = self.sess.run(None, {self.inp: t[i:i + 1].numpy().astype(np.float32)})[0]
            out.append(v.reshape(1, -1))
        v = np.concatenate(out)
        return (v > 0).astype(np.uint8)[:, :96]


IMAGE_METHODS: dict[str, Callable[[], Fingerprinter]] = {
    "PDQ": lambda: PDQ(False),
    "PDQ-dihedral": lambda: PDQ(True),
    "aHash-64": lambda: ImageHash("aHash"),
    "dHash-64": lambda: ImageHash("dHash"),
    "pHash-64": lambda: ImageHash("pHash"),
    "pHash-256": lambda: ImageHash("pHash", 16),
    "wHash-64": lambda: ImageHash("wHash"),
    "BlockMean": lambda: CvImgHash("BlockMean"),
    "MarrHildreth": lambda: CvImgHash("MarrHildreth"),
    "ColorMoment": lambda: CvImgHash("ColorMoment"),
    "Blockhash-256": Blockhash,
    "ISCC-Image-64": lambda: ISCCImage(64),
    "ISCC-Image-256": lambda: ISCCImage(256),
    "DINOv2-S": lambda: DINOv2("S"),
    "DINOv2-B": lambda: DINOv2("B"),
    "DINOv2-S-LSH256": KeyedLSH,
    "OpenCLIP-B32": OpenCLIP,
    "SSCD-mixup": lambda: SSCD("mixup"),
    "SSCD-large": lambda: SSCD("large"),
    "ISC21-1st": ISC21,
    "DINOHash-96": DINOHash,
}

# Measured in the smoke run only. The ISCC semantic code binarises an ISC21-derived descriptor
# (evaluated directly as ISC21-1st) and its reference ONNX model runs at batch 1, about 3 s
# per image on one CPU thread and ~160 ms on the A10G with preprocessing, which does not fit
# the full run's budget of ~290k extractions.
SMOKE_ONLY_METHODS: dict[str, Callable[[], Fingerprinter]] = {
    "ISCC-SCI-256": lambda: ISCCSemantic(256),
}
