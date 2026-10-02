"""Security track: evasion, targeted collision (forgery), descriptor inversion, and the
audio/video counterparts.

    python bench_security.py {evasion,collision,inversion,audio,video,all} [--n N] [--device D]

Threat model. The adversary holds a registered asset (evasion: strip its provenance by
making the copy stop matching) or an unrelated image (collision: make it bind to a
registered asset, the counterfeit-listing case), may run the public fingerprint
extractors, and is bounded by an L-inf pixel budget eps (images, video frames) or a
perturbation-to-signal ratio (audio). Success is judged ONLY by the real extractor
(fingerprints_* adapters) on the uint8/int16-quantised result, at the threshold the
retrieval track calibrated for the method (results/<modality>/<m>.json, pair@1e-06,
falling back to query@0.01; each record says which it used). Gradients come from:
  * white-box PGD through the neural extractors (whole image resized differentiably);
  * surrogate PGD for DCT/mean hashes (hash_surrogates; the gradient source only);
  * transfer: a perturbation crafted on an ensemble (DINOv2-S + SSCD-mixup + OpenCLIP)
    and scored on every image method, including hashes the attacker never modelled.
Inversion asks what a leaked descriptor reveals: a decoder trained on (descriptor,
thumbnail) pairs of distractor images reconstructs 64x64 thumbnails of registered
images from their descriptors; a keyed LSH is tested with and without the key.

Outputs: $BENCH_RESULTS/security/{evasion,collision,inversion,audio,video}.json and
galleries under $BENCH_WORK/out/security/. Each run prints a runtime estimate for the
default N extrapolated from its own per-attack timings.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import hash_surrogates as hs
from analysis_stats import cp_interval
from common import CORPUS, DESC, OUT, RESULTS, host_info, load_rgb, rng_for, to_u8, write_json
from fingerprints_image import IMAGE_METHODS, KeyedLSH, _Torch, pairwise_score

SEC_OUT = OUT / "security"
EPS_EVASION = (2, 4, 8, 16)
EPS_COLLISION = (4, 8, 16)
WHITEBOX = ["DINOv2-S", "DINOv2-B", "OpenCLIP-B32", "SSCD-mixup", "SSCD-large", "ISC21-1st", "DINOv2-S-LSH256"]
ENSEMBLE = ["DINOv2-S", "SSCD-mixup", "OpenCLIP-B32"]
# Defaults sized for ~3 h on one A10G for all five experiments (see the smoke estimate).
DEFAULT_N = {"evasion": 100, "collision": 100, "audio": 60, "video": 30}


# ------------------------------------------------------------------ shared helpers
def manifest() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())


def threshold(modality: str, name: str, fallback: Callable[[], float] | None = None,
              label: str = "fallback:q0.999-nonmatching") -> tuple[float, str]:
    """Calibrated threshold from the retrieval track, or a conservative fallback."""
    f = RESULTS / modality / f"{name}.json"
    if f.exists():
        th = json.loads(f.read_text()).get("thresholds", {})
        for key in ("pair@1e-06", "query@0.01"):
            v = th.get(key)
            if v is not None and np.isfinite(v):
                return float(v), key
    if fallback is None:
        return float("nan"), "unavailable"
    return float(fallback()), label


def image_fallback(name: str) -> Callable[[], float]:
    """99.9th percentile of registered-vs-distractor scores: a stand-in for the calibrated
    threshold when the retrieval metrics have not been run (smoke runs only). Not the max:
    ABO holds colour variants of one product photo, which pin the max near a perfect match."""
    def f() -> float:
        r = np.load(DESC / "reg" / f"{name}.npy")[:200]
        d = np.load(DESC / "dist" / f"{name}.npy")[:2000]
        return float(np.quantile(pairwise_score(r, d, IMAGE_METHODS[name]().metric), 0.999)) + 1e-6
    return f


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


class Lpips:
    def __init__(self, device: str) -> None:
        import lpips

        self.m = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        """LPIPS at max side 512 (both images area-downscaled): the perturbation budget is
        what a viewer sees at screen size, and full-resolution AlexNet LPIPS dominated runtime."""
        def t(x):
            v = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
            s = 512 / max(v.shape[-2:])
            if s < 1:  # on CPU: MPS lacks non-divisible adaptive pooling
                v = F.interpolate(v, scale_factor=s, mode="area")
            return v.to(self.device) * 2 - 1
        with torch.no_grad():
            return float(self.m(t(a), t(b)).item())


def summarise(rows: list[dict], key: str = "success") -> dict:
    """Success rate over items the method matched BEFORE the attack (score_clean >= t);
    items already below threshold are counted in n_unmatched_clean, not as successes."""
    excluded = [r for r in rows if r.get("unmatched_clean")]
    rows = [r for r in rows if not r.get("unmatched_clean")]
    k = sum(bool(r[key]) for r in rows)
    n = len(rows)
    fin = lambda f: [r[f] for r in rows if r.get(f) is not None and np.isfinite(r[f])]
    return {"n": n, "n_unmatched_clean": len(excluded), "success": k, "rate": k / n if n else float("nan"),
            "ci": cp_interval(k, n),
            "psnr_mean": float(np.mean(fin("psnr"))) if fin("psnr") else float("nan"),
            "lpips_mean": float(np.mean(fin("lpips"))) if fin("lpips") else float("nan"),
            "score_drop_mean": float(np.mean([r["score_clean"] - r["score_adv"] for r in rows])) if rows else float("nan")}


def seed_of(*keys: object) -> int:
    """Reproducible seed (Python's hash() of strings is salted per process)."""
    return int(rng_for("security", *keys).integers(0, 2**31 - 1))


def to_t(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)


def from_t(t: torch.Tensor) -> np.ndarray:
    """Quantise to uint8 exactly as a saved image would be, back to float."""
    return to_u8(t.detach()[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()).astype(np.float32) / 255.0


# ------------------------------------------------------------------ differentiable views
class TorchView:
    """Differentiable embedding of a neural image method, mirroring its extractor:
    whole-image square resize (bicubic, antialiased), normalisation, forward, L2-norm.
    For DINOv2-S-LSH256 the output is the pre-sign projection (soft code)."""

    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.m: _Torch = IMAGE_METHODS[name]()
        self.m.load(device)
        self.device = device
        for p in getattr(self.m.m, "parameters", lambda: [])():
            p.requires_grad_(False)
        self.P = torch.from_numpy(self.m.P).to(device) if isinstance(self.m, KeyedLSH) else None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        s = self.m.size
        t = hs.resize(x, s, s, "bicubic").clamp(0, 1)
        mean = torch.tensor(self.m.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.m.std, device=x.device).view(1, 3, 1, 1)
        v = F.normalize(self.m.forward((t - mean) / std).float(), dim=1)
        return v @ self.P if self.P is not None else v


def neural_loss(view: TorchView, ref: torch.Tensor, sign: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """sign=+1: minimise similarity to ref (evasion); sign=-1: maximise it (collision)."""
    if view.P is not None:
        b = ref.sign()
        def lsh(x):
            z = view(x)
            return sign * torch.tanh(4 * z / (z.abs().mean() + 1e-6)).mul(b).mean()
        return lsh
    return lambda x: sign * (view(x) * ref).sum()


def hash_loss(name: str, ref_bits: np.ndarray, sign: float, device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    b = torch.from_numpy(ref_bits.astype(np.float32) * 2 - 1).to(device)[None]
    f = hs.SURROGATES[name]
    if name == "ISCC-Image-64":
        return lambda x: sign * (f(x, box=hs.trim_box(x)) * b).mean()
    return lambda x: sign * (f(x) * b).mean()


def pgd(x0: torch.Tensor, loss: Callable, eps: float, steps: int, seed: str,
        done: Callable[[np.ndarray], bool] | None = None, every: int = 5) -> tuple[np.ndarray, int]:
    """L-inf PGD (sign steps of eps/8, uniform random start), minimising `loss`. Stops early
    once `done` (a real-extractor check on the quantised image) holds."""
    g = torch.Generator(device="cpu").manual_seed(seed_of(seed))
    delta = ((torch.rand(x0.shape, generator=g) * 2 - 1) * eps).to(x0.device)
    step = eps / 8
    used = steps
    for i in range(steps):
        delta.requires_grad_(True)
        l = loss((x0 + delta).clamp(0, 1))
        (grad,) = torch.autograd.grad(l, delta)
        with torch.no_grad():
            delta = (delta - step * grad.sign()).clamp(-eps, eps)
            delta = (x0 + delta).clamp(0, 1) - x0
        if done is not None and (i + 1) % every == 0 and done(from_t(x0 + delta)):
            used = i + 1
            break
    return from_t(x0 + delta), used


class Real:
    """Real extractors, loaded once."""

    def __init__(self, device: str) -> None:
        self.device, self.cache = device, {}

    def __call__(self, name: str):
        if name not in self.cache:
            m = IMAGE_METHODS[name]()
            m.load(self.device if m.gpu else "cpu")
            self.cache[name] = m
        return self.cache[name]

    def score(self, name: str, x: np.ndarray, ref_desc: np.ndarray) -> float:
        m = self(name)
        return float(m.score(m.extract_query([x]), ref_desc[None] if ref_desc.ndim == 1 else ref_desc)[0, 0])


# ------------------------------------------------------------------ 1. evasion
def evasion(n: int, device: str, steps: int) -> dict:
    man = manifest()["image"]
    src = man["sets"]["pos"][:n]
    real, lp = Real(device), Lpips(device)
    ths = {m: threshold("image", m, image_fallback(m)) for m in IMAGE_METHODS if (DESC / "reg" / f"{m}.npy").exists()}
    out: dict[str, Any] = {"params": {"n": len(src), "eps_255": EPS_EVASION, "steps": steps, "step": "eps/8"},
                           "thresholds": ths, "whitebox": {}, "surrogate": {}, "transfer": {}}
    timing: dict[str, float] = {}
    imgs = {i: load_rgb(CORPUS / man["items"][i]["path"], 1024) for i in src}
    refs_cache: dict[tuple[str, str], np.ndarray] = {}

    def ref_of(name: str, i: str) -> np.ndarray:
        if (name, i) not in refs_cache:
            refs_cache[(name, i)] = real(name).extract([imgs[i]])[0]
        return refs_cache[(name, i)]

    for name in WHITEBOX + list(hs.SURROGATES):
        kind = "whitebox" if name in WHITEBOX else "surrogate"
        if name not in ths:
            continue
        t, _ = ths[name]
        view = TorchView(name, device) if kind == "whitebox" else None
        out[kind][name] = {}
        t0, runs = time.time(), 0
        for eps in EPS_EVASION:
            rows = []
            for i in src:
                x = imgs[i]
                ref = ref_of(name, i)
                x0 = to_t(x, device)
                if kind == "whitebox":
                    with torch.no_grad():
                        r = view(x0)[0]
                    loss = neural_loss(view, r, +1)
                else:
                    loss = hash_loss(name, ref, +1, device)
                done = lambda xa: real.score(name, xa, ref) < t
                xa, used = pgd(x0, loss, eps / 255, steps, f"ev:{name}:{eps}:{i}", done)
                s_adv, s_clean = real.score(name, xa, ref), real.score(name, x, ref)
                rows.append({"item": i, "success": s_adv < t, "unmatched_clean": s_clean < t, "score_clean": s_clean,
                             "score_adv": s_adv,
                             "steps": used, "psnr": psnr(xa, x), "lpips": lp(xa, x)})
                runs += 1
            out[kind][name][str(eps)] = {"summary": summarise(rows), "rows": rows}
            print(f"evasion {kind} {name} eps={eps}: {out[kind][name][str(eps)]['summary']['rate']:.2f}", flush=True)
        timing[name] = (time.time() - t0) / max(1, runs)
        if view is not None:
            del view
    # transfer from the ensemble to every method
    views = [TorchView(m, device) for m in ENSEMBLE if m in ths]
    t0, runs = time.time(), 0
    for eps in EPS_EVASION:
        per: dict[str, list] = {}
        for i in src:
            x = imgs[i]
            x0 = to_t(x, device)
            with torch.no_grad():
                refs = [v(x0)[0] for v in views]
            loss = lambda xx: sum((v(xx) * r).sum() for v, r in zip(views, refs)) / len(views)
            xa, _ = pgd(x0, loss, eps / 255, steps, f"tr:{eps}:{i}")
            q = {"psnr": psnr(xa, x), "lpips": lp(xa, x)}
            for name, (t, _) in ths.items():
                try:
                    ref = ref_of(name, i)
                    s0, s1 = real.score(name, x, ref), real.score(name, xa, ref)
                except Exception as e:  # a method that cannot run here is recorded, not hidden
                    per.setdefault(name, []).append({"item": i, "error": repr(e)[:200]})
                    continue
                per.setdefault(name, []).append({"item": i, "success": s1 < t, "unmatched_clean": s0 < t,
                                                 "score_clean": s0, "score_adv": s1, **q})
            runs += 1
        out["transfer"][str(eps)] = {m: {"summary": summarise([r for r in rows if "error" not in r]),
                                         "errors": sum("error" in r for r in rows)} for m, rows in per.items()}
        print(f"transfer eps={eps}: " + ", ".join(f"{m}={v['summary']['rate']:.2f}" for m, v in out["transfer"][str(eps)].items()), flush=True)
    timing["transfer"] = (time.time() - t0) / max(1, runs)
    out["seconds_per_attack"] = timing
    out["estimate_default_n_hours"] = _estimate(timing, DEFAULT_N["evasion"], len(EPS_EVASION))
    return out


def _estimate(per_attack: dict[str, float], n: int, n_eps: int) -> float:
    return float(sum(per_attack.values()) * n * n_eps / 3600)


# ------------------------------------------------------------------ 2. collision
def collision(n: int, device: str, steps: int) -> dict:
    man = manifest()["image"]
    srcs, tgts = man["sets"]["neg"], man["sets"]["pos"]
    pairs = [(srcs[j % len(srcs)], tgts[(j * 7 + 3) % len(tgts)]) for j in range(n)]
    real, lp = Real(device), Lpips(device)
    reg_ids = json.loads((DESC / "reg" / "ids.json").read_text())
    idx_ids = reg_ids + json.loads((DESC / "dist" / "ids.json").read_text())
    out: dict[str, Any] = {"params": {"n": n, "eps_255": EPS_COLLISION, "steps": steps}, "methods": {}}
    timing = {}
    for name in WHITEBOX + list(hs.SURROGATES):
        if not (DESC / "reg" / f"{name}.npy").exists():
            continue
        t, tsrc = threshold("image", name, image_fallback(name))
        index = np.concatenate([np.load(DESC / "reg" / f"{name}.npy"), np.load(DESC / "dist" / f"{name}.npy")])
        m = real(name)
        view = TorchView(name, device) if name in WHITEBOX else None
        rec = {"threshold": t, "threshold_source": tsrc}
        t0, runs = time.time(), 0
        for eps in EPS_COLLISION:
            rows = []
            for a, b in pairs:
                xs = load_rgb(CORPUS / man["items"][a]["path"], 1024)
                xt = load_rgb(CORPUS / man["items"][b]["path"], 1024)
                tgt = m.extract([xt])[0]
                x0 = to_t(xs, device)
                if view is not None:
                    with torch.no_grad():
                        r = view(to_t(xt, device))[0]
                    loss = neural_loss(view, r, -1)
                else:
                    loss = hash_loss(name, tgt, -1, device)
                tj = idx_ids.index(b)

                def hit(xa):
                    s = m.score(m.extract_query([xa]), index)[0]
                    return int(np.argmax(s)) == tj and s[tj] >= t

                xa, used = pgd(x0, loss, eps / 255, steps, f"co:{name}:{eps}:{a}", hit)
                s = m.score(m.extract_query([xa]), index)[0]
                rows.append({"src": a, "target": b, "success": bool(int(np.argmax(s)) == tj and s[tj] >= t),
                             "score_clean": float(m.score(m.extract_query([xs]), tgt[None])[0, 0]), "score_adv": float(s[tj]),
                             "target_rank": int((s > s[tj]).sum()), "steps": used, "psnr": psnr(xa, xs), "lpips": lp(xa, xs)})
                runs += 1
            summ = summarise(rows)
            summ["score_gain_mean"] = -summ.pop("score_drop_mean")
            rec[str(eps)] = {"summary": summ, "rows": rows}
            print(f"collision {name} eps={eps}: {summ['rate']:.2f}", flush=True)
        timing[name] = (time.time() - t0) / max(1, runs)
        out["methods"][name] = rec
    out["seconds_per_attack"] = timing
    out["estimate_default_n_hours"] = _estimate(timing, DEFAULT_N["collision"], len(EPS_COLLISION))
    return out


# ------------------------------------------------------------------ 3. inversion
INV_METHODS = ["PDQ", "pHash-64", "ISCC-Image-64", "DINOv2-S", "SSCD-mixup", "OpenCLIP-B32", "DINOv2-S-LSH256"]


class Decoder(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(dim, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 4 * 4 * 512))
        chans = [512, 256, 128, 64, 32]
        self.up = torch.nn.Sequential(*[torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="nearest"), torch.nn.Conv2d(chans[i], chans[i + 1], 3, padding=1),
            torch.nn.GroupNorm(8, chans[i + 1]), torch.nn.GELU()) for i in range(4)])
        self.out = torch.nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, z):
        return torch.sigmoid(self.out(self.up(self.fc(z).view(-1, 512, 4, 4))))


def thumbs(ids: list[str], man: dict, tag: str) -> np.ndarray:
    """64x64 thumbnails (area resize of the stored asset), cached."""
    SEC_OUT.mkdir(parents=True, exist_ok=True)
    f = SEC_OUT / f"thumbs_{tag}_{len(ids)}.npy"
    if f.exists():
        return np.load(f)
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image

    def one(i):
        with Image.open(CORPUS / man["items"][i]["path"]) as im:
            return np.asarray(im.convert("RGB").resize((64, 64), Image.BOX), np.uint8)

    with ThreadPoolExecutor(16) as ex:
        arr = np.stack(list(ex.map(one, ids)))
    np.save(f, arr)
    return arr


def _codes(name: str, set_name: str, wrong_key: bool = False) -> np.ndarray:
    if wrong_key:
        v = np.load(DESC / set_name / "DINOv2-S.npy")
        P = np.random.default_rng(12345).standard_normal((384, 256)).astype(np.float32)
        return (v @ P > 0).astype(np.float32) * 2 - 1
    d = np.load(DESC / set_name / f"{name}.npy").astype(np.float32)
    return d * 2 - 1 if IMAGE_METHODS[name]().metric == "hamming" else d


def inversion(device: str, epochs: int) -> dict:
    import lpips
    from skimage.metrics import structural_similarity

    man = manifest()["image"]
    tr_ids = json.loads((DESC / "dist" / "ids.json").read_text())
    te_ids = json.loads((DESC / "reg" / "ids.json").read_text())
    Xtr = torch.from_numpy(thumbs(tr_ids, man, "dist")).permute(0, 3, 1, 2).float() / 255
    Xte = torch.from_numpy(thumbs(te_ids, man, "reg")).permute(0, 3, 1, 2).float() / 255
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    dino = TorchView("DINOv2-S", device)
    with torch.no_grad():
        orig_emb = np.load(DESC / "reg" / "DINOv2-S.npy")  # full-resolution registered descriptors

    def evaluate(rec: torch.Tensor) -> dict:
        r = rec.clamp(0, 1)
        with torch.no_grad():
            l = torch.cat([lp(r[i:i + 64].to(device) * 2 - 1, Xte[i:i + 64].to(device) * 2 - 1).flatten().cpu()
                           for i in range(0, len(r), 64)])
            emb = torch.cat([dino(r[i:i + 64].to(device)).cpu() for i in range(0, len(r), 64)]).numpy()
        ssim = [structural_similarity(r[i].permute(1, 2, 0).numpy(), Xte[i].permute(1, 2, 0).numpy(), channel_axis=2,
                                      data_range=1.0) for i in range(len(r))]
        top1 = (emb @ orig_emb.T).argmax(1) == np.arange(len(r))
        k = int(top1.sum())
        return {"ssim": float(np.mean(ssim)), "lpips": float(l.mean()), "reid_top1": k / len(r),
                "reid_ci": cp_interval(k, len(r)), "chance": 1 / len(r)}

    res: dict[str, Any] = {"params": {"train": len(tr_ids), "test": len(te_ids), "epochs": epochs, "size": 64}, "methods": {}}
    mean = Xtr.mean(0, keepdim=True).expand(len(Xte), -1, -1, -1)
    res["methods"]["mean-image"] = evaluate(mean)
    gallery = {"original": Xte[:8], "mean-image": mean[:8]}
    variants = [(m, m, False) for m in INV_METHODS if (DESC / "dist" / f"{m}.npy").exists()]
    if (DESC / "dist" / "DINOv2-S.npy").exists() and (DESC / "dist" / "DINOv2-S-LSH256.npy").exists():
        variants.append(("DINOv2-S-LSH256 (wrong key)", "DINOv2-S-LSH256", True))
    for label, name, wrong in variants:
        t0 = time.time()
        Ztr = torch.from_numpy(_codes(name, "dist", wrong))
        Zte = torch.from_numpy(_codes(name, "reg", False))
        torch.manual_seed(0)
        dec = Decoder(Ztr.shape[1]).to(device)
        opt = torch.optim.AdamW(dec.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = max(1, epochs * math.ceil(len(Ztr) / 256))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        for ep in range(epochs):
            perm = torch.randperm(len(Ztr))
            for i in range(0, len(Ztr), 256):
                b = perm[i:i + 256]
                z, x = Ztr[b].to(device), Xtr[b].to(device)
                y = dec(z)
                loss = (y - x).abs().mean() + lp(y * 2 - 1, x * 2 - 1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        dec.eval()
        with torch.no_grad():
            rec = torch.cat([dec(Zte[i:i + 256].to(device)).cpu() for i in range(0, len(Zte), 256)])
        r = evaluate(rec)
        r["train_seconds"] = time.time() - t0
        res["methods"][label] = r
        gallery[label] = rec[:8]
        print(f"inversion {label}: ssim={r['ssim']:.3f} lpips={r['lpips']:.3f} reid={r['reid_top1']:.3f} (chance {r['chance']:.4f})", flush=True)
    SEC_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SEC_OUT / "inversion_gallery.npz", **{k: (v.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                                                                for k, v in gallery.items()})
    _gallery_png(gallery, SEC_OUT / "inversion_gallery.png")
    return res


def _gallery_png(g: dict[str, torch.Tensor], path: Path) -> None:
    from PIL import Image, ImageDraw

    rows = list(g)
    W = 8
    canvas = Image.new("RGB", (160 + W * 68, len(rows) * 68), "white")
    d = ImageDraw.Draw(canvas)
    for r, k in enumerate(rows):
        d.text((4, r * 68 + 28), k[:26], fill="black")
        for c in range(min(W, len(g[k]))):
            canvas.paste(Image.fromarray((g[k][c].permute(1, 2, 0).numpy() * 255).astype(np.uint8)), (160 + c * 68, r * 68 + 2))
    canvas.save(path)


# ------------------------------------------------------------------ 4. audio
AUDIO_EVAL = ["Chromaprint", "ISCC-Audio-64", "audfprint", "CLAP"]
SNR_DB = (40, 30, 20)
PITCH_GRID = (0.25, 0.5, 1, 1.5, 2, 3, 4, 6)
TEMPO_GRID = (1.02, 1.05, 1.1, 1.2, 1.3, 1.5)


def audio(n: int, device: str, steps: int) -> dict:
    import soundfile as sf
    import torchaudio

    from attacks import _ff
    from fingerprints_audio import AUDIO_METHODS

    man = manifest().get("audio")
    if not man:
        return {"skipped": "no audio corpus in manifest"}
    items = man["sets"]["reg"][:n]
    systems = {}
    for name in AUDIO_EVAL:
        try:
            s = AUDIO_METHODS[name]()
            s.load(device if s.gpu else "cpu")
            systems[name] = s
        except Exception as e:
            print(f"audio: {name} unavailable: {e!r}")

    def fallback(name):
        def f():
            s = systems[name]
            import pickle

            reg = pickle.load(open(DESC / "audio_reg" / f"{name}.pkl", "rb"))[:20]
            dist = pickle.load(open(DESC / "audio_dist" / f"{name}.pkl", "rb"))[:20]
            return max(s.pair(a, b) for a in reg for b in dist) + 1e-6
        return f

    ths = {nm: threshold("audio", nm, fallback(nm), "fallback:max-reg-vs-dist") for nm in systems}
    out: dict[str, Any] = {"params": {"n": len(items), "snr_db": SNR_DB, "steps": steps, "pitch_grid": PITCH_GRID,
                                      "tempo_grid": TEMPO_GRID}, "thresholds": ths, "clap_pgd": {}, "magnitude": {}}
    load = lambda i: sf.read(CORPUS / man["items"][i]["path"], dtype="float32")
    t0 = time.time()
    if "CLAP" in systems:
        clap = systems["CLAP"]
        for p in clap.m.parameters():
            p.requires_grad_(False)

        def emb(y: torch.Tensor, sr: int) -> torch.Tensor:
            x = torchaudio.functional.resample(y, sr, 48000)
            win = 480000
            segs = [x[i:i + win] for i in range(0, max(1, len(x) - win // 2), win)] or [x]
            segs = [F.pad(s, (0, win - len(s))) if len(s) < win else s for s in segs]
            e = clap.m.get_audio_embedding_from_data(x=torch.stack(segs), use_tensor=True).mean(0)
            return F.normalize(e, dim=0)

        for snr in SNR_DB:
            rows = []
            for i in items:
                y, sr = load(i)
                y = y if y.ndim == 1 else y.mean(1)
                y0 = torch.from_numpy(y).to(device)
                with torch.no_grad():
                    r = emb(y0, sr)
                eps = float(np.sqrt(np.mean(y ** 2)) * 10 ** (-snr / 20))
                g = torch.Generator().manual_seed(seed_of("audio", snr, i))
                delta = ((torch.rand(len(y), generator=g) * 2 - 1) * eps).to(device)
                for _ in range(steps):
                    delta.requires_grad_(True)
                    (grad,) = torch.autograd.grad((emb(y0 + delta, sr) * r).sum(), delta)
                    with torch.no_grad():
                        delta = (delta - eps / 8 * grad.sign()).clamp(-eps, eps)
                ya = np.clip(y + delta.detach().cpu().numpy(), -1, 1)
                ya = np.round(ya * 32767) / 32767  # int16 quantisation of a saved file
                row = {"item": i, "snr_achieved": float(10 * np.log10(np.mean(y ** 2) / max(1e-12, np.mean((ya - y) ** 2))))}
                for nm, s in systems.items():
                    ref, q = s.extract(y, sr), s.extract(ya.astype(np.float32), sr)
                    sa = float(s.pair(q, ref))
                    row[nm] = {"score_clean": float(s.pair(ref, ref)), "score_adv": sa, "success": bool(sa < ths[nm][0])}
                rows.append(row)
            summ = {}
            for nm in systems:
                ok = [r[nm] for r in rows if r[nm]["score_clean"] >= ths[nm][0]]  # matched before the attack
                k = sum(o["success"] for o in ok)
                summ[nm] = {"success": k, "n": len(ok), "n_unmatched_clean": len(rows) - len(ok),
                            "rate": k / len(ok) if ok else float("nan"), "ci": cp_interval(k, len(ok))}
            out["clap_pgd"][str(snr)] = {**summ, "snr_achieved_mean": float(np.mean([r["snr_achieved"] for r in rows])),
                                         "rows": rows}
            print(f"audio CLAP-PGD snr={snr}: " + ", ".join(f"{nm}={v['rate']:.2f}" for nm, v in out['clap_pgd'][str(snr)].items() if isinstance(v, dict)), flush=True)
    t_pgd = (time.time() - t0) / max(1, len(items) * len(SNR_DB))
    # minimal pitch / tempo change that breaks each method
    t0 = time.time()
    semi = lambda k: 2 ** (k / 12)
    for nm, s in systems.items():
        rows = []
        for i in items:
            y, sr = load(i)
            y = y if y.ndim == 1 else y.mean(1)
            ref = s.extract(y, sr)
            br = {}
            for kind, grid, filt in (("pitch_semitones", PITCH_GRID,
                                      lambda k: f"asetrate={sr}*{semi(k):.6f},aresample={sr},atempo={1 / semi(k):.6f}"),
                                     ("tempo_factor", TEMPO_GRID, lambda k: f"atempo={k}")):
                br[kind] = None
                for k in grid:
                    if s.pair(s.extract(_ff(y, sr, filt(k)), sr), ref) < ths[nm][0]:
                        br[kind] = k
                        break
            rows.append({"item": i, **br})
        out["magnitude"][nm] = {"rows": rows, **{
            f"{kind}_median": _median_break([r[kind] for r in rows], grid)
            for kind, grid in (("pitch_semitones", PITCH_GRID), ("tempo_factor", TEMPO_GRID))}}
        print(f"audio magnitude {nm}: {out['magnitude'][nm]['pitch_semitones_median']} st, tempo {out['magnitude'][nm]['tempo_factor_median']}", flush=True)
    t_mag = (time.time() - t0) / max(1, len(items))
    out["seconds_per_item"] = {"clap_pgd_per_snr": t_pgd, "magnitude_all_methods": t_mag}
    out["estimate_default_n_hours"] = (t_pgd * len(SNR_DB) + t_mag) * DEFAULT_N["audio"] / 3600
    return out


def _median_break(vals: list, grid: tuple) -> float | str:
    """Median breaking magnitude, counting 'never broke within the grid' as above the grid."""
    v = sorted(float(x) if x is not None else float("inf") for x in vals)
    if not v:
        return float("nan")
    m = v[len(v) // 2]
    return m if np.isfinite(m) else f">{grid[-1]}"


# ------------------------------------------------------------------ 5. video
SPEED_GRID = (1.05, 1.1, 1.25, 1.5, 2.0)
CROP_GRID = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def video(n: int, device: str, steps: int) -> dict:
    from fingerprints_video import TMK, VIDEO_METHODS

    man = manifest().get("video")
    if not man:
        return {"skipped": "no video corpus in manifest"}
    clips = man["sets"]["reg"][:n]
    systems, skipped = {}, {}
    for name in VIDEO_METHODS:
        try:
            s = VIDEO_METHODS[name]()
            s.load(device if s.gpu else "cpu")
            path0 = CORPUS / man["items"][clips[0]]["path"]
            s.extract(path0)  # probe: vpdq import / tmk binary / ffmpeg signature filter
            systems[name] = s
        except Exception as e:
            skipped[name] = repr(e)[:300]
    ths = {}
    for nm, s in systems.items():
        def fb(nm=nm, s=s):
            if nm == "TMK+PDQF":
                return 0.7  # Meta's default level-2 threshold
            a = [s.extract(CORPUS / man["items"][c]["path"]) for c in man["sets"]["reg"][:6]]
            return max(s.pair(a[i], a[j]) for i in range(len(a)) for j in range(len(a)) if i != j) + 1e-6
        ths[nm] = threshold("video", nm, fb, "fallback:max-over-other-reg-clips" if nm != "TMK+PDQF" else "fallback:meta-default-c2")
    out: dict[str, Any] = {"params": {"n": len(clips), "eps_255": (4, 8), "steps": steps, "speed_grid": SPEED_GRID,
                                      "crop_grid": CROP_GRID}, "skipped": skipped, "thresholds": ths,
                           "frame_pgd": {}, "magnitude": {}}
    work = Path(tempfile.mkdtemp(prefix="fpsec-video-"))

    def pair(nm, q: Path, r: Path) -> float:
        s = systems[nm]
        if nm == "TMK+PDQF":
            qa, ra = work / "q.tmk", work / "r.tmk"
            qa.write_bytes(s.extract(q))
            ra.write_bytes(s.extract(r))
            return TMK.batch_scores([qa], [ra]).get((str(qa), str(ra)), (np.nan, np.nan))[1]
        return float(s.pair(s.extract(q), s.extract(r)))

    views = [TorchView(m, device) for m in ("SSCD-mixup", "DINOv2-S")]
    t0 = time.time()
    for eps in (4, 8):
        rows = []
        for c in clips:
            src = CORPUS / man["items"][c]["path"]
            frames, fps = _read_all(src)
            adv = []
            for i in range(0, len(frames), 16):
                x0 = torch.from_numpy(frames[i:i + 16]).permute(0, 3, 1, 2).float().div(255).to(device)
                with torch.no_grad():
                    refs = [v(x0) for v in views]
                loss = lambda xx: sum((v(xx) * r).sum() for v, r in zip(views, refs))
                g = torch.Generator().manual_seed(seed_of("video", c, i, eps))
                delta = ((torch.rand(x0.shape, generator=g) * 2 - 1) * eps / 255).to(device)
                for _ in range(steps):
                    delta.requires_grad_(True)
                    (grad,) = torch.autograd.grad(loss((x0 + delta).clamp(0, 1)), delta)
                    with torch.no_grad():
                        delta = (delta - eps / 255 / 8 * grad.sign()).clamp(-eps / 255, eps / 255)
                adv.append(to_u8((x0 + delta).clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()))
            dst = work / f"{c}_eps{eps}.mp4"
            _write(np.concatenate(adv), fps, dst)
            row = {"clip": c}
            for nm in systems:
                try:
                    sc = pair(nm, dst, src)
                    row[nm] = {"score_adv": sc, "success": bool(sc < ths[nm][0])}
                except Exception as e:
                    row[nm] = {"error": repr(e)[:200]}
            rows.append(row)
        out["frame_pgd"][str(eps)] = {nm: _vsum(rows, nm) for nm in systems}
        print(f"video frame-PGD eps={eps}: " + ", ".join(f"{nm}={v['rate']:.2f}" for nm, v in out['frame_pgd'][str(eps)].items()), flush=True)
    t_pgd = (time.time() - t0) / max(1, 2 * len(clips))
    t0 = time.time()
    for nm in systems:
        rows = []
        for c in clips:
            src = CORPUS / man["items"][c]["path"]
            br = {}
            for kind, grid, vf in (("speed", SPEED_GRID, lambda k: f"setpts=PTS/{k}"),
                                   ("crop_keep", CROP_GRID, lambda k: f"crop=iw*{k}:ih*{k}")):
                br[kind] = None
                for k in grid:
                    dst = work / f"{c}_{kind}_{k}.mp4"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf(k), "-c:v", "libx264", "-crf", "23",
                                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
                    if pair(nm, dst, src) < ths[nm][0]:
                        br[kind] = k
                        break
            rows.append({"clip": c, **br})
        out["magnitude"][nm] = {"rows": rows, "speed_median": _median_break([r["speed"] for r in rows], SPEED_GRID),
                                "crop_keep_median": _median_keep([r["crop_keep"] for r in rows])}
    t_mag = (time.time() - t0) / max(1, len(clips))
    out["seconds_per_clip"] = {"frame_pgd_per_eps": t_pgd, "magnitude_all_methods": t_mag}
    out["estimate_default_n_hours"] = (2 * t_pgd + t_mag) * DEFAULT_N["video"] / 3600
    return out


def _median_keep(vals: list) -> float | str:
    """Median crop fraction that broke the match; 'never broke' ranks below the grid."""
    v = sorted((x if x is not None else 0.0) for x in vals)
    if not v:
        return float("nan")
    m = v[len(v) // 2]
    return m if m > 0 else f"<{CROP_GRID[-1]}"


def _vsum(rows: list[dict], nm: str) -> dict:
    ok = [r[nm] for r in rows if "success" in r.get(nm, {})]
    k = sum(o["success"] for o in ok)
    return {"n": len(ok), "success": k, "rate": k / len(ok) if ok else float("nan"), "ci": cp_interval(k, len(ok)),
            "errors": len(rows) - len(ok)}


def _read_all(path: Path) -> tuple[np.ndarray, float]:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(s)]), fps


def _write(frames: np.ndarray, fps: float, dst: Path) -> None:
    h, w = frames.shape[1:3]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
                          "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    assert p.wait() == 0


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["evasion", "collision", "inversion", "audio", "video", "all"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    runs = ["evasion", "collision", "inversion", "audio", "video"] if args.experiment == "all" else [args.experiment]
    for r in runs:
        t0 = time.time()
        n = args.n or DEFAULT_N.get(r, 0)
        if r == "evasion":
            res = evasion(n, dev, args.steps)
        elif r == "collision":
            res = collision(n, dev, args.steps)
        elif r == "inversion":
            res = inversion(dev, args.epochs)
        elif r == "audio":
            res = audio(n, dev, args.steps)
        else:
            res = video(n, dev, max(10, args.steps // 2))
        res.update({"experiment": r, "device": dev, "wall_seconds": time.time() - t0, "host": host_info()})
        write_json(RESULTS / "security" / f"{r}.json", res)
        est = res.get("estimate_default_n_hours")
        print(f"== {r}: {time.time() - t0:.0f}s" + (f"; estimated {est:.2f} h at default N on this device" if est else ""), flush=True)


if __name__ == "__main__":
    main()
