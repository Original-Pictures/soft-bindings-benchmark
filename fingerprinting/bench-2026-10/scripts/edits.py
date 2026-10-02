"""Deterministic partial edits of registered originals, with ground-truth masks.

Why: a provenance service must (a) still find the registered original when part of the
image was changed, and (b) show *where* it was changed. Both need edited copies whose
edited pixels are known exactly, so every edit here is generated, not collected.

Sources: the ABO `pos` set (registered originals, max side 1024 px, as registered).
Region: a smooth blob (union of 1-4 random ellipses, feathered 3 px) whose area is a
fraction a of the image, centred on a pixel with content (local standard deviation
above the image's background level: ABO product shots are often on white, and an edit
of empty background is both trivial to hide and uninformative).

Edit types (licences of the generators in parentheses):
  splice        region pasted from an ABO `bg` image (never registered or queried)
  copymove      region copied from another place of the same image
  inpaint-telea classical inpainting, OpenCV (Apache-2.0)
  inpaint-lama  LaMa big-lama torchscript (Apache-2.0; advimman/lama weights repackaged by
                enesmsahin/simple-lama-inpainting, Apache-2.0)
  recolor       hue rotation by 60-180 degrees inside the region
  textreplace   region bounding box filled with a flat colour and new text (a relabelled
                product or price)
Stable Diffusion 2 inpainting was planned as a reference generator; its Hugging Face
repository (stabilityai/stable-diffusion-2-inpainting) no longer serves the weights
(HTTP 401 on 2026-09-26), so it is not used.

Post-processing after the edit: none, jpeg75, resize0.75+jpeg80, crop90 (the watermark
bench's attack functions). The GT mask is kept in the ORIGINAL frame; for crop90 a
validity mask marks the original pixels still present in the query.

Design (balanced sample, not the full factorial): each source gets `per_source` specs
drawn as a Latin-square-like cycle over (type x area x post) with a per-source offset,
so every level of every factor appears equally often overall (500 x 12 = 6000 edits).

Output: $BENCH_WORK/out/edits/<src>__<type>_<area>_<post>.png, .mask.png (GT, original
frame, uint8 0/255), .valid.png (crop90 only), and index.json.

Usage: python edits.py [--n 500] [--per-source 12] [--workers 8] [--device cuda]
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import attacks
from common import BENCH_WORK, CORPUS, OUT, load_rgb, rng_for, to_u8

EDIT_DIR = OUT / "edits"
TYPES = ["splice", "copymove", "inpaint-telea", "inpaint-lama", "recolor", "textreplace"]
AREAS = [0.01, 0.03, 0.10, 0.25]
POSTS = ["none", "jpeg75", "resize0.75_jpeg80", "crop90"]
GPU_TYPES = {"inpaint-lama"}


def spec_name(t: str, a: float, p: str) -> str:
    return f"{t}_{a:g}_{p}"


def design(sources: list[str], per_source: int) -> list[tuple[str, str, float, str]]:
    """Balanced (type, area, post) assignment: a global cycle through the factorial grid,
    each source taking the next `per_source` cells after a seeded shuffle of the grid."""
    grid = list(itertools.product(TYPES, AREAS, POSTS))
    rng = rng_for("edit-design")
    order = [grid[i] for i in rng.permutation(len(grid))]
    out, k = [], 0
    for s in sources:
        for _ in range(per_source):
            t, a, p = order[k % len(order)]
            out.append((s, t, a, p))
            k += 1
    return out


# ------------------------------------------------------------------ region
def content_map(x: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.blur(g, (15, 15))
    sd = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mu * mu, 0))
    return sd > max(4.0, np.percentile(sd, 40))


def blob_mask(shape: tuple[int, int], area: float, rng: np.random.Generator, content: np.ndarray) -> np.ndarray:
    """Float mask in [0,1] (feathered), area ~= `area` of the image, centred on content."""
    import cv2

    h, w = shape
    ys, xs = np.nonzero(content)
    if len(ys) == 0:
        ys, xs = np.array([h // 2]), np.array([w // 2])
    j = int(rng.integers(0, len(ys)))
    cy, cx = int(ys[j]), int(xs[j])
    target = area * h * w
    m = np.zeros((h, w), np.uint8)
    n = int(rng.integers(1, 5))
    for _ in range(n):
        r = np.sqrt(target / n / np.pi)
        ay, ax = r * rng.uniform(0.6, 1.6), r * rng.uniform(0.6, 1.6)
        oy, ox = rng.normal(0, r * 0.6, 2)
        cv2.ellipse(m, (int(cx + ox), int(cy + oy)), (max(2, int(ax)), max(2, int(ay))),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
    # rescale to the target area (ellipses overlap / leave the frame)
    for _ in range(6):
        cur = m.sum()
        if cur == 0 or abs(cur - target) / target < 0.1:
            break
        k = max(3, int(abs(np.sqrt(target / np.pi) - np.sqrt(cur / np.pi))) * 2 + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, ker) if cur < target else cv2.erode(m, ker)
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3.0)
    return np.clip(soft, 0, 1)


# ------------------------------------------------------------------ edit operators
_LAMA = {}


def _lama(device: str):
    import torch

    if device not in _LAMA:
        _LAMA[device] = torch.jit.load(str(BENCH_WORK / "weights" / "lama" / "big-lama.pt"), map_location=device).eval()
    return _LAMA[device]


def inpaint_lama(x: np.ndarray, hard: np.ndarray, device: str) -> np.ndarray:
    import torch

    h, w = hard.shape
    ph, pw = (-h) % 8, (-w) % 8
    xi = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    mi = np.pad(hard, ((0, ph), (0, pw)))
    with torch.no_grad():
        t = torch.from_numpy(xi.transpose(2, 0, 1))[None].float().to(device)
        k = torch.from_numpy(mi[None, None].astype(np.float32)).to(device)
        y = _lama(device)(t, k)[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return y[:h, :w].astype(np.float32)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(attacks._font("NotoSans-Regular.ttf"), size)


def apply_edit(x: np.ndarray, soft: np.ndarray, kind: str, rng: np.random.Generator, bg: np.ndarray | None,
               device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (edited image, GT mask as float in [0,1] = soft mask actually applied)."""
    import cv2

    h, w = soft.shape
    hard = (soft > 0.5).astype(np.uint8)
    m3 = soft[..., None]
    if kind == "splice":
        src = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA) if bg is not None else rng.random((h, w, 3), np.float32)
        return (x * (1 - m3) + src * m3).astype(np.float32), soft
    if kind == "copymove":
        dy, dx = int(rng.integers(-h // 3, h // 3 + 1)), int(rng.integers(-w // 3, w // 3 + 1))
        dy = dy or h // 4
        shifted = np.roll(np.roll(x, dy, 0), dx, 1)
        return (x * (1 - m3) + shifted * m3).astype(np.float32), soft
    if kind == "inpaint-telea":
        y = cv2.inpaint(to_u8(x), hard, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0
        return (x * (1 - m3) + y * m3).astype(np.float32), soft
    if kind == "inpaint-lama":
        dil = cv2.dilate(hard, np.ones((7, 7), np.uint8))
        y = inpaint_lama(x, dil, device)
        return (x * (1 - m3) + y * m3).astype(np.float32), soft
    if kind == "recolor":
        hsv = cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(30, 91))) % 180  # OpenCV hue is 0-179 (x2 degrees)
        # A hue turn is invisible on white/grey product surfaces; tint them (saturation
        # >= 110, value <= 235) so the edit changes the pixels it claims to change.
        hsv[..., 1] = np.maximum(hsv[..., 1], 110)
        hsv[..., 2] = np.minimum(hsv[..., 2], 235)
        y = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return (x * (1 - m3) + y * m3).astype(np.float32), soft
    if kind == "textreplace":
        ys, xs = np.nonzero(hard)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        im = Image.fromarray(to_u8(x))
        d = ImageDraw.Draw(im)
        fill = tuple(int(c) for c in rng.integers(0, 256, 3))
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill)
        txt = rng.choice(["SALE $9.99", "NEW MODEL", "GENUINE", "50% OFF", "LIMITED"])
        fs = max(8, int((y1 - y0) * 0.45))
        f = _font(fs)
        while fs > 8 and d.textlength(txt, font=f) > (x1 - x0) * 0.95:
            fs -= 2
            f = _font(fs)
        ink = tuple(255 - c for c in fill)
        d.text((x0 + 2, y0 + (y1 - y0 - fs) // 2), txt, font=f, fill=ink)
        gt = np.zeros((h, w), np.float32)
        gt[y0:y1, x0:x1] = 1.0
        return np.asarray(im, np.float32) / 255.0, gt
    raise ValueError(kind)


def post_process(y: np.ndarray, post: str, rng) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Returns (query, crop box in original pixels (t, l, h, w) or None)."""
    if post == "none":
        return y, None
    if post == "jpeg75":
        return attacks.jpeg(75)(y), None
    if post == "resize0.75_jpeg80":
        return attacks.jpeg(80)(attacks.resize(0.75)(y)), None
    if post == "crop90":
        h, w = y.shape[:2]
        ch, cw = round(h * 0.9), round(w * 0.9)
        t, l = (h - ch) // 2, (w - cw) // 2
        return attacks.center_crop(0.9)(y), (t, l, ch, cw)
    raise ValueError(post)


# ------------------------------------------------------------------ driver
def manifest() -> dict:
    return json.loads((CORPUS / "corpus_manifest.json").read_text())


def _paths(src: str, t: str, a: float, p: str) -> tuple[Path, Path, Path]:
    # String concatenation, not with_suffix: spec names contain dots ("0.03_none").
    stem = str(EDIT_DIR / f"{src}__{spec_name(t, a, p)}")
    return Path(stem + ".png"), Path(stem + ".mask.png"), Path(stem + ".valid.png")


def make_one(job: tuple[str, str, str, float, str, list[str], str]) -> dict:
    src, src_path, t, a, p, bg_paths, device = job
    out, mpath, vpath = _paths(src, t, a, p)
    rec = {"src": src, "type": t, "area": a, "post": p, "key": f"{src}|{spec_name(t, a, p)}",
           "path": str(out.relative_to(OUT)), "mask": str(mpath.relative_to(OUT))}
    if out.exists() and mpath.exists():
        rec.update(json.loads(Path(str(out) + ".json").read_text()))
        return rec
    x = load_rgb(Path(src_path), 1024)
    rng = rng_for("edit", src, t, a, p)
    soft = blob_mask(x.shape[:2], a, rng, content_map(x))
    bg = load_rgb(Path(bg_paths[int(rng.integers(0, len(bg_paths)))]), 1024) if t == "splice" else None
    y, gt = apply_edit(x, soft, t, rng, bg, device)
    q, box = post_process(y, p, rng)
    EDIT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_u8(q)).save(out)
    Image.fromarray(to_u8(gt)).save(mpath)
    changed = np.abs(y - x).max(-1) > 2 / 255
    g = gt > 0.5
    extra = {"area_actual": float(g.mean()), "crop_box": box,
             # share of the GT region whose pixels actually changed (inpainting a flat
             # background can reproduce it exactly)
             "visible_frac": float(changed[g].mean()) if g.any() else 0.0}
    if box is not None:
        v = np.zeros(gt.shape, np.uint8)
        tt, ll, hh, ww = box
        v[tt:tt + hh, ll:ll + ww] = 255
        Image.fromarray(v).save(vpath)
        extra["valid"] = str(vpath.relative_to(OUT))
    Path(str(out) + ".json").write_text(json.dumps(extra))
    rec.update(extra)
    return rec


def build(n: int, per_source: int, workers: int, device: str) -> list[dict]:
    man = manifest()["image"]
    items = man["items"]
    sources = sorted(man["sets"]["pos"])[:n]
    bg_paths = [str(CORPUS / items[i]["path"]) for i in man["sets"]["bg"]]
    plan = design(sources, per_source)
    jobs = [(s, str(CORPUS / items[s]["path"]), t, a, p, bg_paths, device) for s, t, a, p in plan]
    cpu_jobs = [j for j in jobs if j[2] not in GPU_TYPES]
    gpu_jobs = [j for j in jobs if j[2] in GPU_TYPES]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        recs = pool.map(make_one, [(*j[:6], "cpu") for j in cpu_jobs], chunksize=4)
    t_cpu = time.time() - t0
    recs += [make_one(j) for j in gpu_jobs]  # LaMa in the main process on the accelerator
    t_all = time.time() - t0
    recs.sort(key=lambda r: r["key"])
    idx = {"design": {"types": TYPES, "areas": AREAS, "posts": POSTS, "per_source": per_source, "n_sources": len(sources),
                      "skipped": {"inpaint-sd": "stabilityai/stable-diffusion-2-inpainting weights not served (HTTP 401, 2026-09-26)"}},
           "timing_s": {"cpu_pool": t_cpu, "total": t_all, "workers": workers, "device": device}, "edits": recs}
    (EDIT_DIR / "index.json").write_text(json.dumps(idx, indent=1))
    print(f"{len(recs)} edits ({len(gpu_jobs)} LaMa) in {t_all:.0f}s (cpu pool {t_cpu:.0f}s)")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--per-source", type=int, default=12)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    build(args.n, args.per_source, args.workers, dev)


if __name__ == "__main__":
    main()
