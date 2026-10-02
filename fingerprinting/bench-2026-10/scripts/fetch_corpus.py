"""Build the benchmark corpora under $BENCH_WORK/corpus and write corpus_manifest.json.

Every item records its source URL, licence and the sha256 of the stored file, so a
rerun can verify it holds the same media. Sampling is seeded (common.SEED).

Image
  ABO (Amazon Berkeley Objects, CC BY 4.0). Products come from the listings metadata:
    reg    registered originals (one main image per product, min side >= 800 px), stored
           at max side 1024 px (JPEG q95): the "asset" a provenance service fingerprints
    pos    subset of reg used as positive query sources (attacked)
    hard   another image of a pos product (different photo, same product): must NOT match
    dist   distractor images from other products, added to the index only
    neg    images of further products, attacked and queried: never registered
    bg     backgrounds for the overlay-onto-background attack (never registered/queried)
  DISC21 (ISC2021 dev, CC BY-NC 4.0, research only): a subset fetched member-by-member
  from the official zips: GT-matched dev queries with their references, unmatched dev
  queries, and extra references from references_0.zip as distractors.
Audio
  music  FMA-large tracks under CC BY / CC BY-SA / CC0 that are not in fma_medium,
         mono 16-bit WAV at 22.05 kHz, 30 s clips (reg / dist / neg split)
  speech LibriSpeech test-clean + dev-clean (CC BY 4.0), utterances >= 8 s
Video
  Blender open movies (CC BY 3.0) and Xiph derf 1080p sequences, cut into 5 s clips,
  scaled to 1280 px wide, stored as lossless-ish H.264 CRF 12 (the registered asset).

Usage: python fetch_corpus.py [--scale smoke|full] [--only image,disc21,audio,video] [--workers 32]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import subprocess
import tarfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import BENCH_DIR, CORPUS, SEED, rng_for, sha256_file, write_json

ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
DISC = "https://dl.fbaipublicfiles.com/image_similarity_challenge/public"
FMA = "https://os.unil.cloud.switch.ch/fma"
MANIFEST = CORPUS / "corpus_manifest.json"

SCALES = {
    "full": dict(reg=5000, pos=2000, hard=1000, dist=100_000, neg=2000, bg=200,
                 disc_gt=3000, disc_negq=3000, disc_extra=22_000,
                 music_reg=600, music_dist=4000, music_neg=400, speech_reg=400, speech_dist=1600, speech_neg=200,
                 video_films=None),
    "smoke": dict(reg=120, pos=60, hard=30, dist=1500, neg=60, bg=10,
                  disc_gt=60, disc_negq=60, disc_extra=400,
                  music_reg=12, music_dist=60, music_neg=8, speech_reg=12, speech_dist=40, speech_neg=6,
                  video_films=1),
}

_lock = threading.Lock()


def _get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "op-fp-bench"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
    raise RuntimeError(url)


def _save_jpeg(data: bytes, dst: Path, max_side: int) -> tuple[int, int]:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=95, subsampling=0)
        return im.size


def _pmap(fn, items, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------ ABO
def abo_products(root: Path) -> tuple[dict, dict]:
    """image_id -> (h, w, path) and product key -> [image ids] (main first)."""
    meta = {}
    with gzip.open(root / "images.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            meta[row["image_id"]] = (int(row["height"]), int(row["width"]), row["path"])
    products: dict[str, list[str]] = {}
    with tarfile.open(root / "abo-listings.tar") as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".json.gz"):
                continue
            for line in gzip.open(tar.extractfile(m), "rt"):
                d = json.loads(line)
                main = d.get("main_image_id")
                if not main or main not in meta:
                    continue
                others = [i for i in d.get("other_image_id", []) if i in meta and i != main]
                # Listings of one item in several marketplaces share images: key by main image.
                prev = products.get(main, [main])
                products[main] = prev + [i for i in others if i not in prev]
    return meta, products


def fetch_abo(sc: dict, workers: int) -> dict:
    root = CORPUS / "abo"
    root.mkdir(parents=True, exist_ok=True)
    for name, url in (("images.csv.gz", f"{ABO}/images/metadata/images.csv.gz"),
                      ("abo-listings.tar", f"{ABO}/archives/abo-listings.tar"),
                      ("LICENSE-CC-BY-4.0.txt", f"{ABO}/LICENSE-CC-BY-4.0.txt")):
        if not (root / name).exists():
            (root / name).write_bytes(_get(url))
    meta, products = abo_products(root)
    # An image id used by two products (shared photo) must not end up on both sides.
    owner: dict[str, int] = {}
    for key, ims in products.items():
        for i in ims:
            owner[i] = owner.get(i, 0) + 1
    keys = sorted(k for k, ims in products.items() if all(owner[i] == 1 for i in ims))
    rng = np.random.default_rng(SEED)
    rng.shuffle(keys)

    def big(i, m):
        h, w, _ = meta[i]
        return min(h, w) >= m

    reg_keys = [k for k in keys if big(k, 800)][: sc["reg"]]
    used = set(reg_keys)
    rest = [k for k in keys if k not in used]
    pos_keys = reg_keys[: sc["pos"]]
    hard = [(k, next(i for i in products[k][1:] if big(i, 500))) for k in pos_keys
            if any(big(i, 500) for i in products[k][1:])][: sc["hard"]]
    neg_keys = [k for k in rest if big(k, 800)][: sc["neg"]]
    used |= set(neg_keys)
    bg_keys = [k for k in rest if k not in used and big(k, 800)][: sc["bg"]]
    used |= set(bg_keys)
    # Distractors may come from any other product, including ones whose photos are shared
    # with sibling listings, as long as no image of a used product is taken. Exact photo
    # reuse across products is removed by image id here; near-duplicates are handled by
    # dedup_distractors() after fingerprinting (paper Sec. 3).
    used_imgs = {i for k in used for i in products[k]} | {h for _, h in hard}
    all_keys = sorted(products)
    drng = np.random.default_rng(SEED + 1)
    drng.shuffle(all_keys)
    dist_imgs, seen = [], set(used_imgs)
    for k in all_keys:
        if k in used or any(i in used_imgs for i in products[k]):
            continue
        # One image per distractor product keeps the distractors independent draws.
        cands = [i for i in products[k] if big(i, 300) and i not in seen]
        if cands:
            pick = cands[int(drng.integers(0, len(cands)))]
            dist_imgs.append(pick)
            seen.update(products[k])
        if len(dist_imgs) >= sc["dist"]:
            break

    sets = {"reg": reg_keys, "pos": pos_keys, "hard": [i for _, i in hard], "neg": neg_keys, "bg": bg_keys,
            "dist": dist_imgs}
    todo = sorted({i for v in sets.values() for i in v})
    print(f"ABO: {len(todo)} images to fetch ({ {k: len(v) for k, v in sets.items()} })")

    def one(i):
        dst = root / "img" / f"{i}.jpg"
        if not dst.exists():
            _save_jpeg(_get(f"{ABO}/images/original/{meta[i][2]}"), dst, 1024)
        return i

    done = [0]

    def tick(i):
        one(i)
        with _lock:
            done[0] += 1
            if done[0] % 2000 == 0:
                print(f"  {done[0]}/{len(todo)}", flush=True)

    _pmap(tick, todo, workers)
    items = {i: {"path": f"abo/img/{i}.jpg", "url": f"{ABO}/images/original/{meta[i][2]}", "licence": "CC BY 4.0"}
             for i in todo}
    return {"sets": sets, "hard_pairs": hard, "items": items}


# ------------------------------------------------------------------ DISC21 subset
def fetch_disc21(sc: dict, workers: int) -> dict:
    from remotezip import RemoteZip

    root = CORPUS / "disc21"
    root.mkdir(parents=True, exist_ok=True)
    gt_path = root / "dev_ground_truth.csv"
    if not gt_path.exists():
        gt_path.write_bytes(_get(f"{DISC}/dev_ground_truth.csv"))
    gt = [l.strip().split(",") for l in gt_path.read_text().splitlines() if l.strip()]
    gt = [(q, r) for q, r in gt if r]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(gt))[: sc["disc_gt"]]
    pairs = sorted(gt[i] for i in idx)
    matched = {q for q, _ in gt}
    with RemoteZip(f"{DISC}/dev_queries.zip") as z:
        allq = sorted(n.split("/")[-1][:-4] for n in z.namelist() if n.endswith(".jpg"))
    unmatched = [q for q in allq if q not in matched]
    negq = sorted(unmatched[i] for i in rng.permutation(len(unmatched))[: sc["disc_negq"]])
    need_refs = {r for _, r in pairs}
    # reference Rxxxxxx lives in references_{int(x)//50000}.zip
    by_zip: dict[int, list[str]] = {}
    for r in need_refs:
        by_zip.setdefault(int(r[1:]) // 50_000, []).append(r)
    extra_pool = [f"R{n:06d}" for n in range(0, 50_000)]
    extra = [r for r in (extra_pool[i] for i in rng.permutation(len(extra_pool))) if r not in need_refs][: sc["disc_extra"]]
    by_zip.setdefault(0, []).extend(extra)

    def pull(zip_url: str, members: list[tuple[str, Path]]):
        chunks = [members[i::workers] for i in range(workers)]

        def work(chunk):
            if not chunk:
                return
            with RemoteZip(zip_url) as z:
                names = set(z.namelist())
                for m, dst in chunk:
                    if dst.exists():
                        continue
                    name = m if m in names else next(n for n in names if n.endswith("/" + m.split("/")[-1]))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(z.read(name))

        _pmap(work, chunks, workers)

    pull(f"{DISC}/dev_queries.zip", [(f"images/queries/{q}.jpg", root / "queries" / f"{q}.jpg")
                                      for q in sorted({q for q, _ in pairs} | set(negq))])
    for zi, refs in sorted(by_zip.items()):
        pull(f"{DISC}/references_{zi}.zip", [(f"images/references/{r}.jpg", root / "refs" / f"{r}.jpg") for r in sorted(refs)])
        print(f"DISC21 references_{zi}: {len(refs)}")
    items = {}
    for q in {q for q, _ in pairs} | set(negq):
        items[q] = {"path": f"disc21/queries/{q}.jpg", "url": f"{DISC}/dev_queries.zip#{q}.jpg", "licence": "CC BY-NC 4.0 (DISC21)"}
    for r in need_refs | set(extra):
        items[r] = {"path": f"disc21/refs/{r}.jpg", "url": f"{DISC}/references_{int(r[1:]) // 50_000}.zip#{r}.jpg",
                    "licence": "CC BY-NC 4.0 (DISC21)"}
    return {"pairs": pairs, "negq": negq, "refs": sorted(need_refs | set(extra)), "items": items}


# ------------------------------------------------------------------ audio
AUDIO_SR = 22050
OPEN_FMA = ("Attribution", "Attribution-ShareAlike", "CC0", "Public Domain", "Creative Commons Attribution 4.0",
            "Creative Commons Attribution-ShareAlike")


def _to_wav(src: bytes | Path, dst: Path, seconds: float | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-v", "error", "-y", "-i", "-" if isinstance(src, bytes) else str(src)]
    if seconds:
        args += ["-t", str(seconds)]
    args += ["-ac", "1", "-ar", str(AUDIO_SR), "-sample_fmt", "s16", str(dst)]
    subprocess.run(args, input=src if isinstance(src, bytes) else None, check=True)


def fetch_audio(sc: dict, workers: int) -> dict:
    import pandas as pd
    from remotezip import RemoteZip

    root = CORPUS / "audio"
    root.mkdir(parents=True, exist_ok=True)
    tracks_csv = root / "fma_tracks.csv"
    if not tracks_csv.exists():
        with RemoteZip(f"{FMA}/fma_metadata.zip") as z:
            tracks_csv.write_bytes(z.read("fma_metadata/tracks.csv"))
    t = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    lic = t[("track", "license")].fillna("")
    subset = t[("set", "subset")]
    ok = lic.apply(lambda s: any(s.startswith(p) or p in s for p in OPEN_FMA) and "NonCommercial" not in s
                   and "NoDerivatives" not in s and "Noncommercial" not in s)
    dur = t[("track", "duration")]
    cand = sorted(int(i) for i in t.index[ok & (subset == "large") & (dur >= 60)])
    rng = np.random.default_rng(SEED)
    cand = [cand[i] for i in rng.permutation(len(cand))]
    n_music = sc["music_reg"] + sc["music_dist"] + sc["music_neg"]
    music = cand[: int(n_music * 1.15)]  # some large-archive members are missing or unreadable

    def member(tid: int) -> str:
        return f"fma_large/{tid // 1000:03d}/{tid:06d}.mp3"

    got: list[int] = []
    chunks = [music[i::workers] for i in range(workers)]

    def work(chunk):
        with RemoteZip(f"{FMA}/fma_large.zip") as z:
            for tid in chunk:
                dst = root / "music" / f"{tid:06d}.wav"
                if not dst.exists():
                    try:
                        _to_wav(z.read(member(tid)), dst, 30)
                    except Exception:
                        continue
                import soundfile as sf

                if sf.info(dst).duration >= 29.5:
                    with _lock:
                        got.append(tid)

    _pmap(work, chunks, workers)
    got = [tid for tid in music if tid in set(got)][:n_music]
    m_reg, m_dist, m_neg = (got[: sc["music_reg"]], got[sc["music_reg"]: sc["music_reg"] + sc["music_dist"]],
                            got[sc["music_reg"] + sc["music_dist"]:])
    items = {f"m{tid:06d}": {"path": f"audio/music/{tid:06d}.wav", "url": f"{FMA}/fma_large.zip#{member(tid)}",
                             "licence": str(lic[tid])} for tid in got}

    # LibriSpeech: test-clean + dev-clean, utterances >= 8 s.
    sp = root / "librispeech"
    for part in ("test-clean", "dev-clean"):
        if not (sp / "LibriSpeech" / part).exists():
            sp.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(_get(f"https://www.openslr.org/resources/12/{part}.tar.gz"))) as tar:
                tar.extractall(sp, filter="data")
    import soundfile as sf

    flacs = sorted((sp / "LibriSpeech").rglob("*.flac"))
    long = [f for f in flacs if sf.info(f).duration >= 8.0]
    long = [long[i] for i in rng.permutation(len(long))]
    n_sp = sc["speech_reg"] + sc["speech_dist"] + sc["speech_neg"]
    speech = long[:n_sp]

    def conv(f):
        dst = root / "speech" / (f.stem + ".wav")
        if not dst.exists():
            _to_wav(f, dst)
        return f.stem

    sids = _pmap(conv, speech, workers)
    for f, s in zip(speech, sids):
        items[f"s{s}"] = {"path": f"audio/speech/{s}.wav", "url": f"https://www.openslr.org/resources/12#{f.name}",
                          "licence": "CC BY 4.0 (LibriSpeech)"}
    s_reg, s_dist, s_neg = (sids[: sc["speech_reg"]], sids[sc["speech_reg"]: sc["speech_reg"] + sc["speech_dist"]],
                            sids[sc["speech_reg"] + sc["speech_dist"]:])
    sets = {"reg": [f"m{t:06d}" for t in m_reg] + [f"s{s}" for s in s_reg],
            "dist": [f"m{t:06d}" for t in m_dist] + [f"s{s}" for s in s_dist],
            "neg": [f"m{t:06d}" for t in m_neg] + [f"s{s}" for s in s_neg]}
    print("audio:", {k: len(v) for k, v in sets.items()})
    return {"sets": sets, "items": items, "sr": AUDIO_SR}


# ------------------------------------------------------------------ video
# download.blender.org answers 403 to AWS address ranges; on the GPU host the films are
# relayed from a workstation download through the bench bucket and checked against these.
FILM_SHA256 = {
    "Sintel.2010.1080p.mkv": "97f1dbc66231df42ad49bd8c29aa174b8f48933058e47e7157d4ba63d93a8efa",
    "tears_of_steel_720p.mov": "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f",
    "elephantsdream-720-h264-st-aac.mov": "6fc295e0e92835316ec95929ccfc664b05ed2434f673e6f5e684f98f7558a547",
    "bbb_sunflower_1080p_30fps_normal.mp4": "ae51005850b0ff757fe60c3dd7a12d754d3cd2397d87d939b55235e457f97658",
}
FILMS = [  # (id, url, licence, archive member or None)
    ("sintel", "https://download.blender.org/durian/movies/Sintel.2010.1080p.mkv", "CC BY 3.0 (Blender Foundation)", None),
    ("tos", "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov", "CC BY 3.0 (Blender Foundation)", None),
    ("ed", "https://download.blender.org/ED/elephantsdream-720-h264-st-aac.mov", "CC BY 2.5 (Blender Foundation)", None),
    ("bbb", "https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip",
     "CC BY 3.0 (Blender Foundation)", "bbb_sunflower_1080p_30fps_normal.mp4"),
]
XIPH = ["aspen_1080p", "blue_sky_1080p25", "controlled_burn_1080p", "crowd_run_1080p50", "dinner_1080p30",
        "ducks_take_off_1080p50", "factory_1080p30", "in_to_tree_1080p50", "life_1080p30", "old_town_cross_1080p50",
        "park_joy_1080p50", "pedestrian_area_1080p25", "red_kayak_1080p", "riverbed_1080p25", "rush_field_cuts_1080p",
        "rush_hour_1080p25", "snow_mnt_1080p", "speed_bag_1080p", "station2_1080p25", "sunflower_1080p25",
        "touchdown_pass_1080p", "tractor_1080p25", "west_wind_easy_1080p"]
CLIP_S = 5.0
VIDEO_ENC = ["-vf", "scale=1280:-2:flags=bicubic,fps=25", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
             "-pix_fmt", "yuv420p", "-an"]


def _duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def fetch_video(sc: dict, workers: int) -> dict:
    root = CORPUS / "video"
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    films = FILMS[: sc["video_films"]] if sc["video_films"] else FILMS
    items, clips = {}, []
    for fid, url, lic, member in films:
        dst = src / (member or url.split("/")[-1])
        if not dst.exists():
            subprocess.run(["curl", "-sSfL", "-o", str(dst) + ".part", url], check=True)
            if member:
                import zipfile

                with zipfile.ZipFile(str(dst) + ".part") as z:
                    dst.write_bytes(z.read(member))
                Path(str(dst) + ".part").unlink()
            else:
                Path(str(dst) + ".part").rename(dst)
        if dst.name in FILM_SHA256 and sha256_file(dst) != FILM_SHA256[dst.name]:
            raise SystemExit(f"{dst.name}: sha256 mismatch")
        # Skip the opening titles and the end credits (text on black matches everything).
        dur = _duration(dst)
        starts = np.arange(30.0, dur - 120.0 - CLIP_S, CLIP_S)
        clips += [(f"{fid}_{int(s):05d}", dst, float(s), lic, url) for s in starts]
    xiph = XIPH[:2] if sc["video_films"] else XIPH
    for x in xiph:
        dst = src / f"{x}.y4m"
        if not dst.exists():
            subprocess.run(["curl", "-sSfL", "-o", str(dst), f"https://media.xiph.org/video/derf/y4m/{x}.y4m"], check=True)
        clips.append((f"xiph_{x}", dst, 0.0, "Xiph.org derf collection (research use)", f"https://media.xiph.org/video/derf/y4m/{x}.y4m"))

    def cut(c):
        cid, p, s, lic, url = c
        out = root / "clips" / f"{cid}.mp4"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.3f}", "-i", str(p), "-t", str(CLIP_S), *VIDEO_ENC,
                            str(out)], check=True)
        return cid

    ids = _pmap(cut, clips, max(1, workers // 4))
    for (cid, p, s, lic, url) in clips:
        items[cid] = {"path": f"video/clips/{cid}.mp4", "url": f"{url}#t={s:.1f},{s + CLIP_S:.1f}", "licence": lic}
    rng = np.random.default_rng(SEED)
    order = [ids[i] for i in rng.permutation(len(ids))]
    n_reg = max(4, len(order) // 4)
    n_neg = max(2, len(order) // 12)
    sets = {"reg": sorted(order[:n_reg]), "neg": sorted(order[n_reg:n_reg + n_neg]), "dist": sorted(order[n_reg + n_neg:])}
    print("video:", {k: len(v) for k, v in sets.items()})
    return {"sets": sets, "items": items, "clip_seconds": CLIP_S}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=SCALES)
    ap.add_argument("--only", default="image,disc21,audio,video")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    sc = SCALES[args.scale]
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    man["scale"] = args.scale
    fns = {"image": fetch_abo, "disc21": fetch_disc21, "audio": fetch_audio, "video": fetch_video}
    for part in args.only.split(","):
        man[part] = fns[part](sc, args.workers)
        items = man[part]["items"]
        for k, it in items.items():
            it["sha256"] = sha256_file(CORPUS / it["path"])
        write_json(MANIFEST, man)
        print(f"{part}: {len(items)} items hashed")
    # The repo keeps a copy without per-item URLs of bulk sets (size) but with every hash.
    write_json(BENCH_DIR / "corpus_manifest.json", man)


if __name__ == "__main__":
    main()
