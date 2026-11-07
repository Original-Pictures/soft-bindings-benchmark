"""Complete an existing copy of the image corpus and verify it against corpus_manifest.json.

Downloads the ABO photos missing under $BENCH_WORK/corpus with fetch_corpus's own download and storage code, then
checks the SHA-256 of every ABO and DISC21 file the image track reads. Exits non-zero if a file is missing or
differs.

Usage: python fetch_missing.py [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import CORPUS, sha256_file  # noqa: E402
from fetch_corpus import _get, _save_jpeg  # noqa: E402

MAX_SIDE = 1024  # fetch_corpus.fetch_abo stores every ABO photo at this longest side


def corpus_items() -> dict[str, dict]:
    """Path -> manifest entry for every ABO and DISC21 file of the image track."""
    man = json.loads((CORPUS / "corpus_manifest.json").read_text())
    return {it["path"]: it for part in ("image", "disc21") for it in man[part]["items"].values()}


def fetch(item: dict) -> None:
    """Download one ABO photo and store it as fetch_corpus.fetch_abo does."""
    dst = CORPUS / item["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    _save_jpeg(_get(item["url"]), dst, MAX_SIDE)


def check(item: dict) -> str:
    """'ok', 'missing' or 'differs' for one file."""
    path = CORPUS / item["path"]
    if not path.exists():
        return "missing"
    return "ok" if sha256_file(path) == item["sha256"] else "differs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    items = corpus_items()
    todo = [it for p, it in items.items() if p.startswith("abo/") and not (CORPUS / p).exists()]
    print(f"{len(items)} files in the manifest, {len(todo)} ABO photos to fetch", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(fetch, todo))
        status = dict(zip(items, pool.map(check, items.values(), chunksize=256)))
    bad = {p: s for p, s in status.items() if s != "ok"}
    print(f"verified {len(status) - len(bad)} of {len(status)} files against the manifest SHA-256", flush=True)
    for p, s in sorted(bad.items())[:20]:
        print(f"  {s}: {p}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
