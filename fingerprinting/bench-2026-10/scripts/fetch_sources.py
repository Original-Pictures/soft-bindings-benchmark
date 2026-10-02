"""Fetch pinned upstream source trees for every benchmarked fingerprint method (and the
watermarks used in the combination track).

Each repo is downloaded as a GitHub tarball at an exact commit (resolved once and
then pinned in sources.lock.json) and extracted under $BENCH_WORK/src/<name>.
The tarball sha256 and the repo's SPDX licence (GitHub licence API) are recorded
so the licence table in the report is reproducible from the lock file.

Usage: python fetch_sources.py [--resolve]   (--resolve re-pins to current HEAD)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from common import BENCH_WORK, SCRIPTS

REPOS: dict[str, str] = {
    # fingerprint methods
    "blockhash-python": "commonsmachinery/blockhash-python",
    "ThreatExchange": "facebook/ThreatExchange",  # PDQ reference, TMK+PDQF, vPDQ
    "sscd-copy-detection": "facebookresearch/sscd-copy-detection",
    "ISC21-Descriptor-Track-1st": "lyakaap/ISC21-Descriptor-Track-1st",
    "dinohash": "proteus-photos/dinohash-perceptual-hash",
    "audfprint": "dpwe/audfprint",
    "Olaf": "JorenSix/Olaf",
    "neural-audio-fp": "mimbres/neural-audio-fp",
    "neural-music-fp": "raraz15/neural-music-fp",  # NMFP (GPL-3.0), reference row
    "iscc-core": "iscc/iscc-core",
    # watermarks for the combination track (same pins as the 2026-09 watermark bench)
    "videoseal": "facebookresearch/videoseal",
    "trustmark": "adobe/trustmark",
    "audioseal": "facebookresearch/audioseal",
}
LOCK = SCRIPTS / "sources.lock.json"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "op-fp-bench"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def resolve(repo: str) -> dict[str, str]:
    sha = json.loads(_get(f"https://api.github.com/repos/{repo}/commits/HEAD"))["sha"]
    try:
        spdx = json.loads(_get(f"https://api.github.com/repos/{repo}/license"))["license"]["spdx_id"]
    except Exception as exc:  # repo without a detectable LICENSE file
        spdx = f"UNDETECTED ({type(exc).__name__})"
    return {"repo": repo, "commit": sha, "spdx": spdx}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve", action="store_true")
    args = ap.parse_args()
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    for name, repo in REPOS.items():
        if args.resolve or name not in lock:
            lock[name] = resolve(repo)
        entry = lock[name]
        dest = BENCH_WORK / "src" / name
        if dest.exists() and (dest / ".op-commit").read_text().strip() == entry["commit"]:
            print(f"{name}: present @ {entry['commit'][:12]}")
            continue
        blob = _get(f"https://codeload.github.com/{repo}/tar.gz/{entry['commit']}")
        digest = hashlib.sha256(blob).hexdigest()
        if entry.get("tarball_sha256") and entry["tarball_sha256"] != digest:
            raise SystemExit(f"{name}: tarball sha256 changed {entry['tarball_sha256']} -> {digest}")
        entry["tarball_sha256"] = digest
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            top = tar.getnames()[0].split("/")[0]
            tar.extractall(dest.parent, filter="data")
        (dest.parent / top).rename(dest)
        (dest / ".op-commit").write_text(entry["commit"])
        print(f"{name}: {entry['commit'][:12]} {entry['spdx']} sha256={digest[:16]}")
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
