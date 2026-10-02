"""Pack the arXiv submission: only the files the compile needs, comments stripped.

arXiv publishes the source, so LaTeX comments are removed (an escaped \\% is kept).
Build products, raster previews, scripts and verification notes are left out; the
compiled bibliography ships as main.bbl because arXiv does not need to run BibTeX.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "arxiv-submission.tar.gz"

TEX = ["main.tex", "sections/*.tex", "generated/numbers.tex", "generated/tables/*/*.tex", "figures/*.tex"]
BINARY = ["figures/*.pdf", "main.bbl"]
SKIP: set[str] = set()


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        # A % not preceded by a backslash starts a comment.
        m = re.search(r"(?<!\\)%", line)
        if m:
            kept = line[: m.start()].rstrip()
            if not kept:
                continue
            line = kept + "%"
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    if not (PAPER / "main.bbl").exists():
        raise SystemExit("main.bbl missing: compile with BibTeX first")
    files: list[tuple[Path, str]] = []
    for pattern in TEX + BINARY:
        for p in sorted(PAPER.glob(pattern)):
            rel = p.relative_to(PAPER).as_posix()
            if rel not in SKIP:
                files.append((p, rel))
    with tarfile.open(OUT, "w:gz") as tar:
        for p, rel in files:
            if p.suffix == ".tex":
                data = strip_comments(p.read_text()).encode()
                info = tarfile.TarInfo(rel)
                info.size = len(data)
                info.mtime = int(p.stat().st_mtime)
                import io

                tar.addfile(info, io.BytesIO(data))
            else:
                tar.add(p, arcname=rel)
    size = OUT.stat().st_size / 1e6
    print(f"packed {len(files)} files into {OUT.name} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
