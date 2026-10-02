#!/usr/bin/env bash
# Build the unified paper from the two benchmark papers' results.
#
#   ./build.sh           regenerate both papers' numbers and tables (their build_numbers.py),
#                        assemble them here under wm:/fp: prefixes, compile with pdfLaTeX in the
#                        same TeX Live container as the source papers, pack arXiv, and recompile
#                        the packed source without BibTeX.
#
# Figures are taken as the source papers last rendered them (their build.sh renders them).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
IMAGE=texlive/texlive:latest-medium
# tlmgr's mirror sometimes refuses a request that follows another closely; retry, and stop with a clear
# message rather than letting pdflatex fail silently on a missing newtx.
TEX_SETUP='for i in 1 2 3 4 5; do kpsewhich newtxmath.sty >/dev/null && break; tlmgr install newtx fontaxes >/dev/null 2>&1 || sleep 10; done; kpsewhich newtxmath.sty >/dev/null || { echo "newtx could not be installed" >&2; exit 1; }'

(cd "$ROOT/watermarking/bench-2026-09/paper/scripts" && uv run -q --with numpy --with scipy python build_numbers.py)
(cd "$ROOT/fingerprinting/bench-2026-10/paper/scripts" && uv run -q --with numpy --with scipy python build_numbers.py)
python3 scripts/assemble.py

docker run --rm -v "$PWD":/work -w /work "$IMAGE" sh -c "$TEX_SETUP;
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null &&
  bibtex main >/dev/null &&
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null &&
  pdflatex -interaction=nonstopmode -halt-on-error main | grep -E 'Warning|Overfull|undefined' || true"

python3 scripts/pack_arxiv.py

# A fresh directory name each time: Docker Desktop can serve a stale view of a bind-mounted directory
# that was deleted and recreated moments earlier, and pdflatex then never starts.
CHECK=$(mktemp -d .arxiv-check.XXXXXX)
tar -xzf arxiv-submission.tar.gz -C "$CHECK"
docker run --rm -v "$PWD/$CHECK":/work -w /work "$IMAGE" sh -c "$TEX_SETUP;
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null &&
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null && echo 'arXiv-style compile: OK'" ||
  { echo "arXiv-style compile FAILED:" >&2; grep -A8 '^!' "$CHECK/main.log" >&2 || tail -20 "$CHECK/main.log" >&2; exit 1; }
cp "$CHECK/main.pdf" ../soft-bindings-preprint.pdf
rm -rf "$CHECK"
echo "built paper/main.pdf, paper/arxiv-submission.tar.gz and soft-bindings-preprint.pdf"
