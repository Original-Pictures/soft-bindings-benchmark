"""Assemble the unified paper's generated inputs from the two benchmark papers.

The watermarking and fingerprinting papers each regenerate their numbers, tables and
figures from their own results (their build.sh). This script copies those products into
combined/paper under a prefix, so that the two benchmarks' labels and values cannot collide:

  \\val{key}          -> \\val{wm:key} / \\val{fp:key}       generated/numbers.tex
  tab:x, sec:x, fig:x -> tab:wm-x, sec:wm-x, fig:wm-x         generated/tables/{wm,fp}/*.tex
  figures/name.pdf    -> figures/wm_name.pdf, figures/fp_name.pdf

Nothing here computes a number. Run after both papers' build_numbers.py and figures.py.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
SOURCES = {"wm": ROOT / "watermarking" / "bench-2026-09" / "paper",
           "fp": ROOT / "fingerprinting" / "bench-2026-10" / "paper"}
VAL = re.compile(r"\\expandafter\\def\\csname val@(.*?)\\endcsname\{(.*)\}\s*$")


def prefix_labels(text: str, p: str) -> str:
    """Namespace LaTeX labels and references of one source paper."""
    return re.sub(r"\\(label|ref|autoref|eqref)\{(tab|sec|fig|eq|app):([^}]*)\}",
                  lambda m: rf"\{m.group(1)}{{{m.group(2)}:{p}-{m.group(3)}}}", text)


def prefix_values(text: str, p: str) -> str:
    return re.sub(r"\\val\{([^}]*)\}", lambda m: rf"\val{{{p}:{m.group(1)}}}", text)


# Combined-study terminology; numerical table entries remain unchanged.
WM_CAPTION_ORIGINAL = r"""\caption{Watermark and fingerprint on the same attacked copies of watermarked assets, pooled over all transformations. WM: the watermark yields the embedded key; FP: the fingerprint identifies the asset at its operating point (images pair-level $10^{-7}$, audio and video query-level 1\%); either: the cascade recovers the asset. Drift: share of watermarked assets whose own fingerprint falls below the threshold against the unmarked original.}"""
WM_CAPTION_COMBINED = r"""\caption{Watermark and fingerprint on the same attacked copies of watermarked assets, pooled over transformations. WM: exact-key recovery for TrustMark and AudioSeal, but expected-key verification for Video Seal; the latter does not establish unique registry identification. FP: correct fingerprint identification at a threshold calibrated to pair-level FPR $10^{-7}$ for images or query-level FPR 1\% for audio and video; achieved held-out rates are reported separately. Both and Either are the intersection and union of these success criteria, not measured cascade accuracy. Drift: share of watermarked assets whose own fingerprint falls below the threshold against the unmarked original.}"""


def main() -> int:
    out = [r"% Assembled by scripts/assemble.py from the two benchmark papers. Do not edit.",
           r"\makeatletter",
           r"\newcommand{\val}[1]{\ifcsname val@#1\endcsname\csname val@#1\endcsname\else"
           r"\PackageError{numbers}{Undefined value #1}{Run the source papers' build_numbers.py and assemble.py}\fi}"]
    for p, src in SOURCES.items():
        n = 0
        for line in (src / "generated" / "numbers.tex").read_text().splitlines():
            m = VAL.match(line)
            if m:
                out.append(rf"\expandafter\def\csname val@{p}:{m.group(1)}\endcsname{{{m.group(2)}}}")
                n += 1
        tdir = HERE / "generated" / "tables" / p
        tdir.mkdir(parents=True, exist_ok=True)
        for t in sorted((src / "generated" / "tables").glob("*.tex")):
            table = prefix_values(prefix_labels(t.read_text(), p), p)
            if p == "fp" and t.name == "wm.tex":
                if table.count(WM_CAPTION_ORIGINAL) != 1:
                    raise ValueError("Watermark table caption changed; review the combined-study wording")
                table = table.replace(WM_CAPTION_ORIGINAL, WM_CAPTION_COMBINED)
            (tdir / t.name).write_text(table)
        for f in sorted((src / "figures").glob("*.pdf")):
            shutil.copy2(f, HERE / "figures" / f"{p}_{f.name}")
        for f in sorted((src / "figures").glob("*.tex")):
            (HERE / "figures" / f"{p}_{f.name}").write_text(prefix_values(prefix_labels(f.read_text(), p), p))
        print(f"{p}: {n} values, tables and figures copied")
    # One bibliography: the union of both papers' entries (shared keys are identical entries).
    bib, seen = [], set()
    for src in SOURCES.values():
        for m in re.finditer(r"(@\w+\{([^,]+),.*?\n\})", (src / "refs.bib").read_text(), re.S):
            if m.group(2).strip() not in seen:
                seen.add(m.group(2).strip())
                bib.append(m.group(1))
    (HERE / "refs.bib").write_text("% Assembled by scripts/assemble.py from both papers' refs.bib. Do not edit.\n\n"
                                   + "\n\n".join(bib) + "\n")
    print(f"refs.bib: {len(bib)} entries")
    out.append(r"\makeatother")
    (HERE / "generated" / "numbers.tex").write_text("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
