"""Source-level bootstrap intervals and paired tests (the paper's cluster_ci.py) with the ISCC rerun's records.

Runs paper/scripts/cluster_ci.py, unchanged, on a copy of results-gpu/ whose ISCC records come from one rerun
(compare.merged_results) and keeps its output as results-iscc/iscc-sdk-<version>/_cluster_ci.json. With the paper's
ISCC versions the output is checked against the paper's _cluster_ci.json.

The paper's planned comparisons pair PDQ with ISCC-Image-64 only. The comparison of PDQ with ISCC-Image-256, which
the update brings level, is tested on its own with the same functions (not part of the paper's Holm family) and
kept under "added".

    python stats.py <iscc-sdk version>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import compare

SCRIPTS = compare.BENCH / "paper" / "scripts"
ADDED_PAIRS = [("PDQ", "ISCC-Image-256")]
ADDED = f"""
import json, sys, analysis as A, cluster_ci as C
C.PAIRED = {ADDED_PAIRS!r}
recs = {{m.name: m for m in A.load()["image"]}}
pqs = {{n: C.PerQuery(A.RAW / "image" / "_perquery" / f"{{n}}.npz", recs[n].rec["n_ref"]) for p in C.PAIRED for n in p}}
json.dump(C.paired(pqs, A.OP, 2000, C.SEED), sys.stdout)
"""


def run(raw: Path, *args: str) -> str:
    """Run a Python program in paper/scripts with PAPER_RESULTS set to a results directory."""
    env = {**os.environ, "PAPER_RESULTS": str(raw)}
    return subprocess.run([sys.executable, *args], cwd=SCRIPTS, env=env, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def main(version: str) -> None:
    results = compare.BENCH / "results-iscc" / f"iscc-sdk-{version}"
    with tempfile.TemporaryDirectory() as tmp:
        raw = compare.merged_results(Path(tmp) / "results", results)
        run(raw, "cluster_ci.py")
        out = json.loads((raw / "_cluster_ci.json").read_text())
        out["added"] = json.loads(run(raw, "-c", ADDED))
    (results / "_cluster_ci.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {results / '_cluster_ci.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
