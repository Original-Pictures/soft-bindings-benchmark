"""Compare the ISCC rerun with the paper's recorded results and write results-iscc/RESULTS.md.

Reads results-gpu/ (the paper's records), results-iscc/iscc-sdk-0.9.5/ (this rerun with the ISCC versions the paper
ran) and results-iscc/iscc-sdk-0.9.7/ (the released update, which also holds the Semantic-Code methods the paper
did not report: ISCC-SCI-64, ISCC-SCI-256 and ISCC-Combined-512), including their security records, no-cleanup
duplicate-rule records and source-level bootstrap (iscc/stats.py). The last section rebuilds the paper's numbers with
its own paper/scripts/build_numbers.py from a copy of results-gpu/ whose ISCC Image-Code records come from the update,
and lists every number cited in the text that changes.

    uv run --no-project --with numpy==2.4.6 --with scipy==1.17.1 python iscc/compare.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
REPO = BENCH.parents[1]
PAPER = BENCH / "results-gpu"
OLD = BENCH / "results-iscc" / "iscc-sdk-0.9.5"
NEW = BENCH / "results-iscc" / "iscc-sdk-0.9.7"
OUT = BENCH / "results-iscc" / "RESULTS.md"
ISCC = ("ISCC-Image-64", "ISCC-Image-256")  # the paper's ISCC methods
ADDED = ("ISCC-SCI-64", "ISCC-SCI-256", "ISCC-Combined-512")  # run with the update only; the paper reports none
SCALE = ("PDQ", "DINOv2-B", "ISC21-1st")  # the paper's records, for scale
OP = "pair@1e-07"
NOT_RESULTS = {"host", "timing", "disc21_source"}  # host details and the paper's provenance note
VAL = re.compile(r"\\csname val@(.+?)\\endcsname\{(.*)\}$")
CITE = re.compile(r"\\val\{fp:([^}]*)\}")


# ------------------------------------------------------------------ inputs
def read_json(path: Path) -> dict:
    """JSON from a file or from its gzipped copy."""
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_name(path.name + ".gz").read_bytes()))


def image_records(results: Path) -> dict:
    """Method name -> image-track record."""
    return {p.stem: read_json(p) for p in sorted((results / "image").glob("*.json")) if not p.name.startswith("_")}


def wm_records(results: Path) -> dict:
    """Method name -> record of the image part of the watermark-combination track."""
    return read_json(results / "wmcombo" / "wmcombo.json")["image"]["methods"]


# ------------------------------------------------------------------ helpers
def differences(a, b, path: str = "") -> list[str]:
    """Paths at which two records differ; numbers are compared up to float32 rounding (muAP sums float32 scores)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted((a.keys() | b.keys()) - (NOT_RESULTS if not path else set()))
        return [d for k in keys for d in differences(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        same = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-9)
        return [] if same else [path]
    return [] if a == b else [path]


def pct(x) -> str:
    """A rate in percent with one decimal."""
    return "–" if x is None or not math.isfinite(x) else f"{100 * x:.1f}"


def max_bits(threshold, nbits: int) -> str:
    """Most bits that may differ for a match at a similarity threshold."""
    if threshold is None or not math.isfinite(threshold) or threshold > 1:
        return "none"
    return str(math.floor((1 - threshold) * nbits + 1e-9))


def table(header: list[str], rows: list[list[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def overall(rec: dict, key: str):
    """A pooled figure of an image-track record."""
    return rec["per_attack"]["_all"].get(key)


# ------------------------------------------------------------------ sections
def reproduction(paper: dict, old: dict, wm_paper: dict, wm_old: dict) -> str:
    """Whether the rerun with the paper's ISCC versions gives the paper's records."""
    lines = []
    for m in ISCC:
        diff = differences(paper[m], old[m])
        lines.append(f"- {m}, image track: " + ("identical to the paper's record: every figure, threshold and score "
                                                "histogram, up to float32 rounding." if not diff else
                                                f"{len(diff)} figures differ: {', '.join(diff[:12])}"))
    for m in ISCC:
        a, b = unattacked_hits(wm_paper[m]), unattacked_hits(wm_old[m])
        agree = sum(a[s] == b[s] for s in a)
        lines.append(f"- {m}, watermark track: the paper records {pct(wm_paper[m]['drift_below_threshold'])} % of "
                     f"marked photos past the threshold, this rerun {pct(wm_old[m]['drift_below_threshold'])} %; "
                     f"whether the unattacked marked copy finds its unmarked original agrees for {agree} of {len(a)} "
                     f"photos.")
    return "\n".join(lines)


def unattacked_hits(rec: dict) -> dict:
    """Source -> whether its unattacked marked copy finds the unmarked original at the threshold."""
    return {s: ok for s, a, ok in zip(rec["source"], rec["attack"], rec["fp_ok_registry_unmarked"]) if a == "none"}


def runs(paper: dict, new: dict) -> list[tuple[str, str, dict]]:
    """(method, run, record) rows: the paper's ISCC methods in the paper and with the update, the added methods
    with the update, and the paper's methods for scale."""
    rows = [(m, run, rec[m]) for m in ISCC for run, rec in (("paper", paper), ("update", new))]
    return rows + [(m, "update", new[m]) for m in ADDED] + [(m, "paper", paper[m]) for m in SCALE]


def bits_at(rec: dict, target: str) -> str:
    """Most bits that may differ for a match at a target, for binary codes."""
    meta = rec["method"]
    return max_bits(rec["thresholds"].get(target), meta["dim"]) if meta["metric"] == "hamming" else "–"


def image_summary(paper: dict, new: dict) -> str:
    """Headline figures of the image track per method and run."""
    rows = [[m, run, pct(overall(r, f"TPR@{OP}")), pct(overall(r, "TPR@pair@1e-08")), pct(overall(r, "TPR@query@0.01")),
             pct(overall(r, "R@1")), f"{overall(r, 'muAP'):.3f}", bits_at(r, OP), bits_at(r, "pair@1e-08"),
             pct(overall(r, f"FPRq@{OP}")), pct(r["hard_negative"].get(OP, {}).get("rate")), pct(r["disc21"]["R@1"]),
             f"{r['disc21']['muAP']:.3f}"] for m, run, r in runs(paper, new)]
    return table(["Method", "Run", "Detected at 10⁻⁷ (headline)", "Detected at 10⁻⁸", "Detected at query-level 1 %",
                  "Recall@1", "µAP", "Bits that may differ at 10⁻⁷", "Bits that may differ at 10⁻⁸",
                  "Held-out queries matched at 10⁻⁷", "Other photo of the product bound at 10⁻⁷", "DISC21 Recall@1",
                  "DISC21 µAP"], rows)


def ranking(paper: dict, new: dict) -> str:
    """All image methods by detection at the headline operating point, with the update's ISCC records."""
    merged = {**paper, **{m: new[m] for m in ISCC + ADDED}}
    # Ties are broken by case-folded name so the order does not depend on how the host sorts paths
    # (Windows paths compare case-insensitively, POSIX paths do not).
    order = sorted(merged, key=lambda m: (-overall(merged[m], f"TPR@{OP}"), m.casefold()))
    before = sorted(paper, key=lambda m: (-overall(paper[m], f"TPR@{OP}"), m.casefold()))
    rows = []
    for rank, m in enumerate(order, 1):
        note = (f"paper: {pct(overall(paper[m], f'TPR@{OP}'))} %, rank {before.index(m) + 1}" if m in ISCC else
                "added, not in the paper" if m in ADDED else "")
        rows.append([str(rank), m, merged[m]["method"]["tier"], pct(overall(merged[m], f"TPR@{OP}")), note])
    return table(["Rank", "Method", "Tier", "Detected at 10⁻⁷, %", ""], rows)


def per_attack(paper: dict, new: dict) -> str:
    """Recall@1 per transformation per method and run."""
    cols = [(m, run, r) for m, run, r in runs(paper, new) if m != "ISCC-SCI-64"]
    family = paper["PDQ"]["attack_family"]
    rows = [[a, family[a], *(pct(r["per_attack"][a]["R@1"]) for _, _, r in cols)] for a in family]
    return table(["Transformation", "Family", *(f"{m} {run}" for m, run, _ in cols)], rows)


def watermark(wm_paper: dict, wm_old: dict, wm_new: dict) -> str:
    """Watermark drift and detection of attacked watermarked copies per method: the paper, the rerun with the
    paper's ISCC versions and the update."""
    figures = [("Marked photos that no longer match their unmarked original", lambda r: r["drift_below_threshold"]),
               ("Attacked marked copies identified, registry holds the marked asset",
                lambda r: r["summary"]["_all"]["fp"]),
               ("Attacked marked copies identified, registry holds the unmarked original",
                lambda r: r["summary_unmarked_registry"]["_all"]["fp"])]
    rows = []
    for m in ISCC + ADDED + SCALE:
        recs = [wm.get(m) for wm in (wm_paper, wm_old, wm_new)]
        rows += [[m, label, *(pct(get(r)) if r else "–" for r in recs)] for label, get in figures]
    return table(["Method", "", "Paper", "Rerun, paper's ISCC versions", "Update"], rows)


RUNS = ("paper", "rerun, paper's ISCC versions", "update")
EPS = ("2", "4", "8", "16")
ANCHORS = ("PDQ", "DINOv2-S", "SSCD-mixup", "OpenCLIP-B32")  # scored by the rerun as reference points


def security_records(name: str) -> list[dict]:
    """A security record of the paper, the rerun with its ISCC versions and the update."""
    return [read_json(r / "security" / name) for r in (PAPER, OLD, NEW)]


def transfer(evs: list[dict]) -> str:
    """Attacked originals no longer matched after the ensemble transfer attack, per budget."""
    rows = []
    for m in ISCC + ("ISCC-SCI-256",) + ANCHORS:
        runs = (("paper", evs[0]), ("rerun", evs[1])) if m in ANCHORS else zip(RUNS, evs)
        for run, ev in runs:
            per = [ev["transfer"][e].get(m) for e in EPS]
            if any(per):
                rows.append([m, run, *(pct(p["summary"]["rate"]) if p else "–" for p in per)])
    return table(["Method", "Run", *(f"{e} grey levels" for e in EPS)], rows)


def rate_at(rows: list[dict], t: float):
    """Evasion rate at another threshold, among items matched before the attack (build_numbers' rescoring)."""
    ok = [r for r in rows if r["score_clean"] >= t]
    return sum(r["score_adv"] < t for r in ok) / len(ok) if ok else None


def surrogate(evs: list[dict], images: list[dict]) -> str:
    """ISCC-Image-64 evaded by the surrogate attack, per budget, at pair-level 10⁻⁶ and 10⁻⁷."""
    m = "ISCC-Image-64"
    rows = []
    for run, ev, rec in zip(RUNS, evs, images):
        per = ev["surrogate"][m]
        rows.append([run, *(f"{pct(per[e]['summary']['rate'])} / {pct(rate_at(per[e]['rows'], rec[m]['thresholds'][OP]))}"
                            for e in EPS)])
    return table(["Run", *(f"{e} grey levels" for e in EPS)], rows)


def collision(cos: list[dict]) -> str:
    """Unrelated images made to bind to a registered original as ISCC-Image-64, per budget."""
    m = "ISCC-Image-64"
    return table(["Run", "4 grey levels", "8 grey levels", "16 grey levels"],
                 [[run, *(pct(co["methods"][m][e]["summary"]["rate"]) for e in ("4", "8", "16"))]
                  for run, co in zip(RUNS, cos)])


def interval(x: dict, value: str, ci: str) -> str:
    """A rate with its 95 % bootstrap interval, in percent."""
    return f"{pct(x[value])} [{pct(x[ci][0])}, {pct(x[ci][1])}]"


def bootstrap(cis: list[dict]) -> str:
    """Source-level intervals of the ISCC methods and the paired tests that involve them."""
    rows = [[m, run, interval(ci["methods"][f"image/{m}"][OP], "tpr", "tpr_ci"),
             interval(ci["methods"][f"image/{m}"], "r1", "r1_ci")] for m in ISCC for run, ci in zip(RUNS, cis)]
    pairs = [[f"{r['a']} − {r['b']}", run, interval(r, "diff", "diff_ci"), f"{r['p_holm']:.4f}"]
             for run, ci in zip(RUNS, cis) for r in ci["paired"] if involves_iscc(r)]
    added = [[f"{r['a']} − {r['b']}", run, interval(r, "diff", "diff_ci"), f"{r['p_signflip']:.4f}"]
             for run, ci in zip(RUNS[1:], cis[1:]) for r in ci.get("added", [])]
    paired = [[r for r in ci["paired"] if involves_iscc(r)] for ci in cis[:2]]
    same = all(not differences(cis[0]["methods"][f"image/{m}"], cis[1]["methods"][f"image/{m}"]) for m in ISCC) and \
        not differences(*paired)
    check = ("With the paper's ISCC versions, `cluster_ci.py` reproduces the paper's ISCC intervals and paired tests."
             if same else "With the paper's ISCC versions, the ISCC intervals or paired tests DIFFER from the paper's.")
    return "\n\n".join([check,
                        table(["Method", "Run", "Detected at 10⁻⁷ [95 % CI]", "Recall@1 [95 % CI]"], rows),
                        "Planned paired comparisons that involve ISCC (difference in detection at 10⁻⁷, Holm-adjusted "
                        "p over the paper's eight comparisons):",
                        table(["A − B", "Run", "Difference [95 % CI]", "p (Holm)"], pairs),
                        "Added comparison, tested on its own with the same sign-flip test (not part of the paper's "
                        "planned family):",
                        table(["A − B", "Run", "Difference [95 % CI]", "p"], added)])


def sensitivity() -> str:
    """Detection of the ISCC methods without the near-duplicate cleanup."""
    rows = []
    for m in ISCC:
        for run, res in zip(RUNS, (PAPER, OLD, NEW)):
            p = read_json(res / "image_dedup-none" / f"{m}.json")["per_attack"]["_all"]
            rows.append([m, run, pct(p["TPR@query@0.01"]), pct(p[f"TPR@{OP}"])])
    return table(["Method", "Run", "Detected at query-level 1 %", "Detected at 10⁻⁷"], rows)


# ------------------------------------------------------------------ the paper's cited numbers
def merged_results(dst: Path, run: Path = NEW) -> Path:
    """A copy of results-gpu/ whose ISCC Image-Code records come from a rerun (by default the update): image and
    watermark tracks and, where the rerun has them, security, the no-cleanup duplicate rule and the source-level
    bootstrap."""
    shutil.copytree(PAPER, dst)
    for m in ISCC:
        shutil.copy(run / "image" / f"{m}.json", dst / "image" / f"{m}.json")
        shutil.copy(run / "image" / "_perquery" / f"{m}.npz", dst / "image" / "_perquery" / f"{m}.npz")
        if (run / "image_dedup-none" / f"{m}.json").exists():
            shutil.copy(run / "image_dedup-none" / f"{m}.json", dst / "image_dedup-none" / f"{m}.json")
    wm = read_json(PAPER / "wmcombo" / "wmcombo.json")
    wm["image"]["methods"].update({m: wm_records(run)[m] for m in ISCC})
    (dst / "wmcombo" / "wmcombo.json.gz").unlink()
    (dst / "wmcombo" / "wmcombo.json").write_text(json.dumps(wm))
    merge_security(dst / "security", run / "security")
    merge_cluster(dst / "_cluster_ci.json", run / "_cluster_ci.json")
    return dst


def merge_security(dst: Path, src: Path) -> None:
    """Put a rerun's ISCC entries into the paper's evasion and collision records."""
    if (src / "evasion.json").exists():
        ev, new = read_json(dst / "evasion.json"), read_json(src / "evasion.json")
        ev["surrogate"].update(new["surrogate"])
        ev["thresholds"].update({m: new["thresholds"][m] for m in ISCC})
        for eps, per in new["transfer"].items():
            ev["transfer"][eps].update({m: {k: v for k, v in per[m].items() if k != "rows"} for m in ISCC})
        (dst / "evasion.json").write_text(json.dumps(ev))
    if (src / "collision.json").exists():
        co = read_json(dst / "collision.json")
        co["methods"].update(read_json(src / "collision.json")["methods"])
        (dst / "collision.json").write_text(json.dumps(co))


def involves_iscc(row: dict) -> bool:
    """Whether a paired comparison involves one of the paper's ISCC methods."""
    return row["a"] in ISCC or row["b"] in ISCC


def merge_cluster(dst: Path, src: Path) -> None:
    """Put a rerun's ISCC intervals and its paired tests that involve ISCC into the paper's _cluster_ci.json."""
    if not src.exists():
        return
    ci, new = read_json(dst), read_json(src)
    ci["methods"].update({f"image/{m}": new["methods"][f"image/{m}"] for m in ISCC})
    pairs = {(r["a"], r["b"]): r for r in new["paired"]}
    ci["paired"] = [pairs[(r["a"], r["b"])] if involves_iscc(r) else r for r in ci["paired"]]
    dst.write_text(json.dumps(ci))


def rebuild(raw: Path) -> dict[str, str]:
    """File name -> content of every file build_numbers.py generates (numbers.tex and the tables) from a results
    directory."""
    env = {**os.environ, "PAPER_RESULTS": str(raw)}
    build = "import json, sys, build_numbers as b; json.dump({p.name: s for p, s in b.build().items()}, sys.stdout)"
    out = subprocess.run([sys.executable, "-c", build], cwd=BENCH / "paper" / "scripts", env=env, check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    return json.loads(out)


def values(numbers_tex: str) -> dict[str, str]:
    """Key -> value of every \\val in a numbers.tex."""
    return dict(m.groups() for m in map(VAL.search, numbers_tex.splitlines()) if m)


def paper_sections() -> Path:
    """Sections of the combined paper: paper/ at the root of the public repository, combined/paper/ in the
    private research repository it is exported from."""
    for d in (REPO / "paper" / "sections", REPO / "combined" / "paper" / "sections"):
        if d.is_dir():
            return d
    raise FileNotFoundError("no paper/sections or combined/paper/sections under " + str(REPO))


def citations() -> dict[str, list[str]]:
    """Key -> paper sections that cite it."""
    out: dict[str, list[str]] = {}
    for f in sorted(paper_sections().glob("*.tex")):
        for key in sorted(set(CITE.findall(f.read_text(encoding="utf-8")))):
            out.setdefault(key, []).append(f.stem)
    return out


def cited_changes() -> str:
    """Every number cited in the paper's text that changes with the update's ISCC records, after a check that
    the paper's results rebuild to its committed numbers.tex."""
    committed = (BENCH / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8")
    base = rebuild(PAPER)
    with tempfile.TemporaryDirectory() as tmp:
        update = rebuild(merged_results(Path(tmp) / "results"))
    check = ("Rebuilt from the unchanged `results-gpu/`, `numbers.tex` is identical to the committed one."
             if base["numbers.tex"] == committed else
             "Rebuilt from the unchanged `results-gpu/`, `numbers.tex` DIFFERS from the committed one.")
    tables = ", ".join(f"`{n}`" for n in sorted(base) if n != "numbers.tex" and base[n] != update[n])
    before, after = values(base["numbers.tex"]), values(update["numbers.tex"])
    rows = [[f"`{k}`", ", ".join(where), before.get(k, "–"), after.get(k, "–")]
            for k, where in sorted(citations().items()) if before.get(k) != after.get(k)]
    return "\n\n".join([check, table(["Number (`\\val{fp:…}`)", "Cited in", "Paper", "With the update"], rows),
                        f"Generated tables that change: {tables or 'none'}."])


# ------------------------------------------------------------------ report
def report() -> str:
    paper, old, new = image_records(PAPER), image_records(OLD), image_records(NEW)
    wm_paper, wm_old, wm_new = wm_records(PAPER), wm_records(OLD), wm_records(NEW)
    evs, cos = security_records("evasion.json"), security_records("collision.json")
    cis = [read_json(r / "_cluster_ci.json") for r in (PAPER, OLD, NEW)]
    return "\n\n".join([
        "# ISCC rerun: results",
        "Generated by `iscc/compare.py` from `results-gpu/` (the paper), `results-iscc/iscc-sdk-0.9.5/` (this "
        "rerun with the ISCC versions the paper ran: iscc-sdk 0.9.5, iscc-core 1.4.0, iscc-lib 0.6.0) and "
        "`results-iscc/iscc-sdk-0.9.7/` (the update: iscc-sdk 0.9.7, iscc-core 1.5.0, iscc-lib 0.7.0, iscc-sci 0.4.1). "
        "The update also runs the Semantic-Code, which the paper measured in its pilot only: ISCC-SCI-256 with the "
        "paper's own adapter, ISCC-SCI-64 as its first 64 bits, and ISCC-Combined-512 as ISCC-SCI-256 followed by "
        "ISCC-Image-256, compared as one code. Each method has its own calibrated thresholds. Rates in percent. "
        f"{', '.join(SCALE)} are the paper's records, for scale.",
        "## The rerun reproduces the paper", reproduction(paper, old, wm_paper, wm_old),
        "## Image track", image_summary(paper, new),
        "## Ranking at the headline operating point", ranking(paper, new),
        "## Recall@1 per transformation", per_attack(paper, new),
        "## Watermark-combination track (TrustMark Q, 300 product photos)", watermark(wm_paper, wm_old, wm_new),
        "## Source-level bootstrap and paired tests", bootstrap(cis),
        "## Security track: evasion and forgery",
        "The paper's image attacks that involve ISCC, on its first 100 registered originals with its seeds, budgets "
        "(grey levels of 255) and 40 PGD steps. Success is judged by the real extractor at the pair-level 10⁻⁶ "
        "threshold of each run's own record, as `bench_security.py` does.",
        "Transfer attack crafted on DINOv2-S, SSCD-mixup and OpenCLIP-B32 (the attacker never models ISCC): share "
        "of originals that no longer match. The perturbations were made once and scored with both ISCC versions; "
        "PDQ and the ensemble methods were scored at the paper's thresholds as reference points.", transfer(evs),
        "Surrogate attack on ISCC-Image-64 (gradients from `hash_surrogates.py`, whose trim box follows the installed "
        "iscc-sdk): share evaded at 10⁻⁶ / at 10⁻⁷.", surrogate(evs, [paper, old, new]),
        "Forgery: unrelated images made to become the top match of a registered original above the 10⁻⁶ threshold "
        "(ISCC-Image-64).", collision(cos),
        "## Duplicate-rule sensitivity: no near-duplicate cleanup",
        "The \"two of three\" rule cannot be rerun: it needs the duplicate evidence of the paper's work directory, "
        "which the repository does not carry.", sensitivity(),
        "## Numbers cited in the paper that change",
        "Rebuilt with the paper's `build_numbers.py` from `results-gpu/` with the update's ISCC records in place of "
        "the paper's: image and watermark tracks, source-level bootstrap and paired tests, security (evasion and "
        "collision) and the no-cleanup duplicate rule. Edit retrieval and localisation, latency, cross-host "
        "stability, inversion and the \"two of three\" duplicate rule keep the paper's ISCC values.",
        cited_changes(),
    ]) + "\n"


if __name__ == "__main__":
    OUT.write_text(report(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
