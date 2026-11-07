"""Tests for compare.py on the committed records of the paper and of this rerun.

    uv run --no-project --with pytest --with numpy==2.4.6 --with scipy==1.17.1 --with pillow==12.3.0 pytest iscc
"""

import json

import compare as c


def test_differences_of_identical_records_are_empty():
    rec = c.image_records(c.PAPER)["PDQ"]
    assert c.differences(rec, rec) == []


def test_differences_tolerate_float32_rounding_only():
    assert c.differences({"a": 0.090875014}, {"a": 0.090875231}) == []
    assert c.differences({"a": 0.4266}, {"a": 0.4391}) == ["/a"]


def test_differences_ignore_host_details_at_top_level_only():
    assert c.differences({"host": "x", "timing": 1}, {"host": "y", "timing": 2}) == []
    assert c.differences({"r": {"host": "x"}}, {"r": {"host": "y"}}) == ["/r/host"]


def test_max_bits_from_thresholds():
    assert c.max_bits(0.937501, 64) == "3"
    assert c.max_bits(0.9296885, 256) == "17"
    assert c.max_bits(0.8671885, 256) == "33"
    assert c.max_bits(1.000001, 64) == "none"


def test_values_parse_committed_numbers():
    vals = c.values((c.BENCH / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8"))
    assert vals["wm/image/ISCC-Image-64/drift/pct"] == "62.0"
    assert vals["image/ISCC-Image-64/TPR@pair@1e-07/pct"] == "42.7"


def test_citations_include_the_abstract_drift():
    assert "abstract" in c.citations()["wm/image/ISCC-Image-64/drift/pct"]


def test_unattacked_hits_cover_every_marked_photo():
    assert len(c.unattacked_hits(c.wm_records(c.PAPER)["ISCC-Image-64"])) == 300


def test_rerun_with_paper_versions_matches_paper_records():
    paper, old = c.image_records(c.PAPER), c.image_records(c.OLD)
    assert all(c.differences(paper[m], old[m]) == [] for m in c.ISCC)


def test_rebuild_reproduces_committed_numbers():
    committed = (c.BENCH / "paper" / "generated" / "numbers.tex").read_text(encoding="utf-8")
    assert c.rebuild(c.PAPER)["numbers.tex"] == committed


def test_rate_at_rescores_items_matched_before_the_attack():
    rows = [{"score_clean": 0.95, "score_adv": 0.90}, {"score_clean": 0.95, "score_adv": 0.96},
            {"score_clean": 0.92, "score_adv": 0.80}]
    assert c.rate_at(rows, 0.94) == 0.5
    assert c.rate_at(rows, 0.99) is None


def test_involves_iscc():
    assert c.involves_iscc({"a": "PDQ", "b": "ISCC-Image-64"})
    assert not c.involves_iscc({"a": "PDQ", "b": "PDQ-dihedral"})


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_merge_security_replaces_only_iscc_entries(tmp_path):
    paper = {"surrogate": {"PDQ": 1, "ISCC-Image-64": 1}, "thresholds": {"PDQ": [0.8, "p"], "ISCC-Image-64": [0.9, "p"],
             "ISCC-Image-256": [0.88, "p"]}, "transfer": {"2": {"PDQ": {"summary": 1}, "ISCC-Image-64": {"summary": 1},
                                                                "ISCC-Image-256": {"summary": 1}}}}
    rerun = {"surrogate": {"ISCC-Image-64": 2}, "thresholds": {"PDQ": [0.7, "q"], "ISCC-Image-64": [0.91, "p"],
             "ISCC-Image-256": [0.87, "p"]}, "transfer": {"2": {"PDQ": {"summary": 2}, "ISCC-Image-64": {"summary": 2,
                                                                "rows": []}, "ISCC-Image-256": {"summary": 2}}}}
    write(tmp_path / "dst" / "evasion.json", paper)
    write(tmp_path / "dst" / "collision.json", {"methods": {"PDQ": 1, "ISCC-Image-64": 1}})
    write(tmp_path / "src" / "evasion.json", rerun)
    write(tmp_path / "src" / "collision.json", {"methods": {"ISCC-Image-64": 2}})
    c.merge_security(tmp_path / "dst", tmp_path / "src")
    ev = c.read_json(tmp_path / "dst" / "evasion.json")
    assert ev["surrogate"] == {"PDQ": 1, "ISCC-Image-64": 2}
    assert ev["thresholds"]["PDQ"] == [0.8, "p"] and ev["thresholds"]["ISCC-Image-64"] == [0.91, "p"]
    assert ev["transfer"]["2"] == {"PDQ": {"summary": 1}, "ISCC-Image-64": {"summary": 2},
                                   "ISCC-Image-256": {"summary": 2}}
    assert c.read_json(tmp_path / "dst" / "collision.json")["methods"] == {"PDQ": 1, "ISCC-Image-64": 2}


def test_merge_cluster_replaces_only_iscc_entries(tmp_path):
    rows = [{"a": "PDQ", "b": "ISCC-Image-64", "v": 1}, {"a": "PDQ", "b": "PDQ-dihedral", "v": 1}]
    write(tmp_path / "paper.json", {"methods": {"image/PDQ": 1, "image/ISCC-Image-64": 1, "image/ISCC-Image-256": 1},
                                    "paired": rows})
    write(tmp_path / "rerun.json", {"methods": {"image/PDQ": 2, "image/ISCC-Image-64": 2, "image/ISCC-Image-256": 2},
                                    "paired": [{**r, "v": 2} for r in rows]})
    c.merge_cluster(tmp_path / "paper.json", tmp_path / "rerun.json")
    ci = c.read_json(tmp_path / "paper.json")
    assert ci["methods"] == {"image/PDQ": 1, "image/ISCC-Image-64": 2, "image/ISCC-Image-256": 2}
    assert [r["v"] for r in ci["paired"]] == [2, 1]


def test_merged_results_carry_the_update(tmp_path):
    merged = c.merged_results(tmp_path / "results")
    assert c.image_records(merged)["ISCC-Image-256"] == c.image_records(c.NEW)["ISCC-Image-256"]
    assert c.wm_records(merged)["PDQ"] == c.wm_records(c.PAPER)["PDQ"]
    assert c.wm_records(merged)["ISCC-Image-64"] == c.wm_records(c.NEW)["ISCC-Image-64"]
    ev, new, paper = (c.read_json(r / "security" / "evasion.json") for r in (merged, c.NEW, c.PAPER))
    assert ev["transfer"]["2"]["ISCC-Image-64"]["summary"] == new["transfer"]["2"]["ISCC-Image-64"]["summary"]
    assert ev["transfer"]["2"]["PDQ"] == paper["transfer"]["2"]["PDQ"]
    ci = c.read_json(merged / "_cluster_ci.json")
    assert ci["methods"]["image/ISCC-Image-256"] == c.read_json(c.NEW / "_cluster_ci.json")["methods"]["image/ISCC-Image-256"]
    assert ci["methods"]["image/PDQ"] == c.read_json(c.PAPER / "_cluster_ci.json")["methods"]["image/PDQ"]


def outcomes(per_eps: dict) -> dict:
    """Budget -> per-item (success, adversarial score, steps) of an attack record; LPIPS and PSNR are left out,
    because they differ in the last digits between CPU and GPU."""
    return {e: [(r["success"], r["score_adv"], r["steps"]) for r in x["rows"]] for e, x in per_eps.items()
            if e.isdigit()}


def test_rerun_with_paper_versions_reproduces_security_and_bootstrap():
    paper, old = c.security_records("collision.json")[:2]
    assert outcomes(paper["methods"]["ISCC-Image-64"]) == outcomes(old["methods"]["ISCC-Image-64"])
    ev_paper, ev_old = c.security_records("evasion.json")[:2]
    assert outcomes(ev_paper["surrogate"]["ISCC-Image-64"]) == outcomes(ev_old["surrogate"]["ISCC-Image-64"])
    ci_paper, ci_old = (c.read_json(r / "_cluster_ci.json") for r in (c.PAPER, c.OLD))
    assert all(c.differences(ci_paper["methods"][f"image/{m}"], ci_old["methods"][f"image/{m}"]) == [] for m in c.ISCC)
