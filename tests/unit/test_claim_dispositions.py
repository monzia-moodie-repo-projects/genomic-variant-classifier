"""Package 1D, stage 1D-0: scientific claim dispositions (owner ruling 2026-11-08). Author: Monzia Moodie.

The registry (docs/claims/) is read and verified as committed; every reader refusal is exercised on a tampered copy; the
current-claim documents may cite a Run 15 figure only beside its disposition; the corrected historical record opens with
its correction; the live holdout design withholds its verdict; and the committed records are exactly what the committed
derivation writes from the evidence they hold.
"""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation import claim_disposition as cd
from genomic_variant_classifier.evaluation import run15_claim_evidence as ev

import derive_run15_claim_dispositions as derive  # scripts/ is on sys.path (tests/conftest.py)

ROOT = Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "docs" / "claims"
IDS = ("run15.c3.unseen_gene_generalization", "run15.test.ensemble_evaluation", "run15.validation.ensemble_evaluation")


@pytest.fixture(scope="module")
def registry() -> cd.ClaimRegistry:
    return cd.load_registry(ROOT)


def _copy(tmp_path: Path) -> Path:
    dst = tmp_path / "claims"
    shutil.copytree(CLAIMS, dst)
    return dst


def _rewrite(path: Path, mutate, *, redigest: bool) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))
    mutate(rec)
    if redigest:
        rec["record_sha256"] = cd.record_digest(rec)
    path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def _first(d: Path) -> Path:
    return d / "CLAIM-0001_run15.c3.unseen_gene_generalization.json"


# --------------------------------------------------------------------------------------------------------------------
# The committed registry
# --------------------------------------------------------------------------------------------------------------------

def test_the_committed_registry_verifies_and_holds_the_three_run15_claims(registry):
    assert registry.claim_ids == tuple(sorted(IDS))
    assert [r.sequence for r in registry.records] == [1, 2, 3]
    assert registry.records[0].previous_record_sha256 is None
    for prev, rec in zip(registry.records, registry.records[1:]):
        assert rec.previous_record_sha256 == prev.record_sha256
    for cid in IDS:
        rec = registry.current(cid)
        assert rec.disposition is cd.ClaimDisposition.INVALIDATED_EVALUATION and rec.eligible is False
        assert rec.raw["inflation_estimate"] is None
        assert {"GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE", "GENE_IDENTITY_SPLIT_LEAKAGE"} <= set(rec.defect_ids)
    assert "GNN_HOLDOUT_LABEL_LEAKAGE" in registry.current(IDS[0]).defect_ids
    assert "GNN_HOLDOUT_LABEL_LEAKAGE" not in registry.current(IDS[1]).defect_ids


@pytest.mark.parametrize("cid", IDS)
def test_an_invalidated_claim_cannot_be_required_as_evidence(registry, cid):
    with pytest.raises(cd.InadmissibleClaimError, match="INVALIDATED_EVALUATION"):
        registry.require_admissible(cid)


def test_an_unknown_claim_is_refused_not_assumed_valid(registry):
    with pytest.raises(cd.ClaimRegistryError, match="no claim record"):
        registry.current("run16.test.ensemble_evaluation")


def test_the_measured_own_label_identity_holds_on_every_evaluated_row(registry):
    """The evidence itself: 0 mismatches on all three evaluations, and the rows are the run's own split sizes."""
    rows = {}
    for cid in IDS:
        d = {x["defect_id"]: x["evidence"] for x in registry.current(cid).raw["invalidating_defects"]}
        own = d["GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE"]
        assert own["mismatches"] == 0 and own["feature_auroc_alone"] > 0.7
        rows[cid] = own["rows"]
    assert rows == {IDS[0]: 213436, IDS[1]: 304711, IDS[2]: 146329}


def test_every_cited_figure_is_a_metric_of_its_own_record(registry):
    """The cited figures are the record's historical metrics at four decimals -- not a hand list that can drift."""
    for cid in IDS:
        rec = registry.current(cid)
        metrics = rec.raw["historical_observation"]["metrics"]
        assert set(rec.cited_figures) == {f"{v:.4f}" for v in metrics.values()}


def test_the_records_bind_their_inputs_and_the_run_code(registry):
    for cid in IDS:
        obs = registry.current(cid).raw["historical_observation"]
        paths = [a["path"] for a in obs["source_artifacts"]]
        assert len(paths) == len(set(paths)) == len(ev.ARTIFACTS)
        prov = obs["code_provenance"]
        assert prov["run_code_commit_pre_rewrite"] == "4b9de1c319739824955bb165c295e56e444626ce"
        facts = prov["facts"]
        assert facts["count_computed_before_split"]["line"] < facts["gene_symbol_string_split"]["line"]
        assert [(f["pre_rewrite"], f["rewritten"]) for f in prov["leakage_fixes_after_the_run"]] == list(derive.FIXES)


def test_the_committed_records_are_exactly_what_the_derivation_writes_from_their_evidence(registry):
    """Rebuild every record with the committed code from the evidence the records hold; the bytes must be identical.
    A hand edit to any figure, statement or order would differ (the full re-derivation from Run 15's files is
    scripts/derive_run15_claim_dispositions.py --check, run where the files are)."""
    c3, test, val = (registry.current(c).raw for c in IDS)

    def evidence(rec):
        d = {x["defect_id"]: x["evidence"] for x in rec["invalidating_defects"]}
        lovd = d.get("VARIANT_LEVEL_CLINICAL_CLASS_FEATURE", {"annotated_evaluated_rows": 0})
        return {"own_label_count": d["GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE"],
                "gene_token_overlap": d["GENE_IDENTITY_SPLIT_LEAKAGE"],
                "lovd_annotated_rows": lovd["annotated_evaluated_rows"]}, d, lovd

    m_c3, d_c3, lovd_c3 = evidence(c3)
    m_c3["gnn_order_in_log"] = d_c3["GNN_HOLDOUT_LABEL_LEAKAGE"]
    m_test, _, _ = evidence(test)
    m_val, _, _ = evidence(val)
    crit = val["historical_observation"]["criterion"]
    line_no = int(crit.split("line ")[1].split(" ")[0])
    measured = {
        "artifacts": c3["historical_observation"]["source_artifacts"],
        "metrics_json": {**test["historical_observation"]["metrics"],
                         **{f"val_{k}": v for k, v in val["historical_observation"]["metrics"].items()}},
        "unseen_gene_holdout_json": c3["historical_observation"]["metrics"],
        "c3_holdout": m_c3, "test": m_test, "validation": m_val, "lovd_log_line": lovd_c3["run_log"],
        "validation_verdict_line": {"line": line_no, "timestamp": None, "text": crit.split(": ", 1)[1]},
    }
    rebuilt = ev.build_records(measured, recorded_utc=c3["recorded_utc"],
                               code_provenance=c3["historical_observation"]["code_provenance"])
    for rec, rb in zip(registry.records, rebuilt):
        assert rec.path.read_bytes() == ev.render_record(rb).encode("ascii"), rec.path.name


def test_the_record_files_are_ascii_lf_with_one_trailing_newline():
    for p in sorted(CLAIMS.glob("CLAIM-*.json")):
        raw = p.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"}\n") and not raw.endswith(b"\n\n")
        raw.decode("ascii")


# --------------------------------------------------------------------------------------------------------------------
# Every refusal of the reader, on a tampered copy
# --------------------------------------------------------------------------------------------------------------------

def test_a_copy_of_the_registry_loads(tmp_path):
    assert cd.load_registry_dir(_copy(tmp_path)).claim_ids == tuple(sorted(IDS))


def test_an_edited_record_breaks_its_own_digest(tmp_path):
    d = _copy(tmp_path)
    _rewrite(_first(d), lambda r: r["historical_observation"]["metrics"].update(auroc=0.5), redigest=False)
    with pytest.raises(cd.ClaimRegistryError, match="canonical digest"):
        cd.load_registry_dir(d)


def test_a_consistently_rewritten_past_record_breaks_the_chain(tmp_path):
    d = _copy(tmp_path)
    _rewrite(_first(d), lambda r: r["historical_observation"]["metrics"].update(auroc=0.5), redigest=True)
    with pytest.raises(cd.ClaimRegistryError, match="breaks the chain"):
        cd.load_registry_dir(d)


def test_a_gap_in_the_sequence_is_refused(tmp_path):
    d = _copy(tmp_path)
    (d / "CLAIM-0002_run15.test.ensemble_evaluation.json").unlink()
    with pytest.raises(cd.ClaimRegistryError, match="sequence 3 where 2 is next"):
        cd.load_registry_dir(d)


def test_a_file_name_naming_another_claim_is_refused(tmp_path):
    d = _copy(tmp_path)
    _first(d).rename(d / "CLAIM-0001_run15.c3.something_else.json")
    with pytest.raises(cd.ClaimRegistryError, match="the file name names"):
        cd.load_registry_dir(d)


def test_a_stray_file_is_refused(tmp_path):
    d = _copy(tmp_path)
    (d / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(cd.ClaimRegistryError, match="not claim records"):
        cd.load_registry_dir(d)


@pytest.mark.parametrize("payload,match", [
    ('{"schema": 1, "schema": 2}', "duplicate key"),
    ('{"a": NaN}', "non-finite literal"),
    ('{"a": ', "not JSON"),
])
def test_malformed_json_is_refused(tmp_path, payload, match):
    d = _copy(tmp_path)
    _first(d).write_text(payload, encoding="utf-8")
    with pytest.raises(cd.ClaimRegistryError, match=match):
        cd.load_registry_dir(d)


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(inflation_estimate=0.01), "no inflation estimate"),
    (lambda r: r.update(scientific_claim_eligible=True), "scientific_claim_eligible"),
    (lambda r: r.update(invalidating_defects=[]), "at least one defect"),
    (lambda r: r.update(hypothesis_status="REFUTED"), "resolves no hypothesis"),
    (lambda r: r.update(disposition="SUPERSEDED", scientific_claim_eligible=False, invalidating_defects=[]),
     "lists no invalidating defect|names its"),
    (lambda r: r.update(replacement_claim_id="run16.c3.x"), "only a SUPERSEDED claim"),
    (lambda r: r.update(cited_figures=["0.9988", "0.9988"]), "duplicate entries"),
    (lambda r: r.update(cited_figures=["99%"]), "does not match"),
    (lambda r: r.update(extra_field=1), "unexpected"),
    (lambda r: r["historical_observation"]["metrics"].update(auroc=True), "JSON number"),
    (lambda r: r["historical_observation"].update(verdict="GREEN"), "verdict"),
    (lambda r: r["claim"].update(claim_id="Run15 C3"), "does not match"),
    (lambda r: r.update(recorded_utc="2026-11-08 14:00"), "recorded_utc"),
])
def test_schema_violations_are_refused_even_with_a_valid_digest(tmp_path, mutate, match):
    d = _copy(tmp_path)
    _rewrite(_first(d), mutate, redigest=True)
    with pytest.raises(cd.ClaimRegistryError, match=match):
        cd.load_registry_dir(d)


def test_a_partial_replacement_must_partition_the_defects(tmp_path):
    d = _copy(tmp_path)
    p = d / "CLAIM-0002_run15.test.ensemble_evaluation.json"
    _rewrite(p, lambda r: r["partial_replacements"][0].update(does_not_address_defects=[]), redigest=True)
    with pytest.raises(cd.ClaimRegistryError, match="partition"):
        cd.load_registry_dir(d)


def test_a_superseded_claim_must_name_a_different_replacement():
    rec = json.loads(_first(CLAIMS).read_text(encoding="utf-8"))
    rec.update(disposition="SUPERSEDED", invalidating_defects=[], partial_replacements=[],
               hypothesis_status="NOT_RESOLVED_BY_THIS_EXPERIMENT", replacement_claim_id=rec["claim"]["claim_id"])
    rec["record_sha256"] = cd.record_digest(rec)
    with pytest.raises(cd.ClaimRegistryError, match="different"):
        cd.validate_record(rec)
    rec["replacement_claim_id"] = "run17.c3.unseen_gene_generalization"
    rec["record_sha256"] = cd.record_digest(rec)
    assert cd.validate_record(rec).disposition is cd.ClaimDisposition.SUPERSEDED


# --------------------------------------------------------------------------------------------------------------------
# The documents
# --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("doc", cd.GOVERNED_DOCUMENTS)
def test_no_current_document_cites_an_invalidated_figure_without_its_disposition(registry, doc):
    path = ROOT / doc
    assert path.is_file(), doc
    found = cd.find_unqualified_citations(path.read_text(encoding="utf-8"), registry)
    assert not found, "\n".join(f"{doc}:{c.line}: {c.figure} (claims {c.claims}) -- {c.text!r}" for c in found)


def test_the_governed_documents_do_cite_run15_so_the_check_is_not_vacuous(registry):
    figures = registry.inadmissible_figures()
    cited = [doc for doc in cd.GOVERNED_DOCUMENTS
             if any(f in (ROOT / doc).read_text(encoding="utf-8") for f in figures)]
    assert {"README.md", "docs/METRICS.md", "docs/runs/RUN17_SCOPE.md"} <= set(cited)


def test_the_citation_check_finds_bare_and_wrongly_qualified_citations(registry):
    bare = "Run 15 reached 0.9984 on test.\n\nunrelated paragraph"
    assert [c.figure for c in cd.find_unqualified_citations(bare, registry)] == ["0.9984"]
    wrong_claim = "0.9988 INVALIDATED_EVALUATION (run15.test.ensemble_evaluation)"
    assert [c.figure for c in cd.find_unqualified_citations(wrong_claim, registry)] == ["0.9988"]
    no_word = "0.9988 (run15.c3.unseen_gene_generalization)"
    assert cd.find_unqualified_citations(no_word, registry)
    ok = "first line 0.9988\nsecond line INVALIDATED_EVALUATION `run15.c3.unseen_gene_generalization`"
    assert cd.find_unqualified_citations(ok, registry) == []


def test_each_table_row_and_fenced_line_is_its_own_unit(registry):
    table = ("| run | auroc |\n| 15 | 0.9984 INVALIDATED_EVALUATION run15.test.ensemble_evaluation |\n"
             "| 15-UGH | 0.9988 |")
    found = cd.find_unqualified_citations(table, registry)
    assert [(c.line, c.figure) for c in found] == [(3, "0.9988")]
    fenced = "```\nAUROC 0.9984\n```\nINVALIDATED_EVALUATION run15.test.ensemble_evaluation"
    assert [c.figure for c in cd.find_unqualified_citations(fenced, registry)] == ["0.9984"]


@pytest.mark.parametrize("text,hits", [("0.99841", 0), ("10.9984", 0), ("0.9984.", 1), ("(0.9984)", 1), ("x0.9984", 1)])
def test_a_figure_matches_only_as_a_whole_number(registry, text, hits):
    assert len(cd.find_unqualified_citations(text, registry)) == hits


@pytest.mark.parametrize("doc,ids", sorted(cd.CORRECTED_HISTORICAL_DOCUMENTS.items()))
def test_a_corrected_historical_record_opens_with_its_correction(doc, ids):
    head = "\n".join((ROOT / doc).read_text(encoding="utf-8").splitlines()[:cd.CORRECTION_BANNER_WITHIN_LINES])
    assert cd.CORRECTION_BANNER in head
    for cid in ids:
        assert cid in head, cid


def test_the_probe_docstring_no_longer_states_that_run15_postdates_the_fix():
    doc = ast.get_docstring(ast.parse((ROOT / "scripts" / "probe_run14_univariate_leakage.py").read_text(encoding="utf-8")))
    assert "Run 15 POSTDATES the fix, so its" not in doc
    assert "CORRECTED 2026-11-08" in doc and "run15.test.ensemble_evaluation" in doc


# --------------------------------------------------------------------------------------------------------------------
# The live design and the falsifier verdict
# --------------------------------------------------------------------------------------------------------------------

def test_the_known_invalid_holdout_design_withholds_its_verdict():
    v = cd.design_verdict(cd.UGH_DESIGN_ID, 0.9988, 0.95)
    assert v["verdict"] == "WITHHELD" and v["design_status"] == "INVALID_DESIGN"
    assert v["claim_it_cannot_support"] == IDS[0]
    assert set(v["defects"]) == {"GNN_HOLDOUT_LABEL_LEAKAGE", "GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE",
                                 "GENE_IDENTITY_SPLIT_LEAKAGE"}
    assert cd.design_verdict(cd.UGH_DESIGN_ID, 0.5, 0.95)["verdict"] == "WITHHELD"


def test_the_holdout_design_defects_are_named_in_the_c3_record(registry):
    assert set(cd.KNOWN_INVALID_DESIGNS[cd.UGH_DESIGN_ID]["defects"]) <= set(registry.current(IDS[0]).defect_ids)


def test_a_design_not_known_invalid_reports_pass_or_fail_and_types_are_strict():
    assert cd.design_verdict("some.other.design", 0.96, 0.95)["verdict"] == "PASS"
    assert cd.design_verdict("some.other.design", 0.94, 0.95)["verdict"] == "FAIL"
    with pytest.raises(ValueError):
        cd.design_verdict(cd.UGH_DESIGN_ID, np.float64(0.9), 0.95)


def test_the_driver_withholds_the_c3_verdict_through_design_verdict():
    src = (ROOT / "scripts" / "run_phase2_eval.py").read_text(encoding="utf-8")
    assert 'm_ugh["c3_falsifier"] = design_verdict(UGH_DESIGN_ID, m_ugh["auroc"], 0.95)' in src
    assert src.index('m_ugh["c3_falsifier"] = design_verdict') < src.index(
        'writer.save_json("unseen_gene_holdout_metrics.json", m_ugh)')
    assert '"PASS" if m_ugh["auroc"] >= 0.95 else "FAIL"' not in src


# --------------------------------------------------------------------------------------------------------------------
# The evidence functions, on synthetic inputs with known answers
# --------------------------------------------------------------------------------------------------------------------

def test_the_own_label_identity_detects_an_inclusive_count_and_its_absence():
    genes = pd.Series(["A", "A", "A", "B", "B"])
    y = np.array([1, 1, 0, 0, 1])
    inclusive = np.array([2, 2, 2, 1, 1])
    out = ev.own_label_count_identity(genes, y, inclusive)
    assert out["mismatches"] == 0 and out["rows"] == 5 and out["positives"] == 3
    assert ev.own_label_count_identity(genes, y, np.zeros(5))["mismatches"] == 5


def test_gene_token_overlap_sees_a_gene_inside_a_multi_gene_string():
    out = ev.gene_token_overlap(pd.Series(["ABCA4;ARHGAP29", "TTN"]), pd.Series(["ABCA4", "BRCA1", "BRCA1"]),
                                np.array([1, 0, 1]))
    assert out["strings_identical_to_a_training_string"] == 0
    assert out["strings_sharing_a_gene_token"] == 1 and out["rows_sharing_a_gene_token"] == 1
    assert out["positives_among_them"] == 1 and out["row_fraction"] == round(1 / 3, 4)


def test_the_holdout_rule_is_run15s_md5_rule():
    genes = pd.Series([f"G{i}" for i in range(400)] + [None])
    mask = ev.run15_holdout_mask(genes)
    expected = [int(hashlib.md5(f"42:{'unknown' if g is None else g}".encode()).hexdigest(), 16) / 2 ** 128 < 0.2
                for g in genes]
    assert mask.tolist() == expected and 40 < mask.sum() < 120


def test_lovd_annotation_is_read_against_the_training_mode():
    train = np.array([-0.0171] * 10 + [15.9, 63.8])
    zero = ev.unannotated_lovd_value(train)
    assert zero == -0.0171
    assert ev.lovd_annotated_rows(np.array([-0.0171, 15.9, -0.0171]), zero) == 1


def test_a_log_line_must_be_unique_and_is_never_given_an_invented_timestamp():
    lines = ["2026-06-06 04:45:22  INFO  x  GNN scores injected into train split (mean=0.159).",
             "  AUROC           0.9984         0.9984  PASS  (target >= 0.9)"]
    stamped = ev._log_line(lines, "GNN scores injected")
    assert stamped["line"] == 1 and stamped["timestamp"] == "2026-06-06 04:45:22"
    printed = ev._log_line(lines, "PASS  (target")
    assert printed["line"] == 2 and printed["timestamp"] is None and printed["text"].startswith("AUROC")
    with pytest.raises(ev.Run15EvidenceError, match="0 line"):
        ev._log_line(lines, "absent")
    with pytest.raises(ev.Run15EvidenceError, match="2 line"):
        ev._log_line(lines + lines, "PASS  (target")
