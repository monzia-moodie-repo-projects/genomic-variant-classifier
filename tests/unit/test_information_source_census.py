"""Stage 1D-1a information-source census (set "ao"): the committed ledger verifies against this repository, and the
gate refuses each defect class it exists to catch -- each refusal shown on the REAL ledger with one defect introduced.
Synthetic modules test the measurement primitives (contract, semantic-closure digests, discovery, surface rules).
Author: Monzia Moodie."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import consumers as K

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 64


@pytest.fixture(scope="module")
def measured():
    return {"engineered": K.engineered_columns(REPO), "discovered": K.discover_sites(REPO)}


@pytest.fixture(scope="module")
def committed():
    return C.strict_load(REPO / C.LEDGER_PATH)


@pytest.fixture()
def ledger(committed):
    return copy.deepcopy(committed)


def verify(ledger, measured):
    return C.verify_census(REPO, ledger, engineered=measured["engineered"], discovered=measured["discovered"])


def entry(ledger, name):
    return next(e for e in [*ledger["entries"], *ledger["information_sources"]] if e["name"] == name)


# ---------------------------------------------------------------------------------------------------------------- the ledger
def test_the_committed_ledger_verifies_with_fresh_measurement(committed):
    report = C.verify_census(REPO, committed)              # builder run and consumer discovery measured here
    lines = report.lines()
    assert C.SCIENTIFIC_TRUTH_LINE in lines
    assert "Scientific truth of all source claims: NOT ESTABLISHED BY STRUCTURAL VALIDATION ALONE" in lines
    assert sum(report.disposition_counts.values()) == len(C.source_contract(REPO).names)
    assert "pending_review" not in report.disposition_counts


def test_the_committed_ledger_is_exactly_what_the_reviewed_builder_writes(committed):
    spec = importlib.util.spec_from_file_location("build_feature_census_1d1a",
                                                  REPO / "scripts" / "build_feature_census_1d1a.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    raw = (REPO / C.LEDGER_PATH).read_bytes()
    assert C.ledger_bytes(builder.build(REPO)) == raw
    assert C.ledger_bytes(committed) == raw                 # canonical: sorted keys, ASCII, LF, one newline
    assert b"\r" not in raw and raw.endswith(b"}\n")


def test_every_pinned_ruling_holds_and_no_other_feature_is_pinned(committed):
    assert len(C.PINNED) == 19
    for name, disposition in C.PINNED.items():
        assert entry(committed, name)["disposition"] == disposition.value


def test_measured_surfaces_match_the_code(committed, measured):
    names = list(C.source_contract(REPO).names)
    surfaces = committed["surfaces"]
    assert sorted(surfaces) == sorted(K.SURFACES)          # the committed form sorts keys
    assert measured["engineered"] == names
    assert surfaces["gnn_node_attributes"]["observed"] == sorted(set(names) - {"gnn_score"})
    assert surfaces["hetero_gnn_node_attributes"]["observed"] == sorted(set(names) - {"gnn_score", "hetero_gnn_score"})
    assert surfaces["calibration"]["observed"] == [] and surfaces["export"]["observed"] == []
    # the retired label count is a GNN node attribute: the measured transitive dependency
    assert "n_pathogenic_in_gene" in entry(committed, "gnn_score")["depends_on"]


def test_the_graph_inventory_includes_the_disease_relations(committed):
    graph = committed["graph_inventory"]
    assert graph["kg_relation_sources"]["clingen"] == "shares_disease"
    assert graph["kg_relation_sources"]["omim"] == "shares_disease"
    assert "database" in graph["string_edge_attribute_channels"]
    for name in ("kg_relation:clingen", "kg_relation:omim", "string_edge_attribute:database",
                 "string_relation:combined_score_threshold"):
        assert entry(committed, name)["disposition"] == "exclude_strict"


def test_the_eve_correction_record_withdraws_the_invalid_sections():
    record = C.strict_load(REPO / "docs/census/measure_1d1_correction.json")   # strict JSON: no NaN
    assert set(record["withdrawn_sections"]) == {"default_indicator_train", "eve_gene_coverage_train"}
    assert not set(record["withdrawn_sections"]) & set(record["retained_sections"])
    assert all(record["eve_source_join_run15"]["eve_score_constant_in"].values())
    assert "eve_path not set" in record["eve_source_join_run15"]["log"]["text"]


# ------------------------------------------------------------------------------------------------ one defect at a time
def test_a_pinned_ruling_cannot_be_changed(ledger, measured):
    entry(ledger, "eve_score")["disposition"] = "admit_strict"
    with pytest.raises(ValueError, match="pinned ruling"):
        verify(ledger, measured)


def test_an_unreviewed_feature_refuses(ledger, measured):
    entry(ledger, "af_raw")["disposition"] = "pending_review"
    with pytest.raises(ValueError, match="unreviewed disposition"):
        verify(ledger, measured)


def test_order_and_completeness_are_authoritative(ledger, measured):
    reordered = copy.deepcopy(ledger)
    reordered["entries"][0], reordered["entries"][1] = reordered["entries"][1], reordered["entries"][0]
    with pytest.raises(ValueError, match="names or order"):
        verify(reordered, measured)
    ledger["entries"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        verify(ledger, measured)


def test_recorded_consumers_must_equal_the_measured_ones(ledger, measured):
    entry(ledger, "af_raw")["consumers"].remove("inference")
    with pytest.raises(ValueError, match="differ from measured"):
        verify(ledger, measured)


def test_a_tampered_surface_inventory_refuses(ledger, measured):
    ledger["surfaces"]["gnn_node_attributes"]["observed"].remove("n_pathogenic_in_gene")
    with pytest.raises(ValueError, match="surface inventory differs"):
        verify(ledger, measured)


def test_every_consumer_site_must_be_classified(ledger, measured):
    del ledger["consumer_sites"]["scripts/run_benchmark.py"]
    with pytest.raises(ValueError, match="unaccounted"):
        verify(ledger, measured)


def test_a_new_unclassified_consumer_refuses(ledger, measured):
    discovered = dict(measured["discovered"]) | {"scripts/new_consumer.py": ["TABULAR_FEATURES"]}
    with pytest.raises(ValueError, match=r"unclassified \['scripts/new_consumer.py'\]"):
        C.verify_census(REPO, ledger, engineered=measured["engineered"], discovered=discovered)


def test_site_classes_come_from_the_vocabulary(ledger, measured):
    ledger["consumer_sites"]["scripts/run_benchmark.py"]["classes"] = ["surface:somewhere_else"]
    with pytest.raises(ValueError, match="unknown site classes"):
        verify(ledger, measured)


def test_a_surface_rule_file_must_be_classified_to_its_surface(ledger, measured):
    ledger["consumer_sites"]["src/genomic_variant_classifier/api/pipeline.py"]["classes"] = ["diagnostic"]
    with pytest.raises(ValueError, match="surface inference: its rule's file"):
        verify(ledger, measured)


def test_a_changed_producer_names_the_entries_to_re_review(ledger, measured):
    symbol = "src/genomic_variant_classifier/data/eve.py::EVEConnector"
    ledger["producer_digests"][symbol] = SHA
    with pytest.raises(ValueError, match=r"producer changed since census capture: .*EVEConnector.*eve_score"):
        verify(ledger, measured)


def test_producer_digests_cover_exactly_the_cited_symbols(ledger, measured):
    ledger["producer_digests"]["src/genomic_variant_classifier/data/eve.py::load_eve_entry_map"] = SHA
    with pytest.raises(ValueError, match="exactly the cited code symbols"):
        verify(ledger, measured)


def test_an_evidence_anchor_must_exist_exactly_once(ledger, measured):
    entry(ledger, "af_raw")["evidence_refs"][0] += " -- not in the file"
    with pytest.raises(ValueError, match="occurs 0 times"):
        verify(ledger, measured)


def test_an_ambiguous_evidence_anchor_refuses(ledger, measured):
    entry(ledger, "gtex_is_eqtl")["evidence_refs"].append(
        'code:src/genomic_variant_classifier/data/real_data_prep.py#("gtex_is_eqtl", 0),')   # two places in the file
    with pytest.raises(ValueError, match="occurs 2 times"):
        verify(ledger, measured)


def test_literature_alone_is_not_repository_checkable_evidence(ledger, measured):
    entry(ledger, "revel_score")["evidence_refs"] = ["doi:10.1002/humu.22768"]
    with pytest.raises(ValueError, match="repository-checkable"):
        verify(ledger, measured)


def test_admission_is_monotone_over_dependencies(ledger, measured):
    entry(ledger, "cadd_high")["depends_on"].append("polyphen2_score")
    with pytest.raises(ValueError, match="admitted while depending on non-admitted"):
        verify(ledger, measured)


def test_a_measured_dependency_cannot_be_dropped(ledger, measured):
    entry(ledger, "gnn_score")["depends_on"].remove("n_pathogenic_in_gene")
    with pytest.raises(ValueError, match="omits measured dependencies"):
        verify(ledger, measured)


def test_a_mixed_origin_relation_cannot_be_admitted(ledger, measured):
    entry(ledger, "string_relation:combined_score_threshold")["disposition"] = "admit_strict"
    with pytest.raises(ValueError, match="not of independent biological origin"):
        verify(ledger, measured)


def test_information_sources_match_the_measured_graph(ledger, measured):
    ledger["information_sources"] = [s for s in ledger["information_sources"] if s["name"] != "kg_relation:omim"]
    with pytest.raises(ValueError, match=r"missing \['kg_relation:omim'\]"):
        verify(ledger, measured)


def test_unreviewed_and_invalid_assessments_refuse(ledger, measured):
    bad = copy.deepcopy(ledger)
    entry(bad, "af_raw")["upstream_supervision"] = "unreviewed"
    with pytest.raises(ValueError, match="upstream supervision not assessed"):
        verify(bad, measured)
    bad = copy.deepcopy(ledger)
    entry(bad, "af_raw")["upstream_overlap"] = "unreviewed"
    with pytest.raises(ValueError, match="upstream overlap not assessed"):
        verify(bad, measured)
    bad = copy.deepcopy(ledger)
    entry(bad, "af_raw")["label_criterion_use"] = "probably_fine"
    with pytest.raises(ValueError, match="invalid label-criterion use"):
        verify(bad, measured)


def test_admission_requires_a_verified_source(ledger, measured):
    entry(ledger, "af_raw")["source_state"] = "unknown"
    with pytest.raises(ValueError, match="admitted without a verified source"):
        verify(ledger, measured)


def test_sources_and_defects_must_resolve(ledger, measured):
    bad = copy.deepcopy(ledger)
    entry(bad, "af_raw")["source_ids"].append("dataset:nowhere")
    with pytest.raises(ValueError, match="unknown dataset"):
        verify(bad, measured)
    ledger["registered_defects"][0]["affects"].append("not_a_feature")
    with pytest.raises(ValueError, match="affects unknown names"):
        verify(ledger, measured)


def test_authority_and_schema_are_exact(ledger, measured):
    bad = copy.deepcopy(ledger)
    bad["authority"]["contract_digest"] = SHA
    with pytest.raises(ValueError, match="authority changed"):
        verify(bad, measured)
    ledger["extra"] = 1
    with pytest.raises(ValueError, match="unexpected ledger schema"):
        verify(ledger, measured)


def test_the_command_line_refuses_a_non_canonical_ledger(tmp_path, committed):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(committed, indent=4, sort_keys=True) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="canonical"):
        C.main(["--repo", str(REPO), "--ledger", str(path)])


def test_strict_json_refuses_duplicates_and_nonfinite():
    with pytest.raises(ValueError, match="duplicate"):
        C.strict_loads('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="nonfinite"):
        C.strict_loads('{"a":NaN}')


# ---------------------------------------------------------------------------------------------- measurement primitives
def _module(tmp_path: Path, text: str, rel: str = "pkg/mod.py") -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


BASE = '''import math
LIMIT = 3


def helper(x):
    return x * LIMIT


def target(x):
    # a comment
    return helper(x) + math.pi


def unrelated():
    return 1
'''


def test_the_closure_digest_follows_meaning_not_bytes(tmp_path):
    path = _module(tmp_path, BASE)
    digest = C.symbol_closure_digest(tmp_path, "pkg/mod.py::target")
    path.write_text(BASE.replace("# a comment", "# reworded entirely").replace("    return 1", "    return 2"))
    assert C.symbol_closure_digest(tmp_path, "pkg/mod.py::target") == digest     # comment and unrelated code
    path.write_text(BASE.replace("LIMIT = 3", "LIMIT = 4"))
    assert C.symbol_closure_digest(tmp_path, "pkg/mod.py::target") != digest     # a referenced constant
    path.write_text(BASE.replace("return x * LIMIT", "return x * LIMIT + 1"))
    assert C.symbol_closure_digest(tmp_path, "pkg/mod.py::target") != digest     # a transitively called helper
    with pytest.raises(ValueError, match="not defined"):
        C.symbol_closure_digest(tmp_path, "pkg/mod.py::absent")


def test_the_whole_file_digest_is_recorded_but_does_not_gate(tmp_path):
    src = 'EXPECTED_TABULAR_FEATURE_COUNT = 2\nTABULAR_FEATURES = ["a", "b"]\n'
    path = _module(tmp_path, src, C.CONTRACT_FILE)
    first = C.source_contract(tmp_path)
    path.write_text(src + "\n# unrelated model code edited\n")
    second = C.source_contract(tmp_path)
    assert first.contract_digest == second.contract_digest and first.file_sha256 != second.file_sha256
    path.write_text(src.replace('["a", "b"]', '["b", "a"]'))
    assert C.source_contract(tmp_path).contract_digest != first.contract_digest
    path.write_text(src + 'TABULAR_FEATURES = ["c"]\n')
    with pytest.raises(ValueError, match="duplicate authoritative"):
        C.source_contract(tmp_path)
    path.write_text(src.replace("= 2", "= 3"))
    with pytest.raises(ValueError, match="does not reconcile"):
        C.source_contract(tmp_path)


def test_discovery_sees_aliases_attributes_and_definitions(tmp_path):
    _module(tmp_path, "from m import TABULAR_FEATURES as _C\n", "src/a.py")
    _module(tmp_path, "import m\nx = m.engineer_features\n", "src/b.py")
    _module(tmp_path, "def build_pyg_dataset():\n    pass\n", "scripts/c.py")
    _module(tmp_path, "y = 1\n", "scripts/d.py")
    assert K.discover_sites(tmp_path) == {"scripts/c.py": ["build_pyg_dataset"], "src/a.py": ["TABULAR_FEATURES"],
                                          "src/b.py": ["engineer_features"]}


def test_the_comprehension_rule_evaluates_only_a_pure_selection():
    cols = ["a", "gnn_score", "hetero_gnn_score"]
    assert K.evaluate_comprehension('c for c in X_train.columns if c != "gnn_score"', cols) == ["a", "hetero_gnn_score"]
    with pytest.raises(ValueError, match="disallowed construct Call"):
        K.evaluate_comprehension('c for c in X_train.columns if __import__("os")', cols)
    with pytest.raises(ValueError, match="iterate X_train.columns"):
        K.evaluate_comprehension("c for c in other.columns", cols)


def test_an_anchor_must_be_unique(tmp_path):
    _module(tmp_path, "x = 1\nx = 1\n", "f.py")
    with pytest.raises(ValueError, match="occurs 2 times"):
        K.require_unique_anchor(tmp_path, "f.py", "x = 1")
    assert K.require_unique_anchor(tmp_path, "f.py", "x = 1\nx") == 1


# ----------------------------------------------------------------------- the reference's EVE and relation semantics
def test_eve_coverage_is_join_membership_and_a_matched_half_is_observed():
    k1 = C.ProteinVariantKey("P12345-1", "R", 123, "H")
    k2 = C.ProteinVariantKey("P12345-1", "R", 124, "H")
    k3 = C.ProteinVariantKey("Q99999-2", "A", 40, "G")
    out = C.annotate_eve([k1, k2, k3], C.EveLookup({k1: 0.5}, frozenset({"P12345-1"}), SHA))
    assert [o.status for o in out] == [C.Coverage.MATCHED_SCORE, C.Coverage.GENE_MODELED_VARIANT_UNSCORED,
                                       C.Coverage.GENE_NOT_MODELED]
    assert out[0].has_observed_score and out[0].score == 0.5 and out[1].score is None
    with pytest.raises(ValueError, match="undeclared protein"):
        C.annotate_eve([k1], C.EveLookup({k1: 0.5}, frozenset(), SHA))
    with pytest.raises(ValueError, match="invalid numeric"):
        C.annotate_eve([k1], C.EveLookup({k1: float("nan")}, frozenset({"P12345-1"}), SHA))


def test_graph_relation_admission_default_denies():
    C.admit_strict_graph_relation(C.GraphRelationEvidence("measured", C.RelationOrigin.INDEPENDENT_BIOLOGY, SHA))
    for origin in (C.RelationOrigin.MIXED, C.RelationOrigin.UNKNOWN, C.RelationOrigin.CLINICAL_DISEASE):
        with pytest.raises(ValueError, match="not independently"):
            C.admit_strict_graph_relation(C.GraphRelationEvidence("shares_pathway", origin, SHA))
    with pytest.raises(ValueError, match="digest"):
        C.admit_strict_graph_relation(C.GraphRelationEvidence("measured", C.RelationOrigin.INDEPENDENT_BIOLOGY, None))


def test_semantic_and_materialization_identities_differ(committed):
    d1 = C.semantic_contract_digest(committed)
    d2 = C.materialization_digest(d1, source_digests={"s": SHA}, parser_digests={"p": SHA}, row_population_digest=SHA)
    d3 = C.materialization_digest(d1, source_digests={"s": "b" * 64}, parser_digests={"p": SHA},
                                  row_population_digest=SHA)
    assert len(d1) == 64 and d1 != d2 != d3
    changed = copy.deepcopy(committed)
    changed["surfaces"]["export"]["note"] = "a measured field"
    assert C.semantic_contract_digest(changed) == d1        # measured fields are not semantics
    entry(changed, "af_raw")["meaning"] += " (reworded)"
    assert C.semantic_contract_digest(changed) != d1


# ------------------------------------------------------------------------------- the EVE correction record's derivation
def _correction_inputs(tmp_path: Path, *, eve_varies: bool = False, log_line: str | None = None):
    import pandas as pd

    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    for part, n in (("train", 6), ("val", 3), ("test", 4)):
        pd.DataFrame({"cadd_phred": [float(i) for i in range(n)],
                      "eve_score": [float(i) if eve_varies else 0.0 for i in range(n)],
                      "dbsnp_af": [0.0] * n}).to_parquet(splits / f"X_{part}.parquet", index=False)
    log = tmp_path / "run.log"
    log.write_text("start\n" + (log_line or "x  WARNING  eve  EVEConnector: eve_path not set -- returning 0.5") + "\nend\n",
                   encoding="utf-8")
    earlier = tmp_path / "measure_1d1_audit.json"
    earlier.write_text('{"rows": {"train": 6}, "default_indicator_train": {}, '
                       '"eve_gene_coverage_train": {"positive_rate_elsewhere": NaN}}', encoding="utf-8")
    return ["--earlier", str(earlier), "--splits", str(splits), "--log", str(log)]


def _correction_script():
    spec = importlib.util.spec_from_file_location("measure_1d1_correction", REPO / "scripts" / "measure_1d1_correction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_correction_withdraws_measures_and_rechecks(tmp_path):
    script = _correction_script()
    args = _correction_inputs(tmp_path)
    out = tmp_path / "out.json"
    assert script.main(args + ["--out", str(out)]) == 0
    record = C.strict_load(out)                                   # strict JSON although the corrected file held NaN
    assert set(record["withdrawn_sections"]) == {"default_indicator_train", "eve_gene_coverage_train"}
    assert record["retained_sections"] == {"rows": {"train": 6}}
    assert record["constant_columns"]["constant_in_every_split"] == ["dbsnp_af", "eve_score"]
    assert script.main(args + ["--check", str(out)]) == 0
    out.write_bytes(out.read_bytes().replace(b'"train": 6', b'"train": 7'))
    assert script.main(args + ["--check", str(out)]) == 1
    with pytest.raises(SystemExit, match="overwrite"):
        script.main(args + ["--out", str(out)])


def test_the_correction_refuses_what_it_cannot_establish(tmp_path):
    script = _correction_script()
    with pytest.raises(SystemExit, match="not constant in every split"):
        script.main(_correction_inputs(tmp_path / "a", eve_varies=True) + ["--out", str(tmp_path / "a.json")])
    with pytest.raises(SystemExit, match="eve_path not set"):
        script.main(_correction_inputs(tmp_path / "b", log_line="EVEConnector: 12 / 40 variants covered")
                    + ["--out", str(tmp_path / "b.json")])
