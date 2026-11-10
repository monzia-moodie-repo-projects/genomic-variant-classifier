"""Stage 1D-1b.2 (set "aq"): the source-absence census, measured by execution (ABSENCE-AS-EVIDENCE-1).
Owner ruling 7f4b69d0 (2026-11-09), section 5, step 1. Author: Monzia Moodie."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import absence_census as AC
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derived():
    return AC.build(REPO)                     # runs the hermetic connector-layer child once


def row(report, name):
    return next(r for r in report["rows"] if r["feature"] == name)


def test_the_committed_census_is_exactly_the_fresh_derivation(derived):
    raw = (REPO / AC.REPORT_PATH).read_bytes()
    assert AC.report_bytes(derived) == raw
    assert b"\r" not in raw and raw.endswith(b"}\n")


def test_raw_inputs_are_measured_by_execution_and_equal_the_source_text():
    engineer, _contract = AC._builder(REPO)
    assert AC.executed_inputs(engineer) == AC.source_text_inputs(REPO)
    assert len(AC.executed_inputs(engineer)) == 67


def test_eighty_nine_features_fabricate_a_value_when_their_source_is_absent(derived):
    s = derived["summary"]
    assert s["features"] == len(TABULAR_FEATURES) == 91 and s["fabricated_under_absence"] == 89
    assert s["missingness_survives"] == ["gene_constraint_oe", "gene_is_constrained"]
    assert s["absent_differs_from_nan"] == []          # 'never joined' and 'joined, not in it' are indistinguishable
    assert s["fabrication_is_silent"] is True


def test_absence_alone_sets_four_indicators_to_one(derived):
    assert derived["summary"]["indicators_set_by_absence"] == ["af_is_absent", "af_is_ultra_rare", "is_snv",
                                                                "revel_pathogenic"]
    assert row(derived, "revel_pathogenic")["raw_inputs_read"] == ["revel_score"]
    assert row(derived, "n_tools_pathogenic")["absent_value"] == 1.0            # the absent REVEL counts as a tool
    assert row(derived, "sift_deleterious")["absent_value"] == 0.0              # the SIFT fill is NOT its threshold


def test_the_indicators_are_reproduced_by_direct_execution():
    out = engineer_features(pd.DataFrame(index=pd.RangeIndex(2)))
    for name in ("af_is_absent", "af_is_ultra_rare", "is_snv", "revel_pathogenic"):
        assert (out[name] == 1).all(), name
    observed = engineer_features(pd.DataFrame({"allele_freq": [0.05, 0.0], "revel_score": [0.1, 0.9],
                                               "ref": ["AT", "A"], "alt": ["A", "G"]}))
    assert list(observed["af_is_ultra_rare"]) == [0, 1] and list(observed["revel_pathogenic"]) == [0, 1]
    assert list(observed["is_snv"]) == [0, 1]


def test_a_fill_that_is_also_a_measurement_is_named(derived):
    named = set(derived["summary"]["fill_equals_an_observed_value"])
    assert {"eve_score", "revel_score", "sift_score", "alphamissense_score", "cadd_phred"} <= named
    assert row(derived, "eve_score")["absent_value"] == 0.5 and row(derived, "eve_score")["fill_equals_an_observed_value"]


def test_every_feature_reads_a_raw_input_and_derived_features_read_only_through_their_dependencies(derived):
    assert all(r["raw_inputs_read"] for r in derived["rows"])
    checked = {r["feature"] for r in derived["lineage_check"]}
    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    assert checked == {e["name"] for e in ledger["entries"] if e["depends_on"]}
    bad = copy.deepcopy(ledger)
    next(e for e in bad["entries"] if e["name"] == "revel_pathogenic")["depends_on"] = ["sift_score"]
    with pytest.raises(ValueError, match=r"revel_pathogenic: reads raw inputs its declared dependencies do not: "
                                         r"\['revel_score'\]"):
        AC.lineage_check(bad, derived["rows"])


def test_the_annotation_chain_runs_hermetically_and_its_values_are_what_a_model_sees(derived):
    layer = derived["connector_layer"]
    assert layer["network_attempts"] == 0 and layer["fixture_rows"] == 3
    assert "allele_freq" in layer["raw_inputs_never_written_by_the_chain"]      # the gnomAD join is outside the chain
    assert layer["raw_inputs_written_by_the_chain"]["eve_score"] == [0.5, 0.5, 0.5]
    assert layer["raw_inputs_written_by_the_chain"]["revel_score"] == [0.5, 0.5, 0.5]
    for r in derived["rows"]:
        assert len(r["all_sources_absent_through_the_annotation_chain"]) == 3
    assert row(derived, "revel_pathogenic")["all_sources_absent_through_the_annotation_chain"] == [1.0, 1.0, 1.0]


def test_the_command_line_checks_byte_for_byte_and_never_overwrites(tmp_path, derived, monkeypatch):
    monkeypatch.setattr(AC, "connector_layer", lambda repo: {
        "features": {r["feature"]: r["all_sources_absent_through_the_annotation_chain"] for r in derived["rows"]},
        "written": derived["connector_layer"]["raw_inputs_written_by_the_chain"],
        "network_attempts": 0, "created": derived["connector_layer"]["files_created_in_the_hermetic_folder"]})
    good = tmp_path / "census.json"
    assert AC.main(["--repo", str(REPO), "--out", str(good)]) == 0
    assert good.read_bytes() == (REPO / AC.REPORT_PATH).read_bytes()
    assert AC.main(["--repo", str(REPO), "--check", str(good)]) == 0
    tampered = json.loads(good.read_text())
    tampered["summary"]["fabricated_under_absence"] = 0
    good.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    assert AC.main(["--repo", str(REPO), "--check", str(good)]) == 1
    with pytest.raises(SystemExit, match="overwrite"):
        AC.main(["--repo", str(REPO), "--out", str(good)])


def test_the_builder_signals_are_measured_in_a_child_and_name_only_the_preserved_features():
    records, warned = AC.builder_signals(REPO)
    assert warned == []
    assert [(name, level) for name, level, _m in records] == [("genomic_variant_classifier.models.variant_ensemble", 20)]
    assert "gene_constraint_oe" in records[0][2] and "gene_is_constrained" in records[0][2]


def test_a_builder_that_warns_about_a_fabricated_value_refuses_the_silent_claim():
    import logging

    engineer, contract = AC._builder(REPO)
    inputs = AC.executed_inputs(engineer)
    with pytest.raises(ValueError, match="signalled about a fabricated value"):
        AC.feature_layer(engineer, contract, inputs, ([("x", logging.WARNING, "filled eve_score with 0.5")], []))
    with pytest.raises(ValueError, match="signalled about a fabricated value"):      # an INFO naming a fabricated feature
        AC.feature_layer(engineer, contract, inputs, ([("x", logging.INFO, "eve_score set to 0.5")], []))
    with pytest.raises(ValueError, match="raised 1 warnings"):
        AC.feature_layer(engineer, contract, inputs, ([], ["UserWarning: x"]))


def test_the_value_census_is_deterministic():
    engineer, contract = AC._builder(REPO)
    inputs = AC.executed_inputs(engineer)
    a, _ = AC.feature_layer(engineer, contract, inputs, ([], []))
    b, _ = AC.feature_layer(engineer, contract, inputs, ([], []))
    assert a == b
    assert np.isfinite([r["absent_value"] for r in a if isinstance(r["absent_value"], float)]).all()
