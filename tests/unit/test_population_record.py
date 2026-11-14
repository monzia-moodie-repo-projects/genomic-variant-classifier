"""Stage 1D-1b.4 (set "as"): the real-data population record (docs/census/POPULATION_ISOLATION_1D1b4.json) is internally
consistent, bound to the code that produced it, measured on the same files as the 1D-1b.3 graph record, and the numbers the
census's registered defects state are the measured ones. The record was measured on the owner's Run 15 split manifests and
HGNC file (scripts/measure_1d1b4_population.py; --check re-measures it); a clone does not hold those files.
Author: Monzia Moodie."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import census as C

REPO = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    raw = (REPO / "docs/census" / name).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    return json.loads(raw)


@pytest.fixture(scope="module")
def record():
    return _load("POPULATION_ISOLATION_1D1b4.json")


@pytest.fixture(scope="module")
def defects():
    return {d["id"]: d["statement"] for d in C.strict_load(REPO / C.LEDGER_PATH)["registered_defects"]}


def test_the_record_is_bound_to_the_code_as_it_is_now(record):
    assert record["schema"] == "gvc.population-isolation-record.1d1b4/1"
    for symbol, digest in record["bound_code"].items():
        assert C.symbol_closure_digest(REPO, symbol) == digest, (
            f"{symbol} changed since the record was measured: re-run scripts/measure_1d1b4_population.py")
    assert "scripts/measure_1d1b4_population.py::measure" in record["bound_code"]


def test_the_inputs_are_the_files_of_the_graph_record(record):
    graph = _load("GRAPH_IDENTITY_1D1b3.json")
    assert record["inputs"]["hgnc"]["sha256"] == graph["inputs"]["hgnc"]["sha256"]
    for key in ("meta_val", "meta_test"):
        assert record["inputs"][key]["sha256"] == graph["inputs"][key]["sha256"]
    assert set(record["inputs"]) == {"hgnc", "meta_train", "meta_val", "meta_test", "y_train", "y_val", "y_test"}
    assert all(C.HEX64.match(v["sha256"]) for v in record["inputs"].values())


def test_the_population_and_the_audit_close(record):
    a = record["audit"]["string_protocol"]
    sizes = a["sizes"]
    by = record["gene_fields"]["by_partition"]
    assert (sizes["train"], sizes["validation"], sizes["test"]) == (by["train"]["rows"], by["val"]["rows"], by["test"]["rows"])
    assert sum(sizes.values()) == record["variant_identity"]["rows"] == record["variant_identity"]["distinct_variant_ids"]
    assert a["verdict"] == "violated" and record["audit"]["vrs_protocol_verdict"] == "violated"
    assert {p: v["shared_variants"] for p, v in a["pairs"].items()} == dict.fromkeys(a["pairs"], 0)
    assert a["gene_incomplete_rows"] == {"train": by["train"]["no_resolved_gene"] + by["train"]["partially_resolved"],
                                         "validation": by["val"]["no_resolved_gene"] + by["val"]["partially_resolved"],
                                         "test": by["test"]["no_resolved_gene"] + by["test"]["partially_resolved"]}
    assert record["strict"] == {"outcome": "refused", "reason": "strict fit refused: split verdict is violated"}


def test_the_conditional_authority_and_its_comparisons_close(record):
    rep = record["conditional"]["report"]
    assert rep["training_rows"] - rep["excluded_gene_incomplete"] - rep["excluded_component_shares_heldout"] \
        == rep["authorized_rows"]
    gd = record["comparison"]["gene_direct_rule"]
    assert gd["superset_of_component_authorization"] and gd["authorized"] - rep["authorized_rows"] \
        == gd["admitted_only_by_gene_direct"] > 0
    old = record["comparison"]["one_gene_rule_1d1b3"]
    assert old["refused_rows_also_excluded_by_components"] == old["refused_training_rows"]
    assert old["authorized"] + old["refused_training_rows"] == rep["training_rows"]
    mix = record["label_mix_pathogenic_percent"]
    assert mix["excluded"] > mix["training"] > mix["authorized"]


def test_weights_checkpoint_and_the_legacy_early_stop(record):
    w = record["weights"]
    assert w["0"]["ess_percent"] == 100.0 and w["0"]["max_to_min_weight"] == 1.0
    assert w["0"]["ess_percent"] > w["0.25"]["ess_percent"] > w["0.5"]["ess_percent"] > w["1.0"]["ess_percent"]
    ck = record["checkpoint"]
    assert ck["component_overlap"] == 0 and abs(ck["positive_rate_inner_validation"] - ck["positive_rate_all"]) < 0.001
    es = record["legacy_early_stop"]
    assert es["early_stopping_rows_sharing_a_symbol_with_inner_training"] / es["early_stopping_rows"] > 0.99
    assert record["training_labels_receipt"] == {"authorized": 1, "denied": 1, "unknown": 1, "masked_calls": 1,
                                                 "masked_authorized": 1, "masked_denied": 1, "masked_unknown": 1}
    assert [s["step"] for s in record["run"]["steps"]][-1] == "labels"


def _n(x: int) -> str:
    return f"{x:,}"


def test_the_census_states_the_measured_numbers(record, defects):
    es = record["legacy_early_stop"]
    s = defects["GNN-EARLY-STOP-ROW-SPLIT-1"]
    share = round(100 * es["early_stopping_rows_sharing_a_symbol_with_inner_training"] / es["early_stopping_rows"], 2)
    for part in (_n(es["training_rows"]), _n(es["early_stopping_rows_sharing_a_symbol_with_inner_training"]),
                 _n(es["early_stopping_rows"]), f"({share}%)", _n(es["early_stopping_rows_in_an_inner_training_component"]),
                 _n(es["early_stopping_symbols_also_inner_training_symbols"]), _n(es["early_stopping_symbols"]),
                 _n(record["conditional"]["report"]["authorized_rows"]),
                 f"{100 * record['checkpoint']['positive_rate_inner_validation']:.3f}%",
                 f"{100 * record['checkpoint']['positive_rate_all']:.3f}%"):
        assert part in s, part
    g = record["gene_fields"]
    s = defects["GENE-FIELD-PROSE-1"]
    by = g["by_partition"]
    assert f"{g['prose_rows']} Run 15 rows ({by['train']['prose']} training, {by['val']['prose']} validation, " \
           f"{by['test']['prose']} test; {g['prose_distinct_fields']} distinct fields)" in s
    c = g["prose_rows_by_consequence"]
    assert (f"({c['intron_variant']} intronic, {c['missense_variant']} missense, {c['3_prime_UTR_variant']} 3' UTR, "
            f"{c['synonymous_variant']} synonymous, {c['non-coding_transcript_variant']} non-coding transcript, "
            f"{c['splice_donor_variant']} splice donor, {c['(none)']} none)") in s
    a = record["audit"]["string_protocol"]
    s = defects["VARIANT-MULTIGENE-ISOLATION-1"]
    no_gene = sum(v["no_resolved_gene"] for v in by.values())
    assert f"{_n(a['components'])} components ({_n(a['components'] - no_gene)} gene groups and {no_gene} rows" in s
    p = a["pairs"]
    assert (f"training-validation {p['train|validation']['shared_genes']} and {p['train|validation']['shared_components']}, "
            f"training-test {p['train|test']['shared_genes']} and {p['train|test']['shared_components']}, "
            f"validation-test {p['validation|test']['shared_genes']} and {p['validation|test']['shared_components']}") in s
    assert f"{a['components_spanning_partitions']} spanning partitions" in s
    s = defects["GENE-SYMBOL-SPLIT-IDENTITY-1"]
    old = record["comparison"]["one_gene_rule_1d1b3"]
    assert f"refuses the {old['refused_training_rows']} training rows of those {len(old['refused_hgnc_genes'])} genes" in s
    assert f"excludes all {old['refused_training_rows']} among the " \
           f"{_n(record['conditional']['report']['excluded_component_shares_heldout'])} training rows" in s
