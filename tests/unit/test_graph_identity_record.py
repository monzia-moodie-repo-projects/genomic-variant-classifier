"""Stage 1D-1b.3 (set "ar"): the real-data records (docs/census/GRAPH_IDENTITY_1D1b3.json and
docs/census/STRING_READER_BENCHMARK_1D1b3.json) are internally consistent, bound to the code that produced them, and the
numbers the census's registered defects state are the measured ones. The records were measured on the owner's source files
(scripts/measure_1d1b3_graph_identity.py, scripts/benchmark_1d1b3_string_reader.py); a clone does not hold those files.
Author: Monzia Moodie."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import census as C

REPO = Path(__file__).resolve().parents[2]
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _load(name: str) -> dict:
    raw = (REPO / "docs/census" / name).read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    return json.loads(raw)


@pytest.fixture(scope="module")
def record():
    return _load("GRAPH_IDENTITY_1D1b3.json")


@pytest.fixture(scope="module")
def bench():
    return _load("STRING_READER_BENCHMARK_1D1b3.json")


def test_the_record_is_bound_to_the_code_as_it_is_now(record):
    for symbol, digest in record["bound_code"].items():
        assert C.symbol_closure_digest(REPO, symbol) == digest, (
            f"{symbol} changed since the record was measured: re-run scripts/measure_1d1b3_graph_identity.py")
    for label, entry in record["inputs"].items():
        assert HEX64.match(entry["sha256"]) and entry["bytes"] > 0 and entry["logical_source"].startswith("dataset:"), label
    registry = C.strict_load(REPO / C.LEDGER_PATH)["datasets"]
    assert {e["logical_source"][8:] for e in record["inputs"].values()} <= set(registry)


def test_exact_pair_accounting(record):
    p = record["string_pairs"]
    assert p["records"] == 2 * p["pairs"] and p["scratch_bytes"] == p["records"] * p["record_bytes"] == p["records"] * 25
    assert HEX64.match(p["content_sha256"]) and p["max_partition_records"] < p["records"]


def test_the_threshold_crosstab_is_complete_and_nested(record):
    t = record["threshold"]
    x = t["crosstab"]
    assert sum(x.values()) == record["string_pairs"]["pairs"]
    assert x["selected_robust|published_fails"] == x["selected_sensitive|published_fails"] == 0    # G1 is inside G2
    assert t["g1_selected"] == x["selected_robust|published_passes"] + x["selected_sensitive|published_passes"]
    assert t["g2_published"] == sum(v for k, v in x.items() if k.endswith("published_passes"))
    assert t["threshold_milli"] == 700 and "database" not in t["channels"] and "textmining" not in t["channels"]
    q = record["quantization"]
    assert q["pairs"] - q["inconsistent_pairs"] == round(q["consistent_fraction"] * q["pairs"])


def test_graph_identities_are_computed_and_reconcile(record):
    mapping = record["mapping"]
    assert sum(mapping["status_counts"].values()) == mapping["proteins"]
    assert len(mapping["collisions"]) == mapping["status_counts"].get("collision_superseded", 0) + \
        mapping["status_counts"].get("collision_ambiguous", 0)
    sel = {"G1": record["threshold"]["g1_selected"], "G2": record["threshold"]["g2_published"],
           "G1_threshold_robust": record["threshold"]["crosstab"]["selected_robust|published_passes"]}
    digests = set()
    for name, g in record["graphs"].items():
        m, r = g["materialization"], g["report"]
        assert C.sha256_bytes(C.canonical_bytes(g["policy"])) == g["policy_sha256"] == m["relation_policy_sha256"], name
        assert m["source_sha256"] == record["inputs"]["string_links"]["sha256"]
        assert m["mapping_sha256"] == mapping["mapping_sha256"]
        assert r["selected_pairs"] == sel[name]
        assert r["edges"] == r["selected_pairs"] - r["selected_pairs_with_unmapped_endpoint"] - \
            r["selected_pairs_becoming_self_loops"], name
        assert g["census"]["strict_admissible"] is False
        digests.add(m["edge_content_sha256"])
    assert len(digests) == 3
    assert record["graphs"]["G1_threshold_robust"]["report"]["edges"] <= record["graphs"]["G1"]["report"]["edges"]


def test_the_legacy_graph_and_its_cache(record):
    legacy = record["legacy_string_graph"]
    assert legacy["edges"] == record["threshold"]["g2_published"]          # one edge per published pair at 700
    assert legacy["cache_returned_for_a_different_links_file"]["same_graph"] is True
    assert record["legacy_cache"]["filenames"] == 1 and record["legacy_cache"]["policies"] == 3


def test_cohort_identity_and_the_label_authority(record):
    c, a = record["cohort"], record["label_authority"]
    assert a["authorized_rows"] + a["refused_training_rows"] + a["heldout_rows"] == c["rows"]
    assert len(a["refused_genes"]) == c["hgnc_ids_whose_symbols_straddle_train_and_heldout"]
    assert sum(c["symbol_resolution"].values()) == c["symbols"] and c["val_test_symbol_overlap"] == 0


def test_pathway_projections(record):
    r = record["reactome"]
    assert r["legacy"]["train_edges_removed_by_adding_heldout_genes"] > 0
    cons = r["global_size_projection"]["induced_subgraph_consistency"]
    assert cons["consistent"] == 1 and cons["removed_by_growth"] == cons["added_by_growth"] == 0
    assert cons["reweighted_by_growth"] == 0
    assert r["hypergraph_equivalence_max_abs_difference"] < 1e-9


def test_the_benchmark_is_layout_independent_and_bound(bench, record):
    assert C.symbol_closure_digest(REPO, bench["reader"]) == bench["reader_sha256"], (
        "the reader changed since the benchmark: re-run scripts/benchmark_1d1b3_string_reader.py")
    runs = bench["runs"]
    assert len(runs) == 5 and {r["content_sha256"] for r in runs} == {record["string_pairs"]["content_sha256"]}
    assert {r["scratch_bytes"] for r in runs} == {record["string_pairs"]["scratch_bytes"]}
    assert bench["links"]["sha256"] == record["inputs"]["string_links"]["sha256"]
    assert len({(r["partitions"], r["chunk_records"]) for r in runs}) == 5


def _fmt(n: int) -> str:
    return f"{n:,}"


def test_the_census_states_the_measured_numbers(record):
    defects = {d["id"]: d["statement"] for d in C.strict_load(REPO / C.LEDGER_PATH)["registered_defects"]}
    x = record["threshold"]["crosstab"]
    for n in (x["selected_robust|published_passes"], x["selected_sensitive|published_passes"],
              x["excluded_sensitive|published_passes"], record["threshold"]["g1_selected"],
              record["threshold"]["g2_published"], record["threshold"]["decided_exactly"]["reconstruction"],
              record["quantization"]["inconsistent_pairs"]):
        assert _fmt(n) in defects["STRING-THRESHOLD-RECONSTRUCTION-1"], n
    p = record["string_pairs"]
    for n in (p["records"], p["pairs"], p["proteins"], p["scratch_bytes"]):
        assert _fmt(n) in defects["STRING-PAIR-ACCOUNTING-1"], n
    legacy = record["reactome"]["legacy"]
    for d in ("KG-COHORT-DEPENDENT-PROJECTION-1", "GRAPH-NODE-UNIVERSE-SPLIT-DEPENDENT-1"):
        assert _fmt(legacy["train_edges_removed_by_adding_heldout_genes"]) in defects[d]
    assert _fmt(legacy["edges_train_universe"]) in defects["KG-COHORT-DEPENDENT-PROJECTION-1"]
    c, a = record["cohort"], record["label_authority"]
    s = defects["GENE-SYMBOL-SPLIT-IDENTITY-1"]
    assert f"{c['hgnc_ids_under_two_or_more_symbols']} HGNC identifiers" in s
    assert f"{c['hgnc_ids_whose_symbols_straddle_train_and_heldout']} of them" in s
    assert f"refuses the {a['refused_training_rows']} training rows" in s
    m = defects["STRING-GENE-MAPPING-1"]
    assert f"{_fmt(record['legacy_string_graph']['edges'])}-edge graph" in m
    assert f"{c['symbols_reaching_string_only_by_hgnc_id']} Run 15" in m
