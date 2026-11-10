"""Stage 1D-1b.2 (set "aq"): bytes bound to the parse, measured receipts, and the EVE and allele-frequency pilots --
each shown equal to the CURRENT connector's numbers (shadow comparison) while recording why a value is or is not
observed. Owner ruling 7f4b69d0 (2026-11-09), sections 4.1, 5 and 19. Author: Monzia Moodie."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.eve import EVEConnector, load_eve_entry_map
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import columns as K
from genomic_variant_classifier.information_sources import materialization as M
from genomic_variant_classifier.information_sources import observation as O

REPO = Path(__file__).resolve().parents[2]
S = O.ObservationState
REGISTRY = sorted(C.strict_load(REPO / C.LEDGER_PATH)["datasets"])


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ the verified reader
def test_the_bytes_parsed_are_the_bytes_hashed_in_one_pass(tmp_path):
    data = b"a,b\n1,2\n3,4\n"
    p = tmp_path / "x.csv"
    p.write_bytes(data)
    got = M.parse_verified(p, pd.read_csv)
    assert got.value.shape == (2, 2) and got.sha256 == _sha(data) and got.n_bytes == len(data)
    assert M.parse_verified(p, pd.read_csv, expected_sha256=_sha(data)).sha256 == _sha(data)
    with pytest.raises(ValueError, match="not the pinned"):
        M.parse_verified(p, pd.read_csv, expected_sha256="0" * 64)


def test_a_parser_that_stops_early_refuses(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"header\n" + b"row\n" * 100_000)              # larger than one buffer

    def first_line(stream):
        return stream.readline()

    with pytest.raises(ValueError, match="stopped before end of file"):
        M.parse_verified(p, first_line)


def test_a_gzipped_source_is_verified_on_its_compressed_bytes(tmp_path):
    raw = gzip.compress(b"x\n1\n2\n", mtime=0)
    p = tmp_path / "x.csv.gz"
    p.write_bytes(raw)
    got = M.parse_verified(p, lambda s: pd.read_csv(gzip.open(s)))
    assert list(got.value["x"]) == [1, 2] and got.sha256 == _sha(raw)


def test_a_columnar_file_is_parsed_from_the_verified_bytes(tmp_path):
    p = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(p, index=False)
    got = M.read_verified_bytes(p)
    assert got.sha256 == _sha(p.read_bytes()) and list(pd.read_parquet(io.BytesIO(got.value))["a"]) == [1, 2]


def test_a_directory_identity_is_its_canonical_manifest():
    a, rows = M.directory_manifest([("b.csv", "1" * 64, 3), ("a.csv", "2" * 64, 5)])
    b, _ = M.directory_manifest([("a.csv", "2" * 64, 5), ("b.csv", "1" * 64, 3)])
    assert a == b and [r["name"] for r in rows] == ["a.csv", "b.csv"]
    assert M.directory_manifest([("a.csv", "2" * 64, 6), ("b.csv", "1" * 64, 3)])[0] != a
    with pytest.raises(ValueError, match="once"):
        M.directory_manifest([("a.csv", "2" * 64, 5), ("a.csv", "1" * 64, 3)])


def test_a_materialised_source_must_be_registered():
    M.check_registered("dataset:eve", REGISTRY)
    for bad in ("dataset:nowhere", "derived:eve_score", "eve"):
        with pytest.raises(ValueError, match="not a dataset registered"):
            M.check_registered(bad, REGISTRY)


# ------------------------------------------------------------------------------------------------------- EVE fixture
EVE_HEADER = "wt_aa,position,mt_aa,EVE_scores_ASM,EVE_classes_75_pct_retained_ASM\n"


@pytest.fixture
def eve_fixture(tmp_path):
    d = tmp_path / "eve"
    d.mkdir()
    # GENEA_HUMAN: R10H scored 0.5 (a GENUINE 0.5), R10C unscored, G12V 0.9. P53_HUMAN -> TP53 by the entry map.
    (d / "GENEA_HUMAN.csv").write_text(EVE_HEADER + "R,10,H,0.5,Uncertain\nR,10,C,,\nG,12,V,0.9,Pathogenic\n")
    (d / "P53_HUMAN.csv").write_text(EVE_HEADER + "R,175,H,0.95,Pathogenic\nA,1,A,,\n")
    # Two domain files that fall back to the same gene prefix: K5E scored differently in each -> ambiguous.
    (d / "DOM_A_HUMAN.csv").write_text(EVE_HEADER + "K,5,E,0.2,Benign\nK,6,E,0.3,Benign\n")
    (d / "DOM_B_HUMAN.csv").write_text(EVE_HEADER + "K,5,E,0.7,Pathogenic\nK,6,E,0.3,Benign\n")
    (d / "notes.txt").write_text("ignored by the legacy glob\n")
    index = tmp_path / "uniprot.parquet"
    pd.DataFrame({"gene_symbol": ["GENEA", "TP53", "tp53x"], "uniprot_id": ["Q1", "P04637", "Q3"],
                  "entry_name": ["GENEA_HUMAN", "P53_HUMAN", " p53x_human "], "sequence": ["M", "M", "M"]}).to_parquet(
        index, index=False)
    frame = pd.DataFrame({
        "variant_id": [f"v{i}" for i in range(9)],
        "gene_symbol": ["GENEA", "GENEA", "GENEA", "TP53", "TP53", "GENEB", "DOM", "DOM", "GENEA"],
        "consequence": ["missense_variant"] * 6 + ["missense_variant", "missense_variant", "stop_gained"],
        "wt_aa": ["R", "R", "G", "R", None, "A", "K", "K", None],
        "protein_pos": pd.array([10, 10, 99, 175, pd.NA, 3, 5, 6, pd.NA], dtype="Int64"),
        "mut_aa": ["H", "C", "V", "H", None, "V", "E", "E", None],
        "protein_change": [None] * 9,
    })
    return d, index, frame


def _source(d, index):
    em = M.entry_map_from_index(index.read_bytes())
    return M.read_eve_source(d, em), em


def test_the_entry_map_from_verified_bytes_equals_the_legacy_loader(eve_fixture):
    _d, index, _f = eve_fixture
    assert M.entry_map_from_index(index.read_bytes()) == load_eve_entry_map(index)
    assert M.entry_map_from_index(index.read_bytes())["P53X_HUMAN"] == "TP53X"


def test_every_eve_file_parses_as_the_legacy_connector_parses_it_keeping_unscored_rows(eve_fixture):
    d, index, _f = eve_fixture
    src, em = _source(d, index)
    assert len(src.manifest) == 4 and src.files_resolved_by_map == 2 and src.files_by_fallback == 2
    for entry in src.manifest:
        assert entry["sha256"] == _sha((d / entry["name"]).read_bytes())
        legacy = EVEConnector(eve_path=None)._parse_single_csv(d / entry["name"], em).reset_index(drop=True)
        mine = src.rows[src.rows["file"] == entry["name"]].dropna(subset=["eve_score"]).reset_index(drop=True)
        assert mine[["gene_symbol", "aa_change"]].equals(legacy[["gene_symbol", "aa_change"]])
        assert np.array_equal(mine["eve_score"].to_numpy(), legacy["eve_score"].to_numpy())
    assert int(src.rows["eve_score"].isna().sum()) == 2                 # the unscored rows the legacy parse drops


def test_eve_tells_a_genuine_half_from_every_kind_of_absence(eve_fixture):
    d, index, frame = eve_fixture
    src, _em = _source(d, index)
    pop = M.population_from_frame(frame, "fixture")
    got = M.materialize_eve(frame, pop, src, registered_datasets=REGISTRY)
    col, det = got.column, {i: v for i, v in enumerate(M.EVE_DETAILS)}
    state = [K.STATE_CODES[c].value for c in col.states]
    detail = [det[int(c)] for c in col.details]
    assert state == ["observed", "not_in_source", "not_in_source", "observed", "bad_mapping", "not_in_source",
                     "bad_mapping", "observed", "unsupported_variant"]
    assert detail == ["", "substitution_unscored", "substitution_not_in_source", "", "missense_without_protein_key",
                      "gene_not_modeled", "ambiguous_source_record", "ambiguous_source_record",
                      "not_a_missense_substitution"]
    assert col.values[0] == 0.5 and col.values[3] == 0.95 and col.values[7] == 0.3     # 0.5 OBSERVED, not absent
    assert got.receipt.content_sha256 == src.content_sha256 and got.receipt.n_population == 9
    assert sum(got.receipt.state_counts.values()) == 9
    assert any("scored more than once" in n for n in got.receipt.notes)


def test_the_eve_projection_equals_the_current_connector_where_its_join_is_defined(eve_fixture):
    d, index, frame = eve_fixture
    src, em = _source(d, index)
    keep = frame["gene_symbol"] != "DOM"                     # the ambiguous gene multiplies legacy rows (below)
    sub = frame[keep].reset_index(drop=True)
    got = M.materialize_eve(sub, M.population_from_frame(sub, "fixture"), src, registered_datasets=REGISTRY)
    full = M.legacy_lookup_from_source(src)
    legacy = M.legacy_eve_values(sub, full)
    assert M.compare_legacy(M.engineer_fill(got.legacy_projection, 0.5), legacy)["identical"]
    # the full connector (its own read, its own cache) gives the same numbers
    connector = EVEConnector(eve_path=d, entry_map_path=index)
    connector._load_cache = lambda key: None
    connector._save_cache = lambda key, df: None
    assert np.array_equal(connector.annotate_dataframe(sub)["eve_score"].to_numpy(), legacy)
    # the restricted lookup changes nothing for this population
    restricted = M.legacy_lookup_from_source(src, got.source_rows_relevant)
    assert len(restricted) < len(full) and np.array_equal(M.legacy_eve_values(sub, restricted), legacy)
    with pytest.raises(ValueError, match="duplicated source keys multiplied"):
        M.legacy_eve_values(frame, full)                     # DOM:K5E and DOM:K6E occur twice: 9 rows become 11


def test_eve_absent_is_source_absent_with_no_digest(eve_fixture):
    _d, _index, frame = eve_fixture
    got = M.materialize_eve(frame, M.population_from_frame(frame, "fixture"), None, registered_datasets=REGISTRY)
    assert got.receipt.state_counts["source_absent"] == 9 and got.receipt.content_sha256 is None
    assert np.all(got.legacy_projection == 0.5)


def test_a_frame_out_of_population_order_refuses(eve_fixture):
    d, index, frame = eve_fixture
    src, _em = _source(d, index)
    pop = M.population_from_frame(frame, "fixture")
    with pytest.raises(ValueError, match="population order"):
        M.materialize_eve(frame.iloc[::-1].reset_index(drop=True), pop, src, registered_datasets=REGISTRY)


# --------------------------------------------------------------------------------------------- allele frequency fixture
@pytest.fixture
def af_fixture(tmp_path):
    g = tmp_path / "gnomad.parquet"
    pd.DataFrame({
        "variant_id": ["1:100:A:G", "1:200:C:T", "1:300:G:A", "1:300:G:A", "1:400:T:C", "1:400:T:C", "chr1:500:A:C",
                       "bad", "1:600:A:T"],
        "allele_freq": [0.0, 1e-5, 0.02, 0.02, 0.1, 0.2, 0.3, 0.5, np.nan],
    }).to_parquet(g, index=False)
    frame = pd.DataFrame({
        "variant_id": ["r0", "r1", "r2", "r3", "r4", "r5", "r6"],
        "chrom": ["1", "1", "1", "1", "1", "1", "1"], "pos": [100, 200, 300, 400, 500, 600, 700],
        "ref": ["A", "C", "G", "T", "A", "A", "G"], "alt": ["G", "T", "A", "C", "C", "T", "C"],
        "allele_freq": [np.nan] * 7,
    })
    return g, frame


def test_an_observed_zero_is_not_an_unavailable_frequency(af_fixture):
    g, frame = af_fixture
    pop = M.population_from_frame(frame, "fixture")
    got = M.materialize_gnomad_af(frame, pop, M.read_verified_bytes(g), registered_datasets=REGISTRY)
    det = dict(enumerate(M.AF_DETAILS))
    state = [K.STATE_CODES[c].value for c in got.column.states]
    detail = [det[int(c)] for c in got.column.details]
    # r4: the source writes 'chr1:500:A:C', which the legacy locus parse cannot read (it strips one non-numeric prefix
    # field, here 'chr1', leaving three fields), so the record is dropped -- by the legacy join and, identically, here.
    assert state == ["observed", "observed", "observed", "bad_mapping", "not_in_source", "not_in_source", "not_in_source"]
    assert detail == ["", "", "", "ambiguous_source_record", "locus_not_in_source", "record_without_value",
                      "locus_not_in_source"]
    assert got.receipt.inputs["dataset:gnomad_af#records"]["unparsable_variant_ids"] == 2
    assert got.column.values[0] == 0.0 and got.column.values[1] == 1e-5      # an OBSERVED zero
    assert got.receipt.inputs["dataset:gnomad_af#records"]["ambiguous_loci"] == 1
    assert got.receipt.content_sha256 == _sha(g.read_bytes())


def test_the_af_projection_equals_the_current_join(af_fixture):
    g, frame = af_fixture
    got = M.materialize_gnomad_af(frame, M.population_from_frame(frame, "fixture"), M.read_verified_bytes(g),
                                  registered_datasets=REGISTRY)
    legacy = M.legacy_af_values(frame, g)
    assert M.compare_legacy(got.legacy_projection, legacy) == {"rows": 7, "equal": 7, "differ": 0, "identical": True}
    assert legacy[3] == 0.1                                    # keep-first resolved the ambiguous locus silently


def test_a_cohort_frequency_takes_precedence_as_in_the_legacy_join(af_fixture):
    g, frame = af_fixture
    frame = frame.copy()
    frame.loc[6, "allele_freq"] = 0.25
    got = M.materialize_gnomad_af(frame, M.population_from_frame(frame, "fixture"), M.read_verified_bytes(g),
                                  registered_datasets=REGISTRY)
    assert M.compare_legacy(got.legacy_projection, M.legacy_af_values(frame, g))["identical"]
    assert any("cohort rows carry their own allele_freq" in n for n in got.receipt.notes)


# ----------------------------------------------------------------------------------------------------------- receipts
def test_the_receipt_is_measured_by_the_materialiser(af_fixture):
    g, frame = af_fixture
    got = M.materialize_gnomad_af(frame, M.population_from_frame(frame, "fixture"), M.read_verified_bytes(g),
                                  registered_datasets=REGISTRY)
    r = got.receipt.as_dict()
    symbol, digest = M.parser_identity("materialize_gnomad_af")
    assert r["parser"] == symbol and r["parser_sha256"] == digest == C.symbol_closure_digest(REPO, symbol)
    assert r["population_sha256"] == M.population_from_frame(frame, "fixture").sha256
    assert r["column_sha256"] == got.column.sha256 and r["schema"] == M.RECEIPT_SCHEMA
    json.dumps(r, allow_nan=False)                             # serialisable, no NaN
