"""Stage 1D-1b.3 (set "ar"): source-bound graph identity -- STRING exact pairs read in one verified pass, the four-way
threshold classification in integer thousandths, canonical HGNC mapping with its collision rule, a graph materialisation
computed from content and refused on any mismatch, the legacy cache key, and pathway relations that keep their source and
survive node-universe growth. Owner rulings f2ec92cd (Unit 1D-1b.3, section III) and 7f4b69d0 (sections 7-10).
Author: Monzia Moodie."""
from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.data import kg_edges as K
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import graph_identity as G

REPO = Path(__file__).resolve().parents[2]
HEADER = " ".join(G.STRING_HEADER)
HGNC_HEADER = "hgnc_id\tsymbol\tname\tstatus\talias_symbol\tprev_symbol"


def _write(path: Path, lines, *, gz: bool = True, header: str = HEADER) -> Path:
    data = (header + "\n" + "".join(f"{x}\n" for x in lines)).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)
    return path


def _both(p1: str, p2: str, scores: str) -> list[str]:
    return [f"{p1} {p2} {scores}", f"{p2} {p1} {scores}"]


PAIRS = _both("9606.B", "9606.A", "0 0 0 10 20 0 0 300") + _both("9606.A", "9606.C", "1 2 3 4 5 6 7 800")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =================================================================================================== exact pairs
def test_every_pair_once_in_canonical_order_from_one_verified_pass(tmp_path):
    p = _write(tmp_path / "links.gz", PAIRS)
    t = G.read_string_pairs(p, expected_sha256=_sha(p), partitions=3)
    assert t.proteins == ("9606.A", "9606.B", "9606.C")
    assert t.a.tolist() == [0, 0] and t.b.tolist() == [1, 2]
    assert t.channels.tolist() == [[0, 0, 0, 10, 20, 0, 0], [1, 2, 3, 4, 5, 6, 7]]
    assert t.combined.tolist() == [300, 800] and t.n_records == 4 and len(t) == 2
    assert t.source_sha256 == _sha(p) and t.n_bytes == p.stat().st_size
    assert t.scratch_bytes == 4 * G._RECORD.itemsize


def test_content_identity_ignores_record_order_compression_and_partitioning(tmp_path):
    a = G.read_string_pairs(_write(tmp_path / "a.gz", PAIRS), partitions=1)
    b = G.read_string_pairs(_write(tmp_path / "b.txt", list(reversed(PAIRS)), gz=False), partitions=7)
    assert a.content_sha256 == b.content_sha256 and a.source_sha256 != b.source_sha256
    c = G.read_string_pairs(_write(tmp_path / "c.gz", PAIRS[:2] + _both("9606.A", "9606.C", "1 2 3 4 5 6 7 801")))
    assert c.content_sha256 != a.content_sha256                       # one thousandth of one score changes the identity


@pytest.mark.parametrize("lines, message", [
    (["9606.A 9606.B 0 0 0 0 0 0 0 1"], "without its reciprocal"),
    (["9606.A 9606.B 0 0 0 0 0 0 0 1", "9606.B 9606.A 0 0 0 0 0 0 0 2"], "different scores"),
    (["9606.A 9606.B 0 0 0 0 0 0 0 1", "9606.A 9606.B 0 0 0 0 0 0 0 1"], "duplicated in one direction"),
    (_both("9606.A", "9606.B", "0 0 0 0 0 0 0 1") + ["9606.A 9606.B 0 0 0 0 0 0 0 1"], "more than twice"),
    (["9606.A 9606.A 0 0 0 0 0 0 0 1"], "paired with itself"),
    (_both("9606.A", "9606.B", "0 0 0 0 0 0 0 1001"), "above 1000"),
    (["9606.A 9606.B 0 0 0 0 0 0 0 01"], "canonical integer"),
    (["9606.A 9606.B 0 0 0 0 0 0 0 -1"], "canonical integer"),
    (["9606.A 9606.B 0 0 0 0 0 0 0 1.0"], "canonical integer"),
    (["9606.A  9606.B 0 0 0 0 0 0 0 1"], "canonical integer"),
    (["9606.A 9606.B 0 0 0 0 0 0 0 1\r"], "carriage return"),
    (["9606.A 9606.B 0 0 0 0 0 0 1"], "canonical integer"),
    (_both("9606.A", "10090.B", "0 0 0 0 0 0 0 1"), "taxa"),
])
def test_records_are_validated_and_never_silently_reduced(tmp_path, lines, message):
    with pytest.raises(ValueError, match=message):
        G.read_string_pairs(_write(tmp_path / "bad.gz", lines), partitions=2, scratch_dir=tmp_path)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("gvc_string_pairs_")]   # scratch removed on failure


def test_header_pin_and_scratch_lifecycle(tmp_path):
    with pytest.raises(ValueError, match="header"):
        G.read_string_pairs(_write(tmp_path / "h.gz", PAIRS, header=HEADER.replace("cooccurence", "cooccurrence")))
    p = _write(tmp_path / "ok.gz", PAIRS)
    with pytest.raises(ValueError, match="not the pinned"):
        G.read_string_pairs(p, expected_sha256="0" * 64)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    G.read_string_pairs(p, scratch_dir=scratch)
    assert list(scratch.iterdir()) == []
    with pytest.raises(ValueError, match="partitions"):
        G.read_string_pairs(p, partitions=0)


def test_protein_info_is_read_strictly(tmp_path):
    rows = "\t".join(G.STRING_INFO_HEADER) + "\n9606.A\tAAA\t10\tx\n9606.B\tBBB\t20\ty\n"
    p = tmp_path / "info.gz"
    p.write_bytes(gzip.compress(rows.encode(), mtime=0))
    info = G.read_string_info(p)
    assert info.names == {"9606.A": "AAA", "9606.B": "BBB"} and info.source_sha256 == _sha(p)
    assert info.n_bytes == p.stat().st_size
    with pytest.raises(TypeError):
        info.names["9606.Z"] = "ZZZ"                                    # read-only: the digest describes this content
    p.write_bytes(gzip.compress((rows + "9606.A\tAAA2\t1\tz\n").encode(), mtime=0))
    with pytest.raises(ValueError, match="listed twice"):
        G.read_string_info(p)
    p.write_bytes(gzip.compress((rows + "9606.C\tCCC\t1\n").encode(), mtime=0))
    with pytest.raises(ValueError, match="fields"):
        G.read_string_info(p)


# ======================================================================================== reconstruction and threshold
def _reference_string_score(values, prior=G.STRING_PRIOR):
    """The owner's reference formula (gvc_1d1b_integrity_strict_reference.string_score), independently re-typed."""
    product = 1.0
    for v in values:
        product *= 1.0 - max(0.0, (v - prior) / (1.0 - prior))
    return (1.0 - product) + product * prior


def test_the_combination_rule_equals_the_reference_and_is_nested():
    rng = np.random.default_rng(7)
    k = rng.integers(0, 1001, size=(500, 7))
    got = G.combine_channels(k)
    want = np.array([_reference_string_score(row / 1000) for row in k])
    assert np.max(np.abs(got - want)) < 1e-12
    assert np.all(G.combine_channels(k[:, :5]) <= got + 1e-15)          # dropping channels never raises the score
    lo, hi = G.channel_bounds(k, G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    assert np.all(lo <= got) and np.all(got <= hi)


def _table(tmp_path, rows) -> G.StringPairTable:
    lines = []
    for i, scores in enumerate(rows):
        lines += _both(f"9606.P{i:03d}", f"9606.Q{i:03d}", scores)
    return G.read_string_pairs(_write(tmp_path / "t.gz", lines))


def test_four_way_classification_with_the_published_flag(tmp_path):
    # One non-zero channel reconstructs to itself, so its half-step bounds are k - 0.5 and k + 0.5 thousandths.
    t = _table(tmp_path, ["0 0 0 0 900 0 0 950",     # selected robust, published passes
                          "0 0 0 0 701 0 0 750",     # selected; upper/lower bounds 700.5/701.5 -> robust
                          "0 0 0 0 900 0 0 650",     # reconstruction passes, published fails -> not selected, robust
                          "0 0 0 0 100 0 0 720",     # excluded robust, published passes
                          "0 0 0 0 699 0 0 710"])    # bounds 698.5..699.5 < 700 -> excluded robust
    c = G.classify_threshold(t, ("experimental",), 700)
    order = {t.proteins[t.a[i]]: i for i in range(len(t))}
    st = {name: G.ThresholdStatus(list(G.ThresholdStatus)[c.status[i]]).value for name, i in order.items()}
    assert st == {"9606.P000": "selected_robust", "9606.P001": "selected_robust", "9606.P002": "excluded_robust",
                  "9606.P003": "excluded_robust", "9606.P004": "excluded_robust"}
    assert c.published_passes.tolist() == [True, True, False, True, True]
    assert c.selected.tolist() == [True, True, False, False, False]
    assert sum(c.crosstab().values()) == len(t)


def test_threshold_sensitive_pairs_are_named_on_both_sides(tmp_path):
    # Every (coexpression, experimental) integer pair in [300, 700): find one whose half-step bounds straddle 0.700 with
    # the reconstruction above it and one with the reconstruction below it (both clear of floating-point error).
    co, ex = np.meshgrid(np.arange(300, 700), np.arange(300, 700), indexing="ij")
    m = np.zeros((co.size, 7), dtype=np.int64)
    m[:, 3], m[:, 4] = co.ravel(), ex.ravel()
    rec = G.combine_channels(m) * 1000
    lo, hi = (x * 1000 for x in G.channel_bounds(m, G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED))
    straddle = (lo < 700) & (700 <= hi) & (np.abs(rec - 700) > 1e-6)
    above = m[np.flatnonzero(straddle & (rec >= 700))[0]]
    below = m[np.flatnonzero(straddle & (rec < 700))[0]]
    t = _table(tmp_path, [f"0 0 0 {above[3]} {above[4]} 0 0 800", f"0 0 0 {below[3]} {below[4]} 0 0 800"])
    c = G.classify_threshold(t, ("coexpression", "experimental"), 700)
    assert sorted(list(G.ThresholdStatus)[s].value for s in c.status) == ["excluded_sensitive", "selected_sensitive"]
    assert c.crosstab()["selected_sensitive|published_passes"] == 1
    assert c.crosstab()["excluded_sensitive|published_passes"] == 1


def test_comparisons_at_the_threshold_are_decided_exactly(tmp_path):
    # One channel: the combination returns the channel itself, so 700 reconstructs to EXACTLY 0.700, which a float
    # evaluation may place just below (real STRING v12: 52 such pairs under the five non-curated channels).
    t = _table(tmp_path, ["0 0 0 0 700 0 0 700", "0 0 0 0 699 0 0 699", "0 0 0 0 0 0 0 41"])
    assert abs(G.combine_channels(t.channels[:1, 4:5])[0] * 1000 - 700) < 1e-9
    assert G.combine_exact([G.Fraction(700, 1000)]) == G.Fraction(7, 10)
    c = G.classify_threshold(t, ("experimental",), 700)
    assert [list(G.ThresholdStatus)[s].value for s in c.status] == ["selected_sensitive", "excluded_robust",
                                                                     "excluded_robust"]
    assert c.decided_exactly["reconstruction"] == 1
    # At a threshold of 50 thousandths the float evaluation of a single channel 50 is 49.999999999999986: a float comparison
    # would exclude the pair; the exact decision selects it.
    low = _table(tmp_path, ["0 0 0 0 50 0 0 60"])
    assert G.combine_channels(low.channels[:, 4:5])[0] * 1000 < 50
    c50 = G.classify_threshold(low, ("experimental",), 50)
    assert c50.selected.tolist() == [True] and c50.decided_exactly["reconstruction"] == 1
    m = G.model_consistency(t, G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    # all channels 0: both bounds are the prior, exactly 41 thousandths, floored exactly (a float may give 40.999...)
    assert m["inconsistent_pairs"] == 0 and m["bounds_floored_exactly"] >= 2


def test_exact_and_float_combination_agree_away_from_boundaries():
    rng = np.random.default_rng(11)
    k = rng.integers(0, 1001, size=(200, 5))
    approx = G.combine_channels(k) * 1000
    exact = np.array([float(G._exact_milli(row, 0)) for row in k])
    assert np.max(np.abs(approx - exact)) < 1e-9


def test_the_threshold_is_integer_thousandths(tmp_path):
    t = _table(tmp_path, ["0 0 0 0 800 0 0 900"])
    for bad in (0.7, True, 0, 1001, "700"):
        with pytest.raises(ValueError, match="integer number of thousandths"):
            G.classify_threshold(t, ("experimental",), bad)
    with pytest.raises(ValueError, match="distinct members"):
        G.classify_threshold(t, ("experimental", "experimental"), 700)


def test_quantization_model_consistency_is_measured(tmp_path):
    good = int(math.floor(G.combine_channels(np.array([[0, 0, 0, 0, 400, 0, 300]]))[0] * 1000))
    t = _table(tmp_path, [f"0 0 0 0 400 0 300 {good}", f"0 0 0 0 400 0 300 {good + 5}"])
    r = G.model_consistency(t, G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    assert r["pairs"] == 2 and r["inconsistent_pairs"] == 1 and r["max_miss_thousandths"] >= 4


# ============================================================================================ canonical gene identity
def _hgnc(tmp_path, rows) -> G.HgncIndex:
    p = tmp_path / "hgnc.txt"
    p.write_bytes((HGNC_HEADER + "\n" + "".join("\t".join(r) + "\n" for r in rows)).encode("utf-8"))
    return G.read_hgnc(p)


HGNC_ROWS = [("HGNC:1", "AAA", "a", "Approved", "", "OLDA"),
             ("HGNC:2", "BBB", "b", "Approved", "ALIASB", '"OLDB|SHARED"'),
             ("HGNC:3", "CCC", "c", "Approved", "", "SHARED"),
             ("HGNC:4", "C1orf2", "d", "Approved", "", "")]


def test_hgnc_resolution_is_exact_and_ambiguity_is_named(tmp_path):
    h = _hgnc(tmp_path, HGNC_ROWS)
    S = G.MappingStatus
    assert h.resolve("AAA") == ("HGNC:1", S.APPROVED_SYMBOL)
    assert h.resolve("OLDB") == ("HGNC:2", S.PREVIOUS_SYMBOL_UNIQUE)
    assert h.resolve("SHARED") == (None, S.PREVIOUS_SYMBOL_AMBIGUOUS)
    assert h.resolve("ALIASB") == (None, S.UNRESOLVED)               # aliases are not canonical identity
    assert h.resolve("C1ORF2") == (None, S.UNRESOLVED)               # case is part of an HGNC symbol
    assert h.resolve("C1orf2") == ("HGNC:4", S.APPROVED_SYMBOL)
    with pytest.raises(ValueError, match="repeated"):
        _hgnc(tmp_path, HGNC_ROWS + [("HGNC:1", "ZZZ", "z", "Approved", "", "")])
    with pytest.raises(ValueError, match="malformed"):
        _hgnc(tmp_path, [("1", "AAA", "a", "Approved", "", "")])


def test_canonical_collisions_are_resolved_by_approved_symbol_precedence_and_reported(tmp_path):
    h = _hgnc(tmp_path, HGNC_ROWS)
    info = {"9606.P1": "AAA", "9606.P2": "OLDA", "9606.P3": "OLDB", "9606.P4": "SHARED", "9606.P5": "NOPE"}
    m, rep = G.map_string_proteins(G.StringInfo(info, "1" * 64, 0), h)
    st = {r.protein_id: (r.hgnc_id, r.status.value) for r in m.rows}
    assert st == {"9606.P1": ("HGNC:1", "approved_symbol"), "9606.P2": (None, "collision_superseded"),
                  "9606.P3": ("HGNC:2", "previous_symbol_unique"), "9606.P4": (None, "previous_symbol_ambiguous"),
                  "9606.P5": (None, "unresolved")}
    assert rep["collisions"] == [{"hgnc_id": "HGNC:1", "kept": "9606.P1",
                                  "claimants": [["9606.P1", "AAA", "approved_symbol"],
                                                ["9606.P2", "OLDA", "previous_symbol_unique"]]}]
    info2 = {"9606.Q1": "OLDA", "9606.Q2": "OLDA2"}
    h2 = _hgnc(tmp_path, [("HGNC:1", "AAA", "a", "Approved", "", '"OLDA|OLDA2"')])
    m2, rep2 = G.map_string_proteins(G.StringInfo(info2, "1" * 64, 0), h2)
    assert {r.status.value for r in m2.rows} == {"collision_ambiguous"} and rep2["collisions"][0]["kept"] is None
    m3, _ = G.map_string_proteins(G.StringInfo({**info, "9606.P5": "CCC"}, "1" * 64, 0), h)
    assert m3.mapping_sha256 != m.mapping_sha256


def test_a_mapping_cannot_be_constructed_non_injective():
    row = G.ProteinGeneRow
    with pytest.raises(ValueError, match="no two proteins"):
        G.ProteinGeneMapping((row("a", "X", "HGNC:1", G.MappingStatus.APPROVED_SYMBOL, "dataset:hgnc"),
                              row("b", "Y", "HGNC:1", G.MappingStatus.PREVIOUS_SYMBOL_UNIQUE, "dataset:hgnc")),
                             "1" * 64, "2" * 64)
    with pytest.raises(ValueError, match="exactly when"):
        G.ProteinGeneMapping((row("a", "X", None, G.MappingStatus.APPROVED_SYMBOL, "dataset:hgnc"),), "1" * 64, "2" * 64)


# =================================================================================================== graph identity
def _policy(src: str, info: str, hg: str, **kw) -> G.GraphPolicy:
    base = dict(source="dataset:string", release="12.0", source_sha256=src, info_sha256=info,
                mapping_source="dataset:hgnc", mapping_source_sha256=hg, regime=G.Regime.G1,
                selection_channels=("experimental",), attribute_channels=("experimental", "coexpression"),
                threshold_milli=700, quantization=G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    base.update(kw)
    return G.GraphPolicy(**base)


def test_policy_rules_for_the_three_regimes():
    d = ("1" * 64, "2" * 64, "3" * 64)
    with pytest.raises(ValueError, match="G0"):
        _policy(*d, regime=G.Regime.G0)
    with pytest.raises(ValueError, match="curation-reduced"):
        _policy(*d, selection_channels=("experimental", "database"))
    with pytest.raises(ValueError, match="quantization"):
        _policy(*d, quantization=None)
    with pytest.raises(ValueError, match="published combined"):
        _policy(*d, regime=G.Regime.G2, quantization=None)
    g2 = _policy(*d, regime=G.Regime.G2, selection_channels=G.STRING_CHANNELS, quantization=None)
    with pytest.raises(ValueError, match="thousandths"):
        _policy(*d, threshold_milli=0.7)
    with pytest.raises(ValueError, match="measured lower-case"):
        _policy("x" * 64, *d[1:])
    with pytest.raises(ValueError, match="unsupported"):
        _policy(*d, node_universe="cohort")
    digests = {g2.policy_sha256, _policy(*d).policy_sha256, _policy(*d, threshold_robust_only=True).policy_sha256,
               _policy(*d, threshold_milli=701).policy_sha256, _policy(*d, release="12.5").policy_sha256,
               _policy(*d, attribute_channels=("experimental",)).policy_sha256}
    assert len(digests) == 6                                            # every field is identity


def _graph_inputs(tmp_path):
    lines = (_both("9606.A", "9606.B", "0 0 0 100 900 0 0 950") + _both("9606.A", "9606.C", "0 0 0 0 950 0 0 960") +
             _both("9606.B", "9606.D", "0 0 0 0 990 0 0 995") + _both("9606.C", "9606.D", "0 0 0 0 100 0 0 800") +
             _both("9606.A", "9606.E", "0 0 0 0 980 0 0 985"))
    links = _write(tmp_path / "links.gz", lines)
    t = G.read_string_pairs(links)
    h = _hgnc(tmp_path, [("HGNC:1", "AAA", "", "Approved", "", "OLDC"), ("HGNC:2", "BBB", "", "Approved", "", ""),
                         ("HGNC:4", "DDD", "", "Approved", "", "")])
    info = {"9606.A": "AAA", "9606.B": "BBB", "9606.C": "OLDC", "9606.D": "DDD", "9606.E": "UNKNOWN"}
    m, _ = G.map_string_proteins(G.StringInfo(info, "9" * 64, 0), h)
    return t, m, h


def test_materialised_identity_is_computed_and_reductions_are_counted(tmp_path):
    t, m, h = _graph_inputs(tmp_path)
    pol = _policy(t.source_sha256, "9" * 64, h.source_sha256)
    g = G.materialize_string_graph(t, m, pol)
    # A-B, A-C (C superseded by A's approved claim -> unmapped), B-D, A-E (E unresolved): selected on experimental >= 700
    assert g.edges == (("HGNC:1", "HGNC:2"), ("HGNC:2", "HGNC:4"))
    assert g.attributes.tolist() == [[900, 100], [990, 0]]
    assert g.nodes == ("HGNC:1", "HGNC:2", "HGNC:4")                    # the source-defined universe, isolated or not
    assert g.report["selected_pairs"] == 4 and g.report["selected_pairs_with_unmapped_endpoint"] == 2
    assert g.report["unmapped_endpoint_status_counts"] == {"collision_superseded": 1, "unresolved": 1}
    assert G.verify_graph_consistency(pol, g) == g.materialization
    assert g.materialization.relation_policy_sha256 == pol.policy_sha256
    assert g.materialization.mapping_sha256 == m.mapping_sha256 and g.materialization.source_sha256 == t.source_sha256


def test_the_consistency_check_recomputes_identity_and_refuses_every_mismatch(tmp_path):
    import dataclasses

    t, m, h = _graph_inputs(tmp_path)
    pol = _policy(t.source_sha256, "9" * 64, h.source_sha256)
    g = G.materialize_string_graph(t, m, pol)
    with pytest.raises(ValueError, match="different graph policy"):
        G.verify_graph_consistency(_policy(t.source_sha256, "9" * 64, h.source_sha256, threshold_milli=701), g)
    tampered = dataclasses.replace(g, edges=g.edges[:1], attributes=g.attributes[:1])
    with pytest.raises(ValueError, match="edge_content_sha256"):
        G.verify_graph_consistency(pol, tampered)
    renamed = dataclasses.replace(g, nodes=g.nodes + ("HGNC:9",))
    with pytest.raises(ValueError, match="node_universe_sha256"):
        G.verify_graph_consistency(pol, renamed)
    forged = dataclasses.replace(g, materialization=dataclasses.replace(g.materialization, mapping_sha256="0" * 64))
    with pytest.raises(ValueError, match="mapping_sha256"):
        G.verify_graph_consistency(pol, forged)
    with pytest.raises(ValueError, match="pinned source"):
        G.materialize_string_graph(t, m, _policy("0" * 64, "9" * 64, h.source_sha256))
    with pytest.raises(ValueError, match="pinned protein-info"):
        G.materialize_string_graph(t, m, _policy(t.source_sha256, "8" * 64, h.source_sha256))


def test_g2_and_the_threshold_robust_graph(tmp_path):
    t, m, h = _graph_inputs(tmp_path)
    g2 = G.materialize_string_graph(t, m, _policy(t.source_sha256, "9" * 64, h.source_sha256, regime=G.Regime.G2,
                                                  selection_channels=G.STRING_CHANNELS, quantization=None))
    assert g2.report["selected_pairs"] == 5 and ("HGNC:1", "HGNC:2") in g2.edges and g2.report["threshold_crosstab"] is None
    robust = G.materialize_string_graph(t, m, _policy(t.source_sha256, "9" * 64, h.source_sha256,
                                                      threshold_robust_only=True))
    assert set(robust.edges) <= set(G.materialize_string_graph(
        t, m, _policy(t.source_sha256, "9" * 64, h.source_sha256)).edges)


def test_the_legacy_cache_key_cannot_tell_policies_apart():
    assert G.legacy_string_cache_name(REPO) == "string_graph_{threshold}.pkl"
    d = ("1" * 64, "2" * 64, "3" * 64)
    policies = [_policy(*d), _policy(*d, threshold_robust_only=True), _policy(*d, release="11.5"),
                _policy(*d, regime=G.Regime.G2, selection_channels=G.STRING_CHANNELS, quantization=None),
                _policy("4" * 64, *d[1:])]
    r = G.legacy_cache_collisions(REPO, policies)
    assert r == {"template": "string_graph_{threshold}.pkl", "policies": 5, "filenames": 1,
                 "policies_per_filename": {"string_graph_700.pkl": 5}}


def test_the_census_does_not_admit_any_string_regime_strictly():
    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    d = ("1" * 64, "2" * 64, "3" * 64)
    for pol in (_policy(*d), _policy(*d, regime=G.Regime.G2, selection_channels=G.STRING_CHANNELS, quantization=None)):
        r = G.census_admission(pol, ledger)
        assert r["strict_admissible"] is False
        assert r["information_sources"]["string_relation:combined_score_threshold"] == "exclude_strict"
        assert r["information_sources"]["string_edge_attribute:experimental"] == "conditional_only"


# =============================================================================================== pathway relations
def test_gene_sets_are_keyed_by_stable_id_and_resolved_canonically(tmp_path):
    h = _hgnc(tmp_path, HGNC_ROWS)
    gmt = tmp_path / "sets.gmt"
    gmt.write_bytes(b"Pathway one\tR-1\tAAA\tOLDB\tNOPE\nPathway one\tR-2\tCCC\tAAA\t\nSmall\tR-3\tAAA\n")
    s = G.read_gene_sets(gmt, h, source_id="reactome", relation="shares_pathway")
    assert s.sets == {"R-1": frozenset({"HGNC:1", "HGNC:2"}), "R-2": frozenset({"HGNC:3", "HGNC:1"}),
                      "R-3": frozenset({"HGNC:1"})}
    assert s.unresolved_members == 1 and s.duplicate_names == 1 and s.relation_key == ("reactome", "shares_pathway")
    assert s.source_sha256 == _sha(gmt) and s.hgnc_sha256 == h.source_sha256
    gmt.write_bytes(b"A\tR-1\tAAA\tBBB\nB\tR-1\tCCC\tAAA\n")
    with pytest.raises(ValueError, match="repeated"):
        G.read_gene_sets(gmt, h, source_id="reactome", relation="shares_pathway")


def _source(sets: dict[str, set[str]]) -> G.GeneSetSource:
    return G.GeneSetSource("reactome", "shares_pathway", {k: frozenset(v) for k, v in sets.items()},
                           {k: k for k in sets}, "1" * 64, "2" * 64, 0, {}, 0)


SETS = {"P1": {"a", "b", "c"}, "P2": {"a", "b"}, "P3": {"c", "d", "e", "f"}}


def test_weights_use_the_global_size_and_sum_exactly():
    proj = G.project_pathways(_source(SETS), {"a", "b", "c", "d", "e", "f"}, max_set_size=None)
    assert proj.edges[("a", "b")] == math.fsum([1 / 2, 1 / 1]) and proj.edges[("c", "d")] == 1 / 3
    small = G.project_pathways(_source(SETS), {"a", "c", "d"}, max_set_size=None)
    assert small.edges == {("a", "c"): 1 / 2, ("c", "d"): 1 / 3}         # |S_s| stays global inside a small universe
    capped = G.project_pathways(_source(SETS), {"a", "b", "c", "d", "e", "f"}, max_set_size=3)
    assert capped.sets_over_cap == 1 and ("c", "d") not in capped.edges


def test_node_universe_growth_never_removes_an_edge_but_the_legacy_projection_does():
    small_u, large_u = {"c", "d"}, {"c", "d", "e", "f"}
    new_small = G.project_pathways(_source(SETS), small_u, max_set_size=3)
    new_large = G.project_pathways(_source(SETS), large_u, max_set_size=3)
    assert G.induced_subgraph_consistency(new_small.edges, small_u, new_large.edges)["consistent"] == 1
    # The legacy rule caps the set size AFTER restricting to the cohort: P3 has 2 cohort genes in the small universe
    # (kept) and 4 in the large one (dropped), so adding e and f deletes the c-d edge.
    legacy = {k: list(v) for k, v in SETS.items()}
    old_small = {e: 1.0 for e in K.co_membership_edges(legacy, max_set_size=3, restrict_to=small_u)}
    old_large = {e: 1.0 for e in K.co_membership_edges(legacy, max_set_size=3, restrict_to=large_u)}
    r = G.induced_subgraph_consistency(old_small, small_u, old_large)
    assert r["consistent"] == 0 and r["removed_by_growth"] == 1


def test_the_factorised_operator_equals_the_weighted_clique_projection():
    rng = np.random.default_rng(3)
    genes = [f"g{i}" for i in range(12)]
    sets = {f"S{j}": set(rng.choice(genes, size=int(rng.integers(2, 7)), replace=False)) for j in range(9)}
    x = {g: float(rng.normal()) for g in genes[:10]}                   # g10, g11 are outside the universe
    proj = G.project_pathways(_source(sets), x.keys(), max_set_size=None).edges
    explicit = {g: 0.0 for g in x}
    for (u, v), w in proj.items():
        explicit[u] += w * x[v]
        explicit[v] += w * x[u]
    implicit = G.hypergraph_matvec(sets, x)
    assert max(abs(implicit[g] - explicit[g]) for g in x) < 1e-12


# ============================================================== line endings (set "ar" revision B, Windows V1 2026-11-10)
def _crlf(path: Path, text: str, *, gz: bool = False) -> Path:
    data = text.replace("\n", "\r\n").encode("utf-8")       # exactly what Path.write_text(text) writes on Windows
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)
    return path


def _one_crlf_row(path: Path, head: str, rows: list[str], *, gz: bool = False) -> Path:
    """Every line LF except the LAST row, which ends CRLF (a file appended to by a Windows text-mode write): the header
    check alone cannot see it, so this case proves each reader checks its rows too."""
    data = (head + "".join(r + "\n" for r in rows[:-1]) + rows[-1] + "\r\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)
    return path


_CR_SOURCES = {    # reader: (header line or "", data rows); in every source the last field of a row is NOT validated
    "string_links": (HEADER + "\n", PAIRS),
    "string_info": ("\t".join(G.STRING_INFO_HEADER) + "\n", ["9606.A\tAAA\t1\tx", "9606.B\tBBB\t1\ty"]),
    "hgnc": ("hgnc_id\tsymbol\tstatus\tprev_symbol\talias_symbol\tname\n",          # the required columns are NOT last
             ["HGNC:1\tAAA\tApproved\t\t\ta", "HGNC:2\tBBB\tApproved\t\t\tb"]),
    "gene_sets": ("", ["P\tR-1\tAAA\tBBB", "Q\tR-2\tCCC\tAAA"]),
}


@pytest.mark.parametrize("where", ["every_line", "one_row"])
@pytest.mark.parametrize("reader", sorted(_CR_SOURCES))
def test_carriage_returns_are_refused_by_every_reader(tmp_path, reader, where):
    """Windows V1 of set "ar" (2026-11-10) failed because a fixture written by Path.write_text had CRLF line endings. A
    CRLF GMT had been SILENTLY accepted with the last member of every set unresolved; every reader now refuses a carriage
    return by name, on every platform, in its header and in every row."""
    h = _hgnc(tmp_path, HGNC_ROWS)
    head, rows = _CR_SOURCES[reader]
    gz = reader.startswith("string")
    p = tmp_path / ("src.gz" if gz else "src.txt")
    if where == "every_line":
        _crlf(p, head + "".join(r + "\n" for r in rows), gz=gz)
    else:
        _one_crlf_row(p, head, rows, gz=gz)
    with pytest.raises(ValueError, match="carriage return"):
        if reader == "string_links":
            G.read_string_pairs(p)
        elif reader == "string_info":
            G.read_string_info(p)
        elif reader == "hgnc":
            G.read_hgnc(p)
        else:
            G.read_gene_sets(p, h, source_id="reactome", relation="shares_pathway")
    if where == "one_row":       # the same file with the carriage return removed is accepted: only the CR is refused
        data = p.read_bytes()
        raw = (gzip.decompress(data) if gz else data).replace(b"\r", b"")
        p.write_bytes(gzip.compress(raw, mtime=0) if gz else raw)
        if reader == "string_links":
            assert len(G.read_string_pairs(p)) == 2
        elif reader == "string_info":
            assert G.read_string_info(p).names == {"9606.A": "AAA", "9606.B": "BBB"}
        elif reader == "hgnc":
            assert G.read_hgnc(p).approved == {"AAA": "HGNC:1", "BBB": "HGNC:2"}
        else:
            got = G.read_gene_sets(p, h, source_id="reactome", relation="shares_pathway").sets
            assert got == {"R-1": frozenset({"HGNC:1", "HGNC:2"}), "R-2": frozenset({"HGNC:3", "HGNC:1"})}


def test_fixtures_for_byte_strict_readers_are_written_as_bytes():
    """The error class behind the Windows V1 failure: a test module that feeds the byte-strict readers of graph_identity
    may not write a fixture with Path.write_text unless it fixes the line ending (newline=...); bytes are identical on
    every platform. (Measured 2026-11-10: the 316 such calls in the 108 test files that do not import graph_identity are
    all at 5b7335f6 unchanged, and its Windows V1 passed every test.)"""
    import ast

    offenders = []
    for path in sorted((REPO / "tests").rglob("*.py")):
        tree = ast.parse(path.read_bytes())
        imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in n.names} | {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        if not any("graph_identity" in m for m in imports):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "write_text" and \
                    not any(k.arg == "newline" for k in n.keywords):
                offenders.append(f"{path.relative_to(REPO).as_posix()}:{n.lineno}")
    assert offenders == [], offenders
