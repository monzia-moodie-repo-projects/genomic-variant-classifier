"""Stage 1D-1b.4 (set "as"): population identity, the split audit and the fit authorities. Every probe that found a defect
in a reference package (AUDIT_1D1b4_REFERENCES_2026-11-13.md: T1, T7, E3, E4, P1, P2, P10) is a test here.
Owner ruling 68ca322f; reference packages cee6713d, 7ba10b3b, 7b277d7c. Author: Monzia Moodie."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import graph_identity as G
from genomic_variant_classifier.information_sources import population as P

PT, VI = P.Partition, P.VariantIdentity


def row(i, v, genes, p, unresolved=(), ident=VI.VRS_VERIFIED):
    return P.PopulationRow(i, v, ident, frozenset(genes), tuple(unresolved), p)


def clean():
    return [row("t1", "v1", ["HGNC:1"], PT.TRAIN), row("t2", "v2", ["HGNC:5"], PT.TRAIN),
            row("va", "v3", ["HGNC:2"], PT.VALIDATION), row("te", "v4", ["HGNC:3"], PT.TEST)]


def _hgnc():
    return G.HgncIndex({"AAA": "HGNC:1", "BBB": "HGNC:2"}, {"OLDA": frozenset({"HGNC:1"}),
                                                           "TWICE": frozenset({"HGNC:1", "HGNC:2"})}, {}, "a" * 64, 2)


# ------------------------------------------------------------------------------------------------ identity
@pytest.mark.parametrize("bad", [dict(row_id=5), dict(row_id=b"r"), dict(variant_id=" "), dict(variant_identity=True),
                                 dict(variant_identity="vrs_verified"), dict(gene_ids={"HGNC:1"}),
                                 dict(gene_ids=frozenset({"HGNC:01"})), dict(gene_ids=frozenset({"GAS8"})),
                                 dict(gene_ids=frozenset({" HGNC:1"})), dict(unresolved_gene_tokens=["LOC1"]),
                                 dict(unresolved_gene_tokens=("",)), dict(partition="train")])
def test_rows_are_strictly_typed(bad):
    """E3 (untyped identities), E4 (a caller boolean for normalization) and LABEL-TYPE-IDENTITY-1's gene strings."""
    base = dict(row_id="r", variant_id="v", variant_identity=VI.VRS_VERIFIED, gene_ids=frozenset({"HGNC:1"}),
                unresolved_gene_tokens=(), partition=PT.TRAIN)
    base.update(bad)
    with pytest.raises((TypeError, ValueError)):
        P.PopulationRow(**base)


def test_gene_fields_resolve_to_sets_and_partial_resolution_is_unknown():
    h = _hgnc()
    assert P.resolve_gene_field("AAA;BBB", h) == P.GeneField(frozenset({"HGNC:1", "HGNC:2"}), (), False)
    assert P.resolve_gene_field(" AAA | OLDA ", h) == P.GeneField(frozenset({"HGNC:1"}), (), False)
    got = P.resolve_gene_field("AAA;LOC100", h)
    assert got.gene_ids == {"HGNC:1"} and got.unresolved_tokens == ("LOC100",)
    assert not row("r", "v", got.gene_ids, PT.TRAIN, got.unresolved_tokens).gene_complete
    assert P.resolve_gene_field("TWICE", h).unresolved_tokens == ("TWICE",)      # an ambiguous previous symbol
    assert P.resolve_gene_field("", h) == P.GeneField(frozenset(), (), False)
    assert not row("r", "v", (), PT.TRAIN).gene_complete
    with pytest.raises(TypeError):
        P.resolve_gene_field(float("nan"), h)


def test_clinvar_prose_in_a_gene_field_is_flagged_and_never_a_gene():
    """GENE-FIELD-PROSE-1: 32 Run 15 rows carry ClinVar's description of a large variant instead of gene symbols."""
    got = P.resolve_gene_field("covers 26 genes, none of which curated to show dosage sensitivity", _hgnc())
    assert got.prose and not got.gene_ids
    assert got.unresolved_tokens == ("covers 26 genes", "none of which curated to show dosage sensitivity")


def test_components_link_through_genes_and_variants_and_ignore_order():
    rs = [row("a", "v1", ["HGNC:1"], PT.TRAIN), row("b", "v2", ["HGNC:1", "HGNC:2"], PT.TRAIN),
          row("c", "v3", ["HGNC:2"], PT.TRAIN), row("d", "v4", [], PT.TRAIN), row("e", "v4", ["HGNC:9"], PT.TRAIN)]
    c = P.split_components(rs)
    assert c["a"] == c["b"] == c["c"] and c["d"] == c["e"] != c["a"]
    assert dict(P.split_components(reversed(rs))) == dict(c)
    with pytest.raises(TypeError):
        c["a"] = "x"
    with pytest.raises(ValueError, match="duplicate"):
        P.split_components(rs + [rs[0]])


# ------------------------------------------------------------------------------------------------ audit
def test_clean_split_is_supported_only_for_supplied_identities():
    a = P.audit_split(clean())
    assert a["verdict"] == "supported_for_supplied_identities" and a["components_spanning_partitions"] == 0


@pytest.mark.parametrize("change,verdict", [
    (("te", "v4", ["HGNC:1"], PT.TEST), "violated"),                       # shared gene
    (("te", "v1", ["HGNC:3"], PT.TEST), "violated"),                       # shared variant
    (("va", "v3", ["HGNC:2", "HGNC:1"], PT.VALIDATION), "violated"),       # shared through a multi-gene row
    (("te", "v4", [], PT.TEST), "unknown"),                                # no resolved gene (P1)
    (("te", "v4", ["HGNC:3"], PT.TEST, ("LOC123",)), "unknown"),           # partially resolved
])
def test_audit_verdicts(change, verdict):
    rs = [r for r in clean() if r.row_id != change[0]] + [row(*change)]
    assert P.audit_split(rs)["verdict"] == verdict


def test_the_validation_test_pair_is_audited_too():
    """Measured on Run 15: validation and test share 54 genes in 53 components; a two-way audit never looks."""
    rs = clean() + [row("te2", "v9", ["HGNC:2"], PT.TEST)]
    a = P.audit_split(rs)
    assert a["verdict"] == "violated" and a["pairs"]["validation|test"]["shared_genes"] == 1
    assert a["pairs"]["train|validation"] == {"shared_genes": 0, "shared_variants": 0, "shared_components": 0}


def test_unnormalized_variants_make_a_vrs_protocol_unknown_but_not_a_declared_string_protocol():
    rs = [row(r.row_id, r.variant_id, r.gene_ids, r.partition, ident=VI.UNNORMALIZED) for r in clean()]
    assert P.audit_split(rs)["verdict"] == "unknown"
    assert P.audit_split(rs, P.SplitProtocol(require_vrs=False))["verdict"] == "supported_for_supplied_identities"


def test_an_empty_required_partition_is_not_evaluated_and_calibration_is_audited_when_present():
    a = P.audit_split([r for r in clean() if r.partition is not PT.TEST])               # P2
    assert a["verdict"] == "not_evaluated" and a["sizes"]["test"] == 0
    rs = clean() + [row("ca", "v9", ["HGNC:1"], PT.CALIBRATION)]
    a = P.audit_split(rs)
    assert a["verdict"] == "violated" and a["pairs"]["train|calibration"]["shared_genes"] == 1
    assert P.audit_split([r for r in clean() if r.partition is not PT.TEST],
                         P.SplitProtocol((PT.TRAIN, PT.VALIDATION)))["verdict"] == "supported_for_supplied_identities"
    for bad in (dict(required=(PT.VALIDATION, PT.TEST)), dict(required=(PT.TRAIN, PT.TRAIN)),
                dict(required=[PT.TRAIN]), dict(require_vrs=1)):
        with pytest.raises((TypeError, ValueError)):
            P.SplitProtocol(**bad)


# ------------------------------------------------------------------------------------------------ authorization
def _t1_rows():
    return [row("a", "v1", ["HGNC:1"], PT.TRAIN), row("b", "v2", ["HGNC:1", "HGNC:2"], PT.TRAIN),
            row("c", "v3", ["HGNC:2"], PT.TEST), row("d", "v4", ["HGNC:4"], PT.VALIDATION),
            row("z", "v5", ["HGNC:7"], PT.TRAIN), row("u", "v6", ["HGNC:8"], PT.TRAIN, ("LOC9",))]


def test_strict_refuses_a_split_that_is_not_supported():
    with pytest.raises(PermissionError, match="violated"):
        P.authorize_fit(_t1_rows(), P.FitAuthority.STRICT)
    assert P.authorize_fit(clean(), P.FitAuthority.STRICT).authorized_rows == {"t1", "t2"}


def test_conditional_authorizes_only_clean_components():
    """T1: 'a' shares no gene with a held-out row but is joined to one through 'b'; it must not be authorized. 'u' is
    partially resolved, so its gene set is unknown."""
    a = P.authorize_fit(_t1_rows(), P.FitAuthority.CONDITIONAL_COMPONENT)
    assert a.authorized_rows == {"z"}
    assert a.report["claims_permitted"] == "exploratory only"
    assert (a.report["excluded_gene_incomplete"], a.report["excluded_component_shares_heldout"]) == (1, 2)
    with pytest.raises(TypeError):
        a.report["authorized_rows"] = 99


def test_one_incomplete_training_row_does_not_disqualify_its_whole_component():
    """Measured on Run 15 (2026-11-13): excluding whole components for one incomplete row cost 34,760 rows. Only the
    incomplete row itself is excluded; it carries no label into the fit, so it links nothing."""
    rs = [row("x1", "v1", ["HGNC:10"], PT.TRAIN), row("x2", "v2", ["HGNC:10"], PT.TRAIN, ("LOC55",)),
          row("x3", "v3", ["HGNC:11", "HGNC:12"], PT.TRAIN, ("LOC56",)), row("x4", "v4", ["HGNC:12"], PT.TEST),
          row("x5", "v5", ["HGNC:11"], PT.TRAIN), row("x6", "v6", ["HGNC:13"], PT.VALIDATION),
          row("x7", "v7", ["HGNC:14"], PT.TEST, ("LOC57",))]
    a = P.authorize_fit(rs, P.FitAuthority.CONDITIONAL_COMPONENT)
    assert a.authorized_rows == {"x1", "x5"}
    assert a.report["excluded_gene_incomplete"] == 2 and a.report["heldout_gene_incomplete_rows"] == 1


def test_a_heldout_row_without_a_resolved_gene_still_links_through_its_variant():
    rs = clean() + [row("t3", "v4", ["HGNC:6"], PT.TRAIN)]          # v4 is the test row's variant
    assert P.authorize_fit(rs, P.FitAuthority.CONDITIONAL_COMPONENT).authorized_rows == {"t1", "t2"}


def test_authorization_digests_bind_authority_protocol_population_and_rows():
    s = P.authorize_fit(clean(), P.FitAuthority.STRICT)
    c = P.authorize_fit(clean(), P.FitAuthority.CONDITIONAL_COMPONENT)
    assert s.authorized_rows == c.authorized_rows and s.digest != c.digest
    more = clean() + [row("t3", "v8", ["HGNC:6"], PT.TRAIN)]
    assert P.authorize_fit(more, P.FitAuthority.STRICT).digest != s.digest
    loose = P.authorize_fit(clean(), P.FitAuthority.STRICT, P.SplitProtocol(require_vrs=False))
    assert loose.digest != s.digest
    with pytest.raises(ValueError, match="init=False"):
        dataclasses.replace(s, digest="0" * 64)                     # the digest is computed, never supplied
    with pytest.raises(ValueError, match="registered"):
        P.FitAuthorization(P.FitAuthority.STRICT, P.SplitProtocol(), "a" * 64, frozenset({"x"}), frozenset({"y"}), {})
    with pytest.raises(PermissionError, match="empty"):
        P.FitAuthorization(P.FitAuthority.STRICT, P.SplitProtocol(), "a" * 64, frozenset(), frozenset({"y"}), {})
    with pytest.raises(PermissionError, match="not evaluable"):
        P.authorize_fit([r for r in clean() if r.partition is not PT.TEST], P.FitAuthority.CONDITIONAL_COMPONENT)


# ------------------------------------------------------------------------------------------------ weights and checkpoints
def test_tempered_weights_have_mean_one_and_report_ess():
    g = ["big"] * 99 + ["small"]
    w0, w5, w1 = (P.tempered_component_weights(g, a) for a in (0, 0.5, 1.0))
    assert all(abs(x["weights"].mean() - 1) < 1e-12 for x in (w0, w5, w1))
    assert w0["max_to_min_weight"] == pytest.approx(1) and w1["max_to_min_weight"] == pytest.approx(99)
    assert w0["effective_sample_size"] > w5["effective_sample_size"] > w1["effective_sample_size"]
    assert w1["component_mass_range"][0] == pytest.approx(w1["component_mass_range"][1])
    with pytest.raises(ValueError):
        w1["weights"].setflags(write=True)
    for bad in (True, 1.5, -0.1, float("nan"), np.float64(0.5)):
        with pytest.raises(ValueError):
            P.tempered_component_weights(g, bad)


def _bimodal(seed):
    """Components that are mostly benign (2% positive) or mostly pathogenic (90%), as genes are. Measured 2026-11-13 on
    seeds 0, 1, 2 and 4: the stratified choice lands within 0.002 of the overall positive rate; plain GroupKFold folds
    miss it by 0.017 to 0.040."""
    rng = np.random.default_rng(seed)
    g, y = [], []
    for c in range(80):
        n, p = int(rng.integers(5, 80)), float(rng.choice([0.02, 0.9]))
        g += [f"c{c}"] * n
        y += list((rng.random(n) < p).astype(np.int64))
    return np.array(y, dtype=np.int64), np.array(g)


@pytest.mark.parametrize("seed", [0, 1, 2, 4])
def test_checkpoint_split_is_component_disjoint_stratified_and_deterministic(seed):
    y, g = _bimodal(seed)
    tr, va, rec = P.component_checkpoint_split(y, g, fraction=0.15, seed=3)
    assert not set(g[tr]) & set(g[va]) and rec["component_overlap"] == 0
    assert abs(rec["positive_rate_inner_validation"] - rec["positive_rate_all"]) < 0.005
    assert np.array_equal(tr, P.component_checkpoint_split(y, g, fraction=0.15, seed=3)[0])
    assert len(tr) + len(va) == len(y) and not tr.flags.writeable


@pytest.mark.parametrize("labels", [[True, False] * 20, [1.0, 0.0] * 20])
def test_checkpoint_split_refuses_bool_and_float_labels(labels):
    with pytest.raises(TypeError):
        P.component_checkpoint_split(np.array(labels), np.array([f"c{i // 2}" for i in range(40)]))


def test_checkpoint_split_refuses_too_few_components_and_bad_settings():
    with pytest.raises(ValueError, match="too few"):
        P.component_checkpoint_split(np.array([0, 1] * 10, dtype=np.int64), np.array(["a"] * 10 + ["b"] * 10))
    y, g = _bimodal(0)
    for kw in (dict(fraction=1), dict(fraction=0.6), dict(seed=1.0)):
        with pytest.raises((TypeError, ValueError)):
            P.component_checkpoint_split(y, g, **kw)
