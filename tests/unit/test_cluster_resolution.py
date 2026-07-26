"""Battery for the gene-cluster resolver.

Every guard here is proven FALSIFIABLE: for each refusal there is a paired case
that must succeed, so a resolver that refused everything would fail this file
just as loudly as one that accepted everything.

The central claim under test is that partition equivalence is namespace-free:
Ensembl gene identifiers and HUGO Gene Nomenclature Committee symbols name the
same genes with different strings, and the resolver must accept them as
interchangeable while still refusing labelings that genuinely disagree about
which rows belong together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.evaluation.capabilities import MetricStatus
from genomic_variant_classifier.evaluation.cluster_resolution import (
    FINDING_IDENTIFIER_INCOMPLETE,
    FINDING_IDENTIFIER_REQUIRED,
    FINDING_PARTITIONS_DISAGREE,
    SOURCE_CLUSTER_ID,
    SOURCE_GENE_ID,
    SOURCE_GENE_SYMBOL,
    SOURCE_VERIFIED_GENE_PARTITION,
    ClusterResolution,
    partitions_equivalent,
    resolve_gene_clusters,
)


# ---------------------------------------------------------------------------
# Reference implementation, kept deliberately slow and obvious.
# ---------------------------------------------------------------------------
def _reference_partitions_equivalent(a, b) -> bool:
    """The groupby-nunique formulation, used only to cross-check the fast path.

    This is the clearer statement of the property. The shipped implementation
    factorises instead, and the randomised test below proves the two agree, so
    the optimisation cannot drift away from the definition.
    """
    sa = pd.Series(list(a), dtype="object").reset_index(drop=True)
    sb = pd.Series(list(b), dtype="object").reset_index(drop=True)
    if len(sa) != len(sb):
        return False

    def miss(s):
        na = pd.isna(s).to_numpy()
        empty = s.map(lambda v: isinstance(v, str) and v.strip() == "").to_numpy(dtype=bool)
        return na | empty

    ma, mb = miss(sa), miss(sb)
    if not np.array_equal(ma, mb):
        return False
    keep = ~ma
    if not keep.any():
        return True
    frame = pd.DataFrame({"a": sa[keep].astype(str), "b": sb[keep].astype(str)})
    a_to_b = frame.groupby("a")["b"].nunique()
    b_to_a = frame.groupby("b")["a"].nunique()
    return bool((a_to_b == 1).all() and (b_to_a == 1).all())


# ---------------------------------------------------------------------------
# Group 1 -- partition equivalence is namespace-free
# ---------------------------------------------------------------------------
def test_different_namespaces_naming_the_same_genes_are_equivalent():
    """The case that makes raw string equality wrong."""
    gene_id = ["ENSG00000012048", "ENSG00000012048", "ENSG00000141510", "ENSG00000141510"]
    gene_symbol = ["BRCA1", "BRCA1", "TP53", "TP53"]
    assert partitions_equivalent(gene_id, gene_symbol) is True
    # and the strings are, of course, nowhere equal
    assert not any(x == y for x, y in zip(gene_id, gene_symbol))


def test_one_identifier_split_across_two_symbols_is_not_equivalent():
    assert partitions_equivalent(["ENSG1", "ENSG1"], ["BRCA1", "TP53"]) is False


def test_two_identifiers_collapsed_into_one_symbol_is_not_equivalent():
    assert partitions_equivalent(["ENSG1", "ENSG2"], ["BRCA1", "BRCA1"]) is False


def test_row_order_does_not_matter_as_long_as_both_are_reordered_together():
    a = ["ENSG1", "ENSG2", "ENSG1", "ENSG2"]
    b = ["BRCA1", "TP53", "BRCA1", "TP53"]
    assert partitions_equivalent(a, b) is True
    order = [3, 1, 0, 2]
    assert partitions_equivalent([a[i] for i in order], [b[i] for i in order]) is True


def test_reordering_only_one_side_breaks_equivalence():
    a = ["ENSG1", "ENSG1", "ENSG2", "ENSG2"]
    b = ["BRCA1", "TP53", "BRCA1", "TP53"]
    assert partitions_equivalent(a, b) is False


def test_asymmetric_missingness_is_a_disagreement():
    assert partitions_equivalent(["ENSG1", None], ["BRCA1", "BRCA1"]) is False
    assert partitions_equivalent(["ENSG1", "ENSG1"], ["BRCA1", None]) is False


def test_empty_string_counts_as_missing_not_as_a_cluster():
    assert partitions_equivalent(["ENSG1", ""], ["BRCA1", None]) is True
    assert partitions_equivalent(["ENSG1", "  "], ["BRCA1", np.nan]) is True


def test_all_missing_on_both_sides_is_vacuously_equivalent():
    assert partitions_equivalent([None, None], [np.nan, ""]) is True


def test_length_mismatch_is_not_equivalent():
    assert partitions_equivalent(["a", "b"], ["x"]) is False


def test_empty_input_is_equivalent():
    assert partitions_equivalent([], []) is True


def test_mixed_scalar_types_do_not_crash_and_group_by_identity():
    a = [1, 1, 2, 2]
    b = ["X", "X", "Y", "Y"]
    assert partitions_equivalent(a, b) is True
    assert partitions_equivalent([1, "1"], ["X", "Y"]) is True   # distinct labels
    assert partitions_equivalent([1, "1"], ["X", "X"]) is False  # 1 and "1" differ


@pytest.mark.parametrize("seed", list(range(40)))
def test_fast_path_agrees_with_reference_formulation(seed):
    """Randomised cross-check: the factorise path must match groupby-nunique.

    Without this, an O(n) optimisation could silently diverge from the property
    it claims to compute.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    n_a = int(rng.integers(1, 6))
    n_b = int(rng.integers(1, 6))
    a = [f"A{rng.integers(0, n_a)}" for _ in range(n)]
    b = [f"B{rng.integers(0, n_b)}" for _ in range(n)]
    # inject missingness on a random subset of rows, sometimes symmetric
    for i in range(n):
        if rng.random() < 0.15:
            a[i] = None
        if rng.random() < 0.15:
            b[i] = None
    assert partitions_equivalent(a, b) == _reference_partitions_equivalent(a, b)


# ---------------------------------------------------------------------------
# Group 2 -- resolver contract
# ---------------------------------------------------------------------------
def test_cluster_id_wins_outright_and_is_not_cross_checked():
    meta = pd.DataFrame({
        "cluster_id": ["c1", "c1", "c2"],
        "gene_id": ["ENSG1", "ENSG2", "ENSG2"],      # deliberately disagreeing
        "gene_symbol": ["BRCA1", "TP53", "TP53"],
    })
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.OK
    assert r.source == SOURCE_CLUSTER_ID
    assert r.partition_verified is False   # declared, not derived
    assert r.n_clusters == 2
    assert r.usable is True


def test_only_gene_id_is_used_and_recorded():
    meta = pd.DataFrame({"gene_id": ["ENSG1", "ENSG1", "ENSG2"]})
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.OK
    assert r.source == SOURCE_GENE_ID
    assert r.partition_verified is False
    assert r.n_clusters == 2


def test_only_gene_symbol_is_used_and_recorded():
    meta = pd.DataFrame({"gene_symbol": ["BRCA1", "TP53", "TP53"]})
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.OK
    assert r.source == SOURCE_GENE_SYMBOL
    assert r.partition_verified is False
    assert r.n_clusters == 2


def test_both_columns_equivalent_records_dual_provenance():
    meta = pd.DataFrame({
        "gene_id": ["ENSG1", "ENSG1", "ENSG2"],
        "gene_symbol": ["BRCA1", "BRCA1", "TP53"],
    })
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.OK
    assert r.source == SOURCE_VERIFIED_GENE_PARTITION
    assert r.source != SOURCE_CLUSTER_ID, "derived provenance must not masquerade as supplied"
    assert r.partition_verified is True
    assert r.n_clusters == 2


def test_both_columns_disagreeing_refuses_rather_than_choosing():
    meta = pd.DataFrame({
        "gene_id": ["ENSG1", "ENSG1"],
        "gene_symbol": ["BRCA1", "TP53"],
    })
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.FAILED
    assert r.finding == FINDING_PARTITIONS_DISAGREE
    assert r.values is None
    assert r.source is None
    assert r.usable is False


def test_neither_column_present_is_insufficient_support():
    meta = pd.DataFrame({"consequence": ["missense_variant"] * 3})
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.finding == FINDING_IDENTIFIER_REQUIRED
    assert r.values is None
    assert r.usable is False


def test_meta_none_is_insufficient_support():
    r = resolve_gene_clusters(None)
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.finding == FINDING_IDENTIFIER_REQUIRED
    assert r.usable is False


def test_partial_missingness_is_refused_not_pooled_or_singletonised():
    """The anti-conservative trap: unlabelled rows must not silently become
    one giant pseudo-gene, nor one singleton cluster each."""
    meta = pd.DataFrame({"gene_symbol": ["BRCA1", "BRCA1", None, "TP53"]})
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.finding == FINDING_IDENTIFIER_INCOMPLETE
    assert r.n_missing == 1
    assert r.values is None


def test_entirely_missing_column_is_identifier_required():
    meta = pd.DataFrame({"gene_symbol": [None, None, ""]})
    r = resolve_gene_clusters(meta)
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.finding == FINDING_IDENTIFIER_REQUIRED
    assert r.n_missing == 3


def test_empty_frame_with_no_columns_is_identifier_required():
    r = resolve_gene_clusters(pd.DataFrame())
    assert r.status is MetricStatus.INSUFFICIENT_SUPPORT
    assert r.finding == FINDING_IDENTIFIER_REQUIRED


# ---------------------------------------------------------------------------
# Group 3 -- the result object itself
# ---------------------------------------------------------------------------
def test_resolution_is_frozen():
    r = resolve_gene_clusters(pd.DataFrame({"gene_id": ["a", "b"]}))
    with pytest.raises(Exception):
        r.status = MetricStatus.FAILED   # type: ignore[misc]


def test_values_are_object_dtype_and_positionally_aligned():
    meta = pd.DataFrame({"gene_symbol": ["BRCA1", "TP53", "BRCA1"]},
                        index=[10, 11, 12])   # non-default index must not leak
    r = resolve_gene_clusters(meta)
    assert isinstance(r.values, np.ndarray)
    assert r.values.dtype == object
    assert list(r.values) == ["BRCA1", "TP53", "BRCA1"]
    assert r.n_rows == 3


def test_status_is_a_metricstatus_not_a_bare_string():
    r = resolve_gene_clusters(pd.DataFrame({"gene_id": ["a"]}))
    assert isinstance(r.status, MetricStatus)
    assert r.status.value == "ok"


def test_unusable_resolutions_never_carry_values():
    """One assertion covering every refusal path, so a future branch that
    returns labels alongside a refusal cannot slip through."""
    refusals = [
        resolve_gene_clusters(None),
        resolve_gene_clusters(pd.DataFrame()),
        resolve_gene_clusters(pd.DataFrame({"consequence": ["x"]})),
        resolve_gene_clusters(pd.DataFrame({"gene_symbol": [None, None]})),
        resolve_gene_clusters(pd.DataFrame({"gene_symbol": ["A", None]})),
        resolve_gene_clusters(pd.DataFrame({"gene_id": ["E1", "E1"],
                                            "gene_symbol": ["A", "B"]})),
    ]
    for r in refusals:
        assert r.values is None
        assert r.usable is False
        assert r.finding, "every refusal must name a machine-readable finding"
        assert r.status is not MetricStatus.OK


def test_every_success_path_carries_values_and_no_finding():
    successes = [
        resolve_gene_clusters(pd.DataFrame({"cluster_id": ["c", "c"]})),
        resolve_gene_clusters(pd.DataFrame({"gene_id": ["E1", "E2"]})),
        resolve_gene_clusters(pd.DataFrame({"gene_symbol": ["A", "B"]})),
        resolve_gene_clusters(pd.DataFrame({"gene_id": ["E1", "E2"],
                                            "gene_symbol": ["A", "B"]})),
    ]
    for r in successes:
        assert r.usable is True
        assert r.values is not None
        assert r.finding is None
        assert r.status is MetricStatus.OK
        assert r.n_missing == 0


def test_partition_verified_only_when_two_columns_were_compared():
    verified = resolve_gene_clusters(pd.DataFrame({
        "gene_id": ["E1", "E2"], "gene_symbol": ["A", "B"]}))
    assert verified.partition_verified is True
    assert verified.source == SOURCE_VERIFIED_GENE_PARTITION

    for meta in (pd.DataFrame({"cluster_id": ["c", "d"]}),
                 pd.DataFrame({"gene_id": ["E1", "E2"]}),
                 pd.DataFrame({"gene_symbol": ["A", "B"]})):
        r = resolve_gene_clusters(meta)
        assert r.partition_verified is False
        assert r.source != SOURCE_VERIFIED_GENE_PARTITION
