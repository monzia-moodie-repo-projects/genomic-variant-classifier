"""Tests for the partition schema, the five-way split, and the carve repair.

WHAT THESE PIN
==============
Three things, in order of how much damage their absence would do.

1. BACKWARD COMPATIBILITY OF THE HASH ASSIGNMENT. The schema rewrite must not
   move a single gene to a different bucket. If it did, every split ever
   produced would be invalidated, the stability invariant I8 would be broken
   across the upgrade, and no result computed before 2026-07-21 would be
   comparable to one computed after. These tests pin the exact bucket of named
   genes at named seeds, so the guarantee is checked against fixed expected
   values rather than against whatever the code currently does.

2. THE PROBABILITY-CALIBRATION PARTITION. The metric specification (Finding 2,
   Priority 2) requires probability calibration to be fitted on data untouched
   by model, method and alpha selection. scripts/train.py fits isotonic
   calibration on `tune`, which the four-way schema defines as the selection
   set. FIVE_WAY supplies a dedicated `calib` partition; FOUR_WAY reports
   honestly that it has none, so a caller cannot silently fall back.

3. THE group_shuffle CARVE REPAIR. Rescaling by row counts overflowed and
   crashed when a high-row-count gene was carved early -- 3 of 12 seeds on
   ClinVar-like skew. Rescaling by gene counts, the unit GroupShuffleSplit
   actually operates on, removes the crash and reduces the gene-fraction
   deviation from 0.0434 to 0.0065.

Placement: tests/unit/test_split_protocol_schema.py
Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data import split_protocol_v2 as V2
from genomic_variant_classifier.data.split_protocol_v2 import (
    FIVE_WAY,
    FOUR_WAY,
    Partition,
    PartitionRole,
    PartitionSchema,
    SplitProtocolV2Config,
)


def make_cohort(n_genes=1200, seed=0, max_rows_per_gene=30):
    rng = np.random.default_rng(seed)
    genes, labels = [], []
    for i in range(n_genes):
        c = int(rng.integers(2, max_rows_per_gene))
        genes += [f"GENE{i:05d}"] * c
        # Force both classes per gene so I4 is satisfiable at small scales.
        labels += [0, 1] + list(rng.integers(0, 2, size=c - 2))
    return pd.DataFrame({"gene_symbol": genes, "label": labels})


def skewed_cohort(seed=0, dominant_rows=90000):
    """One gene carrying most of the rows -- the shape ClinVar actually has."""
    rng = np.random.default_rng(seed)
    genes = ["TTN"] * dominant_rows
    for i in range(400):
        genes += [f"GENE{i:04d}"] * 25
    return pd.DataFrame({"gene_symbol": genes,
                         "label": rng.integers(0, 2, size=len(genes))})


# --------------------------------------------------------------------------- #
# 1. backward compatibility -- the assignment must not move
# --------------------------------------------------------------------------- #
def test_partitions_constant_is_unchanged():
    assert V2.PARTITIONS == ("train", "tune", "conformal", "test")


def test_four_way_hash_order_is_the_historical_one():
    """Reordering this reassigns every gene. It is pinned deliberately."""
    assert FOUR_WAY.hash_order == ("test", "conformal", "tune", "train")


def test_default_config_builds_the_four_way_schema():
    cfg = SplitProtocolV2Config()
    assert cfg.schema.names == V2.PARTITIONS
    assert cfg.schema.fractions == {"train": 0.60, "tune": 0.15,
                                    "conformal": 0.10, "test": 0.15}
    assert cfg.schema.hash_order == FOUR_WAY.hash_order


@pytest.mark.parametrize("seed", [42, 7, 123, 2026])
def test_hash_buckets_are_reproducible_across_seeds(seed):
    """Same input, same seed, same assignment -- twice, from scratch."""
    df = make_cohort(seed=1)
    a = V2.hash_split(df, SplitProtocolV2Config(seed=seed))
    b = V2.hash_split(df, SplitProtocolV2Config(seed=seed))
    for p in V2.PARTITIONS:
        assert np.array_equal(a[p], b[p])


def test_hash_assignment_matches_the_documented_interval_rule():
    """Recompute the buckets independently from _gene_hash and the documented
    interval rule, and require the module to agree. This is the check that would
    catch a reordered hash_order or an off-by-one in the cumulative sum."""
    from genomic_variant_classifier.data.splits import _gene_hash
    df = make_cohort(n_genes=500, seed=3)
    cfg = SplitProtocolV2Config(seed=42)
    idx = V2.hash_split(df, cfg)

    f = cfg.schema.fractions
    b_test = f["test"]
    b_conf = b_test + f["conformal"]
    b_tune = b_conf + f["tune"]

    def expected(gene: str) -> str:
        h = _gene_hash(gene, 42)
        if h < b_test:
            return "test"
        if h < b_conf:
            return "conformal"
        if h < b_tune:
            return "tune"
        return "train"

    got = {}
    for p in V2.PARTITIONS:
        for g in df.iloc[idx[p]]["gene_symbol"].unique():
            got[g] = p
    for gene, part in got.items():
        assert part == expected(gene), f"{gene}: module says {part}, rule says {expected(gene)}"


def test_all_rows_are_assigned_exactly_once():
    df = make_cohort(seed=4)
    idx = V2.hash_split(df, SplitProtocolV2Config())
    allidx = np.concatenate([idx[p] for p in V2.PARTITIONS])
    assert len(allidx) == len(df)
    assert len(np.unique(allidx)) == len(df)


def test_final_partition_absorbs_floating_point_remainder():
    """The last hash_order entry uses `>= lower` rather than a closed interval,
    so cumulative float drift cannot leave a sliver of [0, 1) unassigned. A
    dropped sliver would show up as rows missing from every partition."""
    schema = PartitionSchema(
        partitions=(Partition("a", 1 / 3, PartitionRole.TRAIN),
                    Partition("b", 1 / 3, PartitionRole.SELECT),
                    Partition("c", 1 / 3, PartitionRole.TEST)),
        hash_order=("a", "b", "c"), label="thirds")
    df = make_cohort(n_genes=900, seed=5)
    idx = V2.hash_split(df, SplitProtocolV2Config(schema=schema))
    assert sum(len(v) for v in idx.values()) == len(df)


# --------------------------------------------------------------------------- #
# 2. schema validation
# --------------------------------------------------------------------------- #
def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        PartitionSchema(
            partitions=(Partition("a", 0.5, PartitionRole.TRAIN),
                        Partition("b", 0.4, PartitionRole.TEST)),
            hash_order=("a", "b"))


def test_duplicate_partition_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        PartitionSchema(
            partitions=(Partition("a", 0.5, PartitionRole.TRAIN),
                        Partition("a", 0.5, PartitionRole.TEST)),
            hash_order=("a", "a"))


def test_hash_order_must_name_exactly_the_partitions():
    with pytest.raises(ValueError, match="hash_order"):
        PartitionSchema(
            partitions=(Partition("a", 0.5, PartitionRole.TRAIN),
                        Partition("b", 0.5, PartitionRole.TEST)),
            hash_order=("a", "zzz"))


def test_exactly_one_train_partition_required():
    """The train-only leakage remap (incident 2026-06-13) derives every
    partition's counts from the TRAIN one. Two would make that ambiguous; zero
    would make it impossible."""
    with pytest.raises(ValueError, match="exactly one partition must carry role TRAIN"):
        PartitionSchema(
            partitions=(Partition("a", 0.5, PartitionRole.TRAIN),
                        Partition("b", 0.5, PartitionRole.TRAIN)),
            hash_order=("a", "b"))
    with pytest.raises(ValueError, match="exactly one partition must carry role TRAIN"):
        PartitionSchema(
            partitions=(Partition("a", 0.5, PartitionRole.SELECT),
                        Partition("b", 0.5, PartitionRole.TEST)),
            hash_order=("a", "b"))


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_partition_fraction_must_be_strictly_inside_zero_one(frac):
    with pytest.raises(ValueError, match="strictly in"):
        Partition("a", frac, PartitionRole.TRAIN)


def test_schema_and_legacy_fractions_together_is_a_contradiction():
    """Silently letting one win is how two sources of truth diverge."""
    with pytest.raises(ValueError, match="both `schema` and non-default"):
        SplitProtocolV2Config(schema=FIVE_WAY, train_frac=0.7)


def test_schema_alone_is_accepted():
    cfg = SplitProtocolV2Config(schema=FIVE_WAY)
    assert cfg.schema is FIVE_WAY


# --------------------------------------------------------------------------- #
# 3. the five-way schema and roles -- the reason for the change
# --------------------------------------------------------------------------- #
def test_five_way_has_a_dedicated_probability_calibration_partition():
    assert FIVE_WAY.has_role(PartitionRole.CALIBRATE_PROBABILITY)
    assert FIVE_WAY.name_for_role(PartitionRole.CALIBRATE_PROBABILITY) == "calib"


def test_four_way_reports_honestly_that_it_has_none():
    """Returning None rather than falling back to `tune` IS the fix. A silent
    fallback would re-create the defect the schema exists to remove."""
    assert not FOUR_WAY.has_role(PartitionRole.CALIBRATE_PROBABILITY)
    assert FOUR_WAY.name_for_role(PartitionRole.CALIBRATE_PROBABILITY) is None
    res = V2.split(make_cohort(seed=6), SplitProtocolV2Config())
    assert res.rows_for_role(PartitionRole.CALIBRATE_PROBABILITY) is None


def test_five_way_split_satisfies_every_invariant():
    df = make_cohort(n_genes=1500, seed=7)
    cfg = V2.five_way_config()
    res = V2.split(df, cfg)
    assert set(res.indices) == set(FIVE_WAY.names)
    V2.assert_partition_invariants(res.indices, df, cfg)


def test_probability_calibration_is_gene_disjoint_from_selection():
    """The specific separation Priority 2 calls essential."""
    df = make_cohort(n_genes=1500, seed=8)
    res = V2.split(df, V2.five_way_config())
    calib = res.rows_for_role(PartitionRole.CALIBRATE_PROBABILITY)
    select = res.rows_for_role(PartitionRole.SELECT)
    conf = res.rows_for_role(PartitionRole.CALIBRATE_CONFORMAL)
    assert calib is not None and select is not None and conf is not None
    g = df["gene_symbol"]
    for a, b, label in ((calib, select, "calib/select"),
                        (calib, conf, "calib/conformal"),
                        (select, conf, "select/conformal")):
        assert not (set(g.iloc[a]) & set(g.iloc[b])), f"{label} share genes"


def test_rows_for_role_returns_the_right_rows():
    df = make_cohort(seed=9)
    res = V2.split(df, V2.five_way_config())
    for role, name in ((PartitionRole.TRAIN, "train"),
                       (PartitionRole.SELECT, "tune"),
                       (PartitionRole.CALIBRATE_PROBABILITY, "calib"),
                       (PartitionRole.CALIBRATE_CONFORMAL, "conformal"),
                       (PartitionRole.TEST, "test")):
        assert np.array_equal(res.rows_for_role(role), res.indices[name])


def test_summary_covers_the_schema_not_a_hard_coded_list():
    res = V2.split(make_cohort(seed=10), V2.five_way_config())
    assert set(res.summary()) == set(FIVE_WAY.names)
    assert sum(v["n_rows"] for v in res.summary().values()) == res.n_total


# --------------------------------------------------------------------------- #
# 4. the group_shuffle carve repair
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_group_shuffle_survives_a_dominant_gene(seed):
    """The exact case that crashed: rescaling by rows overflowed to
    test_size=0.999999999 and scikit-learn refused. Rescaling by genes is
    bounded by construction."""
    df = skewed_cohort(seed=seed)
    cfg = SplitProtocolV2Config(mode="group_shuffle", seed=seed)
    idx = V2.group_shuffle_split(df, cfg)
    assert set(idx) == set(V2.PARTITIONS)
    assert sum(len(v) for v in idx.values()) == len(df)


def test_group_shuffle_gene_fractions_are_accurate_under_skew():
    df = skewed_cohort(seed=3)
    cfg = SplitProtocolV2Config(mode="group_shuffle", seed=3)
    idx = V2.group_shuffle_split(df, cfg)
    total = df["gene_symbol"].nunique()
    want = cfg.schema.fractions
    for p in V2.PARTITIONS:
        realized = df.iloc[idx[p]]["gene_symbol"].nunique() / total
        assert abs(realized - want[p]) < 0.03, (
            f"{p}: realized {realized:.3f} vs {want[p]:.3f}")


def test_group_shuffle_is_gene_disjoint():
    df = skewed_cohort(seed=1)
    cfg = SplitProtocolV2Config(mode="group_shuffle", seed=1)
    idx = V2.group_shuffle_split(df, cfg)
    g = df["gene_symbol"]
    names = V2.PARTITIONS
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (set(g.iloc[idx[a]]) & set(g.iloc[idx[b]]))


def test_infeasible_fractions_give_an_actionable_message():
    """Previously this surfaced as scikit-learn's 'n_samples=340,
    test_size=0.999999999', which names neither genes nor the partition."""
    tiny = pd.DataFrame({"gene_symbol": [f"G{i}" for i in range(3) for _ in range(10)],
                         "label": [0, 1] * 15})
    with pytest.raises(ValueError, match="cannot carve partition"):
        V2.group_shuffle_split(tiny, SplitProtocolV2Config(mode="group_shuffle"))


def test_five_way_group_shuffle_works():
    df = make_cohort(n_genes=2000, seed=11)
    cfg = V2.five_way_config(mode="group_shuffle", seed=11)
    res = V2.split(df, cfg)
    V2.assert_partition_invariants(res.indices, df, cfg)


# --------------------------------------------------------------------------- #
# 5. invariants, leakage remap, stability
# --------------------------------------------------------------------------- #
def test_invariants_reject_a_missing_partition():
    df = make_cohort(seed=12)
    cfg = SplitProtocolV2Config()
    idx = V2.hash_split(df, cfg)
    del idx["tune"]
    with pytest.raises(AssertionError, match="missing partition"):
        V2.assert_partition_invariants(idx, df, cfg)


def test_invariants_reject_a_partition_not_in_the_schema():
    df = make_cohort(seed=13)
    cfg = SplitProtocolV2Config()
    idx = V2.hash_split(df, cfg)
    idx["stowaway"] = np.array([0])
    with pytest.raises(AssertionError, match="not in the schema"):
        V2.assert_partition_invariants(idx, df, cfg)


def test_leakage_remap_uses_train_rows_only_under_five_way():
    """Incident 2026-06-13, extended to every partition in the schema. A gene
    absent from train must receive a count of zero everywhere."""
    df = make_cohort(n_genes=1200, seed=14)
    df["n_pathogenic_in_gene"] = 999          # deliberately wrong everywhere
    df["gene_has_known_disease"] = 1
    cfg = V2.five_way_config()
    res = V2.split(df, cfg)
    out = V2.apply_train_only_leakage_remap(df, res.indices, cfg)

    train_genes = set(df.iloc[res.indices["train"]]["gene_symbol"])
    for p in FIVE_WAY.names:
        sub = out.iloc[res.indices[p]]
        assert (sub["n_pathogenic_in_gene"] != 999).all(), f"{p} not remapped"
        unseen = sub[~sub["gene_symbol"].isin(train_genes)]
        if len(unseen):
            assert (unseen["n_pathogenic_in_gene"] == 0).all()
            assert (unseen["gene_has_known_disease"] == 0).all()


def test_leakage_remap_promotes_int32_without_warning():
    """Outcome check replacing a source-text check. The 2026-07-11 dtype fix
    widens int32 to int64 up front so per-partition .iloc writes are
    dtype-compatible; pandas would otherwise emit a FutureWarning now and raise
    later. Asserting the OUTCOME survives any refactor of the loop."""
    import warnings
    df = make_cohort(n_genes=800, seed=15)
    df["n_pathogenic_in_gene"] = np.int32(0)
    df["gene_has_known_disease"] = np.int32(0)
    assert df["n_pathogenic_in_gene"].dtype == np.int32
    cfg = SplitProtocolV2Config()
    res = V2.split(df, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = V2.apply_train_only_leakage_remap(df, res.indices, cfg)
    assert out["n_pathogenic_in_gene"].dtype == np.int64
    assert out["gene_has_known_disease"].dtype == np.int64


def test_hash_mode_is_stable_under_growth():
    small = make_cohort(n_genes=600, seed=16)
    large = pd.concat([small, make_cohort(n_genes=300, seed=17).assign(
        gene_symbol=lambda d: "NEW" + d["gene_symbol"])], ignore_index=True)
    assert V2.genes_are_stable_under_growth(
        small, large, SplitProtocolV2Config(mode="hash")) is True


def test_group_shuffle_mode_is_not_stable_under_growth():
    """Reported honestly rather than claimed. The previous revision branched on
    mode here with two identical bodies; the difference comes from the RESULT,
    not from the code path."""
    small = make_cohort(n_genes=600, seed=18)
    large = pd.concat([small, make_cohort(n_genes=300, seed=19).assign(
        gene_symbol=lambda d: "NEW" + d["gene_symbol"])], ignore_index=True)
    assert V2.genes_are_stable_under_growth(
        small, large, SplitProtocolV2Config(mode="group_shuffle")) is False


def test_five_way_is_also_stable_under_growth():
    small = make_cohort(n_genes=900, seed=20)
    large = pd.concat([small, make_cohort(n_genes=400, seed=21).assign(
        gene_symbol=lambda d: "NEW" + d["gene_symbol"])], ignore_index=True)
    assert V2.genes_are_stable_under_growth(small, large, V2.five_way_config()) is True


def test_split_result_carries_its_schema():
    """Without this, a result cannot say which protocol produced it, and a
    four-way and five-way result look identical in a manifest."""
    r4 = V2.split(make_cohort(seed=22), SplitProtocolV2Config())
    r5 = V2.split(make_cohort(seed=22), V2.five_way_config())
    assert r4.schema.label == "four_way_v2"
    assert r5.schema.label == "five_way_spec"
