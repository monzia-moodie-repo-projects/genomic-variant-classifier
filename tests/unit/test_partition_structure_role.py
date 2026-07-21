"""Tests for PartitionRole.STRUCTURE and the six-way partition schema.

WHY THIS FILE EXISTS
====================
Panel Q evaluates whether a representation has coherent structure. Its
specification is explicit that this is a MODEL-SELECTION activity: it chooses a
representation, preprocessing, dimensionality, distance geometry, clustering
algorithm, cluster count, noise handling, stability thresholds and a biological
interpretation. Performing those choices on the locked test partition is
selection on test.

The specification therefore requires (docs/PANEL_Q_unsupervised_structure.md,
"Partition policy"):

  * discovery, cluster-count selection, geometry comparison and biological
    interpretation occur on a dedicated STRUCTURE partition;
  * that partition is gene-disjoint from train, tune, probability calibration,
    conformal calibration AND test;
  * test admits only a predeclared replication of a solution frozen on
    STRUCTURE.

Panel Q shipped on 2026-07-21 (5dcb932) with no partition to run on. This file
covers the partition that unblocks it.

THE DEFECT FOUND WHILE BUILDING IT
-----------------------------------
PartitionSchema.__post_init__ capped roles at one occurrence by ENUMERATING
three of them -- SELECT, CALIBRATE_CONFORMAL, TEST. That list was already stale:
commit 5b1c82b had added CALIBRATE_PROBABILITY the same morning and had not
added it here. So two partitions could both declare CALIBRATE_PROBABILITY, the
schema would be ACCEPTED, and name_for_role() would silently return whichever
was declared first. train.py would fit the probability calibrator on one
partition while a second believed it held the role -- a silent ambiguity in
precisely the role the five-way schema exists to serve.

It was found by PROBING a six-partition schema, not by reading. The repair is
not to extend the list but to remove it: every role except TRAIN is capped, so a
future enum member cannot be forgotten. test_no_role_except_train_may_repeat is
parametrized over PartitionRole itself, so it covers roles that do not exist
yet.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.split_protocol_v2 import (
    FIVE_WAY,
    FOUR_WAY,
    PARTITIONS,
    Partition,
    PartitionRole,
    PartitionSchema,
    SIX_WAY,
    SplitProtocolV2Config,
    five_way_config,
    six_way_config,
    split,
)


@pytest.fixture(scope="module")
def cohort():
    """900 genes with heavy-tailed variant counts, as ClinVar per-gene counts are."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(900):
        gene = f"GENE{i:04d}"
        for _ in range(int(rng.integers(3, 12))):
            rows.append({"gene_symbol": gene,
                         "label": int(rng.integers(0, 2)),
                         "n_pathogenic_in_gene": int(rng.integers(0, 5))})
    return pd.DataFrame(rows)


def _gene_sets(df, result, schema):
    return {p: frozenset(df.gene_symbol.iloc[result.indices[p]]) for p in schema.names}


# --------------------------------------------------------------------------- #
# 1. the role cap -- the repair that removes a list rather than extending it
# --------------------------------------------------------------------------- #
# Parametrized over PartitionRole itself, MINUS train. Filtering in the
# parametrize list rather than skipping inside the body matters: a permanent
# skip is a silent loss of coverage that shows up in every run forever, and the
# suite's skip count is a monitored signal. The enum-driven property survives --
# a role added tomorrow is covered without anyone editing this line.
@pytest.mark.parametrize(
    "role", [r for r in PartitionRole if r is not PartitionRole.TRAIN])
def test_no_role_except_train_may_repeat(role):
    """Parametrized over PartitionRole ITSELF, so a role added tomorrow is
    covered without anyone remembering to add it here. That is the whole point:
    the previous check enumerated three roles and CALIBRATE_PROBABILITY was
    already missing from the list on the day it was written."""
    with pytest.raises(ValueError, match="appears 2 times"):
        PartitionSchema(
            partitions=(Partition("train", 0.50, PartitionRole.TRAIN),
                        Partition("a", 0.25, role),
                        Partition("b", 0.25, role)),
            hash_order=("a", "b", "train"), label="probe")


def test_the_duplicate_message_names_the_offending_partitions():
    """'role X appears 2 times' without saying WHICH two sends the reader
    hunting through a schema definition."""
    with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
        PartitionSchema(
            partitions=(Partition("train", 0.50, PartitionRole.TRAIN),
                        Partition("a", 0.25, PartitionRole.CALIBRATE_PROBABILITY),
                        Partition("b", 0.25, PartitionRole.CALIBRATE_PROBABILITY)),
            hash_order=("a", "b", "train"), label="probe")


def test_calibrate_probability_specifically_may_not_repeat():
    """Named explicitly because this was the hole. A parametrized test that
    silently stopped covering this role would not be noticed."""
    with pytest.raises(ValueError, match="calibrate_probability"):
        PartitionSchema(
            partitions=(Partition("train", 0.50, PartitionRole.TRAIN),
                        Partition("c1", 0.25, PartitionRole.CALIBRATE_PROBABILITY),
                        Partition("c2", 0.25, PartitionRole.CALIBRATE_PROBABILITY)),
            hash_order=("c1", "c2", "train"), label="probe")


@pytest.mark.parametrize("n_train", [0, 2])
def test_train_is_exactly_one_not_at_most_one(n_train):
    """The train-only leakage remap (incident 2026-06-13) derives every
    partition's counts from TRAIN, so it must be unambiguous AND present."""
    parts = [Partition("other", 0.5, PartitionRole.TEST)]
    if n_train == 0:
        parts.append(Partition("more", 0.5, PartitionRole.SELECT))
    else:
        parts = [Partition("t1", 0.4, PartitionRole.TRAIN),
                 Partition("t2", 0.3, PartitionRole.TRAIN),
                 Partition("x", 0.3, PartitionRole.TEST)]
    with pytest.raises(ValueError, match="exactly one partition must carry role TRAIN"):
        PartitionSchema(partitions=tuple(parts),
                        hash_order=tuple(p.name for p in parts), label="probe")


# --------------------------------------------------------------------------- #
# 2. the role itself
# --------------------------------------------------------------------------- #
def test_structure_role_exists_and_is_distinct():
    assert PartitionRole.STRUCTURE.value == "structure"
    assert len({r.value for r in PartitionRole}) == len(list(PartitionRole))


def test_structure_role_is_json_serialisable():
    """Roles end up in run manifests; a bare Enum would serialise as a repr."""
    import json
    assert json.loads(json.dumps({"role": PartitionRole.STRUCTURE}))["role"] == "structure"


# --------------------------------------------------------------------------- #
# 3. SIX_WAY shape
# --------------------------------------------------------------------------- #
def test_six_way_has_six_partitions_summing_to_one():
    assert len(SIX_WAY.partitions) == 6
    assert sum(SIX_WAY.fractions.values()) == pytest.approx(1.0, abs=1e-9)


def test_six_way_serves_every_role():
    for role in PartitionRole:
        assert SIX_WAY.has_role(role), f"SIX_WAY has no partition for {role.value}"


def test_six_way_structure_partition_is_named_structure():
    assert SIX_WAY.name_for_role(PartitionRole.STRUCTURE) == "structure"


def test_structure_genes_come_from_train_not_from_evaluation():
    """Narrowing test, calib or conformal to fund exploratory analysis would
    trade a guarantee for a convenience. Every 0.07 comes out of train."""
    for name in ("tune", "calib", "conformal"):
        assert SIX_WAY.fractions[name] == FIVE_WAY.fractions[name], (
            f"{name} changed between FIVE_WAY and SIX_WAY; the structure "
            "partition must be funded from train only")
    assert SIX_WAY.fractions["train"] < FIVE_WAY.fractions["train"]
    taken = FIVE_WAY.fractions["train"] - SIX_WAY.fractions["train"]
    given = SIX_WAY.fractions["structure"]
    test_shift = FIVE_WAY.fractions["test"] - SIX_WAY.fractions["test"]
    assert taken == pytest.approx(given + test_shift, abs=1e-9)


def test_the_earlier_schemas_are_untouched():
    """FOUR_WAY carries the historical hash order; changing it moves every gene."""
    assert FOUR_WAY.names == ("train", "tune", "conformal", "test")
    assert FOUR_WAY.hash_order == ("test", "conformal", "tune", "train")
    assert FIVE_WAY.names == ("train", "tune", "calib", "conformal", "test")
    assert PARTITIONS == FOUR_WAY.names


@pytest.mark.parametrize("schema,expected", [
    (FOUR_WAY, None), (FIVE_WAY, None), (SIX_WAY, "structure")])
def test_only_six_way_offers_a_structure_partition(schema, expected):
    """FOUR_WAY and FIVE_WAY must return None rather than substituting a
    partition. 'This schema has no structure partition' is the honest answer,
    and Panel Q must be able to detect it and refuse."""
    assert schema.name_for_role(PartitionRole.STRUCTURE) == expected


# --------------------------------------------------------------------------- #
# 4. the specification requirement, measured on a real split
# --------------------------------------------------------------------------- #
def test_structure_is_gene_disjoint_from_every_other_partition(cohort):
    """The specification names all five by name. This checks all fifteen pairs,
    because a structure partition disjoint from test but overlapping train is
    just as invalid."""
    res = split(cohort, six_way_config(seed=42))
    gs = _gene_sets(cohort, res, SIX_WAY)
    names = SIX_WAY.names
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (gs[a] & gs[b]), f"{a} and {b} share genes"


@pytest.mark.parametrize("other", ["train", "tune", "calib", "conformal", "test"])
def test_structure_is_disjoint_from_each_named_partition(cohort, other):
    """Named individually so a failure says which requirement broke."""
    res = split(cohort, six_way_config(seed=42))
    gs = _gene_sets(cohort, res, SIX_WAY)
    assert not (gs["structure"] & gs[other])


def test_structure_partition_is_non_empty_and_substantial(cohort):
    """A structure partition too small for a stable silhouette is a partition in
    name only. Panel Q's own guard refuses below two clusters."""
    res = split(cohort, six_way_config(seed=42))
    gs = _gene_sets(cohort, res, SIX_WAY)
    assert len(gs["structure"]) >= 30
    assert len(res.indices["structure"]) > 0


@pytest.mark.parametrize("seed", [42, 7, 123, 2026])
def test_disjointness_holds_across_seeds(cohort, seed):
    res = split(cohort, six_way_config(seed=seed))
    gs = _gene_sets(cohort, res, SIX_WAY)
    assert not (gs["structure"] & gs["test"])
    assert not (gs["structure"] & gs["train"])


def test_rows_for_role_reaches_the_structure_partition(cohort):
    """Panel Q will ask by ROLE, never by partition name -- that is the
    mechanism by which it cannot accidentally run on test."""
    res = split(cohort, six_way_config(seed=42))
    rows = res.rows_for_role(PartitionRole.STRUCTURE)
    assert rows is not None and len(rows) > 0
    assert set(rows) == set(res.indices["structure"])


@pytest.mark.parametrize("cfg_factory", [SplitProtocolV2Config, five_way_config])
def test_rows_for_role_returns_none_when_the_schema_has_no_structure(cohort, cfg_factory):
    """Not an empty array -- None. An empty result would let a caller run Panel Q
    on nothing and report a vacuous pass."""
    res = split(cohort, cfg_factory(seed=42))
    assert res.rows_for_role(PartitionRole.STRUCTURE) is None


# --------------------------------------------------------------------------- #
# 5. migration is a re-split, and must not be assumed otherwise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("partition", ["test", "conformal", "calib", "tune"])
@pytest.mark.parametrize("seed", [42, 7])
def test_migration_from_five_way_preserves_every_other_partition(cohort, partition, seed):
    """THE MIGRATION GUARANTEE. SIX_WAY splits FIVE_WAY's train interval in two
    and leaves the first four intervals exactly where they were, so the locked
    test set is untouched, a fitted probability calibrator stays valid, and
    model selection need not be repeated.

    The first hash order tried put `structure` between `calib` and `tune`, which
    shifted `tune` by 0.07 and silently invalidated every model-selection
    decision for no benefit. A test asserting the opposite property failed, and
    that failure is what found it."""
    five = split(cohort, five_way_config(seed=seed))
    six = split(cohort, six_way_config(seed=seed))
    g5 = frozenset(cohort.gene_symbol.iloc[five.indices[partition]])
    g6 = frozenset(cohort.gene_symbol.iloc[six.indices[partition]])
    assert g5 == g6, (
        f"{partition} changed between FIVE_WAY and SIX_WAY. Check SIX_WAY's "
        "hash_order: structure must sit between tune and train.")


@pytest.mark.parametrize("seed", [42, 7, 123, 2026])
def test_structure_is_exactly_what_train_gave_up(cohort, seed):
    """Not merely 'train shrank' -- the genes that left train are precisely the
    structure partition, and train gains nothing in exchange."""
    five = split(cohort, five_way_config(seed=seed))
    six = split(cohort, six_way_config(seed=seed))
    t5 = frozenset(cohort.gene_symbol.iloc[five.indices["train"]])
    t6 = frozenset(cohort.gene_symbol.iloc[six.indices["train"]])
    st = frozenset(cohort.gene_symbol.iloc[six.indices["structure"]])
    assert t5 - t6 == st, "structure is not exactly the genes train released"
    assert not (t6 - t5), "train gained genes it did not have under FIVE_WAY"
    assert t6 < t5


def test_the_guarantee_does_not_extend_to_four_way(cohort):
    """FOUR_WAY has different fractions throughout. Claiming a guarantee that
    does not hold would be worse than claiming none."""
    four = split(cohort, SplitProtocolV2Config(seed=42))
    six = split(cohort, six_way_config(seed=42))
    g4 = frozenset(cohort.gene_symbol.iloc[four.indices["test"]])
    g6 = frozenset(cohort.gene_symbol.iloc[six.indices["test"]])
    assert g4 != g6


def test_partition_invariants_hold_under_six_way(cohort):
    """Coverage, disjointness, fraction accuracy, non-emptiness and both-classes
    are enforced by the shared checker; this confirms six partitions pass it."""
    res = split(cohort, six_way_config(seed=42))
    total = sum(len(res.indices[p]) for p in SIX_WAY.names)
    assert total == len(cohort)
    seen = np.concatenate([res.indices[p] for p in SIX_WAY.names])
    assert len(np.unique(seen)) == len(cohort)


# --------------------------------------------------------------------------- #
# 6. the helper
# --------------------------------------------------------------------------- #
def test_six_way_config_uses_the_six_way_schema():
    assert six_way_config().schema is SIX_WAY


def test_six_way_config_forwards_overrides():
    assert six_way_config(seed=7).seed == 7


def test_six_way_config_still_refuses_contradictory_fractions():
    """Supplying both a schema and legacy fraction fields specifies the same
    thing twice; one would silently win."""
    with pytest.raises(ValueError, match="both `schema` and non-default"):
        six_way_config(train_frac=0.9)
