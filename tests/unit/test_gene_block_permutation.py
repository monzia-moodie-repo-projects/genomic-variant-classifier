"""Gene-block permutation: naming, provenance, measured drift, and the opt-in
size-stratified mode.

WHY THIS FILE EXISTS
====================
`permute_covariate_by_group` was renamed to `permute_covariate_by_gene_block`
and `permutation_unit` changed from "group" to "gene_block" on 2026-07-21. The
old name said only that groups were involved; it did not say that whole blocks
are exchanged as units, which is the property the null rests on. "Gene-block
permutation" is the standard name for the scheme.

WHAT THE PROBE FOUND, AND WHAT IT DID NOT
-----------------------------------------
Gene-block permutation with UNEQUAL block sizes necessarily changes the
row-level marginal distribution of the covariate: swapping a 10-row gene's value
with a 4-row gene's moves 10 rows out of one level and 4 into it.

Measured on 1,149 rows across 150 genes with a PURELY gene-level covariate --
zero genes carrying more than one value -- the mean total variation distance
between observed and permuted marginals was 0.0145 (standard deviation 0.0076),
and it was ZERO in 0 of 200 permutations.

Two claims made while investigating this were WRONG and were corrected from the
same data that produced them:

  * "at zero within-gene variation the permutation is exact" -- false; drift at
    p=0.00 was 0.0171, LARGER than at p=0.05 (0.0160). Within-block
    heterogeneity is not the cause; unequal block SIZES are.
  * the relationship is monotone -- false; across p = 0.00, 0.05, 0.15, 0.35,
    0.60, 1.00 the drift ran 0.0171, 0.0160, 0.0264, 0.0458, 0.0630, 0.0172. It
    PEAKS near 0.60, because at full heterogeneity the first value is itself a
    uniform draw.

THE CONSEQUENCE IS NEGLIGIBLE, WHICH IS WHY THE DEFAULT DID NOT CHANGE.
Adjusted mutual information is chance-corrected against exactly the marginals
that drift. Unstratified against size-stratified p-values, six seeds each:

    association 0.0   0.6024 vs 0.6013   -0.0011
    association 0.2   0.0659 vs 0.0548   -0.0111
    association 0.4   0.0033 vs 0.0033    0.0000
    association 0.6   0.0033 vs 0.0033    0.0000

So the drift is MEASURED AND REPORTED per run rather than restructured away.
Size-stratified permutation is available opt-in and removes the drift exactly,
at the cost of freezing any block alone in its size stratum.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.evaluation.clustering_metrics import (
    DEFAULT_BLOCK_HETEROGENEITY_ADVISORY,
    REASON_BLOCK_COVARIATE_HETEROGENEOUS,
    count_blocks_frozen_in_size_strata,
    permutation_null_ami,
    permute_covariate_by_gene_block,
)


def _gene_level_cohort(n_genes=150, lo=4, hi=12, seed=0):
    """A covariate that IS a gene-level property: one value per gene."""
    rng = np.random.default_rng(seed)
    g, cov, clu = [], [], []
    for i in range(n_genes):
        n = int(rng.integers(lo, hi))
        v = int(rng.integers(0, 3))
        g += [i] * n
        cov += [v] * n
        clu += [int(rng.integers(0, 3))] * n
    return np.array(g), np.array(cov), np.array(clu)


def _heterogeneous_cohort(p_within, n_genes=150, seed=0):
    """A covariate that varies WITHIN genes, violating the representative rule."""
    rng = np.random.default_rng(seed)
    g, cov, clu = [], [], []
    for i in range(n_genes):
        n = int(rng.integers(4, 12))
        base = int(rng.integers(0, 3))
        for _ in range(n):
            g.append(i)
            cov.append(int(rng.integers(0, 3)) if rng.random() < p_within else base)
            clu.append(int(rng.integers(0, 3)))
    return np.array(g), np.array(cov), np.array(clu)


def _unique_size_cohort():
    """Every gene has a DIFFERENT number of rows, so size stratification freezes
    all of them. This is the cost case, and no other fixture creates it."""
    g, cov, clu = [], [], []
    rng = np.random.default_rng(5)
    for i in range(12):
        n = i + 2                       # 2, 3, 4, ... 13 -- all distinct
        v = int(rng.integers(0, 3))
        g += [i] * n
        cov += [v] * n
        clu += [int(rng.integers(0, 3))] * n
    return np.array(g), np.array(cov), np.array(clu)


# --------------------------------------------------------------------------- #
# 1. the rename
# --------------------------------------------------------------------------- #
def test_the_scheme_is_named_gene_block_not_group():
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=30, seed=0)
    assert out["permutation_scheme"] == "gene_block"
    assert out["permutation_unit"] == "gene_block"


def test_the_row_scheme_is_still_named_row():
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=None, n_permutations=30, seed=0)
    assert out["permutation_scheme"] == "row"


@pytest.mark.parametrize("key", ["n_groups", "n_groups_with_multiple_covariate_values"])
def test_the_old_keys_are_retained_for_existing_manifests(key):
    """Run manifests already carry these. Dropping them would break every
    historical record's schema for a cosmetic gain."""
    g, cov, clu = _gene_level_cohort(n_genes=40)
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0)
    assert key in out


# --------------------------------------------------------------------------- #
# 2. the representative rule, recorded rather than implied
# --------------------------------------------------------------------------- #
def test_the_representative_rule_is_recorded():
    """A reader told only that twelve blocks carried multiple values still
    cannot tell WHICH value was used."""
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0)
    assert out["representative_rule"] == "first_value_per_block"


def test_the_row_scheme_has_no_representative_rule():
    """Row permutation collapses nothing, so claiming a rule would be false."""
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=None, n_permutations=20, seed=0)
    assert out["representative_rule"] is None


def test_every_member_of_a_block_still_shares_one_value():
    """The defining property of block permutation. Testing the mechanism is more
    fundamental and less brittle than testing a downstream quantile."""
    g, cov, _ = _gene_level_cohort()
    out = permute_covariate_by_gene_block(cov, g, np.random.default_rng(1))
    for block in np.unique(g):
        assert len(np.unique(out[g == block])) == 1


def test_the_multiset_of_block_values_is_preserved():
    g, cov, _ = _gene_level_cohort()
    _, inv = np.unique(g, return_inverse=True)
    first = np.array([int(np.flatnonzero(inv == j)[0]) for j in range(inv.max() + 1)])
    out = permute_covariate_by_gene_block(cov, g, np.random.default_rng(1))
    before = sorted(cov[first].tolist())
    after = sorted(out[first].tolist())
    assert before == after


# --------------------------------------------------------------------------- #
# 3. the marginal drift, measured rather than assumed
# --------------------------------------------------------------------------- #
def test_the_marginal_drift_is_measured_and_reported():
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=100, seed=0)
    assert "marginal_tvd_mean" in out and "marginal_tvd_max" in out
    assert out["marginal_tvd_mean"] > 0.0


def test_drift_is_nonzero_even_for_a_purely_gene_level_covariate():
    """THE CORRECTION. I claimed the permutation was exact when no gene carries
    more than one value. It is not: unequal block SIZES alone force the
    row-level marginal to move. Measured 0.0145 mean over 200 permutations,
    zero in none of them."""
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=100, seed=0)
    assert out["n_blocks_with_multiple_covariate_values"] == 0
    assert out["marginal_tvd_max"] > 0.0, (
        "no permutation moved the marginal; the fixture no longer has unequal "
        "block sizes and this test has stopped testing anything")


def test_equal_block_sizes_produce_no_drift_without_stratifying():
    """The converse, which isolates SIZE as the cause: when every block has the
    same number of rows, unstratified permutation already preserves the
    marginal exactly."""
    g = np.repeat(np.arange(40), 5)
    rng = np.random.default_rng(2)
    cov = np.repeat(rng.integers(0, 3, 40), 5)
    clu = np.repeat(rng.integers(0, 3, 40), 5)
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=60, seed=0)
    assert out["marginal_tvd_max"] == 0.0


# --------------------------------------------------------------------------- #
# 4. the opt-in size-stratified mode
# --------------------------------------------------------------------------- #
def test_size_stratified_removes_the_drift_exactly():
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=100, seed=0,
                               size_strata=True)
    assert out["marginal_tvd_mean"] == 0.0
    assert out["marginal_tvd_max"] == 0.0
    assert out["permutation_scheme"] == "gene_block_size_stratified"
    assert out["size_strata"] is True


def test_size_stratified_still_permutes_whole_blocks():
    g, cov, _ = _gene_level_cohort()
    out = permute_covariate_by_gene_block(cov, g, np.random.default_rng(1),
                                          size_strata=True)
    for block in np.unique(g):
        assert len(np.unique(out[g == block])) == 1


def test_size_stratified_only_swaps_blocks_of_equal_size():
    """The mechanism behind the exact preservation: a swap must move the same
    number of rows in each direction."""
    g, cov, _ = _gene_level_cohort()
    _, inv = np.unique(g, return_inverse=True)
    sizes = np.array([int((inv == j).sum()) for j in range(inv.max() + 1)])
    first = np.array([int(np.flatnonzero(inv == j)[0]) for j in range(inv.max() + 1)])
    before = cov[first]
    after = permute_covariate_by_gene_block(
        cov, g, np.random.default_rng(3), size_strata=True)[first]
    for size in np.unique(sizes):
        idx = np.flatnonzero(sizes == size)
        assert sorted(before[idx].tolist()) == sorted(after[idx].tolist())


def test_the_default_is_unstratified():
    """Stratifying buys nothing measurable in the p-value and costs frozen
    blocks. If this default ever flips, the cost must be argued for."""
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0)
    assert out["size_strata"] is False


# --------------------------------------------------------------------------- #
# 5. the COST of stratifying, which must never be assumed small
# --------------------------------------------------------------------------- #
def test_blocks_alone_in_their_size_stratum_are_counted():
    """A block alone in its stratum can only swap with itself: its value is
    FROZEN across every permutation. My synthetic ClinVar-scale fixture froze
    zero blocks, which made the cost look free -- but sizes drawn from a bounded
    uniform are not ClinVar's heavy tail, where large genes are frequently
    unique in size. This fixture forces the condition instead of hoping for it."""
    g, cov, clu = _unique_size_cohort()
    assert count_blocks_frozen_in_size_strata(g) == 12, (
        "fixture no longer gives every block a distinct size; the cost case is "
        "not being exercised")
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=40, seed=0,
                               size_strata=True)
    assert out["n_blocks_frozen_in_stratum"] == 12


def test_a_fully_frozen_stratification_cannot_move_anything():
    """The pathological end of the cost: when every block is frozen, the 'null'
    reproduces the observed covariate exactly and carries no information."""
    g, cov, _ = _unique_size_cohort()
    for seed in range(5):
        out = permute_covariate_by_gene_block(cov, g, np.random.default_rng(seed),
                                              size_strata=True)
        assert np.array_equal(out, cov)


def test_frozen_count_is_zero_when_not_stratifying():
    """Reporting a stratification cost for an unstratified run would be false."""
    g, cov, clu = _unique_size_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0)
    assert out["n_blocks_frozen_in_stratum"] == 0


# --------------------------------------------------------------------------- #
# 6. the heterogeneity advisory -- a recorded diagnostic that now acts
# --------------------------------------------------------------------------- #
def test_a_heterogeneous_covariate_raises_the_advisory(caplog):
    """n_groups_with_multiple_covariate_values was recorded from the start and
    NOTHING acted on it -- a repository-wide search found one test asserting it
    equals zero and no other consumer. That is the recorded-but-unacted-on
    pattern this project has removed four times today."""
    g, cov, clu = _heterogeneous_cohort(0.6)
    with caplog.at_level("WARNING"):
        out = permutation_null_ami(clu, cov, groups=g, n_permutations=30, seed=0)
    assert out["fraction_blocks_with_multiple_covariate_values"] > \
        DEFAULT_BLOCK_HETEROGENEITY_ADVISORY
    assert out.get("advisory") == REASON_BLOCK_COVARIATE_HETEROGENEOUS
    assert "carry more than one covariate value" in caplog.text


def test_a_gene_level_covariate_raises_no_advisory(caplog):
    """An advisory that fires on healthy input becomes noise and stops being
    read."""
    g, cov, clu = _gene_level_cohort()
    with caplog.at_level("WARNING"):
        out = permutation_null_ami(clu, cov, groups=g, n_permutations=30, seed=0)
    assert out["fraction_blocks_with_multiple_covariate_values"] == 0.0
    assert "advisory" not in out
    assert "carry more than one covariate value" not in caplog.text


def test_the_advisory_warns_and_does_not_refuse():
    """Refusing what the previous implementation accepted is a regression, not a
    stricter standard -- the lesson from the 427-row calibration cohort earlier
    the same day."""
    g, cov, clu = _heterogeneous_cohort(1.0)
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=30, seed=0)
    assert out["status"] == "ok"
    assert np.isfinite(out["permutation_p_value"])


def test_the_advisory_threshold_is_configurable():
    g, cov, clu = _gene_level_cohort()
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0,
                               heterogeneity_advisory=-1.0)
    assert out.get("advisory") == REASON_BLOCK_COVARIATE_HETEROGENEOUS


# --------------------------------------------------------------------------- #
# 7. provenance completeness and determinism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", [
    "permutation_scheme", "permutation_unit", "representative_rule",
    "n_blocks", "n_blocks_with_multiple_covariate_values",
    "fraction_blocks_with_multiple_covariate_values",
    "marginal_tvd_mean", "marginal_tvd_max", "size_strata",
    "n_blocks_frozen_in_stratum", "p_value_sidedness",
    "n_permutations", "seed", "null_mean", "null_p95", "permutation_p_value",
])
def test_the_full_provenance_record_is_present(key):
    """Everything a reader needs to judge the null, in one place."""
    g, cov, clu = _gene_level_cohort(n_genes=40)
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0)
    assert key in out


def test_the_p_value_sidedness_is_stated():
    """One-sided upper: the question is whether the observed agreement EXCEEDS
    the null, never whether it falls short. Leaving this implicit invites a
    reader to halve or double it."""
    g, cov, clu = _gene_level_cohort(n_genes=40)
    out = permutation_null_ami(clu, cov, groups=g, n_permutations=20, seed=0)
    assert out["p_value_sidedness"] == "one_sided_upper"


@pytest.mark.parametrize("strata", [False, True])
def test_the_permutation_is_deterministic_in_its_seed(strata):
    g, cov, clu = _gene_level_cohort(n_genes=40)
    a = permutation_null_ami(clu, cov, groups=g, n_permutations=25, seed=7,
                             size_strata=strata)
    b = permutation_null_ami(clu, cov, groups=g, n_permutations=25, seed=7,
                             size_strata=strata)
    assert a == b


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="covariate has"):
        permute_covariate_by_gene_block(np.arange(10), np.arange(11),
                                        np.random.default_rng(0))
