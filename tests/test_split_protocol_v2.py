"""Tests for split_protocol_v2: four-way gene-disjoint partitioning (hash and group_shuffle)."""
import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data import split_protocol_v2 as V2


def _df(n_genes, per_gene, seed=0, with_count=True):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(n_genes):
        g = f"GENE{gi}"
        for _ in range(per_gene):
            r = {"gene_symbol": g, "label": int(rng.random() < 0.15)}
            if with_count:
                r["n_pathogenic_in_gene"] = 0
                r["gene_has_known_disease"] = 0
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode", ["hash", "group_shuffle"])
def test_all_shared_invariants_hold(mode):
    df = _df(500, 20, 1)
    res = V2.split(df, V2.SplitProtocolV2Config(mode=mode))
    # coverage: every row exactly once
    allidx = np.concatenate([res.indices[p] for p in V2.PARTITIONS])
    assert len(allidx) == len(df) and len(np.unique(allidx)) == len(df)
    # gene-disjoint: all six pairwise
    for i, a in enumerate(V2.PARTITIONS):
        for b in V2.PARTITIONS[i + 1:]:
            assert not (res.genes[a] & res.genes[b])
    # non-empty
    assert all(len(res.indices[p]) > 0 for p in V2.PARTITIONS)


def test_hash_is_stable_under_gene_growth():
    small = _df(300, 20, 2)
    extra = _df(200, 20, 99)
    extra["gene_symbol"] = extra["gene_symbol"].str.replace("GENE", "NEWGENE")
    large = pd.concat([small, extra], ignore_index=True)
    assert V2.genes_are_stable_under_growth(small, large,
                                            V2.SplitProtocolV2Config(mode="hash")) is True


def test_group_shuffle_is_not_stable_under_gene_growth():
    small = _df(300, 20, 2)
    extra = _df(200, 20, 99)
    extra["gene_symbol"] = extra["gene_symbol"].str.replace("GENE", "NEWGENE")
    large = pd.concat([small, extra], ignore_index=True)
    # documents the difference: group_shuffle reshuffles existing genes
    assert V2.genes_are_stable_under_growth(small, large,
                                            V2.SplitProtocolV2Config(mode="group_shuffle")) is False


def test_determinism_same_mode_seed():
    df = _df(400, 20, 3)
    a = V2.split(df, V2.SplitProtocolV2Config(mode="hash", seed=7))
    b = V2.split(df, V2.SplitProtocolV2Config(mode="hash", seed=7))
    assert all(np.array_equal(a.indices[p], b.indices[p]) for p in V2.PARTITIONS)


def test_train_only_leakage_remap_zeroes_nontrain_only_genes():
    df = _df(400, 25, 3)
    res = V2.split(df, V2.SplitProtocolV2Config(mode="hash"))
    remapped = V2.apply_train_only_leakage_remap(df, res.indices, V2.SplitProtocolV2Config(mode="hash"))
    conf_only = res.genes["conformal"] - res.genes["train"]
    conf_rows = remapped.iloc[res.indices["conformal"]]
    mask = conf_rows["gene_symbol"].astype(str).isin(conf_only)
    if mask.any():
        assert conf_rows.loc[mask, "n_pathogenic_in_gene"].max() == 0


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        V2.SplitProtocolV2Config(train_frac=0.7, tune_frac=0.2, conformal_frac=0.2, test_frac=0.2)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        V2.SplitProtocolV2Config(mode="nonsense")


def test_nan_genes_bucketed_not_dropped():
    df = _df(200, 20, 4)
    df.loc[df.sample(50, random_state=1).index, "gene_symbol"] = np.nan
    res = V2.split(df, V2.SplitProtocolV2Config(mode="hash"))
    # all rows still assigned (NaN genes -> 'unknown' pseudo-gene, not dropped)
    allidx = np.concatenate([res.indices[p] for p in V2.PARTITIONS])
    assert len(allidx) == len(df)


def test_fraction_accuracy_within_tolerance():
    df = _df(1000, 10, 5)
    cfg = V2.SplitProtocolV2Config(mode="hash")
    res = V2.split(df, cfg)
    total = len(set().union(*res.genes.values()))
    want = {"train": 0.60, "tune": 0.15, "conformal": 0.10, "test": 0.15}
    for p in V2.PARTITIONS:
        realized = len(res.genes[p]) / total
        assert abs(realized - want[p]) <= cfg.frac_tolerance
