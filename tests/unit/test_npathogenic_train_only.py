"""
test_npathogenic_train_only.py  --  Monzia Moodie

Regression tests for Option-A Level 1 (INCIDENT_2026-06-13).

n_pathogenic_in_gene must be recomputed from TRAIN rows only AFTER the
gene-disjoint split, with unseen val/test genes collapsing to 0, and the
derived gene_has_known_disease (= count > 0) tracking it in lockstep.

Measured leakage (probe scripts/audit_npathogenic_leakage.py on
outputs/run15_rerun_report/full/splits, 2026-06-13):
    gene-disjoint (0 shared genes); 91.0% of test rows leaked;
    lone-feature test AUROC 0.7181 corpus-wide vs 0.5000 train-only.
After the fix the test/val feature carries no leakage (unseen genes -> 0),
so a lone-feature test score is exactly chance.
"""
import numpy as np
import pandas as pd

from genomic_variant_classifier.data.real_data_prep import (
    DataPrepConfig,
    DataPrepPipeline,
)


def _synth_cohort(n_genes: int = 60, seed: int = 7) -> pd.DataFrame:
    """Gene-disjoint-able cohort with a corpus-wide (leaky) count baked in,
    exactly as enrich_gene_counts() would have produced pre-split."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(n_genes):
        g = f"GENE{gi:03d}"
        n = int(rng.integers(6, 30))
        p = rng.uniform(0.15, 0.85)  # gene-level pathogenic propensity (real signal)
        for _ in range(n):
            rows.append((g, int(rng.random() < p)))
    df = pd.DataFrame(rows, columns=["gene_symbol", "label"])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    corpus = df[df.label == 1].groupby("gene_symbol").size()
    df["n_pathogenic_in_gene"] = df["gene_symbol"].map(corpus).fillna(0).astype(int)
    df["gene_has_known_disease"] = (df["n_pathogenic_in_gene"] > 0).astype(int)
    df["dummy_feat"] = rng.normal(size=len(df))  # so X is multi-column
    return df


def _run_split(df: pd.DataFrame):
    cfg = DataPrepConfig(
        group_column="gene_symbol",
        test_fraction=0.20,
        val_fraction=0.10,
        random_state=0,
        require_both_classes=False,  # small synthetic data; skip class-balance gate
    )
    pipe = DataPrepPipeline(cfg)
    X = df[["n_pathogenic_in_gene", "gene_has_known_disease", "dummy_feat"]].copy()
    y = df["label"].reset_index(drop=True)
    groups = df["gene_symbol"].reset_index(drop=True)
    result = pipe._gene_aware_split(X, y, groups)
    return result, X, y, groups


def test_npathogenic_is_train_only_after_split():
    df = _synth_cohort()
    result, X, y, groups = _run_split(df)
    (X_train, X_test, X_val, y_train, y_test, y_val,
     train_idx, test_idx, val_idx) = result

    g_train = groups.iloc[train_idx].reset_index(drop=True)
    g_test = groups.iloc[test_idx].reset_index(drop=True)
    g_val = groups.iloc[val_idx].reset_index(drop=True)

    expected = (
        pd.Series((np.asarray(y_train) == 1).astype(int))
        .groupby(g_train.values).sum()
    )

    def _exp(genes):
        return genes.map(expected).fillna(0).astype(int).to_numpy()

    np.testing.assert_array_equal(
        X_train["n_pathogenic_in_gene"].to_numpy(), _exp(g_train))
    np.testing.assert_array_equal(
        X_test["n_pathogenic_in_gene"].to_numpy(), _exp(g_test))
    np.testing.assert_array_equal(
        X_val["n_pathogenic_in_gene"].to_numpy(), _exp(g_val))


def test_unseen_test_val_genes_collapse_to_zero():
    df = _synth_cohort()
    result, X, y, groups = _run_split(df)
    (X_train, X_test, X_val, y_train, y_test, y_val,
     train_idx, test_idx, val_idx) = result

    train_genes = set(groups.iloc[train_idx])
    test_genes = set(groups.iloc[test_idx])
    val_genes = set(groups.iloc[val_idx])

    # gene-disjointness sanity (the split contract this fix relies on)
    assert train_genes.isdisjoint(test_genes)
    assert train_genes.isdisjoint(val_genes)

    # the leakage removal: unseen val/test genes carry count 0
    assert (X_test["n_pathogenic_in_gene"] == 0).all()
    assert (X_val["n_pathogenic_in_gene"] == 0).all()


def test_gene_has_known_disease_recomputed_in_lockstep():
    df = _synth_cohort()
    result, X, y, groups = _run_split(df)
    X_train, X_test, X_val = result[0], result[1], result[2]
    for Xs in (X_train, X_test, X_val):
        expected = (Xs["n_pathogenic_in_gene"] > 0).astype(int)
        pd.testing.assert_series_equal(
            Xs["gene_has_known_disease"].astype(int),
            expected, check_names=False)


def test_train_rows_preserve_genuine_gene_biology():
    # We removed leakage, not signal: train genes keep a non-zero, non-degenerate
    # count distribution (train-gene counts are legitimate variant-level biology).
    df = _synth_cohort()
    X_train = _run_split(df)[0][0]
    assert (X_train["n_pathogenic_in_gene"] > 0).any()
    assert X_train["n_pathogenic_in_gene"].nunique() > 1
