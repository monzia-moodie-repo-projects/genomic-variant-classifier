"""test_level2_leakfree_oof.py  --  Monzia Moodie

Level-2 tests (INCIDENT_2026-06-13): the stacking OOF must use gene-disjoint
inner CV with per-fold train-only n_pathogenic_in_gene recompute, so fold-held-out
genes contribute 0 inside the OOF (no inner leakage), while gene_symbol=None
reproduces the legacy path.

Synthetic-proof reference: leaky StratifiedKFold+full-count OOF 0.7755 vs
leak-free GroupKFold+recompute 0.6633.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from genomic_variant_classifier.models.variant_ensemble import (
    VariantEnsemble,
    EnsembleConfig,
)


def _cohort(n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(n_genes):
        g = f"G{gi}"
        n = int(rng.integers(8, 20))
        p = rng.uniform(0.2, 0.8)
        for _ in range(n):
            lab = int(rng.random() < p)
            rows.append((g, lab, rng.normal(lab * 0.5, 1.0)))
    df = pd.DataFrame(rows, columns=["gene", "y", "sig"]).sample(
        frac=1, random_state=1).reset_index(drop=True)
    full = df[df.y == 1].groupby("gene").size()
    X = pd.DataFrame({
        "n_pathogenic_in_gene": df["gene"].map(full).fillna(0).astype(int),
        "gene_has_known_disease": (df["gene"].map(full).fillna(0) > 0).astype(int),
        "sig": df["sig"],
    })
    return X, df["y"].to_numpy(), df["gene"]


def test_leakfree_oof_zeros_unseen_genes():
    """The core leak-free property: every fold-val row sees n_pathogenic_in_gene=0
    (its gene is held out of that fold's train under GroupKFold)."""
    X, y, gene = _cohort()
    cv = list(GroupKFold(n_splits=3).split(X, y, groups=gene))
    ens = VariantEnsemble.__new__(VariantEnsemble)  # bypass heavy __init__
    seen = []

    class Recorder(LogisticRegression):
        def predict_proba(self, Xin):
            seen.append(np.asarray(Xin)[:, 0].copy())  # col 0 = n_pathogenic_in_gene
            return super().predict_proba(Xin)

    oof = ens._leakfree_oof("random_forest", Recorder(max_iter=300),
                            X, None, y, gene, cv)
    assert len(oof) == len(y)
    assert np.isfinite(oof).all()
    assert (np.concatenate(seen) == 0).all(), "fold-val rows must see npig=0"


def test_leakfree_oof_recomputes_gene_has_known_disease():
    """gene_has_known_disease must track the recomputed count (0 for unseen)."""
    X, y, gene = _cohort()
    cv = list(GroupKFold(n_splits=3).split(X, y, groups=gene))
    ens = VariantEnsemble.__new__(VariantEnsemble)
    seen_ghkd = []

    class Recorder(LogisticRegression):
        def predict_proba(self, Xin):
            seen_ghkd.append(np.asarray(Xin)[:, 1].copy())  # col 1 = ghkd
            return super().predict_proba(Xin)

    ens._leakfree_oof("random_forest", Recorder(max_iter=300), X, None, y, gene, cv)
    assert (np.concatenate(seen_ghkd) == 0).all(), "unseen genes -> ghkd 0"


def _light_ensemble():
    ens = VariantEnsemble(EnsembleConfig(n_folds=3))
    keep = {"logistic_regression", "random_forest"}
    for k in list(ens.base_estimators):
        if k not in keep:
            ens.base_estimators.pop(k, None)
    return ens


def test_fit_accepts_gene_symbol_and_runs():
    X, y, gene = _cohort()
    seq = pd.Series([""] * len(y))
    ens = _light_ensemble()
    ens.fit(X, seq, pd.Series(y), gene_symbol=gene)
    assert ens.trained_models_
    proba = ens.predict_proba(X, seq)
    assert proba.shape[0] == len(y)
    assert np.isfinite(proba).all()


def test_fit_backward_compatible_without_gene_symbol():
    X, y, gene = _cohort()
    seq = pd.Series([""] * len(y))
    ens = _light_ensemble()
    ens.fit(X, seq, pd.Series(y))  # gene_symbol=None -> legacy StratifiedKFold path
    assert ens.trained_models_
