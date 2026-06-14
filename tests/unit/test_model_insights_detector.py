"""test_model_insights_detector.py -- Monzia Moodie
Hermetic tests for the model-insights detector over a synthetic oof_predictions.parquet (RunArtifactWriter
schema). Covers per-model metrics, leakage-suspicion + degenerate + AUROC/AUPRC-gap flags, gene-disjoint
pass/fail, MCC-not-AUROC ranking, and latest-run discovery.
"""
import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation import model_insights_detector as D


def _oof(n=400, seed=0, disjoint=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    # a STRONG model: prob correlated with label; a WEAK model: noisy; a DEGENERATE model: constant
    strong = np.clip(0.15 * rng.standard_normal(n) + 0.8 * y + 0.1, 0, 1)
    weak = np.clip(0.4 * rng.standard_normal(n) + 0.5, 0, 1)
    const = np.full(n, 0.5)
    folds = (np.arange(n) % 4)
    if disjoint:
        genes = [f"G{f}_{i}" for i, f in enumerate(folds)]      # unique per row -> disjoint across folds
    else:
        genes = [f"SHARED{i % 3}" for i in range(n)]             # same genes across folds -> overlap
    return pd.DataFrame({"variant_id": np.arange(n), "gene_symbol": genes, "fold": folds,
                         "label": y, "strong_prob": strong, "weak_prob": weak,
                         "const_prob": const, "ensemble_prob": strong})


def test_prob_columns_and_per_model_metrics():
    oof = _oof()
    cols = D.prob_columns(oof)
    assert set(cols) == {"strong_prob", "weak_prob", "const_prob", "ensemble_prob"}
    ms = {m.model: m for m in D.per_model_metrics(oof)}
    assert ms["strong"].auroc > ms["weak"].auroc          # strong discriminates better
    assert 0.0 <= ms["weak"].auroc <= 1.0
    assert ms["const"].auroc is None and "degenerate" in ms["const"].note  # constant -> degenerate


def test_degenerate_flag():
    flags = D.integrity_flags(D.per_model_metrics(_oof()))
    assert any(f.startswith("DEGENERATE_OOF[const]") for f in flags)


def test_leakage_suspicion_flag():
    # build a near-perfect model -> AUROC >= 0.99 -> leakage suspicion
    n = 300
    y = (np.arange(n) % 3 == 0).astype(int)
    perfect = y * 0.98 + 0.01                              # almost separable
    oof = pd.DataFrame({"label": y, "leaky_prob": perfect})
    flags = D.integrity_flags(D.per_model_metrics(oof))
    assert any(f.startswith("LEAKAGE_SUSPICION[leaky]") for f in flags)


def test_gene_disjoint_pass_and_fail():
    ok, msg = D.gene_disjoint_check(_oof(disjoint=True))
    assert ok and "disjoint" in msg
    bad, msg2 = D.gene_disjoint_check(_oof(disjoint=False))
    assert not bad and "LEAKAGE RISK" in msg2


def test_ranking_is_by_mcc_not_auroc():
    ranked = D.rank_by_balanced(D.per_model_metrics(_oof()))
    assert ranked[0] in ("strong", "ensemble")            # strong/ensemble lead; const excluded (None mcc)
    assert "const" not in ranked


def test_analyze_bundles_everything_and_flags_violation():
    res = D.analyze(_oof(disjoint=False))
    assert res["gene_disjoint"] is False
    assert res["flags"][0].startswith("GENE_DISJOINT_VIOLATION")
    assert res["ranking_by_mcc"] and isinstance(res["metrics"], list)


def test_discover_latest_run(tmp_path):
    assert D.discover_latest_run(str(tmp_path)) is None    # empty -> None
    rd = tmp_path / "run99" / "full"
    rd.mkdir(parents=True)
    _oof().to_parquet(rd / "oof_predictions.parquet")
    found = D.discover_latest_run(str(tmp_path))
    assert found is not None and found.name == "full"
