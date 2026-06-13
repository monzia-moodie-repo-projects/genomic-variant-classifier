"""Regression guard for ClinicalEvaluator.evaluate(meta=...).

Locks the contract surfaced in Run 16: meta=None leaves the per-consequence
and per-gene breakdowns empty, a complete meta populates both, and a meta whose
consequence column is misnamed silently yields an empty consequence breakdown
while gene_errors still populate (the trap that left Run 16's eval_report bare).
Author: Monzia Moodie
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.evaluator import ClinicalEvaluator


def _synthetic(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.15, n), 0.0, 1.0)
    meta = pd.DataFrame(
        {
            "consequence": rng.choice(
                ["missense_variant", "stop_gained",
                 "synonymous_variant", "splice_donor_variant"], n),
            "gene_symbol": rng.choice([f"GENE{i}" for i in range(12)], n),
        }
    )
    return y, p, meta


def test_meta_none_yields_empty_breakdowns():
    y, p, _ = _synthetic()
    r = ClinicalEvaluator(n_bootstrap=20).evaluate(
        y_true=y, y_proba=p, meta=None, model_name="t")
    assert not r.consequence_breakdown
    assert not r.gene_errors


def test_meta_full_populates_both_breakdowns():
    y, p, meta = _synthetic()
    r = ClinicalEvaluator(n_bootstrap=20).evaluate(
        y_true=y, y_proba=p, meta=meta, model_name="t")
    assert len(r.consequence_breakdown) > 0
    assert len(r.gene_errors) > 0
    row = r.consequence_breakdown[0]
    for attr in ("consequence", "n_total", "n_pathogenic", "auroc", "auprc"):
        assert hasattr(row, attr)


def test_wrong_consequence_column_is_silently_empty():
    y, p, meta = _synthetic()
    meta_bad = meta.rename(columns={"consequence": "most_severe_consequence"})
    r = ClinicalEvaluator(n_bootstrap=20).evaluate(
        y_true=y, y_proba=p, meta=meta_bad, model_name="t")
    assert not r.consequence_breakdown          # silent-empty trap
    assert len(r.gene_errors) > 0               # gene side still works
