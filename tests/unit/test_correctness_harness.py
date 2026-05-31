"""
Failing-first tests for the AutoKernel-style correctness harness
(agent_layer/harness/correctness_harness.py).

Each test injects ONE instance of a real Run-11..13 failure class and asserts
the matching harness stage flags it. Correctness is gated BEFORE any AUROC is
recorded ("all stages must pass before performance is measured").

Stages:
  1 smoke       - every active base estimator imports and fits on a tiny slice
  2 config      - required estimator init attributes present (KAN test_size etc.)
  3 sanity      - fasta_seq is real (not the "A"*101 dummy); no constant preds
  4 determinism - same seed -> identical OOF
  5 zero-audit  - non-binary feature columns are not ~all-zero (silent-zero)

Ground truth (verified 2026-05-30, HEAD 25b5eaf):
  - EnsembleConfig kwargs: n_folds, random_state, calibrate, class_weight,
    n_jobs, model_dir, skip_catboost, skip_svm, skip_kan, skip_mc_dropout
    (NO skip_cnn; cnn_1d pruned via base_estimators.pop).
  - fit/evaluate/predict_proba are all (X_tab, X_seq, y).
  - base_estimators built in _build_estimators(); live keys include
    random_forest, xgboost, lightgbm, logistic_regression, gradient_boosting,
    catboost, tabular_nn, cnn_1d, kan, mc_dropout, deep_ensemble.
  - base_estimators is CLEARED during fit (Issue H) -> enumerate before fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import re

# The module under test does not exist yet -> these tests fail at import,
# which is the intended RED state before implementation.
harness = pytest.importorskip(
    "genomic_variant_classifier.agent_layer.harness.correctness_harness",
    reason="correctness_harness not implemented yet (failing-first)",
)

# Shared reference fixture + dead-connector allowlist, bound off the module
# object that importorskip already returned (single source of truth; see
# correctness_harness.build_reference_slice / KNOWN_ZERO_DEFAULT).
build_reference_slice = harness.build_reference_slice
KNOWN_ZERO_DEFAULT = harness.KNOWN_ZERO_DEFAULT


def _tiny_slice(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A minimal variant frame the harness can run engineer_features on."""
    rng = np.random.default_rng(seed)
    label = (rng.uniform(0, 1, n) < 0.5).astype(int)
    return pd.DataFrame(
        {
            "variant_id": [f"syn:{i}" for i in range(n)],
            "gene_symbol": rng.choice([f"GENE{i}" for i in range(6)], n),
            "chrom": rng.choice([str(c) for c in range(1, 23)], n),
            "pos": rng.integers(1, 1_000_000, n),
            "ref": rng.choice(list("ACGT"), n),
            "alt": rng.choice(list("ACGT"), n),
            # label leaked into a feature so models can actually learn on the slice
            "alphamissense_score": label * 0.6 + rng.uniform(0, 0.4, n),
            "fasta_seq": ["".join(rng.choice(list("ACGT"), 101)) for _ in range(n)],
            "label": label,
            # Populate the columns engineer_features consumes via df.get(...) with
            # small non-zero values, so a genuinely-complete slice passes the
            # stage-5 zero-audit. Absent columns are the silent-zero mechanism
            # itself (.get returns a default), which is exactly what stage 5 flags.
            "allele_freq": rng.uniform(1e-4, 0.5, n),
            "alphafold_plddt": rng.uniform(20, 95, n),
            "clingen_validity_score": rng.uniform(0.1, 1.0, n),
            "codon_position": rng.integers(1, 4, n),
            "dbsnp_af": rng.uniform(1e-4, 0.5, n),
            "dist_to_active_site": rng.uniform(1, 500, n),
            "dist_to_splice_site": rng.uniform(1, 500, n),
            "esm2_delta_norm": rng.uniform(0.1, 5.0, n),
            "exon_number": rng.integers(1, 30, n),
            "gnn_score": rng.uniform(0.1, 0.9, n),
            "has_uniprot_annotation": rng.integers(0, 2, n),
            "hgmd_is_disease_mutation": rng.integers(0, 2, n),
            "hgmd_n_reports": rng.integers(0, 20, n),
            "is_canonical_splice": rng.integers(0, 2, n),
            "loeuf": rng.uniform(0.05, 2.0, n),
            "lovd_variant_class": rng.integers(1, 6, n),
            "maxentscan_score": rng.uniform(-5, 12, n),
            "mis_z": rng.uniform(-3, 5, n),
            "n_pathogenic_in_gene": rng.integers(0, 50, n),
            "omim_is_autosomal_dominant": rng.integers(0, 2, n),
            "omim_n_diseases": rng.integers(0, 10, n),
            "pli_score": rng.uniform(0.0, 1.0, n),
            "secondary_structure_context": rng.integers(1, 4, n),
            "solvent_accessibility": rng.uniform(0.0, 1.0, n),
            "syn_z": rng.uniform(-3, 5, n),
        }
    )


def test_stage1_smoke_flags_estimator_that_raises_on_fit():
    """An estimator raising during fit must be reported by the smoke stage."""
    df = _tiny_slice()

    class _Exploding:
        def fit(self, *a, **k):
            raise RuntimeError("simulated LGBM-uncompiled / CUDA-not-built failure")

        def predict_proba(self, *a, **k):
            raise RuntimeError("unreachable")

    report = harness.run_correctness_harness(
        df, inject_estimators={"exploding_model": _Exploding()}
    )
    assert not report.passed
    assert any(
        "exploding_model" in f and report_stage(f) == 1
        for f in report.failures
    ), f"smoke stage did not flag the exploding estimator: {report.failures}"


def test_stage2_config_flags_kan_missing_test_size():
    """KAN active but missing the test_size init attribute must be flagged."""
    df = _tiny_slice()
    report = harness.run_correctness_harness(
        df, simulate_kan_missing_test_size=True
    )
    assert not report.passed
    assert any("test_size" in f for f in report.failures), report.failures


def test_stage3_sanity_flags_dummy_sequence():
    """fasta_seq forced to the 'A'*101 dummy must be flagged (CNN-dummy class)."""
    df = _tiny_slice()
    df["fasta_seq"] = ["A" * 101] * len(df)
    report = harness.run_correctness_harness(df)
    assert not report.passed
    assert any("dummy" in f.lower() or "fasta_seq" in f for f in report.failures), report.failures


def test_stage5_zero_audit_flags_all_zero_nonbinary_feature():
    """A non-binary feature that is ~all-zero must be flagged (silent-zero class)."""
    df = _tiny_slice()
    df["alphamissense_score"] = 0.0  # a continuous feature gone entirely zero
    report = harness.run_correctness_harness(df, zero_rate_threshold=0.95)
    assert not report.passed
    assert any("alphamissense_score" in f and "zero" in f.lower() for f in report.failures), report.failures


def test_complete_slice_only_flags_known_zero_defaults():
    """A fully-populated slice: stages 1-4 pass, and stage-5 findings are a SUBSET
    of the documented KNOWN_ZERO_DEFAULT allowlist (the dead-connector set). Any
    stage-5 finding OUTSIDE that set is a real regression; any non-stage-5 failure
    means a correctness stage broke."""
    df = build_reference_slice()
    report = harness.run_correctness_harness(df)

    non_stage5 = [f for f in report.failures if not f.startswith("[stage 5]")]
    assert not non_stage5, f"stages 1-4 must pass on a complete slice; got: {non_stage5}"

    flagged = set()
    for f in report.failures:
        m = re.search(r"feature '([^']+)'", f)
        if m:
            flagged.add(m.group(1))
    unexpected = flagged - KNOWN_ZERO_DEFAULT
    assert not unexpected, (
        f"stage 5 flagged columns outside the known dead-connector allowlist "
        f"(possible regression or new silent-zero): {sorted(unexpected)}"
    )


def report_stage(failure_msg: str) -> int:
    """Helper: failure messages are prefixed '[stage N] ...'."""
    import re
    m = re.match(r"\[stage (\d+)\]", failure_msg)
    return int(m.group(1)) if m else -1


def test_clingen_validity_score_preserves_fractional_input():
    """Regression (INCIDENT clingen int-truncation): engineer_features must NOT
    truncate fractional clingen_validity_score to int. Fractional ClinGen
    confidence (e.g. 2.5, 0.7) must survive into the engineered feature."""
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    df = pd.DataFrame(
        {
            "variant_id": ["v0", "v1", "v2", "v3"],
            "gene_symbol": ["GENE0", "GENE1", "GENE2", "GENE3"],
            "chrom": ["1", "2", "7", "X"],
            "pos": [100, 200, 300, 400],
            "ref": ["A", "C", "G", "T"],
            "alt": ["T", "G", "C", "A"],
            "consequence": [
                "missense_variant", "synonymous_variant",
                "stop_gained", "intron_variant",
            ],
            "allele_freq": [1e-4, 1e-3, 1e-2, 0.0],
            "fasta_seq": ["ACGT" * 25 + "A"] * 4,
            "clingen_validity_score": [2.5, 0.7, 3.2, 0.0],
        }
    )
    out = engineer_features(df)
    got = list(out["clingen_validity_score"])
    assert got == [2.5, 0.7, 3.2, 0.0], (
        "clingen_validity_score was truncated/altered; expected fractional values "
        f"preserved, got {got}"
    )
