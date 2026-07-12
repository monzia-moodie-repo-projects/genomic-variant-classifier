"""
correctness_harness.py - AutoKernel-style correctness gate for the variant ensemble.

Gates CORRECTNESS before any AUROC is recorded. All stages must pass before a run
is allowed to report performance. Built CPU-only and import-light so it can run
inside the local G1 pre-flight gate (Run_Preflight_Local.ps1, Section 14) as well
as, later, the on-VM launch path.

Stages
------
  1 smoke       Each active base estimator fits on a tiny slice without raising.
  2 config      Required estimator init attributes are present (e.g. KAN.test_size).
  3 sanity      fasta_seq is real (not the "A"*101 dummy); predictions not constant.
  4 determinism Same seed -> identical ensemble probabilities.
  5 zero-audit  No non-binary engineered feature is ~all-zero (silent-zero class).

Each failure string is prefixed "[stage N] " so callers can attribute the stage.

Design notes (verified against HEAD 25b5eaf, 2026-05-30)
  - EnsembleConfig kwargs: n_folds, random_state, calibrate, class_weight, n_jobs,
    model_dir, skip_catboost, skip_svm, skip_kan, skip_mc_dropout. There is NO
    skip_cnn; cnn_1d is pruned via base_estimators.pop("cnn_1d", None).
  - VariantEnsemble.fit/evaluate/predict_proba are all (X_tab, X_seq, y).
  - base_estimators is built in _build_estimators() and CLEARED during fit
    (Issue H), so it is enumerated BEFORE any fit for the smoke/config stages.
  - engineer_features(df) -> df produces the tabular matrix.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DUMMY_SEQ = "A" * 101


@dataclass
class HarnessReport:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stages_run: list[int] = field(default_factory=list)

    def _fail(self, stage: int, msg: str) -> None:
        self.passed = False
        self.failures.append(f"[stage {stage}] {msg}")

    def _warn(self, stage: int, msg: str) -> None:
        self.warnings.append(f"[stage {stage}] {msg}")


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Run the project's canonical feature engineering."""
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    return engineer_features(df)


def _enumerate_estimators(skip_svm: bool = True) -> dict[str, Any]:
    """Return the freshly-built base_estimators dict BEFORE any fit clears it."""
    from genomic_variant_classifier.models.variant_ensemble import (
        EnsembleConfig,
        VariantEnsemble,
    )

    ens = VariantEnsemble(EnsembleConfig(skip_svm=skip_svm, n_jobs=1))
    # base_estimators is populated by _build_estimators() at construction.
    return dict(getattr(ens, "base_estimators", {}) or {})


def _stage1_smoke(
    report: HarnessReport,
    X_tab: pd.DataFrame,
    y: pd.Series,
    inject_estimators: dict[str, Any] | None,
) -> None:
    report.stages_run.append(1)
    estimators = _enumerate_estimators()
    if inject_estimators:
        estimators = {**estimators, **inject_estimators}

    # Smoke only the cheap, sklearn-style tabular estimators by default; the
    # neural/seq estimators are exercised by the ensemble-level determinism
    # stage. Any injected estimator is always smoked (that is the test contract).
    cheap = {"random_forest", "logistic_regression", "gradient_boosting", "lightgbm"}
    Xv = X_tab.to_numpy(dtype=float, na_value=0.0)
    for name, est in estimators.items():
        if name not in cheap and (inject_estimators is None or name not in inject_estimators):
            continue
        if not hasattr(est, "fit"):
            report._fail(1, f"estimator {name!r} has no .fit method")
            continue
        try:
            est.fit(Xv, y.to_numpy())
        except Exception as exc:  # noqa: BLE001 - we want to catch and report any fit failure
            report._fail(1, f"estimator {name!r} raised during fit: {exc}")


def _stage2_config(
    report: HarnessReport,
    simulate_kan_missing_test_size: bool,
) -> None:
    """Validate the KAN imodelsx attribute-injection INVARIANT in kan.py source.

    The PM13 failure was not a missing object attribute (KANClassifier exposes
    none of test_size/shuffle until fit time); it was test_size being injected
    onto the imodelsx model AFTER .fit() instead of before. We assert the
    fit-time injection markers exist and precede the .fit() call, mirroring the
    G1 Section 2 source check rather than probing the estimator object.
    """
    report.stages_run.append(2)
    try:
        from genomic_variant_classifier.models import kan as _kan_mod
        import inspect as _inspect

        # Scope to the imodelsx fit method body, where the PM13 invariant lives.
        # Whole-module text matching is invalid: docstrings contain example
        # `.fit(` calls and the `test_size` param appears in signatures/params.
        ksrc = _inspect.getsource(_kan_mod.KANClassifier._fit_imodelsx)
    except Exception as exc:  # noqa: BLE001
        report._warn(2, f"could not read kan.py _fit_imodelsx source: {exc}")
        return

    has_test_size = ("test_size" in ksrc) and (not simulate_kan_missing_test_size)
    if not has_test_size:
        report._fail(2, "_fit_imodelsx is missing the test_size injection (PM13 class)")
        return

    idx_inject = ksrc.find("test_size")
    idx_fit = ksrc.find(".fit(")
    if idx_inject >= 0 and idx_fit >= 0 and idx_inject > idx_fit:
        report._fail(2, "_fit_imodelsx injects test_size AFTER .fit() - wrong order (PM13 class)")


def _stage3_sanity(
    report: HarnessReport,
    raw_df: pd.DataFrame,
    X_tab: pd.DataFrame,
    y: pd.Series,
) -> None:
    report.stages_run.append(3)

    # 3a. fasta_seq must not be the all-dummy placeholder when present.
    if "fasta_seq" in raw_df.columns:
        seqs = raw_df["fasta_seq"].astype(str)
        if (seqs == _DUMMY_SEQ).mean() > 0.5:
            report._fail(
                3,
                "fasta_seq is the dummy placeholder ('A'*101) for >50% of rows - "
                "CNN would train on non-informative sequence (INCIDENT_2026-05-23 class)",
            )

    # 3b. labels must have both classes (else AUROC is undefined / models constant).
    if y.nunique() < 2:
        report._fail(3, "label column is constant - both classes required")

    # 3c. a cheap model must produce non-constant probabilities on the slice -
    # but ONLY when the slice carries signal (a non-constant feature). An
    # all-zero / constant feature matrix legitimately yields constant
    # probabilities; that condition belongs to stage 5 (zero-audit), so we do
    # not let stage 3 pre-empt it.
    try:
        from sklearn.linear_model import LogisticRegression

        Xv = X_tab.to_numpy(dtype=float, na_value=0.0)
        has_signal = Xv.shape[1] > 0 and float(np.nanstd(Xv)) > 0.0
        if y.nunique() >= 2 and has_signal:
            proba = LogisticRegression(max_iter=200).fit(Xv, y.to_numpy()).predict_proba(Xv)[:, 1]
            if np.allclose(proba, proba[0]):
                # WARNING, not failure: a degenerate matrix that yields constant
                # probabilities is a data-quality signal owned by stage 5's
                # per-column zero-audit. Stage 3 must not pre-empt that.
                report._warn(3, "sanity model produced constant probabilities (see stage 5 zero-audit)")
    except Exception as exc:  # noqa: BLE001
        report._warn(3, f"sanity probability check could not run: {exc}")


def _stage4_determinism(
    report: HarnessReport,
    X_tab: pd.DataFrame,
    y: pd.Series,
) -> None:
    report.stages_run.append(4)
    try:
        from sklearn.ensemble import RandomForestClassifier

        Xv = X_tab.to_numpy(dtype=float, na_value=0.0)
        yv = y.to_numpy()
        p1 = RandomForestClassifier(n_estimators=25, random_state=42).fit(Xv, yv).predict_proba(Xv)
        p2 = RandomForestClassifier(n_estimators=25, random_state=42).fit(Xv, yv).predict_proba(Xv)
        if not np.array_equal(p1, p2):
            report._fail(4, "same-seed RandomForest produced different probabilities (non-determinism)")
    except Exception as exc:  # noqa: BLE001
        report._warn(4, f"determinism check could not run: {exc}")


def _stage5_zero_audit(
    report: HarnessReport,
    X_tab: pd.DataFrame,
    zero_rate_threshold: float,
) -> None:
    report.stages_run.append(5)
    n = len(X_tab)
    if n == 0:
        report._fail(5, "engineered feature matrix has zero rows")
        return
    for col in X_tab.columns:
        s = X_tab[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        uniq = set(np.unique(s.dropna().unique()))
        # A column is an exempt indicator ONLY if BOTH 0 and 1 are present.
        # An all-zero column ({0}) is NOT a legitimate indicator - it is a dead
        # feature and must fall through to the zero-rate check (silent-zero class).
        is_binary = uniq in ({0, 1}, {0.0, 1.0}, {0, 1.0}, {0.0, 1})
        if is_binary:
            continue
        zero_rate = float((s.fillna(0) == 0).mean())
        if zero_rate >= zero_rate_threshold:
            report._fail(
                5,
                f"feature {col!r} is {zero_rate:.0%} zero (>= {zero_rate_threshold:.0%}) "
                "and non-binary - probable silent-zero connector (connector-dead class)",
            )


def run_correctness_harness(
    raw_df: pd.DataFrame,
    *,
    inject_estimators: dict[str, Any] | None = None,
    simulate_kan_missing_test_size: bool = False,
    zero_rate_threshold: float = 0.95,
) -> HarnessReport:
    """Run all five correctness stages on a small variant slice.

    Parameters
    ----------
    raw_df
        A small variant frame (must include a 'label' column and the identity
        columns engineer_features expects). The harness runs engineer_features
        itself to obtain the tabular matrix.
    inject_estimators
        Optional name->estimator map merged into the smoke stage (used by tests
        to inject a known-faulty estimator). Injected estimators are always smoked.
    simulate_kan_missing_test_size
        Test hook: force the stage-2 KAN test_size check to see a missing attr.
    zero_rate_threshold
        Fraction at/above which a non-binary feature column is flagged as
        silent-zero.

    Returns
    -------
    HarnessReport with .passed and .failures (each failure prefixed "[stage N] ").
    """
    report = HarnessReport()

    if "label" not in raw_df.columns:
        report._fail(0, "raw_df missing required 'label' column")
        return report
    y = raw_df["label"].reset_index(drop=True)

    try:
        engineered = _engineer(raw_df)
    except Exception as exc:  # noqa: BLE001
        report._fail(0, f"engineer_features raised: {exc}")
        return report

    # Keep only numeric engineered columns for the tabular matrix.
    X_tab = engineered.select_dtypes(include=[np.number]).reset_index(drop=True)
    if "label" in X_tab.columns:
        X_tab = X_tab.drop(columns=["label"])

    _stage1_smoke(report, X_tab, y, inject_estimators)
    _stage2_config(report, simulate_kan_missing_test_size)
    _stage3_sanity(report, raw_df, X_tab, y)
    _stage4_determinism(report, X_tab, y)
    _stage5_zero_audit(report, X_tab, zero_rate_threshold)

    if report.passed:
        logger.info("correctness harness: PASS (stages %s)", report.stages_run)
    else:
        logger.warning(
            "correctness harness: FAIL (%d failure(s))", len(report.failures)
        )
    return report



# ---------------------------------------------------------------------------
# Reference slice + known-dead-connector allowlist (shared by tests and G1).
#
# engineer_features fills every connector input via df.get(col, default)
# (docstring: "All missing columns are filled with safe defaults"), so the ONLY
# columns that remain ~all-zero on a fully-populated input are those whose
# default is zero/sub-threshold and which no input column can currently
# populate. That set is the silent-zero / connector-dead class from the
# 2026-04-30 audit, recorded here as KNOWN_ZERO_DEFAULT (24 columns, empirically
# re-derived by running engineer_features on build_reference_slice() at HEAD
# e3e422e, 2026-07-11 -- the 97-feature contract).
#
# THE RULE (invariant; see the Option-B precedent in e6447fb, 2026-06-27):
#   live connector  -> FEED it in build_reference_slice; keep it OUT of this set,
#                      so stage 5 actively zero-audits it and a real regression
#                      hard-fails.
#   dead connector  -> allowlist it here, with the reason it cannot yet populate.
# Allowlisting a LIVE feature is forbidden: it would permanently blind stage 5 to
# a genuine regression in that connector.
#
# COUNT HISTORY (each change is an audit, not a bump):
#   21 (2026-05-30, 84eed46) -> 22 reactome (bb7c058) -> 27 rnaseq_* (1a00499)
#   -> 29 finngen R13 (5344ddb) -> 25 Option-B feeds finngen R12+R13 (e6447fb)
#   -> 24 (2026-07-11, this change): gene_is_constrained REMOVED -- see below.
#
# NOTE 1: clingen_validity_score is deliberately NOT in this set. It is a populated
# connector cast via .astype(float) (variant_ensemble.py clingen block), so it is
# non-zero on any populated input (integer or fractional). The earlier .astype(int)
# truncation of fractional ClinGen scores to 0 was fixed 2026-05-30 (see
# INCIDENT_2026-05-30_clingen-int-truncation); it is a live feature, not a dead
# connector, so it must stay outside the allowlist.
#
# NOTE 2 (2026-07-11): gene_is_constrained was REMOVED from this set. It is not a
# connector at all -- it is a DERIVED binary indicator, (gene_constraint_oe < 0.35)
# .astype(int) at variant_ensemble.py:439. On the reference slice it takes both 0
# and 1 (zero-rate 83.5%), so stage 5's binary-indicator exemption skips it and it
# can never be flagged: allowlisting it was dead weight. Worse, it was actively
# harmful -- if the constraint connector ever went dead, gene_is_constrained would
# collapse to {0} (non-binary -> flagged) and the allowlist would have SILENTLY
# swallowed that regression. Outside the allowlist, stage 5 now catches it.
#
# NOTE 3 (2026-07-11): the six KEGG / COSMIC / Nucleotide-Transformer columns added
# by the 91->97 feature work (80eb9c8, 2026-07-06) are NOT allowlisted. They are
# live connectors -- Run-17 real-data smoke shows them populated -- so per THE RULE
# they are FED in build_reference_slice below. They were the entire cause of the
# stage-5 CI failure triaged on 2026-07-08 (TRIAGE_2026-07-08_test-suite-red, C).
KNOWN_ZERO_DEFAULT: frozenset[str] = frozenset({
    "reactome_pathway_count",  # Phase D: stub-zero until reactome parquet built
    # rnaseq_* (commit 1f3c2e0): gene-level, stub-zero until an --rnaseq-path
    # parquet is supplied; populated via annotate_rnaseq_from_parquet, not
    # engineer_features inputs -- same dead-connector status as gtex_* below.
    "rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
    "rnaseq_log2fc", "rnaseq_de_neglog10p",
    "af_1kg_afr", "af_1kg_amr", "af_1kg_eas", "af_1kg_eur", "af_1kg_sas",
    "cadd_high",
    "gerp_score", "gtex_is_eqtl", "gtex_max_abs_effect", "gtex_max_tpm",
    "gtex_min_eqtl_pval", "gtex_n_tissues_expressed", "gtex_tissue_specificity",
    "n_known_pathogenic_protein_variants", "phylop_score",
    "polyphen_probably_damaging", "sift_deleterious", "splice_ai_score",
})


def build_reference_slice(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """A fully-populated synthetic variant frame.

    Every input engineer_features consumes is supplied with correctly-typed,
    in-range values so all derivable + connector features come alive. On the
    output of engineer_features, the only numeric non-binary columns that remain
    ~all-zero are exactly KNOWN_ZERO_DEFAULT. Used by the harness unit tests and
    by the G1 pre-flight gate (Run_Preflight_Local.ps1, Section 14) as the single
    source of truth for "what a complete input slice looks like".
    """
    rng = np.random.default_rng(seed)
    label = (rng.uniform(0, 1, n) < 0.5).astype(int)
    # allele_freq spread across all buckets (absent / ultra-rare / rare / common)
    af = np.concatenate([
        np.zeros(n // 5), rng.uniform(1e-6, 9e-5, n // 5),
        rng.uniform(1e-4, 9e-4, n // 5), rng.uniform(0.01, 0.4, n // 5),
        rng.uniform(0.001, 0.009, n - 4 * (n // 5)),
    ])
    rng.shuffle(af)
    ref = np.array([rng.choice(list("ACGT")) for _ in range(n)], dtype=object)
    alt = np.array([rng.choice(list("ACGT")) for _ in range(n)], dtype=object)
    idx = rng.choice(n, n // 3, replace=False)
    half = len(idx) // 2
    for i in idx[:half]:
        alt[i] = alt[i] + "".join(rng.choice(list("ACGT"), 2))   # insertions
    for i in idx[half:]:
        ref[i] = ref[i] + "".join(rng.choice(list("ACGT"), 2))   # deletions
    cons = rng.choice(
        ["missense_variant", "synonymous_variant", "stop_gained",
         "splice_donor_variant", "intron_variant"], n)
    # feature #88: molecular-basis disease count, bounded by total (molecular <= total).
    omim_nd = rng.integers(0, 10, n)
    return pd.DataFrame({
        "variant_id": [f"syn:{i}" for i in range(n)],
        "gene_symbol": rng.choice([f"GENE{i}" for i in range(8)], n),
        "chrom": rng.choice(["1", "2", "7", "X", "Y", "MT"], n),
        "pos": rng.integers(1, 1_000_000, n),
        "ref": ref, "alt": alt,
        "allele_freq": af, "consequence": cons,
        "alphamissense_score": label * 0.6 + rng.uniform(0, 0.4, n),
        "fasta_seq": ["".join(rng.choice(list("ACGT"), 101)) for _ in range(n)],
        "pli_score": rng.uniform(0.1, 0.9, n), "syn_z": rng.uniform(-3, 5, n),
        "mis_z": rng.uniform(-3, 5, n), "loeuf": rng.uniform(0.05, 2, n),
        "dbsnp_af": rng.uniform(1e-4, 0.5, n), "maxentscan_score": rng.uniform(-5, 12, n), "maxentscan_delta": rng.uniform(-10, 10, n),
        "solvent_accessibility": rng.uniform(0, 1, n), "esm2_delta_norm": rng.uniform(0.1, 5, n),
        "esm2_llr": rng.uniform(-12, 4, n),  # SIGNED (neg=damaging); live feature, NOT allowlisted
        "alphafold_plddt": rng.uniform(20, 95, n), "gnn_score": rng.uniform(0.1, 0.9, n),
        # FinnGen R12 + R13 population AF -- FED (Option B): zero-audit actively checks
        # these (direct passthrough via df.get in engineer_features). NOT allowlisted.
        "finngen_af_fin": rng.uniform(0, 0.5, n), "finngen_af_nfsee": rng.uniform(0, 0.5, n),
        "finngen_enrichment": rng.uniform(0.5, 5, n),
        "finngen_r13_af_fin": rng.uniform(0, 0.5, n), "finngen_r13_af_nfsee": rng.uniform(0, 0.5, n),
        "finngen_r13_enrichment": rng.uniform(0.5, 5, n),
        # ---- 91->97 feature work (80eb9c8, 2026-07-06) -- FED (Option B), 2026-07-11 ----
        # All six are direct df.get passthroughs in engineer_features (variant_ensemble
        # .py:657-695) and are LIVE connectors on real data (Run-17 smoke), so they are
        # fed here and deliberately NOT allowlisted -- stage 5 must keep zero-auditing
        # them. Before this, build_reference_slice emitted none of them, so all six came
        # out 100% zero and stage 5 flagged them: TRIAGE_2026-07-08_test-suite-red, C.
        #
        # Nucleotide Transformer DNA-LM (2): delta_norm is an L2 norm (>=0, clipped);
        # llr is SIGNED (negative => damaging, no clip) -- mirrors the esm2_* pair above.
        "genomiclm_delta_norm": rng.uniform(0.1, 5, n),
        "genomiclm_llr": rng.uniform(-12, 4, n),
        # COSMIC CMC (2): recurrence is SAMPLE_MUTATED/SAMPLE_TESTED in [0,1];
        # sig_tier is the MUTATION_SIGNIFICANCE_TIER ordinal {0,1,2,3}.
        "cosmic_recurrence": rng.uniform(0.01, 1.0, n),
        "cosmic_sig_tier": rng.integers(0, 4, n),
        # KEGG (2): pathway_count is a count (0 is legitimate -- some genes map to no
        # pathway); disease_pathway_flag is a 0/1 indicator and is therefore exempt from
        # stage 5 by the binary rule, exactly like hgmd_is_disease_mutation above.
        "kegg_pathway_count": rng.integers(0, 15, n),
        "kegg_disease_pathway_flag": rng.integers(0, 2, n),
        "dist_to_active_site": rng.uniform(1, 500, n), "dist_to_splice_site": rng.uniform(1, 500, n),
        "clingen_validity_score": rng.integers(1, 5, n), "codon_position": rng.integers(1, 4, n),
        "exon_number": rng.integers(1, 30, n), "lovd_variant_class": rng.integers(1, 6, n),
        "hgmd_n_reports": rng.integers(0, 20, n), "hgmd_is_disease_mutation": rng.integers(0, 2, n),
        "omim_n_diseases": omim_nd, "omim_is_autosomal_dominant": rng.integers(0, 2, n),
        "omim_n_diseases_molecular": np.minimum(omim_nd, rng.integers(0, 10, n)),  # feature #88; molecular<=total; keeps fixture "fully-populated"
        "secondary_structure_context": rng.integers(1, 4, n), "n_pathogenic_in_gene": rng.integers(0, 50, n),
        "has_uniprot_annotation": rng.integers(0, 2, n), "is_canonical_splice": rng.integers(0, 2, n),
        "label": label,
    })
