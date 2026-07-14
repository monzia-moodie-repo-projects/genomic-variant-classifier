"""A feature declared REAL must actually exist in the data. Silent-zeros must not train.

Created 2026-07-13 (roadmap 6.21).

WHAT THIS PROTECTS
------------------
Connectors do not crash when their source file is missing. They return ZEROS. From omim.py:105:

    gene_table = self._get_gene_table()
    result = df.copy()
    if gene_table.empty:
        result["omim_n_diseases"]            = DEFAULT_N_DISEASES   # 0
        result["omim_n_diseases_molecular"]  = DEFAULT_N_DISEASES   # 0
        result["omim_is_autosomal_dominant"] = DEFAULT_IS_AD        # 0
        return result

No log. No warning. No raise. The column arrives full of zeros and TRAINS. And in
variant_ensemble's own feature builder:

    feats["omim_n_diseases"] = df.get("omim_n_diseases", pd.Series([0] * len(df), ...))

`.get()` with a zero default does the same thing one layer up.

IT HAS ALREADY HAPPENED, AT SCALE. Run 15 trained, evaluated, and published with **36 of its
78 features CONSTANT ZERO** -- 46% of the feature space, across 1,038,974 variants. Whole
sources were silently stubbed: GTEx (6 features), 1000 Genomes (5), FinnGen (3),
AlphaFold/protein structure (4), splice/MaxEntScan (4), UniProt (2), OMIM (2), ESM-2, EVE,
dbSNP, PhyloP, ClinGen, codon_position, gene constraint.

The reported AUROC of 0.998 was produced by the 38 features that were real. Nothing in the
pipeline said a word. It surfaced only on 2026-07-13, when the drift work forced someone to
actually look at the values in the matrix -- not at the file list, not at the logs, at the
VALUES.

WHY THE LAUNCHER'S ABORT GATES ARE NOT ENOUGH
---------------------------------------------
scripts/launch_run17_*.sh now hard-aborts (exit 8) if the OMIM genemap2 file is missing, and
similarly for PhyloP, dbSNP, AlphaFold, ClinGen, EVE, UniProt, FinnGen. Right instinct, wrong
LAYER: those check that a FILE EXISTS, which is a PROXY for the feature being populated.

A present-but-empty file, a schema change, a renamed column, or a failed gene-symbol join all
sail straight through a file-existence check and still deliver a column of zeros. That is
roadmap section 7, root pattern (c): a gate that checks a proxy instead of the thing it
protects is not a gate.

    A file that exists is not a feature that varies.

This guard asserts the thing itself, against the actual data, at the moment of fit -- before a
single model is trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    EXPECTED_TABULAR_FEATURE_COUNT,
    TABULAR_FEATURES,
    EnsembleConfig,
    VariantEnsemble,
    feature_census,
    format_feature_census,
)

N_BIG = 12_000      # above EnsembleConfig.zero_variance_min_rows (10_000)
N_SMALL = 500       # below it


def _frame(n: int, dead: list[str] | None = None) -> pd.DataFrame:
    """A frame with every declared tabular feature varying, except those named in `dead`."""
    rng = np.random.default_rng(20260713)
    X = pd.DataFrame(
        {f: rng.normal(size=n) for f in TABULAR_FEATURES},
        columns=list(TABULAR_FEATURES),
    )
    for col in dead or []:
        X[col] = 0.0
    return X


def _ensemble(tmp_path, **cfg) -> VariantEnsemble:
    return VariantEnsemble(EnsembleConfig(model_dir=str(tmp_path), **cfg))


# ---------------------------------------------------------------------------
# THE GUARD MUST FAIL. Watch it fail, or it is not a guard.
# ---------------------------------------------------------------------------

def test_hgmd_is_no_longer_a_declared_feature(tmp_path):
    """HGMD was REMOVED 2026-07-13: no license, and variant-level label leakage.

    Pinned as a test because a feature that is 'obviously' gone is exactly the kind of thing
    that gets restored by a well-meaning merge. hgmd_is_disease_mutation is a near-copy of the
    ClinVar-Pathogenic target; reintroducing it as a variant-level feature would hand the
    model an answer key and wreck it on the novel variants it exists to score.
    """
    assert "hgmd_is_disease_mutation" not in TABULAR_FEATURES
    assert "hgmd_n_reports" not in TABULAR_FEATURES
    assert EXPECTED_TABULAR_FEATURE_COUNT == len(TABULAR_FEATURES) == 95


def test_a_single_dead_feature_raises(tmp_path):
    ens = _ensemble(tmp_path)
    X = _frame(N_BIG, dead=["phylop_score"])

    with pytest.raises(ValueError, match="ZERO-VARIANCE FEATURES"):
        ens._assert_no_dead_features(X)


def test_the_error_names_the_feature_and_points_at_the_fix(tmp_path):
    """A guard that fires without saying WHICH feature or WHAT to do is a guard you disable."""
    ens = _ensemble(tmp_path)
    X = _frame(N_BIG, dead=["gtex_max_tpm", "esm2_delta_norm"])

    with pytest.raises(ValueError) as exc:
        ens._assert_no_dead_features(X)

    msg = str(exc.value)
    assert "gtex_max_tpm" in msg
    assert "esm2_delta_norm" in msg
    assert "PHASE_2_FEATURES" in msg          # the honest fix
    assert "omim.py:105" in msg               # the usual root cause
    assert "Population Stability Index" in msg  # why drift can never catch it either


def test_the_run15_casualties_are_all_caught(tmp_path):
    """The columns Run 15 actually shipped dead. Regression, with names.

    36 of 78 features were constant zero in Run 15 -- 46% of the feature space -- and the
    published AUROC of 0.998 came from the 38 that were real. These are the ones still
    declared today (HGMD's two have since been removed outright).
    """
    run15_dead = [
        "phylop_score", "eve_score", "gtex_max_tpm", "gtex_is_eqtl",
        "dbsnp_af", "omim_n_diseases", "omim_is_autosomal_dominant",
        "clingen_validity_score", "alphafold_plddt", "af_1kg_afr",
        "finngen_af_fin", "esm2_delta_norm", "maxentscan_score", "codon_position",
    ]
    ens = _ensemble(tmp_path)
    X = _frame(N_BIG, dead=run15_dead)

    with pytest.raises(ValueError) as exc:
        ens._assert_no_dead_features(X)

    for col in run15_dead:
        assert col in str(exc.value), f"{col} shipped dead in Run 15 and is not named here"


def test_the_census_names_the_source_and_the_flag_to_fix(tmp_path):
    """A list of dead column names is a shrug. The operator needs the FLAG."""
    X = _frame(N_BIG, dead=["gtex_max_tpm", "af_1kg_afr", "alphafold_plddt"])
    report = format_feature_census(feature_census(X), len(X))

    assert "--gtex-path" in report
    assert "--kg" in report
    assert "--alphafold-path" in report
    assert "GTEx" in report and "1000 Genomes" in report and "AlphaFold" in report


def test_an_all_nan_column_is_dead_too(tmp_path):
    """np.nanstd of an empty slice is NaN, not 0.0 -- the check must not miss this."""
    ens = _ensemble(tmp_path)
    X = _frame(N_BIG)
    X["phylop_score"] = np.nan

    with pytest.raises(ValueError, match="phylop_score"):
        ens._assert_no_dead_features(X)


def test_a_constant_nonzero_column_is_also_dead(tmp_path):
    """It is ZERO VARIANCE that kills the feature, not the value zero."""
    ens = _ensemble(tmp_path)
    X = _frame(N_BIG)
    X["gerp_score"] = 4.2

    with pytest.raises(ValueError, match="gerp_score"):
        ens._assert_no_dead_features(X)


# ---------------------------------------------------------------------------
# THE GUARD MUST NOT FIRE WHEN IT SHOULDN'T. Negative-test in both directions.
# ---------------------------------------------------------------------------

def test_healthy_data_passes_silently(tmp_path):
    ens = _ensemble(tmp_path)
    ens._assert_no_dead_features(_frame(N_BIG))   # must not raise


def test_small_fixtures_warn_but_do_not_raise(tmp_path, caplog):
    """A constant binary flag in 500 synthetic rows is sampling, not a dead feature.

    Without this, the guard would turn every unit-test fixture in the repo red and would be
    switched off within a day -- which is how guards die.
    """
    ens = _ensemble(tmp_path)
    X = _frame(N_SMALL, dead=["gtex_is_eqtl"])

    with caplog.at_level("WARNING"):
        ens._assert_no_dead_features(X)   # must NOT raise

    assert "too" in caplog.text and "small to be conclusive" in caplog.text
    assert "gtex_is_eqtl" in caplog.text


def test_the_row_threshold_is_the_thing_that_decides(tmp_path):
    """Same dead column, same data -- only the row count differs. Below: warn. Above: raise."""
    dead = ["gtex_is_eqtl"]

    ens_lo = _ensemble(tmp_path, zero_variance_min_rows=50_000)
    ens_lo._assert_no_dead_features(_frame(N_BIG, dead=dead))   # 12k < 50k -> warn

    ens_hi = _ensemble(tmp_path, zero_variance_min_rows=1_000)
    with pytest.raises(ValueError):
        ens_hi._assert_no_dead_features(_frame(N_BIG, dead=dead))   # 12k > 1k -> raise


# ---------------------------------------------------------------------------
# THE ESCAPE HATCH MUST BE LOUD AND MUST BE RECORDED
# ---------------------------------------------------------------------------

def test_opt_out_does_not_raise_but_records_the_fact(tmp_path, caplog):
    """`allow_zero_variance_features=True` tolerates it -- it does not HIDE it.

    The dead features are written to `zero_variance_features_`, so the run's artifacts carry
    the fact that it trained on columns that do not exist. A finding in a log is a comment; a
    finding in the artifacts is evidence.
    """
    ens = _ensemble(tmp_path, allow_zero_variance_features=True)
    X = _frame(N_BIG, dead=["gtex_max_tpm", "esm2_delta_norm"])

    with caplog.at_level("ERROR"):
        ens._assert_no_dead_features(X)      # must not raise

    assert "TOLERATED BY CONFIG" in caplog.text
    assert set(ens.zero_variance_features_) == {"gtex_max_tpm", "esm2_delta_norm"}


def test_default_config_is_armed(tmp_path):
    """The guard is only worth anything if it is ON by default. Assert the default."""
    assert EnsembleConfig(model_dir=str(tmp_path)).allow_zero_variance_features is False
    assert EnsembleConfig(model_dir=str(tmp_path)).zero_variance_min_rows == 10_000
