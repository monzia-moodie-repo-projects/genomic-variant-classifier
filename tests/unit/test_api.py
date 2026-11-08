"""
tests/unit/test_api.py
=======================
Unit tests for the Phase 3 REST API layer.

Coverage
--------
  src/genomic_variant_classifier/api/schemas.py           → VariantRequest validation, score_to_classification
  src/genomic_variant_classifier/models/variant_ensemble  → engineer_features, TABULAR_FEATURES
  src/genomic_variant_classifier/api/pipeline.py          → InferencePipeline.predict_single / predict_batch
  src/genomic_variant_classifier/api/main.py              → /health, /info, /predict, /batch endpoints

Package 1, set 1a (2026-11-07): the pipeline is built around the DECLARED convex blend (tests/unit/
_prediction_support.py), and predictions are schema-3.0.0 records -- full-precision score, one tie rule,
identities, units and statuses; the five-tier classification and "confidence" are gone from serving.

Run with:
    pytest tests/unit/test_api.py -v --tb=short

All tests run without a real trained model on disk: the ensemble and scaler
are replaced by mocks / minimal fakes.

PHASE_2_FEATURES note
---------------------
  codon_position requires VEP/HGVSc and is not in TABULAR_FEATURES.
  splice_ai_score was promoted to TABULAR_FEATURES in commit 66b2740.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from genomic_variant_classifier.models.variant_ensemble import EXPECTED_TABULAR_FEATURE_COUNT
from genomic_variant_classifier.api.attribution import (
    RuntimeModelBinding,
    served_model_roster,
)
from tests.unit._prediction_support import FixedProba, blend_pipeline


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_variant_row() -> dict:
    """Bare-minimum variant dict (only required fields)."""
    return {
        "chrom": "17",
        "pos":   41276045,
        "ref":   "C",
        "alt":   "T",
    }


@pytest.fixture
def full_variant_row() -> dict:
    """Variant dict with all optional fields populated."""
    return {
        "chrom":              "17",
        "pos":                41276045,
        "ref":                "C",
        "alt":                "T",
        "allele_freq":        0.0,
        "consequence":        "missense_variant",
        "gene_symbol":        "BRCA1",
        "cadd_phred":         32.0,
        "sift_score":         0.02,
        "polyphen2_score":    0.99,
        "revel_score":        0.88,
        "phylop_score":       6.5,
        "gerp_score":         5.2,
        "alphamissense_score":  0.92,
        "gene_constraint_oe":  0.08,
        "n_pathogenic_in_gene": 150,
        "has_uniprot_annotation": 1,
        "n_known_pathogenic_protein_variants": 12,
    }


def _make_trained_models(proba: float = 0.85) -> dict:
    """Two fitted, picklable members that always predict `proba` for the positive class."""
    return {"lightgbm": FixedProba(proba), "xgboost": FixedProba(proba)}


def _make_pipeline(proba: float = 0.85):
    from genomic_variant_classifier.api.pipeline import PipelineMetadata
    return blend_pipeline(
        _make_trained_models(proba),
        scaler   = None,
        metadata = PipelineMetadata(
            fit_input_rows = 1_197_216,
            n_features    = EXPECTED_TABULAR_FEATURE_COUNT,
            model_version = "phase2",
        ),
    )


RECORD_KEYS = {
    "prediction_schema_version", "variant_id", "source_row_key", "source_observation_id", "prediction_unit_id",
    "ordered_rows_sha256", "result_block_sha256", "combiner", "positive_class", "pathogenicity_score",
    "model_call", "tie_rule", "predictive_entropy_nats", "roster_research_view", "weighted_blend_diagnostic",
    "preprocessing_status", "calibration_status", "decision_policy", "feature_attribution", "coverage_interval",
}


# ---------------------------------------------------------------------------
# 1. Schema validation — VariantRequest
# ---------------------------------------------------------------------------

class TestVariantRequestSchema:

    def test_minimal_valid_input(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="17", pos=43094692, ref="G", alt="A")
        assert v.chrom == "17"
        assert v.ref   == "G"

    def test_chr_prefix_stripped(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="chr17", pos=1, ref="A", alt="T")
        assert v.chrom == "17"

    def test_chrM_normalised_to_MT(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="chrM", pos=1, ref="A", alt="G")
        assert v.chrom == "MT"

    def test_M_normalised_to_MT(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="M", pos=1, ref="A", alt="G")
        assert v.chrom == "MT"

    def test_alleles_uppercased(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="1", pos=100, ref="aac", alt="a")
        assert v.ref == "AAC"
        assert v.alt == "A"

    def test_pos_must_be_positive(self):
        from pydantic import ValidationError
        from genomic_variant_classifier.api.schemas import VariantRequest
        with pytest.raises(ValidationError):
            VariantRequest(chrom="1", pos=0, ref="A", alt="T")

    def test_optional_scores_default_none(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="1", pos=1, ref="A", alt="T")
        assert v.alphamissense_score is None
        assert v.allele_freq is None
        assert v.n_pathogenic_in_gene is None
        assert v.has_uniprot_annotation is None

    def test_allele_freq_bounds(self):
        from pydantic import ValidationError
        from genomic_variant_classifier.api.schemas import VariantRequest
        with pytest.raises(ValidationError):
            VariantRequest(chrom="1", pos=1, ref="A", alt="T", allele_freq=1.5)

    def test_variant_id_derived(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(chrom="17", pos=43094692, ref="G", alt="A")
        assert v._variant_id == "17:43094692:G:A"

    def test_batch_input_min_one(self):
        from pydantic import ValidationError
        from genomic_variant_classifier.api.schemas import BatchPredictRequest
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=[])

    def test_batch_input_max_size(self):
        from pydantic import ValidationError
        from genomic_variant_classifier.api.schemas import BatchPredictRequest, MAX_BATCH_SIZE
        variants = [{"chrom": "1", "pos": i + 1, "ref": "A", "alt": "T"}
                    for i in range(MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=variants)

    def test_model_dump_roundtrip(self):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(
            chrom="17", pos=43094692, ref="G", alt="A",
            consequence="missense_variant",
            alphamissense_score=0.94,
            allele_freq=0.0,
        )
        d = v.model_dump()
        assert d["chrom"] == "17"
        assert d["alphamissense_score"] == 0.94

    def test_all_optional_fields_accepted(self, full_variant_row):
        from genomic_variant_classifier.api.schemas import VariantRequest
        v = VariantRequest(**full_variant_row)
        assert v.alphamissense_score             == pytest.approx(0.92)
        assert v.gene_constraint_oe              == pytest.approx(0.08)
        assert v.n_pathogenic_in_gene            == 150
        assert v.has_uniprot_annotation          == 1
        assert v.n_known_pathogenic_protein_variants == 12


# ---------------------------------------------------------------------------
# 2. BatchPredictRequest validation
# ---------------------------------------------------------------------------

class TestBatchPredictRequest:

    def test_empty_list_rejected(self):
        from pydantic import ValidationError
        from genomic_variant_classifier.api.schemas import BatchPredictRequest
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=[])

    def test_over_limit_rejected(self):
        from pydantic import ValidationError
        from genomic_variant_classifier.api.schemas import BatchPredictRequest, MAX_BATCH_SIZE, VariantRequest
        variants = [
            VariantRequest(chrom="1", pos=i + 1, ref="A", alt="T")
            for i in range(MAX_BATCH_SIZE + 1)
        ]
        with pytest.raises(ValidationError):
            BatchPredictRequest(variants=variants)

    def test_single_variant_accepted(self):
        from genomic_variant_classifier.api.schemas import BatchPredictRequest, VariantRequest
        req = BatchPredictRequest(
            variants=[VariantRequest(chrom="1", pos=100, ref="A", alt="T")]
        )
        assert len(req.variants) == 1


# ---------------------------------------------------------------------------
# 3. score_to_classification
# ---------------------------------------------------------------------------

class TestScoreToClassificationRemoved:
    """Package 1, set 1c (2026-11-08): the five-tier helper and its import-time, current-directory-relative, silently
    defaulting threshold loader are REMOVED. Serving dropped them in 3.0.0 (gate G6.2); validate_external now reads an
    explicit, predictor-bound decision policy (evaluation/decision_policy.py)."""

    def test_the_helpers_are_gone(self):
        from genomic_variant_classifier.api import schemas
        for name in ("score_to_classification", "CLASSIFICATION_THRESHOLDS", "_load_thresholds",
                     "_DEFAULT_THRESHOLDS"):
            assert not hasattr(schemas, name), name

    def test_importing_the_schemas_reads_no_file(self, tmp_path):
        """A thresholds file in the current directory used to change what a script classified with, silently."""
        import os
        import subprocess
        import sys as _sys

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "classification_thresholds.json").write_text("{}", encoding="utf-8")
        probe = ("import builtins, io, sys\n"
                 "opened = []\n"
                 "real = io.open\n"
                 "def spy(file, *a, **k):\n"
                 "    opened.append(str(file)); return real(file, *a, **k)\n"
                 "builtins.open = io.open = spy\n"
                 "import genomic_variant_classifier.api.schemas\n"
                 "print([f for f in opened if 'threshold' in f or f.endswith('.json')])\n")
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
               "THRESHOLDS_PATH": str(tmp_path / "models" / "classification_thresholds.json")}
        r = subprocess.run([_sys.executable, "-c", probe], cwd=tmp_path, env=env, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "[]", r.stdout


# ---------------------------------------------------------------------------
# 4. engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    """Tests for engineer_features() in genomic_variant_classifier.models.variant_ensemble."""

    def test_output_columns_match_tabular_features(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features, TABULAR_FEATURES
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert list(feats.columns) == TABULAR_FEATURES

    def test_no_undeclared_missingness_in_output(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        # DUPLICATE-1A / CONSTRAINTFILL-1, 2026-08-10. This assertion read
        #     assert not feats.isnull().any().any()
        # which was true ONLY because engineer_features swept the entire matrix
        # with fillna(0.0) at line 1028. A test that forbids NaN without asking
        # WHAT FILLED IT certifies the fabrication as correctness -- and that is
        # how gene_constraint_oe survived for months as a bit-identical alias of
        # loeuf, and how a missing constraint value was recorded as "perfectly
        # tolerant of loss of function".
        #
        # The replacement is STRICTLY STRONGER: it still catches an unexpected
        # null, and it additionally distinguishes declared missingness from
        # undeclared. The permission list is imported from the same constant the
        # model preprocessor uses, so the two cannot drift.
        from genomic_variant_classifier.models.model_preprocessing import (
            DECLARED_MISSINGNESS,
        )

        undeclared = sorted(
            c for c in feats.columns
            if feats[c].isnull().any() and c not in DECLARED_MISSINGNESS
        )
        assert not undeclared, (
            "columns carry missing values with no declared missing-value "
            "policy: {}".format(undeclared)
        )

    def test_snv_detection(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats["is_snv"].iloc[0]   == 1
        assert feats["is_indel"].iloc[0] == 0

    def test_insertion_is_indel(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "ACGT"}])
        feats = engineer_features(df)
        assert feats["is_snv"].iloc[0]       == 0
        assert feats["is_indel"].iloc[0]     == 1
        assert feats["is_insertion"].iloc[0] == 1
        assert feats["is_deletion"].iloc[0]  == 0

    def test_deletion_is_indel(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "ACGT", "alt": "A"}])
        feats = engineer_features(df)
        assert feats["is_indel"].iloc[0]     == 1
        assert feats["is_deletion"].iloc[0]  == 1
        assert feats["is_insertion"].iloc[0] == 0
        assert feats["is_snv"].iloc[0]       == 0

    def test_missense_consequence(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "consequence": "missense_variant"}])
        feats = engineer_features(df)
        assert feats["is_missense"].iloc[0]          == 1
        assert feats["in_coding"].iloc[0]            == 1
        assert feats["is_loss_of_function"].iloc[0]  == 0

    def test_lof_consequence(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "consequence": "stop_gained"}])
        feats = engineer_features(df)
        assert feats["is_loss_of_function"].iloc[0] == 1
        assert feats["is_missense"].iloc[0]         == 0

    def test_frameshift_is_lof(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "AC", "alt": "A",
                             "consequence": "frameshift_variant"}])
        feats = engineer_features(df)
        assert feats["is_loss_of_function"].iloc[0] == 1

    def test_splice_site_consequence(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "consequence": "splice_donor_variant"}])
        feats = engineer_features(df)
        assert feats["is_splice"].iloc[0]            == 1
        assert feats["is_loss_of_function"].iloc[0]  == 1

    def test_alphamissense_defaults_to_05_when_missing(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats["alphamissense_score"].iloc[0] == pytest.approx(0.5)

    def test_af_raw_defaults_to_zero_when_missing(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}])
        feats = engineer_features(df)
        assert feats["af_raw"].iloc[0]        == pytest.approx(0.0)
        assert feats["af_is_absent"].iloc[0]  == 1
        assert feats["af_is_common"].iloc[0]  == 0

    def test_af_is_common_for_high_af(self):
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        df = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T",
                             "allele_freq": 0.33}])
        feats = engineer_features(df)
        assert feats["af_is_absent"].iloc[0]  == 0
        assert feats["af_is_common"].iloc[0]  == 1

    def test_codon_position_in_tabular_features(self):
        """codon_position is derived by VEPConnector (Phase 4) and is now in TABULAR_FEATURES."""
        from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, PHASE_2_FEATURES
        assert "codon_position" in TABULAR_FEATURES, (
            "codon_position should be in TABULAR_FEATURES after Phase 4 VEP promotion."
        )
        assert "codon_position" not in PHASE_2_FEATURES

    def test_batch_consistency(self):
        """engineer_features must produce identical rows regardless of batch size."""
        from genomic_variant_classifier.models.variant_ensemble import engineer_features
        single = pd.DataFrame([{"chrom": "1", "pos": 100, "ref": "A", "alt": "T",
                                 "consequence": "missense_variant", "allele_freq": 0.001}])
        batch  = pd.concat([single, single], ignore_index=True)
        feats_s = engineer_features(single)
        feats_b = engineer_features(batch)
        pd.testing.assert_frame_equal(
            feats_b.iloc[[0]].reset_index(drop=True),
            feats_s.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# 5. InferencePipeline
# ---------------------------------------------------------------------------

class TestInferencePipeline:

    def test_predict_single_returns_a_schema_3_record(self):
        pipe   = _make_pipeline(proba=0.85)
        result = pipe.predict_single({"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"})
        assert set(result.keys()) == RECORD_KEYS
        assert result["prediction_schema_version"] == "3.0.0"
        assert result["variant_id"] == "17:43094692:G:A" and result["source_row_key"] == "position:0"
        assert result["source_observation_id"] != result["prediction_unit_id"]
        for gone in ("classification", "confidence", "uncertainty_epistemic", "uncertainty_aleatoric",
                     "top_features"):
            assert gone not in result

    def test_the_score_is_full_precision_never_rounded(self):
        p = 0.123456789012345678
        result = _make_pipeline(proba=p).predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["pathogenicity_score"] == p            # was round(score, 4) == 0.1235

    @pytest.mark.parametrize("proba,call", [(0.95, 1), (0.5, 1), (0.5 - 2.0 ** -53, 0), (0.05, 0), (0.0, 0),
                                            (1.0, 1)])
    def test_one_tie_rule(self, proba, call):
        result = _make_pipeline(proba=proba).predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        assert result["model_call"] == call and result["tie_rule"] == "positive_if_p_ge_0.5"

    def test_unqualified_outputs_are_reported_unavailable(self):
        result = _make_pipeline().predict_single({"chrom": "17", "pos": 1, "ref": "G", "alt": "A"})
        for key in ("decision_policy", "feature_attribution", "coverage_interval", "weighted_blend_diagnostic"):
            assert result[key]["status"] == "unavailable" and result[key]["reason"]
        assert result["calibration_status"] == "not_qualified_by_this_record"
        view = result["roster_research_view"]
        assert view["between_member_spread_variance"] == {
            "value": 0.0, "unit": "probability_squared", "status": "defined", "reason": None}
        assert view["disagreement_status"] == "exact_agreement"

    def test_predict_batch_length(self):
        pipe    = _make_pipeline(proba=0.9)
        results = pipe.predict_batch([
            {"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"},
            {"chrom": "1",  "pos": 925952,   "ref": "G", "alt": "T"},
        ])
        assert len(results) == 2

    def test_predict_batch_empty_returns_empty(self):
        pipe    = _make_pipeline()
        results = pipe.predict_batch([])
        assert results == []

    def test_scaler_applied_when_present(self):
        from sklearn.preprocessing import StandardScaler

        scaler = MagicMock(spec=StandardScaler)
        scaler.transform.side_effect = lambda X: X  # identity

        pipe = blend_pipeline(_make_trained_models(0.8), scaler=scaler)
        pipe.predict_single({"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
        assert scaler.transform.called

    def test_save_and_load_roundtrip(self, tmp_path):
        """Roundtrip via from_variant_ensemble — no VariantEnsemble dependency at load time."""
        from sklearn.linear_model import LogisticRegression
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble, EnsembleConfig, TABULAR_FEATURES

        cfg = EnsembleConfig(n_folds=2, model_dir=tmp_path / "models")
        ens = VariantEnsemble(cfg)
        ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=50)}

        rng   = np.random.default_rng(42)
        n     = 20
        X_tab = pd.DataFrame(rng.random((n, len(TABULAR_FEATURES))), columns=TABULAR_FEATURES)
        y     = pd.Series([0] * 10 + [1] * 10)
        # X_seq=None: the roster on line 454 is logistic_regression only, so no
        # model takes the sequence branch. Before Part 3 (ff97c34) the signature
        # demanded a value and this test fabricated one -- a pd.Series, which is
        # also the wrong shape (roadmap 6.28: production passes a 2-column frame).
        ens.fit(X_tab, None, y)

        pipe = InferencePipeline.from_variant_ensemble(ens)
        out  = tmp_path / "pipeline.joblib"
        pipe.save(out)
        from tests.unit._admission_support import AllowForTests, register
        registry = register(tmp_path / "registry.v1.json", out, feature_names=pipe._declared_features(),
                            roster=list(pipe.trained_models),
                            predictor_contract_sha256=pipe.predictor_contract_sha256())
        loaded = InferencePipeline.load(out, consumer="test", registry_path=registry, authority=AllowForTests())
        # G7.2 (package 1, set 1c): only evidence bound to the selected predictor is carried; without a completed run
        # it is unavailable WITH a reason, and the removed unbound scalars refuse instead of reading 0.0.
        assert loaded.metadata.evaluation_evidence["status"] == "unavailable"
        assert loaded.metadata.fit_input_rows == 20
        assert loaded.metadata.feature_names == [str(f) for f in loaded._declared_features()]
        from genomic_variant_classifier.prediction_contract import ContractError
        with pytest.raises(ContractError, match="PIPEMETA-1"):
            loaded.metadata.val_auroc
        assert loaded.combiner.digest == pipe.combiner.digest
        assert "logistic_stacker" in loaded.research_candidates      # carried, never called
        result = loaded.predict_single({"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
        assert set(result) == RECORD_KEYS
        assert result["combiner"]["combiner_sha256"] == pipe.combiner.digest
        assert result["preprocessing_status"] == "fitted_training_policy_applied"

    def test_predict_proba_applies_the_missing_value_policy(self, tmp_path):
        """PIPELINE-FILL-1: value coverage for the SECOND call site.

        engineer_features is invoked TWICE in pipeline.py -- once in
        predict_proba and once in predict_proba_with_uncertainty -- in two
        methods that duplicate forty lines and share no _prepare
        (PIPELINE-PREP-DUP-1). test_save_and_load_roundtrip covers the second
        via predict_single. This covers the FIRST.

        WHY THE EXISTING COVERAGE IS NOT ENOUGH. test_catboost.py exercises
        predict_proba, but builds its pipeline from MagicMock estimators that
        accept any input and return a canned array: they verify DataFrame-
        versus-ndarray dispatch and CANNOT DETECT NaN. Its numeric_variant_df
        fixture also populates gene_constraint_oe, so no missing value ever
        reaches the estimator. Container coverage, not value coverage.

        This test uses a REAL LogisticRegression, which raises

            ValueError: Input X contains NaN.
            LogisticRegression does not accept missing values encoded as NaN

        so a missing value genuinely breaks it if the policy is not applied.
        """
        from sklearn.linear_model import LogisticRegression
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        from genomic_variant_classifier.models.variant_ensemble import (
            VariantEnsemble, EnsembleConfig, TABULAR_FEATURES,
        )

        cfg = EnsembleConfig(n_folds=2, model_dir=tmp_path / "models")
        ens = VariantEnsemble(cfg)
        ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=50)}
        rng = np.random.default_rng(11)
        X_tab = pd.DataFrame(rng.random((20, len(TABULAR_FEATURES))),
                             columns=TABULAR_FEATURES)
        y = pd.Series([0] * 10 + [1] * 10)
        ens.fit(X_tab, None, y)

        pipe = InferencePipeline.from_variant_ensemble(ens)
        assert pipe.preprocessor_ is not None, (
            "from_variant_ensemble did not carry the fitted policy; this test "
            "would then exercise the LEGACY path and prove nothing")

        # A BARE variant. No connector supplies gnomAD constraint, so
        # engineer_features correctly returns NaN for gene_constraint_oe and
        # gene_is_constrained -- the same input that broke predict_single.
        bare = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}])

        proba = pipe.predict_proba(bare)
        assert np.isfinite(proba).all(), (
            "predict_proba returned non-finite probabilities: {}".format(proba))
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_the_bare_variant_WOULD_break_an_unprotected_pipeline(self, tmp_path):
        """THE CONTROL. Without it the test above proves nothing.

        A passing assertion on a benign input is not evidence that a policy
        works -- only that nothing needed one. This demonstrates the SAME input
        genuinely breaks the SAME estimator when the fitted policy is absent,
        which is the legacy-artefact path _apply_missing_value_policy
        deliberately passes through unchanged.
        """
        from sklearn.linear_model import LogisticRegression
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        from genomic_variant_classifier.models.variant_ensemble import (
            VariantEnsemble, EnsembleConfig, TABULAR_FEATURES,
        )

        cfg = EnsembleConfig(n_folds=2, model_dir=tmp_path / "models")
        ens = VariantEnsemble(cfg)
        ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=50)}
        rng = np.random.default_rng(11)
        X_tab = pd.DataFrame(rng.random((20, len(TABULAR_FEATURES))),
                             columns=TABULAR_FEATURES)
        ens.fit(X_tab, None, pd.Series([0] * 10 + [1] * 10))

        pipe = InferencePipeline.from_variant_ensemble(ens)
        pipe.preprocessor_ = None          # a pipeline pickled before 2026-08-11

        bare = pd.DataFrame([{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}])
        with pytest.raises(ValueError, match="NaN"):
            pipe.predict_proba(bare)

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        p = tmp_path / "bad.joblib"
        joblib.dump({"not": "a pipeline"}, p)
        from tests.unit._admission_support import AllowForTests, register
        # Registered and ALLOWED, so the ONLY thing left to raise TypeError is the object-type check
        # (called bare, a missing-keyword TypeError used to satisfy this test for the wrong reason).
        registry = register(tmp_path / "registry.v1.json", p, feature_names=["x"], roster=["m"])
        with pytest.raises(TypeError, match="Expected InferencePipeline"):
            InferencePipeline.load(p, consumer="test", registry_path=registry, authority=AllowForTests())


# ---------------------------------------------------------------------------
# 6. FastAPI endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    """A READY world: the pipeline's real saved bytes, a registry record identifying THIS fitted predictor (exact
    roster and predictor contract), promoted to production, resolved by the production resolver.

    Package 1, set 1a (2026-11-07): /predict and /batch refuse unless the runtime binding is ready, so every
    prediction test runs against an attributed predictor. The unattributed world is `unattributed_client` below.
    """
    from fastapi.testclient import TestClient
    import genomic_variant_classifier.api.main as api_main
    from tests.unit._prediction_support import ready_binding

    saved = (api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING)
    pipe = _make_pipeline(proba=0.92)
    binding, _artifact, _registry = ready_binding(tmp_path, pipe)
    assert binding.is_ready, binding
    api_main._PIPELINE = pipe
    api_main._RUNTIME_MODEL_BINDING = binding
    try:
        yield TestClient(api_main.app)
    finally:
        api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING = saved


@pytest.fixture
def unattributed_client():
    """An UNATTRIBUTED world: a pipeline object with no artifact bytes.

    The name is kept because many tests request it, but the world it
    constructs is now explicit. A pipeline injected straight into the
    process is loaded and usable and has NO PROVENANCE, so this client
    is alive and NOT READY, and `/info` reports
    `no_artifact_identity`. Until 2026-08-07 the suite asserted this
    world was "ok".

    BOTH globals are set and BOTH are restored. A binding surviving a
    pipeline swap is a failure mode PROD-1 creates, and this fixture is
    where it would first appear. Deliberately not autouse and it does
    not request `monkeypatch`: the root conftest.py records what happens
    when an autouse fixture drags monkeypatch into the autouse group.

    tests/unit/test_runtime_attribution.py carries the fixtures for the
    attributed worlds, including one that drives the real lifespan.
    """
    from fastapi.testclient import TestClient
    import genomic_variant_classifier.api.main as api_main

    saved = (api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING)
    pipe = _make_pipeline(proba=0.92)
    api_main._PIPELINE = pipe
    api_main._RUNTIME_MODEL_BINDING = RuntimeModelBinding.unattributed(
        served_model_roster(pipe),
        detail="injected directly by the test fixture; no artifact bytes "
               "were measured")
    try:
        yield TestClient(api_main.app)
    finally:
        api_main._PIPELINE, api_main._RUNTIME_MODEL_BINDING = saved


class TestHealthEndpoint:

    def test_health_reports_degraded_without_attribution(self, unattributed_client):
        """PROD-1. A process that loaded bytes it cannot identify is
        ALIVE and NOT READY. An orchestrator should stop routing
        inference traffic to it rather than restart it, and for a
        clinical-facing service "I loaded a model but cannot establish
        what it is" must not be operationally green."""
        r = unattributed_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"]     is True
        assert body["live"]             is True
        assert body["ready"]            is False
        assert body["model_attributed"] is False
        assert body["status"]           == "degraded"
        assert "uptime_seconds"         in body

    def test_health_includes_uptime(self, client):
        r = client.get("/health")
        assert isinstance(r.json()["uptime_seconds"], float)

    def test_health_model_not_ready_when_no_artifact(self):
        from fastapi.testclient import TestClient
        import genomic_variant_classifier.api.main as api_main

        # The global is RESTORED (it leaked into every later test until 2026-11-07).
        saved = api_main._PIPELINE
        api_main._PIPELINE = None
        try:
            c = TestClient(api_main.app)
            r = c.get("/health")
            assert r.status_code == 200
            body = r.json()
            assert body["model_loaded"] is False
            assert body["status"]       == "degraded"
        finally:
            api_main._PIPELINE = saved


class TestInfoEndpoint:

    def test_info_reports_attribution_and_no_metric(self, unattributed_client):
        """The four assertions this replaced pinned constants written on
        2026-03-25 and never updated through Runs 9 to 16. The suite
        DEFENDED them, which is why they survived. `/info` now reports
        what it is serving; it reports no metric at all."""
        r = unattributed_client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["api_version"]  == "3.0.0"
        assert body["model_loaded"] is True
        assert body["attribution"]["resolution_status"] == (
            "no_artifact_identity")
        assert body["attribution"]["model_version"] is None
        assert body["attribution"]["record_id"] is None
        for gone in ("holdout_auroc", "training_auroc",
                     "training_auprc", "pipeline_version",
                     "description"):
            assert gone not in body
        assert body["n_features"]        == EXPECTED_TABULAR_FEATURE_COUNT
        assert len(body["feature_names"]) == EXPECTED_TABULAR_FEATURE_COUNT
        # Phase 4: all features promoted, phase2_features_remaining is now empty
        assert isinstance(body["phase2_features_remaining"], list)

    def test_info_phase2_remaining_is_empty(self, client):
        """All Phase 2 features have been promoted to TABULAR_FEATURES in Phase 4."""
        r = client.get("/info")
        remaining = r.json()["phase2_features_remaining"]
        assert isinstance(remaining, list)
        assert len(remaining) == 0, (
            f"Expected empty phase2_features_remaining, got: {remaining}"
        )


class TestPredictEndpoint:

    def test_predict_single_valid(self, client):
        payload = {
            "chrom": "17", "pos": 43094692, "ref": "G", "alt": "A",
            "consequence": "missense_variant",
            "gene_symbol": "BRCA1",
            "alphamissense_score": 0.94,
            "allele_freq": 0.0,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        pred = body["prediction"]
        assert set(pred) == RECORD_KEYS
        assert pred["pathogenicity_score"] == 0.92            # passed through unchanged: no clip, no rounding
        assert pred["model_call"] == 1 and pred["tie_rule"] == "positive_if_p_ge_0.5"
        assert pred["variant_id"] == "17:43094692:G:A"

    def test_an_unattributed_model_refuses_inference_before_any_member_call(self, unattributed_client):
        """Owner ruling 2026-11-07 (decision 38f15c7a): a binding that is not READY prevents scoring -- 503, outside
        the broad handlers, before feature preparation or any member call. It was a 200 until set 1a."""
        import genomic_variant_classifier.api.main as api_main
        members = list(api_main._PIPELINE.trained_models.values())
        for route, payload in (("/predict", {"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}),
                               ("/batch", {"variants": [{"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}]})):
            r = unattributed_client.post(route, json=payload)
            assert r.status_code == 503, (route, r.status_code, r.text)
            assert "not ready for inference" in r.json()["detail"]
        assert all(m.calls == [] for m in members), "a member was called before the readiness refusal"

    def test_predictions_carry_the_ready_record_identity(self, client):
        body = client.post("/predict", json={"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}).json()
        assert body["model_record_id"].startswith("synthetic-ready-")
        assert body["model_version"] == "synthetic-ready"
        assert "pipeline_version" not in body

    def test_predict_chrom_normalised(self, client):
        payload = {"chrom": "chr17", "pos": 43094692, "ref": "g", "alt": "a"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["prediction"]["variant_id"].startswith("17:")

    def test_predict_missing_required_field(self, client):
        r = client.post("/predict", json={"chrom": "17", "pos": 1, "ref": "G"})
        assert r.status_code == 422

    def test_predict_invalid_allele_freq(self, client):
        payload = {"chrom": "1", "pos": 1, "ref": "A", "alt": "T", "allele_freq": 2.0}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_predict_indel(self, client):
        payload = {"chrom": "17", "pos": 43115726, "ref": "AAC", "alt": "A",
                   "gene_symbol": "BRCA1"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_predict_sex_chromosome(self, client):
        r = client.post("/predict", json={"chrom": "X", "pos": 153_000_000, "ref": "C", "alt": "T"})
        assert r.status_code == 200

    def test_predict_mitochondrial(self, client):
        r = client.post("/predict", json={"chrom": "MT", "pos": 8993, "ref": "T", "alt": "G"})
        assert r.status_code == 200

    def test_predict_503_when_no_model(self):
        from fastapi.testclient import TestClient
        import genomic_variant_classifier.api.main as api_main
        original = api_main._PIPELINE
        api_main._PIPELINE = None
        try:
            c = TestClient(api_main.app, raise_server_exceptions=False)
            r = c.post("/predict", json={"chrom": "1", "pos": 1, "ref": "A", "alt": "T"})
            assert r.status_code == 503
        finally:
            api_main._PIPELINE = original


class TestBatchEndpoint:

    def test_batch_two_variants(self, client):
        payload = {"variants": [
            {"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"},
            {"chrom": "1",  "pos": 925952,   "ref": "G", "alt": "T"},
        ]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"] == 2 == len(body["predictions"])
        assert body["n_model_call_positive"] + body["n_model_call_negative"] == 2
        assert [p["source_row_key"] for p in body["predictions"]] == ["position:0", "position:1"]
        assert [p["variant_id"] for p in body["predictions"]] == ["17:43094692:G:A", "1:925952:G:T"]
        assert body["model_version"] == "synthetic-ready"

    def test_batch_counts_partition_the_batch(self, client):
        """With proba=0.92 every model call is positive; the counts partition the batch (the 2.0.0 counts
        matched "Pathogenic" by case-sensitive substring and silently omitted every "Likely pathogenic")."""
        payload = {"variants": [
            {"chrom": "1", "pos": i + 1, "ref": "A", "alt": "T"}
            for i in range(5)
        ]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"]         == 5
        assert body["n_model_call_positive"] == 5
        assert body["n_model_call_negative"] == 0
        for gone in ("n_pathogenic", "n_benign", "n_uncertain"):
            assert gone not in body

    def test_batch_empty_rejected(self, client):
        r = client.post("/batch", json={"variants": []})
        assert r.status_code == 422

    def test_batch_chrom_normalisation(self, client):
        payload = {"variants": [{"chrom": "chr1", "pos": 1, "ref": "A", "alt": "T"}]}
        r = client.post("/batch", json=payload)
        assert r.status_code == 200
        assert r.json()["predictions"][0]["variant_id"].startswith("1:")

    def test_batch_503_when_no_model(self):
        from fastapi.testclient import TestClient
        import genomic_variant_classifier.api.main as api_main
        original = api_main._PIPELINE
        api_main._PIPELINE = None
        try:
            c = TestClient(api_main.app, raise_server_exceptions=False)
            r = c.post("/batch", json={"variants": [
                {"chrom": "1", "pos": 1, "ref": "A", "alt": "T"}
            ]})
            assert r.status_code == 503
        finally:
            api_main._PIPELINE = original


# ---------------------------------------------------------------------------
# 7. /rsid/{rs_id} endpoint
# ---------------------------------------------------------------------------

class TestRsidEndpoint:

    def test_rsid_unknown_when_no_index(self, client):
        """With no dbSNP index loaded, /rsid always returns known=false."""
        import genomic_variant_classifier.api.main as api_main
        original = api_main._DBSNP_INDEX
        api_main._DBSNP_INDEX = None
        try:
            r = client.get("/rsid/rs12345678")
            assert r.status_code == 200
            body = r.json()
            assert body["known"] is False
            assert body["rs_id"] == "rs12345678"
        finally:
            api_main._DBSNP_INDEX = original

    def test_rsid_known_with_index(self, client):
        """With a mock dbSNP index, /rsid resolves locus and returns prediction."""
        import pandas as pd
        import genomic_variant_classifier.api.main as api_main

        mock_index = pd.DataFrame([{
            "rs_id": "rs12345678",
            "chrom": "1",
            "pos": 925952,
            "ref": "G",
            "alt": "T",
        }]).set_index("rs_id")

        original = api_main._DBSNP_INDEX
        api_main._DBSNP_INDEX = mock_index
        try:
            r = client.get("/rsid/rs12345678")
            assert r.status_code == 200
            body = r.json()
            assert body["known"] is True
            assert body["chrom"] == "1"
            assert body["pos"] == 925952
            assert body["ref"] == "G"
            assert body["alt"] == "T"
            # Pipeline is loaded in the client fixture, so prediction should be present
            assert body["prediction"] is not None
            assert 0.0 <= body["prediction"]["pathogenicity_score"] <= 1.0
        finally:
            api_main._DBSNP_INDEX = original

    def test_rsid_normalises_prefix(self, client):
        """rs-ID without 'rs' prefix is accepted and normalised."""
        import genomic_variant_classifier.api.main as api_main

        original = api_main._DBSNP_INDEX
        api_main._DBSNP_INDEX = None
        try:
            r = client.get("/rsid/12345678")
            assert r.status_code == 200
            body = r.json()
            assert body["rs_id"] == "rs12345678"
        finally:
            api_main._DBSNP_INDEX = original

    def test_rsid_no_prediction_when_no_model(self):
        """When pipeline is not loaded, /rsid still resolves locus but no prediction."""
        import pandas as pd
        from fastapi.testclient import TestClient
        import genomic_variant_classifier.api.main as api_main

        mock_index = pd.DataFrame([{
            "rs_id": "rs99999999",
            "chrom": "2",
            "pos": 100,
            "ref": "A",
            "alt": "T",
        }]).set_index("rs_id")

        orig_pipeline = api_main._PIPELINE
        orig_index    = api_main._DBSNP_INDEX
        api_main._PIPELINE    = None
        api_main._DBSNP_INDEX = mock_index
        try:
            c = TestClient(api_main.app)
            r = c.get("/rsid/rs99999999")
            assert r.status_code == 200
            body = r.json()
            assert body["known"] is True
            assert body["prediction"] is None
        finally:
            api_main._PIPELINE    = orig_pipeline
            api_main._DBSNP_INDEX = orig_index


# ---------------------------------------------------------------------------
# 8. Auth header (X-API-Key)
# ---------------------------------------------------------------------------

class TestApiKeyAuth:

    def test_dev_mode_no_key_required(self, client):
        """When API_KEYS env var is empty, all requests pass without a key."""
        import genomic_variant_classifier.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset()   # simulate dev mode
        try:
            r = client.get("/info")
            assert r.status_code == 200
        finally:
            auth_module._VALID_KEYS = original

    def test_valid_key_accepted(self, client):
        """A request with a valid X-API-Key is accepted."""
        import genomic_variant_classifier.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/info", headers={"X-API-Key": "test-key"})
            assert r.status_code == 200
        finally:
            auth_module._VALID_KEYS = original

    def test_invalid_key_rejected(self, client):
        """A request with a wrong key returns 401."""
        import genomic_variant_classifier.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/info", headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 401
        finally:
            auth_module._VALID_KEYS = original

    def test_missing_key_rejected(self, client):
        """A request with no key returns 401 when API_KEYS is set."""
        import genomic_variant_classifier.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/info")
            assert r.status_code == 401
        finally:
            auth_module._VALID_KEYS = original

    def test_health_always_open(self, client):
        """/health must not require auth even when API_KEYS is set."""
        import genomic_variant_classifier.api.auth as auth_module
        original = auth_module._VALID_KEYS
        auth_module._VALID_KEYS = frozenset({"test-key"})
        try:
            r = client.get("/health")
            assert r.status_code == 200
        finally:
            auth_module._VALID_KEYS = original


# ---------------------------------------------------------------------------
# 9. Package 1 serving contract (2026-11-07): pairing by identity, rate identity
# ---------------------------------------------------------------------------

class TestServingContract:

    def _batch(self, client):
        return client.post("/batch", json={"variants": [
            {"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"},
            {"chrom": "1",  "pos": 925952,   "ref": "G", "alt": "T"},
        ]})

    def test_a_reordered_result_is_refused_not_paired_by_position(self, client):
        import genomic_variant_classifier.api.main as api_main
        original = api_main._PIPELINE.predict_batch
        api_main._PIPELINE.predict_batch = lambda rows: list(reversed(original(rows)))
        try:
            r = self._batch(client)
        finally:
            del api_main._PIPELINE.predict_batch
        assert r.status_code == 500 and "is not the submitted variant" in r.json()["detail"]

    def test_a_result_of_the_wrong_cardinality_is_refused(self, client):
        import genomic_variant_classifier.api.main as api_main
        original = api_main._PIPELINE.predict_batch
        api_main._PIPELINE.predict_batch = lambda rows: original(rows)[:1]
        try:
            r = self._batch(client)
        finally:
            del api_main._PIPELINE.predict_batch
        assert r.status_code == 500 and "2 variants submitted but 1 results returned" in r.json()["detail"]

    def test_single_and_batch_share_one_serializer(self, client):
        single = client.post("/predict", json={"chrom": "17", "pos": 43094692, "ref": "G", "alt": "A"}).json()
        batch = self._batch(client).json()
        s, b = single["prediction"], batch["predictions"][0]
        assert set(s) == set(b) == RECORD_KEYS
        for key in ("pathogenicity_score", "model_call", "tie_rule", "combiner", "predictive_entropy_nats",
                    "roster_research_view", "decision_policy"):
            assert s[key] == b[key], key

    def test_the_rate_limit_identity_is_the_digest_of_the_api_key(self):
        import hashlib

        from starlette.requests import Request

        import genomic_variant_classifier.api.main as api_main

        def request(headers):
            return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
                            "client": ("10.0.0.7", 1234), "method": "GET", "path": "/"})

        keyed = api_main._rate_key(request([("X-API-Key", "secret-key")]))
        assert keyed == "api-key-sha256:" + hashlib.sha256(b"secret-key").hexdigest()
        assert "secret-key" not in keyed
        # A Bearer header is NOT how this API authenticates, so it is not a rate identity either.
        assert api_main._rate_key(request([("Authorization", "Bearer secret-key")])) == "remote-address:10.0.0.7"
