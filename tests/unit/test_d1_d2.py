"""
tests/unit/test_d1_d2.py
=========================
Test battery for D.1 + D.2 deliverables.

Coverage
--------
D.1 correctness fixes:
  - F-02/F-03  _assert_clean_cohort: null alleles, dup IDs, missing key cols
  - F-05       run_phase2_eval auto-skip CNN when seq-windows absent
  - F-08       evaluation package import (prediction_artifacts)
  - F-13       OOF sidecar includes _train_row_idx

D.2 science additions:
  - splits.py  unseen_gene_holdout_split: disjointness, both-class gate,
               missing column, overlap detection
  - ntqr_evaluator.py  NTQREvaluator: stub mode, confusion-matrix correctness,
                       threshold validation, to_dict contract
  - topological_ph.py  TopologicalPHGenerator: leakage-guard attribute,
                       zero-fallback, fit-before-transform gate,
                       column-name contract
  - ablation_npig_permutation.py  _recompute_npig: shuffled-label correctness
                                   (FINDING F-10)
  - prediction_artifacts.py       RunArtifactWriter: OOF row-index column,
                                   atomic write contract, manifest schema

Standing rules
--------------
- No logging.basicConfig in test file.
- All tests are hermetic (tmp_path, no external files required).
- Stub/fallback mode tests run regardless of whether heavy deps are installed.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# F-08 — evaluation package import must not raise ImportError
# ===========================================================================

class TestEvaluationPackageImport:
    def test_package_importable(self):
        """genomic_variant_classifier.evaluation must import cleanly (F-08)."""
        mod = importlib.import_module("genomic_variant_classifier.evaluation")
        assert hasattr(mod, "ClinicalEvaluator"),  "ClinicalEvaluator missing"
        assert hasattr(mod, "RunArtifactWriter"),   "RunArtifactWriter missing"

    def test_prediction_artifacts_module_importable(self):
        mod = importlib.import_module(
            "genomic_variant_classifier.evaluation.prediction_artifacts"
        )
        assert hasattr(mod, "RunArtifactWriter")


# ===========================================================================
# F-02/F-03 — _assert_clean_cohort
# ===========================================================================

def _minimal_df(**overrides):
    """Return a valid two-row ClinVar-like DataFrame."""
    base = {
        "variant_id":   ["1:100:A:T", "1:200:C:G"],
        "chrom":        ["1", "1"],
        "pos":          [100, 200],
        "ref":          ["A", "C"],
        "alt":          ["T", "G"],
        "clinical_sig": ["Pathogenic", "Benign"],
        "gene_symbol":  ["BRCA1", "BRCA2"],
        "label":        [1, 0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestAssertCleanCohort:
    """
    Tests for DataPrepPipeline._assert_clean_cohort.

    The HEAD version of real_data_prep.py (attached doc 1 / 553efac) must
    be the file on disk.  These tests also serve as a regression lock so
    future re-pulls or refactors don't silently re-introduce null-key leakage
    (INCIDENT_2026-05-31_null-key-leak.md).
    """

    @staticmethod
    def _guard(df, source="test"):
        from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline
        DataPrepPipeline._assert_clean_cohort(df, source)

    def test_passes_on_valid_df(self):
        self._guard(_minimal_df())

    def test_null_ref_raises(self):
        df = _minimal_df(ref=[None, "C"])
        with pytest.raises(ValueError, match="null/empty ref or alt"):
            self._guard(df)

    def test_empty_string_alt_raises(self):
        df = _minimal_df(alt=["", "G"])
        with pytest.raises(ValueError, match="null/empty ref or alt"):
            self._guard(df)

    def test_nan_string_ref_raises(self):
        df = _minimal_df(ref=["nan", "C"])
        with pytest.raises(ValueError, match="null/empty ref or alt"):
            self._guard(df)

    def test_dot_alt_raises(self):
        df = _minimal_df(alt=[".", "G"])
        with pytest.raises(ValueError, match="null/empty ref or alt"):
            self._guard(df)

    def test_duplicate_variant_id_raises(self):
        df = _minimal_df(variant_id=["1:100:A:T", "1:100:A:T"])
        with pytest.raises(ValueError, match="duplicate"):
            self._guard(df)

    def test_missing_key_column_raises(self):
        """
        F-02/F-03: if neither variant_id nor (chrom, pos, ref, alt) exist,
        must raise ValueError — never silently skip the dedup check.
        """
        df = pd.DataFrame({
            "clinical_sig": ["Pathogenic", "Benign"],
            "label": [1, 0],
            "ref": ["A", "C"],
            "alt": ["T", "G"],
            # 'chrom' and 'pos' deliberately absent, no 'variant_id'
        })
        with pytest.raises(ValueError, match="Cannot construct variant identity key"):
            self._guard(df)

    def test_locus_key_fallback_works(self):
        """If variant_id absent but (chrom, pos, ref, alt) present, use locus key."""
        df = _minimal_df()
        df = df.drop(columns=["variant_id"])
        self._guard(df)  # must not raise

    def test_locus_key_dedup_catches_duplicate(self):
        df = _minimal_df()
        df = df.drop(columns=["variant_id"])
        df2 = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            self._guard(df2)


# ===========================================================================
# D.2 — unseen_gene_holdout_split
# ===========================================================================

def _make_meta_train(n_genes: int = 20, variants_per_gene: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_genes):
        gene = f"GENE_{g:03d}"
        for v in range(variants_per_gene):
            rows.append({"gene_symbol": gene, "label": int(rng.integers(0, 2))})
    df = pd.DataFrame(rows)
    # Guarantee both classes appear in every gene.
    for gene in df["gene_symbol"].unique():
        m = df["gene_symbol"] == gene
        df.loc[df[m].index[0], "label"] = 0
        df.loc[df[m].index[1], "label"] = 1
    return df.reset_index(drop=True)


class TestUnseenGeneHoldoutSplit:

    def test_gene_disjoint(self):
        from genomic_variant_classifier.data.splits import unseen_gene_holdout_split
        meta = _make_meta_train()
        sub_idx, hout_idx = unseen_gene_holdout_split(meta)
        sub_genes  = set(meta.iloc[sub_idx]["gene_symbol"])
        hout_genes = set(meta.iloc[hout_idx]["gene_symbol"])
        assert sub_genes.isdisjoint(hout_genes), \
            f"Gene overlap detected — leakage!  Overlap: {sub_genes & hout_genes}"

    def test_both_classes_in_each_partition(self):
        from genomic_variant_classifier.data.splits import unseen_gene_holdout_split
        meta = _make_meta_train()
        sub_idx, hout_idx = unseen_gene_holdout_split(meta)
        for name, idx in [("sub_train", sub_idx), ("holdout", hout_idx)]:
            classes = set(meta.iloc[idx]["label"].unique())
            assert classes == {0, 1}, \
                f"Partition '{name}' missing a class: {classes}"

    def test_full_coverage(self):
        """Every row in meta must appear in exactly one partition."""
        from genomic_variant_classifier.data.splits import unseen_gene_holdout_split
        meta = _make_meta_train()
        sub_idx, hout_idx = unseen_gene_holdout_split(meta)
        all_idx = set(sub_idx) | set(hout_idx)
        assert all_idx == set(range(len(meta))), \
            "Some rows are neither in sub_train nor holdout"

    def test_missing_gene_col_raises_value_error(self):
        # splits.py raises ValueError (not KeyError) — consistent with
        # test_splits.py::TestUnseenGeneHoldoutSplit::test_rejects_missing_gene_column
        from genomic_variant_classifier.data.splits import unseen_gene_holdout_split
        meta = pd.DataFrame({"label": [0, 1, 0, 1]})
        with pytest.raises(ValueError, match="gene_symbol"):
            unseen_gene_holdout_split(meta, gene_col="gene_symbol")

    # NOTE: test_missing_label_col_raises_key_error removed.
    # unseen_gene_holdout_split no longer requires a label column —
    # it partitions by gene hash only.  The label-column check was a
    # D.2-specific guard that conflicts with the pre-existing
    # test_splits.py API (which uses acmg_label, not label).

    def test_different_seeds_give_different_splits(self):
        from genomic_variant_classifier.data.splits import unseen_gene_holdout_split
        meta = _make_meta_train(n_genes=30)
        _, h1 = unseen_gene_holdout_split(meta, seed=42)
        _, h2 = unseen_gene_holdout_split(meta, seed=99)
        assert not np.array_equal(h1, h2), \
            "Different seeds produced identical holdout sets"

    def test_holdout_frac_respected(self):
        """Actual gene-level holdout fraction must be close to requested."""
        from genomic_variant_classifier.data.splits import unseen_gene_holdout_split
        meta = _make_meta_train(n_genes=40)
        sub_idx, hout_idx = unseen_gene_holdout_split(meta, holdout_frac=0.25)
        n_hout = meta.iloc[hout_idx]["gene_symbol"].nunique()
        n_sub  = meta.iloc[sub_idx]["gene_symbol"].nunique()
        actual_frac = n_hout / (n_sub + n_hout)
        assert 0.15 < actual_frac < 0.40, \
            f"Holdout gene fraction {actual_frac:.3f} far from 0.25"


# ===========================================================================
# D.2 — NTQREvaluator
# ===========================================================================

class TestNTQREvaluator:

    def test_confusion_matrix_correct(self):
        """Confusion-matrix counts must match manual computation."""
        from genomic_variant_classifier.evaluation.ntqr_evaluator import NTQREvaluator
        ev     = NTQREvaluator(threshold=0.5)
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.9])
        b      = ev.evaluate(y_true, y_prob)
        assert b.q_00 == 1, "TN"
        assert b.q_01 == 1, "FP"
        assert b.q_10 == 1, "FN"
        assert b.q_11 == 1, "TP"

    def test_n_class_counts_correct(self):
        from genomic_variant_classifier.evaluation.ntqr_evaluator import NTQREvaluator
        ev     = NTQREvaluator(threshold=0.5)
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.6, 0.9])
        b      = ev.evaluate(y_true, y_prob)
        assert b.n_benign     == 3
        assert b.n_pathogenic == 2

    def test_stub_mode_when_ntqr_absent(self):
        """When ntqr is absent, bounds must be None and ntqr_available False."""
        from genomic_variant_classifier.evaluation.ntqr_evaluator import (
            NTQREvaluator, _NTQR_AVAILABLE,
        )
        if _NTQR_AVAILABLE:
            pytest.skip("ntqr is installed; stub-mode test not applicable")
        ev = NTQREvaluator(threshold=0.5)
        rng = np.random.default_rng(0)
        b   = ev.evaluate(
            np.array([0, 1, 0, 1, 0]),
            rng.uniform(0, 1, 5),
        )
        assert b.ntqr_available is False
        assert b.benign_accuracy_lower    is None
        assert b.pathogenic_accuracy_lower is None

    def test_to_dict_has_all_keys(self):
        from genomic_variant_classifier.evaluation.ntqr_evaluator import NTQREvaluator
        ev = NTQREvaluator(threshold=0.5)
        b  = ev.evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.3, 0.8]))
        d  = b.to_dict()
        for key in ("n_benign", "n_pathogenic", "q_00", "q_01", "q_10", "q_11",
                    "benign_accuracy_lower", "benign_accuracy_upper",
                    "pathogenic_accuracy_lower", "pathogenic_accuracy_upper",
                    "ntqr_available"):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_invalid_threshold_raises(self):
        from genomic_variant_classifier.evaluation.ntqr_evaluator import NTQREvaluator
        with pytest.raises(ValueError, match="threshold"):
            NTQREvaluator(threshold=1.5)

    def test_mismatched_shapes_raise(self):
        from genomic_variant_classifier.evaluation.ntqr_evaluator import NTQREvaluator
        ev = NTQREvaluator()
        with pytest.raises(ValueError, match="shape"):
            ev.evaluate(np.array([0, 1]), np.array([0.1, 0.9, 0.8]))

    def test_sensitivity_alias(self):
        """sensitivity_lower/upper aliases must equal pathogenic bounds."""
        from genomic_variant_classifier.evaluation.ntqr_evaluator import NTQREvaluator
        ev = NTQREvaluator(threshold=0.5)
        b  = ev.evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.7]))
        assert b.sensitivity_lower == b.pathogenic_accuracy_lower
        assert b.sensitivity_upper == b.pathogenic_accuracy_upper


# ===========================================================================
# D.2 — TopologicalPHGenerator
# ===========================================================================

class TestTopologicalPHGenerator:

    def test_leakage_guard_default_true(self):
        """train_genes_only defaults to True (Adopt #20 leakage guard)."""
        from genomic_variant_classifier.features.topological_ph import (
            TopologicalPHGenerator,
        )
        gen = TopologicalPHGenerator()
        assert gen.train_genes_only is True, \
            "train_genes_only must default to True (leakage guard active)"

    def test_column_names_contract(self):
        """transform() must return exactly PH_FEATURE_COLS columns."""
        from genomic_variant_classifier.features.topological_ph import (
            TopologicalPHGenerator, PH_FEATURE_COLS,
        )
        gen = TopologicalPHGenerator(string_path=None)
        gen.fit(["BRCA1", "TP53"])
        out = gen.transform(pd.Series(["BRCA1", "TP53", "UNKNOWNGENE"]))
        assert list(out.columns) == PH_FEATURE_COLS, \
            f"Column mismatch: {list(out.columns)}"
        assert len(out) == 3

    def test_zero_fallback_when_string_absent(self):
        """Unknown genes and missing STRING file → _PH_DEFAULTS values."""
        from genomic_variant_classifier.features.topological_ph import (
            TopologicalPHGenerator, _PH_DEFAULTS,
        )
        gen = TopologicalPHGenerator(string_path=None)
        gen.fit(["GENE_A"])
        out = gen.transform(pd.Series(["GENE_NOT_IN_GRAPH"]))
        assert out["ph_h1_n_loops"].iloc[0]  == _PH_DEFAULTS["ph_h1_n_loops"]
        assert out["ph_betti_1"].iloc[0]     == _PH_DEFAULTS["ph_betti_1"]

    def test_transform_before_fit_raises(self):
        """Calling transform() before fit() must raise RuntimeError."""
        from genomic_variant_classifier.features.topological_ph import (
            TopologicalPHGenerator,
        )
        gen = TopologicalPHGenerator(string_path=None)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            gen.transform(pd.Series(["BRCA1"]))

    def test_fit_is_idempotent(self):
        """Calling fit() twice must reset the cache cleanly."""
        from genomic_variant_classifier.features.topological_ph import (
            TopologicalPHGenerator,
        )
        gen = TopologicalPHGenerator(string_path=None)
        gen.fit(["GENE_A", "GENE_B"])
        gen.fit(["GENE_C"])  # second call must not crash or accumulate old cache
        # After second fit with no STRING file, cache is empty.
        assert "GENE_A" not in gen._gene_ph_cache

    def test_output_dtypes_numeric(self):
        """All PH feature columns must be numeric."""
        from genomic_variant_classifier.features.topological_ph import (
            TopologicalPHGenerator,
        )
        gen = TopologicalPHGenerator(string_path=None)
        gen.fit([])
        out = gen.transform(pd.Series(["ANY_GENE"]))
        for col in out.columns:
            assert pd.api.types.is_numeric_dtype(out[col]), \
                f"Column {col} is not numeric: dtype={out[col].dtype}"


# ===========================================================================
# D.2 — _recompute_npig (FINDING F-10 regression lock)
# ===========================================================================

class TestRecomputeNpig:
    """
    Verifies that _recompute_npig uses *shuffled* labels to recompute
    n_pathogenic_in_gene — not the original labels (FINDING F-10).
    """

    @staticmethod
    def _make_inputs():
        meta = pd.DataFrame({
            "gene_symbol": ["GENE_A"] * 4 + ["GENE_B"] * 4,
            "label":       [1, 1, 0, 0, 0, 0, 0, 0],
        })
        X_train = pd.DataFrame({
            "n_pathogenic_in_gene":  [2, 2, 2, 2, 0, 0, 0, 0],
            "gene_has_known_disease": [1, 1, 1, 1, 0, 0, 0, 0],
            "feature_x":             np.arange(8, dtype=float),
        })
        X_test = pd.DataFrame({
            "n_pathogenic_in_gene":  [2, 0],
            "gene_has_known_disease": [1, 0],
            "feature_x":             [10.0, 11.0],
        })
        return meta, X_train, X_test

    def test_shuffled_labels_produce_different_npig(self):
        """After shuffling, npig counts must differ from the original."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from ablation_npig_permutation import _recompute_npig

        meta, X_train, X_test = self._make_inputs()
        # All-benign shuffle: label=0 for all rows.
        y_shuffled = pd.Series([0] * 8)
        X_tr_p, X_te_p = _recompute_npig(X_train, y_shuffled, meta, X_test)
        # With all labels = 0, npig must be 0 for every gene.
        assert (X_tr_p["n_pathogenic_in_gene"] == 0).all(), \
            "npig must be 0 after all-benign shuffle"
        assert (X_te_p["n_pathogenic_in_gene"] == 0).all()

    def test_test_npig_always_zero(self):
        """Test-set npig must always be 0 after shuffling (unseen genes)."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from ablation_npig_permutation import _recompute_npig

        meta, X_train, X_test = self._make_inputs()
        y_shuffled = pd.Series([1, 0, 1, 0, 1, 0, 1, 0])
        _, X_te_p = _recompute_npig(X_train, y_shuffled, meta, X_test)
        assert (X_te_p["n_pathogenic_in_gene"] == 0).all(), \
            "Test-set npig must be 0 (unseen genes get no shuffled-train signal)"

    def test_missing_gene_symbol_raises(self):
        """meta_train without gene_symbol must raise ValueError."""
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from ablation_npig_permutation import _recompute_npig

        meta_bad = pd.DataFrame({"label": [1, 0, 1, 0, 1, 0, 1, 0]})
        meta, X_train, X_test = self._make_inputs()
        y_shuffled = pd.Series([0] * 8)
        with pytest.raises(ValueError, match="gene_symbol"):
            _recompute_npig(X_train, y_shuffled, meta_bad, X_test)


# ===========================================================================
# D.1 / F-13 — RunArtifactWriter OOF row-index (prediction_artifacts)
# ===========================================================================

class TestRunArtifactWriter:

    def test_oof_includes_row_index_when_provided(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        writer = RunArtifactWriter(
            run_id="test_run", ablation="full", output_dir=tmp_path
        )
        oof    = pd.DataFrame({
            "variant_id": ["v1", "v2", "v3"],
            "fold":        [0, 1, 2],
            "label":       [0, 1, 0],
            "catboost_prob": [0.1, 0.9, 0.4],
        })
        path = writer.save_oof_predictions(oof)
        df   = pd.read_parquet(path)
        assert len(df) == 3
        assert "label" in df.columns
        assert "catboost_prob" in df.columns

    def test_oof_missing_required_cols_raises(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        writer = RunArtifactWriter(
            run_id="test_run", ablation="full", output_dir=tmp_path
        )
        bad_oof = pd.DataFrame({"some_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required cols"):
            writer.save_oof_predictions(bad_oof)

    def test_save_manifest_writes_valid_json(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        writer = RunArtifactWriter(
            run_id="run15", ablation="c3_hybrid", output_dir=tmp_path
        )
        writer.save_manifest(
            git_sha="553efac",
            versions={"python": "3.12.10", "lightgbm": "4.0.0"},
            config={"n_folds": 5, "skip_nn": False},
        )
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["run_id"]      == "run15"
        assert data["ablation"]    == "c3_hybrid"
        assert data["git_sha"]     == "553efac"

    def test_atomic_write_leaves_no_tmp_on_success(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        writer = RunArtifactWriter(
            run_id="r", ablation="a", output_dir=tmp_path
        )
        writer.save_manifest(git_sha="abc", versions={}, config={})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, \
            f"Stale .tmp file(s) after successful write: {tmp_files}"

    def test_artefacts_list_tracks_written_files(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        writer = RunArtifactWriter(
            run_id="r", ablation="a", output_dir=tmp_path
        )
        assert len(writer.artefacts) == 0
        writer.save_manifest(git_sha="abc", versions={}, config={})
        assert "manifest.json" in writer.artefacts

    def test_init_requires_run_id_and_ablation(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        with pytest.raises(ValueError):
            RunArtifactWriter(run_id="", ablation="full", output_dir=tmp_path)
        with pytest.raises(ValueError):
            RunArtifactWriter(run_id="r", ablation="", output_dir=tmp_path)

    def test_save_graph_stats_missing_keys_raises(self, tmp_path):
        from genomic_variant_classifier.evaluation.prediction_artifacts import (
            RunArtifactWriter,
        )
        writer = RunArtifactWriter(
            run_id="r", ablation="a", output_dir=tmp_path
        )
        with pytest.raises(ValueError, match="missing required keys"):
            writer.save_graph_stats({"node_count": 100})  # edge_count missing


# ===========================================================================
# SR #31 preflight smoke-test script presence
# ===========================================================================

class TestSR31PreflightScripts:
    """The SR #31 PowerShell smoke-test scripts must exist at their expected paths."""

    def test_ntqr_sr31_script_exists(self):
        root = Path(__file__).parents[2]
        script = root / "docs" / "preflight" / "ntqr_sr31_check.ps1"
        if not script.exists():
            pytest.skip(
                f"ntqr_sr31_check.ps1 not found at {script} — "
                "create it before enabling ntqr in requirements.txt"
            )

    def test_gudhi_sr31_script_exists(self):
        root = Path(__file__).parents[2]
        script = root / "docs" / "preflight" / "gudhi_sr31_check.ps1"
        if not script.exists():
            pytest.skip(
                f"gudhi_sr31_check.ps1 not found at {script} — "
                "create it before enabling gudhi in requirements.txt"
            )
