"""Test-isolation prerequisite (owner rulings 2026-09-22 and 2026-09-23). Real writes, real outcomes.

Nothing is written unless an explicit destination is given; where one is given, real writes land there;
a failed checkpoint is RECORDED, never only logged; importing a script has no filesystem side effect;
reports go to the artifact root, never the repository by default.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from genomic_variant_classifier.models.variant_ensemble import EnsembleConfig, VariantEnsemble

REPO = Path(__file__).resolve().parents[2]


def _cohort(n_genes=24, seed=0):
    rng = np.random.default_rng(seed)
    genes, labels, sig = [], [], []
    for g in range(n_genes):
        k = int(rng.integers(8, 16))
        for _ in range(k):
            lab = int(rng.random() < 0.5)
            genes.append(f"G{g}"); labels.append(lab); sig.append(rng.normal(lab, 1.0))
    X = pd.DataFrame({"sig": sig, "noise": rng.normal(size=len(sig))})
    return X, pd.Series(labels), pd.Series(genes)


def _light(**cfg):
    ens = VariantEnsemble(EnsembleConfig(n_folds=3, **cfg))
    for k in list(ens.base_estimators):
        if k not in {"logistic_regression", "random_forest"}:
            ens.base_estimators.pop(k)
    return ens


def _files(root: Path):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


class TestCheckpointDirectory:
    def test_construction_creates_nothing_and_defaults_to_no_checkpoints(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = EnsembleConfig()
        VariantEnsemble(cfg)
        assert cfg.model_dir is None
        assert _files(tmp_path) == []

    def test_an_explicit_model_dir_is_not_created_at_construction(self, tmp_path):
        target = tmp_path / "later"
        cfg = EnsembleConfig(model_dir=str(target))
        assert cfg.model_dir == target and isinstance(cfg.model_dir, Path)
        assert not target.exists()

    def test_fit_without_model_dir_writes_nothing_and_says_so(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        X, y, g = _cohort()
        ens = _light()
        ens.fit(X, None, y, gene_symbol=g)
        assert _files(tmp_path) == []
        assert ens.checkpoint_status_ == {"logistic_regression": "disabled", "random_forest": "disabled"}

    def test_fit_with_model_dir_really_writes_there(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        X, y, g = _cohort()
        ens = _light(model_dir=tmp_path / "ckpt")
        ens.fit(X, None, y, gene_symbol=g)
        assert set(ens.checkpoint_status_.values()) == {"saved"}
        written = {p.name for p in (tmp_path / "ckpt").iterdir()}
        assert {"logistic_regression.joblib", "random_forest.joblib"} <= written
        assert _files(tmp_path) == sorted({"ckpt", *(f"ckpt/{n}" for n in written)})

    def test_a_failed_checkpoint_is_recorded_not_only_logged(self, tmp_path):
        blocker = tmp_path / "not_a_directory"
        blocker.write_text("a file where the checkpoint directory should be")
        X, y, g = _cohort()
        ens = _light(model_dir=blocker)
        ens.fit(X, None, y, gene_symbol=g)
        assert all(v.startswith("failed: ") for v in ens.checkpoint_status_.values()), ens.checkpoint_status_

    def test_save_refuses_without_a_destination(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        X, y, g = _cohort()
        ens = _light()
        ens.fit(X, None, y, gene_symbol=g)
        with pytest.raises(ValueError, match="needs a destination"):
            ens.save()
        assert _files(tmp_path) == []
        ens.save(tmp_path / "export" / "ensemble.joblib")
        assert (tmp_path / "export" / "ensemble.joblib").is_file()


class TestCatBoostOutput:
    def test_the_ensemble_builds_catboost_with_training_files_off(self):
        pytest.importorskip("catboost")
        cb = VariantEnsemble(EnsembleConfig(skip_catboost=False)).base_estimators["catboost"]
        assert cb.allow_writing_files is False and cb.backend_output_root is None
        copy = clone(cb)
        assert copy.allow_writing_files is False and copy.backend_output_root is None

    def test_a_real_fit_and_a_clone_fit_write_nothing(self, tmp_path, monkeypatch):
        pytest.importorskip("catboost")
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        monkeypatch.chdir(tmp_path)
        X, y, _ = _cohort()
        est = CatBoostVariantClassifier(iterations=5, depth=2, early_stopping_rounds=None, cat_feature_names=[])
        for model in (est, clone(est)):
            model.fit(X, y)
            assert model._model.get_params()["allow_writing_files"] is False
        assert _files(tmp_path) == []

    def test_output_settings_passed_through_kwargs_are_refused(self):
        pytest.importorskip("catboost")
        from genomic_variant_classifier.models.catboost_wrapper import CatBoostVariantClassifier
        X, y, _ = _cohort()
        with pytest.raises(ValueError, match="Output options must use the policy"):
            CatBoostVariantClassifier(iterations=2, cat_feature_names=[], train_dir="anywhere").fit(X, y)


class TestNoImportTimeWrites:
    def test_importing_train_py_writes_nothing(self, tmp_path):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "scripts")])}
        r = subprocess.run([sys.executable, "-c", "import train"], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert _files(tmp_path) == [], "importing scripts/train.py created files"


class TestFreshnessReport:
    def test_the_report_goes_to_the_injected_directory(self, tmp_path):
        from genomic_variant_classifier.agent_layer.agents.database_freshness_monitor_agent import (
            DatabaseFreshnessMonitorAgent)
        agent = DatabaseFreshnessMonitorAgent.__new__(DatabaseFreshnessMonitorAgent)
        agent._report_dir = tmp_path / "reports_here"
        out = agent._write_report({"upstream": [], "local": [], "changes": []})
        assert out.parent == tmp_path / "reports_here" and out.is_file()

    def test_report_directory_resolution_order(self, tmp_path):
        from genomic_variant_classifier.agent_layer import config
        from genomic_variant_classifier.agent_layer.agents.database_freshness_monitor_agent import (
            DatabaseFreshnessMonitorAgent)
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        ss = SharedState(state_file=tmp_path / "state.json")
        default = DatabaseFreshnessMonitorAgent(ss)
        scoped = DatabaseFreshnessMonitorAgent(ss, root=str(tmp_path / "r"))
        explicit = DatabaseFreshnessMonitorAgent(ss, root=str(tmp_path / "r"), report_dir=str(tmp_path / "x"))
        assert default._report_dir == config.DATA_FRESHNESS_REPORT_DIR        # production: artifact root
        assert scoped._report_dir == tmp_path / "r" / "reports" / "data_freshness"   # hermetic scope
        assert explicit._report_dir == tmp_path / "x"                         # explicit wins

    def test_the_default_is_the_artifact_reports_root(self):
        from genomic_variant_classifier.agent_layer import config
        assert config.DATA_FRESHNESS_REPORT_DIR == config.LITERATURE_DIGEST_DIR.parent / "data_freshness"
