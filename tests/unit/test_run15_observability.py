"""Regression tests for scripts/run15_observability.py per_model parsing.

A7 (2026-05-27): the log-grep parser expected the "==>" prefix but the master log
emits per-model metrics via Python logger without that prefix. Fix: read
structured outputs (per_model_metrics.csv, per_model_metrics_val.csv,
models/*_meta.json) directly, with log-grep as fallback.

Coverage:
  1. read_per_model_metrics_files reads OOF AUROC from *_meta.json
  2. Reads test metrics from per_model_metrics.csv
  3. Reads val metrics from per_model_metrics_val.csv
  4. Returns empty dict when outputs_dir is missing
  5. Returns empty dict when outputs_dir has no structured files
  6. Fixed log-grep regex matches Python-logger format (real Run 14 lines)
  7. Fixed log-grep regex still matches legacy "==>" prefix (backward compat)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run15_observability.py"


def _import_observability_module():
    """Load scripts/run15_observability.py by path (scripts/ is not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run15_observability", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def obs_mod():
    return _import_observability_module()


@pytest.fixture
def synthetic_outputs(tmp_path):
    """Synthetic outputs dir mirroring outputs/run14/full/ shape."""
    out = tmp_path / "run14_synth" / "full"
    (out / "models").mkdir(parents=True)
    for name, oof in [("catboost", 0.998), ("xgboost", 0.997), ("kan", 0.992)]:
        (out / "models" / f"{name}_meta.json").write_text(
            json.dumps({
                "name": name,
                "oof_auroc": oof,
                "saved_at_utc": "2026-05-26T11:00:00",
                "n_samples": 1000,
            }),
            encoding="utf-8",
        )
    (out / "per_model_metrics.csv").write_text(
        ",auroc,auprc,f1_macro,f1_weighted,mcc,brier\n"
        "catboost,0.9975,0.9912,0.9632,0.9761,0.9276,0.0166\n"
        "xgboost,0.9974,0.9906,0.9769,0.9852,0.9539,0.012\n"
        "kan,0.9896,0.968,0.9422,0.9628,0.8847,0.03\n",
        encoding="utf-8",
    )
    (out / "per_model_metrics_val.csv").write_text(
        ",auroc,auprc,f1_macro,f1_weighted,mcc,brier\n"
        "catboost,0.9975,0.9908,0.9739,0.9845,0.948,0.0132\n"
        "xgboost,0.9974,0.9896,0.9776,0.9869,0.9552,0.0109\n"
        "kan,0.9914,0.9722,0.9606,0.9771,0.9218,0.0189\n",
        encoding="utf-8",
    )
    return out


class TestReadPerModelMetricsFiles:
    """Cover the structured-file reader added in the A7 fix."""

    def test_reads_oof_from_meta_jsons(self, obs_mod, synthetic_outputs):
        result = obs_mod.read_per_model_metrics_files(synthetic_outputs)
        assert "catboost" in result
        assert result["catboost"]["oof_auroc"] == pytest.approx(0.998)
        assert result["xgboost"]["oof_auroc"] == pytest.approx(0.997)
        assert result["kan"]["oof_auroc"] == pytest.approx(0.992)
        assert result["catboost"]["saved_at_utc"] == "2026-05-26T11:00:00"
        assert result["catboost"]["n_samples"] == 1000

    def test_reads_test_metrics_from_csv(self, obs_mod, synthetic_outputs):
        result = obs_mod.read_per_model_metrics_files(synthetic_outputs)
        assert result["catboost"]["test_auroc"] == pytest.approx(0.9975)
        assert result["catboost"]["test_auprc"] == pytest.approx(0.9912)
        assert result["catboost"]["test_f1_macro"] == pytest.approx(0.9632)
        assert result["catboost"]["test_mcc"] == pytest.approx(0.9276)
        assert result["catboost"]["test_brier"] == pytest.approx(0.0166)

    def test_reads_val_metrics_from_csv(self, obs_mod, synthetic_outputs):
        result = obs_mod.read_per_model_metrics_files(synthetic_outputs)
        assert result["catboost"]["val_auroc"] == pytest.approx(0.9975)
        assert result["xgboost"]["val_auprc"] == pytest.approx(0.9896)

    def test_returns_empty_when_outputs_dir_missing(self, obs_mod, tmp_path):
        result = obs_mod.read_per_model_metrics_files(tmp_path / "does_not_exist")
        assert result == {}

    def test_returns_empty_when_no_structured_files(self, obs_mod, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = obs_mod.read_per_model_metrics_files(empty)
        assert result == {}


class TestParseLogForPerModelMetricsLogger:
    """Cover the log-grep fallback path. Pattern A regex was fixed in A7."""

    def test_python_logger_format_matches(self, obs_mod):
        """Real Run 14 log format (verified 2026-05-27 vs outputs/run14/run14_master.log).

        Pre-A7 the regex required "==>" prefix and matched zero lines on this
        format. Post-A7 it accepts "<model> OOF AUROC: <val>" anywhere on a line.
        """
        log = (
            "2026-05-26 10:49:47  INFO      genomic_variant_classifier.models.variant_ensemble    random_forest OOF AUROC: 0.9978\n"
            "2026-05-26 10:50:10  INFO      genomic_variant_classifier.models.variant_ensemble    xgboost OOF AUROC: 0.9984\n"
            "2026-05-26 11:55:38  INFO      genomic_variant_classifier.models.variant_ensemble    kan OOF AUROC: 0.9921\n"
        )
        result = obs_mod.parse_log_for_per_model_metrics(log)
        assert result["random_forest"]["oof_auroc"] == pytest.approx(0.9978)
        assert result["xgboost"]["oof_auroc"] == pytest.approx(0.9984)
        assert result["kan"]["oof_auroc"] == pytest.approx(0.9921)

    def test_shell_script_arrow_format_still_matches(self, obs_mod):
        """Backward compat: the legacy '==>' prefix format still resolves."""
        log = "==> random_forest OOF AUROC: 0.9999\n"
        result = obs_mod.parse_log_for_per_model_metrics(log)
        assert result["random_forest"]["oof_auroc"] == pytest.approx(0.9999)