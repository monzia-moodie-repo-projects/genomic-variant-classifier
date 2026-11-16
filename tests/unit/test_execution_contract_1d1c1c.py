"""Stage 1D-1c.1C: the information regime, the experiment declaration and the fit context (ruling aa999b96 decision B,
revised section 8; audit R5).

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources.admission import REGIMES
from genomic_variant_classifier.information_sources.allele_frequency import Arm
from genomic_variant_classifier.models.execution import (
    READINESS_GATES,
    ExecutionProtocol,
    ExperimentIntent,
    FitContext,
    InformationRegime,
    TrainingReadinessError,
    add_execution_arguments,
    fit_context_from_arguments,
    require_scientific_readiness,
    scientific_fit_context,
)
from genomic_variant_classifier.models import fit_context as fit_context_module
from genomic_variant_classifier.models.feature_semantics import FeatureSemantics as FS
from genomic_variant_classifier.models.member import ProjectionProtocol as PP

ROOT = Path(__file__).resolve().parents[2]
D = "0" * 64


def _manifest(**over):
    m = {"experiment_id": "x", "regime": "strict", "arm": "retain_reported_zeros", "projection": "capability_aware_p",
         "feature_semantics": "gvc.features/2", "source_policy_sha256": D, "feature_roster_sha256": D,
         "population_sha256": D, "split_sha256": D}
    m.update(over)
    return m


def test_the_regime_vocabulary_is_the_admission_authoritys():
    assert {r.value for r in InformationRegime} == set(REGIMES)


def test_a_manifest_without_a_regime_or_with_an_extra_key_is_refused():
    m = _manifest(); del m["regime"]
    with pytest.raises(ValueError, match="missing"):
        ExperimentIntent.from_manifest(m)
    with pytest.raises(ValueError, match="extra"):
        ExperimentIntent.from_manifest(_manifest(default_regime="strict"))
    with pytest.raises(ValueError, match="unknown regime"):
        ExperimentIntent.from_manifest(_manifest(regime="strictt"))
    with pytest.raises(ValueError):
        ExperimentIntent.from_manifest(_manifest(population_sha256="abc"))


def test_a_valid_declaration_reuses_the_af_arm_vocabulary_and_its_identity_binds_the_regime():
    a = ExperimentIntent.from_manifest(_manifest())
    b = ExperimentIntent.from_manifest(_manifest(regime="conditional"))
    assert a.arm is Arm.RETAIN_REPORTED_ZEROS and a.regime is InformationRegime.STRICT
    assert a.identity != b.identity


def test_a_fit_context_cannot_be_constructed_directly():
    with pytest.raises(TypeError, match="named constructor|direct construction"):
        FitContext(ExecutionProtocol.HISTORICAL_REPLAY, FS.V1_LEGACY, PP.LEGACY_UNPROJECTED, "x")


def test_the_constructors_enforce_epoch_and_protocol_pairs():
    h = FitContext.historical_replay("r")
    assert (h.protocol, h.semantics, h.projection) == (ExecutionProtocol.HISTORICAL_REPLAY, FS.V1_LEGACY,
                                                        PP.LEGACY_UNPROJECTED)
    with pytest.raises(ValueError):
        FitContext.engineering_qualification(FS.V1_LEGACY, PP.CAPABILITY_AWARE, "r")
    with pytest.raises(ValueError):
        FitContext.engineering_qualification(FS.V2_OBSERVED_AF, PP.LEGACY_UNPROJECTED, "r")
    with pytest.raises(ValueError):
        FitContext.engineering_qualification(FS.V2_OBSERVED_AF, PP.CAPABILITY_AWARE, "  ")
    a = FitContext.engineering_qualification(FS.V2_OBSERVED_AF, PP.CAPABILITY_AWARE, "r").as_record()
    b = FitContext.engineering_qualification(FS.V2_OBSERVED_AF, PP.COMMON_FOLD_MEDIAN, "r").as_record()
    assert a["digest"] != b["digest"] and not FitContext.historical_replay("r").scientific_artifact


def test_a_scientific_fit_is_refused_naming_every_readiness_gate():
    intent = ExperimentIntent.from_manifest(_manifest())
    with pytest.raises(TrainingReadinessError) as exc:
        scientific_fit_context(intent)
    for gate in READINESS_GATES:
        assert gate in str(exc.value)
    with pytest.raises(ValueError):
        require_scientific_readiness("strict")        # a string is not a declared regime


def test_no_scientific_fit_context_can_exist_at_this_stage():
    assert not hasattr(FitContext, "scientific")
    with pytest.raises(PermissionError, match="no scientific fit context"):
        FitContext(ExecutionProtocol.SCIENTIFIC, FS.V2_OBSERVED_AF, PP.CAPABILITY_AWARE, "r",
                   _token=fit_context_module._TOKEN)
    with pytest.raises(TypeError, match="ExperimentIntent"):
        scientific_fit_context(_manifest())


def test_execution_re_exports_the_one_fit_context_and_the_light_module_imports_no_information_source():
    assert FitContext is fit_context_module.FitContext and ExecutionProtocol is fit_context_module.ExecutionProtocol
    tree = ast.parse((ROOT / "src/genomic_variant_classifier/models/fit_context.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m and ("information_sources" in m or m.endswith(".execution"))}


def _args(*argv):
    p = argparse.ArgumentParser()
    add_execution_arguments(p)
    return p.parse_args(list(argv))


@pytest.mark.parametrize("argv,match", [
    ((), "--execution-protocol is required"),
    (("--execution-protocol", "scientific"), "--information-regime is required"),
    (("--execution-protocol", "scientific", "--information-regime", "strict"), "no certifying authority"),
    (("--execution-protocol", "historical_replay", "--information-regime", "strict"), "no information regime"),
])
def test_launcher_flags_refuse_before_any_data_is_read(argv, match):
    with pytest.raises(SystemExit, match=match):
        fit_context_from_arguments(_args(*argv), "launcher")


def test_launcher_flags_declare_a_historical_replay():
    ctx = fit_context_from_arguments(_args("--execution-protocol", "historical_replay"), "launcher")
    assert ctx.protocol is ExecutionProtocol.HISTORICAL_REPLAY and "launcher" in ctx.reason


@pytest.mark.parametrize("script", ["scripts/train.py", "scripts/run_phase2_eval.py", "scripts/run9_ablations.py"])
def test_every_ensemble_fit_in_a_launcher_passes_its_fit_context(script):
    tree = ast.parse((ROOT / script).read_text(encoding="utf-8"))
    fits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "fit" and isinstance(n.func.value, ast.Name) and "ensemble" in n.func.value.id]
    assert fits, f"{script}: no ensemble fit found; the check is not reaching the code"
    missing = [n.lineno for n in fits if not any(k.arg == "authorization" for k in n.keywords)]
    assert not missing, f"{script}: ensemble fits without authorization at lines {missing}"
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "fit_context_from_arguments" in calls
