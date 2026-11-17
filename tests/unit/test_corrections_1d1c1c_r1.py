"""Set "au.1": the corrective revision of stage 1D-1c.1C's model-facing core (owner ruling ff77015c, 2026-11-16; the
owner's independent audit archive 4a6ce62c, whose three candidate witnesses are turned here into refusal tests).

* CODE-IDENTITY-STALE-CACHE-1 -- a source edit after the first lookup must change the projection's code identity.
* MARGINAL-RESAMPLING-SAMPLE-UNBOUND-1 -- a changed fitted empirical sample must be refused at transform.
* PROTOCOL-B-CONDITIONAL-BLINDING-CLAIM-1 -- the marginal diagnostic does not establish conditional blinding (an exact
  witness on the real projection), and the draw depends on unrelated features (a stated property, not a repair).
* ENGINEERING-FIT-DEPLOYABLE-1 -- a load is refused unless the admission decision names the recorded fit protocol.

Author: Monzia Moodie
"""
from __future__ import annotations

import pickle
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier import model_admission as MA
from genomic_variant_classifier.code_identity import closure_digest_of_text, source_text, symbol_closure_digest
from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.models import model_preprocessing as MP
from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, derive_family
from genomic_variant_classifier.models.fit_context import ExecutionProtocol

REPO = Path(__file__).resolve().parents[2]
PM = MP.ProjectionMode


def _frame(af, other=None) -> pd.DataFrame:
    values = np.asarray(af, dtype=np.float64)
    family = derive_family(AF_FAMILY, values)
    return pd.DataFrame({**family, "other": np.arange(len(values), dtype=np.float64) if other is None else other})


def _projection(X: pd.DataFrame, mode: PM) -> MP.MemberProjection:
    return MP.MemberProjection(tuple(X.columns), mode.value, "gvc.features/2")


# ------------------------------------------------------------------------------- CODE-IDENTITY-STALE-CACHE-1
def _bound_symbols() -> list[str]:
    """Every code symbol a committed record or the projection binds."""
    found: set[str] = set()
    for record in sorted((REPO / "docs" / "census").glob("*.json")):
        found.update(re.findall(r"(?:src|scripts)/[A-Za-z0-9_/]+\.py::[A-Za-z_][A-Za-z0-9_.]*", record.read_text("utf-8")))
    found.update(MP.projection_code_identity((AF_FAMILY,)))
    return sorted(s for s in found if (REPO / s.split("::")[0]).is_file())


def test_the_closure_from_bytes_equals_the_closure_from_the_file_for_every_bound_symbol():
    symbols = _bound_symbols()
    assert len(symbols) > 50
    for symbol in symbols:
        raw = (REPO / symbol.split("::")[0]).read_bytes()
        assert closure_digest_of_text(symbol, source_text(raw)) == symbol_closure_digest(REPO, symbol), symbol


def test_source_text_decodes_exactly_as_read_text(tmp_path):
    raw = "def f():\r\n    return 1  # x\r\n\rdef g():\n    return 'é'\n".encode("utf-8")
    path = tmp_path / "m.py"
    path.write_bytes(raw)
    assert source_text(raw) == path.read_text(encoding="utf-8")


@pytest.fixture
def copied_models(tmp_path, monkeypatch):
    """The two source files the projection binds, copied to a scratch repository root the projection reads."""
    for rel in (MP._THIS_FILE, MP._SEMANTICS_FILE):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, target)
    monkeypatch.setattr(MP, "_repo_root", lambda: tmp_path)
    return tmp_path / MP._SEMANTICS_FILE


def _edit_formula(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_an_actual_source_edit_changes_the_code_identity_after_the_first_lookup(copied_models):
    before = MP.projection_code_identity((AF_FAMILY,))
    _edit_formula(copied_models, "return ((af >= 0.0001) & (af < 0.001))", "return ((af >= 0.0001) & (af < 0.002))")
    after = MP.projection_code_identity((AF_FAMILY,))
    symbol = f"{MP._SEMANTICS_FILE}::formula_af_is_rare"
    assert after[symbol] != before[symbol]
    assert {k for k in before if before[k] != after[k]} == {symbol}
    _edit_formula(copied_models, "return ((af >= 0.0001) & (af < 0.002))", "return ((af >= 0.0001) & (af < 0.001))")
    assert MP.projection_code_identity((AF_FAMILY,)) == before          # the same bytes, the same identity


def test_a_fitted_projection_refuses_to_transform_after_its_formula_source_is_edited(copied_models):
    X = _frame([1e-5, 2e-4, np.nan, 0.02, 0.5])
    p = _projection(X, PM.FOLD_MEDIAN).fit(X)
    p.transform(X)
    _edit_formula(copied_models, "return (af >= 0.01).astype(np.float64)", "return (af >= 0.02).astype(np.float64)")
    with pytest.raises(RuntimeError, match="no longer matches its recorded identity"):
        p.transform(X)


def test_a_comment_edit_keeps_the_semantic_identity(copied_models):
    X = _frame([1e-5, 2e-4, np.nan, 0.02, 0.5])
    p = _projection(X, PM.FOLD_MEDIAN).fit(X)
    _edit_formula(copied_models, "def formula_af_log10(af: np.ndarray) -> np.ndarray:\n",
                  "def formula_af_log10(af: np.ndarray) -> np.ndarray:  # a comment changes no derivation\n")
    out = p.transform(X)                                   # the closure digest is comment-insensitive by design
    assert list(out.columns)[0] == "af_model_raw"


# ----------------------------------------------------------------------- MARGINAL-RESAMPLING-SAMPLE-UNBOUND-1
def _resampling_projection():
    X = _frame([0.1, 0.2, np.nan, 0.3, 0.4, np.nan])
    return X, _projection(X, PM.MARGINAL_RESAMPLING_DRAW).fit(X)


def test_the_fitted_sample_is_backed_by_immutable_bytes():
    _X, p = _resampling_projection()
    sample = p._sorted_observed_["allele_frequency"]
    assert not sample.flags.writeable and not sample.flags.owndata
    with pytest.raises(ValueError):
        sample.setflags(write=True)


@pytest.mark.parametrize("tamper, match", [
    (lambda s: np.array([0.1, 0.2, 0.3, 0.777]), "changed after fitting"),
    (lambda s: np.array([0.1, 0.2, 0.3]), "count, domain or order"),
    (lambda s: np.array([0.4, 0.3, 0.2, 0.1]), "count, domain or order"),
    (lambda s: np.array([0.1, 0.2, 0.3, 1.5]), "count, domain or order"),
    (lambda s: np.array([0.1, 0.2, 0.3, np.nan]), "count, domain or order"),
    (lambda s: np.array([[0.1, 0.2, 0.3, 0.4]]), "one-dimensional"),
])
def test_a_changed_fitted_sample_is_refused_at_transform(tamper, match):
    X, p = _resampling_projection()
    before = p.transform(X)
    p._sorted_observed_["allele_frequency"] = tamper(p._sorted_observed_["allele_frequency"])
    assert p._identity() == p.identity_                    # the recorded identity alone cannot see this
    with pytest.raises(RuntimeError, match=match):
        p.transform(X)
    assert before.notna().all().all()


def test_a_missing_or_extra_family_sample_is_refused():
    X, p = _resampling_projection()
    p._sorted_observed_ = {}
    with pytest.raises(RuntimeError, match="not the fitted families"):
        p.transform(X)
    X, p = _resampling_projection()
    p._sorted_observed_["other_family"] = np.array([0.5])
    with pytest.raises(RuntimeError, match="not the fitted families"):
        p.transform(X)


def test_the_sample_check_survives_pickling():
    X, p = _resampling_projection()
    q = pickle.loads(pickle.dumps(p))
    pd.testing.assert_frame_equal(q.transform(X), p.transform(X))
    sample = q._sorted_observed_["allele_frequency"]
    if not sample.flags.writeable:                         # an unpickled array may own writable memory
        sample = sample.copy()
        q._sorted_observed_["allele_frequency"] = sample
    sample[0] = 0.15
    with pytest.raises(RuntimeError, match="changed after fitting"):
        q.transform(X)


# ---------------------------------------------------------------------- PROTOCOL-B-CONDITIONAL-BLINDING-CLAIM-1
def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score))


def test_marginal_resampling_hides_missingness_from_af_alone_but_not_from_af_with_the_other_features():
    """The owner's counterexample (ruling ff77015c, section 8.1) on the REAL projection: the observed AF is determined by
    a covariate Z; the surrogate is drawn from the pooled observed sample. AF alone barely reveals which rows were
    unobserved; AF together with Z reveals them, because only an unobserved row can pair a low AF with Z = 1."""
    rng = np.random.default_rng(20261116)
    n = 4000
    z = rng.integers(0, 2, n).astype(np.float64)
    missing = rng.random(n) < 0.5
    af = np.where(z == 1, 0.1, 0.0001)
    af[missing] = np.nan
    X = _frame(af, other=z + rng.normal(0, 1e-9, n))       # Z itself, made unique per row so every key differs
    projected = _projection(X, PM.MARGINAL_RESAMPLING_DRAW).fit(X).transform(X)["af_model_raw"].to_numpy()
    label = missing.astype(int)
    marginal = _auroc(projected, label)
    mismatch = ((projected > 0.01) != (z == 1)).astype(float)   # AF disagreeing with what Z determines when observed
    conditional = _auroc(mismatch, label)
    assert abs(marginal - 0.5) < 0.05, marginal
    assert conditional > 0.70, conditional
    assert mismatch[~missing].sum() == 0                    # an observed row never disagrees: the marker is exact


def test_the_surrogate_of_an_unobserved_row_moves_with_an_unrelated_feature():
    """A stated property of the current B1 draw (ruling ff77015c section 8.2), not a repair: its key is the row's
    non-family content, so perturbing another feature can change the surrogate. Perturbation and attribution studies
    must not read such a change as an effect of the perturbed feature alone."""
    X = _frame(np.r_[np.linspace(0.001, 0.9, 100), np.nan])
    p = _projection(X, PM.MARGINAL_RESAMPLING_DRAW).fit(X)
    values = set()
    for i in range(50):
        Y = X.copy()
        Y.at[100, "other"] = 1000.0 + i
        values.add(float(p.transform(Y)["af_model_raw"].iloc[100]))
    assert len(values) > 1


# ------------------------------------------------------------------------------ ENGINEERING-FIT-DEPLOYABLE-1
def test_the_admission_vocabulary_is_the_fit_context_vocabulary():
    assert set(MA.FIT_PROTOCOLS) == {p.value for p in ExecutionProtocol}
    assert MA.UNRECORDED_FIT not in MA.FIT_PROTOCOLS and MA.SCIENTIFIC_ONLY == {"scientific"}


class _Authority:
    def __init__(self, protocols=None):
        self.protocols = protocols

    def decide(self, record, *, consumer, policy_sha256):
        extra = {} if self.protocols is None else {"fit_protocols": self.protocols}
        return MA.Decision(record.artifact.sha256, consumer, policy_sha256, MA.Outcome.ALLOW, "test decision", **extra)


def _engineering_pipeline(tmp_path):
    from sklearn.linear_model import LogisticRegression

    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, EnsembleConfig, VariantEnsemble
    from tests.unit._admission_support import register
    from tests.unit._fit_contexts import EPOCH1_TEST_FIT

    ens = VariantEnsemble(EnsembleConfig(n_folds=2, model_dir=tmp_path / "models"))
    ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=50)}
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((20, len(TABULAR_FEATURES))), columns=TABULAR_FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ens.fit(X, None, pd.Series([0] * 10 + [1] * 10), authorization=EPOCH1_TEST_FIT)
    pipe = InferencePipeline.from_variant_ensemble(ens)
    out = tmp_path / "pipeline.joblib"
    pipe.save(out)
    registry = register(tmp_path / "registry.v1.json", out, feature_names=pipe._declared_features(),
                        roster=list(pipe.trained_models), predictor_contract_sha256=pipe.predictor_contract_sha256())
    return ens, pipe, out, registry


def test_an_engineering_fit_is_refused_unless_the_decision_names_it(tmp_path):
    from genomic_variant_classifier.api.pipeline import InferencePipeline

    _ens, pipe, out, registry = _engineering_pipeline(tmp_path)
    assert pipe.fit_context["protocol"] == "engineering_qualification"
    for authority in (_Authority(), _Authority(frozenset({"scientific"})), _Authority(frozenset({"historical_replay"}))):
        with pytest.raises(ContainmentError, match="engineering_qualification protocol, which this decision does not"):
            InferencePipeline.load(out, consumer="test", registry_path=registry, authority=authority)
    loaded = InferencePipeline.load(out, consumer="test", registry_path=registry,
                                    authority=_Authority(frozenset({"engineering_qualification"})))
    assert loaded.predictor_contract_sha256() == pipe.predictor_contract_sha256()


def test_an_ensemble_load_is_refused_before_any_member_is_deserialized(tmp_path):
    from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble
    from tests.unit._admission_support import register

    ens, _pipe, _out, _registry = _engineering_pipeline(tmp_path)
    path = tmp_path / "ens" / "ensemble.joblib"
    ens.save(path)
    registry = register(tmp_path / "ens.registry.v1.json", path, feature_names=tuple(ens.tabular_columns_),
                        roster=list(ens.trained_models_))
    shutil.rmtree(path.parent / "ensemble_models")          # a member read would now fail differently
    with pytest.raises(ContainmentError, match="engineering_qualification protocol"):
        VariantEnsemble.load(path, consumer="test", registry_path=registry, authority=_Authority())


def test_an_unrecorded_fit_needs_an_explicit_admission(tmp_path):
    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from tests.unit._admission_support import register
    from tests.unit._prediction_support import FixedProba, blend_pipeline

    pipe = blend_pipeline({"m": FixedProba([0.2])})
    assert pipe.fit_context is None
    out = tmp_path / "legacy.joblib"
    pipe.save(out)
    registry = register(tmp_path / "registry.v1.json", out, feature_names=pipe._declared_features(), roster=["m"],
                        predictor_contract_sha256=pipe.predictor_contract_sha256())
    with pytest.raises(ContainmentError, match="unrecorded protocol"):
        InferencePipeline.load(out, consumer="test", registry_path=registry, authority=_Authority())
    InferencePipeline.load(out, consumer="test", registry_path=registry, authority=_Authority(frozenset({"unrecorded"})))


@pytest.mark.parametrize("protocols", [frozenset(), {"scientific"}, frozenset({"deployable"}), ("scientific",)])
def test_a_decision_that_does_not_name_its_fit_protocols_properly_is_refused(tmp_path, protocols):
    _ens, _pipe, out, registry = _engineering_pipeline(tmp_path)
    with pytest.raises(ContainmentError, match="non-empty frozenset"):
        MA.admit_artifact(out, consumer="test", registry_path=registry, authority=_Authority(protocols))


def test_an_edited_fit_context_record_is_refused(tmp_path):
    _ens, pipe, out, registry = _engineering_pipeline(tmp_path)
    admission = MA.admit_artifact(out, consumer="test", registry_path=registry,
                                  authority=_Authority(frozenset({"scientific"})))
    forged = dict(pipe.fit_context, protocol="scientific")
    with pytest.raises(ContainmentError, match="does not reproduce its digest"):
        MA.require_admitted_fit(admission, forged)
    with pytest.raises(ContainmentError, match="malformed"):
        MA.require_admitted_fit(admission, {"protocol": "scientific"})
    with pytest.raises(ContainmentError, match="only through an Admission"):
        MA.require_admitted_fit(object(), None)
