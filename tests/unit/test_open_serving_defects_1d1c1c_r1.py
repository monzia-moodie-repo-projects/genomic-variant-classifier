"""Set "au.1": executable witnesses of the serving defects owner ruling ff77015c registered for repair before deployment
(further ruling, sections 6 and 7). Each test asserts the defect AS IT IS and that the living roadmap registers it, so a
repair makes the witness fail until the register is updated -- a defect cannot be closed silently, nor left unrecorded.

* RUNTIME-SERVING-IMAGE-1 -- the API image is built on Python 3.11 from a lock compiled under 3.11, while every fit runs
  under the Python 3.12 locks; and it installs neither torch nor imodelsx, so 4 of the 12 tabular members (tabular_nn,
  kan, mc_dropout, deep_ensemble) cannot even be deserialized there (measured 2026-11-16 by unpickling each fitted
  member with both blocked). The shared model libraries are pinned identically in both locks.
* DOCKER-RUNTIME-BASE-UNPINNED-1 -- the builder stage's base image is digest-pinned; the runtime stage's is a tag.
* SERVING-LEGACY-ROSTER-1 -- a projected pipeline declares the whole epoch contract (91 names), not its members' fitted
  roster, so a predictor fitted on a smaller admitted roster is refused at serving (loudly, never silently).

Author: Monzia Moodie
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROADMAP = (REPO / "docs" / "ROADMAP.md").read_text(encoding="utf-8")


def _lock_python(name: str) -> str:
    head = (REPO / name).read_text(encoding="utf-8").splitlines()[:4]
    found = [m.group(1) for line in head for m in [re.search(r"pip-compile with Python (\d+\.\d+)", line)] if m]
    assert len(found) == 1, name
    return found[0]


def _pins(name: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in
            re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", (REPO / name).read_text(encoding="utf-8"), re.M)}


def test_runtime_serving_image_1_is_present_and_registered():
    docker = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ARG PYTHON_VERSION=3\.11$", docker, re.M)
    assert _lock_python("requirements-api.lock") == "3.11"
    assert _lock_python("requirements-train-test-linux.lock") == _lock_python("requirements-train-test-windows.lock") == "3.12"
    api, train = _pins("requirements-api.lock"), _pins("requirements-train-test-linux.lock")
    for shared in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "xgboost", "lightgbm", "catboost"):
        assert api[shared] == train[shared], shared
    assert "torch" in train and "imodelsx" in train
    assert "torch" not in api and "imodelsx" not in api
    assert "RUNTIME-SERVING-IMAGE-1" in ROADMAP


def test_docker_runtime_base_unpinned_1_is_present_and_registered():
    docker = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.11-slim-bookworm@sha256:[0-9a-f]{64} AS builder$", docker, re.M)
    assert re.search(r"^FROM python:\$\{PYTHON_VERSION\}-slim AS api$", docker, re.M)
    assert "DOCKER-RUNTIME-BASE-UNPINNED-1" in ROADMAP


def test_serving_legacy_roster_1_is_present_and_registered():
    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, FeatureSemantics, roster_without_family
    from genomic_variant_classifier.models.member import ProjectionProtocol
    from genomic_variant_classifier.models.variant_ensemble import engineer_features
    from tests.unit._fit_contexts import epoch2_test_fit
    from tests.unit.test_ensemble_projection_1d1c1c import _ens, _fit, _raw

    raw, y, genes = _raw(seed=7)
    X = engineer_features(raw, semantics=FeatureSemantics.V2_OBSERVED_AF)
    X = X[list(roster_without_family(tuple(X.columns), AF_FAMILY))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ens = _fit(_ens(["logistic_regression"]), X, y, genes, epoch2_test_fit(ProjectionProtocol.COMMON_FOLD_MEDIAN))
    pipe = InferencePipeline.from_variant_ensemble(ens)
    assert len(ens.tabular_columns_) == 85 and len(pipe._declared_features()) == 91
    with pytest.raises(ValueError, match="feature names or order differ from the fitted roster"):
        pipe.score(raw.head(5))
    assert "SERVING-LEGACY-ROSTER-1" in ROADMAP
