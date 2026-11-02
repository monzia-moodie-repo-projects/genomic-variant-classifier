"""The pykan fallback of models/kan.py: it fits, it writes nothing into the current directory, and its fit emits no warning.

MEASURED 2026-11-01 (pykan 0.2.8, the version in the training-and-test lock). pykan's KAN defaults to auto_save=True with
ckpt_path='./model': constructing it printed "checkpoint directory created: ./model", and a three-step fit left ./model/0.0,
./model/0.1 and ./model/history.txt in the CURRENT DIRECTORY -- from a checkout, that is a write into the checkout (the defect
class of models/best_gat.pt, 2026-10-11). kan.py now constructs it with auto_save=False.

The fallback runs only when imodelsx cannot be imported, so the suite never reached it: these tests select it explicitly. The same
measurement found that the blanket `warnings.simplefilter("ignore")` around its fit hid nothing (zero warnings with and without
it), so it was removed; the second test pins that a fit emits no warning, so a future pykan that starts warning is SEEN.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest

from genomic_variant_classifier.models import kan as kan_module

logger = logging.getLogger(__name__)

pykan = pytest.importorskip("kan", reason="pykan is not installed in this environment")


@pytest.fixture()
def pykan_backend(monkeypatch, tmp_path):
    """Select the pykan fallback for this test and run it in an empty current directory."""
    monkeypatch.setattr(kan_module, "_PyKAN", pykan.KAN, raising=False)
    monkeypatch.setattr(kan_module, "_KAN_BACKEND", "pykan")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    return X, (X[:, 0] > 0).astype(int)


def test_the_pykan_fallback_fits_and_writes_nothing_into_the_current_directory(pykan_backend):
    X, y = _data()
    model = kan_module.KANClassifier(hidden_sizes=[4], max_iter=3, random_state=0).fit(X, y)
    proba = model.predict_proba(X)
    assert model._backend_used == "pykan"
    assert proba.shape == (60, 2) and np.all(np.isfinite(proba)) and np.allclose(proba.sum(axis=1), 1.0)
    assert sorted(p.name for p in pykan_backend.iterdir()) == [], "pykan wrote into the current directory"


def test_a_pykan_fit_emits_no_warning(pykan_backend):
    X, y = _data()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        kan_module.KANClassifier(hidden_sizes=[4], max_iter=3, random_state=0).fit(X, y)
    assert [(w.category.__name__, str(w.message)) for w in seen] == []


def test_the_negative_control_pykan_with_its_defaults_does_write(pykan_backend):
    """Without auto_save=False the same construction DOES write ./model -- so the first test's empty directory is evidence."""
    pykan.KAN(width=[4, 4, 1], grid=5, k=3, seed=0)
    assert (pykan_backend / "model").is_dir()
