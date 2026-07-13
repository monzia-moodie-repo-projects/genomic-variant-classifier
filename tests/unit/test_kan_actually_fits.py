"""Every base model in the roster must actually FIT. Importing is not fitting.

Added 2026-07-13.

WHY THIS FILE EXISTS
--------------------
On 2026-07-13, the first Continuous Integration run after the silent-model-drop fix
(commit 7d42409) went RED with:

    src/genomic_variant_classifier/models/kan.py:281:  in fit
        self._fit_imodelsx(X, y)
    .../imodelsx/kan/kan_sklearn.py:86:  in fit
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    E   NameError: name 'test_size' is not defined

    RuntimeError: Base model 'kan' FAILED during out-of-fold (OOF) prediction, so it could
    not be fitted and would have been silently dropped from the ensemble.

imodelsx 1.0.13 -- the latest release -- ships a `KANClassifier.fit` that references
`test_size`, `random_state` and `shuffle` as BARE NAMES. They are not parameters of `fit`
and not locals, so Python looks for module globals, does not find them, and raises. The
method cannot run at all, out of the box, anywhere.

THE PART THAT MATTERS: THIS HAD BEEN HAPPENING FOR MONTHS, INVISIBLY.
---------------------------------------------------------------------
The bug was "handled" two ways, and only one of them was real:

  1. kan.py set INSTANCE ATTRIBUTES (`self._imodelsx_model.test_size = 0.2`, etc.) before
     fitting. This was DEAD CODE. A NameError on a bare name cannot be fixed by setting an
     attribute on `self`. It never executed a single useful instruction, and it carried a
     comment claiming it fixed the bug. The 2026-05-28 KAN audit recorded the bug as
     "handled twice" -- one of those two handlers did nothing.

  2. A `sed -i` into the INSTALLED site-packages file, duplicated across
     launch_run11_vm.sh, launch_run16_vm.sh and launch_run16.py. THIS is what actually
     worked -- on the developer's laptop and on the Run 11 / Run 16 rented instances.

     It was never applied in Continuous Integration. Never in Docker. And it is NOT in
     scripts/vm_bootstrap_run.sh -- THE RUN 17 PATH.

So KAN raised NameError in every Continuous Integration run; the old bare `except
Exception` in VariantEnsemble.fit swallowed it; the ensemble trained TWELVE models instead
of thirteen and reported entirely normal metrics.

WHY NO GATE CAUGHT IT
---------------------
vm_bootstrap_run.sh, section "E. IMPORT + GPU GATE", checks:

    python -c "import imodelsx.kan.kan_sklearn"                       -> ok
    python -c "from ...models.kan import KANClassifier"               -> ok

**The bug is in fit(). Importing succeeds.** Every gate went green. Run 17 would have
provisioned a fresh instance, passed pre-flight, trained for eleven hours, silently dropped
the Kolmogorov-Arnold Network, and published a twelve-model algorithm comparison that never
mentioned it -- in a project whose stated first-class goal is comparing these algorithms.

An import check cannot see a bug in fit(). So this file FITS THINGS.

    A model that imports is not a model that trains.
"""
from __future__ import annotations

import numpy as np
import pytest

from genomic_variant_classifier.models.kan import (
    _KAN_BACKEND,
    KANClassifier,
)
from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig,
    VariantEnsemble,
)


def _xy(n: int = 60, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# The gate that was missing.
# ---------------------------------------------------------------------------
def test_kan_actually_fits_and_predicts():
    """THE TEST THAT WOULD HAVE CAUGHT IT. Fit KAN for real; do not merely import it."""
    X, y = _xy()
    kan = KANClassifier(hidden_sizes=[8], max_iter=2)

    kan.fit(X, y)                       # raised NameError in CI before 2026-07-13

    proba = kan.predict_proba(X)
    assert proba.shape == (len(y), 2), f"expected (n, 2) probabilities; got {proba.shape}"
    assert np.all((proba >= 0.0) & (proba <= 1.0)), "probabilities must lie in [0, 1]"
    assert np.allclose(proba.sum(axis=1), 1.0), "probability rows must sum to 1"


def test_the_imodelsx_repair_is_present_when_that_backend_is_active():
    """The in-process repair must have run -- not a sed, not an instance attribute.

    If imodelsx is the active backend, `KANClassifier.fit` is only runnable because
    _repair_imodelsx_kan_bare_names() injected the module globals the upstream code looks
    for. Assert the module globals exist. If a future imodelsx fixes fit() properly, the
    repair reports 'already-sane' and this still passes -- see the status assertion below.
    """
    if _KAN_BACKEND != "imodelsx":
        pytest.skip(f"imodelsx is not the active KAN backend (backend={_KAN_BACKEND!r})")

    import imodelsx.kan.kan_sklearn as ks

    from genomic_variant_classifier.models.kan import _IMODELSX_KAN_REPAIR

    assert _IMODELSX_KAN_REPAIR in {"repaired", "already-sane"}, (
        f"the imodelsx KAN repair did not run (status={_IMODELSX_KAN_REPAIR!r}). "
        f"KANClassifier.fit() will raise NameError."
    )

    if _IMODELSX_KAN_REPAIR == "repaired":
        for name in ("test_size", "random_state", "shuffle"):
            assert hasattr(ks, name), (
                f"module global {name!r} is missing from imodelsx.kan.kan_sklearn. "
                f"Upstream fit() reads it as a BARE NAME and will raise NameError."
            )


def test_kan_honours_its_own_random_state():
    """The repair must not cost us per-estimator seeding -- on EITHER installed form.

    Upstream `__init__` accepts random_state and discards it (verified 2026-07-13:
    hasattr(m, 'random_state') is False after construction). So whichever binding the
    installed `fit` reads -- `self.random_state` (sed-patched form) or the module global
    (pristine form) -- it must carry THIS estimator's seed, not the default 42.

    If the per-fit re-bind in `_fit_imodelsx` is ever removed, every KAN in the project
    silently collapses onto seed 42 and the ensemble quietly loses diversity. Nothing else
    would notice.
    """
    if _KAN_BACKEND != "imodelsx":
        pytest.skip(f"imodelsx is not the active KAN backend (backend={_KAN_BACKEND!r})")

    import imodelsx.kan.kan_sklearn as ks

    X, y = _xy()
    kan = KANClassifier(hidden_sizes=[8], max_iter=2, random_state=1234)
    kan.fit(X, y)

    inner = kan._imodelsx_model

    # Binding 1 -- what the SED-PATCHED form reads (`self.random_state`).
    assert getattr(inner, "random_state", None) == 1234, (
        "the inner imodelsx estimator's `self.random_state` is "
        f"{getattr(inner, 'random_state', '<ABSENT>')!r}, not 1234. On a sed-patched "
        "install, fit() reads exactly this attribute -- and upstream __init__ NEVER sets "
        "it, so _fit_imodelsx is the only thing that can. Restore that assignment."
    )

    # Binding 2 -- what the PRISTINE form reads (module global), when the repair is active.
    from genomic_variant_classifier.models.kan import _IMODELSX_KAN_REPAIR

    if _IMODELSX_KAN_REPAIR == "repaired":
        assert ks.random_state == 1234, (
            "the module global `imodelsx.kan.kan_sklearn.random_state` is "
            f"{ks.random_state!r}, not 1234. On a PRISTINE install, fit() resolves the bare "
            "name `random_state` here. The per-fit re-bind in _fit_imodelsx is gone, so "
            "every KAN now shares seed 42 regardless of configuration."
        )


def test_upstream_init_really_does_discard_its_own_parameters():
    """PIN THE UPSTREAM DEFECT. Everything else in this file follows from it.

    imodelsx 1.0.13's KANClassifier.__init__ accepts test_size / random_state / shuffle and
    NEVER STORES THEM. That single fact explains both failure modes:

        pristine form     -> fit() reads bare names   -> NameError
        sed-patched form  -> fit() reads self.<name>  -> AttributeError

    and it is why the instance-attribute assignments in _fit_imodelsx are load-bearing
    rather than, as was briefly assumed on 2026-07-13, dead code.

    IF THIS TEST FAILS, THAT IS GOOD NEWS: upstream has fixed __init__, and the whole repair
    apparatus (module globals, instance attributes, the sed in the launch scripts) can be
    re-examined. Do not delete the test -- re-measure, then simplify.
    """
    if _KAN_BACKEND != "imodelsx":
        pytest.skip(f"imodelsx is not the active KAN backend (backend={_KAN_BACKEND!r})")

    import imodelsx.kan.kan_sklearn as ks

    fresh = ks.KANClassifier(hidden_layer_sizes=[8])
    discarded = [n for n in ("test_size", "random_state", "shuffle") if not hasattr(fresh, n)]

    assert discarded == ["test_size", "random_state", "shuffle"], (
        f"imodelsx KANClassifier.__init__ now STORES {sorted(set(('test_size','random_state','shuffle')) - set(discarded))} "
        f"(it used to discard all three). The upstream bug this project works around has "
        f"changed. Re-measure both installed forms before touching "
        f"_repair_imodelsx_kan_bare_names() or _fit_imodelsx."
    )


def test_which_imodelsx_source_form_is_installed_here():
    """Make the environment divergence VISIBLE. It is the root defect, not a footnote.

    Two different source forms of `imodelsx/kan/kan_sklearn.py` exist in the wild:

        PRISTINE     `test_size=test_size`       -- what pip installs. CI, Docker, Run 17.
        SED-PATCHED  `test_size=self.test_size`  -- the developer's .venv312, and the
                                                    Run 11 / Run 16 rented instances, via a
                                                    `sed -i` in the launch scripts.

    The developer's site-packages has been MUTATED since 2026-05. Local tests have therefore
    been exercising a code path that no clean machine has, and "it passes on my machine" was
    load-bearing -- which is precisely how KAN came to be silently dropped from every
    Continuous Integration run without anyone noticing.

    This test does not fail on either form (the repair handles both, deliberately, so that a
    stale VM image cannot silently lose KAN). It exists to PRINT which form is present, so
    the divergence can never again be invisible.
    """
    if _KAN_BACKEND != "imodelsx":
        pytest.skip(f"imodelsx is not the active KAN backend (backend={_KAN_BACKEND!r})")

    import inspect

    import imodelsx.kan.kan_sklearn as ks

    src = inspect.getsource(ks)
    pristine = "test_size=test_size" in src
    patched = "test_size=self.test_size" in src

    assert pristine or patched, (
        "imodelsx/kan/kan_sklearn.py matches NEITHER known source form. The library has "
        "changed shape entirely; re-measure before trusting the KAN repair."
    )
    assert not (pristine and patched), "both forms present -- a partial sed? investigate."

    form = "PRISTINE (bare names)" if pristine else "SED-PATCHED (self.<name>)"
    print(f"\n  imodelsx source form installed here: {form}")
    print(f"  file: {ks.__file__}")
    print("  Both forms are repaired; see _repair_imodelsx_kan_bare_names().")


# ---------------------------------------------------------------------------
# Generalise it: NO base model may be un-fittable.
# ---------------------------------------------------------------------------
def test_every_base_model_in_the_roster_can_be_fitted():
    """Fit EVERY configured base model. An un-fittable model is an absent model.

    Before 2026-07-13 a base model whose fit raised was silently dropped, so 'the roster has
    13 models' and 'the run trained 13 models' were different statements and nothing checked
    the second one. `VariantEnsemble.fit` now raises instead -- but that only fires during a
    full ensemble fit. This test exercises each estimator directly, so a broken model is
    named individually rather than surfacing as one opaque ensemble failure.
    """
    roster = VariantEnsemble(EnsembleConfig()).base_estimators
    assert len(roster) >= 10, f"roster looks truncated: {sorted(roster)}"

    X, y = _xy(n=80)
    failures: dict[str, str] = {}

    for name, model in roster.items():
        if name == "cnn_1d":
            # cnn_1d consumes the one-hot DNA sequence, not the tabular matrix; it is
            # exercised by its own tests. Fitting it on X here would be meaningless.
            continue
        try:
            from sklearn.base import clone
            clone(model).fit(X, y)
        except Exception as exc:  # noqa: BLE001 - we WANT every failure, not the first
            failures[name] = f"{type(exc).__name__}: {exc}"

    assert not failures, (
        "these base models CANNOT BE FITTED, and before 2026-07-13 each would have been "
        "silently dropped from the ensemble -- appearing in the report not as a failure but "
        "as an algorithm that was never a candidate:\n  "
        + "\n  ".join(f"{n}: {e}" for n, e in sorted(failures.items()))
    )
