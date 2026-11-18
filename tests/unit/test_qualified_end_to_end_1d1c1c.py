"""Set "av": the fresh-process end-to-end proof of owner ruling aa999b96 revised section 15 ("what would constitute a
successful 1C"), on synthetic sources. No scientific performance claim follows; these tests establish correctness.

  1-2  the AF source holds an observed value, an observed zero and an unobserved value; the canonical matrix keeps all
       three (test_qualified_preparation_1d1c1c.py proves the states; here they reach the fitted members);
  3-5  each member's fitted projection consumes the canonical matrix under an explicit engineering-qualification
       context and the compiled strict roster (loeuf withheld through the admission authority); the members are
       serialised with their projections;
  6    a FRESH PROCESS loads the stored canonical matrix and the ensemble through the admission route and reproduces
       the predictions bit for bit;
  7    another source policy, another canonical matrix, a missing or edited source binding are refused -- in the fresh
       process too;
  8    an audit-only column changed in the cohort frame changes neither the matrix nor the predictions;
  +    a fresh process with numpy's processor dispatch turned off (another log10 implementation) prepares the same
       identity (CANONICAL-IDENTITY-ENVIRONMENT-1).
  Points 9 (stochastic-member schedules) and 10 (historical artifacts untouched) are held elsewhere:
  tests/unit/test_tabular_nn_mc_dropout.py and the installer's preservation stage.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import admission as AD
from genomic_variant_classifier.information_sources import allele_frequency as A
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import qualified_preparation as Q
from genomic_variant_classifier.models.member import ProjectionProtocol as PP
from tests.unit import _qualified_support as S
from tests.unit._admission_support import register
from tests.unit._fit_contexts import epoch2_test_fit
from tests.unit.test_ensemble_projection_1d1c1c import _ens

FRESH = textwrap.dedent("""
    import sys
    from pathlib import Path
    import numpy as np
    from genomic_variant_classifier.information_sources import qualified_preparation as Q
    from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble
    from tests.unit._admission_support import AllowForTests
    store, ident, payload, model, registry, out = sys.argv[1:7]
    ens = VariantEnsemble.load(Path(model), consumer="test", registry_path=Path(registry), authority=AllowForTests())
    if payload == "by-binding":      # the deterministic selector for exact replay: the representation the binding names
        payload = Q.payload_for(Path(store), ident, ens.source_binding_["representation_sha256"])
    cm = Q.load_canonical(Path(store), ident, payload)
    try:
        np.save(out, Q.score_canonical(ens, cm))
        print("SCORED")
    except PermissionError as exc:
        print("REFUSED", exc)
""")


DISPATCH = textwrap.dedent("""
    import sys
    from pathlib import Path
    import numpy._core._multiarray_umath as m
    from tests.unit import _qualified_support as S
    cm, _ = S.prepared(Path(sys.argv[1]))
    enabled = sorted(k for k in m.__cpu_dispatch__ if m.__cpu_features__.get(k))
    print("IDENTITY", cm.identity)
    print("DISPATCHED", " ".join(enabled) or "none")
""")


def _roster():
    raw = (S.REPO / C.LEDGER_PATH).read_bytes()
    compiled = AD.compiled_roster(C.strict_loads(raw.decode("ascii")), ledger_sha256=C.sha256_bytes(raw),
                                  regime="strict", withheld=frozenset({"loeuf"}))
    return list(Q.fit_roster(compiled))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    cm, inp = S.prepared(tmp / "sources")
    y = pd.Series(inp["cohort"]["label"].to_numpy())
    genes = pd.Series(inp["cohort"]["gene_symbol"].to_numpy())
    ens = _ens(["logistic_regression", "random_forest"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ens.fit(cm.features[_roster()], None, y, gene_symbol=genes, authorization=epoch2_test_fit(PP.COMMON_FOLD_MEDIAN))
    Q.bind_fit(ens, cm)
    store = tmp / "store"
    ident = Q.write_canonical(cm, store).identity
    model = tmp / "ensemble.joblib"
    ens.save(model)
    registry = register(tmp / "registry.v1.json", model, feature_names=tuple(ens.tabular_columns_),
                        roster=list(ens.trained_models_))
    return {"tmp": tmp, "cm": cm, "inp": inp, "ens": ens, "store": store, "ident": ident, "model": model,
            "registry": registry, "proba": Q.score_canonical(ens, cm)}


def _fresh(f, ident: str, payload: str, out_name: str):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(S.REPO / "src"), str(S.REPO)]),
           "PYTHONDONTWRITEBYTECODE": "1"}
    out = f["tmp"] / out_name
    r = subprocess.run([sys.executable, "-c", FRESH, str(f["store"]), ident, payload, str(f["model"]), str(f["registry"]),
                        str(out)], capture_output=True, text=True, env=env, cwd=str(f["tmp"]), timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout, out


def test_the_members_consume_the_canonical_matrix_through_fitted_projections(fitted):
    ens = fitted["ens"]
    assert list(ens.tabular_columns_) == _roster() and "loeuf" not in ens.tabular_columns_
    assert ens.fit_context_["protocol"] == "engineering_qualification"
    assert set(ens.member_projections_) == set(ens.trained_models_)
    assert {d["projection"]["mode"] for d in ens.member_projections_.values()} == {"fold_median"}
    af = fitted["cm"].features["af_raw"]
    assert af.isna().any() and (af == 0).any() and (af > 0).any()      # all three states reach the members


def test_a_fresh_process_reproduces_the_predictions_bit_for_bit(fitted):
    stdout, out = _fresh(fitted, fitted["ident"], "by-binding", "fresh.npy")
    assert "SCORED" in stdout
    fresh = np.load(out)
    assert fresh.shape == fitted["proba"].shape and np.array_equal(fresh.view(np.uint64), fitted["proba"].view(np.uint64))


def test_a_fresh_process_refuses_a_matrix_prepared_under_another_policy(fitted):
    arm_b = A.AFSelectionPolicy(unknown_base_treatment=S.POLICY.unknown_base_treatment, arm=A.Arm.WITHHOLD_REPORTED_ZEROS)
    other, _ = S.prepared(fitted["tmp"] / "arm_b", policy=arm_b)
    ident, payload = Q.write_canonical(other, fitted["store"])
    stdout, out = _fresh(fitted, ident, payload, "refused.npy")
    assert "REFUSED" in stdout and "canonical matrix differs" in stdout and not out.exists()
    with pytest.raises(PermissionError, match="canonical_identity"):
        Q.score_canonical(fitted["ens"], other)


def test_a_missing_edited_or_replaced_binding_is_refused(fitted):
    import copy

    ens = copy.copy(fitted["ens"])
    ens.source_binding_ = None
    with pytest.raises(PermissionError, match="records no source binding"):
        Q.score_canonical(ens, fitted["cm"])
    ens.source_binding_ = {**fitted["ens"].source_binding_, "roster": _roster()[:-1]}
    with pytest.raises(PermissionError, match="does not reproduce its digest"):
        Q.score_canonical(ens, fitted["cm"])
    with pytest.raises(ValueError, match="never replaced"):
        Q.bind_fit(fitted["ens"], fitted["cm"])


def test_an_audit_only_change_leaves_the_matrix_and_the_predictions_unchanged(fitted):
    frame = fitted["inp"]["cohort"].assign(source_db="re-curated")
    again = Q.prepare_qualified(frame, fitted["inp"]["population"], **fitted["inp"]["args"])
    assert again.identity == fitted["cm"].identity
    assert np.array_equal(Q.score_canonical(fitted["ens"], again), fitted["proba"])


def test_a_fresh_process_on_numpys_baseline_paths_prepares_the_same_identity(fitted):
    import numpy._core._multiarray_umath as m

    # The child disables every dispatched path this process has enabled AND every one this process's own environment already
    # disables (measured 2026-11-17: a suite run with NPY_DISABLE_CPU_FEATURES set left nothing "enabled" here, and an empty
    # value in the child then re-enabled them all).
    enabled = {k for k in m.__cpu_dispatch__ if m.__cpu_features__.get(k)}
    inherited = set(os.environ.get("NPY_DISABLE_CPU_FEATURES", "").replace(",", " ").split())
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(S.REPO / "src"), str(S.REPO)]),
           "PYTHONDONTWRITEBYTECODE": "1", "NPY_DISABLE_CPU_FEATURES": " ".join(sorted(enabled | inherited))}
    r = subprocess.run([sys.executable, "-c", DISPATCH, str(fitted["tmp"] / "dispatch_sources")], capture_output=True,
                       text=True, env=env, cwd=str(fitted["tmp"]), timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "DISPATCHED none" in r.stdout, r.stdout                       # the child ran on numpy's baseline paths only
    assert f"IDENTITY {fitted['cm'].identity}" in r.stdout, r.stdout


def test_scientific_training_stays_refused():
    from genomic_variant_classifier.models.fit_context import ExecutionProtocol, FitContext

    with pytest.raises(TypeError, match="direct construction is refused"):
        FitContext(ExecutionProtocol.SCIENTIFIC, None, None, "x")
    assert Q.TRAINING.startswith("refused")
