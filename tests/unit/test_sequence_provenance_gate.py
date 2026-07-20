"""The ensemble refuses sequence it cannot vouch for.

Before 2026-07-20 the check asked one question -- "is X_seq None?" -- of one parameter. It
never saw X_seq_cal_ext, so a run could fit cnn_1d on real sequence and calibrate it on
fabricated sequence in silence. It could not distinguish a real window from an invented one,
nor a verified attachment from one whose `ok` mask seq_window_join fabricated as all-True.

Every refusal below has a matching NEGATIVE CONTROL showing the same input passes when no
sequence model is active. A gate that refuses everything is not a gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.data.seq_window_join import (
    OK_COL, REF_WIN_COL, ALT_WIN_COL, attach_delta_windows,
)
from genomic_variant_classifier.models.variant_ensemble import (
    EnsembleConfig, SequenceWindows, VariantEnsemble, SEQUENCE_MODELS,
)


def _frame(n: int, ok=True) -> pd.DataFrame:
    return pd.DataFrame({
        REF_WIN_COL: ["A" * 101] * n,
        ALT_WIN_COL: ["C" * 101] * n,
        OK_COL: [ok] * n if isinstance(ok, bool) else ok,
    })


def _att(n: int = 200, ok=True):
    return attach_delta_windows(_frame(n, ok))


def _ens(**kw) -> VariantEnsemble:
    return VariantEnsemble(EnsembleConfig(**kw))


def _gate(ens, inputs, models=None, method="fit"):
    return ens._require_sequence_windows(
        inputs, models if models is not None else ens.base_estimators, method)


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------

def test_a_real_attachment_satisfies_the_protocol():
    assert isinstance(_att(), SequenceWindows)


def test_a_bare_frame_does_not_satisfy_the_protocol():
    assert not isinstance(_frame(5), SequenceWindows)
    assert not isinstance(pd.Series(["A" * 101] * 5), SequenceWindows)


def test_the_protocol_is_structural_not_nominal():
    """variant_ensemble must not IMPORT the data layer; any matching shape qualifies.

    PARSED, NOT STRING-MATCHED. The first version of this test read

        assert "import WindowAttachment" not in src

    and failed -- on the SequenceWindows docstring, which says "This module does not import
    WindowAttachment". It fired on the prose asserting the very property it checks.

    Two checks in the installer that produced this file were converted from matching to
    parsing an hour earlier for exactly this reason, and this one was left behind. An import
    is an ast node; a mention is not. Thirteenth occurrence of this failure mode in one
    session, and the durable form has never changed: parse, do not match.
    """
    import ast
    import pathlib

    import genomic_variant_classifier.models.variant_ensemble as VE

    tree = ast.parse(pathlib.Path(VE.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)

    pkg = "genomic_variant_classifier.data"
    leaks = sorted(m for m in modules if m == pkg or m.startswith(pkg + "."))
    assert not leaks, (
        f"variant_ensemble imports {leaks} from the data layer. The SequenceWindows protocol "
        f"exists so the models layer does not have to depend on the data layer; a real import "
        f"means the structural-typing argument no longer holds."
    )


# ---------------------------------------------------------------------------
# The four refusals
# ---------------------------------------------------------------------------

def test_none_is_refused_when_a_sequence_model_is_active():
    ens = _ens()
    with pytest.raises(ValueError, match="received None"):
        _gate(ens, {"X_seq": None})


def test_a_bare_frame_is_refused_as_provenance_less():
    ens = _ens()
    with pytest.raises(ValueError, match="bare DataFrame"):
        _gate(ens, {"X_seq": _frame(200)})


def test_a_series_is_refused_and_named_as_a_series():
    ens = _ens()
    with pytest.raises(ValueError, match="bare Series"):
        _gate(ens, {"X_seq": pd.Series(["A" * 101] * 200)})


def test_unverified_provenance_is_refused_even_when_fully_usable():
    """The trap: a "rows" attachment reports 100% usable because `ok` was assumed."""
    att = attach_delta_windows(pd.DataFrame({
        REF_WIN_COL: ["A" * 101] * 200, ALT_WIN_COL: ["C" * 101] * 200,
    }))
    assert att.provenance == "rows" and att.n_usable == 200
    with pytest.raises(ValueError, match="UNVERIFIED provenance"):
        _gate(_ens(), {"X_seq": att})


def test_too_few_usable_rows_is_refused():
    ens = _ens(seq_min_usable_rows=500, seq_min_usable_fraction=0.0)
    with pytest.raises(ValueError, match="below the floor"):
        _gate(ens, {"X_seq": _att(200)})


def test_too_low_a_usable_fraction_is_refused():
    ok = [True] * 150 + [False] * 50          # 75% usable
    ens = _ens(seq_min_usable_rows=1, seq_min_usable_fraction=0.95)
    with pytest.raises(ValueError, match="below the floor of 0.95"):
        _gate(ens, {"X_seq": _att(200, ok=ok)})


# ---------------------------------------------------------------------------
# X_seq_cal_ext -- the hole this closes
# ---------------------------------------------------------------------------

def test_the_calibration_partition_is_checked_too():
    """THE REGRESSION THIS FILE EXISTS FOR.

    Between ff97c34 (2026-07-19) and 2026-07-20 the check ran on X_seq alone. A run with real
    train windows and a placeholder calibration partition passed, fitting cnn_1d on real
    sequence and calibrating it on fabricated sequence.
    """
    with pytest.raises(ValueError, match="X_seq_cal_ext"):
        _gate(_ens(), {"X_seq": _att(), "X_seq_cal_ext": None})


def test_the_refusal_names_the_offending_parameter_not_just_the_method():
    with pytest.raises(ValueError, match="`X_seq_cal_ext`"):
        _gate(_ens(), {"X_seq": _att(), "X_seq_cal_ext": _frame(10)})


def test_fit_does_not_demand_a_calibration_sequence_when_there_is_no_partition():
    """THE REGRESSION THIS PAIR EXISTS FOR, and it is the opposite of the one above.

    X_seq_cal_ext=None is the NORMAL case: it means no external calibration partition, which
    is what scripts/train.py:561 and run_phase2_eval.py:590 both do. The first version of this
    gate demanded it unconditionally and would have refused every non-v2 run.

    Asserted through fit() rather than through the gate directly, because the conditional
    lives at the CALL SITE -- checking the gate in isolation would pass while production broke.
    The failure this pins is a ValueError mentioning X_seq_cal_ext; any other exception from
    deeper in fit() is not this test's concern.
    """
    ens = _ens()
    ens.base_estimators = {k: v for k, v in ens.base_estimators.items()
                           if k not in SEQUENCE_MODELS}
    tab = pd.DataFrame(np.random.default_rng(0).random((40, 3)), columns=list("abc"))
    try:
        ens.fit(tab, None, pd.Series([0, 1] * 20))
    except Exception as exc:                                   # noqa: BLE001
        assert "X_seq_cal_ext" not in str(exc), (
            "fit() refused a missing calibration sequence when no calibration partition was "
            "supplied; the gate is demanding an input that is not in play"
        )


# ---------------------------------------------------------------------------
# Negative controls -- the gate must NOT fire without a sequence model
# ---------------------------------------------------------------------------

def test_nothing_is_refused_when_no_sequence_model_is_active():
    ens = _ens()
    for name in list(ens.base_estimators):
        if name in SEQUENCE_MODELS:
            ens.base_estimators.pop(name)
    out = _gate(ens, {"X_seq": None, "X_seq_cal_ext": _frame(3)})
    assert out["X_seq"] is None
    assert isinstance(out["X_seq_cal_ext"], pd.DataFrame)


def test_a_good_attachment_passes_and_is_resolved_to_its_frame():
    out = _gate(_ens(), {"X_seq": _att()})
    assert isinstance(out["X_seq"], pd.DataFrame)
    assert list(out["X_seq"].columns) == [REF_WIN_COL, ALT_WIN_COL]


def test_thresholds_are_per_run_overridable_so_a_smoke_run_is_not_refused():
    """EnsembleConfig.zero_variance_min_rows idiom: armed by default, adjustable per run."""
    tiny = _att(20)
    with pytest.raises(ValueError):
        _gate(_ens(), {"X_seq": tiny})
    out = _gate(_ens(seq_min_usable_rows=10), {"X_seq": tiny})
    assert isinstance(out["X_seq"], pd.DataFrame)


def test_defaults_are_armed():
    cfg = EnsembleConfig()
    assert cfg.seq_require_verified_provenance is True
    assert cfg.seq_min_usable_rows == 100
    assert cfg.seq_min_usable_fraction == 0.95


# ---------------------------------------------------------------------------
# The message
# ---------------------------------------------------------------------------

def test_the_refusal_states_the_remedy_and_that_no_compute_was_spent():
    with pytest.raises(ValueError) as e:
        _gate(_ens(), {"X_seq": None})
    msg = str(e.value)
    assert "attach_delta_windows" in msg
    assert "not its `.windows`" in msg
    assert "base_estimators.pop" in msg
    assert "--skip-cnn" in msg
    assert "no compute was spent" in msg


def test_the_refusal_names_every_active_sequence_model():
    with pytest.raises(ValueError) as e:
        _gate(_ens(), {"X_seq": None})
    assert "cnn_1d" in str(e.value)
