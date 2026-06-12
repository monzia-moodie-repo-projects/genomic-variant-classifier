"""Regression guard: variant_ensemble.py must stay ASCII so its log messages
cannot crash a cp1252 console (the smoke's non-fatal Greek-delta logging error).

Targets this one module on purpose -- evaluator.py keeps intentional box-drawing
Unicode in its report, which prints via train.py's reconfigured stdout.
Author: Monzia Moodie.
"""
from __future__ import annotations

from pathlib import Path

_VE = (
    Path(__file__).resolve().parents[2]
    / "src" / "genomic_variant_classifier" / "models" / "variant_ensemble.py"
)


def test_variant_ensemble_is_ascii():
    assert _VE.exists(), f"not found: {_VE}"
    data = _VE.read_bytes()
    offenders = [(i, b) for i, b in enumerate(data) if b > 127]
    assert not offenders, (
        f"{len(offenders)} non-ASCII byte(s) in variant_ensemble.py "
        f"(first few: {offenders[:5]}) -- log messages must be ASCII"
    )
