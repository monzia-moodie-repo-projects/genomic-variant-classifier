"""Tests for train.py's UTF-8 stdio guard (_force_utf8_stdio).

Confirms the helper tolerates streams without .reconfigure and streams whose
.reconfigure raises, and calls reconfigure(utf-8, replace) on real ones -- so
the evaluator's box-drawing report separator (U+2500) and the blend-AUROC
delta log can't crash a Windows cp1252 console. Author: Monzia Moodie.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import train as train_mod  # noqa: E402


def test_missing_reconfigure_tolerated():
    class NoRecon:
        pass

    train_mod._force_utf8_stdio([NoRecon()])  # must not raise


def test_raising_reconfigure_tolerated():
    class Raising:
        def reconfigure(self, **kwargs):
            raise ValueError("boom")

    train_mod._force_utf8_stdio([Raising()])  # must not raise


def test_real_stream_reconfigured_utf8_replace():
    calls = []

    class Recordable:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    train_mod._force_utf8_stdio([Recordable()])
    assert calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_default_streams_no_raise():
    train_mod._force_utf8_stdio()  # real stdout/stderr; idempotent, must not raise
