"""Test ScalableSVM._resolve_n_jobs caps joblib workers safely (WinError 1450 fix).

cpu_count is mocked so results are deterministic regardless of the test host
(a 1-core box would otherwise mask the malformed-env bug).
"""

from __future__ import annotations

import os

from genomic_variant_classifier.models.scalable_svm import ScalableSVM


def _bare(n_jobs=-1):
    # bypass __init__ (its signature has several params); only n_jobs is needed
    s = ScalableSVM.__new__(ScalableSVM)
    s.n_jobs = n_jobs
    return s


def test_default_cap(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.delenv("GVC_SVM_NJOBS", raising=False)
    s = _bare()
    assert s._resolve_n_jobs(15) == 4    # default cap on an 8-core box
    assert s._resolve_n_jobs(2) == 2     # never exceed n_tasks
    assert s._resolve_n_jobs(1) == 1


def test_env_override(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    s = _bare()
    monkeypatch.setenv("GVC_SVM_NJOBS", "1")
    assert s._resolve_n_jobs(15) == 1
    monkeypatch.setenv("GVC_SVM_NJOBS", "6")
    assert s._resolve_n_jobs(15) == 6    # valid value may raise the cap (bounded by cpu)
    monkeypatch.setenv("GVC_SVM_NJOBS", "garbage")
    assert s._resolve_n_jobs(15) == 4    # malformed -> default cap, NOT uncapped (the bug)


def test_explicit_positive_n_jobs(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.delenv("GVC_SVM_NJOBS", raising=False)
    s = _bare(n_jobs=2)
    assert s._resolve_n_jobs(15) == 2    # explicit 2 stays under default cap 4
    assert s._resolve_n_jobs(1) == 1
