"""The reports namespace must not activate the reporting capability.

REPORTS-EAGER-IMPORT-1
======================
report_generator.py imports seaborn and jinja2 UNGUARDED (lines 45 and 46) and
executes sns.set_style() at MODULE SCOPE (line 57). Neither package is in
requirements-api.lock, while matplotlib -- two lines above them in the same
import block -- is present at line 24.

So importing `genomic_variant_classifier.reports` used to require both packages
and mutate process-wide plotting configuration, merely because something
traversed the namespace.

    Importing a namespace must not activate an optional capability.

Latent rather than firing: `python -X importtime -c "import api.main"` shows no
seaborn, jinja2, reports or report_generator in the API's import graph. One
import statement away from breaking the image.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


#: TWO probes, split by COST -- measured, not guessed.
#:
#:     lazy namespace import         0.03 s
#:     full report_generator import  1.91 s
#:
#: A 64x difference. An earlier version used ONE probe that resolved
#: ReportGenerator, so every property paid the expensive path and the file took
#: 11.17 s -- SLOWER than the eight separate subprocesses it replaced. The
#: right axis is cost, not count.
#:
#: A fresh process is still REQUIRED: in this one the suite has already
#: imported half the world, so sys.modules here proves nothing about what a
#: given import pulls in.

_WATCH = ("'seaborn', 'jinja2', "
          "'genomic_variant_classifier.reports.report_generator'")

#: CHEAP: namespace only. Never resolves a re-exported name.
_PROBE_NAMESPACE = """
import json, sys
import genomic_variant_classifier.reports as R
print(json.dumps({
    'loaded': [m for m in (%s) if m in sys.modules],
    'dir_has_all': all(n in dir(R) for n in R.__all__),
    'all_names': list(R.__all__),
}))
""" % _WATCH

#: EXPENSIVE: resolves a name, so it pays the full import. Everything that
#: needs the module resolved is asked HERE, once.
_PROBE_RESOLVED = """
import json, sys
import genomic_variant_classifier.reports as R
try:
    R.definitely_not_a_real_name
    unknown_raises = False
except AttributeError as exc:
    unknown_raises = 'definitely_not_a_real_name' in str(exc)
gen = R.ReportGenerator
from genomic_variant_classifier.reports import report_generator as RG
print(json.dumps({
    'loaded': [m for m in (%s) if m in sys.modules],
    'attribute_resolved': gen is not None,
    'unknown_raises': unknown_raises,
    'identity': all(getattr(R, n) is getattr(RG, n) for n in R.__all__),
}))
""" % _WATCH

_CACHE = {}


def _probe(which: str) -> dict:
    """Run one probe in a fresh interpreter, once per session."""
    if which not in _CACHE:
        code = _PROBE_NAMESPACE if which == "namespace" else _PROBE_RESOLVED
        out = subprocess.run([sys.executable, "-B", "-c", code],
                             capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            pytest.fail("the {} probe failed:\n{}".format(
                which, out.stderr[-1200:]))
        try:
            _CACHE[which] = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception as exc:                               # noqa: BLE001
            pytest.fail("could not parse the {} probe ({}): {!r}".format(
                which, exc, out.stdout[-400:]))
    return _CACHE[which]


def test_importing_the_namespace_does_NOT_load_report_generator():
    """THE DEFECT, as a test. Measured in a FRESH interpreter."""
    loaded = _probe("namespace")["loaded"]
    assert "genomic_variant_classifier.reports.report_generator" not in loaded, (
        "importing the reports namespace pulled in {} -- the capability was "
        "activated by traversal".format(loaded))


def test_importing_the_namespace_does_NOT_load_seaborn_or_jinja2():
    """Neither is in requirements-api.lock. If the API ever imports this
    package, an eager load would break the image at import time."""
    loaded = _probe("namespace")["loaded"]
    for pkg in ("seaborn", "jinja2"):
        assert pkg not in loaded, (
            "{} was imported by traversing the reports namespace".format(pkg))


def test_accessing_a_re_exported_name_DOES_load_the_module():
    """Laziness must defer, not break. Asking for the name resolves it."""
    r = _probe("resolved")
    assert r["attribute_resolved"] is True
    assert "genomic_variant_classifier.reports.report_generator" in r["loaded"]


def test_the_submodule_import_path_still_works():
    """Every measured consumer uses this form; it must be untouched.

    Run in THIS process deliberately -- the question is whether the import
    works, not what else it loads, so a subprocess would buy nothing.
    """
    from genomic_variant_classifier.reports.report_generator import (
        ReportGenerator, bootstrap_metric)
    assert callable(bootstrap_metric)
    assert ReportGenerator is not None


def test_every_advertised_name_actually_resolves():
    """__all__ must not promise a name the module cannot supply -- that would
    be a lazy re-export that fails only when someone finally uses it."""
    from genomic_variant_classifier import reports
    for name in reports.__all__:
        assert getattr(reports, name) is not None, name


def test_an_unknown_attribute_raises_AttributeError():
    from genomic_variant_classifier import reports
    with pytest.raises(AttributeError) as exc:
        reports.definitely_not_a_real_name
    assert "definitely_not_a_real_name" in str(exc.value)
    assert _probe("resolved")["unknown_raises"] is True


def test_dir_reports_the_lazy_names():
    """dir() must stay honest, or tooling and tab-completion silently lose the
    public interface."""
    from genomic_variant_classifier import reports
    listed = dir(reports)
    for name in reports.__all__:
        assert name in listed, name
    assert _probe("namespace")["dir_has_all"] is True


def test_the_re_exported_objects_are_the_SAME_objects():
    """A lazy proxy that returned copies would break identity comparisons and
    isinstance checks in ways that surface far from here."""
    from genomic_variant_classifier import reports
    from genomic_variant_classifier.reports import report_generator
    for name in reports.__all__:
        assert getattr(reports, name) is getattr(report_generator, name), name
    assert _probe("resolved")["identity"] is True
