"""Conformal prediction subpackage: from-scratch numpy, no external dependency.

Modules
-------
scores      nonconformity scores -- Least Ambiguous set-valued Classifier (LAC),
            Adaptive Prediction Sets (APS), Regularized APS (RAPS)
split       split conformal, and the finite-sample conformal_quantile used by
            every other module here
mondrian    class-conditional (Mondrian) conformal -- coverage per class rather
            than on average, which is what marginal coverage hides
grouped     group-conditional conformal over arbitrary partitions
coverage    coverage and efficiency diagnostics
calibrate   probability calibration helpers
ordinal     ordinal conformal for the American College of Medical Genetics and
            Genomics / Association for Molecular Pathology five-tier scale;
            every prediction set is a CONTIGUOUS interval of adjacent tiers

Every module on disk must appear in the import below and in __all__.
tests/unit/test_conformal_package_exports.py enforces that by walking this
directory and asserting each module is reachable as a package attribute. The
list went stale twice -- once for calibrate, once for ordinal -- because nothing
connected "a file exists here" to "its name appears on that line". Now something
does, and adding a submodule without exporting it turns the suite red.
"""
from . import (  # noqa: F401
    calibrate,
    coverage,
    grouped,
    mondrian,
    ordinal,
    scores,
    split,
)

__all__ = [
    "calibrate",
    "coverage",
    "grouped",
    "mondrian",
    "ordinal",
    "scores",
    "split",
]
