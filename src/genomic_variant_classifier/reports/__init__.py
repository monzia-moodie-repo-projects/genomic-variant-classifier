"""
src/genomic_variant_classifier/reports
===========
Report generation package for the Genomic Variant Classifier.

CHANGES FROM PHASE 1:
  - This __init__.py did not exist; src/reports/ was not a Python package,
    so `from genomic_variant_classifier.reports.report_generator import ...` raised ModuleNotFoundError
    even after the module was written to disk (Issue D fixed).

REPORTS-EAGER-IMPORT-1 (2026-08-13): the re-export is now LAZY.

    Importing a namespace must not activate an optional capability.

This module previously imported report_generator at package-import time. That
module imports seaborn and jinja2 UNGUARDED and executes sns.set_style() at
module scope, so merely touching `genomic_variant_classifier.reports` required
both packages and mutated process-wide plotting configuration.

Neither is in requirements-api.lock. The API image would therefore fail at
import on any path reaching this package -- latent today, because
`python -X importtime -c "import api.main"` shows api.main does NOT reach
reports/, but one import statement away from firing.

MEASURED BEFORE CHANGING IT. Every consumer imports the SUBMODULE directly:

    tests/unit/test_bootstrap_reconciliation.py:123, :217
        from genomic_variant_classifier.reports.report_generator import bootstrap_metric
    tests/unit/test_bootstrap_reconciliation.py:231
        from genomic_variant_classifier.reports import report_generator
    tests/unit/test_core.py:1061, :1068, :1079, :1103, :1187
        from genomic_variant_classifier.reports.report_generator import ...

ZERO callers use the re-exported names. The eight-name import block served no
caller and guaranteed the eager activation, so making it lazy preserves the
public interface for a future caller while removing a cost nobody was paying
for deliberately.

The original defect this file was created for -- src/reports/ not being a
package -- is fixed by the file EXISTING, not by its re-exports.

Author: Monzia Moodie
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ReportGenerator",
    "ValidationMetrics",
    "bootstrap_metric",
    "compute_variant_phenotype_association",
    "plot_calibration",
    "plot_feature_importance",
    "plot_pr_curves",
    "plot_roc_curves",
]


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name on first access, not at package import.

    PEP 562 module __getattr__. Python calls this only when normal attribute
    lookup fails, so it runs at most once per name per process and costs
    nothing on the path that does not use it.
    """
    if name in __all__:
        from genomic_variant_classifier.reports import report_generator
        return getattr(report_generator, name)
    raise AttributeError(
        "module {!r} has no attribute {!r}".format(__name__, name))


def __dir__() -> list:
    """Keep tab-completion and dir() honest despite the lazy lookup."""
    return sorted(set(__all__) | set(globals()))
