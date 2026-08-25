"""Report feature-drift readiness. An adapter -- it owns no scientific policy.

Created 2026-08-24.

WHAT THIS IS
============
A thin command-line adapter over
`genomic_variant_classifier.monitoring.drift_readiness`. It serialises a domain
result and projects it into whatever vocabulary a caller needs. It decides
nothing.

    repository/domain state
            |
            v
    current_feature_drift_readiness()
            |
            v
    DriftReadiness
            |
            v
    THIS ADAPTER  ->  canonical JSON
                  ->  GitHub Actions output lines

WHY AN ADAPTER AND NOT YAML
===========================
`.github/workflows/drift_monitor.yml` previously authored `drift_level` itself.
A workflow that authors semantic state is a second author of it, and the
combination `feature_drift_checked=false` beside `drift_level=none` was
constructible because three fields were written independently.

Here all four lines derive from ONE record, so that combination cannot arise.

WHY THIS DOES NOT WRITE TO $GITHUB_OUTPUT ITSELF
================================================
It prints. The caller redirects. A domain-adjacent command that opens a
GitHub-specific file becomes untestable without GitHub and unusable without it,
and this project has already recorded what happens when a scientific step can
only run inside one execution venue.

    python -m genomic_variant_classifier.monitoring.drift_readiness  -- not this
    python scripts/report_drift_readiness.py --format json
    python scripts/report_drift_readiness.py --format github >> "$GITHUB_OUTPUT"

EXIT CODES
==========
    0  a readiness verdict was produced and printed
    1  this program failed

The verdict itself is NOT encoded in the exit code. That inversion -- domain
state projected into a process exit rather than read out of one -- is the same
correction `EXIT_NOT_CHECKED` made inside `run_drift_monitor.py`, and repeating
the old shape here would reintroduce it one layer up. A caller wanting the
verdict reads the record.

Acronyms: JSON = JavaScript Object Notation; CLI = command-line interface.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_package_importable() -> None:
    """Make `src/` importable when this script is run from a checkout.

    Prepends rather than appends so a same-named module elsewhere on the path
    cannot shadow the package -- DOWNLOADSHADOW-1, observed when a probe run
    from a downloads directory bound the wrong `catalogue` module and then
    measured a defect it had itself created.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Report whether a feature-drift assessment may proceed.")
    parser.add_argument(
        "--format", choices=("json", "github"), default="json",
        help="canonical JSON record, or GitHub Actions output lines")
    args = parser.parse_args(argv)

    _ensure_package_importable()
    from genomic_variant_classifier.monitoring.drift_readiness import (
        current_feature_drift_readiness,
        render_github_output_lines,
        render_json,
        validate_document,
        as_document,
    )

    readiness = current_feature_drift_readiness()

    # The record is validated before it is projected. An adapter that emits an
    # unvalidated document is how two producers of one record diverge.
    validate_document(as_document(readiness))

    if args.format == "json":
        sys.stdout.write(render_json(readiness))
    else:
        for line in render_github_output_lines(readiness):
            sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
