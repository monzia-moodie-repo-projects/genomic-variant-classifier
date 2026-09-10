"""cleanup_apply: inspect and report. Application is UNAVAILABLE in this version.

Author: Monzia Moodie

WHAT THIS REPAIRS
=================
MEASURED 2026-09-08, executing the previous `scripts/cleanup_apply.py` with a
working directory outside any Git working tree:

    ### (F)  2 eligible (4.0B), 0 SKIPPED ###
      would-delete  2.0B  scripts/dump_thing.py
      would-delete  2.0B  scripts/patch_thing.py

Category F's condition was `ok = not tracked(p)`, and `tracked()` returned
False whenever `git ls-files` exited non-zero -- which it always does outside a
repository. Failed inspection became deletion eligibility. The docstring
"Refuses to delete a tracked path under any circumstance" was literally true
and operationally empty.

WHAT THIS DELIBERATELY DOES NOT DO
==================================
It does not delete. `--apply` returns a nonzero status BEFORE any discovery,
and this module imports no executor: there is no destructive code path to
reach, dormant or otherwise.

That is a deliberate compatibility change, not an oversight. Application
requires an approved configuration owner and an authorization binding that are
not installed. Supplying a digest on the command line would be manufacturing
approval, so no such option exists.

    inspection  : completed
    application : unavailable
    authorization: not established
    deletion attempts: 0

ACTIVATION OBLIGATION
=====================
This disabled state is an intermediate release with a recorded obligation, not
a finished feature. Activation is a separately reviewed unit and requires ALL
of: the approved configuration owner installed; production composition
obtaining approval from the established binding mechanism; every destructive
route passing through the executor; category eligibility and supporting
evidence checked; repository and runtime protection enforced; the actual
command-line interface passing end-to-end positive and negative tests; and
platform behaviour qualified for the supported environment.

A configuration file existing, or a digest being supplied, is not activation.

EXIT STATUS
  0  inspection completed
  2  repository selection or inspection failed
  3  an application request was refused. NOT a successful dry run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NO PATH BOOTSTRAP.
#
# MEASURED 2026-09-10: the previous version computed
# `Path(__file__).resolve().parent.parent / "src"`, which is correct for a
# script in scripts/ and WRONG for one in scripts/forensics/ -- where this
# file actually lives. The pending-unit fit census caught the destination
# error, and the bootstrap had inherited it.
#
# The repository's own convention has no bootstrap: scripts/
# retire_backup_artifacts.py imports
# `from genomic_variant_classifier.repository_hygiene import backup_artifacts`
# directly, relying on the installed project. Following it removes the depth
# dependency rather than correcting the depth.

from genomic_variant_classifier.repository_hygiene.cleanup_categories import (
    inspect_cleanup_candidates, render_proposal)                # noqa: E402
from genomic_variant_classifier.repository_hygiene.repository_inspection import (
    InspectionError, RepositorySelectionError, select_repository)  # noqa: E402

APPLICATION_UNAVAILABLE = 3
INSPECTION_FAILED = 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect cleanup candidates. Application is unavailable.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--apply", action="store_true",
                        help="refused: application is unavailable")
    args = parser.parse_args(argv)

    # BEFORE ANY OTHER WORK. Discovery must not run for an application
    # request, so the refusal cannot be mistaken for a completed dry run and
    # no code capable of mutation is reached.
    if args.apply:
        print("Cleanup application is unavailable in this version: the "
              "approved configuration and authorization provider is not "
              "installed. No cleanup action was attempted.", file=sys.stderr)
        return APPLICATION_UNAVAILABLE

    try:
        repository = select_repository(args.repo_root)
    except RepositorySelectionError as exc:
        print("Repository selection failed: {}".format(exc), file=sys.stderr)
        return INSPECTION_FAILED

    # EXPECTED inspection failures are caught deliberately. A programming
    # error must remain an unsuccessful execution, not become an empty
    # category, so no bare `except Exception` appears here.
    try:
        proposal = inspect_cleanup_candidates(repository)
    except InspectionError as exc:
        print("Inspection failed: {}".format(exc), file=sys.stderr)
        return INSPECTION_FAILED

    render_proposal(proposal)
    if proposal.inspection_status != "completed":
        print("Inspection was INCOMPLETE: {} failure(s). Zero candidates in "
              "an incomplete category is not a measured absence.".format(
                  proposal.inspection_failures), file=sys.stderr)
        return INSPECTION_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
