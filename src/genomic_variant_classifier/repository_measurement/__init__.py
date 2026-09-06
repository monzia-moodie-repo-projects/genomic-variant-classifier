"""Epistemic semantics of repository inspections. ADR-0005.

This package answers one question: what does a repository observation actually
establish? It is the typed owner of the `Observation` role ADR-0001 declared on
2026-08-21 and never implemented.

WHAT IT OWNS
    declare what repository population was inspected
    describe how completely it was inspected
    state what the resulting evidence licenses

WHAT IT DOES NOT OWN
    finding lifecycle, finding status, carried-item lifecycle, project state,
    decision-record status, scientific-data provenance, durable-record
    placement, retention, publication, transaction semantics, Git
    orchestration, and the analysis mechanisms themselves.

    measurement evidence is not state authority

NO RE-EXPORTS. MEASURED 2026-09-05 across the neighbouring infrastructure
packages: `transactions` 0 re-export lines, `repository_records` 0, `paths` 0.
`provenance` has 1 and `conformal` 2, and both are domain packages rather than
infrastructure. Consumers import from modules explicitly. This is the
repository's demonstrated convention, not an invented one.

STANDARD LIBRARY ONLY. This package must remain importable when almost
everything else is broken, because diagnostic and repair tools need it exactly
then. `tests/unit/test_repository_measurement_isolation.py` enforces that by
parsing the sources.

Author: Monzia Moodie
"""
from __future__ import annotations
