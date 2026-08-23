"""Durable machine evidence is not documentation, and not runtime state.

ADR-0004. This package is the OPERATIONAL AUTHORITY for the `records/` plane.

WHY THIS EXISTS AS CODE AND NOT AS PROSE
========================================
ADR-0004 declines to carry the role-to-root mapping, and says why: a record
stating `role X -> records/foo` while code says `records/bar` is the identical
defect this programme spent its length removing -- a README mirroring internal
state, four installers each carrying a private notion of "neutral", nine
attestations in three shapes under one version.

    One semantic concept, one typed owner.
    The mapping that performs the placement must be the mapping that is executed.

WHAT WAS MEASURED, 2026-08-22
=============================
Twenty-six machine-evidence documents were committed across SIX directories --
docs/audits/evidence/2026-07-09 and 2026-07-24, docs/incidents,
docs/measurements, docs/migrations, docs/verified -- while eleven install
attestations lived outside version control entirely, cited by eleven commit
messages. `docs/archive/`, assigned DEVELOPMENT_NOTEBOOK, held three files, none
of them documentation: a stranded git worktree recovery artifact.

The cause was not carelessness. Machine records never acquired an architectural
layer, so each subsystem filed them under whichever documentation noun was
convenient. Naming a better noun would have preserved the category error.

FOUR ORTHOGONAL AXES
====================
    ArtifactRole              what kind of record is this, hence where it lives
    DisclosureClass           may these exact bytes be published
    PreservationDisposition   may this artifact be preserved verbatim at all
    RetentionClass            how long is it kept

Answering one does not answer another. An artifact may be an installation
attestation, publishable verbatim, admitted with a defect note, and permanent --
four independent facts.

Author: Monzia Moodie
"""
from __future__ import annotations
