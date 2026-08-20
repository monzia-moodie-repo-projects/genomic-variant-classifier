"""Rollback state is transactional and ephemeral, and lives outside the tree.

    A transaction may temporarily own a rollback state. The repository never
    does. Git owns history; the incident system owns evidence; secret material
    expires.

INSTALLER-TRANSACTION-1. This package exists because `<target>.bak_<timestamp>`
was asked to serve four incompatible purposes at once -- crash recovery,
undo, historical archive and security evidence -- and served none of them
reliably.

MEASURED 2026-08-19, in two sweeps:

    148 artefacts, 17,640,928 bytes   matching `*.bak_*`
    107 artefacts,  2,828,345 bytes   matching `*.bak`, `*.orig`, `*.rej`

The second sweep existed because the first tool scanned ONE shape of four and
reported "remaining .bak_* artefact(s): 0". One of the 148 was a
credential-bearing `.env` backup; the manifest recording its retirement was
later overwritten by a routine run, because it was addressed by a name the next
event reused.

Three defects, one root: rollback state inside the repository, under names no
single filter reliably covers, cleaned up only when someone remembered.

THE LIFECYCLE

    VERIFY -> PLAN -> SNAPSHOT -> APPLY -> POST-WRITE VERIFY -> GATE -> COMMIT

and on any failure

    FAILURE -> ROLLBACK -> VERIFY RESTORATION -> DESTROY THE JOURNAL

THE INVARIANTS

    success       the repository keeps the changes and NOTHING else
    failure       the repository is byte-identical to how it was found
    interruption  a journal survives OUTSIDE the repository, in a non-terminal
                  state, and the next invocation can discover it

WHAT THIS PACKAGE DOES NOT REBUILD
Persistence is JsonStateStore -- atomic write via mkstemp, fsync and os.replace
in the same directory, schema identification, a generation counter, and a load
that RAISES on damage rather than reporting emptiness. Location is
RuntimePaths.transaction_journal, the machine-scoped fifth path domain.

A third `_atomic_write` was deliberately not added: representation_artifact.py
already documents its copy of the idiom from RunArtifactWriter, and
consolidating those two is a separate unit.

Author: Monzia Moodie
"""

from __future__ import annotations
