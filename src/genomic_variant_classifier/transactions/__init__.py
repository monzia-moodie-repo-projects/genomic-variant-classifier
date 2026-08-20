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


STEP 3B: CRASH CONSISTENCY (2026-08-20)
The step-3 primitive passed 38 tests and 12 sabotage mutations, and every one
of those mutations was EXCEPTION-driven. Not one killed a process. Two defects
were then found by inspection and DEMONSTRATED before repair:

    WRITE-AHEAD VIOLATION. patch() captured the preimage, wrote the new bytes,
    and only THEN persisted the target record. Reproduced by capturing,
    writing, and dropping the object: the file read MUTATED, the preimage
    existed on disk, and the manifest's target list was EMPTY. The journal was
    discoverable and unrecoverable.

    A FAILED ROLLBACK RECORDED ITSELF AS SUCCESSFUL. ROLLED_BACK was set before
    the failures were examined, so a corrupted preimage left the repository
    unrestored, the journal retained, the state terminal, discovery blind, and
    a retry a no-op. Four bad properties from one misordered assignment.

What changed: write-ahead ordering with fsynced preimages; ROLLING_BACK and
RECOVERY_REQUIRED so a failed rollback stays discoverable and retryable;
recover_transaction(), which reconstructs from the manifest and preimages
ALONE and therefore works in a process that never saw the transaction object;
clean-tree enforcement, because HEAD not moving says nothing about a concurrent
editor; and unresolved journals blocking a new transaction.

SECRET TARGETS ARE NOW REFUSED BY DEFAULT
One abstraction cannot promise both "no persistent secret preimage" and
"arbitrary secret mutations are crash-recoverable" without another trusted
store. Credential provisioning is a different AUTHORITY -- environment
injection, an operating-system credential store, a hosted secret store -- not a
special case of a source-tree patch. That is the same principle the path
domains follow.

    An exception-safe transaction is not necessarily a crash-safe transaction.

Verified across five real SIGTERM kill points -- after preparation, after
write-ahead, after mutation, after the mutation mark, and during the gate --
with recovery performed in a THIRD process each time and zero journals left
behind.

Author: Monzia Moodie
"""

from __future__ import annotations
