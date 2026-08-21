"""The repository contains no installer residue.

INSTALLER-TRANSACTION-1, step 4. The first payload installed by a
RepositoryTransaction rather than by a script writing files directly.

    An installer is one atomic repository state transition. There are no
    transactional "payload files" and nontransactional "bookkeeping files".

WHAT THIS INVARIANT IS FOR
MEASURED across 2026-08-19 and 2026-08-20, in three sweeps:

    148 artefacts matching `*.bak_*`            17,640,928 bytes
    107 artefacts matching `*.bak/.orig/.rej`    2,828,345 bytes
     25 artefacts in the final hygiene sweep      1,663,985 bytes

Every one was written by an installer that backed a file up before editing it
and never removed the backup on success. What was designed as a rollback
IMPLEMENTATION DETAIL became a permanent archival system by omission, and it
was invisible to `git status` because `.gitignore` carries `*.bak_*`.

The transaction primitive keeps its rollback state OUTSIDE the repository, in
a machine-scoped journal, and destroys it on commit. This test is the assertion
that the practice actually holds -- that no installer, however carefully
written, has quietly reintroduced the habit.

THE TEST CARRIES NO POLICY
It states the repository property and nothing else. Every judgement about what
counts as residue -- which filename shapes, which directories are declared
scratch, when a backup is a relocated predecessor rather than detritus -- lives
in repository_hygiene.backup_artifacts.

That matters because the alternative is four authorities for one ontology:

    the retirement tool's policy
    the iterator's policy
    this test's exception list
    each installer's own idea

This project has already repaired precisely that class of divergence twice --
once for the pattern lists, once for the classification vocabulary. An
exception list here would rebuild it.

WHY THE JOURNAL NEEDS NO EXCEPTION
MEASURED 2026-08-20: the transaction journal resolves to
<LOCALAPPDATA>/GenomicVariantClassifier/transactions, outside the repository.
So repository hygiene and transaction hygiene are two clean invariants rather
than one detector taught about both concepts:

    repository hygiene   no backup/temp residue in the tree
    transaction hygiene  no incomplete journals

Author: Monzia Moodie
"""
from __future__ import annotations

import os
from pathlib import Path

from genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths
from genomic_variant_classifier.repository_hygiene import backup_artifacts as H
from genomic_variant_classifier.transactions.repository_transaction import (
    incomplete_transactions,
)

_REPO = Path(__file__).resolve().parents[2]


def test_the_repository_contains_no_backup_detritus():
    """THE INVARIANT.

    A failure here names the files. The remedy is
    `python scripts/retire_backup_artifacts.py --repo-root . --manifest
    docs/incidents/BACKUP_RETIREMENT_<date>_<event>.json --apply`, which
    classifies each artefact and retires only what it can prove is redundant.
    """
    detritus = sorted(H.iter_repository_detritus(_REPO))
    assert detritus == [], (
        "{} backup-shaped file(s) are present in the repository: {}"
        .format(len(detritus), detritus))


def test_the_invariant_is_capable_of_FAILING(tmp_path):
    """A check that cannot fail is not a check.

    MEASURED 2026-08-20: an earlier iterator reported zero detritus on this
    repository while EIGHT ordinary artefacts were present, because a
    relocation false positive excluded them all. It passed, and it was
    VACUOUS.

    So the invariant is exercised against a repository built to violate it.
    """
    repo = tmp_path / "dirty"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_bytes(b"x = 1\n")
    (repo / "src" / "mod.py.bak_2026-01-01_000000").write_bytes(b"x = 0\n")
    found = sorted(H.iter_repository_detritus(repo))
    assert found == ["src/mod.py.bak_2026-01-01_000000"], found


#: The transaction currently installing this suite, if any.
#:
#: INVARIANT-SELF-REFERENCE-1 (2026-08-21). The acceptance gate runs INSIDE the
#: apply transaction -- deliberately, so that a failure still has a rollback --
#: which means that transaction's own journal exists and is non-terminal while
#: this test looks at it.
#:
#: MEASURED: the first transactional install failed on exactly this, with
#: 4976 passed / 10 skipped / 1 failed. The repository-hygiene assertion held
#: throughout; only the journal assertion was false, and it was false because
#: the install was in flight.
#:
#: "No incomplete journals" is a QUIESCENT-REPOSITORY property. Asserting it
#: during an install asks whether a thing is finished while it is happening.
#:
#: The runner exports its identifier, following the GVC_ environment convention
#: already used by runtime_paths (GVC_PROJECT_ROOT, GVC_ARTIFACT_ROOT,
#: GVC_STATE_ROOT, GVC_CACHE_ROOT). The exclusion is therefore NARROW and
#: EXPLICIT: exactly one named transaction, never a blanket tolerance.
ACTIVE_TRANSACTION_ENV = "GVC_ACTIVE_TRANSACTION"


def test_no_incomplete_transaction_journals_remain():
    """TRANSACTION hygiene, which is a separate question from REPOSITORY
    hygiene precisely because the journal lives outside the tree.

    An interrupted installer leaves a journal in a non-terminal state.
    `recover_transaction()` reconciles it from the manifest and preimages
    alone; this asserts none is outstanding -- EXCEPT the one transaction that
    is installing this suite right now, if there is one.

    A blanket "ignore journals during tests" would make this assertion vacuous
    whenever it mattered most. Naming the single active transaction keeps every
    other journal in scope.
    """
    journal_root = resolve_runtime_paths().transaction_journal
    active = os.environ.get(ACTIVE_TRANSACTION_ENV, "").strip()
    pending = incomplete_transactions(journal_root)
    if active:
        pending = [p for p in pending
                   if p.get("transaction_id") != active]
    assert pending == [], (
        "{} unresolved transaction journal(s) under {}{}: {}".format(
            len(pending), journal_root,
            " (excluding the active {})".format(active[:12]) if active else "",
            [p.get("transaction_id", "?") for p in pending]))


def test_the_exclusion_is_ONE_named_transaction_not_a_blanket(monkeypatch,
                                                              tmp_path):
    """The exclusion must not become a tolerance.

    Setting the variable to one identifier must hide THAT journal and no other.
    A future edit that skipped the assertion whenever the variable is merely
    PRESENT would pass the test above and quietly disable the invariant.
    """
    import json
    root = tmp_path / "journals"
    for name, state in (("aaaa1111", "applying"), ("bbbb2222", "applying")):
        d = root / name
        (d / "preimages").mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({
            "schema": "gvc.repository-transaction", "schema_version": 2,
            "generation": 1, "updated_at": "x",
            "values": {"transaction_id": name, "state": state,
                       "repo_root": str(tmp_path), "targets": []}}),
            encoding="utf-8")

    everything = incomplete_transactions(root)
    assert len(everything) == 2, everything

    monkeypatch.setenv(ACTIVE_TRANSACTION_ENV, "aaaa1111")
    active = os.environ.get(ACTIVE_TRANSACTION_ENV, "").strip()
    filtered = [p for p in incomplete_transactions(root)
                if p.get("transaction_id") != active]
    assert [p["transaction_id"] for p in filtered] == ["bbbb2222"], filtered
    monkeypatch.undo()


def test_the_journal_root_is_outside_the_repository():
    """The property that lets the two invariants stay separate.

    If the journal lived inside the tree, every hygiene check would need an
    exception for live recovery state -- and an exception is where a detector
    stops being able to distinguish residue from anything else.
    """
    paths = resolve_runtime_paths()
    journal = str(paths.transaction_journal)
    assert not journal.startswith(str(paths.project_root)), journal
    assert paths.project_root not in paths.transaction_journal.parents


def test_this_test_declares_no_exceptions_of_its_own():
    """A STRUCTURAL assertion, and the reason it is structural.

    The failure mode being guarded is a future edit that adds a local
    allowance -- an ignored path, a tolerated shape -- rather than changing the
    shared classifier. That divergence would not fail any behavioural test; it
    would simply make this invariant quietly weaker.

    So the check reads this file's own source and refuses if it names an
    exception vocabulary that belongs in backup_artifacts.
    """
    import ast
    import io

    source = io.open(__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    banned = {"SCRATCH_ROOTS", "BACKUP_SHAPES", "NOT_THIS_REPOSITORY",
              "ALLOWED", "EXCEPTIONS", "IGNORE", "TOLERATED"}
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    defined.add(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    defined.update(e.id for e in tgt.elts
                                   if isinstance(e, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    collisions = sorted(defined & banned)
    assert not collisions, (
        "this test defines its own exception vocabulary {}; that policy "
        "belongs in repository_hygiene.backup_artifacts, which is the single "
        "classification authority".format(collisions))
