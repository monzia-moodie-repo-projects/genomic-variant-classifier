"""What belongs in a repository, and what is residue.

    A filter that reports zero is not evidence of zero. It is evidence about
    the filter.

INSTALLER-TRANSACTION-1 step 5. This package exists because the same three
questions were being answered by whatever list happened to be nearest.

MEASURED 2026-08-19: the retirement tool scanned `*.bak_*` alone, retired 148
artefacts, and reported "remaining .bak_* artefact(s): 0" while 107 more sat
beside them in shapes it never looked for -- `*.bak`, `*.orig`, `*.rej`, and
the `.pre_<name>.bak` form every `scripts/apply_*.py` writes.

MEASURED 2026-08-20, before this package was written:

    SECRET_PATTERNS  defined TWICE   transactions/repository_transaction.py:94
                                     scripts/retire_backup_artifacts.py:109
    SECRET_CANARIES  defined TWICE   the same two modules

Eleven and seven entries respectively, verified IDENTICAL at runtime -- element
for element, order included. One copy had been written by transcribing the
other. Nothing enforced the agreement, and a third consumer was about to
arrive.

THREE QUESTIONS THAT WERE BEING CONFLATED

    NOT_THIS_REPOSITORY   .venv312, .git, node_modules -- contents that are not
                          this project's artefacts at all.

    SCRATCH_ROOTS         .af_fix_work -- working space where rollback
                          artefacts are PERMITTED.

    BACKUP_SHAPES         the filename shapes that indicate rollback residue.

`.gitignore` answers "should git normally show this path?". It does NOT answer
"may this path legitimately contain rollback detritus?". Deriving hygiene from
ignore rules would mean anyone adding `some_directory/` silently confers
scratch legitimacy -- the semantic coupling this project keeps removing.

Scratch roots are DECLARED here. A test asserts that the declaration and
`.gitignore` correspond, without either deriving meaning from the other.

WHY DECLARATION RATHER THAN OUTCOME
MEASURED 2026-08-20: the retirement tool deleted a backup inside `.af_fix_work`
whose original was resolvable. The twelve real scratch files had survived every
earlier sweep only because THEIR originals happened to be untracked -- an
accident of that data, not a policy. An outcome one approves of, produced by a
mechanism one has not checked, is not evidence the mechanism is right.

RELOCATION IS PROVEN, NEVER GUESSED
`scripts/verify_written_cohorts.py.bak` was classified an unclassified orphan
because no file existed at its derived path. Manual investigation then found
`scripts/forensics/verify_written_cohorts.py` -- tracked, arriving at 0b93d30
("archive 62 spent forensic scripts"), 171 lines against the backup's 171,
differing in exactly two prose passages where the canonical file records that a
defect was fixed. A superseded relocated preimage.

`resolve_relocation()` performs that search automatically, and admits only two
proofs: exact blob identity in git history, or exactly one tracked file sharing
the basename. Ambiguity returns nothing, because a classifier that guesses is
worse than one that refuses.

CONSUMERS
    scripts/retire_backup_artifacts.py    classification and retirement
    transactions/repository_transaction   the secret vocabulary
    tests/unit/test_no_detritus.py        the repository-wide invariant

Each imports; none restates. A test asserts identity of OBJECT, not merely of
value, so a future copy fails rather than drifting in silence.

Author: Monzia Moodie
"""

from __future__ import annotations
