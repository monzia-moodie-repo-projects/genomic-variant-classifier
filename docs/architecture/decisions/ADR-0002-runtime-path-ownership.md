# ADR-0002 -- Runtime path ownership and the repository certification surface

**Author: Monzia Moodie**
**Status:** accepted
**Date:** 2026-08-21
**Authority:** normative
**Domains:** execution
**Measured at commit:** 084ece5

---

## Context

Two architecture sketches produced during 2026-08-21 described the runtime
filesystem topology incorrectly. Both were caught by live measurement before
implementation. This ADR records the measured topology, the reasoning that makes
it load-bearing, and the invariants that will prevent a future sentence from
moving it.

### Measured topology, 2026-08-21 at 084ece5

```
project_root         C:\Projects\genomic-variant-classifier
artifact_root        C:\Projects\genomic-variant-classifier      (== project_root)
state_root           C:\Projects\genomic-variant-classifier\.gvc-state
cache_root           C:\Users\monzi\AppData\Local\GenomicVariantClassifier
transaction_journal  C:\Users\monzi\AppData\Local\GenomicVariantClassifier\transactions
```

`RuntimePaths` declares four dataclass FIELDS -- `project_root` (line 133),
`artifact_root` (134), `state_root` (135), `cache_root` (136) -- and four
PROPERTIES: `reports_root` (139), `literature_scout_state` (148),
`orchestrator_state` (158), `transaction_journal` (163). `installation_lock_root`
is not declared and is introduced by this ADR.

## Decision

### 1. Superseding correction -- transaction and lease roots

An earlier architecture sketch stated:

```
transaction_journal_root = state_root / "transactions"
```

**That description is incorrect and MUST NOT be implemented.**

The executable runtime-path resolver defines:

```
transaction_journal = cache_root / "transactions"
```

This is intentional and load-bearing. Transaction journals are recovery state
associated with a local execution environment, not repository state. Moving them
beneath `project_root` would subject crash-recovery state to repository
snapshots, repository-hygiene enumeration, `.gitignore`, worktree copying,
archive and export behaviour, git operations, and accidental deletion during
repository cleanup -- and it would destroy the reason repository hygiene and
transaction hygiene need no `.gitignore` exception.

The installation lease similarly resides at:

```
installation_lock_root = cache_root / "installation-locks"
```

Neither location may be nested beneath `project_root`.

`incomplete_transactions()` was read at 084ece5 and confirmed to skip any
directory lacking `manifest.json`, so a `locks/` child of the journal root would
not currently be miscounted. That accident does not justify the co-location.
Journals hold recovery-protocol state; locks hold mutual-exclusion state.
Different lifecycle, different schema, different cleanup rules, therefore
sibling directories.

**When this ADR conflicts with descriptive handoffs or session documents, the
`RuntimePaths` resolver and its invariant tests are authoritative.**

### 2. `installation_lock_root` is added as a property

It mirrors `transaction_journal` in shape, which keeps the resolver's four-field
core unchanged:

```python
@property
def installation_lock_root(self) -> Path:
    return self.cache_root / "installation-locks"
```

Lease identity derives from the canonical resolved repository path, not the
basename:

```python
repo_identity = sha256(str(repo.resolve()).encode("utf-8")).hexdigest()
```

Correctness comes from an operating-system exclusive lock, never from the mere
existence of a file. The lock file carries diagnostic metadata only.

### 3. `artifact_root == project_root` is a supported default

The resolver deliberately permits repository identity and artifact identity to
coincide by default and provides injection so that tests and deployments can
separate them. The coincidence is **SUPPORTED DEFAULT / SEMANTICALLY
OVERLOADED**, not a defect.

The defect is treating `artifact_root` as though it always denotes a
generated-only subtree. It does not, and never did.

### 4. `.gvc-state` is repository-local volatile operational state

Measured 2026-08-21: `.gvc-state/` exists, contains
`literature_scout/state.json` (15,301 bytes), is untracked, is hidden by the
root-anchored rule `.gitignore:103:/.gvc-state/`, and does not appear in
`git status --untracked-files=all`.

It is therefore mutable operational state living inside repository identity but
outside git identity. It is classified as volatile and excluded from transaction
delta certification **by semantic declaration**.

Migrating it out of the checkout entirely is the superior long-term design, but
that is a separate change requiring a caller census and a backward-compatible
resolver. It is explicitly NOT part of D0, D3, D4, D5 or D6.

### 5. The certification surface, and why it is not a snapshot policy

The transaction does not take a snapshot because snapshots are interesting. It
defines **the filesystem surface over which an atomic repository transition is
certified**. The type is named accordingly:

```python
@dataclass(frozen=True)
class RepositoryCertificationSurface:
    root: Path
    explicitly_volatile_roots: tuple[Path, ...] = ()
```

That name forces the right question -- "why is this path outside
certification?" -- rather than "would it be convenient to exclude this
directory?"

### 6. Exclusions fail closed

A proposed policy excluded `paths.artifact_root`. Because
`artifact_root == project_root`, that exclusion would have removed the entire
repository from the certification surface, and the delta detector would have
compared an empty set against an empty set and passed unconditionally. That is
the same vacuity family as `RELOCATION-FALSE-POSITIVE-1`.

Construction therefore refuses:

- any volatile root equal to the repository root;
- any volatile root that is not a strict descendant of the repository root;
- duplicate volatile roots;
- volatile roots that contain one another, which would present two policy
  decisions where only one exists.

### 7. Certification scope is never derived from `.gitignore`

The implementation MUST NOT contain logic of the form:

```python
if git_ignored(path):
    skip(path)
```

That would make `.gitignore` capable of silently changing the transaction's
observation surface. Ignored files remain inside the certification surface
unless a volatile root declares otherwise.

Git status remains available as a cheap secondary diagnostic, invoked with
`--porcelain=v2 -z --untracked-files=all` and consumed as bytes. It is never the
correctness authority, because git status cannot report arbitrary mutations to
ignored files.

## Invariants (tests, not prose)

```python
def test_journal_and_lease_are_outside_the_repository():
    p = resolve_runtime_paths()
    assert not _is_within(p.transaction_journal, p.project_root)
    assert not _is_within(p.installation_lock_root, p.project_root)


def test_journal_and_lease_are_siblings_under_cache_root():
    p = resolve_runtime_paths()
    assert p.transaction_journal == p.cache_root / "transactions"
    assert p.installation_lock_root == p.cache_root / "installation-locks"
    assert p.transaction_journal != p.installation_lock_root


def test_certification_surface_refuses_an_exclusion_equal_to_the_root():
    with pytest.raises(CertificationSurfaceError):
        RepositoryCertificationSurface(root=repo, explicitly_volatile_roots=(repo,))


def test_certification_surface_refuses_a_non_descendant_exclusion():
    with pytest.raises(CertificationSurfaceError):
        RepositoryCertificationSurface(root=repo, explicitly_volatile_roots=(outside,))
```

`_is_within` is an explicit helper rather than `Path.is_relative_to`, so the
predicate is testable and its failure mode is visible. It is also **mandatory**:
string-prefix containment is forbidden throughout the transaction and
installation code, because

```
C:\foo\bar2
```

starts with

```
C:\foo\bar
```

as text while being no descendant of it. Existing probes and installers that
use `str(path).startswith(str(root))` are to be migrated.

```python
def is_within(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def is_strict_descendant(child: Path, parent: Path) -> bool:
    return child.resolve() != parent.resolve() and is_within(child, parent)
```

## Open, deliberately not decided here

| Item | State |
|---|---|
| `STATE-STORE-OWNERSHIP-1` -- state identity and ownership are encoded by path convention rather than a registry. Three surfaces exist: the orchestrator SharedState, `data/agent_state.json`, and `.gvc-state/literature_scout/state.json`. None is a defect individually | CONFIRMED architectural debt |
| `STATE-ROOT-EXTERNALIZATION-1` -- move runtime-mutable state out of the checkout | DEFERRED, needs a caller census |
| Renaming `artifact_root` to `artifact_namespace_root` | DEFERRED, not worth compatibility churn |

### Refuted, and deliberately NOT registered

`TRANSACTION-ABANDONED-NON-TERMINAL-1` is **REFUTED**. `incomplete_transactions`
excludes only `COMMITTED` and `ROLLED_BACK` because only those two mean nothing
further is owed. `ABANDONED` means active execution gave up while recovery work
remains owed, and its declared transition target is `ROLLING_BACK`. Treating it
as terminal would conflate process-lifecycle terminality with repository-
obligation terminality -- precisely the semantic compression ADR-0001 prohibits.
The behaviour is intentional and correct. If the name keeps inviting the
confusion, a later rename to `EXECUTION_ABANDONED` is available, but the
enumeration is not to be churned now.
