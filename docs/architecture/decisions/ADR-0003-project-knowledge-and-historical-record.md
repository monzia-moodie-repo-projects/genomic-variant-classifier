# ADR-0003 -- Project knowledge, documentation, and historical-record ownership

**Author: Monzia Moodie**
**Status:** accepted
**Date:** 2026-08-22
**Authority:** normative
**Domains:** execution, data_schema, project_state, historical_repository_record
**Measured at commit:** 31c279a
**Supersedes:** the implicit governance model in which `docs/ROADMAP.md` carried
current state and permanent history simultaneously, and `README.md` mirrored
internal implementation state as executable claims.

---

## Context

Two artifacts had accumulated authority they could not discharge.

`docs/ROADMAP.md` was measured on 2026-08-21 at **7,019 lines, 466,826 bytes,
324 headings**, and was simultaneously acting as roadmap, chronological journal,
architectural record, defect ledger, backlog, identity document, implementation
plan and status report. Its most recent delta was dated **2026-08-08** while the
repository stood three commits past it. Its own heading history shows the
mechanism: `CURRENT STATE SNAPSHOT 2026-07-12`, then `2026-07-15 -- SUPERSEDES`,
then `2026-07-18 -- SUPERSEDES`, and a running count of open items reading
nineteen, twenty-one, twenty-nine, thirty-three, and finally **FIFTY-FOUR**.
That is an append-only notebook impersonating a current-state document.

`README.md` had become an executable mirror of internal state.
`tests/unit/test_readme_claims.py` -- 725 lines, ten tests -- exists because a
2026-07-14 audit found the README stating the feature count in **nine places
with four different values**, the test count in three places with three values,
HGMD as an integrated source whose columns were constant zero, and a training
command using a flag that had never existed. The response was correct for its
moment and solved the wrong architectural problem: it made a public document
into a state mirror rather than removing the duplicated state.

Two symptoms, one cause. Both documents were answering questions that belonged
to different owners.

## Decision

### 1. Authority is assigned by question, never by convenience

ADR-0001 established that authority is typed by domain rather than ranked
globally. This record extends that lattice to documentation and history.

```
AuthorityDomain
    PUBLIC_PROJECT_IDENTITY         README.md
    CURRENT_PROGRAM_STATE           docs/ROADMAP.md
    DEVELOPMENT_NOTEBOOK            docs/archive/
    ARCHITECTURAL_DECISION          docs/architecture/decisions/
    SCIENTIFIC_POLICY               docs/science/
    DATA_SCHEMA                     versioned schemas and registries
    EXECUTABLE_CONTRACT             typed code and configuration
    MEASURED_EXECUTION_EVIDENCE     attestations and manifests
    HISTORICAL_REPOSITORY_RECORD    git
```

Never say "the ROADMAP is authoritative." Say `docs/ROADMAP.md` owns
`CURRENT_PROGRAM_STATE`. No artifact holds vague global supremacy, and every
future conflict is then mechanically resolvable.

The placement question is one sentence:

> **Who needs this fact, for what decision, over what time horizon?**

A visitor deciding whether the programme is interesting -> README. The developer
deciding what to work on next -> ROADMAP. A future investigator reconstructing
what happened and why -> Archive. Code deciding whether an operation is
permitted -> executable contract. A reviewer determining whether an operation
really ran -> attestation. A developer deciding what is normative -> a decision
record. A user determining what changed between releases -> changelog.

### 2. Every plane declares what it MUST NOT own

Architecture erodes through convenience -- *"the ROADMAP is already open, I will
put this there."* Negative constraints are therefore normative:

```
README.md
  OWNS      stable public scientific identity: the problem, the modalities,
            the research principles, major demonstrated results, licensing and
            research-use limits, entry points, links to detail
  MUST NOT  exact test counts, feature counts, agent counts, model rosters,
            active defect counts, internal governance incidents, session
            history, temporary blockers, sprint sequencing, architectural
            disputes, failed installer attempts, mutable dependency topology

docs/ROADMAP.md
  OWNS      current programme state, priorities, blockers, decisions pending,
            near-term sequencing, validation obligations, a BOUNDED recent-
            transitions window
  MUST NOT  discharged history, immutable historical evidence, executable
            runtime configuration, measurement attestations

docs/archive/
  OWNS      what happened and why: observations, experiments, failures,
            reasoning, corrections, decision alternatives, chronology
  MUST NOT  current state, normative rules

docs/architecture/decisions/
  OWNS      accepted architectural decisions within declared domains
  MUST NOT  chronology, status, measurement results
```

The Archive may omit neither consequential events nor their reasoning. The
README may omit internal detail. The ROADMAP may omit discharged history. That
asymmetry is deliberate.

### 3. Canonical locations, reserved now to prevent another naming cycle

Three locations were proposed for decision records during 2026-08-21 --
`docs/architecture/decisions/`, `docs/adr/`, and a dated strategy subdirectory.
The first is canonical and the others are **rejected**: a flat top-level
directory weakens the emerging documentation ontology, and an accepted record
must outlive the strategy session that produced it.

```
docs/
  ROADMAP.md                        CURRENT_PROGRAM_STATE
  CHANGELOG.md                      user- and developer-facing deltas
  architecture/decisions/           ARCHITECTURAL_DECISION  (canonical)
  archive/                          DEVELOPMENT_NOTEBOOK
    notebook/<yyyy>/<mm>/
    findings/  incidents/  experiments/  migrations/  legacy/
  project/                          machine-readable current-state registry
  science/  validation/  data/  models/  results/
```

A canonical location for a normative document class is **not a runtime
preference**. Installers offer no override. Tests needing an alternate location
parameterise the lower-level function, never the production interface.

`docs/measurements/` (353 references) and `docs/audits/` (23) exist and are
**not yet assigned a plane**. They must be classified before any migration
touches them.

### 4. INVARIANT-HANDOFF-1 is repository law

> **No assertion may be retired until its owned invariant has another PROVEN
> owner.**

"Proven" means proven by a deliberate break, not by inspection. The required
history is three steps:

```
A  new owner added; old owner still present
B  falsification proves the new owner detects the defect
C  old assertion removed
```

The period of duplicated enforcement is not waste; it is the handoff proof.

Every retirement records an `InvariantMigration`: the invariant identifier, old
owners, new owners, the falsification fixture, the proof, and the effective
commit.

This law was applied before it was written. A census on 2026-08-22 over the
**entire tracked corpus** -- 1,573 files, 1,565 textual -- found three of the
five invariants `test_readme_claims.py` enforces had **no other owner**: the
model roster, the agent registry, and the drift-monitor exit code, the last with
only a comment in the ratchet as a non-README reference. Commit `31c279a` added
their owners, with six of nine tests as negative controls, while retiring
nothing.

A count of files referencing a symbol is **not** a count of invariant owners.
Nineteen files referenced `base_estimators`; none compared the runtime roster to
a declared list.

### 5. SuiteTransition -- identities, not counts

The suite ratchet detects **accidental test loss**. It is not a measure of
assurance: replacing ten coarse tests with five sharper ones plus three domain
invariants reduces the count and increases assurance. Its conceptual role is
demoted accordingly, and it is retained.

Every suite-size change declares a transition:

```python
class SuiteTransitionKind(StrEnum):
    ADDITION = "addition"                          # delta > 0
    NEUTRAL = "neutral"                            # delta == 0
    DELIBERATE_RETIREMENT = "deliberate_retirement" # delta < 0


@dataclass(frozen=True)
class SuiteTransition:
    kind: SuiteTransitionKind
    expected_added_nodeids: frozenset[str]
    expected_removed_nodeids: frozenset[str]
    invariant_migrations: tuple[InvariantMigration, ...]
    justification: str
```

**A count is not an identity.** A delta of `-5` is equally consistent with
removing the five intended tests and with removing three intended plus two
unrelated. The installer therefore compares the collected node-identity SETS
before and after, and refuses unless the added and removed sets equal the
declared ones exactly. It also cross-checks pytest's summary count against the
number of parsed identities and refuses if two measurements of the same thing
disagree.

`DELIBERATE_RETIREMENT` requires a non-empty justification, every retired
identity named, and an `InvariantMigration` for each invariant whose ownership
moves. No installer in the repository could execute a negative delta before this
record; `build_plan` refuses when the delta is not positive. That path is now
admissible **and documented**, which is stricter than being impossible and
unexplained.

Demonstrated in production at `31c279a`: nine identities declared, nine added,
zero removed, count and identity cross-checked, measurement transaction rolled
back so the count was measured rather than computed.

### 6. Typed freshness -- two-dimensional, and measurable

"Updated at the end of every session" is too vague to enforce, which is how
`ROADMAP-STALE-1` happened. The ROADMAP carries reconciliation front matter:

```yaml
schema: gvc.roadmap
schema_version: 1
state_reconciled_at: <timestamp>
state_reconciled_commit: <sha>
archive_through: <entry id>
```

Staleness is then computed, not judged:

```
FRESH                 relevant commit distance == 0
RECONCILIATION_DUE    relevant commit distance > 0
STALE                 distance exceeds threshold, or a session closed
                      without reconciliation
```

Relevance matters: a cosmetic documentation edit must not force a rewrite, and a
document thirty minutes old can be stale if twelve consequential commits landed.

### 7. Counts are rendered, never primary

`5213` tests, `95` features, `13` models, `22` agents, `54` open items: each is a
measurement of a state at a time. Counts belong in executable contracts that
enforce them and in generated summaries that display them. They never define
identity, and prose must not become a second unenforced authority for the same
number.

This does not weaken any fail-loud contract.
`EXPECTED_TABULAR_FEATURE_COUNT` and `tests/EXPECTED_SUITE_SIZE` remain
enforcing invariants.

### 8. Migration is of authority, not of files

A migration that moves authority records an **Authority Migration Manifest** at
`docs/project/migrations/`, naming for each artifact its old and new authority,
every invariant migration with its proof status, every artifact move with source
and destination blob object identifiers, retired interfaces, open
reconciliations, and completion criteria. It is a transition witness, not a new
authority plane, and becomes archival evidence when the migration closes.

**Archival moves are proven by blob object identifier, not by rename
recognition.** Measured 2026-08-22 in a scratch repository: `git mv` with
unchanged bytes yields the identical blob object identifier at both paths
(`f4e1a7a297bb4962` before and after commit), because blobs are content-addressed --
that is a guarantee. But git stores **no rename entity**; the commit object
records a tree and a parent. `git log --follow` is similarity detection, and it
was broken deliberately by renaming with a rewrite, at which point history
stopped. So the manifest asserts

```
source_blob_oid == destination_blob_oid
```

and does not rely on git recognising anything.

### 9. Epoch boundary

Artifacts written before this migration assume the superseded governance model.
Rather than declaring each inexplicably wrong:

```
Knowledge Architecture Epoch 0    multiplexed ROADMAP, executable README mirror
Knowledge Architecture Epoch 1    typed authority domains, current-state
                                  ROADMAP, historical Archive, non-load-bearing
                                  README, explicit invariant ownership
```

The migration manifest records `from_epoch`, `to_epoch`, and the boundary
commit.

## Consequences

- `render_readme()` is retired after the README ceases to carry the badge; its
  only production caller is `install_no_detritus.py`, which D7 retires.
  `render_ratchet()` remains: the ratchet is a genuine executable contract while
  the badge is presentation.
- `test_readme_claims.py` is decomposed, not merely weakened, and only under a
  `DELIBERATE_RETIREMENT` transition with every invariant handed off first.
- `docs/ROADMAP.md` is preserved by move, not by copy, and a new current-state
  ROADMAP is reconstructed from it.
- The 7,019-line document is **not** discarded. It is primary historical
  evidence.

## Open, deliberately not decided here

| Item | State |
|---|---|
| `docs/measurements/` and `docs/audits/` plane assignment | UNASSIGNED |
| Capability registry superseding `AGENT_REGISTRY_LINEAGE_2026_08` | PLANNED |
| `DOWNLOADS-SHADOWS-TOP-LEVEL-MODULES-1` general mitigation | CONFIRMED, mitigated per-installer only |
| `TRANSACTION-GIT-FAILURE-FAILS-OPEN-1` | CONFIRMED, unaddressed |
| `RESOURCE-HANDLE-LEAK-1` | CONFIRMED, four shipped sites |
| `ATTESTATION-SCHEMA-DRIFT-1` | CONFIRMED, carried forward |
