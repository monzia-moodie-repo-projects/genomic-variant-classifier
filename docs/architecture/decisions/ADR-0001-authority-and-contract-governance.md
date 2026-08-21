# ADR-0001 -- Authority and contract governance

**Author: Monzia Moodie**
**Status:** accepted
**Date:** 2026-08-21
**Authority:** normative
**Measured at commit:** 084ece5

---

## Context

Between 2026-08-19 and 2026-08-21 the project accumulated, in rapid succession,
a session handoff, a starting prompt, four generations of adopted ruling, and
nine strategy documents. Several of them made overlapping claims about the same
semantic objects. Three concrete contradictions reached the point of nearly
being implemented:

1. An architecture sketch stated `transaction_journal_root = state_root /
   "transactions"`. The executable resolver defines it as `cache_root /
   "transactions"`. Implementing the sketch would have moved crash-recovery
   state inside the repository and destroyed the separation that makes
   repository hygiene and transaction hygiene two independent invariants.

2. A proposed snapshot policy excluded `artifact_root`. On the measured
   installation `artifact_root == project_root`, so the exclusion would have
   removed the entire repository from the delta-certification surface and made
   the invariant vacuous.

3. A handoff and the project memory described the roadmap as having "three
   tiers" and "17 items + 10 carried-forward defects". Structural enumeration of
   all 324 headings in `docs/ROADMAP.md` found zero headings containing "tier",
   a phase-based top-level ontology, and a most-recent open-item heading reading
   "OPEN -- FIFTY-FOUR items".

None of these was a coding error. Each was descriptive prose being treated as
executable architectural truth.

Two earlier formulations were proposed to fix this. The first was a single
nine-level global authority hierarchy. The second was two domain-scoped
five-level hierarchies. Both are superseded here, because both assume authority
reduces to one linear ranking, and it does not. Executable code should outrank
prose about where `transaction_journal` resolves. Executable code should NOT
outrank the scientific charter about whether a causal claim requires
falsification, because code can simply violate an intended scientific policy.

## Decision

Authority is **typed by domain**. There is no global total order.

```
AuthorityDomain
    EXECUTION            where code runs, what paths resolve to, what a
                         transaction certifies
    DATA_SCHEMA          feature schemas, evidence schemas, claim schemas,
                         installer specifications
    SCIENTIFIC_POLICY    what constitutes evidence, a mechanism claim,
                         falsification, external validation
    PROJECT_STATE        what is implemented, in progress, blocked, or planned
    HISTORICAL_REPOSITORY_RECORD
                         what bytes were committed, by which commit, with what
                         commit metadata
    EXECUTION_EVIDENCE   what a run, gate, or transaction actually observed
```

`HISTORICAL_REPOSITORY_RECORD` and `EXECUTION_EVIDENCE` are separate domains
because git history is authoritative for what was committed and is NOT
authoritative for what happened operationally. That distinction is not academic:
it is exactly how a false acceptance line reading `0 passed, 0 skipped, 0
failed` became committed truth for commit `f125187`, while the real gate result
-- 4978 passed, 10 skipped, 0 failed -- existed only in runtime output. A commit
message must never outrank a machine-generated attestation about what a run
observed.

Within each domain, the ordering is:

```
EXECUTION
    1. executable invariant (resolver + its tests)
    2. schema / contract
    3. accepted architecture decision record
    4. generated documentation
    5. narrative documentation

DATA_SCHEMA
    1. versioned machine-readable schema
    2. executable validator
    3. accepted architecture decision record
    4. generated documentation
    5. narrative documentation

SCIENTIFIC_POLICY
    1. accepted scientific charter / architecture decision record
    2. typed claim schema
    3. executable enforcement
    4. README
    5. session prose

PROJECT_STATE
    1. machine-readable current-state registry
    2. generated summary
    3. ROADMAP narrative
    4. handoff
    5. conversation

HISTORICAL_REPOSITORY_RECORD
    1. commit graph and committed bytes
    2. tags and signed release metadata
    3. changelog
    4. session records and incident records
    5. handoffs and conversation

EXECUTION_EVIDENCE
    1. machine-readable install or run attestation
    2. transaction manifests
    3. test reports and external continuous-integration records
    4. runtime logs
    5. commit message prose
```

The last line of `EXECUTION_EVIDENCE` is deliberate. A commit message is
human-oriented, untyped, unversioned, difficult to query, and mutable under
rebase or cherry-pick. It is a derived narrative that summarises and points at
an attestation. It is never the primary evidence object.

The question a reader must ask is never "which document is higher?" but
**"authoritative about what?"**

### Document authority metadata

Every governance document carries front-matter declaring its authority:

```yaml
authority: normative | descriptive | historical_evidence | proposal
domains: [execution, data_schema, scientific_policy, project_state, historical_record]
status: accepted | superseded | draft
supersedes: []
superseded_by: []
measured_at_commit: <short sha>
```

A document with `authority: descriptive` or `authority: proposal` **cannot**
settle a question in any domain. It can only propose.

### Rolling-name evidence files

The author records rulings in files named `decision.txt` and outputs in files
named `output.txt`, because files on disk are read reliably where pasted text is
not. These names are therefore a **sequence**, not a collision: the most recent
file is the current ruling, and each earlier one is superseded rather than
overwritten by accident.

Preserved copies are renamed on ingest to
`decision_<NN>_<YYYY-MM-DD>.txt` and recorded in a manifest carrying the
original filename, the receipt date, the SHA-256, the byte count, the
line-ending kind, whether the file ends with a newline, and explicit
`supersedes` / `superseded_by` edges. The bytes are preserved exactly; only the
filename disambiguates.

### Counts are rendered, never primary

No count is architecture. `5213` tests, `95` features, `13` models, `22` agents,
`54` open items -- each is a measurement of a state at a time. Counts belong in
executable contracts that enforce them and in generated summaries that display
them. They do not belong in identity prose.

This does not weaken any fail-loud contract. `EXPECTED_TABULAR_FEATURE_COUNT`
and `tests/EXPECTED_SUITE_SIZE` remain enforcing invariants. The rule is that
prose must not become a second, unenforced authority for the same number.

## The governing design law

Every defect recorded across 2026-08-19 to 2026-08-21 shares one shape:
**semantic compression**.

```
one field meant scheduling AND reachability
one enum meant lifecycle AND validation AND certification AND blockers
one root meant repository identity AND artifact namespace
one roadmap meant current state AND history
one ratchet entry meant collection provenance AND acceptance evidence
one authority hierarchy attempted to govern unrelated epistemic domains
one filename meant four different rulings
```

The correction is uniform:

```
one semantic concept        ->  one typed owner
derived presentation        !=  source of truth
direct evidence             >   arithmetic reconstruction
```

The third line was added on 2026-08-21 after a draft correction note claimed a
gate result of 4978 passed / 10 skipped was "corroborated" by the preceding
ratchet entry plus a collection count. It is not. A collection count constrains
the sum of passed, skipped and failed; it does not determine their
distribution. Direct evidence for that gate already existed in the transaction
proof record, which made the inference both weak and unnecessary.

## Consequences

- ADR-0002 records runtime path ownership under EXECUTION.
- ADR-0003 will record model role and benchmark lineage.
- ADR-0004 will record evidence and feature contracts under DATA_SCHEMA.
- Handoffs and session records are `historical_evidence` and can never
  become architectural law.
- When an accepted ADR conflicts with descriptive prose, the ADR governs within
  its declared domains, and the prose is annotated with a superseding
  correction placed beside it rather than edited in place.
- A canonical location for a normative document class is not a runtime
  preference. Accepted architecture decision records live at
  `docs/architecture/decisions/` and the installer offers no override. Tests
  that need an alternate location parameterise the lower-level installation
  function, never the production command-line interface.
- Four machine-readable artifact types carry the four epistemic roles that
  markdown, roadmaps, commit messages and ratchets were previously being asked
  to carry at once:

```
Observation    what was measured                 (census attestations)
Finding        classification of observations    (the defect register)
Decision       normative ruling                  (these records)
Attestation    that an operation executed, and its outcome
```

## Status of superseded formulations

| Formulation | Status |
|---|---|
| Single nine-level global authority hierarchy | SUPERSEDED by this ADR |
| Two domain-scoped five-level hierarchies | SUPERSEDED by this ADR |
| "Handoff is authoritative where it contradicts the ruling" | REJECTED |
| "The commit message owns acceptance evidence" | SUPERSEDED -- a typed install attestation owns it; the commit message summarises |
| `docs/adr/` as the record location | REJECTED -- flattens records into a top-level special case |
| Records under `docs/strategy/<date>/` | REJECTED -- strategy is dated evidence and proposal space; an accepted record must outlive the session that produced it |
