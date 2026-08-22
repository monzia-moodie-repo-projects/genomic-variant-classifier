# ADR-0004 -- Repository records, evidence preservation, and authority succession

**Author: Monzia Moodie**
**Status:** accepted
**Date:** 2026-08-22
**Authority:** normative
**Domains:** meta, data_schema, project_state, historical_repository_record
**Measured at commit:** 0999af0

---

## Context

ADR-0003 assigned an authority domain to every documentation plane and left two
directories unassigned. Measurement then showed the problem was neither two
directories nor a naming choice.

**Durable machine evidence is committed in six different places.** Measured
2026-08-22 over the tracked corpus, twenty-six JavaScript Object Notation
documents:

```
docs/audits/evidence/2026-07-09      1      docs/measurements     2
docs/audits/evidence/2026-07-24     14      docs/migrations       1
docs/incidents                       7      docs/verified         1
```

Two of those directories -- `docs/migrations/` and `docs/verified/` -- had no
plane in ADR-0003 either, so the unassigned-plane problem was four, not two.

And `docs/archive/`, which ADR-0003 assigned to `DEVELOPMENT_NOTEBOOK`, already
held three files, none of them documentation:

```
docs/archive/worktree-agent-afd2a1fa-uncommitted.patch          76,314 B
docs/archive/worktree-agent-afd2a1fa/worktree_modifications.patch  182 B
docs/archive/worktree-agent-afd2a1fa/worktree_status_at_removal    235 B
```

That is a stranded git worktree recovery artifact. **A plane was assigned to a
directory whose existing contents belong to a different one.**

Eleven install attestations, meanwhile, live outside version control entirely,
cited by eleven commit messages.

The common cause is not carelessness. It is a missing layer: **machine records
never acquired their own architectural home**, so each subsystem filed them
under whichever documentation noun was convenient. Adding `docs/evidence/` would
improve the noun and preserve the category error. An installation attestation is
closer to a signed instrument record than to a prose report; a migration
manifest is not documentation *about* a migration, it is part of the migration's
evidentiary record.

## Decision

### A. Repository records are a first-class plane

```
documentation  !=  repository record  !=  runtime state
```

The transaction work already separated runtime state from repository state. This
is the analogous separation for durable evidence. The canonical root is
`records/`, a sibling of `docs/`, not a child.

`docs/` holds what humans author to explain or govern the project. `records/`
holds durable facts emitted, captured or preserved about what actually happened.

The six planes:

| Plane | Canonical role | NOT authoritative for |
|---|---|---|
| Public identity | external scientific description | internal governance, dynamic counts |
| Current programme | present state and intended future | discharged history |
| Normative architecture | decisions and contracts | observations, execution evidence |
| Development notebook | human-authored chronological reasoning | current state |
| Historical documents | frozen superseded narrative | current state |
| **Repository records** | **durable machine evidence and provenance** | **narrative interpretation** |

Only the minimum structure any unit needs is created. **Empty directories are
not created to satisfy a diagram**, and cannot be committed anyway: git tracks
files, not directories.

### B. ArtifactRole determines placement; paths do not determine semantics

**THE OPERATIONAL AUTHORITY IS CODE, NOT THIS RECORD.** The role enumeration
and the role-to-root mapping live in
`src/genomic_variant_classifier/repository_records/roles.py`. This record is
normative about the ARCHITECTURE; it is not a second operational copy of the
mapping.

That distinction is not pedantic. A record stating `role X -> records/foo` while
code states `role X -> records/bar` is the identical defect this programme has
spent its length removing: a README mirroring internal state, four installers
each carrying a private notion of "neutral", nine attestations in three shapes
under one version. One semantic concept, one typed owner. **The mapping that
performs the placement must be the mapping that is executed.**

Illustrative only, and subordinate to the module:

```python
# ILLUSTRATIVE. See repository_records.roles for the authority.
class ArtifactRole(StrEnum):
    INSTALLATION_ATTESTATION = "installation_attestation"
    EXECUTION_MEASUREMENT = "execution_measurement"
    AUDIT_RESULT = "audit_result"
    INCIDENT_EVIDENCE = "incident_evidence"
    MIGRATION_RECORD = "migration_record"
    VERIFICATION_RESULT = "verification_result"
    RECOVERY_ARTIFACT = "recovery_artifact"
```

The registry is bound by a test, not by this prose:

```python
def test_every_artifact_role_has_exactly_one_canonical_root():
    assert set(CANONICAL_RECORD_ROOTS) == set(ArtifactRole)
    assert len(set(CANONICAL_RECORD_ROOTS.values())) == len(ArtifactRole)
```

### B2. Disclosure is a second axis, orthogonal to role

`ArtifactRole` answers *what kind of record is this, and therefore where does it
belong*. It does **not** answer *may these exact bytes safely enter a public
repository*. Those are different predicates, and conflating them would make a
measured ruling about eleven benign attestations into a universal licence.

```python
class DisclosureClass(StrEnum):
    PUBLIC_VERBATIM = "public_verbatim"
    PUBLIC_DERIVATIVE = "public_derivative"
    RESTRICTED_VERBATIM = "restricted_verbatim"
    HASH_ONLY_PUBLIC = "hash_only_public"
```

The order is fixed: **classify role, classify disclosure, then validate
preservation, then choose storage.** Never *this is evidence, therefore commit
its bytes*.

For the eleven measured attestations the ruling is
`INSTALLATION_ATTESTATION` and `PUBLIC_VERBATIM`, so no byte changes.

### B2b. Disclosure is not preservation eligibility

`DisclosureClass` answers *may these bytes be published*. It does not answer
*may this artifact be preserved verbatim at all*. Those separate whenever an
artifact is corrupt, truncated, non-reproducible, or of disputed provenance.

```python
class PreservationDisposition(StrEnum):
    ADMITTED_VERBATIM = "admitted_verbatim"
    ADMITTED_WITH_DEFECT_NOTE = "admitted_with_defect_note"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"
```

`ADMITTED_WITH_DEFECT_NOTE` is the important one. A malformed historical
attestation is still historical evidence; the defect belongs in the manifest,
not in the bytes. Preservation validity and interchange validity are orthogonal:

```
preservation_valid = True      these are exactly the bytes that existed
interchange_valid  = False     these bytes do not satisfy the schema
```

Both may be true of one artifact simultaneously, and a preservation system that
cannot express that will eventually be asked to repair evidence in order to
store it.

### B3. A record has three identities, and a filename is none of them

Retaining a historical basename does **not** make eleven commit citations
resolve. Git does not turn a filename in a commit message into a locator. The
claim must become a repository property:

```python
@dataclass(frozen=True)
class RecordId:
    """Durable logical identity. Never reused, never renumbered, never derived
    from a path, a filename or an ordinal."""
    value: str


@dataclass(frozen=True)
class ArtifactInstance:
    """One concrete byte sequence at one location."""
    content_sha256: str
    canonical_path: str
    size_bytes: int


@dataclass(frozen=True)
class RecordIdentity:
    record_id: RecordId
    instance: ArtifactInstance
    legacy_aliases: tuple[str, ...]
```

A record and an artifact instance are not the same thing. A record may later be
copied, mirrored, re-encoded for interchange, or exist as several byte
sequences; keeping them one type would force a rename or a renumber the first
time that happens, and a renumbered durable identity is not durable.

**Identifier allocation is a governed operation.** A sequential
`ATT-INSTALL-0001` scheme is deceptive: it implies a global ordering that
nothing enforces, and it invites renumbering. The identifier is
`REC-<uuid4-hex>`, allocated **once, at preservation**, and never regenerated.

`uuid.uuid7()` is unavailable: measured 2026-08-22, `uuid6`, `uuid7` and
`uuid8` entered the standard library in Python 3.14, and continuous integration
runs 3.11 and 3.12. `uuid4` it is.

That randomness has a consequence worth stating before it bites. A future index
proven by `actual == render_index(scan(...))` can only be deterministic if the
projection **reads** identifiers from the canonical manifests rather than
minting them. If a renderer ever allocates, the equality check fails on every
run for a reason unrelated to drift -- a check failing for the wrong reason,
which this project treats as a defect in its own right.

> **Allocation happens once, at preservation. Every later projection is a pure
> function of stored metadata.**

**The citation set is a COMMITTED artifact, not a walk of git history.** A test
collecting citations from `git log --all` would be silently vacuous in a shallow
checkout -- `actions/checkout` defaults to `fetch-depth: 1`, so such a walk sees
one commit, collects one citation, asserts it resolves, and passes. That is the
family of the vacuous detritus iterator and the liveness gate whose default
invocation could not fail.

So the manifest carries `legacy_aliases`, the test asserts manifest against
artifacts against aliases -- history-independent and non-vacuous in any checkout
-- and reconciliation against git history remains a PROBE, run where full
history exists.

The inversion is the point: `destination(artifact) = f(ArtifactRole)`, not
`meaning(artifact) ~= directory someone happened to choose`.

Consequently the four unassigned directories receive **no directory-wide plane
ruling**. Their contents are classified artifact by artifact. A word-frequency
heuristic over `docs/audits/` was distorted by a single 5.5-megabyte
JavaScript Object Notation file contributing 21,388 matches -- concrete evidence
that directory-wide semantic classification is unsafe.

### B4. Provenance is a relation, not a location

Where an artifact came from is a fact about the artifact. Encoding it in a path
loses it the moment the artifact moves.

```python
class ProvenanceRelation(StrEnum):
    EMITTED_BY_INSTALLER = "emitted_by_installer"
    CAPTURED_FROM_EXTERNAL_TOOL = "captured_from_external_tool"
    IMPORTED_FROM_STAGING = "imported_from_staging"
    DERIVED_FROM_RECORD = "derived_from_record"
    RECOVERED_FROM_INCIDENT = "recovered_from_incident"
```

The eleven attestations are `EMITTED_BY_INSTALLER` and then
`IMPORTED_FROM_STAGING`. Both are true and both matter: the first says what
produced them, the second says how they entered the repository. A later
migration must be able to answer "where did this come from" without inferring it
from a directory name -- which is the same failure the whole record plane
exists to end.

### B5. Retention is a policy, not an accident

Immutability is not the same as permanence.

```python
class RetentionClass(StrEnum):
    PERMANENT_EVIDENCE = "permanent_evidence"
    SUPERSEDABLE_SNAPSHOT = "supersedable_snapshot"
    TRANSIENT_DIAGNOSTIC = "transient_diagnostic"
```

Without this, `records/` accumulates forever and eventually someone deletes
material under time pressure with no policy to appeal to. Install attestations
are `PERMANENT_EVIDENCE`. A future large diagnostic dump may not be, and saying
so in advance is cheaper than arguing about it later. Retention is orthogonal to
disclosure and to role.

### C. Three validation policies, never aliases for one another

```
AUTHORING POLICY      what newly authored artifacts must look like
PRESERVATION POLICY   what historical artifacts must RETAIN
INTERCHANGE POLICY    what structured data must satisfy to be interpreted
```

`VERBATIM-IMPORT-NOT-AUTHORING-1`, measured 2026-08-22: every install
attestation was written with `json.dumps()` or `to_json()`, neither of which
appends a newline. **Eleven of eleven end without one.** The authoring predicate
requires a trailing newline, so an importer reusing it would refuse every file
it exists to preserve, and adding one would change the bytes -- destroying the
byte identity that is the entire preservation claim.

```python
def validate_authored_text(data: bytes) -> None: ...
def validate_verbatim_artifact(data: bytes) -> None: ...
def validate_attestation_document(data: bytes) -> None: ...
```

> **An archival importer verifies the artifact as found. It does not retrofit
> current repository formatting conventions onto historical evidence.**

**Admission precedes preservation.** For an artifact ADMITTED to verbatim
preservation, preservation is byte-exact and admits no mutation: no newline
added, no schema upgraded, no key reordered, no file renamed, no path redacted.

```
admitted_verbatim(A)  =>  H(A_source) == H(A_preserved)
```

and NOT `for all A, A must be publicly committed verbatim`. An artifact that
cannot safely be published verbatim is routed to a restricted preservation
channel. It is never silently redacted and then represented as the historical
original -- a redacted copy presented as the original is a forgery, however
well intentioned.

The eleven measured attestations are admitted `PUBLIC_VERBATIM`: no credentials,
valid JavaScript Object Notation, pure ASCII. The ten carrying an absolute
interpreter path and the eleven carrying a platform string publish those
permanently, and that is accepted -- they are provenance. Portability is a
**prospective** schema decision, never a reason to alter history.

### D. AUTHORITY-SUCCESSION-1

An authority cutover is one atomic transition, not a move followed by a
reconstruction:

```
old authority identified          historical predecessor preserved verbatim
new authority materialized        all live consumers resolved
succession explicitly declared    behavioural gate green
```

Invariant: **exactly one current authority before, exactly one after.** No
committed state may have two live authorities or none.

This is not theoretical. Measured 2026-08-22: `tests/unit/test_changelog_encoding.py`
contains two cases whose identifiers are the literal string `ROADMAP.md` while
both dereference the live `docs/ROADMAP.md` -- one asserting it exists, one
reading it. **A bare archival move leaves suite identity unchanged and makes the
suite fail.** That is not a defect in `SuiteTransition`; it is the boundary of
what membership can prove.

### E. MIGRATION-SCOPE-ISOLATION-1

> An unresolved classification blocks a migration unit only when the unit
> changes the storage, authority, interpretation, references or lifecycle state
> of an artifact covered by that unresolved classification.

`Block(U, C) <=> AffectedArtifacts(U) intersect Scope(C) != {}`

So the roadmap cutover proceeds while the legacy evidence containers remain
unclassified. It also forbids the opposite error: if a later migration rewrites
citations into them, the unresolved ruling becomes relevant again.

### F. Structure may classify structure. It may not classify semantics.

`PROBE-CLASSIFIER-COARSE-1`, 2026-08-22. A probe reported
`ArchiveRootState: LEGACY_MIXED` from a structural rule -- tracked, no index,
therefore mixed. The contents were homogeneous recovery evidence. A structural
observation was rendered as a semantic verdict.

```python
@dataclass(frozen=True)
class StructuralCensus:      # what a probe may report
    tracked_files: int
    directories: tuple[str, ...]
    indexed: bool
    external_references: tuple[str, ...]


class SemanticDisposition(StrEnum):    # what requires examination or a ruling
    HUMAN_ARCHIVE = "human_archive"
    RECOVERY_EVIDENCE = "recovery_evidence"
    MACHINE_EVIDENCE = "machine_evidence"
    MIXED = "mixed"
    UNCLASSIFIED = "unclassified"
```

This generalises every instrument defect found in this programme:

```
symbol reference    != invariant ownership
text occurrence     != semantic dependency
directory shape     != semantic role
count equality      != identity equality
identity equality   != passing
format convention   != preservation invariant
citation            != dependency
refusal             != evidence for the refusal
grep miss           != absence
```

The last was added 2026-08-22: a line-oriented search for
`"continue to resolve"` returned nothing while the phrase was plainly present,
wrapped across a line break, contradicting another section of this very record.
**A line-oriented tool cannot answer a question about prose.** Search the
flattened text, or read the section.

The recurring root: **two superficially similar predicates treated as if one
implied the other.** Before enforcing a rule, classify what kind of rule it is
and what kind of object it governs.

### G. A new record schema requires a typed owner in the same unit

`ATTESTATION-SCHEMA-DRIFT-1` occurred because nine documents were hand-built as
dictionaries under one unchanging version. **A preservation manifest is itself a
durable record.** Introducing `gvc.installation-attestation-archive` as an
unvalidated hand-built dictionary would reproduce that defect one level up, in
the unit preserving the evidence of having fixed it.

Every new record schema ships a validator and negative controls in the unit that
introduces it. A preservation unit is therefore an ADDITION transition, not a
NEUTRAL one.

Stronger still: **one typed object owns construction, serialization, validation
and schema version.** Not a dictionary built by hand and validated afterwards --
that only makes drift detectable. Typed construction makes it hard to express.

```
typed construction -> semantic validation -> deterministic serialization
                   -> schema validation   -> round-trip validation
```

Serialization is `json.dumps(payload, indent=2, sort_keys=True,
ensure_ascii=True)` with a terminating newline. Sorted keys and fixed indent are
fully deterministic; a compact separator form is no more deterministic and makes
a durable record that reviewers must audit into a single unreadable line.
**Determinism does not require unreadability.**

Note that the manifest is AUTHORED and the artifacts it indexes are PRESERVED,
so the two obey different policies -- and the directory layout should make that
structural rather than relying on a validator after checkout:

```gitattributes
records/attestations/installations/artifacts/**  -text
records/**/*.json                                 text eol=lf
```

### H. A new top-level plane is measured before it is created

`records/` is not merely a directory. Before its first write, measure:
`.gitignore` (proven not to swallow it), `.gitattributes` (line endings for the
artifact types it will hold), `.dockerignore` (read by an existing test; whether
records ship inside an image), packaging configuration (whether a source
distribution includes it), and any test walking the repository root.

`records/measurements/` will eventually hold multi-megabyte artifacts. Whether
those enter a container image is a decision, not a formality -- so it is decided
here as a DEFAULT rather than rediscovered by every later subtree:

```
git repository        INCLUDE
source checkout       INCLUDE
container runtime     EXCLUDE by default
Python wheel          EXCLUDE
source distribution   EXCLUDE unless deliberately required
scientific archive    INCLUDE when explicitly packaged
```

Durable evidence is not runtime code. Shipping it in every container build buys
image bloat, cache churn and disclosure expansion in exchange for a dependency
that does not exist.

The preflight measures ten things before the first write, and reports facts
rather than verdicts -- structure may classify structure, not semantics:

```
 1  .gitignore              records/ must NOT be ignored
 2  .gitattributes          preserved bytes must not be EOL-normalized
 3  .dockerignore           expression of the EXCLUDE default
 4  packaging config        wheel and source-distribution surfaces
 5  root-walking tests      must tolerate a new tracked root
 6  CI checkout depth       fetch-depth decides whether a history walk can fail
 7  path collisions         no existing records/ path, no case-fold collision
 8  detritus walk cost      a new root joins the hygiene walk
 9  case sensitivity        Windows working tree, case-insensitive filesystem
10  reserved names          no path component is a reserved device name
```

Point 6 is the one that has already produced a defect class elsewhere:
`actions/checkout` defaults to `fetch-depth: 1`, so a check walking git history
in continuous integration may see one commit and pass vacuously.

### I. The plane has one root README, and the index is a projection

`records/README.md` sits at the **plane root**, not inside a family directory.
It states what a repository record is, what the plane is authoritative for, and
what it must never hold. A README inside
`records/attestations/installations/` would establish per-family documentation
as the norm and quietly recreate the scatter this record exists to end.

`records/index.yaml`, when it arrives, is a **materialized projection** over the
canonical per-family manifests -- never a second authority. It is proven by
regenerating it and asserting equality:

```
actual_index == render_index(scan_canonical_manifests())
```

That is the same discipline as the ratchet and the badge rendered from one
measured count, and it is why identifier allocation must not happen inside a
renderer.

### J. Genesis cardinality belongs to the manifest, not to a test

The preservation unit begins with eleven artifacts. Writing `assert len(...) ==
11` into a test would make the archive unable to grow without editing the
assertion, and an assertion edited whenever it fails is not an assertion.

The archive manifest carries `genesis_cardinality: 11` as an immutable field of
its version-1 semantics. The tests then assert what actually matters:

```
len(artifacts) >= genesis_cardinality        the archive may grow, never shrink
every genesis legacy alias still present     no original may be dropped
```

An empty or truncated archive fails; a twelfth attestation does not.

## Consequences

- The eleven attestations land at `records/attestations/installations/` under
  their **original basenames**. Basename retention is a NECESSARY CONDITION for
  citation resolution and is not itself resolution: a filename in a commit
  message is not a locator, and git will not make it one. Resolution is a
  repository property established by `legacy_aliases` in the manifest and
  proven by a test. Stable record identifiers are additional metadata, never a
  replacement filename.
- The manifest carries portable metadata only. Absolute local paths remain
  untouched inside the historical artifacts and are not copied into the index.
- `docs/archive/` narrows to frozen human-authored historical documents.
  `ARCHIVE-SEMANTIC-COLLISION-1` is recorded: the worktree recovery artifacts
  belong at `records/recovery/worktrees/afd2a1fa/` and are **not** moved by the
  attestation unit. Misclassification is recorded now, migrated later.
- ADR-0003's assignment of `DEVELOPMENT_NOTEBOOK` to `docs/archive/` is narrowed
  by this record, not retracted. Its reasoning stands; its scope was measured
  afterwards to be wider than its contents.

## Open, deliberately not decided here

| Item | State |
|---|---|
| Artifact-by-artifact classification of the four legacy containers | PLANNED |
| Whether the DECIDED exclusion defaults are correctly EXPRESSED in `.dockerignore` and packaging configuration | UNMEASURED, gates the first write. The POLICY is decided in section H; only its expression is unmeasured. |
| `records/index.yaml` and the boundary test over it | PLANNED |
| Migration of the worktree recovery artifacts | PLANNED, not in the attestation unit |
| Prospective portable environment block in a future attestation schema | PLANNED |
