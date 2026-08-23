# Repository records

**Author: Monzia Moodie**

Durable machine evidence about what actually happened. Not documentation about
it, and not runtime state.

```
documentation  !=  repository record  !=  runtime state
```

`docs/` holds what humans author to explain or govern the project. `records/`
holds facts emitted, captured or preserved. The transaction work already
separated runtime state from repository state; this is the analogous separation
for durable evidence, ruled by ADR-0004.

---

## Why this plane exists

Measured 2026-08-22 over the tracked corpus: **twenty-six machine-evidence
documents committed across six directories** -- `docs/audits/evidence/2026-07-09`
and `2026-07-24`, `docs/incidents`, `docs/measurements`, `docs/migrations`,
`docs/verified` -- two of which had no assigned plane at all. Meanwhile every
install attestation lived **outside version control entirely**, cited by commit
messages that could not resolve.

The cause was not carelessness. Machine records never acquired an architectural
layer, so each subsystem filed them under whichever documentation noun was
convenient. Adding `docs/evidence/` would have improved the noun and preserved
the category error. An installation attestation is closer to a signed instrument
record than to a prose report.

---

## The operational authority is code

Placement follows **role**, and the role-to-root mapping lives in
`src/genomic_variant_classifier/repository_records/roles.py` -- not in this file
and not in ADR-0004.

```
destination(artifact) = f(ArtifactRole)
```

A document stating `role X -> records/foo` while code states `records/bar` is the
defect this programme spent its length removing: a README mirroring internal
state, four installers each carrying a private notion of "neutral", nine
attestations in three shapes under one schema version. **The mapping that
performs the placement must be the mapping that is executed.**

So this file describes the plane. It does not define it, and it enumerates
nothing that code already owns.

---

## Four orthogonal axes

Answering one does not answer another, and collapsing them turns a measured
ruling about specific artifacts into a universal licence.

| Axis | Question |
|---|---|
| `ArtifactRole` | what kind of record is this, and therefore where does it live |
| `DisclosureClass` | may these exact bytes be published |
| `PreservationDisposition` | may this artifact be preserved verbatim at all |
| `RetentionClass` | how long is it kept |

`ProvenanceRelation` records where an artifact came from, as a fact about the
artifact rather than about its location -- a path loses that the moment the
artifact moves.

---

## Preservation is not authoring

ADR-0004 section C separates three validation policies that must never alias one
another:

```
AUTHORING      what newly authored artifacts must look like
PRESERVATION   what historical artifacts must RETAIN
INTERCHANGE    what structured data must satisfy to be interpreted
```

Measured 2026-08-23: **every install attestation ends without a trailing
newline**, because `json.dumps` does not append one. The authoring predicate
requires one. An importer reusing it would refuse every file this archive exists
to hold, and adding a newline would change the bytes -- destroying the byte
identity that is the entire preservation claim.

> An archival importer verifies the artifact **as found**. It does not retrofit
> current repository formatting conventions onto historical evidence.

That asymmetry is visible in a single directory: the manifest is authored and
ends with a newline; the artifacts it indexes are preserved and do not. It is
also structural, in `.gitattributes`:

```
records/**/*.json                                text eol=lf
records/attestations/installations/artifacts/**  -text
```

Order is load-bearing there. Later rules win, and the general rule matches the
artifacts directory too, so the `-text` rule must come last.

---

## A basename is not a locator

Retaining a historical filename does **not** make a commit citation resolve.
Git does not turn a filename in a commit message into a locator. Resolution is a
repository property, established by `legacy_aliases` in a manifest and proven by
a test.

The citation set is a **committed artifact**, never a walk of git history. A
test deriving it from `git log --all` would be silently vacuous in a shallow
checkout: `actions/checkout` defaults to `fetch-depth: 1` -- measured 2026-08-22
across ten invocations in four workflows, none declaring a depth -- so such a
walk sees one commit, asserts one citation resolves, and passes.

---

## What this plane must never hold

- Runtime state. That belongs to the paths ADR-0002 governs.
- Documentation. Narrative interpretation of evidence belongs in `docs/`.
- A second copy of anything code already owns.

## What is not here yet

`docs/audits/evidence/`, `docs/incidents`, `docs/measurements`,
`docs/migrations` and `docs/verified` still hold committed machine evidence.
`EVIDENCE-DISPOSITION-INCONSISTENT-1` is open, and those artifacts will be
classified **individually** rather than moved wholesale because a directory now
exists. `docs/archive/` likewise still holds a stranded worktree recovery
artifact -- `ARCHIVE-SEMANTIC-COLLISION-1`.

A generated global index over multiple record families is deliberately deferred:
there is little value in an indexing subsystem while exactly one family exists,
and when it arrives it must be a deterministic projection over the per-family
manifests, never a second authority.
