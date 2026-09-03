# SESSION 2026-09-03 -- one epoch retired, one introduced

**Author: Monzia Moodie**
**Commits:** `13daa3f`, `01cd4b4`, `63c9f52`, `b04e826`
**Ratchet:** 5905 -> 6067
**Preceding head:** `b9503fb` (the 2026-09-02 session record)
**Ending head:** `b04e826`, pushed, `+0 -0`

The work spans 2026-09-02 evening into 2026-09-03 by Coordinated Universal
Time: `b04e826` is stamped `2026-09-03T00:06:46Z`. The record is dated by its
final commit.

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `13daa3f` | ACQUISITION-EXACT-IDENTITY (3A+) | ADDITION +18 | 5905 -> 5923 | 5908p/15s |
| `01cd4b4` | SOURCE-EVIDENCE-EPOCH-V4 (3A++.0) | ADDITION +41 | 5923 -> 5964 | 5949p/15s |
| `63c9f52` | CANONICAL-DIGEST-SCHEMA (3A++.1) | ADDITION +29 | 5964 -> 5993 | 5978p/15s |
| `b04e826` | SOURCE-EVIDENCE-EPOCH-V5 (3A++.2) | +77 / -3 | 5993 -> 6067 | 6052p/15s |

Phase 1C units 3A+ through 3A++.2 of the adopted design authority (its twelfth
revision, 2,538 lines, adopted 2026-09-02).

---

## 1. `13daa3f` -- a July retrieval record satisfied August evidence

`SOURCE-ACQUISITION-KEY-ONLY-MATCH-1`, MEASURED BY CONSTRUCTION.
`SourceManifest.__post_init__` matched an acquisition to its evidence on
`a.identity.key`. A `SourceArtifactKey` is `(source, artifact_kind, product)`;
`release_id`, `coordinate_context` and `artifact_sha256` are NOT in it.

Demonstrated by building it:

```
evidence     clinvar/primary_release  release 2026-08  digest aaaa...
acquisition  clinvar/primary_release  release 2026-07  digest bbbb...   ACCEPTED

evidence     clinvar/primary_release  GRCh38
acquisition  clinvar/primary_release  GRCh37                            ACCEPTED
```

`CoordinateContext` exists precisely because "GRCh37 and GRCh38 coordinates are
NOT interchangeable" -- its own error message says so -- and the acquisition
match could not see it.

THE ORIGINAL WAS NOT CARELESS. `SourceEvidenceManifest` enforces one dependency
per key, so `evidence.keys` is a natural uniqueness set. The defect is that
UNIQUENESS WITHIN EVIDENCE and CORRESPONDENCE BETWEEN AN ACQUISITION AND ITS
EVIDENCE are different questions.

### Three gaps, all one mistake in three disguises

Equality versus object identity was untested for all three fields, each hidden
by a different language behaviour:

| field | why `is` passed by accident |
|---|---|
| `coordinate_context` | the tests reused one module-level singleton |
| `release_id` | `"2026-08"` is a short INTERNED literal |
| `artifact_sha256` | `"a" * 64` is CONSTANT-FOLDED into the code object |

MEASURED: `f() is f()` returns `True` for the folded form and `False` for a
runtime-built one. A suite can be blind to a whole class of defect because the
language optimises the very values it uses. Eleven boundaries, eleven detected
after those repairs; reverting the match to keys fails SIXTEEN tests.

---

## 2. `01cd4b4` -- freeze what v4/schema3 emitted, before its retirement

`EVIDENCE-DOMAIN-V4-PAYLOAD-SCHEMA3-1`, MEASURED in `provenance/source.py`:
line 105 declared `drift-source-evidence-manifest-v4` while line 486 digested a
payload carrying `"schema_version": 3`.

The corpus records that rather than cleaning it, so a future reader can
determine why v4 emitted schema 3 without archaeology.

### Canonical JSON, not pickle -- and it needs no hash-seed pin

MEASURED: five runs under `PYTHONHASHSEED=random` produce byte-identical
output, because `SourceDependency.as_record` returns
`sorted(r.value for r in self.roles)` and `SourceEvidenceManifest.of` sorts
dependencies. The migration corpus needed `PYTHONHASHSEED=0` precisely because
it pickles a frozenset of a string-based enum.

### Two structural gaps

EVERY assertion read the FIXTURE, so a corpus could stay perfectly
self-consistent while the code that produced it drifted. A differential test
now regenerates through the committed script's own `build()`.

And the corpus uses the role set observation-plus-label -- ONE OF ONLY THREE
subsets of five roles whose natural iteration order equals its sorted order.
MEASURED across seeds 0, 1, 7 and 42: twenty to twenty-three of the twenty-six
non-trivial subsets iterate UNSORTED. A test that samples can be blind where a
test that asserts the property cannot.

---

## 3. `63c9f52` -- one canonical identity epoch, one version authority

`CanonicalDigestSchema` derives the domain AND the record's `schema_version`
from one field, so the state `domain v5, record schema 4` is unrepresentable.

PROVEN SEMANTIC-ZERO on transformation identity, which was already coherent at
v1. The same probe ran against the pre- and post-conversion modules, swapping
only `transformation.py`: TWELVE digests identical. And the pickle frozen at
`2d90c23`, before this type existed, still produces
`eda4cf34c0bf866342edee305852c08043adb6d0fb2b6cfc798cd9b891c9df4f`.

That separation is why the next unit's digest changes are attributable to the
migration and not to the abstraction carrying it.

Thirteen boundaries sabotaged, twelve detected. The thirteenth was MEASURED as
a no-op rather than argued: the guard makes it unreachable and `canonical_json`
sorts keys, so the serialisation is byte-identical.

---

## 4. `b04e826` -- v4/schema3 retired for v5/schema5

THE FIRST UNIT IN THIS ARC THAT DELIBERATELY CHANGES SCIENTIFIC DIGESTS.

WHY v5 AND NOT v4/schema4: the v4 epoch historically described schema-3
records. Correcting the literal under the same domain would make ONE nominal
domain describe TWO canonical schemas -- exactly what domain versioning
prevents.

| entry | before | after |
|---|---|---|
| `evidence_multi_authority` | `be04468ca802b6e3` | `a685c4929843361d` |
| `evidence_three_gencode_products` | `fbc25f4bd7faf276` | `fa2cfcc0266f15f6` |
| `manifest_clinvar` | `c2e23041fd6e0ad3` | `6193c19e0fe4b37b` |
| `transformation_all_component_kinds` | `eda4cf34c0bf8663` | UNCHANGED |

For all THIRTEEN frozen cases, stripping the epoch metadata leaves canonical
records that are EQUAL, and the digest PARTITION is preserved: thirteen cases,
twelve distinct digests, one deliberate pair. The migration changed values
without changing WHICH manifests are the same.

NEITHER FROZEN CORPUS WAS REGENERATED. The tests became differential; the
fixtures stayed witnesses.

### `describe()` renders the digest

`describe()` embeds the evidence digest, so it changes legitimately while its
structure -- dependency count, authority count, assemblies -- must not.
Exempting the whole field would have discarded a real assertion, so the two
halves are compared separately.

---

## 5. Four refusals, and what each bought

| refused by | what it caught |
|---|---|
| the transition guard | a blanket `if removed: raise`, wrong for three anticipated inversions |
| the installer's own audit | a justification reading "an auditor that nothing called acquires a caller", inherited through six template generations |
| the installer | my check demanded a literal that Unit 3A++.1 had DELIBERATELY removed thirty minutes earlier |
| the acceptance gate | `test_source_release.py` pinned `EVIDENCE_DOMAIN.endswith("-v4")` |

THE LAST IS THE ONE THE SANDBOX COULD NOT FIND. That file holds 84 identities
absent from my sandbox; only the full 6,067-identity run reaches it.
`6051 passed, 1 failed` is a precise instrument.

Re-pinning it to `-v5` would merely reschedule the breakage, so it now asserts
the CLAIM -- the pre-product epoch is retired, the domain carries a version,
and it equals the derived authority -- and a new installer check refuses any
POSITIVE literal pin in any test payload.

That check needed three refinements, each found by running it against the real
files: the first would have refused the frozen-corpus assertion that correctly
stays at v4; the second tripped on my own NEGATIVE pin asserting v3 is retired.

---

## 6. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Equality versus identity untested for three fields | sabotage, three times |
| 2 | Asserted a corpus was self-consistent, not differential | sabotage: unsorted roles changed no test |
| 3 | Chose a role set that was one of only three already sorted | sabotage across four seeds |
| 4 | Fabricated a digest `1cca5b0f5a9d0a6b` and tested against it | two audit checks reported FAIL against a correct attestation |
| 5 | `manifest.keys` yields objects, not tuples | `TypeError` on the first run |
| 6 | `dependency_order` is SIX fields, not three | 52 failures on the first run |
| 7 | Under-predicted the v5 breakage as three sites | seven node identities across five functions |
| 8 | A digest that ignored its content passed every test | sabotage |
| 9 | Demanded a literal the previous unit had removed | the installer refused |
| 10 | A forbidden-pin check too broad, then still too broad | run against the real payloads, twice |

ERROR 4 IS THE WORST. An expected value in an audit must be READ FROM AN
ARTIFACT, never recalled. A fabricated expected value produces a FAIL that
looks like a repository defect. Every audit since chains `before_digest`
against the prior attestation FILE.

---

## 7. Findings

### Closed
`SOURCE-ACQUISITION-KEY-ONLY-MATCH-1` at `13daa3f`.
`EVIDENCE-DOMAIN-V4-PAYLOAD-SCHEMA3-1` at `b04e826`.

### Still open
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` -- closes only when a real
computation records, persists, reloads and consumes evidence downstream.
`SOURCE-IDENTITY-ERROR-NOT-EXPORTED-1`,
`INSTALLER-HEADER-UNDERSTATES-A-MIXED-TRANSITION-1`,
`FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1`, `HASHING-MIGRATION-PENDING`,
`RESOURCE-WARNING-FROM-UNCLOSED-READS-1`,
`CONFIG-DECLARES-A-PATH-NOTHING-READS-1`,
`CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1`,
`VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1`,
`AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`,
`MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`,
`CONNECTOR-SOURCE-NAMES-DISAGREE-WITH-THE-MANIFEST-1`,
`DATABASE-CONNECTORS-NOT-BYTE-EXACT-BY-TRANSCRIPT-1`,
`GATE-TIMING-NOISE-EXCEEDS-TREND-1`.

---

## 8. Ending state

```
HEAD     b04e826, pushed, +0 -0
ratchet  6067
gate     6052 passed, 15 skipped, 0 failed, 33 pre-existing warnings
suite    5a4b3dc70ea207c6 -> 784f3ba581a399ec
```

## 9. Next intended action

Unit 3A++.3: differential epoch-conformance tests, then 3B.0 -- a
REPOSITORY-WIDE authority census for media type, content type, encoding and
materialization concepts, across every tracked file type and not merely Python
modules. The design authority is explicit that keyword presence is not
authority, and that a closed `MediaType` enumeration must not be invented
before that census.
