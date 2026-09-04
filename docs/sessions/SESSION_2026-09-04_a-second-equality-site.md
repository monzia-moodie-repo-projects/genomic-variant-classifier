# SESSION 2026-09-04 -- a second equality site, and a family already named

**Author: Monzia Moodie**
**Commits:** `2f8dcb9`, `f9b4075`
**Ratchet:** 6110 -> 6136
**Preceding head:** `beadea8` (the 2026-09-03 part-2 session record)
**Ending head:** `f9b4075`, pushed, `+0 -0`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `2f8dcb9` | IDENTITY-LAWS (3A++.4b) | ADDITION +24 | 6110 -> 6134 | 6119p/15s/33w |
| `f9b4075` | SECOND-EQUALITY-SITE (3A++.4c) | ADDITION +2 | 6134 -> 6136 | 6121p/15s/33w |

Phase 1C units 3A++.4b and 3A++.4c, landed consecutively in one session with no
unrelated production work between them, as the design authority requires.

The unit the previous record announced as next was
`IdentityConformanceContract`. The design authority's fifteenth revision
FORBADE it by name, along with `IdentityLaw`, `IdentityRelation`,
`IdentityCase`, `IdentityRegistry` and `IdentityFamily`, ruling instead for a
thin kernel of three assertions. A framework before a third identity family
would create concepts without consumers.

---

## 1. `2f8dcb9` -- three laws, two primitives, no framework

A test-only kernel at `tests/support/identity_laws.py`:

```
assert_identity_equivalence_preserved(before, after)
assert_all_identities_distinct(cases)
assert_orthogonal_change(*, before, after, changed)

_validate_cases(cases, *, where)
_pairwise_relation(cases, *, where="cases")
```

It states what must hold BETWEEN identities and never what an identity IS,
which stays in `provenance/source.py` and `provenance/transformation.py`.

### Why not reuse

MEASURED 2026-09-03, a law-authority census over 234 source, 365 test and 452
script modules: seventy-seven candidate equality or grouping functions, and NOT
ONE with the semantics scientific identity requires.

| candidate | why it cannot serve |
|---|---|
| `partitions_equivalent` | collapses `None`, `""`, whitespace and not-a-number |
| `evaluate_partition_agreement` | returns a metric, not a relation |
| `exact_duplicate_groups` | operates on pandas columns |
| `legacy_values_equal` | treats not-a-number as equal to itself |

For provenance `None != ""` and `"GRCh38" != "grch38"` unless a canonical
schema says otherwise, and any such normalisation belongs at an ADMISSION
BOUNDARY, never inside a comparison.

The census exposed a limit of its own method, recorded as
`LAW-AUTHORITY-CENSUS-BODY-SCAN-MISSES-DELEGATED-NORMALISATION-1`:
`partitions_equivalent` contains none of the tokens the scan searched for,
because the normalisation happens in a helper it calls.

### Three consumers migrated, collection-neutral

Seventy-three node identities before and seventy-three after, proven by
collecting BOTH trees and diffing the sorted sets. Not computed.

An inline `partition()` closure became the equivalence law. Distinctness runs
on TWELVE cases: the frozen v4 corpus holds thirteen cases in twelve classes
with one deliberate pair, `same_key_different_release` built from the same
identity as `clinvar_grch38`. The design authority's earlier text said eleven,
which sets aside BOTH members and surrenders the check that `clinvar_grch38`
differs from every other case. Twelve is the maximal set on which the law is
true.

Orthogonality now asserts BOTH halves. The assertion it replaced proved only
that the transformation digest HELD; a migration that did nothing would also
have satisfied it.

---

## 2. `f9b4075` -- the site the mandatory matrix does not name

Section 19 of the design authority forbids routing orthogonality through
`_pairwise_relation`, because it is a different mathematical statement. That
ruling is correct, and it leaves the kernel with TWO independent token
comparisons. Section 29's mutation 8 names only the first.

MEASURED 2026-09-03: mutating `assert_orthogonal_change` from `!=` to
`is not` **passed all twenty-four tests of the 4b suite**. Every fixture in
that group used short interned literals -- `"aaa"`, `"bbb"`, `"ccc"` -- where
`is` and `==` coincide. The 4b handoff warned about interned literals in its
own section and the fixtures walked into it one section later. This is the
FOURTH occurrence in this arc of equality-versus-identity hidden by a language
optimisation.

Two items close it. Both operands of the new orthogonality test are built at
run time and asserted distinct before use, and neither is a literal, which
keeps the file free of the `SyntaxWarning` that `x is not "literal"` raises --
the gate's warning total is an attested quantity. The static test walks the
parsed kernel, because a behavioural test reaches only a site some fixture
exercises while the kernel docstring bans normalisation everywhere.

The patch is a PURE APPEND, proven on bytes: the postimage's first 10,046
bytes digest to `da56631b6826448db9ad212aa6069fb48f501222d423aedcce61b11e6bc8c2db`,
exactly what 4b installed.

---

## 3. Sabotage

Fourteen executions per unit: ten mandatory mutations and four supplemental
ones for the narrowed `frozenset` contract. All DETECTED, both units.

`S3` restates required mutation 6. Both anchor `if observed != changed:` and
both produce mutant `6b2dc7da53df29cf`. Counting four independent supplemental
contracts would inflate coverage by one, so the evidence DECLARES the overlap
and RECOMPUTES it from mutant digests, refusing if the two disagree in either
direction. Required-mutant uniqueness is asserted too, so the mapping from a
supplemental digest to a required identifier cannot become ambiguous.

Overlap is defined by MUTANT BYTES, never by failure behaviour. Two different
mutants may trigger the same failing set; an identical mutant digest proves
duplication where an identical failure set only corroborates it.

MEASURED across environments: all fourteen mutant programs are byte-identical
between the development sandbox (Python 3.12.3, pytest 9.1.1) and the shipping
environment (Python 3.12.10, pytest 9.0.3), and all fourteen failing-test sets
are identical. The detections are structural.

4c ran every mutation twice, once with the static guard deselected. All
fourteen still failed behaviourally. Zero `GUARD-ONLY`: the guard protects
future sites and rescues no present one.

The evidence schema moved to VERSION 2. Version 1 recorded
`{detected, total, complete, mutations}` and structurally cannot express an
overlap. Leaving both shapes at version 1 would let one nominal version
describe two record shapes -- `EVIDENCE-DOMAIN-V4-PAYLOAD-SCHEMA3-1` exactly,
the defect unit 3A++.2 retired.

---

## 4. The installers, and what they do that earlier ones did not

**Seven targets for 4b, not five.** The epoch-v5 apply log records that a
ratchet-moving unit patches THREE counters from one measured count:
`tests/EXPECTED_SUITE_SIZE`, `README.md` and `docs/ROADMAP.md`. The Session-24
template is NEUTRAL and shows none of it. An installer patching only the
ratchet would have turned the suite red, because `test_readme_claims.py` and
`test_roadmap_claims.py` enforce those documents against the live ratchet.

**Three postimages DERIVED, not shipped.** Each counter is produced from the
real preimage by an anchored substitution matching exactly once, proven on
bytes: the length delta must equal the anchor's length delta and the
replacement must be unique in the postimage. A shipped copy can go stale
between authoring and installing; a derivation cannot.

**The anchors came from the repository's own claim patterns**, read out of the
test modules. That mattered: an earlier search of mine reported NO count site
in the roadmap. The roadmap writes `6,110` with a comma and the authoritative
pattern's class is `[\d,]+`.

**`authored()` is never applied to a derived postimage.** `README.md` carries
sixty non-ASCII bytes, measured.

**One full gate, not two.** The ratchet guard fires in
`pytest_collection_modifyitems`, which runs during `--collect-only`, so the
measurement transaction proves it with a guarded COLLECTION and the apply
transaction runs the single gate.

---

## 5. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Sabotage evidence generated at Python 3.12.3 for a 3.12.10 environment | the dry run refused it; environment-truthfulness as executable policy |
| 2 | Wrote `--tree "<that tree>"` into a command | PowerShell took the brackets literally; `OSError: [Errno 22]`. The hygiene list already forbids this |
| 3 | Added an undeclared `sabotage_evidence` key to the attestation | publication REFUSED after a 1027.54-second gate, before `git add` |
| 4 | A textual derivation replaced one occurrence of two | the 4c tree script hashed the 4b suite and reported a false mismatch |
| 5 | Nineteen instrument defects, all one family, and defect 2 recurred | each caught by a later check disagreeing, or by the repository refusing |
| 6 | Searched only the REPOSITORY, then concluded the coverage probe did not exist | it was in `Downloads` all along, among 1,368 files no search had touched |
| 7 | Hand-rolled `authored()` in three installers | ADR-0004 section C already owns that predicate as `validate_authored_text` |

ERROR 3 IS THE EXPENSIVE ONE. The schema is version 3 and permits no unknown
key; `ATTESTATION-SCHEMA-DRIFT-1` is why. The refusal came after the gate and
after the transaction committed, but BEFORE `git add`, which is where
`PROOF-AFTER-IRREVERSIBILITY-1` requires it. Nothing reached git and nothing
was corrupt. The lawful options were to raise the schema version or not write
the field; a test-only unit does not amend a production schema, so the sabotage
binding moved to the commit message and attestation SHAPE prevalidation moved
INTO THE DRY RUN with a synthetic head and gate. `PROOF-AFTER-IRREVERSIBILITY-1`
said prove before the irreversible step; this adds: prove before the EXPENSIVE
one.

The refusal bought a verification otherwise unavailable. After restore and
re-apply, ALL SEVEN postimages are byte-identical across two independent apply
runs -- the derived counters are deterministic, not merely plausible.

ERROR 6 COST FOUR ROUNDS. Every search covered the repository's 1,106 tracked
Python and PowerShell files and NONE of the 1,368 in
`C:\Users\monzi\Downloads`, where prior-session tooling lives. The search
SPACE was wrong, not the search terms. I then declared absence, which ADR-0004
section F forbids in four words: `grep miss != absence`, added 2026-08-22 after
a wrapped phrase defeated a line-oriented search.

Widening the space, searching FLATTENED text and ranking by score instead of
requiring conjunctions found it immediately:
`Probe_SessionRecordCoverage_2026-08-28.py`, 16,218 bytes, at four signals of
nine -- below five files that merely discuss it. A conjunction matcher had
returned ZERO across the same corpus.

The probe classifies `test(...)` as WORK, so both commits of this session are
judged. It draws coverage from session records AND the changelog, excludes
`CORRECTION_*.md` because such a document names the commit it APPLIES TO, and
compares the newest record's filename date against the changelog's newest
heading. VERIFIED against its own `names_commit` before this record was
finished: both `2f8dcb9` and `f9b4075` are named by both documents.

It also carries, as a comment dated 2026-08-28, the defect I committed on
2026-09-04 -- `records[-1]` selecting the undated kickoff file because
alphabetical order is not chronological order. It was written down in the very
file I could not find.

DEFECT 2 RECURRED ON 2026-09-04, hours after being written into this table.
Checking the changelog entry's trailing blank line, I wrote
`b.endswith(b'\n\n')` inside a quoted here-document, which swallows the
escape, so the test asked for a BACKSLASH followed by the letter n and
reported a correct file as wrong. Naming a defect does not retire it; only a
control does. The reliable form is `data[-2:] == bytes([10, 10])`, which no
quoting layer can reinterpret.

ERROR 5 IS NOT A NEW FAMILY. ADR-0004 section F named its root on 2026-08-22:
"two superficially similar predicates treated as if one implied the other."
Every one of the seventeen is an instance --  `-W always` measuring a different
warning population; `b.endswith(b'\n')` in a here-document testing for
backslash-n; a line-feed counter piping a byte ARRAY; an `ece` substring
matching *predecessor*; a `2>&1` check blind to comments; a probe demanding the
value it already held; a summary line printed unconditionally beside an
acceptance; a count search blind to comma formatting; an incoherent boolean; a
regex matching one indentation level; and `sorted()[-1]` called "newest".

ERROR 7 IS A DUPLICATE AUTHORITY, and section B names that defect precisely:
"four installers each carrying a private notion of neutral". The typed policy
also carries a check mine lacks -- `if not data.strip()`.

---

## 6. Findings

### Registered
`IDENTITY-LAW-CASE-POPULATION-ORDER-SEMANTICS-UNDECIDED-1`. The kernel compares
case populations as ORDERED tuples. Both current consumers derive their
mappings through one deterministic case order, so no present failure requires a
change. Resolve only after measuring whether any consumer constructs mappings
independently and whether mapping order carries diagnostic meaning.

`SUPPORT-PACKAGE-IMPORT-MECHANISM-ASSERTED-IN-PROSE-ONLY-1`.
`tests/support/__init__.py` states that `tests/conftest.py` inserts the
repository root onto `sys.path`. MEASURED: that insertion is CONDITIONAL on an
ancestor holding `scripts/clean_cohort.py`, which is present, so the claim is
accurate in effect while remaining prose with no executable check.

`INSTALLER-AUTHORING-PREDICATE-DUPLICATES-THE-TYPED-POLICY-1`. Three installers
carry a private `authored()` duplicating
`repository_records.validation.validate_authored_text`, which additionally
refuses an empty artifact.

### Still open
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` -- closes only when a real
computation records, persists, reloads and consumes evidence downstream.
`SOURCE-IDENTITY-ERROR-NOT-EXPORTED-1`,
`INSTALLER-HEADER-UNDERSTATES-A-MIXED-TRANSITION-1`,
`FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1`, `HASHING-MIGRATION-PENDING`,
`CONFIG-DECLARES-A-PATH-NOTHING-READS-1`,
`CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1`,
`VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1`,
`AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`,
`MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`,
`CONNECTOR-SOURCE-NAMES-DISAGREE-WITH-THE-MANIFEST-1`,
`DATABASE-CONNECTORS-NOT-BYTE-EXACT-BY-TRANSCRIPT-1`,
`GATE-TIMING-NOISE-EXCEEDS-TREND-1`,
`GATE-WARNING-COUNT-INTERMITTENT-1`,
`GATE-WARNING-COMPOSITION-NOT-ATTESTED-1`,
`UNCLOSED-FILE-HANDLE-SITES-1`,
`RUNNER-GATE-METADATA-ORDER-1`.

The warning total held at 33 across all THREE gates run on 2026-09-04 -- the
refused 3A++.4b run, the applied 3A++.4b run and 3A++.4c -- and equals the
figure in every attestation on disk carrying an acceptance record, back to
ADR-0004 on 2026-08-22. `GATE-WARNING-COUNT-INTERMITTENT-1` did not recur.

### Deferred to a named unit
`3A++.4d` -- identity-law support surface: an explicit `__all__`, an optional
`NewType` refinement, an executable check for the import mechanism above, and
adoption of `validate_authored_text` in place of the private `authored()`. No
behavioural change. None of it is worth amending a byte-pinned kernel whose
mutation anchors exist to make these units reproducible.

---

## 7. Ending state

```
HEAD     f9b4075, pushed, +0 -0
ratchet  6136
gate     6121 passed, 15 skipped, 0 failed, 33 warnings
suite    4a626b2c2832a3889888e7577db12e51b5d3dcb4bcd73626aaf227ba3e7a2651 -> dee9f68faf3713041b3dd551b55c83ed81e6f3e5f9c4b84dffbb9e038420200e
suite    dee9f68faf3713041b3dd551b55c83ed81e6f3e5f9c4b84dffbb9e038420200e -> 48b06e4427cfbea4137569cad83bb22a8d66e85271e18201449088943a6511e9
```

Each unit's `before_digest` equals its predecessor's `after_digest` at all
sixty-four characters, and each `pre_head_oid` equals the predecessor's
`post_head_oid`. The 4b baseline was READ from TWO independent artifacts that
agree: the Session-24 attestation and the last `# after` line of
`tests/EXPECTED_SUITE_SIZE` itself.

Gate durations, four observations of which three are from 2026-09-04:
1006.0 seconds at 6095 (Session 24, 2026-09-03), then 1043.3 and 1127.8 at
6134 and 979.9 at 6136. The largest suite ran fastest, so the variance is
machine state rather than suite size.

The 3A++ sequence is complete. The kernel exists, its laws are sabotage-proven
in the shipping environment, the second equality site is protected, and the
normalisation ban is executable rather than prose.

## 8. Next intended action

Land this record and the changelog entry as one NEUTRAL unit, then re-run
`Probe_SessionRecordCoverage_2026-08-28.py` to confirm fourteen of fourteen
WORK commits named.

Then run `scripts/check_agents_active.py` and register the liveness findings from
actual measurement. The agent-architecture document proposes
`AGENT-FLEET-STALE-1` on a sixty-two-day figure taken on 2026-08-21; a finding
whose content is a number that moves is stale the day it is filed, so it should
be phrased as "no qualifying fleet run record since <timestamp>, measured at
<timestamp>" with the age derived for display only.

Then 3B.0: a REPOSITORY-WIDE authority census for media type, content type,
encoding and materialization, across every tracked file type and not merely
Python modules. The design authority is explicit that keyword presence is not
authority, and ADR-0004 section F adds the sharper form: `grep miss !=
absence`. No `MediaType` vocabulary may be invented before that census exists.

Before 3C, one architectural ruling is owed: three candidate records now answer
the same causal question -- `DerivationStep` from 3C, `FeatureLineage` from the
feature-architecture document, and `ExperimentIdentity` from the agent
document. Section B of ADR-0004 is unambiguous that one semantic concept has
one typed owner.
