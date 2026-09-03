# SESSION 2026-09-03 part 2 -- one grammar, one parser

**Author: Monzia Moodie**
**Commit:** `086e2fa`
**Ratchet:** 6067 -> 6110
**Preceding head:** `80432ac` (the part-1 session record)
**Ending head:** `086e2fa`, pushed, `+0 -0`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `086e2fa` | DOMAIN-GRAMMAR (3A++.3) | ADDITION +43 | 6067 -> 6110 | 6095p/15s/33w |

Phase 1C Unit 3A++.3, the final unit of the 3A++ sequence. A HARDENING, not an
epoch migration: no valid digest moved and the accepted input set narrowed.

---

## 1. Two findings, one shape

Both were invariants the module STATED and no test EXERCISED.

### `DOMAIN-VERSION-SUFFIX-GUARD-ADMITS-A-BARE-V-1`

The version requirement read `domain.rstrip("0123456789").endswith("-v")`.
MEASURED 2026-09-03 by calling it:

```
family-v      ACCEPTED   a version marker with no version
trailing-v    ACCEPTED   same
family-v0     ACCEPTED   a domain CanonicalDigestSchema can never produce
family-v01    ACCEPTED   a second spelling of epoch 1
family-1      refused
no-version    refused
```

Its own error message says a digest whose domain never changes cannot express a
schema change. A domain with no digits has nothing to change, and `v1`, `v01`
and `v001` would have been three byte-distinct namespaces for one numerical
epoch -- three digests for one meaning.

I found this by WRITING THE TEST, not by reading the module. The predicate
reads correctly at a glance; only enumerating its accepted language exposes it.

### `ASCII-INVARIANT-ASSERTED-IN-PROSE-ONLY-1`

`serialization.py` line 54 states that `ensure_ascii=True` exists "so the bytes
cannot depend on a locale or a filesystem encoding".

MEASURED: not one byte of non-ASCII appears in any test file, and every
JavaScript Object Notation fixture is pure ASCII -- the high bytes in the
corpora are PICKLE FRAMING, not payload. Disabling the flag changed NOTHING.

It is not a no-op. For a source spelled with a non-ASCII character:

```
ensure_ascii=True    {"source":"clinv\u00e5r"}    pure ASCII bytes
ensure_ascii=False   {"source":"clinv\xc3\xa5r"}  UTF-8 bytes
```

A source name or release identifier CAN carry a non-ASCII character, and under
the second form the digest of the same scientific evidence could differ between
machines -- a digest changing for a reason having nothing to do with the
evidence.

---

## 2. The census came first

Section XXXV of the design authority requires a repository-wide read-only
census BEFORE authoring. Across 1,713 tracked files, all eleven questions:

| question | answer |
|---|---|
| every live versioned digest domain | all canonical |
| any `v0` domains | 5 occurrences, ALL prose ("regime-v0 runs") |
| any leading-zero domains | none |
| any bare `-v` domains | none (309 regex hits were hyphenated prose) |
| any empty-family domains | none |
| any non-ASCII domains | none |
| frozen historical domains | `drift-source-evidence-manifest-v4` only |
| positive literal epoch pins | 1 in code, on the FROZEN corpus, legitimate |
| non-finite floats in a record | NONE |
| direct `domain_digest` callers | 1 production, 3 test files |
| `CanonicalDigestSchema` constructions | 2 production, 1 test file |

So the hardening was PROVEN semantic-zero before a line changed. VERIFIED
after: the evidence digest `6ba0bbdd46a9b6bf` and the transformation digest
`226334b9744a02c3` are unchanged at all sixty-four characters.

---

## 3. One parser, not two validators

`parse_versioned_domain` is now the ONLY definition of the epoch grammar:

```
VersionedDomain          = Family "-v" PositiveCanonicalInteger
PositiveCanonicalInteger = [1-9][0-9]*
```

`CanonicalDigestSchema.__post_init__` PARSES the domain it would emit through
that same function and refuses if the parser disagrees. Previously the two
layers validated independently -- the class required `version >= 1` while the
primitive accepted any digits, or none at all.

That is the dual-authority pathology this arc removed from source evidence, one
layer down. It would have been easy to fix the predicate in place and leave two
rules; the design authority was explicit that this is the moment not to.

### GVC Canonical JSON v1

The serialisation contract is now written down rather than inferred: sorted
keys, compact separators, ASCII output, no locale, no filesystem encoding,
non-finite numbers REFUSED, and NO implicit Unicode normalisation.

The last is deliberate. `"\u00e9"` and `"e\u0301"` are different code point
sequences and a registry may distinguish them; normalising inside a serialiser
would decide an admission-policy question in the wrong place.

---

## 4. Eighteen boundaries sabotaged

Fifteen detected, two measured NO-OPs, one detected-unattributable, ONE REAL GAP.

### The gap

Removing the ASCII DOMAIN check changed no test. Every Unicode test passed a
non-ASCII PAYLOAD; none passed a non-ASCII DOMAIN, and the grammar's `.+`
accepts any character. Without the guard the failure surfaces later as a
`UnicodeEncodeError` from `.encode("ascii")` -- a different exception type,
from a different line, carrying no explanation. A test now asserts the guard
PRECEDES the encode.

### The two no-ops, distinguished by removing the redundancy

Widening the family group from `.+` to `.*` is caught by the explicit
empty-family guard. So BOTH were sabotaged together, and the grammar test then
fails. A no-op is not a test weakness, but it can conceal an untested guard.

Removing the boolean-version check leaves `version=True` producing `f-vTrue`,
which the PARSER refuses. That is the cross-check doing exactly its job:
`CanonicalDigestSchema` no longer needs its own opinion about what a version is.

---

## 5. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | `git grep -c "-v0"` -- git parsed `-v` as the INVERT-MATCH option | section B of the census reported zero files while `regime-v0` plainly existed |
| 2 | Assumed `SourceAcquisition.as_record` existed | `AttributeError`; the class exposes `canonical_key`, and acquisitions never reach a digest at all |
| 3 | Wrote a test asserting the word "version" appeared in a message I had just rewritten without it | four failures on the first run |
| 4 | Proposed `-W always` for the warning census | it disables once-per-location deduplication, so it would have measured a different quantity than the 33 already recorded |
| 5 | Said `ResourceWarning` was "almost certainly" the warning cause | three captured summaries contain NONE; every one of the 33 is a scikit-learn metric warning |
| 6 | FABRICATED A SUITE DIGEST -- AGAIN | the dry run's own line read `digest unchanged 4a626b2c2832a38` while this record claimed `ea5a4d4cee1eb0aa`, a value in no artifact |

ERROR 1 IS THE SHARPEST. An argument beginning with a hyphen was reinterpreted
by the layer beneath -- the same class as the PowerShell here-string quoting
failure earlier in this arc. `-e` fixes it.

ERROR 5 was reasoning from a plausible mechanism instead of from evidence, and
it is the shape this project punishes most reliably.

ERROR 6 IS THE SAME ERROR I RECORDED ONE UNIT AGO. The previous session record
states, as its own error 4, that "an expected value in an audit must be READ
FROM AN ARTIFACT, never recalled" -- and this record, written immediately
after, asserted an `after_digest` of `ea5a4d4cee1eb0aa` that appears in no
attestation. The real value is `4a626b2c2832a388`, printed by the dry run.

A WRITTEN RULE DID NOT PREVENT THE SECOND OCCURRENCE. So the remedy is
mechanical rather than dispositional, as section XXV of the design authority
requires: this unit's installer now EXTRACTS every suite digest from the
session record and verifies it against the predecessor attestation on disk. A
transcribed digest that no attestation contains is REFUSED before it can reach
a commit.

---

## 6. Findings

### Closed
`DOMAIN-VERSION-SUFFIX-GUARD-ADMITS-A-BARE-V-1`,
`ASCII-INVARIANT-ASSERTED-IN-PROSE-ONLY-1`.

### Registered
`GATE-WARNING-COUNT-INTERMITTENT-1`. Across 44 attestations the warning total
is 33 in 42 runs, 914 once on 2026-08-26, and 37 once on 2026-09-03 -- the last
across an IDENTICAL suite, 41 minutes apart, with only two markdown files
changing. A subsequent full run returned 33 with the same four-group
composition, so the excursion did not reproduce.

`GATE-WARNING-COMPOSITION-NOT-ATTESTED-1`. The attestation records
`"warnings": 33` and nothing else, so the 37 run left no trace of what it saw.
The composition is captured only when the gate FAILS. The design authority's
remedy -- a structured warning group list and a fingerprint -- belongs to its
own tooling unit and must not contaminate a provenance unit.

`UNCLOSED-FILE-HANDLE-SITES-1` supersedes the resource half of the old
`RESOURCE-WARNING-FROM-UNCLOSED-READS-1`: the unmanaged reads are real, but no
observed warning has ever been attributed to them.

### Still open
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` -- closes only when a real
computation opens evidence, captures it, persists the manifest, reloads it, and
downstream logic consumes it.
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
`GATE-TIMING-NOISE-EXCEEDS-TREND-1`.

---

## 7. Ending state

```
HEAD     086e2fa, pushed, +0 -0
ratchet  6110
gate     6095 passed, 15 skipped, 0 failed, 33 warnings
suite    784f3ba581a399ec -> 4a626b2c2832a388
```

Phase 3A++ is complete: 3A++.0 froze the outgoing epoch, 3A++.1 introduced
`CanonicalDigestSchema` and proved it semantic-zero, 3A++.2 migrated source
evidence to v5/schema5, and 3A++.3 gave the grammar one parser.

## 8. Next intended action

Unit 3A++.4: `IdentityConformanceContract`. The design authority places it
before 3B deliberately -- several more identity families are coming, and this
is the last cheap moment to encode the laws once rather than duplicate bespoke
test logic in every future subsystem.

The four relations are already exercised repeatedly across two families:
SAME (invariant transformations), DIFFERENT (scientifically material changes),
REFUSED (ambiguous constructions), and ORTHOGONAL (a change in one family must
not move another). The equivalence-partition check from the v5 migration is the
strongest of them and should be first-class.

Then 3B.0: a repository-wide authority census for media type, content type,
encoding and materialization -- across every tracked file type, not merely
Python -- before any `MediaType` vocabulary is defined.
