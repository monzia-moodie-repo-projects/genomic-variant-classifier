# SESSION 2026-08-27 -- what already governs this

**Author: Monzia Moodie**
**Commits:** `e54c328`, `694da7f`
**Ratchet:** 5583 -> 5591 -> 5642
**Preceding head:** `e12f5c8`
**Ending head:** `694da7f`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `e54c328` | ROADMAP-PROVENANCE | ADDITION +8 | 5583 -> 5591 | 5576p/15s, 1111.3s |
| `694da7f` | DRIFT-PHASE-1B-REPRESENTATION | ADDITION +51 | 5591 -> 5642 | 5627p/15s, 1451.4s |

`ROADMAP-PROVENANCE-CLAIM-STALE-1` closes. DRIFT-1 Phase 1A completes as
measurement; Phase 1B lands as code.

---

## 1. A correction to the record at `e54c328`

**`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1` was ALREADY CLOSED before that
commit claimed to close it.**

MEASURED 2026-08-27 by reading two files in full:

- `tests/unit/test_roadmap_claims.py::test_every_claim_equals_its_live_source`
  binds NINE claim sites to SEVEN live quantities with exact equality, and its
  `CLAIM_SITES` table already carries
  `"snapshot: suite size": (r"Test suite \| \*\*([\d,]+) collected\*\*", "suite")`
  -- the identical regular expression I re-implemented.
- `tests/unit/test_readme_claims.py::test_readme_test_count_equals_the_suite_size_ratchet_exactly`
  binds EVERY README claim site to `tests/EXPECTED_SUITE_SIZE`, and was rebuilt
  on 2026-07-14 because its first version let a stale number through: a
  tolerance of 50 hid a real 17-test drift.

So three of the eight cases installed at `e54c328` DUPLICATE existing coverage,
and the originals are stronger -- they check more sites and carry
vanishing-site guards mine lack. `DUPLICATE-COUNTER-BINDING-1`.

The four that survive are genuinely new: the provenance guard against a
frozen-commit claim, which `CLAIM_SITES` structurally cannot express because it
checks that a NUMBER equals its source and never that the PROSE around it is
honest about where it came from.

**The commit is not amended.** It is pinned by digest and preserved, and the
correction lives here.

### How the duplicate reached a commit

The roadmap probe read the roadmap and never asked what already tested it. That
is the defect the previous ruling named as Layer B -- *"did I discover every
existing owner, consumer and test of this concept?"* -- adopted the same day
and not yet applied.

---

## 2. `ROADMAP-PROVENANCE-CLAIM-STALE-1`

`docs/ROADMAP.md` asserted, in TWO places, that every figure in its
current-state table *"was MEASURED on 2026-08-23 at `f2b93ff`"*. Eleven
consecutive installers then patched the collected count -- 5,436 through 5,583
-- with a SAME-WIDTH substitution and left both sentences standing. The figure
was correct throughout; the PROVENANCE was false from the second commit onward.

A same-width substitution is invisible to a length check, and nothing read the
prose for eleven commits.

**AND READING WAS NOT ENOUGH.** The roadmap was finally read in full, ONE claim
was found and repaired -- and the new test then FAILED on the repaired file,
naming a SECOND claim in the v3 note that the reading had missed.

Reading found one. The predicate found both.

---

## 3. DRIFT-1 Phase 1A: what already governs this

Ten questions, answered by reading rather than by inference. Of the eight facts
Phase 1 needs, SEVEN already have owners:

| fact | owner |
|---|---|
| candidate discovery | `ClinVarTracker.compare` -- computes all three population relations and DISCARDS them |
| population identity | `EvaluationPopulation.membership_fingerprint` |
| feature contract | `TABULAR_FEATURES` + `EXPECTED_TABULAR_FEATURE_COUNT` |
| feature materialization | `engineer_features` |
| preprocessor identity | `PreprocessorIdentity` + `policy_fingerprint` |
| allele equality | `diffcore.column_equal_series` |
| locus derivation | `helpers.locus_key` |
| **source release identity** | **NONE** |

### The eighth had no owner, and my probe said it did

Layer B reported `evaluation/moe_identity.py` as the owner. It matched
`anchor_manifest_sha256` -- a field of `ExpertLineage` about MECHANISTIC ANCHOR
SETS, unrelated to data releases. All three of its output lines were wrong.

`PROBE-AUTHORITY-MATCH-UNVERIFIED-1`: Layer B matches TERMS, not CONCEPTS. It
answers "does this string appear in code" and reports it as "does an owner
exist". Nothing verified that a discovered authority was the right one.

### Variant identity is format-canonical, not biology-canonical

`make_variant_id` is documented as *"Canonical variant identifier:
source:chrom:pos:ref:alt"* -- and the sentence after it says what that means:
*"variants from different sources at the same locus can be matched."* It
canonicalises the STRING FORMAT.

Four differently-named functions each READ as though they solved the biological
case, and none does:

- `test_normalized_allele_equivalence` compares ABSENT-allele tokens --
  `None`, `na`, `.`, `-`, `""`, `null` -- not left-alignment.
- `normalize_allele` maps every empty representation to `<EMPTY>`; `ACGT` stays
  `ACGT`.
- `locus_key` strips the source prefix and nothing else.
- `make_variant_id` formats.

So two spellings of one indel -- `pos=100 ref=AT alt=A` against
`pos=99 ref=GAT alt=GA` -- remain two identities, and
`genuinely_new = new_ids - old_ids` counts the second as novel. The
canonicalisation sabotage would FAIL today, and that makes it a
release-blocking dependency of Phase 1D rather than a refinement of it.

`CANONICAL-MEANS-FORMAT-NOT-BIOLOGY-1`. The word is used for both concepts in
one codebase, which is why a Layer-B scan for `canonical` matched 24,702 lines.

---

## 4. Phase 1B: two identity types that invent no fingerprint

`RepresentationIdentity` says what the COLUMNS are. `SourceManifest` says WHICH
RELEASES produced them. Population identity -- which ROWS -- stays with
`evaluation.population`.

MEASURED before authoring: `policy_fingerprint` is a PURE function of a policy
mapping. No estimator, no fitted state, no data. So a representation carries a
preprocessing digest without instantiating anything -- the property that made
Phase 1B small, and one I stated as an expectation and then verified.

### A manifest, not a release

The semantic plane joins many sources, each with its own cadence. Same ClinVar
variants, new dbNSFP release, CADD moves: the POPULATION did not drift, the
MEASUREMENT PROCESS did. A representation carrying only ClinVar's release
identity cannot tell those apart, so `SourceManifest` carries the complete set
and `differing_releases` names WHICH source moved.

### Two digests derived, one stored

`feature_contract_digest` and `SourceManifest.digest` are PROPERTIES. Storing a
digest beside the data it digests is two fields for one fact.

The attestation schema answered this differently on 2026-08-26 for a reason
that does not apply here: `pre_head` cannot be derived from `pre_head_oid`
because git chooses the abbreviation length, so version 3 records both and
BINDS them. These are derivable, so they are derived.

### Sabotage found two defects in this unit's own design

Order-independence was enforced TWICE -- in `of()` and again in `digest` -- so
removing either left the other and NEITHER could be shown to matter. Repaired
to one authority.

And a test could not isolate `release_id`: the fixture changed it alongside
`artifact_sha256`, so dropping it from the record left the digest still
changing. It proved "something matters", not "the release identifier is part of
the identity".

Fifteen guards sabotaged, fifteen detected after the repairs.

---

## 5. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Installed a duplicate counter binding without asking what already tested it | reading `test_roadmap_claims.py`, which carries the identical regular expression |
| 2 | Layer B reported a false owner for source-release identity | reading `moe_identity.py`, which is about mechanistic anchors |
| 3 | Excluded `.venv` by `in p.parts`, an equality test against a directory named `.venv-drift` | 27,000 third-party matches flooded a scan and truncated it |
| 4 | Attributed two truncated captures to output limits | the third printed `UnicodeEncodeError: '\u2192'` -- my ad-hoc reads had no encoding guard, though every probe I write does |
| 5 | Published pre-correction test files under the correct names | the installer refused on a digest mismatch |
| 6 | A probe claimed `render_ratchet` PREPENDS | it appends -- `kept + entry + count` -- so my "newest/oldest" labels were inverted |
| 7 | A completeness checker reported 0 of 295 lines captured | its segmentation pattern never matched; the capture was complete |

Errors 1 and 2 are one error: **a match is not an authority.** Errors 3, 4, 6
and 7 are another: **a tool's own output believed without checking the tool.**

---

## 6. Findings

### Closed
`ROADMAP-PROVENANCE-CLAIM-STALE-1`.

### Corrected
`RATCHET-MOVING-UNITS-RENDER-THREE-COUNTERS-1` -- already closed by
`test_roadmap_claims.py` and `test_readme_claims.py` before `e54c328`.

### Resolved by measurement
`ATTACHED-ARTIFACT-STATE-AMBIGUOUS-1` -- the delivered `ROADMAP_post.md` is an
installer PAYLOAD carrying 5,583; the repository postimage carries 5,591,
rendered on top. Same size, different digest, exactly as designed.

`TRANSACTION-CANNOT-EXPRESS-DELETION-1` -- the JOURNAL already supports it.
`_write_ahead` captures the preimage and records `existed_before`;
`_restore_target` branches on it. What blocks deletion is the ATTESTATION
target vocabulary: version 3 types `post_sha256` as 64 hexadecimal characters
and a deleted target has no postimage. Expressing deletion needs schema
version 4.

### Registered
- `DUPLICATE-COUNTER-BINDING-1`
- `PROBE-AUTHORITY-MATCH-UNVERIFIED-1`
- `CANONICAL-MEANS-FORMAT-NOT-BIOLOGY-1`
- `REPO-READTHROUGH-LACKS-ENCODING-GUARD-1`
- `PROBE-EXCLUSION-EXACT-MATCH-1`
- `PAYLOAD-STALE-IN-OUTPUTS-1`
- `RATCHET-FILE-GROWS-UNBOUNDED-1` -- 462,691 bytes, 7,377 comment lines, ONE
  bare integer. The roadmap's predecessor was archived at 466,826.
- `RATCHET-GROWTH-SERIES-INCOMPLETE-1` -- my probe's 52-entry extraction covers
  roughly 65 KB of 462 KB, and `test_readme_claims.py` cites "entry 1966". Do
  not build a migration on that series.

---

## 7. Ending state

```
HEAD                    694da7f
ratchet                 5642
gate                    5627 passed, 15 skipped, 0 failed, 0 errors
new subpackage          monitoring/drift/ -- created inside a transaction,
                        directory intents exercised in a delivered installer
                        for the first time
working tree            clean, including untracked
```

### GATE-DURATION-INCREASED-1, twelfth observation

```
892 901 908 | 1403 1354 1364 1570 1400 1410 1305 | 1089 | 1333 1094 1134 1097 1111 | 1451
```

1451.4s is the longest since the band shifted, and this unit added 51 tests --
but they are pure-dataclass cases that run in 0.06 seconds in isolation, so
they cannot account for 340 seconds. Something else moved and nothing measures
it. Recorded, not explained.

## 8. Next intended action

DRIFT-1 Phase 1C: regenerate the reference from the current canonical pipeline
under ONE representation -- NOT by upgrading a 78-feature Run-15 matrix to 95.

`DriftReferenceProfile.from_reference` accepts any frame, so the capability
exists today; `format_version` already hard-errors on mismatch, so the
migration boundary exists too. What does not exist is a reference carrying
population identity, representation identity or source-release identity, and
Phase 1B has now built the last two.
