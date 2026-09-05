# CORRECTION 2026-09-04 part 3 -- a closed finding, repeated in the correction about repeating closed findings

**Author: Monzia Moodie**
**Applies to:** `acd1561` (`docs/sessions/CORRECTION_2026-09-04_part2_a-list-carried-forward-drifts-both-ways.md` and its changelog entry)
**Status:** the corrected claim is recorded here; `acd1561` is not amended.

---

## 0. What went wrong

`acd1561` withdrew `RUNNER-GATE-METADATA-ORDER-1` from a still-open list
because it had closed fourteen days earlier, and registered
`STALE-BACKLOG-CARRIED-A-CLOSED-FINDING-1` over it.

The same document carried a closed finding as open, in the paragraph
immediately following.

---

## 1. WITHDRAWN: the claim that the auditor has no caller

`acd1561` states, and its changelog entry repeats:

> Its subject `audit_data_tree.py` has ten mentions across tracked files and
> zero invocations.

MEASURED 2026-09-04. That was true on 2026-08-30 and false when I wrote it.

`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1` was CLOSED at `fd6cd4e` on 2026-08-30 --
*feat(preflight): the auditor stops being a comment that happens to be
executable* -- five days before I repeated the claim. I carried the sentence
verbatim from `CORRECTION_2026-08-30_a-tool-that-already-knew.md` and never
re-measured it.

`STALE-CLAIM-REPEATED-IN-THE-CORRECTION-ABOUT-STALE-CLAIMS-1`.

### The closure is proven at all three levels, by execution

`tests/unit/test_data_tree_gate.py`, 14 tests, ALL PASSED, ZERO SKIPPED, in
15.43 seconds at `2c94ae3`:

```
existence     audit_rows and audit_tree are callable
invocation    run_all drove a monkeypatched gate; seen["called"] is true
consequence   the sentinel row returned through run_all's rows, and
              return_code == 2 holds exactly when a FAIL row is present,
              both from one AuditReport
```

The skip paths were the reason for running it verbosely: `_preflight()` skips
if the module will not import, and `test_run_all_actually_calls_the_data_tree_gate`
skips if `run_all` raises. Neither fired. A skipped test reads as green, and
that guard's own text concedes it *"MUST NOT skip on the development machine
or in CI"*.

### Why the original measurement was wrong, which is the reusable part

The wiring loads by path -- `importlib.util.spec_from_file_location` -- so a
scan for `import audit_data_tree` reports ZERO while the gate is demonstrably
called. The guard records that the same scan counted 31 "invocations" of
`preflight_data_guard`, EVERY ONE A LINE OF MARKDOWN PROSE.

Wrong in both directions at once, from one static search.

---

## 2. RESTATED: fifteen findings were called "coherent" on a predicate that
does not establish state

`acd1561` reports that fifteen of the seventeen entries are coherent, each
progressing from an incident or correction, through `Registered and OPEN`, to
`Still open` in every subsequent session record.

That is a statement about NARRATIVE CONSISTENCY. It is not a statement about
current state, and the adopted governing plan rules the distinction directly:
*do not infer finding state from narrative headings*, because those headings
have accumulated heterogeneous semantics over months.

MEASURED at `2c94ae3` -- every commit naming each identifier, across all
branches, by fixed-string match:

```
identifier                                          commits    kinds
AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1                       5    docs + ONE feat
DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1            5    docs + ONE feat
SOURCE-IDENTITY-ERROR-NOT-EXPORTED-1                      1    docs only
FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1                  1    docs only
CONFIG-DECLARES-A-PATH-NOTHING-READS-1                    1    docs only
CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1                1    docs only
VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1               1    docs only
AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1            1    docs only
MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1          2    docs only
CONNECTOR-SOURCE-NAMES-DISAGREE-WITH-THE-MANIFEST-1       1    docs only
DATABASE-CONNECTORS-NOT-BYTE-EXACT-BY-TRANSCRIPT-1        1    docs only
GATE-TIMING-NOISE-EXCEEDS-TREND-1                         1    docs only
GATE-WARNING-COUNT-INTERMITTENT-1                         1    docs only
GATE-WARNING-COMPOSITION-NOT-ATTESTED-1                   1    docs only
INSTALLER-HEADER-UNDERSTATES-A-MIXED-TRANSITION-1         0    NONE
HASHING-MIGRATION-PENDING                                 0    NONE
UNCLOSED-FILE-HANDLE-SITES-1                              0    NONE
```

**The auditor is the only identifier whose `feat` commit closed it.** That is
why it is the only closure I missed, and finding it was luck: a ratchet entry
happened to read *"An auditor acquires a caller"*.

The correct status for the fourteen `docs`-only entries and the three named by
no commit is **UNDETERMINED BY AUTHORITY** -- no authority has ruled, so
neither "open" nor "closed" is established. Calling them coherent was accurate
about the records and silent about the state.

---

## 3. A `feat` commit naming a finding is NOT a closure

`accdf49` -- *feat(monitoring): the factory stops stringifying whatever it is
handed* -- names `DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1`. It does not
close it. Its message states the condition under which the finding WILL close.

`docs/ROADMAP.md` states the rule in its own words: *a mention is not a
closure: a commit may cite an item to say it is open, deferred, or blocking.*

That finding is OPEN, and it is open by MEASUREMENT rather than by narrative:
re-measured 2026-09-04 at `42780f4` across all 1,063 tracked Python files with
zero parse failures, the six source-kernel types have ZERO construction sites
under `src/`, and `SourceRegistry` is imported by exactly one file -- its own
test. The twenty-three sites elsewhere are two `scripts/dev/` tools.

---

## 4. What is not claimed

That the fourteen are closed. Nothing suggests they are. The claim withdrawn
here is that heading sequence established they were open.

That the three unnamed findings are inert. `HASHING-MIGRATION-PENDING`,
`INSTALLER-HEADER-UNDERSTATES-A-MIXED-TRANSITION-1` and
`UNCLOSED-FILE-HANDLE-SITES-1` are named by no commit on any branch, which
means only that no commit message mentions them.

That commit-message search is a sufficient authority test. It is a HIGH-RECALL
DISCOVERY step. A finding could be closed by a commit whose message never
names it -- which is precisely how the auditor's closure would have been
missed had `fd6cd4e` been titled differently.

That the rest of `acd1561` is affected. `RUNNER-GATE-METADATA-ORDER-1`'s
withdrawal stands on four independent records including a commit and a
test-bearing repair. `COUNTED-LINES-NOT-ITEMS-1` stands: the list holds
seventeen identifiers, and line 279 carries two.

---

## 4a. A fabricated digest, made while building the unit that carries this

`FABRICATED-DIGEST-5`. The installer publishing this record pinned
`ENTRY_SHA` as a sixty-four character value of which THIRTY-TWO were invented.
My own validation had printed `hexdigest()[:32]`; I wrote sixty-four. The
measured value is
`1c8ff206ccebcc3f5c2ea3270c330e69bd739f06e8add648ea0b13f9512fcf9c`, and
exactly 32 of 64 characters agreed -- the signature of the mechanism.

This is the FOURTH fabricated digest in this programme and the SECOND on
2026-09-04. `FABRICATED-DIGEST-4` was caught an hour earlier, and its warning
was written into that same installer's docstring: *the abbreviated string is
display metadata, never reusable cryptographic evidence.* The rule was obeyed
for the preimage it named and broken for the payload in the same file.

**That is the shape this whole record is about.** A rule applied to the
REMEMBERED INSTANCE rather than to the CLASS -- the same failure as repeating
the auditor claim in a correction about repeating stale claims.

The repair is therefore mechanical rather than remembered.
`require_full_sha256` refuses anything that is not sixty-four lowercase
hexadecimal digits, `_validate_pins()` runs it over every pinned constant
before the installer does any work, and the digest helper itself passes
through it. `FULL-DIGEST-GUARD-IS-MECHANICAL-1`.

The structural cause is mine: I truncate digests in my own verification output
and then reuse them. Both fabrications came from my display, not from any
external report.

---

## 5. Why this is a correction and not an amendment

`CORRECTION_2026-09-04_part2_a-list-carried-forward-drifts-both-ways.md` is
pinned by digest
`dc903d3d081ff7214a5680ead75741ac7f87d17999a72716ef11ca404c2df24d` in the
attestation for `acd1561`, and `docs/CHANGELOG.md` is append-only. Amending
either would break that binding.

`CORRECTION_2026-08-30_a-tool-that-already-knew.md` is NOT corrected. Its
statement was true on 2026-08-30 and describes the belief at the time
accurately. A historical record that was right when written remains a
historical record.

---

## 6. Status

One claim withdrawn, fifteen restated from *coherent* to *undetermined by
authority*, one finding confirmed closed by three-level executable proof, and
one confirmed open by direct measurement.

No file in the repository is changed by this record beyond its own creation
and the changelog entry accompanying it.
