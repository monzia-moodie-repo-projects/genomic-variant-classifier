# CORRECTION 2026-09-04 part 2 -- a list carried forward drifts in both directions

**Author: Monzia Moodie**
**Applies to:** `05f7868` (`docs/sessions/SESSION_2026-09-04_a-second-equality-site.md`)
**Status:** the corrected claim is recorded here; `05f7868` is not amended.

---

## 0. What went wrong

That record's `### Still open` section, at lines 274 to 291, lists seventeen
finding identifiers. One of them was closed fourteen days before the record
was written:

> `RUNNER-GATE-METADATA-ORDER-1`.

I assembled that list by carrying forward the previous session's, and did not
re-derive it from the records. Nothing in the writing of it would have caught
the error, because nothing in it consulted the repository.

---

## 1. WITHDRAWN: `RUNNER-GATE-METADATA-ORDER-1` from the still-open list

CLOSED at `69ba5f6` on 2026-08-22. Four records agree, and one is a commit:

```
SESSION_2026-08-21_to_08-22_authority-becomes-typed.md:19
    It closed `RUNNER-GATE-METADATA-ORDER-1` at both ends

...:153, under "### Confirmed"
    | `RUNNER-GATE-METADATA-ORDER-1` | acceptance metadata rendered before
      the gate ran; CLOSED by `69ba5f6` |

...:36
    | `69ba5f6` | D3a -- RUNNER-GATE-METADATA-ORDER-1 | `e264e3ca19702caa` |
      5198 passed, 15 skipped, 0 failed, 33 warnings | 954.89s |

commit 69ba5f6
    fix(ratchet): RUNNER-GATE-METADATA-ORDER-1 -- acceptance leaves the ratchet
```

The repair was structural rather than cosmetic. `scripts/install_no_detritus.py`
line 130 now carries the rule it established: *this file certifies COLLECTION
and nothing else; acceptance evidence is owned by the install attestation.*
Acceptance metadata left the ratchet file entirely, so the defect the finding
named cannot recur in the form it named.

`STALE-BACKLOG-CARRIED-A-CLOSED-FINDING-1`.

Had I acted on the carried description rather than re-deriving it, the edit I
would have proposed was to `tests/EXPECTED_SUITE_SIZE` -- the ratchet itself,
bound to three counters and a gate test -- to remove metadata that has not
been there since 2026-08-22.

---

## 2. THE LIST DRIFTS THE OTHER WAY TOO, AND THAT IS WORSE

`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1` was registered OPEN on 2026-08-30. The
document that registered it says so in its own `## 6. Status`:

> Three findings are registered and OPEN:
> `MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1`,
> `AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`,
> `AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`.

Two of those three are on the 2026-09-04 list. The third is not, and no record
closes it. MEASURED at `1fcb1c7`: six mentions, most recent 2026-08-30, under
`## 2. What still stands, on its own measurement`.

Its subject is `audit_data_tree.py` -- ten tracked files name it, all ten were
read, and every mention is documentation, a `.gitignore` comment, or the
script itself. It is the third instance of one shape, after
`preflight_data_guard.py` recording the same of itself and the drift source
kernel.

**A retained closed finding wastes attention. A lost open finding is worse:
nothing points at it.** Both have one cause.

`LIST-CARRIED-FORWARD-DRIFTS-BOTH-WAYS-1`.

---

## 3. A miscount, twice, with the text in front of me

I described the list as holding SIXTEEN identifiers. It holds SEVENTEEN. Line
279 carries two:

```
`FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1`, `HASHING-MIGRATION-PENDING`,
```

I counted lines and reported identifiers. The probe sought seventeen and
printed seventeen sections; it agreed with the record and I did not.

`COUNTED-LINES-NOT-ITEMS-1`.

---

## 4. What the other sixteen actually are

MEASURED at `1fcb1c7` by reading the heading each mention sits under, across
353 tracked markdown files, ordered by each record's own date.

**Fifteen are coherent.** Each shows a progression -- an incident or
correction naming the finding, then `### Registered and OPEN`, then
`### Still open` in every subsequent session record. None contradicts itself.

**`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` is the best evidenced of
them**, and independently re-measured. Twenty mentions across fourteen files,
five commits, and a re-measurement at `42780f4` finding zero construction
sites under `src/` for all six kernel types, with `SourceRegistry` imported by
exactly one file -- its own test. Its closure condition sharpened three times
across the records, and the roadmap now carries the sharpest form.

**The seventeenth needed a document read.** The register showed
`MANIFEST-DECLARES-TWO-SOURCES-IN-ONE-DIRECTORY-1` under a heading beginning
`## 1. REFUTED:` on the same day another record filed it as open. The heading
alone cannot say whether the finding was refuted or something else was.
`CORRECTION_2026-08-30_a-tool-that-already-knew.md` settles it: what was
refuted is the INCIDENT'S PATTERN CLAIM -- that `mim2gene` versus `omim` shows
an established form for one publisher's differently-governed products -- and
this finding is what the refutation PRODUCED. The sequence is coherent.

---

## 5. The register, and what it cost to make it trustworthy

The instrument that produced these figures is external and read-only. Building
it exposed eight defects IN THE INSTRUMENT, every one found by an adversarial
case or by two numbers disagreeing, and none by reading the code:

```
no standard-output reconfiguration     crashed on U+2192 in a heading
traceback written to standard error    flushed 94 lines early, read as absent
false-positive rate calibrated on 8    applied to 1,637 files; 2,408 noise tokens
a gap annotated instead of closed      docs/runs/ excluded by a root list
date rule read filenames only          312 of 904 undated, including six of mine
identifiers split by hard wrapping     one finding hidden, two invented
suppressed token left an empty key     counted as an undated discovery
a substring test where a span was      needed rejoining failed on a short prefix
```

The pattern-versus-corpus choice is the one worth recording. Narrowing the
identifier pattern to remove noise would have DROPPED SIX REAL FINDINGS,
measured. Narrowing the corpus to tracked markdown dropped none. A false
positive costs a glance; a false negative costs a finding.

---

## 6. What is not claimed

That the fifteen coherent findings are open. Their records say so and nothing
contradicts them, which is not the same as a fresh measurement. Only
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` was re-measured directly.

That the register is complete. 902 tokens were discovered across 353 markdown
files; 135 have no dated mention, of which two are canonical shape --
`STATE-ROOT-EXTERNALIZATION-1` and `VERBATIM-IMPORT-NOT-AUTHORING-1`. Those
are real findings the records do not date.

That a heading means open or closed. `### Confirmed` means "confirmed closed"
in one record and could mean "confirmed present" in another. The probe prints
headings and classifies nothing; every reading in this document was done by a
person against the passage.

That the drift is confined to two items. Two were found because two were
checked against evidence. The other 886 identifiers the records name and the
list does not are unexamined.

---

## 6a. One further finding, from choosing this record's date

The coverage probe's section D asserts, at line 324 of
`Probe_SessionRecordCoverage_2026-08-28.py`:

```python
say(out, newest_date == n.group(1),
    "record date equals changelog date", ...)
```

**Strict equality.** The newest changelog heading's date must equal the newest
`SESSION_*.md` filename date. A `CORRECTION_` document is excluded from that
comparison by the probe's own rule, so a correction-only or measurement-only
unit landing on a day with no session record would make the two sides diverge.

That never happened today only because every unit shared 2026-09-04 with
`SESSION_2026-09-04_a-second-equality-site.md`. The invariant holds by
coincidence of date, not by construction.

`SECTION-D-REQUIRES-A-SESSION-RECORD-ON-EVERY-CHANGELOG-DATE-1`.

This record is dated 2026-09-04 because that is the convention two units
already established today, not to satisfy that check. MEASURED: the
`D-CORRECTION` attestation records `finished_at 2026-09-05T01:40:21Z` and its
entry is headed `## 2026-09-04 part 4`; `D-ROADMAP-REPAIR` records
`2026-09-05T02:59:17Z` and its entry is headed `## 2026-09-04 part 5`. Both
crossed midnight in Coordinated Universal Time and both kept the session day.

---

## 7. Why this is a correction and not an amendment

`SESSION_2026-09-04_a-second-equality-site.md` is pinned by digest
`78ce4f1023ce28f38c1c645e069b2f54c392bdebd12054e635443817c66d3165` in the
attestation for `05f7868`. Amending it would break that binding.

The convention is unchanged: corrections sit beside records, never inside
them.

---

## 8. Status

One finding withdrawn from one list, one reported as lost from it, three new
findings registered about how the list was made. Fifteen confirmed by their
own records and one by fresh measurement.

No file in the repository is changed by this record beyond its own creation
and the changelog entry accompanying it.
