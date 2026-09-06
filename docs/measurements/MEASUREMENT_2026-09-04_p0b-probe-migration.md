# MEASUREMENT 2026-09-04 -- four instruments learn to state their own limits

**Author: Monzia Moodie**
**Measured at commit:** 6782617
**Scope:** P0-B.2, P0-B.3 and P0-B.4 of the adopted plan.

---

## 0. What this records

`ADR-0005` gave the `Observation` role a typed owner on 2026-09-04, and
`6782617` built it: `repository_measurement`, six modules, thirty-nine tests.
This records what happened when four existing instruments were made to emit
that schema.

Nothing in this record changes the repository beyond its own creation. The
four probes remain external instruments in `C:\Users\monzi\Downloads`, per the
adopted plan's section 48: *the contract is what should be productized*, not
the analysis logic.

---

## 1. The four instruments, and what each now declares

| instrument | mode | selector | members | complete_census |
|---|---|---|---|---|
| Probe_AuthorityCatalog | CENSUS | `tests/**/test_*.py` | 359 | true |
| Probe_SectionConvention | DISCOVERY | `**/*.md` | 356 | false |
| Probe_StillOpenLedger | CENSUS | `docs/sessions/SESSION_*.md` | 140 | true |
| Probe_FindingRegister | CENSUS | `**/*.md` | 356 | true |

Each emitted a payload that `parse_measurement` accepted and that round-tripped
byte-identically through `serialize_measurement`. None imports the checkout.

### Why the section-convention probe is DISCOVERY

Its analysis coverage is 356/356/356/0 -- COMPLETE -- and its
`complete_census` is nevertheless FALSE. Those are different questions. Every
member was read; the METHOD is a fixed-phrase search over fifteen phrases, and
a convention stated in a phrase the list does not anticipate would not appear.

That distinction is the one the adopted plan's section 58 required, and a
single overloaded "coverage" field could not have expressed it.

MEASURED 2026-09-04: that same probe's zero-hit result for `3B` in
`docs/ROADMAP.md` was read as absence and produced
`ROADMAP-DOES-NOT-RECORD-PHASE-3B-1`, withdrawn on measurement because the
roadmap tracks that work under a different vocabulary.
`KEYWORD-SEARCH-ASSUMED-A-SHARED-VOCABULARY-1`.

### Why the finding register is a CENSUS whose evidence is DISCOVERY

Its enumeration is exhaustive: every tracked Markdown file, every line. Its
identifier pattern is a SHAPE heuristic. So `mode` is `census` and the token
evidence carries `discovery` strength -- the first report in which the two
axes diverge, and the reason they are separate fields.

---

## 2. The corpus-identity proof

```
Probe_FindingRegister     **/*.md at 67826176   356 members
Probe_SectionConvention   **/*.md at 67826176   356 members

membership_sha256   624bf7ad2f2db4cf...   IDENTICAL
```

Two independently written instruments produced the same membership digest.
Equal counts would have proven nothing -- that is exactly
`test_equal_member_counts_do_not_mean_equal_corpus`, and the reason the digest
exists at all. Equal DIGESTS prove they read the same 356 files.

That is what makes one probe's `does_not_prove` able to cite the other's
measurement honestly. The ledger now carries:

> That these are all the open-list sections. This probe matches 8 exact
> heading forms; the section-convention census measured 139 DISTINCT headings
> in the open/unchanged family across 177 occurrences at this same commit.

`LEDGER-MATCHED-EIGHT-OF-ONE-HUNDRED-THIRTY-NINE-HEADING-FORMS-1` is no longer
a finding recorded elsewhere. It is a field inside every report that ledger
emits, computed from `len(wanted)` so it cannot go stale if the section list
changes.

The membership digest was recomputed a THIRD time, from the domain-separation
rule alone, against a payload read from disk. All three agree.

---

## 3. One defect, three times, in four instruments

```
Probe_SectionConvention   line 208   data = blobs.get(oid); if None: continue
Probe_StillOpenLedger     line 277   data = blobs.get(oid); if None: continue
Probe_FindingRegister     line 413   data = blobs.get(oid); if None: continue
Probe_AuthorityCatalog               appends to `failures`, correct from the start
```

A blob `git cat-file --batch` did not return was skipped and counted NOWHERE.
A file in the declared corpus that was never read was invisible in the output.
Same shape as `SILENT-NO-OP-REPLACEMENT-1` and as the empty package directory
that imported successfully while six modules were missing: work not done,
reported as nothing rather than as a failure.

All four were written on 2026-09-04 from the same shape, by me. The one that
got it right was written LAST -- after the earlier three had already taught
the lesson without my noticing I had applied it once.

**The repair was forced, not chosen.** `AnalysisCoverage` refuses
`succeeded + failed != attempted`, so there is no honest way to emit coverage
while dropping members silently. The type found what three readings had passed
over. All three now report `0 NOT RETURNED`, measured rather than assumed.

---

## 4. A defect the transport contract caught

The first authority-catalog payload OMITTED the required `evidence` field. The
probe reported success and wrote the file; `parse_measurement` refused it:

```
MeasurementSchemaError: measurement: missing keys ['evidence']
parser exit: 1
```

That is the case for an instrument not judging its own output. A validator
sharing the emitter's blind spot would have accepted it.

The parser also refuses, verified: an unknown key beside a real one, an
unknown schema version, a member list that does not hash to its own declared
identity, a complete census with a parse failure, and PASS or FAIL on a
discovery measurement. `NOT_JUDGED` remains permitted and is not `PASS`.

---

## 5. P0-B.4: what "authoritative probe" means

The adopted plan's section 47 requires this be MEASURED before any universal
compliance claim, because universal quantifiers need authoritative
populations. Four candidate definitions, all measured at `6782617`:

**A naming rule.** 67 tracked files whose name contains `probe`, across
FIFTEEN locations. Four are not Python: one PowerShell script and three `.txt`
files. A filename habit is not a definition.

**A directory.** `scripts/forensics` holds 70 tracked files under SIXTEEN
leading verbs: `probe` 25, `audit` 13, `verify` 12, `diagnose` 6, plus
`smoke`, `cleanup`, `reconcile`, `scan`, `fix`, `characterize`, `investigate`,
`locate`, `confirm`, `read`, `git_gc`. A mixed drawer. `scripts/probes/` and
`probes/` matched nothing -- and that pathspec exited 0 SILENTLY, which is why
the output was read rather than inferred.

**A manifest.** `git grep` for `PROBES = `, `PROBE_REGISTRY`,
`AUTHORITATIVE_PROBE` and `probe_manifest` exited 1. That is a genuine
measured absence, categorically different from the silent zero above.

**An execution surface.** Six workflow lines mention probes, all in
`teardown_abort_diagnostic.yml`, and exactly one probe is invoked:
`scripts/diagnostics/probe_teardown_abort.py`, in a manually dispatched
diagnostic workflow, not a gate. NONE of the four migrated probes is invoked
anywhere -- no workflow, no gate, no test.

`AUTHORITATIVE-PROBE-HAS-NO-DEFINED-POPULATION-1`. Recorded, not repaired.
Defining that population is a governance decision, not a measurement.

**The acceptance criterion is therefore the bounded one the plan supplies:**
all four currently designated probes declare their corpus. That is now true
and demonstrated over a population that can be named.

---

## 6. The register moved, and the movement is now legible

```
                    acd1561      6782617
markdown files          353          356     +3
tokens discovered       902          925    +23
canonical shape         227          240    +13
review bucket           675          685    +10
total mentions        4,362        4,482   +120
no dated mention        135          133     -2
```

Thirteen commits landed between those states, each adding records that name
identifiers. The growth is expected. What changed is that these numbers now
travel with the corpus digest that produced them, so a future comparison is
between two NAMED POPULATIONS rather than between two bare integers.

`ADR-0005` cites the 886-identifier figure as the reason the `Finding` role
remains deliberately unowned. That figure now carries a statement of who
classified it: the canonical and review buckets are a SHAPE sort made by this
probe's author, not a judgement by the records.

---

## 7. What is not claimed

That any of the four probes is authoritative. No artifact defines that
population.

That the schema is complete. It is version 1, and an unknown version is
refused rather than guessed at.

That the finding namespace should have a register. P3 remains deferred, 886
identifiers remain unclassified, and a monolithic register that falsely
normalised heterogeneous objects would be worse than narrative records.

That these instruments belong in the repository. Section 48 says they may
remain external, and no probe was relocated and no script taxonomy changed.

---

## 8. Known and unrepaired

`Probe_SectionConvention` has no output-file option; its docstring instructs
`*>` redirection, and `REDIRECT-MANGLES-NON-ASCII-1` was measured on
2026-09-04 -- the shell re-encodes through the console code page and U+2192
landed as three mojibake characters. It prints Markdown prose, so the hazard
is live. Adding the option is a separate change; bundling it would be
`TWO-CHANGES-ONE-DESCRIPTION-1`.
