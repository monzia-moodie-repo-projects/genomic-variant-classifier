# CORRECTION 2026-08-30 part 2 -- the next line said it was repaired

**Author: Monzia Moodie**
**Applies to:** `6545b64` (`CORRECTION_2026-08-30_a-tool-that-already-knew.md`)
**Status:** the corrected claim is recorded here; `6545b64` is not amended.

---

## 0. What went wrong

`6545b64` claims a systemic pattern:

> This is the third instance of one shape:
>
>     preflight_data_guard.py   recorded of ITSELF that nothing called it
>     drift source kernel       zero production construction sites
>     audit_data_tree.py        ten mentions, zero invocations

**It is the second instance.** `preflight_data_guard.py` IS invoked, and its
own docstring says so on the line after the one I quoted.

---

## 1. `QUOTED-A-FINDING-PAST-ITS-OWN-REPAIR-1`

I read `scripts/maintenance/preflight_data_guard.py` in full on 2026-08-29.
Lines 22 to 27:

```
22| SECOND, AND WORSE: NOTHING EVER CALLED IT. A repository-wide search across
23| every .py, .sh, .ps1 and .yaml on 2026-07-21 found zero invocations of
24| assert_data_usable or of this file. Its own docstring said it was "importable
25| ..."
26| invoked is not a guard; it is a comment that happens to be executable. It is
27| now wired into preflight_run17.run_all() via storage_gate().
```

I quoted through line 26 and stopped. **Line 27 states the repair.**

The finding is dated 2026-07-21 and was recorded TOGETHER WITH its remedy, in
one paragraph, in a file I had read. I carried the diagnosis forward as present
tense and left the cure behind.

That is the failure `docs/CHANGELOG.md` records against me repeatedly --
carrying a finding forward without re-verifying -- committed here inside a
correction whose entire subject was an unverified claim.

### The mention count should have warned me

`preflight_data_guard` is named in TWENTY tracked files, more than any other
maintenance script:

```
preflight_data_guard   20        consolidate_aliases     8
setup_data_tree        12        sync_data_to_gdrive     7
audit_data_tree        10
```

I reported that number in the same measurement and did not ask why the
supposedly-uncalled script was the most-referenced of the five.

---

## 2. What still stands, on its own measurements

**`AUDITOR-EXISTS-AND-IS-NOT-INVOKED-1`.** Ten tracked files name
`audit_data_tree`; all ten were read line by line, and every one is
documentation, a `.gitignore` comment, or the script itself.

**`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1`.** Every tracked Python file
parsed: `SourceEvidenceManifest`, `SourceManifest`, `SourceArtifactKey`,
`SourceArtifactIdentity`, `SourceDependency` and `SourceRegistry` have zero
production construction sites and 85 test sites between them.

Two instances, both measured. What does NOT stand is the claim that the shape
is systemic rather than local -- that rested on three, and one was wrong.

---

## 3. And `preflight_data_guard` is the more useful finding

It is not a counterexample to be set aside. **It is the repair shape**, already
executed once in this repository.

A read-only guard became `storage_gate()` inside `preflight_run17.run_all()`.
Its `storage_rows()` returns `(severity, message)` rows "in preflight_run17's
gate convention", and `check_free_space` "never raises on a healthy
filesystem... a caller decides what to do with WARN".

**That answers the objection I raised against wiring the auditor.** `6545b64`
says:

> its verdict is a WARNING for orphans, and a gate that fails on three known
> orphans would fail every run until they are declared

A severity-row interface makes that a non-problem: the auditor reports
`[warn] external/gencode: ORPHAN`, the gate records it, and the run proceeds.
That is exactly what the storage guard already does with a thin-disk warning.

So the corrected next step is not "decide whether to wire the auditor". It is
"wire the auditor the way the storage guard was wired", and the pattern is
sitting in `scripts/maintenance/preflight_data_guard.py` lines 249 to 258.

---

## 4. What is NOT claimed

That `setup_data_tree`, `consolidate_aliases` or `sync_data_to_gdrive` are
invoked or not. Only their MENTION counts are measured -- 12, 8 and 7 -- and
mention is not invocation. That measurement is prepared and not yet run.

That wiring the auditor is free. `preflight_run17.run_all()` runs before a
cloud run, not on every gate, so an auditor wired there would report orphans at
run time rather than at commit time. Whether that is the right point is a
decision.

That `storage_gate()` is a complete model. It has been read; whether it
tolerates a tool that exits 2 -- as `audit_data_tree.py` does on a
controlled-tier violation -- is unmeasured.

---

## 5. Status

`QUOTED-A-FINDING-PAST-ITS-OWN-REPAIR-1` registered. The "third instance"
claim in `6545b64` is withdrawn; two instances stand.

No file in the repository is changed by this record.
