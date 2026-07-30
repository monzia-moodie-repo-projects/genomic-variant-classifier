# SESSION 2026-07-30, part two — registry commit 2 lands, and the two guards it invalidated

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Commit:** `c4229f1`, on top of `2044102`
**Ratchet:** 3862 -> 3898 (+36), computed by collection
**Suite:** 3892 passed, 6 skipped, 0 failed, 744.78 s
**Python:** 3.12.10 in `.venv312`

---

## 1. What changed

**Registry commit 2 was cut, hash-verified, and never applied.** Measured at
02:24 today, before anything was installed:

    the repository   23 specified, 17 built, 6 registered absences, registry 17
    the payload      24 specified, 21 built, 3 registered absences, registry 21

Neither `50bb9fa` nor `2044102` touched `metrics.py`, `registry.py`,
`catalogue.py` or `test_metric_catalogue.py`, so the tree was still at commit 1
and the payload had been sitting unapplied for a day. Two independent lines of
evidence confirmed it: the byte sizes on disk were the commit-1 sizes, and
`test_metric_catalogue.py` collected 96 rather than 103, which is exactly what
`3 x 23 + 17 + 10` gives against `3 x 24 + 21 + 10`.

**What lands.** `brier_reliability`, `brier_resolution` and `brier_uncertainty`
— the Murphy decomposition of the Brier score — and `brier_decomposition_residual`,
a metric the original specification did not name, added so the identity can be
AUDITED rather than trusted. It is exactly zero only when bins group identical
forecasts; under interval binning it is the within-bin variance term.

**Installed byte-for-byte.** A delivered payload is immutable once cut, so the
five files were copied verbatim — no re-encoding, no line-ending normalisation
— and each was checked against its recorded hash before the copy and again
after. All five are line-feed terminated, pure American Standard Code for
Information Interchange, no byte-order mark.

---

## 2. The ratchet moved +36 and the prediction was +29

The two files the commit obviously touches contribute +7 and +22. The missing
seven are elsewhere: `test_registry_vocabulary_completion.py:135` parametrises
over `all_metrics()`, and the registry grew from 17 metrics to 21.

**The collected count is a property of the WHOLE SUITE, not of the files a commit
changes.** A hand-count cannot see past the files in front of it. This cost
nothing because the number was computed by a real collection rather than typed —
which is the entire reason that rule exists.

The README badge was then DERIVED from the ratchet. Byte invariants held:
non-ASCII 110, carriage-return line-feed 0, line-feed 502, byte delta +0.

---

## 3. Two guards updated, with derived numbers

The full suite after installing the payload reported **2 failed, 3890 passed,
6 skipped**. Both failures were guards working correctly.

### `test_computation_path_guards.py` — the composition allowance

    kernel(s) invoked more often than any registered metric requires:
    {'brier_score': 2}

`metrics.py:1750`, the last line of the residual:

    return float(brier_score(y, prob) - (reliability - resolution + uncertainty))

Every call site of `brier_score` in the module is three: the definition at 378,
the legacy flat dictionary at 1345, and the residual at 1750. Two invocations
during a report — one for the registered metric, one for the metric DEFINED as
Brier minus the three components. That is the `auprc`/`auprc_gain` shape exactly,
and the guard's own comment states the distinction it exists to make.

**The number had to be exact.** The same test asserts the table in the OTHER
direction at lines 187–192: any name invoked FEWER times than declared fails as
"a blanket licence". An inflated allowance would have failed too.

**And note what is ABSENT from the count:** `metrics.py:1345`, the legacy flat
dictionary. If the report path touched it, `brier_score` would show three
invocations rather than two. The observed 2 is independent evidence that the
authority switch of `0cc663d` is holding.

**A measured inference:** `brier_decomposition_residual` carries
`include_in_evaluation_report=False`, yet it is computed during a report — so
that flag controls DISPLAY, not COMPUTATION. The printed report shows eight
metrics; the evaluator computes the whole registry.

### `test_registry_vocabulary_completion.py` — the declared added-name set

Eleven becomes fifteen. The test enumerates rather than counts, deliberately, and
its own comment records it was updated the same way on 2026-07-29 for the
confusion-matrix family.

**The snapshot is NOT regenerated.** `tests/fixtures/registry_snapshot_2b1.json`
is read by four tests in that file at lines 553, 624, 644 and 666. Making the
difference empty by moving the baseline would leave the other three measuring
nothing.

---

## 4. The finding about the payload's provenance

Registry commit 2 was recorded as **"103 passing in the two test files."** The two
test files — not the suite. Adding four metrics to a registry invalidates every
guard that enumerates or budgets that registry, and there were two. The payload
is not wrong; its verification was scoped to its own tests. A full-suite run at
the time it was cut would have surfaced both.

This is the same shape as the +29 prediction being wrong by seven, and it is the
lesson of the day: **a change to a registry has consequences outside the files
that change.**

---

## 5. Defects found in the instruments used to do this work

1. **A prediction that counted only the visible files.** +29 against a measured
   +36. Recorded above.

2. **A heuristic that read past its own boundary.** To decide which Brier metrics
   enter the evaluation report I scanned fourteen lines after each `name=` for
   `include_in_evaluation_report`. The window spills into the NEXT descriptor, so
   it reported `brier_score` excluded and `brier_uncertainty` included — both
   backwards. `REPORT_METRIC_NAMES` at `registry.py:1038` is authoritative and
   was unchanged. Caught before the wrong result was quoted.

3. **A palimpsest, committed while removing one.** The first draft of the guard
   installer anchored on the THREE CONTINUATION LINES of a comment and left the
   original `UPDATED 2026-07-29` header orphaned above the replacement — two
   conflicting dates and one dangling sentence. This is precisely the defect
   criticised in the payload's own docstring that morning. The anchor now covers
   the whole four-line comment, and the post-check was widened from counting one
   particular date to counting `# UPDATED ` headers.

4. **A verification that did not assert its own precondition.** `--verify` ran the
   tests and reported "2 failed" without first checking whether the edits it
   verifies had been applied. A reader could reasonably conclude the fix was
   broken when it had simply never been installed. A check that cannot
   distinguish "broken" from "absent" reports the wrong thing half the time.

---

## 6. Figures

    ratchet              3862 -> 3898   (+36, computed in 19.64 s)
    README badge         3862 -> 3898   (derived from the ratchet)
    catalogue            23 specified / 17 built / 6 absent
                         -> 24 / 21 / 3
    registry             17 -> 21 metrics
    the four affected files   212 passed + 2 failed  ->  214 passed
    full suite           3892 passed, 6 skipped, 0 failed, 744.78 s
                         3892 + 6 = 3898
    skip set             unchanged, SEVENTH consecutive run
    commit               9 files, 437 insertions, 14 deletions
    backups removed      7, each at its recorded pre-edit size

---

*Written 2026-07-30. The carried-item register decides status; `tests/EXPECTED_SUITE_SIZE`
decides the count.*
