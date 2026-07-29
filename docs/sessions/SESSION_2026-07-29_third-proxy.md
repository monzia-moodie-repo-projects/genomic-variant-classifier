# SESSION 2026-07-29 — the third proxy, and an empty guard that skipped

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `19f020a`, ratchet 3710
**Roadmap position:** CI-r — the last predicated open item
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. CI-r was half right

The item said the frozen report oracle is blind to interval certification, and
that a defect forcing it `False` would therefore be invisible.

**The first half is accurate.** Both captured fields are `False` in all ten
cohorts, because the capture used `n_bootstrap=0`.

**The second half is false, and measurably so.** Forcing the success-path
assignment at `evaluator.py:1247` to `False` FAILS
`test_evaluator_produces_a_certified_interval_when_genes_are_present`. The
property is covered directly and in both directions by
`test_bootstrap_reconciliation`:

    line 310  a certified interval when gene clusters are present
    line 347  an interval that is NEVER certifiable for a variant unit
    line 359  status and certification as INDEPENDENT axes
    line 296  withheld interval, point metrics still computed

Four other suites assert `certification_eligible is True` as well. An oracle
blind to a property is only a gap when NOTHING ELSE asserts it.

## 2. The third proxy

CI-r's predicate read the frozen fixture and checked whether
`auroc_ci_certification_eligible` had one distinct value. It always will: the
fixture is FROZEN BY DESIGN. The predicate described a file nobody intends to
change rather than a gap in coverage.

That is the third register predicate found measuring a proxy:

    CI-q  a text scan matching four docstrings and an unrelated function
    CI-m  whether a function calls another function
    CI-n  whether a function accepts a parameter
    CI-r  a property of a frozen fixture

It now checks that the POSITIVE assertion exists, and is verified to fail when
that assertion is weakened.

## 3. An empty guard that skipped

Discharging the last predicated item left `OPEN_CONDITIONS` empty, and pytest
skipped the parametrised test:

    SKIPPED [1] got empty parameter set for (item)

**A guard reporting success while checking nothing** — the same shape as the
vacuous biconditional in CI-u-3 and the hardcoded predicate in the CI-m audit. It
would also have raised the suite's stable skip surface from seven to eight.

Rewritten as a loop, so an empty open set passes EXPLICITLY. An empty open set is
a legitimate and desirable state; it should be asserted, not skipped.

## 4. Two malformed probes of my own, recorded

A first mutation matched a line already reading `False` -- `False -> False` --
and its 89 passes proved nothing. A second slice removed `_condition_u` along
with the code it targeted, and the register caught it immediately with
`NameError`.

## 5. Verification

Regression `FAILED` list byte-identical at 40. Ratchet 3710 -> 3711 (+1),
measured. Skip surface unchanged at 7.

## 6. The register is now fully discharged

    OPEN          (none predicated)
    DISCHARGED    CI-k CI-l CI-m CI-n CI-o CI-p CI-q CI-r CI-t CI-u
    UNVERIFIABLE  CI-i  CI-j  CI-a
    RULE          CI-s, verified by a dedicated contract test

Every discharge carries an INVERTED predicate: if any condition returns, the
suite goes red rather than the register quietly reverting.

## 7. Files

    tests/unit/test_carried_item_register.py   CI-r rewritten; empty set made explicit
    docs/CARRIED_ITEMS.md                      CI-r discharged

---

*Written 2026-07-29.*
