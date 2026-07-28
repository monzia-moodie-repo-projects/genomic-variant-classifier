# SESSION 2026-07-28 — the verified carried-item register

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `0cc663d`, ratchet 3545
**Roadmap position:** reconciliation, immediately after Tier 1 item 6 completed
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. The defect this closes

Fourteen commits of Tier 1 item 6 accumulated carried items. They were declared
inside per-commit roadmap deltas, and their status changes were recorded in
LATER deltas. An item's current state could therefore only be reconstructed by
grepping a 2,400-line document and reading the deltas in chronological order.

That is two sources of truth for one fact with no divergence detector — the same
defect the metric stack spent fourteen commits removing from the evaluation path.

**It had already gone wrong.** Item CI-l — the transitional label mask
`select_finite_reference_labels` — was retired in commit 2a-1 and still read as
OPEN eleven commits later. The symbol exists nowhere in `src/` as code; the only
occurrences are a docstring reference and a tripwire asserting its absence.

## 2. A namespace collision, found and resolved

`docs/ROADMAP.md` uses single letters for TWO different things:

    root patterns (a)-(d)   recurring failure shapes, lines 1163-1194
    carried items (a)-(s)   deferred work

`carried item (a)` at line 1705 and `root pattern (a)` at line 1163 are unrelated
and sit five hundred lines apart. Carried items now carry the prefix **CI-** and
are never written as a bare letter again.

## 3. What the register is

`docs/CARRIED_ITEMS.md` is the single source of truth for status. Every open item
carries a VERIFICATION PREDICATE — a statement decidable by running code — and
`tests/unit/test_carried_item_register.py` runs them.

**The asymmetry is deliberate.** An OPEN item whose condition has gone is a stale
register: annoying, and caught. A DISCHARGED item whose condition has RETURNED is
a regression: serious, and also caught. Both directions fail, because a register
that detects only one will drift in the other.

An item that cannot be checked goes in an explicit UNVERIFIABLE table, where its
uncheckability is stated rather than implied.

## 4. Status, decided by measurement

    OPEN         CI-m  metrics.evaluate is survivor-filtering
                 CI-n  cohort_version is a weak provenance identity
                 CI-p  to_dict emits NaN while from_dict expects null
                 CI-q  callers still evaluate over unattributed populations
                 CI-r  the report oracle is blind to interval certification
                 CI-s  deferred imports are a load-bearing contract

    DISCHARGED   CI-k  interior-edge agreement coverage        (2b-1)
                 CI-l  the transitional label mask             (2a-1)
                 CI-o  the evaluator abstract-syntax-tree guard (3b-2)

    UNVERIFIABLE CI-i  five Monte Carlo Dropout tests skipped pending a cohort
                 CI-j  four Windows-only tests without continuous-integration coverage
                 CI-a  certification of the canonical seam

**CI-q is worse than recorded.** The item said "migrate callers"; measurement
found `ClinicalEvaluator.evaluate` called at `evaluator.py:1238` WITHOUT
`source_id`, so the package's own batch path produces unattributed populations
and therefore uncertifiable results.

## 5. The register caught its own author, twice

**On its first run.** CI-m's predicate checked for `_clean(` in
`metrics.evaluate` and reported the item CLOSED. The fault was the predicate:
that function filters through `clean_arrays`, and says so in its own docstring —
*"constructs its own population by calling `clean_arrays` and then computes over
the SURVIVORS."* CI-m is very much open.

**In sabotage.** Two mutations survived the first matrix, both weak assertions of
mine rather than gaps in the design:

  * `_discharged_o` checked `"ast" in text`, a substring that matches
    "abstract-syntax-tree", "last", and any mention in a comment. Replacing the
    real import with `import os` went undetected. It now PARSES the module and
    requires both an `import ast` and a call into it.
  * the collision test asserted the phrase `"root pattern"` appeared somewhere.
    The register mentions it twice, so deleting the line that actually EXPLAINS
    the collision left the test green. It now requires both namespaces named WITH
    THEIR RANGES.

## 6. Verification

Regression `FAILED` list byte-identical at 40. Sabotage: **six mutations, six
detected, zero undetected** after the two weak assertions were tightened —
covering both drift directions, a predicate orphaned from the register, an item
listed in both sections, a returning condition, and a reintroduced module-scope
import.

Ratchet 3545 -> 3558 (+13), measured.

## 7. Files

    docs/CARRIED_ITEMS.md                        NEW, the register
    tests/unit/test_carried_item_register.py     NEW, 13 tests

---

*Written 2026-07-28.*
