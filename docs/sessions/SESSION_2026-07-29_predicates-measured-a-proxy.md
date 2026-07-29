# SESSION 2026-07-29 — two register predicates measured a proxy, not the claim

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `f54af68`, ratchet 3709
**Roadmap position:** register audit, after the last defect closed
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What this found

Two carried items reported OPEN whose stated defects **no longer exist**. Neither
predicate could ever have detected that, because both measured a PROXY rather
than the item's claim.

### CI-m — "a caller cannot tell how many observations a number describes"

    predicate asked : does `evaluate` call `clean_arrays`?
    which is        : TRUE BY DESIGN, permanently

Filtering is the design; asking whether it happens says nothing about whether the
counts are reportable. Measured 2026-07-29 on a six-row cohort with one unusable
score:

    n_input 6, n_dropped 1, survivors 5

`CleanArrays` additionally carries three separate drop counts. **A caller can
reconstruct exactly how many observations each number describes.** The counts
were almost certainly added by the fail-closed work of 2026-07-20 that introduced
`CleanArrays`; the item was never updated.

### CI-n — "`cohort_version` is a weak provenance identity"

    predicate asked : does `_derive_population_source_id` ACCEPT a
                      `cohort_version` parameter?
    which is        : TRUE BY DESIGN, permanently

A parameter-name check cannot observe whether an identity is weak. The item said
"the ordered variant-identifier sequence is what actually distinguishes them" --
and the derivation already incorporates exactly that, stating it in its own
docstring: *"Derived from the cohort version, the selected partition, and the
ORDERED `variant_id` sequence."*

Measured: two frames sharing `cohort_version="v2"` and a partition yield DISTINCT
identities when their variants differ, and also when only the variant ORDER
differs.

### CI-r — checked, and genuinely open

Its predicate reads the frozen fixture and finds one distinct value for
`auroc_ci_certification_eligible`. That is the stated condition measured
directly, not a proxy. The item is correctly open.

## 2. Both predicates rewritten to test the claim

CI-m now calls `evaluate` and checks the counts. CI-n now derives three
identities and checks for collision. Both are verified to fail when the behaviour
regresses -- CI-m when `n_input` is removed, CI-n when the variant identifiers
leave the derivation -- and both are INVERTED rather than deleted, so a
regression re-opens the item.

## 3. A limitation recorded rather than papered over

**Sabotage: six mutations, five detected, ONE UNDETECTED.**

| break | detected |
|---|---|
| B1 CI-m regresses: `n_input` is dropped | yes |
| B2 CI-m regresses: `n_dropped` is dropped | yes |
| B3 CI-n regresses: variant identifiers leave the identity | yes |
| B4 CI-m predicate reverts to a hardcoded proxy | **NO** |
| B5 CI-n predicate reverts to a parameter-name check | yes |
| B6 a discharged item is silently listed as open | yes |

B4 replaces the measurement with `n_input, n_dropped = 6, 1`. The predicate still
calls `evaluate`, still returns the correct verdict, and has measured nothing.

A structural guard was added requiring every predicate to perform a call -- it
catches a predicate that measures nothing at all -- but **detecting a predicate
that calls the code and then ignores the result needs dataflow analysis.** That
limitation is written into the guard itself. A guard claiming coverage it does
not have would be worse than one that states its boundary.

The guard's first version also required an import inside the function body and
fired on CI-r, which imports at module scope. A guard that cries wolf on correct
code gets weakened until it catches nothing, so it was relaxed to calls only.

## 4. The pattern this completes

Three register predicates have now been found to measure proxies:

    CI-q  a text scan matching four docstrings and an unrelated function
    CI-m  whether a function calls another function
    CI-n  whether a function accepts a parameter

All three would have reported their items open indefinitely. The register was
built to stop status drifting from code, and its own predicates were the drift.

## 5. Verification

Regression `FAILED` list byte-identical at 40. Legacy report oracle moves only
`schema_version`.

## 6. Files

    tests/unit/test_carried_item_register.py   two predicates rewritten, one guard added
    docs/CARRIED_ITEMS.md                      CI-m and CI-n discharged

Ratchet 3709 -> 3710 (+1), measured.

## 7. The register after this

    OPEN          CI-r  CI-s
    DISCHARGED    CI-k CI-l CI-m CI-n CI-o CI-p CI-q CI-t CI-u
    UNVERIFIABLE  CI-i  CI-j  CI-a

---

*Written 2026-07-29.*
