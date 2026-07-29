# SESSION 2026-07-29 — the README did not know the evaluation stack existed

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Starting HEAD:** `93177c2`, ratchet 3711
**Roadmap position:** documentation freshness
**Python:** 3.12.10 in `.venv312`; declared floor 3.10; continuous integration runs 3.11 and 3.12

---

## 1. What was wrong

The standing instruction is that the README must ALWAYS be fresh. I have updated its
badge fifteen times across this session and never once read its content.

Measured 2026-07-29, occurrences in a 432-line README:

    typed metric registry        0
    MetricResult                 0
    schema version               0
    carried item                 0
    EvaluationPopulation         0
    certification                0
    ModelComparison              0
    legacy projection            0

**Fifteen commits were invisible to a reader of the README.** The typed metric registry,
the schema-versioned report surface, explicit absence representation, provable
like-for-like model comparison, the input gates, and the self-verifying carried-item
register — none of it appeared.

## 2. A claim I nearly "corrected" wrongly

The README states **95 tabular features**. My working assumption was 91, and had I
trusted it I would have edited a correct document into an incorrect one.

Measured: `variant_ensemble.py:193` defines `EXPECTED_TABULAR_FEATURE_COUNT = 95`. The
count grew from 91 and the README tracked it. **The stale value was mine.**

That is the same fault as the six malformed probes of this session, inverted: instead of
constructing a plausible answer, I nearly overwrote a measured one with a remembered one.
The check cost one command.

## 3. What was written

A new section, **"Evaluation as evidence"**, in the document's existing declarative voice —
explaining *why* each design choice was made rather than cataloguing modules. It covers:

  * one computation path, with the abstract-syntax-tree guard and counting wrappers;
  * a refusal as a typed result carrying status, reason and applicability verdict;
  * thresholds as declared provenance rather than numbers in reporting code;
  * one binning for both calibration errors, and the seventeen-day defect that taught it;
  * populations named or admitted unnamed, with attribution governing claims not values;
  * absence made explicit, with the two causes distinguished;
  * model comparisons that prove they compared like for like, and refuse rather than filter;
  * input gates preceding every library call, and the operating-point sweep that moved a
    clinical decision threshold from a sensitivity of 0.90 to 0.50 in silence;
  * deferred work checked by predicate rather than described in prose.

## 4. Every factual claim verified against the code

    undefined_on_cohort exists            OK
    withheld_by_input_gate exists         OK
    UNKNOWN comparison exists             OK
    invalid_ranking_metric exists         OK
    the quoted auprc refusal matches      OK

That last one was checked literally: a single-class cohort yields status `undefined` with
reason `binary_class_support_required`, which is exactly what the README now says. A
documentation claim is an assertion about the code, and this project has spent a day
establishing that assertions get measured.

## 5. Verification

Regression `FAILED` list byte-identical at 40. No code was touched. Ratchet unchanged at
3711; no test added or removed.

## 6. Files

    README.md   432 -> 502 lines; new section "Evaluation as evidence"

---

*Written 2026-07-29.*
