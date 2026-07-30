# SESSION 2026-07-30, part three — the three absent metrics, built from the kernels up

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Commit:** `27f6009`, on top of `7a64574`
**Ratchet:** 3898 -> 3959 (+61), computed by collection
**Suite:** 3953 passed, 6 skipped, 0 failed, 805.07 s
**Python:** 3.12.10 in `.venv312`

---

## 1. What changed, and why it is one commit rather than two

`partial_auroc`, `integrated_calibration_index` and
`adaptive_expected_calibration_error` — the three the catalogue has declared
since commit 1 and the registry did not build. Kernels, registry descriptors,
catalogue statuses, the two guards this trips, and a test module, all together,
because **a kernel that is implemented and not registered is an orphan** and this
project already carries three of those: `calibration_slope_intercept`,
`cluster_bootstrap_ci` and `stratified_evaluate`.

    catalogue   24 specified / 21 built / 3 absent   ->   24 / 24 / 0
    registry    21 metrics                           ->   24

**Zero registered absences for the first time.** Every metric the catalogue
declares is now built, registered, and computed on the single path.

---

## 2. What was measured before any of it was written

`partial_auroc` against scikit-learn's `roc_auc_score(max_fpr=...)`, which
implements the same McClish standardisation and is therefore a genuinely
independent implementation rather than a restatement of ours: **1,000
comparisons across 200 random cohorts** — continuous scores, clipped scores,
heavily tied scores, mixed — over bands 0.05, 0.1, 0.25, 0.5 and the full range.
**Worst absolute difference 2.220e-16.**

The two calibration metrics have no external reference in this environment, so
they are pinned by properties a correct implementation must have: zero on a
perfect forecaster, monotone in the size of an injected miscalibration, and
refusing rather than guessing on a cohort where they are undefined.

---

## 3. Two kernel defects found by that measurement and fixed before delivery

### A strict band restriction dropped the curve's vertical segments

A receiver operating characteristic curve is **vertical wherever a tied block is
all one class**. On a 4,000-row cohort whose lowest-scoring rows were all
positive, four points sat at `fpr = 1.0` with the true-positive rate climbing
from 0.9990 to 1.0. The strict form `fpr < high` discarded all four, and the
trapezoid ran a chord across a region the curve does not occupy. **Over-reported
by 2.5e-07.** The bounds are inclusive now, and an edge is interpolated only
where the curve does not already reach it.

**And my written diagnosis of it was wrong.** I recorded that the discarded
segment was at `fpr = 0`, and the two interpolated values printed directly above
that sentence showed `fpr = 0` was clean at 0.0 both ways. A conclusion
contradicted by the data immediately above it — the third hard-coded verdict
caught in my own diagnostics that day.

### De-duplicated quantile edges collapsed the adaptive binning

On a saturated 5,000-row cohort with 41.5 per cent of predictions at exactly 0.0
and 43.3 per cent at exactly 1.0, five of the eleven decile edges were 0.0 and
five were 1.0. De-duplication left **three** edges, and the 15.2 per cent in
between — hundreds of distinct values, and the only region where calibration can
be resolved — collapsed into a single bin. **The metric failed on exactly the
cohort it was added for.**

A value holding at least one bin's share of the mass now takes a bin **to
itself**, and the remaining bins are distributed over the remaining mass. The
same cohort yields ten bins with both pure leaves isolated and eight across the
middle; a continuous vector still gives exactly 500 per bin with zero deviation.

---

## 4. One the project caught in me

`test_an_implemented_entry_matches_its_descriptor[partial_auroc]` failed the
moment the entry flipped to IMPLEMENTED: the registry said *"Standardised partial
area…"* where the catalogue says *"Partial area…"*.

**The registry yields.** The catalogue is the declaration and the registry
implements it; editing the declaration to match the implementation is the same
move as regenerating a baseline to make a difference empty. The standardisation
is not lost — it is in the description and machine-readable in
`parameters["standardisation"] = "mcclish"`, which is where a declaration belongs
rather than in a label.

My installer's post-checks had verified registration and the band parameter but
**not** display-name agreement, so that check was added.

---

## 5. Two defects in the installer, both caught by testing against the real files

**It crashed where it meant to refuse.** The pre-check set `ok = False` for a
missing file and then read that same file two checks later, raising
`FileNotFoundError`. A refusal that raises is not a refusal. It now stops before
reading anything it has already declared missing.

**`display_name="` contains `name="`.** The guard delimiting one catalogue entry
from the next searched for `name="`, matched `display_name="` sitting between a
metric's name and its status, concluded the status belonged to a later entry, and
**refused all three flips**. That is the CI-q substring defect the carried-item
register documents, which I had cited four times that day and then wrote again.
The boundary is now `SpecifiedMetric(`, which cannot be a substring of a field
name. It refused rather than mis-flipping, which is the right failure mode; the
guard was still wrong.

---

## 6. Declared choices, recorded so they are auditable rather than assumed

**The false-positive band is part of `partial_auroc`'s identity**, exactly as a
threshold is for the confusion family, and is carried by a shared object so the
descriptor and the adapter cannot disagree about it. 0.0 to 0.1 is the region the
specification names at line 344: *"partial AUROC within clinically relevant FPR
regions"*.

**McClish standardisation over the mean true-positive rate.** Under the
alternative a random classifier scores (f_lo + f_hi) / 2 rather than 0.5, so the
value would be comparable neither across bands nor against AUROC. The rejected
option is named in the docstring.

**Isotonic regression rather than a local scatterplot smoother** for the
integrated calibration index: monotone by construction, which is the correct
shape for a calibration curve, and deterministic, where a local smoother's span
would have to become part of the metric's identity as the band is for the
partial area.

**None of the three enters `REPORT_METRIC_NAMES`.** They are computed and
registered; whether they join the eight-line printed report is a separate change
against a surface `test_typed_report_surface.py` pins, and it should be made on
its own terms.

---

## 7. The ratchet moved +61 where 58 was predicted

52 from the new test module, 3 from `implemented_names()` in
`test_metric_catalogue.py`, 3 from `all_metrics()` in
`test_registry_vocabulary_completion.py:135`. The remaining three come from
`test_calibration_validity_contract.py:90`, which parametrises over a
calibration-metric collection — exactly what the two new calibration metrics
join. That third file was not found by the search run beforehand.

**A change to a registry reaches into every collection derived from it, and a
hand search misses at least one.** Third demonstration in a single day, and it
cost nothing every time because the number is computed by a real collection
rather than typed.

---

## 8. Figures

    catalogue          24 / 21 / 3   ->   24 / 24 / 0
    registry           21 metrics    ->   24
    ratchet            3898 -> 3959  (+61, collected in 14.38 s)
    README badge       3898 -> 3959  (derived; non-ascii 110, CRLF 0, LF 502, delta +0)
    the four affected files   250 passed
    full suite         3953 passed, 6 skipped, 0 failed, 805.07 s
                       3953 + 6 = 3959
    skip set           unchanged, EIGHTH consecutive run
    diff               644 insertions, 8 deletions, both exactly as declared
    backups removed    5, each at its recorded pre-edit size

---

*Written 2026-07-30. The carried-item register decides status; `tests/EXPECTED_SUITE_SIZE`
decides the count.*
