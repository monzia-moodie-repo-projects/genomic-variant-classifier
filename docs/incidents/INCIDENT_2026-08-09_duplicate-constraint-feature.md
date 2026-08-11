# INCIDENT 2026-08-09 -- `gene_constraint_oe` was a bit-identical alias of `loeuf`

**Author: Monzia Moodie**
**Opened: 2026-08-09**
**Severity: HIGH -- affects the published Run 15 feature-importance result**
**Status: root-caused; repair specified as DUPLICATE-1A**
**Repository: `github.com/monzia-moodie-repo-projects/genomic-variant-classifier`**

---

## 1. THE FINDING, IN ONE LINE

Two of the ninety-five features in `TABULAR_FEATURES` were **the same column**.

    identical    : True
    max abs diff : 0.0
    correlation  : 1.0

measured on `outputs/run17_prepcheck/full/splits/X_train.parquet`, 1,038,974
rows, on 2026-08-09. The Run 15 matrix shows the same signature: both columns
report `n_unique` 1866, modal fraction 0.055397 and modal value 0.6322374378.

---

## 2. HOW IT WAS FOUND

Not by review. By an instrument that had no interest in the question.

A feature-vitality census, built to detect columns that carry no signal, printed
`gene_constraint_oe` and `loeuf` with byte-identical statistics on every field
it reports. The duplication was visible only because the census enumerates
per-feature statistics rather than a pass/fail count.

**No existing gate could have caught it.** The suite-size ratchet counts tests;
the zero-variance guard tests for constancy and both columns vary; the feature
count guard asserts `EXPECTED_TABULAR_FEATURE_COUNT == len(TABULAR_FEATURES)`
and 95 == 95 held throughout. A duplicated feature is well-formed by every
contract the project had.

---

## 3. ROOT CAUSE

Three lines, in two files.

    connector_gnomad_constraint.py:66   _COL_LOEUF = "lof.oe_ci.upper"  -> loeuf

    variant_ensemble.py:846             feats["gene_constraint_oe"] = df.get(
    variant_ensemble.py:847                 "gene_constraint_oe",
                                            df.get("loeuf", <constant>))

    variant_ensemble.py:849             feats["gene_is_constrained"] =
                                            (feats["gene_constraint_oe"] < 0.35)

The connector names its source correctly. `engineer_features` then falls back to
`loeuf` when a `gene_constraint_oe` column is absent from the frame -- and it
was always absent, because nothing ever produced one. The fallback was not a
fallback; it was the only path.

`monitoring/registry.py:109` records the arrangement plainly:
*"constraint_metrics.tsv -> loeuf-derived oe"*. It was documented. It was
documented as if correct.

### 3.1 The scientific error that justified it

`scripts/patch_constraint_oe_from_loeuf.py:10`:

> *"LOEUF **is** the LoF observed/expected upper-bound fraction, so loeuf is
> the correct source."*

The first clause is true and the conclusion does not follow. **LOEUF is the
upper bound of the 90 per cent confidence interval around the loss-of-function
observed/expected ratio. The ratio itself is the point estimate.** They are
different statistics, and the source column name says so: `lof.oe_ci.upper` is
not `lof.oe`.

The gnomAD constraint table publishes both. `lof.oe` was two columns away in
the same file the connector already reads.

---

## 4. WHAT THIS MEANS FOR THE RUN 15 RESULT -- FOUR CLAIMS

Stated precisely, because it is easy to correct either too little or too much.

1. **`gene_constraint_oe` was not a distinct predictor.** It was an exact
   duplicate of `loeuf`. These two feature slots represented ONE quantity, so
   the 95-feature contract contained **at most 94** distinct quantities. The
   exact count is not established until the pre-transform duplicate gate has
   been run over the whole contract; it may be fewer.
2. **`gene_is_constrained` was a thresholded LOEUF derivative**, despite being
   implemented through the mis-named `gene_constraint_oe`. Its values were
   never affected; its dependency graph was wrong.
3. **Individual importance attributed to `gene_constraint_oe` versus `loeuf` is
   NOT IDENTIFIABLE.** Either duplicate supplies the same split information, so
   feature order, randomness and implementation details determine where
   importance lands.
4. **The valid historical result is family-level.** The roadmap's

   > *"`gene_constraint_oe` REVIVED (Run-14 all-zero -> Run-15 #2 feature)"*

   must become

   > **"The gnomAD pLoF-constraint feature family ranked highly in Run 15.
   > Attribution among `loeuf`, its duplicated `gene_constraint_oe` alias, and
   > the threshold-derived `gene_is_constrained` indicator is not separately
   > identifiable in that run."**

The attribution group is **`{loeuf, gene_constraint_oe, gene_is_constrained}`**.
The third member is not an exact duplicate but is a deterministic transform of
the same underlying signal.

### 4.1 What is NOT claimed -- and what remains UNMEASURED

An earlier draft of this incident said model performance "is not impugned" and
that "tree ensembles are indifferent" to duplication. **Both were too strong,
and are withdrawn.**

For L2-regularised logistic regression, if two identical columns carry
coefficients b1 and b2, only their sum b = b1 + b2 affects the likelihood. For
fixed b the penalty is minimised at b1 = b2 = b/2, giving

    b1^2 + b2^2  =  2 (b/2)^2  =  b^2 / 2

against b^2 for a single column. **Duplication halves the effective
regularisation along that signal direction.** Tree ensembles with column
subsampling, randomised candidate selection or tie-breaking are likewise not
mathematically indifferent to a duplicated predictor.

The defensible statement is:

> **No evidence from DUPLICATE-1 establishes that Run 15's predictive
> performance was materially harmed, and the reported metrics remain empirical
> observations of the model that was actually trained. But performance
> INVARIANCE is not established either: duplicate predictors alter effective
> regularisation in penalised linear models and can alter training paths in
> stochastic and tree ensembles. The demonstrated defect is to interpretation.
> The counterfactual performance without duplication is UNMEASURED.**

---

## 4.2 A SECOND, DISTINCT DEFECT FOUND IN THE SAME BLOCK -- CONSTRAINTFILL-1

Reading `variant_ensemble.py:846-849` verbatim exposed a defect that is
related to DUPLICATE-1 and is **not the same defect**:

    feats["gene_constraint_oe"] = df.get(
        "gene_constraint_oe", df.get("loeuf", pd.Series([1.0] * len(df), ...))
    ).fillna(1.0)
    feats["gene_is_constrained"] = (feats["gene_constraint_oe"] < 0.35).astype(int)

**`.fillna(1.0)` is a biological assertion wearing a default's clothing.** An
observed/expected ratio of 1.0 means observed equals expected -- the gene is
completely tolerant of loss-of-function variation. So a gene with NO constraint
data is asserted to be UNCONSTRAINED, and the model cannot distinguish the two
states. It then propagates: `1.0 < 0.35` is False, so every gene without data
is also recorded as not constrained.

    DUPLICATE-1       a KNOWN LOEUF was mislabelled as observed/expected.
    CONSTRAINTFILL-1  an UNKNOWN constraint was mislabelled as biological
                      neutrality.

The two are filed separately because the distinction will matter historically.

**Scale: not yet measured.** `loeuf` and `gene_constraint_oe` share a modal
fraction of 0.055397 in the Run 15 matrix, which is suggestive of roughly
57,600 filled rows -- but genuine measurements can equal the fill value, so the
imputation footprint must be measured BEFORE `.fillna()` rather than inferred
from a modal fraction afterwards.

**Repair:** DUPLICATE-1A preserves missingness (`NaN`, never `1.0`), and
CONSTRAINTFILL-1 lands immediately afterwards as a coupled commit that gives
every model an explicit missing-value policy fitted on the TRAINING partition
only. Neither may merge alone: the first without the second would leave the
linear, support-vector and neural paths unable to consume the matrix.

Two contracts, held apart:

    DUPLICATE-1A       feature engineering does not invent biological evidence
    CONSTRAINTFILL-1   every model either consumes missing values natively or
                       declares a training-fitted imputation policy

`gene_is_constrained` becomes three-valued -- 1, 0, or NA when LOEUF is
unavailable -- because `np.nan < 0.35` evaluates False and would recreate
CONSTRAINTFILL-1 one layer downstream.

---

## 5. WHY THE SOURCE FILE MADE THE REPAIR NON-TRIVIAL

`gnomad.v4.1.constraint_metrics.tsv`, 55 columns, 211,523 rows. MANE Select
rows only:

    34,962 rows
    17,486 distinct gene symbols
         8 rows with a NULL gene symbol

gnomAD emits each MANE Select transcript **twice**, once per annotation
namespace:

    gene_id '26009'           transcript 'NM_015534.6'      RefSeq
    gene_id 'ENSG00000036549' transcript 'ENST00000370801'  Ensembl

The Ensembl row carries the richer record; the RefSeq row has rank, decile,
chromosome, coding length and exon count as null. **The biological metrics are
identical:** LOEUF agrees within a gene symbol for 17,486 of 17,486 genes, with
zero disagreements. This is namespace duplication, not two competing
measurements.

Three pairing shapes:

| shape | count | example |
|---|---|---|
| pair sharing one symbol | 17,468 | `ZZZ3` |
| pair split by SYMBOL DISAGREEMENT | 5 | `SCHIP1` / `IQCJ-SCHIP1` |
| pair split by null Ensembl symbol vs provisional RefSeq `LOC*` | 8 | `LOC728392` / `ENSG00000286190` |

**Every metric-bearing row has a partner. There are no genuine
single-representation genes** -- only pairs the symbol cannot join.

### 5.1 Three key rulings, each forced by a measurement

- **Never `drop_duplicates`.** The surviving row would depend on source order.
- **Never key on `gene_id` alone.** Measured: `gene_id` is unique PER ROW here,
  so keying on it preserves the duplication rather than resolving it. (General
  advice to prefer stable identifiers over symbols is sound and does not apply
  to this file.)
- **Never let `groupby` drop nulls implicitly.** pandas defaults to
  `dropna=True` and would discard the 8 null-symbol rows silently.

---

## 6. THE REPAIR -- DUPLICATE-1A

Delivered as `constraint_canonicalize.py` with a dedicated test suite covering sabotage-tested conservation, namespace equivalence, the
arithmetic identity, missingness, the dependency graph and hash-collision
contracts. The count is deliberately not restated here -- the suite ratchet
owns it, and a number written in prose is a second copy nobody maintains.

    MANE Select
      -> exclude null gene symbols        (8, recorded; partners keep the metrics)
      -> group by gene symbol, dropna=False, with conservation assertions
      -> assert namespace metric equivalence   (upstream-drift tripwire)
      -> prefer Ensembl; explicit RefSeq fallback
      -> assert uniqueness by symbol

Yielding **17,486 canonical rows**, unique by symbol, with every input row
accounted for: 34,954 retained plus 8 excluded.

**Feature contract, after repair:**

    lof.oe            -> gene_constraint_oe     MODEL FEATURE
    lof.oe_ci.upper   -> loeuf                  MODEL FEATURE
                           |
                           +-> gene_is_constrained   derived DIRECTLY from loeuf

    lof.obs, lof.exp, oe_exceeds_reported_upper_bound   AUDIT RECORD ONLY

`lof.obs`, `lof.exp` and the audit flag are ingested for source validation and
are **deliberately not added to `TABULAR_FEATURES`**. Source-validation
information does not automatically become predictive information, and this
incident repairs an identity rather than growing the roster.

### 6.1 Two invariants the connector now asserts about its own science

**The arithmetic identity.** `lof.oe` must equal `lof.obs / lof.exp`. Measured
across 191,811 rows: maximum absolute error **3.25426e-4**, consistent with
four-decimal publication. Tolerance `5e-4`. Conditional -- only finite triples
with `exp > 0` are checkable, and an uncheckable row must not count as passing.

**Namespace equivalence, at ZERO tolerance.** Two representations of one gene
must agree on every biological metric AND on which values are missing. Today
they agree on all 17,486. If a future release ever disagrees, the build fails
rather than silently preferring one -- which is the reason to collapse rather
than filter.

Two points here were established by sabotage rather than by reasoning:

- **Missingness is part of equivalence.** Comparing after `dropna()` would let
  a populated RefSeq value and a missing Ensembl value look equivalent -- and
  since Ensembl is then preferred, a real measurement would be silently
  replaced by missing data.
- **The tolerances are separate.** `NAMESPACE_ATOL` is 0.0 while
  `OE_ARITHMETIC_ATOL` is 5e-4. The arithmetic check compares an independent
  calculation against published four-decimal fields, so rounding is expected;
  the namespace check compares two ENCODINGS of one record, where no
  calculation occurs and nothing may differ. The constant was created and the
  call site kept passing the arithmetic tolerance, so a 1e-9 disagreement was
  tolerated until a test demanded it raise.

### 6.2 `gene_is_constrained` -- threshold deliberately NOT changed here

The flag stays derived from `loeuf < 0.35`, and its dependency is repaired to
read `loeuf` directly rather than route through the point estimate.

**Why the threshold does not move in this commit.** Applying `< 0.35` to the
point estimate instead of the upper bound was measured:

| criterion | genes | median `lof.exp` |
|---|---|---|
| LOEUF < 0.35 but oe >= 0.1679 -- **lost** | 468 | **92.2** |
| oe < 0.1679 but LOEUF >= 0.35 -- **gained** | 94 | 15.7 |
| both | 874 | 53.7 |

Overlap at the count-matching cut-off is **Jaccard 0.6086** -- roughly a third
of the set turns over. The lost genes are large and well powered; the gained
ones are small and imprecise. **A point-estimate cut-off systematically discards
well-powered constrained genes and admits underpowered ones**, which is exactly
what LOEUF exists to prevent.

**And 0.35 is not current gnomAD guidance.** It is the v2 threshold. gnomAD
v4.1.1 (30 March 2026) recommends **LOEUF < 0.45**, corresponding to the 15th
percentile of 17,063 MANE Select transcripts; the v2 threshold of 0.35 sat at
the 16.7th percentile, which maps to 0.47 in v4.1.1. The comment at the
threshold records this and names **CONSTRAINTPOLICY-1**, so the value is
retained project policy rather than a claim about current practice.

Measured on the local v4.1 substrate, one canonical row per gene:

    LOEUF < 0.35   1,353 genes    7.8th percentile
    LOEUF < 0.45   2,283 genes   13.2th percentile
    LOEUF < 0.60   3,784 genes   21.8th percentile

The 13.2 against gnomAD's 15 is a real v4.1/v4.1.1 difference and is the
measured justification for **DUPLICATE-1B**.

### 6.3 Twelve rows where the point estimate exceeds its reported upper bound

`DNMT3A` (`lof.exp` 106.98), `TET2` (128.09), `LZTR1` (100.58) and nine others
have `lof.oe` between 2.0 and 2.8 against reported upper bounds clustered just
below 2.0. These are **well powered**, so a small-count guard cannot remove
them, and `lof.oe` is arithmetically sound: `267 / 106.98 = 2.4958` for
`DNMT3A`, matching the file.

**The value is kept unclipped.** `min(oe, loeuf)` would manufacture a statistic
gnomAD never published. A descriptive flag records the observation:

    oe_exceeds_reported_upper_bound = (oe.notna() & loeuf.notna() & (oe > loeuf))

and asserts **nothing about why**. An earlier draft of this incident called the
upper bound "censored at 2.0". That inferred a mechanism from a clustering
pattern and is **withdrawn**. Filed as **CONSTRAINT-CI-1**, to be re-measured
after the v4.1.1 migration.

---

## 7. THE GATE THAT WOULD HAVE CAUGHT IT

`exact_duplicate_groups` fingerprints each column with a content hash, groups by
digest, and byte-compares only collisions -- linear in features rather than
quadratic.

Two design points, each established by a sabotage run rather than by reasoning:

**It runs on the PRE-TRANSFORM matrix.** After standardisation every constant
column becomes 0.0, so two unrelated dead features would be reported as
duplicates and the gate would drown in noise.

**Degenerate columns are skipped.** A constant column is a *vitality* failure,
which the vitality contract owns, not a duplicate-signal failure.

---

## 8. WHAT THE SABOTAGE RUNS EXPOSED IN THE GUARD ITSELF

Eight mutations were applied to the canonicaliser. Four were caught immediately.
The other four named properties that were asserted and never exercised: a
`dropna=False` that no fixture could reach, a conservation assertion that never
fired because no clean fixture loses rows, a degenerate-column skip whose
fixture used two *different* constants (different hashes, so the skip was
irrelevant), and a hash-collision path with no collision to exercise it.

The most serious was found only after repairing those. **The conservation check
could be satisfied by cancellation.** Removing the explicit null-symbol filter
and reverting `dropna` together made `groupby` drop the row silently -- yet the
excluded COUNT still reported one, because it was computed from the *intended*
exclusion rather than an observed one. `8 + 1 = 9` balanced and the loss passed.

The identity is now checked twice, and neither can cancel:

    n_grouped  == n_retained                    every retained row reached the loop
    n_retained + n_excluded == n_mane           nothing vanished before it

> **A count of what was meant to be excluded is not a measurement of what was.**

---

## 9. FOLLOW-UP ITEMS FILED

| identifier | scope |
|---|---|
| **DUPLICATE-1A** | this repair: the alias, the canonicaliser, the invariants, the duplicate gate |
| **CONSTRAINTFILL-1** | HIGH. `.fillna(1.0)` asserts biological neutrality for missing constraint. Coupled to DUPLICATE-1A; neither merges alone. A Run 17 gate |
| **DUPLICATE-1B** | migrate the substrate from gnomAD v4.1 to v4.1.1 |
| **CONSTRAINTPOLICY-1** | move `gene_is_constrained` from 0.35 to the v4.1.1-recommended 0.45, with 0.60 as a prespecified sensitivity stratum. **Selection must be external and prespecified, never chosen by whichever value improves a metric** |
| **CONSTRAINT-CI-1** | re-measure the twelve `oe > upper` rows against v4.1.1 before asserting any mechanism |
| **CONSTRAINTKEY-1** | closed inside DUPLICATE-1A; its empirical answer is known |
| **FEATURELINEAGE-1** | lineage groups, so importance can be reported per feature AND per source family. Exact-duplicate detection protects implementation correctness; lineage-aware attribution protects scientific interpretation |
| **CLINVARLOC-1** | measure whether ClinVar ever annotates a variant to a provisional `LOC*` symbol, which determines whether retaining those eight genes is load-bearing or merely correct |
| **DUPGATE-2** | distinguish BIT_IDENTICAL from NUMERIC_DUPLICATE. `Series.equals` compares dtype, so `[1,2,3]` int64 and `[1.0,2.0,3.0]` float64 would evade a value-level detector. Filed with FEATURELINEAGE-1; not a blocker here, because the observed alias was bit-identical |

---

## 10. THE DURABLE LESSON

The defect survived because it was **well-formed**. Every contract the project
had was satisfied: the feature count matched its constant, no duplicate names
existed, both columns varied, and the arrangement was documented in the source
registry.

> **A contract that checks names and counts cannot see that two features are
> the same feature. Identity is a property of values, and only a value-level
> check can measure it.**

And the scientific half:

> **A statistic and a confidence bound on that statistic are different
> quantities. That an upper bound is "the observed/expected upper-bound
> fraction" does not make it the observed/expected ratio.**

---

*End of incident. Written 2026-08-09.*
