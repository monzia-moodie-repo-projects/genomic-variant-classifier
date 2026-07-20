# MEASUREMENT — selection bias in rows without real sequence windows
### 2026-07-19 · tree `d936238` · read-only, no join, no model fitted

---

## 0. PROVENANCE OF THIS MEASUREMENT

| | |
|---|---|
| **Cohort measured** | `data/processed/clinvar_grch38_clean_seq.parquet` |
| | 534.64 MB · 4,399,089 rows · 22 columns · modified 2026-07-18 09:42:56 |
| **Window artifact** | `data/processed/seq_windows/seq_windows.parquet` |
| | 424.79 MB · 4,420,180 rows · 8 columns · written 2026-07-10 21:24:08 |
| **Manifest** | `data/processed/seq_windows/seq_windows.manifest.json` · 797 bytes |
| | builder `delta_window_builder/2026-07-10-stepB` · `build_utc` 2026-07-11T01:24:22.385232+00:00 |
| **Probe script** | `probe_seq_window_artifact_2026-07-19.py` · SHA-256 `1aef7b9bb33ed3d2…` |
| **Measurement script** | `bias_test_seq_windows_2026-07-19.py` · SHA-256 `7f7451ec2098b05f…` |
| **Repository HEAD** | `d936238` |

**Why `clinvar_grch38_clean_seq.parquet` and not a join.** It is the only artifact in
`data/processed` carrying BOTH the cleaned row count (4,399,089) AND the builder's `ok` /
`reason` provenance columns, on the same rows. Joining the cohort to `seq_windows.parquet`
would have required `chrom:pos:ref:alt` to match byte-for-byte across two files —
`seq_window_join._make_key` concatenates those with `.astype(str)` — and a mismatch produces
ZERO matches, which would present as "no selection bias." That is the single most likely way
this measurement could have lied. Reading one file removes the failure mode entirely.

Fourteen cohort-shaped parquets exist in `data/processed` at eight distinct row counts. The
choice above was made on the stated criterion, not by name, and is open to contradiction.

**Validation of the instrument before use.** The measurement script was run against a synthetic
cohort with a deliberately planted bias — 10% pathogenic among usable rows, 40% among unusable,
expected ratio 4.0 — and recovered 4.349 against the sampled truth, correctly identifying the
planted `structural_variant` composition and correctly detecting that rows had been removed by
cleaning. Cramér's V was verified against three cases with known closed-form answers: perfect
independence → 0.000000, perfect association → 1.000000, hand-computed 2 × 2 → 0.200000, exact
to nine decimal places. A measurement tool that has only ever run on unknown data has not been
shown to measure anything.

---

## 1. THE HEADLINE CORRECTION — the quoted figure overstates by thirty-fold

Every prior discussion of this problem, including the module docstring of
`src/genomic_variant_classifier/data/seq_window_join.py`, quotes the manifest:

```
n_rows_built 4,420,180    n_ok 4,398,366    n_poly 21,814    (0.494% of the build)
```

**The trained cohort does not carry 21,814 unusable rows. It carries 723.**

```
rows        4,399,089
ok = True   4,398,366   (99.9836%)
ok = False        723   ( 0.0164%)
ok = null           0
```

0.0164%, not 0.494%. A factor of **30.2**.

### 1.1 The arithmetic closes exactly, four ways

```
4,420,180 − 4,399,089  =  21,091  =  rows in clinvar_grch38_structural.parquet
   21,814 −       723  =  21,091
   19,988 (empty_allele)  +  1,103  =  21,091
    1,771 (manifest non_acgt)  −  668 (cohort non_acgt)  =  1,103
manifest n_ok 4,398,366  =  ok=True here, exactly
```

The cleaned cohort is the built cohort minus the structural set. `clean_cohort.py --apply`
removes **all 19,988** `empty_allele` rows and **1,103 of the 1,771** `non_acgt_allele` rows —
together exactly the 21,091 structural variants.

A 1,103-row discrepancy was flagged as unexplained earlier the same day. It is now explained:
the structural set is not identical to the `empty_allele` set; it is that set plus 1,103
non-ACGT rows.

### 1.2 Why the unusable rows are unusable — cleaned cohort vs. full build

| reason | in cleaned cohort | % of 723 | in full build | removed by cleaning |
|---|---:|---:|---:|---:|
| `empty_allele` | **0** | 0.00% | 19,988 | 19,988 (all) |
| `non_acgt_allele` | 668 | 92.3928% | 1,771 | 1,103 |
| `ref_mismatch` | 53 | 7.3306% | 53 | 0 |
| `fetch_failed` | 2 | 0.2766% | 2 | 0 |
| **total** | **723** | | **21,814** | **21,091** |

`empty_allele` — the structural-variant signature, and 91.63% of the build's failures — is
**entirely absent** from the training cohort. What remains is dominated by alleles containing
non-ACGT characters, plus 53 reference mismatches and 2 fetch failures.

---

## 2. THE BIAS IS REAL, LARGE, AND PRECISELY ESTIMATED

### 2.1 Label prevalence

| pathogenicity | usable | % | unusable | % | ratio |
|---|---:|---:|---:|---:|---:|
| benign | 272,683 | 6.20% | 5 | 0.69% | 0.112 |
| likely_benign | 1,081,569 | 24.59% | 26 | 3.60% | 0.146 |
| likely_pathogenic | 110,631 | 2.52% | 34 | 4.70% | 1.870 |
| **pathogenic** | **382,807** | **8.70%** | **470** | **65.01%** | **7.469** |
| uncertain | 2,550,676 | 57.99% | 188 | 26.00% | 0.448 |
| **total** | **4,398,366** | | **723** | | |

### 2.2 The interval, because 723 is a small group

```
pathogenic | unusable  =  470 / 723            =  65.01%   95% CI [61.53%, 68.48%]
pathogenic | usable    =  382,807 / 4,398,366  =   8.70%   95% CI [ 8.677%,  8.730%]

risk ratio             =  7.469                          95% CI [7.080, 7.880]

pooling pathogenic + likely_pathogenic:
              unusable =  69.71%     usable = 11.22%     risk ratio = 6.214
```

The confidence interval on the risk ratio excludes 1.0 by an enormous margin. 723 rows is small
but entirely sufficient: **this is not a marginal effect and it is not a sampling artifact.**

### 2.3 What kind of variant the unusable rows are

Ordered by count within the unusable group. Every percentage and ratio below was recomputed
from the raw counts before being recorded here; twenty of twenty reconciled with zero
discrepancies, and the unusable counts sum to exactly 723.

| consequence | usable | % | unusable | % | ratio |
|---|---:|---:|---:|---:|---:|
| nonsense | 101,083 | 2.30% | 296 | 40.94% | 17.814 |
| frameshift_variant | 145,003 | 3.30% | 111 | 15.35% | 4.657 |
| intron_variant | 686,855 | 15.62% | 104 | 14.38% | 0.921 |
| missense_variant | 2,488,825 | 56.59% | 62 | 8.58% | 0.152 |
| inframe_insertion | 5,664 | 0.13% | 45 | 6.22% | 48.333 |
| splice_donor_variant | 58,061 | 1.32% | 38 | 5.26% | 3.982 |
| `<null>` | 19,634 | 0.45% | 31 | 4.29% | 9.605 |
| non-coding_transcript_variant | 132,821 | 3.02% | 12 | 1.66% | 0.550 |
| splice_acceptor_variant | 49,496 | 1.13% | 6 | 0.83% | 0.737 |
| inframe_indel | 3,294 | 0.07% | 5 | 0.69% | 9.234 |
| genic_downstream_transcript_variant | 331 | 0.01% | 4 | 0.55% | 73.517 |
| initiator_codon_variant | 6,287 | 0.14% | 3 | 0.41% | 2.903 |
| 5_prime_UTR_variant | 39,564 | 0.90% | 3 | 0.41% | 0.461 |
| synonymous_variant | 577,148 | 13.12% | 1 | 0.14% | 0.011 |
| 3_prime_UTR_variant | 68,109 | 1.55% | 1 | 0.14% | 0.089 |
| inframe_deletion | 13,613 | 0.31% | 1 | 0.14% | 0.447 |
| stop_lost | 1,696 | 0.04% | 0 | 0.00% | 0.000 |
| no_sequence_alteration | 94 | 0.00% | 0 | 0.00% | 0.000 |

**Truncating variants** — nonsense, frameshift, splice donor, splice acceptor — are **62.38%**
of the unusable group against **8.04%** of the usable group, a ratio of **7.758**. That number
is very close to the pathogenic risk ratio of 7.469, and the coincidence is not a coincidence:
truncating variants are both disproportionately pathogenic and disproportionately likely to
involve alleles the window builder cannot represent.

**The mechanism is legible.** These are not randomly missing rows. They are a specific
biological class, and that class is enriched for the outcome being predicted.

---

## 3. CRAMÉR'S V — WHAT IT MEASURED, AND WHY IT WAS THE WRONG STATISTIC

```
Cramér's V (pathogenicity × ok) = 0.02584
Cramér's V (consequence   × ok) = 0.04297
```

By convention those are negligible. Alongside a 7.5× enrichment, that looks like a
contradiction. It is not, and the reason is worth recording because it will recur.

### 3.1 What the statistic is

Cramér's V is an effect size for association between two categorical variables. Build the
contingency table; compute chi-squared as the accumulated squared deviation of observed counts
from the counts expected under independence, each scaled by its expectation; then normalise:

```
V = sqrt( chi2 / (n * min(r-1, c-1)) )
```

For this table: chi-squared = 2,936.48, n = 4,399,089, min(r−1, c−1) = 1, giving V = 0.02584 —
recomputed independently and matching the script's output to five decimal places.

Dividing by *n* is the entire purpose: chi-squared scales linearly with sample size, so
doubling every count doubles chi-squared while changing nothing about the strength of the
relationship. Dividing by min(r−1, c−1) normalises for table shape. V ranges from 0
(independence in this sample) to 1 (one variable determines the other).

### 3.2 Why it read 0.026

Holding the conditional distributions of this measurement EXACTLY fixed and varying only the
relative sizes of the two groups:

```
      723 usable / 723 unusable   ->  V = 0.61601
   10,000 / 10,000                ->  V = 0.61601
1,000,000 / 1,000,000             ->  V = 0.61601
4,398,366 usable /    723 unusable ->  V = 0.02584   (the actual data)
```

Same proportions. V moves from 0.616 — "large" — to 0.026 — "negligible."

**Cramér's V is invariant to overall scale but NOT to marginal imbalance.** It measures
association across the whole table. Since 99.98% of rows are `ok=True`, knowing a row's `ok`
value tells you almost nothing *on average*, and the 723 rows that differ sharply are diluted
into invisibility.

### 3.3 The statistic was mine and it was the wrong choice

The question was never "how associated are `ok` and `pathogenicity` overall." It was **"are the
excluded rows different from the included ones"** — a conditional question about a small
subgroup. Cramér's V answers the first and is structurally incapable of answering the second
when the subgroup is 0.016% of the data.

The right statistic is the risk ratio with a confidence interval, given in §2.2. It is recorded
here rather than quietly substituted, because a measurement chosen for the wrong reason and
then corrected is more informative than one that was right by luck.

**The general form of the error:** a summary statistic can be correctly computed, correctly
implemented, verified against known cases, and still answer a different question than the one
asked. Verifying the implementation says nothing about whether the quantity is the right one.

---

## 4. DEFECTS FOUND WHILE MEASURING

### 4.1 Embedded newline characters in `consequence` — 3 rows

The grouped output contains these as values DISTINCT from their clean counterparts:

```
"missense_variant\n"   2 rows
"intron_variant\n"     1 row
```

Three rows out of 4,399,089 carry a trailing newline inside a categorical string. Any
`group_by` treats `"missense_variant\n"` and `"missense_variant"` as separate categories, so
this silently fragments a category. Small, but it is corruption in a column that feeds a
categorical feature and it should be repaired at its source rather than tolerated.

### 4.2 Null `consequence` — 19,665 rows

19,634 usable and 31 unusable rows carry a null `consequence`, 0.45% of the cohort. Not
investigated here. Whether a null consequence is a data gap or a legitimate category is a
separate question.

### 4.3 Four all-null columns in the cohort schema

`transcript_id`, `allele_freq`, `protein_change` and `fasta_seq` are typed `null` — entirely
empty across all 4,399,089 rows.

`protein_change` being all-null is the direct confirmation of why the ESM-2 and EVE connectors
return zero: the HGVSp parser has nothing to parse. `fasta_seq` is the legacy sequence column
already documented as 100% null.

### 4.4 A gap in this measurement itself

The consequence table shows the top 20 values by unusable count. A twenty-first value exists,
covering **785 usable rows** and 0 unusable rows. It was not displayed and is not recorded here.
It cannot affect any unusable-group figure, but the table above is not exhaustive on the usable
side and should not be quoted as though it were.

---

## 5. WHAT THIS MEANS FOR THE SEQUENCE-PROVENANCE GATE

**The threshold is doing bias control, not merely power control.** That question is settled.
The excluded rows are 7.5× enriched for pathogenic and 7.8× enriched for truncating
consequences; they are a biological class, not a random sample.

**But the cohort-level stakes are currently small.** Dropping every unusable row shifts the
cohort's pathogenic prevalence from 8.712645% to 8.703391% — a change of **0.009254 percentage
points**. At 99.98% coverage, a real and large conditional bias is a rounding error in
aggregate.

Both statements are true simultaneously, and holding them together is the whole point:

> The bias would matter enormously if coverage were 60%. At 99.98% it does not. Nothing
> guarantees coverage stays at 99.98%.

**Therefore the gate should be a FRACTION floor set high, not an absolute floor.** The danger
is not this cohort. It is a future run against a stale, partial or mis-keyed window artifact
where coverage silently drops — and an absolute floor of 100 usable rows would pass happily
while 40% of the cohort trained on fabricated sequence. A fraction floor catches exactly that.

**A second consequence, for how results may be reported.** If a future run ever trains `cnn_1d`
at materially reduced coverage, its ablation delta is partly a measurement of variant class and
must be reported conditional on consequence rather than as a single number. That is a
constraint on scientific claims, not a tuning parameter.

---

## 6. WHAT WAS NOT MEASURED

* **Statistical power.** How many usable windows a convolutional network over 101-base-pair
  windows needs before its output means anything. Requires a learning curve — subsample to 100,
  300, 1k, 3k, 10k, 30k, train, find where the confidence interval stops excluding 0.5 — and a
  graphics-processing unit. The absolute floor cannot be justified without it, which is why the
  gate ships with its threshold marked UNVALIDATED.
* **Per-split coverage.** This measures the whole cohort. Train, test and tune splits are
  gene-disjoint partitions and their individual usable fractions are not measured here. The
  `_att_tune` defect — calibration on placeholder sequence — concerns a split, not the cohort.
* **Tabular-feature distributions between the two groups.** Only `pathogenicity` and
  `consequence` were compared. A fuller comparison across the 95-feature contract would
  characterise the excluded population more completely.
* **Whether the 723 rows are correctly labelled.** A `ref_mismatch` may indicate a cohort or
  reference problem rather than an unrepresentable variant. 53 rows; not examined.
* **The other thirteen cohort artifacts.** This measures one file. Whether
  `clinvar_grch38_clean_seq.parquet` is the artifact the pipeline actually trains on is a
  separate open question — `scripts/train.py:98` defaults `--clinvar` to
  `data/processed/clinvar_grch38.parquet`, the 4,420,180-row March artifact that still contains
  the structural rows.
