# Measurement 2026-07-24 -- cohort delta, review-status inventory, and the sequence-branch gate

Produced by `scripts/probe_cohort_delta_forensics.py`. Raw output at
`docs/measurements/COHORT_DELTA_FORENSICS_2026-07-24.txt`.

## 1. What `clean_cohort.py --apply` actually removed

`clinvar_grch38.parquet` holds 4,420,180 rows; `clinvar_grch38_clean.parquet`
holds 4,399,089. The removal is 21,091 rows, and the number of raw rows whose
`variant_id` is absent from clean is **also 21,091**, so the count delta and the
identity delta are the same quantity -- there is no duplicate-identifier
confound.

| Reason, each row counted once under the first that applies | Rows |
| --- | ---: |
| null `ref` and/or `alt` | 19,988 |
| `ref`/`alt` is a rejected token -- every observed case is `alt = '.'` | **1,103** |
| unaccounted | **0** |
| Total | 21,091 |

**Correction to the prior record.** The project record attributed the entire
21,091 to null-allele structural and copy-number variants, and an earlier
statement in this session repeated that. Only 19,988 rows have a null allele. The
remaining 1,103 carry a valid reference allele and the placeholder `'.'` as the
alternate, and are rejected by the token list at `real_data_prep.py:476`. The
correct statement is that the clean cohort is the raw cohort minus 19,988
null-allele rows **and** 1,103 placeholder-alternate rows.

Observed identifiers embed the literal text `na` where alleles were absent, for
example `clinvar:19:33387843:na:na`. That is the mechanism of
`INCIDENT_2026-05-31_null-key-leak.md` visible in the data.

## 2. Review-status values against the tier map and the missing tokens

Every value falls in exactly one bucket. A value in `MISSING_TOKENS` resolves to
`TIER_MISSING` **before** `REVIEW_STATUS_TIER` is consulted, so it does not raise.

| Column | MAP KEY | MISSING TOKEN | **WOULD RAISE** |
| --- | ---: | ---: | ---: |
| `ReviewStatus` | 3,974,452 | 424,516 (`''`) | **121** |
| `metadata.review_status` | 4,153,808 | 245,148 (`'-'`) | **133** |

Both columns sum to 4,399,089 exactly, with zero nulls and ten distinct values
each.

**One value would raise: `'no classifications from unflagged records'.'** It is
in neither the ten-key map nor the eight-token missing set. Under the raise that
Step 1b introduces, the first production run aborts on exactly these rows and no
others. A tier must be assigned to it in the same commit as the raise.

**A defect in the first run of this measurement, recorded here because the
over-reported figure was circulated.** The first version compared values against
the map alone and reported 669,918 rows as would-raise, because it counted `''`
and `'-'`. The true figure is 254. The count was literally true -- those values
are not map keys -- and the interpretation printed beside it was wrong for
669,664 of the rows it named.

## 3. The case for changing the review-status source, quantified

`metadata.review_status` populates rows that the variant-call-format join leaves
empty. The gain by status:

| Status | Gain |
| --- | ---: |
| criteria provided, single submitter | +131,311 |
| criteria provided, multiple submitters, no conflicts | +26,718 |
| no assertion criteria provided | +12,228 |
| reviewed by expert panel | +4,160 |
| criteria provided, conflicting classifications | +3,768 |
| no classification provided | +1,052 |
| no classification for the single variant | +115 |
| no classifications from unflagged records | +12 |
| practice guideline | +4 |
| **Total** | **+179,368** |

The missing marker falls from 424,516 to 245,148, a reduction of **179,368**.
**The two reconcile exactly.**

`PHASE1_SPEC` section 2 predicts deletions gaining 2,210 to 180,773, a gain of
178,563. The measured gain across all variant types is 179,368, leaving 805
non-deletion rows -- coherent with a join key that mis-normalises indels
generally and deletions overwhelmingly. This is the first independent
corroboration of section 2's mechanism.

After the repair, 245,148 rows still carry no review status and resolve to tier
5. Separately, `clinical_sig = '-'` is also 245,148 rows. Equality of counts is
not identity of rows; a joint count is required before the two sets are stated to
be the same.

## 4. Specification section 3's row counts do not survive contact with the cohort

| Status | Section 3 says | Measured total | Measured delta | Section 3 matches |
| --- | ---: | ---: | ---: | --- |
| criteria provided, conflicting classifications | 3,768 | 157,229 | +3,768 | **the delta** |
| no classification for the single variant | 115 | 512 | +115 | **the delta** |
| no assertion criteria provided | 157,229 | 94,285 | +12,228 | **neither** |

Two figures are source deltas presented under a column headed "Rows". The third
is transposed: **157,229 is the measured total for `criteria provided,
conflicting classifications`**, attributed in the specification to `no assertion
criteria provided`, whose true total is 94,285.

Section 3's *conclusion* -- that the map change moves zero rows into training --
still holds, and on firmer ground than its own table: `Conflicting
classifications of pathogenicity` is in neither `PATHOGENIC_TERMS` nor
`BENIGN_TERMS`, so those rows are dropped at the label filter, line 516, twenty
lines before the tier filter at line 536. The conclusion is provable
analytically. The evidence offered for it is not sound, and section 3 requires
rewriting.

## 5. The sequence-branch gate, read for the first time

`clinvar_grch38_clean_seq.parquet` carries `ok` (boolean) and `reason` (string).

| Outcome | Rows | Share |
| --- | ---: | ---: |
| `ok = true` | 4,398,366 | 99.984% |
| `ok = false` | 723 | 0.016% |
| `ok = null` | 0 | 0.000% |

| Reason | Rows |
| --- | ---: |
| `non_acgt_allele` | 668 |
| `ref_mismatch` | **53** |
| `fetch_failed` | 2 |

The 53 `ref_mismatch` rows matter beyond bookkeeping: the ClinVar reference
allele disagrees with GRCh38 at that coordinate. That is a data-integrity signal
about the cohort, not a window-construction failure, and warrants its own
investigation.

Any model consuming `fasta_seq_ref` or `fasta_seq_alt` must state how the 723
rows are handled. Training on them silently is a defect; dropping them silently
changes the cohort.
