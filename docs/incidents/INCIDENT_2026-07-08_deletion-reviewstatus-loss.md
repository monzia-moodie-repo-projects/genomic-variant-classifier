# INCIDENT 2026-07-08 — Systematic loss of deletions from the training cohort

**Severity:** CRITICAL — affects the scientific validity of reported metrics.
**Status:** OPEN. Root cause localised; mechanism pending one file read. No remediation applied.

**Superseding analysis (added 2026-07-24, metadata only):**
`docs/incidents/INCIDENT_2026-07-08_R2.md`. Revision 2 performs the file read this
status line calls pending, and records three corrections to the text below: two cited
commit hashes do not resolve in this repository; the residual caveat in section 6.1
closes by the opposite mechanism to the one stated; and the affected-runs claim
conflates smoke logs with a full training run -- Run 17 has not executed. The central
measurement and the remedy are unchanged. **This document is otherwise immutable and
is preserved as written on 2026-07-08.** Containment is recorded separately in
`docs/CONTAINMENT_2026-07-24.md`.
**Discovered:** 2026-07-08, incidentally, while investigating an unrelated schema regression.
**Author:** Monzia Moodie
**Affected artifacts:** `data/processed/clinvar_grch38_clean.parquet`,
`clinvar_grch38_clean_seq.parquet`, and every training split derived from them.
**Affected runs:** Runs **15, 16, 17** — deletion-censored cohort (regime v1). Runs **9–14** ran
without any review-tier filter (regime v0): deletions present, but tier 4–6 labels included.
Established by run-log evidence, 2026-07-08 (see §6). The two regimes are not comparable.

**Reproduce:** `python scripts/probe_reviewstatus_gaps.py`,
`python scripts/probe_tier_filter_impact.py`. Raw output in
`outputs/probe_reviewstatus_gaps.txt`, `outputs/probe_tier_filter_impact.txt`.

---

## 1. Executive summary

`scripts/augment_reviewstatus.py` attaches a top-level `ReviewStatus` column to the clean
cohort by joining it against the ClinVar VCF. **The join fails on 98.834% of deletions.**
Line 64 of that script converts every join miss into an empty string:

```python
df["ReviewStatus"] = key.map(vmap).fillna("")   # unmatched -> "" -> tier 5
```

`real_data_prep.py:479` maps `""` to review tier 5. `real_data_prep.py:484` then drops every
row with `review_tier > min_review_tier`, and `launch_run17_baseline.sh:275` passes
`--min-review-tier 3` explicitly.

The result: **a join failure is indistinguishable from a legitimate ClinVar quality
judgement**, and 161,453 deletions — variants ClinVar *did* review, with criteria — were
silently removed from training. The surviving cohort is **0.0521% deletions**. The
missingness is strongly correlated with the label: only **34.556% of pathogenic variants
survive**, against 95.236% of likely-benign.

The correct review statuses were present in the source parquet the entire time, nested in
`metadata.review_status`, agreeing with the VCF-derived column on **all 3,974,573 rows where
both are populated, with zero disagreements**.

---

## 2. Established facts

Each verified this session; each independently reproducible.

**2.1 The filter was deliberate and active.**
- `scripts/launch_run17_baseline.sh:275` — `ARGS="$ARGS --min-review-tier 3 --n-folds 5"`
- `scripts/run_phase2_eval.py:285` — `--min-review-tier` default `3`
- `scripts/run_phase2_eval.py:385` — `DataPrepConfig(min_review_tier=args.min_review_tier, ...)`
- `scripts/smoke_all_models.py:221` — default `3`, forwarded at line 171
- `src/.../real_data_prep.py:215` — `min_review_tier: int = 3  # exclude tier 4-5`
- `src/.../real_data_prep.py:484` — `df = df[df["review_tier"] <= self.config.min_review_tier]`

**2.2 A join miss is coerced into a plausible tier.**
- `scripts/augment_reviewstatus.py:64` — `.fillna("")  # unmatched -> "" -> tier 5`
- `src/.../real_data_prep.py:479` — substring lookup, unmatched → `5`

**2.3 The join fails almost exclusively on deletions.** Blank-`ReviewStatus` rate by class:

| class | total | RS blank | pct_blank |
|---|---:|---:|---:|
| SNV | 4,101,824 | 236,731 | 5.771 |
| **deletion** | **189,468** | **187,258** | **98.834** |
| insertion | 91,219 | 441 | 0.483 |
| MNV/other | 16,578 | 86 | 0.519 |

Insertions are unaffected. The asymmetry is the mechanistic signature.

**2.4 Retention at `min_review_tier=3`, by class.**

| class | total | kept (current) | pct | kept (metadata) | pct |
|---|---:|---:|---:|---:|---:|
| SNV | 4,101,824 | 3,617,151 | 88.184 | 3,617,870 | 88.201 |
| **deletion** | **189,468** | **1,938** | **1.023** | **163,391** | **86.237** |
| insertion | 91,219 | 82,443 | 90.379 | 82,463 | 90.401 |
| MNV/other | 16,578 | 15,142 | 91.338 | 15,143 | 91.344 |

Deletion share of the surviving cohort: **0.0521%** (current) vs **4.2123%** (metadata).
Wrongly excluded deletions: **163,391 − 1,938 = 161,453**.

**2.5 Retention is label-correlated.**

| label | total | kept (current) | pct_kept | kept (metadata) | pct_kept |
|---|---:|---:|---:|---:|---:|
| benign | 272,688 | 249,445 | 91.476 | 265,814 | 97.479 |
| likely_benign | 1,081,595 | 1,030,072 | 95.236 | 1,054,219 | 97.469 |
| likely_pathogenic | 110,665 | 78,361 | 70.809 | 100,184 | 90.529 |
| **pathogenic** | **383,277** | **132,446** | **34.556** | **200,375** | **52.279** |
| uncertain | 2,550,864 | 2,226,350 | 87.278 | 2,258,275 | 88.530 |

Wrongly excluded pathogenic variants: **200,375 − 132,446 = 67,929**.

**2.6 The primary metric's baseline is cohort-dependent.**

| cohort | binary rows | pos | neg | pos_rate |
|---|---:|---:|---:|---:|
| tier-3, `ReviewStatus` (Runs 15–17) | 1,490,324 | 210,807 | 1,279,517 | **14.145%** |
| tier-3, `metadata` (proposed) | 1,620,592 | 300,559 | 1,320,033 | 18.546% |
| unfiltered | 1,848,225 | 493,942 | 1,354,283 | 26.725% |

A random classifier's AUPRC equals `pos_rate`. Any AUPRC compared across these cohorts is
comparing different scales, not different models.

**2.7 The correct data was already present.** `metadata.review_status`, nested in the source
parquet's struct, agrees with `ReviewStatus` on **3,974,573 / 3,974,573** rows where both are
populated — **zero disagreements** — and its own gaps are a strict subset of `ReviewStatus`'s.
It rescues 178,563 deletions (94.2% of all deletions).

**2.8 Validation coverage of `metadata` is thin exactly where it matters.**

| class | both populated | agreeing | validation coverage |
|---|---:|---:|---:|
| SNV | 3,865,093 | 3,865,093 | 94.229% |
| **deletion** | **2,210** | **2,210** | **1.166%** |
| insertion | 90,778 | 90,778 | 99.517% |
| MNV/other | 16,492 | 16,492 | 99.481% |

`metadata`'s deletion values agree with the VCF on every row where the VCF join succeeded —
but that is only 2,210 rows. **We would be relying on `metadata` for 178,563 deletions on the
strength of 2,210 validated examples.** This must be independently validated before adoption
(§7, step 3). It is reassuring, not sufficient.

---

## 3. Mechanism — ESTABLISHED 2026-07-08. It is a coordinate bug, not a ReviewStatus bug.

`augment_reviewstatus.py` and its VCF map build the **identical** key, `chrom:pos:ref:alt`.
There is no case, prefix, or normalisation asymmetry. The join fails because the cohort's
`pos` is wrong.

**Evidence** (`scripts/probe_pos_offset_by_representation.py`, superseding
`probe_vcf_deletion_join.py`; raw output `outputs/probe_vcf_deletion_join.txt`):

| sample group | offset −1 | offset 0 |
|---|---:|---:|
| deletion, `ReviewStatus` blank | **30/30** | 0 |
| deletion, `ReviewStatus` set | 0 | 30/30 |
| insertion | 0 | 30/30 |
| MNV | 0 | 30/30 |

**Corroboration independent of the join.** Cohort row `clinvar:17:43076593:ACTT:A` cannot have
`ref="ACTT"` beginning at 43076593: the reference base there is `C` (a distinct VCF record, C>G,
occupies that position). `ACTT` begins at **43076592**. The cohort's `pos` and its own `ref`
string disagree about which base the allele starts on.

**The two groups separate perfectly on allele representation, not on length:**

| blank → offset −1 | matched → offset 0 |
|---|---|
| `ACTT:A`, `CA:C`, `GA:G`, `AC:A` — `alt == ref[0]` | `AA:C`, `CG:T`, `GG:T`, `TCAGTT…:G` — `alt != ref[0]` |
| **VCF-padded deletions** | **delins** (the first base changes) |

`2,210 / 189,468 = 1.166%` — exactly the delins fraction of the length-class "deletion".

### Root cause

> The cohort's `pos` is ClinVar `variant_summary`'s **`Start`**: the first *altered* reference
> base. Its `ref`/`alt` are `ReferenceAlleleVCF`/`AlternateAlleleVCF`, which begin at
> **`PositionVCF`**. In a padded deletion the padding base is unchanged, so
> `Start = PositionVCF + 1`. For SNVs, delins, and insertions no reference base is removed, so
> `Start = PositionVCF`. **Only padded deletions are shifted, by exactly one.**

Corroborated by `source_id` reading 2, 3, 4 on the first cohort rows — the canonical ClinVar
VariationIDs of the first `AP5Z1` entries in `variant_summary.txt`. A VCF is position-sorted and
would not begin at chr7. The cohort was built from `variant_summary`, mixing its `Start` column
with its `…VCF` allele columns.

### The fix (in the cohort builder, not the artifact)

```python
is_padded_del = (alt.str.len() < ref.str.len()) & ref.str.startswith(alt)
pos_vcf       = pos - is_padded_del.astype(int)
variant_id    = "clinvar:" + chrom + ":" + pos_vcf.astype(str) + ":" + ref + ":" + alt
```

### Escalation — this is far larger than `ReviewStatus`

`.fillna("")` did not *create* the defect; it **concealed** it, by turning a coordinate error into
a plausible-looking review tier. The tier filter then deleted the evidence.

**Every join keyed on `chrom:pos(:ref:alt)` misses these rows**: gnomAD, SpliceAI, phyloP, CADD,
dbNSFP, 1000G, dbSNP, COSMIC, AlphaMissense. The Nucleotide-Transformer sequence windows are
centred on `pos` and are therefore off by one for the same rows. `variant_id` embeds `pos` and is
likewise wrong — internally consistent, but not identifying the true variant.

The consequence differs by regime (§6):

* **Runs 15–17 (v1)** discarded these rows at the tier filter and never saw the corruption.
* **Runs 9–14 (v0)** *kept* them. Their positional annotations would have silently defaulted to
  zero/NaN. Since padded deletions are enriched for loss-of-function and therefore for pathogenic
  labels, **an all-defaults annotation signature correlated with the pathogenic class may have
  been available to the models as a shortcut.** This is a hypothesis, not a finding; it is
  testable against `outputs/run14/` splits and must be tested before any Run-14 number is cited.

### The guard that was never written

Nothing in this pipeline ever asserted

```python
genome[chrom][pos-1 : pos-1+len(ref)] == ref
```

A single reference-consistency post-condition on the cohort would have caught 187,258 rows the
first time the cohort was built. Row counts were guarded, duplicates were guarded, null alleles
were guarded, reconciliation was guarded — **the coordinates were not**.

---

## 4. Scientific consequences

**4.1 Deletion blindness.** With 1,938 deletions in 3,716,674 retained rows, the ensemble has
effectively never trained on a deletion. No claim of "whole-genome variant pathogenicity
classification" is supported for deletions or frameshift variants. Any feature encoding
deletion-specific biology (e.g. `consequence == frameshift_variant` arising from deletions) is
near-degenerate in training. **Untested prediction, worth checking:** the `consequence`
distribution of the surviving cohort should show frameshift variants overwhelmingly derived
from insertions rather than deletions.

**4.2 Label-correlated censoring.** Discarding 65.4% of pathogenic variants while retaining
95.2% of likely-benign ones is not random censoring. It reshapes the class balance, and the
positives that survive are those ClinVar reviewed with criteria — a systematically
better-characterised, plausibly easier subset. This can move measured performance in either
direction and cannot be signed without experiment.

**4.3 AUPRC incomparability.** AUPRC is the project's declared primary metric. Its no-skill
floor is `pos_rate`, which is 14.145% for Runs 15–17. Correcting the join moves it to 18.546%.
Correcting *and* disabling the tier filter moves it to 26.725%. Reported AUPRC values are only
interpretable alongside the cohort's `pos_rate`, which has never been recorded in a run
artifact. **`pos_rate` must become a mandatory postflight field.**

**4.4 What is NOT affected.** Row-level integrity is intact: 0 duplicate `variant_id`, 0
null/bad alleles, the reconciliation identity holds exactly (4,399,089 + 21,091 = 4,420,180).
`clean` and `clean_seq` agree on `ReviewStatus` across all 4,399,089 shared IDs in identical
row order. Nothing is corrupted; a legitimate subset was silently selected.

---

## 5. Two adjacent defects found in the same code path

**5.1 Three divergent implementations of review-tier mapping.**

| location | matching | unmatched default |
|---|---|---|
| `clean_cohort.py:139` `_review_tier` | exact `.map()` | `.fillna(6)` → **6** |
| `augment_reviewstatus.py:32` `_tier_of` | substring `k in s` | **5** |
| `real_data_prep.py:479` | substring `k in s` | **5** |

A blank `ReviewStatus` is tier **5** to the filter and tier **6** to the cohort builder.
Substring matching is additionally load-bearing on dict insertion order. Reconcile to one
function with one documented default.

**5.2 Underscore/space mismatch in the label constants (latent).**
`clean_cohort.py`'s `PATHOGENIC_TERMS` and `BENIGN_TERMS` are written with spaces
(`"likely pathogenic"`); the data uses underscores (`likely_pathogenic`). `_normalize_label`
lowercases and strips but never converts underscores, so **`likely_pathogenic` and
`likely_benign` silently map to `-1` (uncertain)**. Currently inert — the source has zero
duplicate `variant_id`, so the conflict machinery never executes — but it will mis-detect
`pathogenic` vs `likely_benign` conflicts the moment a duplicate appears.

Also: the key `"no classification for the individual variant"` in `REVIEW_STATUS_TIER` is
**dead** — the data says `"no classification for the single variant"`. Harmless under
`clean_cohort`'s fallback (6→6), but it resolves to 5 under the substring implementations.

**5.1 and 5.2 are the same defect class as the incident itself:** ClinVar strings arrive in one
textual convention and are compared against constants written in another, with a silent
fallback absorbing every mismatch.

---

## 6. Scope across runs — RESOLVED 2026-07-08

**Note on a prior error in this document.** An earlier revision dated the `ReviewStatus`
augmentation to 2026-06-12, inferred from the filename
`install_docs_close_2026-06-12_run16-smoke-gate.py`. That is a document *about* the work, not the
work. The commit log gives the true date.

### 6.1 Evidence

```
f24bfc6  2026-06-03 19:48:32 -0400  feat(data): attach ClinVar ReviewStatus to clean cohort
                                    (Path A, enables --min-review-tier)
80ac62c  2026-05-26 05:55:05 -0400  Run 14
```
`outputs/run14/run14_master.log` is dated 2026-05-26 10:46:42. The augmentation landed **eight days
after Run 14 completed**, and one day before Run 15's first log (`run15_run.log`, 2026-06-04 11:31).

`real_data_prep.py:485` emits `"Review tier filter (<=%d): %d -> %d."` on every run where the filter
executed. Grepping every run log:

| log | `Feature matrix: … rows` | `Review tier filter` |
|---|:--:|:--:|
| `run9_ready/regen.log` | yes | **no** |
| `run10a` (×5 logs), `run10b_final` | yes | **no** |
| `run11`, `run12`, `run13` | yes | **no** |
| **`run14/run14_master.log`** | **yes** | **no** |
| `run15_run.log`, `run15_full.log`, `run15_baseline_master.log` | yes | yes |
| `run16_master.log`, `smoke_run16.log`, `smoke_run16b.log` | yes | yes |
| `run17_smoke.log`, `run17_smoke_stage1.log`, `smoke15_vm.log`, `smoke_cnn_tier1/smoke.log` | yes | yes |

**Runs 9–14 built a feature matrix and never applied the review-tier filter. Runs 15, 16, 17 did.**

Residual caveat, both branches converging: the absence of the log line could in principle mean the
`logger.info` did not yet exist rather than that the filter did not run. The filter and the log live
inside the same `if "ReviewStatus" in df.columns:` block (`real_data_prep.py:473-490`), and the column
did not exist before `f24bfc6`. Confirm:
```powershell
git show 80ac62c:src/genomic_variant_classifier/data/real_data_prep.py | Select-String 'ReviewStatus|review_tier'
```

### 6.2 Cohort lineage — three regimes, one undifferentiated metrics table

| regime | runs | tier filter | binary rows | pos_rate | deletion share of cohort |
|---|---|---|---:|---:|---:|
| **v0** unfiltered | 9–14 | none (silent no-op) | ≤1,848,225 | 26.725% | 4.307% (present) |
| **v1** censored | 15, 16, 17 | tier ≤3, VCF `ReviewStatus` | 1,490,324 | 14.145% | **0.0521%** |
| **v2** proposed | 18+ | tier ≤3, `metadata.review_status` | 1,620,592 | 18.546% | 4.2123% |

Run 14's reported test AUROC of 0.9975 was measured on **regime v0**. Run 17's Stage-1 AUROC
0.9960/0.9963 and AUPRC 0.9852/0.9828 were measured on **regime v1**. **They are not comparable, and
no run artifact records which regime produced it.** AUPRC's no-skill floor equals `pos_rate`; that
floor moved from 0.267 to 0.141 across the boundary, with no annotation anywhere.

**Neither v0 nor v1 is defensible.** v0 trains on tier 4–6 labels — "conflicting classifications",
"no assertion criteria provided" — noisy labels across a complete variant spectrum. v1 trains on
clean labels and almost no deletions. v2 is the first regime with both.

The v0 row is an **upper bound**: 1,848,225 is computed before `exclude_conflicting`
(`real_data_prep.py:503-505`), which v0 also applied. The exact figure is printed in
`run14_master.log`. Extract it before citing:
```powershell
Get-ChildItem outputs,logs -Recurse -Include *.log |
  Select-String -Pattern 'Feature matrix: ([\d,]+) rows x (\d+) features' |
  ForEach-Object { [PSCustomObject]@{ Log=(Split-Path $_.Path -Leaf)
                                      Rows=$_.Matches[0].Groups[1].Value
                                      Features=$_.Matches[0].Groups[2].Value } } | Format-Table -AutoSize

Get-ChildItem outputs,logs -Recurse -Include *.log |
  Select-String -Pattern 'Review tier filter \(<=(\d+)\): ([\d,]+) -> ([\d,]+)' |
  ForEach-Object { [PSCustomObject]@{ Log=(Split-Path $_.Path -Leaf)
                                      Tier=$_.Matches[0].Groups[1].Value
                                      Before=$_.Matches[0].Groups[2].Value
                                      After=$_.Matches[0].Groups[3].Value } } | Format-Table -AutoSize
```

### 6.3 Consequence

**No cross-run metric comparison in this project is valid without stating the cohort regime.** The
Run 14 → Run 17 progression on which the roadmap rests spans a cohort boundary that changed the
training-set size, the positive rate, the label-noise profile, and the variant-class spectrum
simultaneously. Every future run artifact must record: cohort version, cohort MD5, schema
fingerprint, `min_review_tier`, `pos_rate`, and per-variant-class row counts (§7, step 5).

---

## 7. Remediation — staged, gated, nothing applied yet

Deliberately ordered so that no scientific change is made before the mechanism is understood
and no cohort is regenerated before the generator is safe.

**Step 1 — Read the mechanism.** `scripts/augment_reviewstatus.py`. Write §3 from the code.

**Step 2 — Fix (c): make the generator safe. Unconditional, zero scientific impact.**
`clean_cohort.py` gains a hard **pre-condition** (no resolvable review column — top-level *or*
nested — raises; never a silent all-tier-5 fallback) and a hard **post-condition**
(`ReviewStatus` asserted present in the written schema). Today it guards rows — duplicates,
null alleles, reconciliation — but never its own output schema, which is the only thing that
changed. Delivered as a complete rewritten file with unit tests, not a patch.

**Step 3 — Independently validate `metadata.review_status` on deletions.** Its deletion values
rest on 2,210 agreeing examples (§2.8). Draw a random sample of ≥500 deletions with
`ReviewStatus` blank and `metadata.review_status` populated, resolve each against the ClinVar
VCF or API, and report agreement with a confidence interval. **Adoption is gated on this.**

**Step 4 — Decide the source, on the evidence.** Either repair the VCF join, or retire it in
favour of `metadata.review_status`. The latter removes a lossy re-derivation of data already in
the source and makes `clean_cohort.py` idempotent. Whichever is chosen, the change is
deliberate, quantified, and documented — never silent.

**Step 5 — Declare a cohort-version boundary.** The corrected cohort is `cohort-v2`. Runs
across the boundary are not comparable. Every run artifact must record: cohort version, cohort
MD5, schema fingerprint, `min_review_tier`, `pos_rate`, and per-class row counts.

**Step 6 — Reconcile §5.1 and §5.2.** One tier function, one default, no dead keys,
underscore-aware label constants.

**Step 7 — Re-baseline.** Re-run Run 14's configuration on `cohort-v2` before any new claim is
made about model performance.

---

## 8. Prevention — mapping to `ANTI_DRIFT_DOCTRINE.md`

**A missing value and a bad value must never share a representation.** `.fillna("")` converted
a join miss into a value that was syntactically valid, semantically wrong, and downstream
*indistinguishable from a legitimate ClinVar judgement of low review quality*. This is the
purest instance of the class this project keeps encountering. It belongs in the failure ledger
as **#19 — silent coercion at a join**.

Concretely, per doctrine §3.2 (contracts and assertions at every boundary):

- A join must report its **miss rate**, and fail loud when it exceeds a declared threshold. A
  98.8% miss on a variant class would have halted this on the first run.
- Sentinel-for-missing (`""`, `-`, `NA`) must be **explicit and distinct** from data. Tier
  should be `Optional[int]`, with `None` propagating to a loud decision, not silently to `5`.
- Per doctrine §3.1, `metadata.review_status` was an **uninventoried member of the trusted
  base** — correct data, present, unused, unknown.
- Per doctrine §3.3, an orchestrator canary asserting *"deletions are ≥1% of the training
  cohort"* — an invariant so weak it looks trivial — would have caught this in seconds. The
  actual value is 0.0521%.

The deepest lesson: **every row-level post-condition in `clean_cohort.py` passed.** Zero
duplicates, zero null alleles, exact reconciliation. The script satisfied every invariant it
declared, and still emitted a cohort that had lost 99% of its deletions — because no invariant
was declared about *composition*. Guards on rows are not guards on populations.

---

## 9. Open questions

1. What is the join key in `augment_reviewstatus.py`, and why do insertions match while
   deletions do not? (§3)
2. Did Run 14 and earlier run without tier filtering? (§6)
3. Does `real_data_prep` use `pathogenicity` as the label column, or `clinical_sig`? All
   `pos_rate` figures in §2.6 assume `pathogenicity`.
4. Is `metadata.review_status` trustworthy for the 178,563 deletions it would rescue? (§7.3)
5. What is the `consequence` distribution of the surviving cohort — are frameshift variants
   present only via insertions? (§4.1)
6. Does `clinvar_grch38_clean_seq.parquet` carry the same censoring into the Nucleotide
   Transformer sequence-window path? (Its `ReviewStatus` matches `clean` on all 4,399,089 rows,
   so almost certainly yes.)
