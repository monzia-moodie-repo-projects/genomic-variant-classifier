# Leiden Open Variation Database -- gene acquisition plan, revision 2, 2026-07-24

**Supersedes revision 1** (`docs/superseded/LOVD_ACQUISITION_PLAN_2026-07-24.md`, SHA-256
`6662699F700C7D2CE08282A3F47AE0AB312D048CE6BE21D5013892EC06F29BD9`), which proposed batches selected by a single axis and named the wrong
seven genes. Revision 1 should be replaced, not kept alongside.

**Status:** decision record. Nothing has been downloaded. Every acquisition is a manual,
per-gene browser action, spaced out, one gene at a time.

**Repository state:** `main` at `715bcfa1bdd8320718c7fa9135834ce1c66d9a59`, 2026-07-23.

**Acronyms on first use.** LOVD = Leiden Open Variation Database. VUS = variant of
uncertain significance. ACMG = American College of Medical Genetics and Genomics.
GRCh37 / GRCh38 = Genome Reference Consortium human build 37 / 38.

---

## 1. Access policy, unchanged and governing

Two internet-protocol-address bans in 2026, both from an automated client looping the
per-gene endpoint. Correspondence from the database administrator dated 2026-03-30
records the mechanism: `modified_since` does not paginate, so each call returned the full
34-megabyte list, and roughly one thousand calls produced about 34 gigabytes of egress
discarded on arrival. The `shared/variants/{GENE}` and `shared/genes/{GENE}` paths are
the human interface and must be rendered server-side; hitting them programmatically is
worse than hitting the application programming interface. One volunteer's unfunded time,
already spent twice cleaning up after generated code.

**No script in this repository may fetch from LOVD.** Every tool built for this decision
prints that it performs no network access, and does not.

---

## 2. What limits the value

From the same correspondence, restated because it bounds every number below:

- Clinical classification is deliberately unstandardised pending ACMG criteria version 4.
- Only the legacy **functional** classification is exposed: `effect_reported` from the
  submitter, `effect_concluded` from the curator.
- `effect_concluded` is sparse for many genes, and how sparse depends on that curator.
- GRCh38 coordinates are sparse wherever submission was against GRCh37.
- Insertion and deletion alleles need reconstruction through VariantValidator.

More genes is more weakly-labelled rows of variable coordinate quality, not
proportionally more gold-standard labels.

---

## 3. Four selection methods failed. Recorded so nobody repeats them.

This section exists because the failures were instructive and because a future reader
would otherwise re-derive them.

**Method 1 -- rank by uncertain-significance count.** Buried every gene whose case rests
on deletion burden. LDLR carries 1,474 uncertain variants, which places it nowhere, and
621 deletions in 4,447 variants -- a 13.96 percent deletion fraction, higher than TTN's
8.36 or DMD's 8.45.

**Method 2 -- two axes, ranked separately.** In Tier 1 the top-25 lists shared 12 genes.
**In Tier 2 they were disjoint -- zero overlap.** CHEK2, with 2,333 uncertain variants and
a 15.70 percent deletion fraction, appeared in neither while being strong on both.

**Method 3 -- Pareto frontier.** Degenerate where it mattered: the Tier 1 absolute
frontier had size **1**, TTN alone, because TTN leads on both axes and dominates all 208
other candidates. The density frontier admitted MUC17 (904 uncertain, **0 deletions**) and
CCDC168 (566 uncertain, **0 deletions**) because their uncertain fractions were maximal.
Pareto assumes both axes are substitutable; a gene with zero deletions is not a trade-off
partner for a deletion-starved cohort, it is a non-starter, and the frontier has no way to
express a floor.

**Method 4 -- floors at the median already-held gene.** 4,510 uncertain and 1,241
deletions. Result: **2 of 209 Tier 1 candidates qualified, and 0 of 21,167 Tier 2**. Only
6 of 21,376 candidates cleared the uncertain floor (0.0281 percent) and 2 cleared the
deletion floor (0.0094 percent). The held median sits near the 99.97th percentile, because
the ten held genes are BRCA1, BRCA2, APC, NF1, MLH1, MSH2, MSH6, PTEN, RB1 and TP53 --
among the most heavily submitted genes in all of ClinVar. They are the extreme tail, not a
neutral sample, and calibrating on them made expansion impossible by construction.

**What all four share.** Each asked *which genes are good enough*. There is no stable
answer: gene sizes in ClinVar span four orders of magnitude continuously, with no natural
threshold anywhere.

---

## 4. The question that has an answer: cumulative coverage

Genes **partition** the cohort -- every variant belongs to exactly one gene -- so coverage
is a plain sum with no double counting and no set-cover machinery.

**Baseline, measured 2026-07-24T05:23:40Z against
`data/processed/clinvar_grch38_clean.parquet` (4,399,089 rows, measured CLEAN):**

| | Held 10 genes |
| --- | ---: |
| variants | 106,561 = **2.422 %** of the cohort |
| uncertain variants | 49,099 |
| deletions | 15,516 |
| average per held gene | 10,656 variants, 4,910 uncertain, 1,552 deletions |

**Genes needed to reach each multiple of current coverage:**

| Multiple | Variants | Uncertain | Deletions |
| --- | ---: | ---: | ---: |
| 1.25x | 1 gene | 1 gene | 2 genes |
| 1.5x | 2 | 2 | 5 |
| 2.0x | 7 | 6 | **15** |
| 3.0x | 20 | 19 | **49** |

**Deletions are the scarcest resource and the flattest curve.** Doubling them takes 15
genes against 6 for uncertain variants and 7 for total variants. That is the single most
decision-relevant fact in this document, and it follows directly from
`docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md`, still OPEN and rated
CRITICAL, which measures the surviving training cohort at 0.0521 percent deletions against
4.2123 percent if the nested field were used.

**Where the deletion curve flattens**, as a percentage of the average held gene:

    rank  1  TTN     211.8%      rank 10  CHEK2   45.8%
    rank  2  ATM     144.1%      rank 12  LDLR    40.0%
    rank  3  TSC2     66.1%      rank 20  RYR1    35.6%
    rank  9  USH2A    51.9%      rank 40  CDH1    25.1%

It crosses fifty percent of the held average **between rank 9 and rank 10**. That is a
landmark visible in the data, not a rule. Where to stop is a cost judgement about manual
browser downloads and it is Monzia's.

---

## 5. Batch 1 -- eleven genes

Selected as **deletion-curve ranks 1 through 10**, because deletions are the scarcest
resource, **plus LDLR** at rank 12 because it closes the configuration drift in section 7.

| Gene | Uncertain | Variants | Deletions | Deletion rank | Uncertain rank | Variant rank | AlphaFold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TTN | 16,140 | 39,316 | 3,286 | 1 | 1 | 1 | NO_MODEL_404 |
| ATM | 9,789 | 19,121 | 2,236 | 2 | 2 | 2 | NO_MODEL_404 |
| **TSC2** | 5,889 | 12,260 | 1,026 | 3 | 4 | 4 | below ceiling |
| PKD1 | 3,651 | 6,889 | 987 | 4 | 13 | 16 | NO_MODEL_404 |
| FBN1 | 3,393 | 9,208 | 983 | 5 | 20 | 10 | NO_MODEL_404 |
| NEB | 4,307 | 12,377 | 878 | 6 | 7 | 3 | NO_MODEL_404 |
| PALB2 | 3,478 | 6,593 | 853 | 7 | 17 | 18 | below ceiling |
| DMD | 3,904 | 10,058 | 850 | 8 | 11 | 7 | ISOFORM_ONLY |
| USH2A | 3,521 | 9,377 | 806 | 9 | 16 | 9 | ISOFORM_ONLY |
| CHEK2 | 2,333 | 4,530 | 711 | 10 | n/a | n/a | below ceiling |
| LDLR | 1,474 | 4,447 | 621 | 12 | n/a | n/a | below ceiling |
| **TOTAL** | **57,879** | **134,176** | **13,237** | | | | |

**Effect, computed:**

| | Held 10 | After Batch 1 | Multiple |
| --- | ---: | ---: | ---: |
| uncertain variants | 49,099 | 106,978 | **2.18x** |
| variants | 106,561 | 240,737 | **2.26x** |
| deletions | 15,516 | 28,753 | **1.85x** |
| cohort variant coverage | 2.422 % | **5.472 %** | |

**Eleven genes more than double the footprint of ten.**

### Two corrections to revision 1

**TSC2 was in Batch 3 and belongs in Batch 1.** It ranks 3rd on deletions, 4th on
uncertain and 4th on variants -- top four on every axis. Only TTN and ATM beat it overall.
Placing it third from last was a clear error produced by ranking on one axis.

**RYR1 was in Batch 1 and drops out.** It is 5th on uncertain and 5th on variants but
**20th on deletions** at a 5.00 percent deletion fraction. Against a deletion-starved
cohort that is the wrong profile for a first batch. It is the strongest Batch 2 candidate.

### Genes in the top 20 of all three curves, by rank sum

    TTN 3   ATM 6   TSC2 11   NEB 16   POLE 24   DMD 26   RYR1 30
    PKD1 33   USH2A 34   FBN1 35   PALB2 42

**POLE** at rank sum 24 is the strongest gene not in Batch 1: 3rd on uncertain, 6th on
variants, 15th on deletions. It leads Batch 2.

---

## 6. Batch 2 -- contingent on Batch 1's measured yield

Ordered by deletion rank, continuing the curve: **POLE, RYR1, CFTR, PKHD1, KMT2D, BRIP1,
TSC1, DICER1, PMS2, ALMS1.**

Notes on three of them. **TSC1** pairs with TSC2; holding one without the other is
incoherent for tuberous sclerosis. **PMS2** is the fourth Lynch-syndrome mismatch-repair
gene, completing MLH1, MSH2 and MSH6 already held -- and gene-set contribution is one of
this project's stated goals, not merely per-variant classification. **BRIP1** completes
the hereditary breast and ovarian panel with PALB2 and CHEK2 in Batch 1 alongside BRCA1,
BRCA2, ATM, TP53 and PTEN.

Deliberately **not** in Batch 2 despite high uncertain counts: OBSCN (3,142 uncertain,
**35 deletions**, 0.60 percent), PRKDC (2,708 / 95), ZNF469 (2,676 / 143), HMCN1
(2,430 / 83), AKAP9 (2,280 / 145). They engage the deletion criterion at close to zero.

---

## 7. Configuration drift -- closed by Batch 1

`scripts/build_lovd_index.py:502` declares the default gene list as
`BRCA1, BRCA2, TP53, PTEN, ATM, MLH1, MSH2, MSH6, APC, LDLR`. The parquet on disk holds
`APC, BRCA1, BRCA2, MLH1, MSH2, MSH6, NF1, PTEN, RB1, TP53` -- 18,006 rows, read directly.

The script names **ATM** and **LDLR**, neither held. The disk holds **NF1** and **RB1**,
neither named. Anyone reading the script would believe ATM and LDLR data exists.

Batch 1 acquires both ATM and LDLR, which closes the drift. Once acquired, the script
default must be rewritten to name what is actually held, including NF1 and RB1. Until
then the default is a false statement in source control.

---

## 8. An open item, now closed

Revision 1 carried an unreconciled count: the census reported 215 lost-structure genes,
the ranking found 209 Tier 1 candidates, and three of the ten held genes are above the
ceiling (APC 2,843 residues, BRCA2 3,418, NF1 2,839), leaving three unaccounted.

**Resolved 2026-07-24.** The three are:

| Gene | Accession | Length | Verdict |
| --- | --- | ---: | --- |
| MUC3B | Q9H195 | 13,477 | SEQUENCE_VERSION_DRIFT |
| SSPOP | A2VEC9 | 5,150 | ISOFORM_ONLY |
| APOLTP | Q6ZTK2 | 3,320 | SEQUENCE_VERSION_DRIFT |

All three have **zero rows in the ClinVar cohort**. They are a mucin, a SCO-spondin
pseudogene and an apolipoprotein-L transmembrane gene -- none carries clinical variant
submissions. This is **not** a gene-symbol join gap, and the AlphaFold structural gap for
these three is irrelevant to the classifier because they contribute no variants. Two of
them are also two of the four sequence-drift cases, which is consistent with the drift
defect's measured cohort impact of only 55 variants.

---

## 9. What must be checked by hand, per gene

None of this is knowable from the repository, and all of it changes a gene's value.

1. Has the curator enabled public bulk download?
2. How populated is `effect_concluded` for this gene?
3. Are coordinates GRCh38, or GRCh37 needing lift-over?
4. How many records does the gene actually hold?

    gene homepage    https://databases.lovd.nl/shared/genes/{GENE}
    bulk download    https://databases.lovd.nl/shared/download/all/gene/{GENE}
    fallback view    https://databases.lovd.nl/shared/variants/{GENE}?format=tab

Save each as
`C:\Projects\genomic-variant-classifier\data\external\lovd\raw\{GENE}_variants.tsv`,
matching the convention the existing ten follow. One gene at a time, by hand, spaced out.

---

## 10. The stopping rule, in coverage terms

Revision 1's rule compared a new gene against the median held gene. That benchmark is now
known to sit at the 99.97th percentile and to reject 21,374 of 21,376 candidates, so it is
withdrawn.

The replacement is the curve itself. After Batch 1 is downloaded and merged, measure per
gene: rows acquired; share carrying GRCh38 coordinates; share with a populated
`effect_concluded`; rows that **join to the cohort** on chromosome, position, reference
and alternate; and of those, how many land on cohort variants currently labelled uncertain
and how many on deletions.

**Continue to Batch 2 only if the realised join yield per gene, expressed as a share of
rows acquired, holds up across Batch 1.** If ninety percent of acquired rows fail to join
-- which `docs/incidents/INCIDENT_2026-07-19_lovd-classification-map-silent-zero.md`
makes a live possibility, having recorded two LOVD coverage figures differing fifteen-fold
-- then the coverage numbers in section 5 are a ceiling nobody reaches, and the whole
exercise needs re-costing before another gene is downloaded.

Report every rate with its denominator. Twelve thousand rows of which forty join is not a
twelve-thousand-row gain.

---

## 11. What this plan does not claim

It does not claim that more LOVD data improves model performance. No such measurement
exists. Its justification is narrower and defensible: **for 215 genes the AlphaFold
structural evidence family is absent or wrong, those genes carry a disproportionate share
of uncertain variants, the cohort is deletion-starved by a known open defect, and
independent curated evidence is the only thing on the table that addresses all three.**
Whether it helps is an ablation question, on a held-out test set, never on out-of-fold
blends.

And every number here measures **where ClinVar evidence is missing, not what LOVD holds**.
A gene topping every curve could still be a poor acquisition if its curator has concluded
nothing.

---

## 12. Provenance

| Figure | Source |
| --- | --- |
| Held gene list, 18,006 rows, 10 genes | `data/external/lovd/lovd_all_variants.parquet`, read directly |
| Per-gene uncertain, variant, deletion counts | `clinvar_grch38_clean.parquet`, 4,399,089 rows, measured CLEAN |
| AlphaFold verdicts, ceiling at 2,699-2,700 residues, 215 lost genes | census 2026-07-24T03:31:29Z, `docs/audits/AUDIT_2026-07-24_alphafold_structural_coverage.md` |
| Coverage curves and multiples | `lovd_coverage_curve_2026-07-24.py`, 2026-07-24T05:23:40Z |
| Failed method 4's qualification counts | `lovd_shortlist_2026-07-24.py`, 2026-07-24T05:17:19Z |
| Failed method 3's frontiers | `lovd_pareto_frontier_2026-07-24.py`, 2026-07-24T05:09:42Z |
| The three unaccounted genes | census cross-referenced against the ranking report, 2026-07-24 |
| Deletion censoring at 0.0521 percent | `docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md` |
| Access policy and value caveats | administrator correspondence 2026-03-30; sessions 2026-04-01, 2026-05-02 |
