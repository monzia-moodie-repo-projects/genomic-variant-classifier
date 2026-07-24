# Leiden Open Variation Database -- gene acquisition plan, 2026-07-24

**Status:** decision record. Nothing here has been downloaded. Every acquisition is a
manual, per-gene, browser action taken by Monzia, spaced out, one gene at a time.

**Repository state:** `main` at `715bcfa1bdd8320718c7fa9135834ce1c66d9a59`, 2026-07-23.

**Acronyms on first use.** LOVD = Leiden Open Variation Database. VUS = variant of
uncertain significance. ACMG = American College of Medical Genetics and Genomics.
GRCh37 / GRCh38 = Genome Reference Consortium human build 37 / 38.

---

## 1. Access policy, restated because it governs everything below

There have been **two internet-protocol-address bans in 2026**, both from an automated
client looping the per-gene endpoint. Correspondence from the database administrator
dated 2026-03-30 records the mechanism and the cost: `modified_since` does not paginate,
so every call returned the full 34-megabyte list, and roughly one thousand calls produced
about 34 gigabytes of egress discarded on arrival. The `shared/variants/{GENE}` and
`shared/genes/{GENE}` paths are the **human interface** and must be rendered server-side;
hitting those programmatically is worse than hitting the application programming
interface. This is one volunteer's unfunded time, already spent twice on cleaning up
after generated code.

**No script in this repository may fetch from LOVD.** The acquisition path is: open one
gene's page in a browser, confirm the curator has enabled bulk download, save the file,
wait, move to the next gene. `rank_lovd_candidate_genes_2026-07-24.py` performs no
network access and states so in its first printed line.

---

## 2. What limits the value, and therefore the batch size

From the same 2026-03-30 correspondence:

- The clinical classification field is **deliberately unstandardised** pending ACMG
  criteria version 4.
- Only the legacy **functional** classification is exposed: `effect_reported` from the
  submitter, `effect_concluded` from the curator. `functionProbablyAffected` is a soft
  signal that would have to be mapped to "likely pathogenic" by hand.
- `effect_concluded` is **sparse for many genes**, and how sparse depends entirely on
  that gene's curator.
- GRCh38 coordinates are **sparse wherever data was submitted against GRCh37**.
- Reference and alternate alleles for insertions and deletions must be **reconstructed
  separately** through VariantValidator.

Thirty genes is therefore not thirty times more gold-standard labels. It is thirty times
more weakly-labelled rows of variable coordinate quality, with a per-gene processing cost
that is real and front-loaded. This plan is staged with a measured stopping rule for
exactly that reason.

The project's own history reinforces the caution.
`docs/incidents/INCIDENT_2026-07-19_lovd-classification-map-silent-zero.md`, still OPEN,
records two LOVD coverage figures that disagreed by a factor of fifteen -- 5,553 inner-join
matches against 369 coverage -- because they measured different quantities. Acquisition
without a yield measurement would repeat that.

---

## 3. What is measured, and what is a prior

This distinction is load-bearing. Do not read the two as equivalent.

**MEASURED, from this repository, 2026-07-24.** Per-gene counts of variants, VUS,
deletions and insertions come from `data/processed/clinvar_grch38_clean.parquet`
(4,399,089 rows, measured CLEAN: zero null or empty alleles, zero duplicate variant
identifiers). AlphaFold verdicts come from the census of 2026-07-24T03:31:29Z. The
currently held gene list comes from reading
`data/external/lovd/lovd_all_variants.parquet` directly: **18,006 rows, 10 genes** --
APC, BRCA1, BRCA2, MLH1, MSH2, MSH6, NF1, PTEN, RB1, TP53.

**A PRIOR, unverified, and possibly wrong.** Which genes have a strong, actively curated
LOVD gene-specific database. LOVD grew out of Leiden University Medical Center's
neuromuscular work, so neuromuscular, connective-tissue, Usher-syndrome and
hereditary-cancer genes have historically been better served than, say, large scaffolding
proteins. **That is recollection, not measurement**, it may be stale, and per-gene
curation status is visible only on the gene's own page. Section 7 is how it gets checked.

---

## 4. The measured basis

**Criterion 1 -- the structural feature family is lost.** The census established that
AlphaFold models nothing above 2,699-2,700 residues. All **215** index accessions above
the ceiling return no model (109), isoform-only models that
`src/genomic_variant_classifier/pipelines/protein_pipeline.py:171` silently substitutes
(102), or sequence drift (4). For those genes `alphafold_plddt`,
`solvent_accessibility`, `secondary_structure_context` and `dist_to_active_site` are
absent or actively wrong. An entire evidence family is missing exactly there.

**Criterion 2 -- deletion burden.**
`docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md`, OPEN and rated
CRITICAL, measures the review-status join failing on 98.834 percent of deletions, leaving
a training cohort that is 0.0521 percent deletions against 4.2123 percent if the nested
field were used. The cohort is deletion-starved.

**Criterion 3 -- uncertain significance.** An independent call on a variant ClinVar
already rates pathogenic with multiple submitters adds little. One on a variant ClinVar
rates uncertain adds a great deal.

**Scale.** The top fifteen Tier 1 genes hold **77,688 VUS across 166,417 variants** --
46.7 percent of the variants in those genes are uncertain, against a cohort-wide
uncertain count of 2,550,864. So fifteen genes cover 3.05 percent of all uncertain
variants in the corpus. TTN and ATM alone carry 25,929 of that 77,688, a third of the
pool.

---

## 5. Batch 1 -- six genes

Ordered by the strength of the combined case, not by VUS alone.

| Gene | VUS | Variants | Deletions | AlphaFold | Why this one |
| --- | ---: | ---: | ---: | --- | --- |
| **ATM** | 9,789 | 19,121 | 2,236 | NO_MODEL_404 | Second-largest VUS pool, no structure at all, **and it is named in `scripts/build_lovd_index.py:502` as though already held when it is not.** Acquiring it closes half the configuration drift in section 6. Ataxia-telangiectasia has a long-standing dedicated LOVD database. Strongest combined case of any gene. |
| **TTN** | 16,140 | 39,316 | 3,286 | NO_MODEL_404 | Largest VUS pool in the corpus by 1.6x, and 41 percent of TTN's variants are uncertain. At 34,350 residues it is the extreme case of the structural gap. **Highest processing cost too** -- expect heavy indel allele reconstruction. Worth doing first precisely because it calibrates the cost of everything else. |
| **DMD** | 3,904 | 10,058 | 850 | ISOFORM_ONLY | The Leiden DMD database is LOVD's founding gene-specific database and is historically the most deletion- and duplication-rich resource on the platform. That is exactly the class the 2026-07-08 incident censors at 98.8 percent. Best expected yield per unit of effort. |
| **RYR1** | 5,676 | 11,050 | 552 | NO_MODEL_404 | Third-largest VUS pool. Malignant hyperthermia and congenital myopathy variants are well curated in the neuromuscular LOVD tradition. |
| **USH2A** | 3,521 | 9,377 | 806 | ISOFORM_ONLY | Usher syndrome has dedicated LOVD-hosted databases. High deletion count, and `protein_pipeline.py` currently attaches an isoform structure to it. |
| **FBN1** | 3,393 | 9,208 | 983 | NO_MODEL_404 | Highest deletion count relative to variant count in the top fifteen. Marfan syndrome curation is mature. |

**Batch 1 totals: 42,423 VUS, 98,130 variants, 8,713 deletions across six genes.**

For comparison, the ten genes already held produced 18,006 LOVD rows in total.

---

## 6. The configuration drift, and what to do about it

`scripts/build_lovd_index.py:502` declares the default gene list as:

    BRCA1, BRCA2, TP53, PTEN, ATM, MLH1, MSH2, MSH6, APC, LDLR

The parquet on disk holds:

    APC, BRCA1, BRCA2, MLH1, MSH2, MSH6, NF1, PTEN, RB1, TP53

**The script names ATM and LDLR, which are not held. The disk holds NF1 and RB1, which
the script does not name.** Anyone reading the script would believe ATM and LDLR data
exists. This must be reconciled in one place; it should not be left as a comment.

**ATM** is resolved by acquiring it -- it ranks second on measurement independently of the
drift.

**LDLR** does not appear in the top thirty of either tier, so its measured case is weak.
Its numbers are in the delivered report and should be looked at before deciding:

```powershell
$j = Get-Content "C:\Users\monzi\Downloads\LOVD_CANDIDATE_RANKING_2026-07-24.json" -Raw | ConvertFrom-Json
$j.tier1 + $j.tier2 | Where-Object { $_.gene -in @("LDLR","APOB","PCSK9","PMS2") } |
  Select-Object gene, n_vus, n_variants, n_del, verdict | Format-Table -AutoSize
```

If LDLR's numbers do not justify the manual cost, remove it from the script default and
add NF1 and RB1, so the default finally describes reality.

---

## 7. What must be checked by hand, per gene, before downloading

None of this is knowable from the repository. All of it changes the value of the gene.

1. **Has the curator enabled public bulk download?** Check the gene homepage for the
   "Download all data" link.
2. **How populated is `effect_concluded`?** A gene whose curator has not concluded
   anything yields submitter-reported functional signal only.
3. **Are coordinates GRCh38 or GRCh37?** GRCh37 records need lift-over before they can
   join the cohort.
4. **What is the record count?** A gene with two hundred records is not worth the same
   effort as one with twenty thousand.

Addresses, to be opened one at a time, by hand, spaced out:

    gene homepage    https://databases.lovd.nl/shared/genes/{GENE}
    bulk download    https://databases.lovd.nl/shared/download/all/gene/{GENE}
    fallback view    https://databases.lovd.nl/shared/variants/{GENE}?format=tab

Save each as `C:\Projects\genomic-variant-classifier\data\external\lovd\raw\{GENE}_variants.tsv`,
matching the convention the existing ten follow.

---

## 8. The stopping rule -- measured, and self-calibrating

Do not commit to Batch 2 before Batch 1 has been processed and measured. The rule below
needs no arbitrary threshold because it calibrates against the data already held.

After Batch 1 is downloaded and merged, measure per gene:

- rows acquired;
- share carrying GRCh38 coordinates;
- share with a populated `effect_concluded`;
- rows that **join to the cohort** on chromosome, position, reference and alternate;
- of those joined rows, how many land on cohort variants currently labelled uncertain.

**Continue to Batch 2 if and only if the median Batch 1 gene contributes more
cohort-joinable rows landing on uncertain variants than the median gene among the ten
already held.** If it contributes fewer, the marginal gene is worth less than the average
gene already in hand, and further acquisition is not justified by this criterion.

Report every rate with its denominator. A gene contributing 12,000 rows of which 40 join
is not a 12,000-row gain.

---

## 9. Batch 2 -- ten genes, contingent on Batch 1

All Tier 1. Ordered by VUS.

| Gene | VUS | Variants | Deletions | AlphaFold |
| --- | ---: | ---: | ---: | --- |
| RYR2 | 5,583 | 9,889 | 343 | NO_MODEL_404 |
| NEB | 4,307 | 12,377 | 878 | NO_MODEL_404 |
| PLEC | 4,285 | 6,668 | 171 | NO_MODEL_404 |
| SYNE1 | 4,111 | 7,048 | 257 | ISOFORM_ONLY |
| PKD1 | 3,651 | 6,889 | 987 | NO_MODEL_404 |
| ALMS1 | 3,589 | 7,342 | 515 | NO_MODEL_404 |
| DSP | 3,399 | 6,084 | 448 | ISOFORM_ONLY |
| FLNC | 3,198 | 6,168 | 348 | ISOFORM_ONLY |
| KMT2D | 3,090 | 7,411 | 586 | NO_MODEL_404 |
| CDH23 | 2,430 | 6,184 | 353 | ISOFORM_ONLY |

Batch 2 totals: 37,643 VUS, 76,060 variants, 4,886 deletions.

Deliberately deferred from Batch 2 despite high VUS: OBSCN (3,142 VUS but only 35
deletions), PRKDC (2,708 / 95), ZNF469 (2,676 / 143), HMCN1 (2,430 / 83), AKAP9
(2,280 / 145). Low deletion counts mean they engage criterion 2 weakly, and my prior on
their LOVD curation is weak. They belong in a later batch if at all.

---

## 10. Batch 3 -- panel completion, a different argument

These are Tier 2: their proteins sit below the AlphaFold ceiling, so their structural
features are almost certainly correct. Measured from the stratified run, 118 of 119
accessions at or below 2,699 residues returned a correct canonical structure -- a drift
rate of 1 in 119, 0.84 percent, 95 percent interval 0.02 to 4.59 percent. **"Not in
census" in the ranking output means "below 2,400 residues and therefore never probed",
not "missing".**

So criterion 1 does not apply. The argument here is **panel coherence**, which the
ranking script cannot see:

- **PMS2** (3,279 VUS) is the fourth Lynch-syndrome mismatch-repair gene. MLH1, MSH2 and
  MSH6 are already held. Acquiring PMS2 completes the set, which matters for any
  gene-set-level inference the project makes -- and gene-set contribution is one of the
  project's stated goals, not just per-variant classification.
- **PALB2** (3,478), **BRIP1** (3,593), **BARD1** (2,508), **RAD50** (2,795) and
  **CDH1** (2,684) complete the hereditary breast and ovarian panel alongside BRCA1,
  BRCA2, ATM, TP53 and PTEN.
- **TSC1** (2,826) and **TSC2** (5,889) are an obligate pair; holding one without the
  other is incoherent for tuberous sclerosis.
- **CFTR** (2,832) and **MYH7** (3,433) are standalone but both have mature curation
  traditions.

**POLE** at 6,301 VUS is the single largest Tier 2 pool and has no panel argument; it
stands on VUS count alone.

---

## 11. Open questions this plan does not settle

| # | Question | Resolution |
| --- | --- | --- |
| 1 | Does a gene's curator permit bulk download? | Manual, section 7 |
| 2 | Is `effect_concluded` populated for a given gene? | Manual, section 7 |
| 3 | Is a gene's LOVD record GRCh37 or GRCh38? | Manual, section 7 |
| 4 | Does a Tier 2 "not in census" gene appear in the UniProt index at all? | The ranking cannot distinguish "below 2,400 residues" from "absent from the index". A gene absent from the index is a separate coverage gap and would need its own check. |
| 5 | What is the actual cohort-join yield per LOVD row? | Batch 1, section 8. The 2026-07-19 incident shows two prior figures differing fifteen-fold, so this must be measured, not assumed. |
| 6 | Is LDLR worth keeping in the script default? | Section 6 command |

---

## 12. What this plan is not

It is not a claim that more LOVD data will improve model performance. No such measurement
has been made, and the 2026-07-19 incident is a standing reminder that LOVD coverage
figures in this project have been misread before. The plan's justification is narrower
and defensible: **for 215 genes the structural evidence family is absent or wrong, those
genes carry a disproportionate share of uncertain-significance variants, the cohort is
deletion-starved by a known open defect, and independent curated evidence is the only
thing on the table that addresses all three at once.** Whether it helps is a question for
an ablation, after acquisition and processing, on a held-out test set -- never on
out-of-fold blends.
