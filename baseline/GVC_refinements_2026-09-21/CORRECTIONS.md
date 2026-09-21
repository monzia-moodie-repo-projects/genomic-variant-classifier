# Corrections to claims I made, 2026-09-20 to 2026-09-21

Each line below is a claim I stated as established that the evidence does not
support. Verified against code or arithmetic before being listed here.

| My claim | What is actually true | How verified |
|---|---|---|
| `added: 0` was "the decisive proof" the defect is closed, from "a tool with no knowledge of the repair" | A tautology. `derive_review_status.py:160` sets ReviewStatus to metadata.review_status; `build_split_registry.py:83-84` then compares those two columns. It could not have failed. It shows internal consistency of one artifact only. | Read both files |
| Commit `0ff10d7` "fix(ingest)" | Ingestion was not fixed. A standalone script was run once. `augment_reviewstatus.py` and the production entry point are unchanged. | Code inspection |
| Train/validation "gene overlap 0" | Disjointness of literal registry strings only. `TTN` and `TTN-AS1;TTN` hash independently; 60,849 validation rows (32.07%) sit in components also in training (reviewer, symbol-level diagnostic). | Registry keys on raw gene_symbol |
| Design effects 3.2x-19.2x mean prior row-level intervals were too narrow by that factor | The ratio compares a whole-gene interval against a class-STRATIFIED row interval at 500 vs 2000 replicates. On data with zero dependence it reports up to 2.55. | Simulation, dependence/ |
| LR added-cell gain "is calibration, not discrimination" (equal AUROC) | Unsupported. Equal AUROC does not imply one model's scores are a monotone transform of the other's. Brier reflects more than calibration. | Reasoning; reviewer concurs |
| Verdicts "stable across 5 seeds" as evidence of robustness | Five bootstrap seeds measure Monte Carlo error of one fixed-prediction analysis, not replication. | Definition |
| "Constraint absent" stratum in the transfer diagnostic | Defined by `loeuf_is_missing` only. It pooled 619 rows with mis_z present (-0.022373) and 2,813 with neither (+0.024584): opposite signs. | Pooled arithmetic reproduces my +0.016115 exactly |
| Log allele frequency was part of the representation fix; raw af was non-linear in log-odds | `allele_freq` is null in all 4,399,089 rows; every AF feature is constant. LightGBM gave af_raw and af_is_absent zero splits, which I saw and did not investigate. | Reviewer schema check; zero-split output in session |
| CONSEQUENCE_SEVERITY verified as "VEP's standard convention" | Verified as fixed, never checked against the data's vocabulary. ClinVar `nonsense` (74,215 rows) is absent from the SO-term dictionary and scores 0. | Dictionary lookup |
| Arm C used "restricted cubic splines" | `SplineTransformer` builds B-splines. | sklearn API |
| "98.5% representational, 1.5% capacity" | Not a causal decomposition. Log lengths and categorical severity changed together, and transformations alter effective regularisation. | Reviewer; design |
| A future ClinVar release provides independent confirmation | A candidate source requiring identity, exposure and time-boundary screening. | Reviewer |
| Deployed `migration.py` | I retyped it rather than copying the reference. Checked: passes all 14 reference tests and yields an identical registry hash over 20,000 genes. Equivalent, but it should have been copied. | Equivalence test |

## Added 2026-09-21, after the second ruling

| My claim | What is actually true | How verified |
|---|---|---|
| The review-status defect was newly discovered this session | `docs/PHASE1_SPEC_2026-07-24_deletion-repair.md` specified it in July: same line, same 3,974,573 agreeing rows, same prevalence shift. Commit `ce5731a` implemented the spec's changes 2 and 3; change 1, the source switch, was never implemented and the spec was never marked partial. I never searched the project's own measurement history. | Read the spec; `git show ce5731a` line 118 still has the VCF join |
| The weighting identity explains the "reversal" | It proves the two means DIFFER. A reversal additionally needs opposite signs. The tool now reports `means_differ` and `sign_reversal` separately. | Reviewer counterexample, now a test |
| Constraint harm concentrates in the four largest groups | Within R5, BRCA1 and ALMS1 improve; only TTN and TSC2 worsen. The variant-vs-group gap is 65.8% from R1's 1,629 small groups, 18.0% from R5. | Named per-group output |
| The closure check was a pure tautology | It establishes round-trip consistency of the write. It cannot establish source authority or production wiring. | Reviewer; accepted |
| The 2,000-vs-500 replicate asymmetry compounds the inflation | Unmeasured. It changes Monte Carlo precision; I asserted a direction without quantifying it. | Reviewer; accepted |

## Upstream origins located this session (not in the July spec)

`scripts/patch_clinvar_alleles.py` builds `variant_id` from variant_summary `chrom`/`pos` plus VCF `ref`/`alt`. For deletions, `Start` is the first deleted base while VCF alleles are anchored one base earlier, so every deletion identity mixes two coordinate conventions. This is why 99.92% of failed deletion joins recover at pos-1. Prediction, NOT yet measured: other coordinate-keyed annotation joins should fail for deletions the same way.

The same script's `parse_consequence` returns `parts[1]` of `split(",")[0]`: it discards every Sequence Ontology accession and every consequence after the first, and its docstring calls ClinVar labels "VEP-style".

`ClinVarConnector.fetch` ends in `_to_canonical`, which drops `GeneID`, `HGNC_ID` and `AlleleID`, and hard-codes `allele_freq = None` pending an ETL join that never populated this cohort.

## Added 2026-09-21, after identity resolution

| My claim | What is actually true | How verified |
|---|---|---|
| Constraint "harms probability quality" (session finding 3, as restated after repair) | Detectable only in validation rows whose gene component also appears in training. In components unseen in training the effect is not detected. Superseded by the derived statement in docs/MEASUREMENT_2026-09-21_identity-resolved-components.md. | Leakage-stratified re-evaluation of the existing predictions |
| Train/validation gene overlap is zero; the reviewer estimated 32.07% | Through resolved source GeneIDs, 42.40% of evaluated validation rows share a gene component with training (43.35% counting enhancer links). | build_components.py, spans from full membership |
| The census parser fix would add 8 relation rows | It added 16: each of the 8 records links two genes. I predicted without counting per-record contributions. | Census 001 vs 002 |
| A generated document is drift-free because its numbers come from data | My first generator hard-coded its CONCLUSIONS, asserting "all present" beside "3 of 5". Every conclusion is now derived under declared rules, and cross-output inconsistency is flagged. | Fixture that contradicts the prose |
| Model columns can be detected by value range | That selected a feature, af_raw, as a model. Selection is now explicit; auto-detection is labelled and constant columns are named. | Stratified run output |
