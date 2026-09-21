# MEASUREMENT 2026-09-20 — The LR/LightGBM gap is 98.5% representational

> **Status 2026-09-21 — core finding confirmed on unseen gene components; several statements corrected.**
> Read with `docs/MEASUREMENT_2026-09-21_gate-a-identity-and-leakage.md` (Gate A) and `baseline/GVC_refinements_2026-09-21/CORRECTIONS.md`.
>
> - Confirmed: split by resolved gene-component leakage (Gate A section 8), `lr_representation`
>   improves on `lr_current` and remains detectably behind LightGBM in both the unseen and leaked
>   strata, and in the unseen stratum neither contrast is carried by a few components.
> - "Gene overlap 0" measured literal registry strings; see Gate A section 6.
> - The stated reason for excluding constraint, that it degrades LightGBM on unseen genes, is not
>   supported; see the companion document's status block.
> - `allele_freq` is null in every cohort row, so `af_raw` and the log10 allele-frequency term were
>   constants. The transformed arm's gain came from the length transforms and categorical severity;
>   "`af_raw` ... badly non-linear" is withdrawn.
> - Consequence severity, including its categorical encoding, was computed from a vocabulary mapping
>   that scored ClinVar `nonsense` and other unmapped terms as 0.
> - `lr_splines` used B-splines (`SplineTransformer`), not restricted cubic splines.
> - "Attributable to model capacity" is not a causal decomposition.
> - The design-effect values use the flawed width ratio described in the companion status block.

Ruling step 7. Four arms over the same admissible information (core tier),
differing only in how that information is encoded for the linear model.
Evaluated on the VALIDATION partition; the test partition is recorded in
`exposure_ledger_v1` as `test_feedback`.

Constraint features were EXCLUDED: the companion measurement showed they
degrade LightGBM probability quality on unseen genes, so including them would
confound the representation question with a term that harms the strongest arm.

Tuning: NONE for any arm, all library defaults. "Comparable tuning effort" is
satisfied by construction rather than by judgement. This leaves open whether
tuning would reorder the arms.

## Arms

Train 1,124,720 rows / 12,385 genes / prevalence 0.186605.
Validation 189,729 rows / 1,807 genes / prevalence 0.197519. Gene overlap 0.

| Arm | Features | AUROC | AUPRC | Brier |
|---|---:|---:|---:|---:|
| lr_current (reference) | 9 | 0.7487 | 0.7130 | 0.08346 |
| lr_representation | 19 | 0.9637 | 0.8475 | 0.05496 |
| lr_splines | 29 | 0.9635 | 0.8476 | 0.05513 |
| lightgbm | 9 | 0.9642 | 0.8542 | 0.05453 |

`lr_current` and `lightgbm` reproduce the constraint run core tier exactly
(0.08346 / 0.05453), confirming nothing moved between experiments.

`lr_representation` uses log1p lengths, log10 allele frequency, and consequence
severity as an unordered CATEGORY rather than a numeric rank. `lr_splines` uses
restricted cubic splines on the raw continuous terms.

AUROC and AUPRC carry NO uncertainty estimate and are not established. Only the
paired Brier contrasts below have intervals.

## Result

Gene-cluster bootstrap, n_boot 2000, all verdicts stable across seeds 0-4.

Against the reference, all three improved arms exclude zero:
lr_representation -0.028501, lr_splines -0.028335, lightgbm -0.028935.

Direct contrasts between the strong arms, which the four-arm run did NOT test:

| Contrast | Delta | 95% CI (seed 0) | Design effect | Verdict |
|---|---:|---|---:|---|
| lr_representation - lightgbm | +0.000434 | [+0.000279, +0.000590] | 2.06 | excludes zero |
| lr_splines - lightgbm | +0.000601 | [+0.000407, +0.000792] | 2.37 | excludes zero |
| lr_representation - lr_splines | -0.000166 | [-0.000314, -0.000044] | 3.61 | excludes zero |

Full ordering, every adjacent pair distinguishable:
**lightgbm < lr_representation < lr_splines << lr_current**

## Reading

Original gap 0.02893. Re-encoding closes **98.50%** of it. The remaining
**1.50%** (0.000434 Brier) is statistically detectable in 5/5 seeds and is
attributable to model capacity.

So the answer is neither "purely representational" nor "capacity": the gap is
overwhelmingly a specification failure in the linear model, with a small real
residual that gradient boosting still captures.

The specification failure is concrete. Raw `ref_len`, `alt_len` and `af_raw`
are badly non-linear in log-odds -- the length/pathogenicity relationship is a
step function (about 11% pathogenic at length 1, 60-65% above), which a linear
term on a standardised raw value cannot bend into a threshold. Two independent
remedies, log transform and spline basis, both recover nearly all the
performance, which is what a functional-form problem looks like.

Whether 0.000434 Brier matters is a decision question about interpretability
versus accuracy. This measurement does not answer it and does not try to.

## Design-effect observation

Design effects here are 1.9-3.7, far below the 8-16x seen in the tier and
repair contrasts. Consistent explanation: these arms make very similar
predictions, so their per-row difference is much less gene-correlated than a
difference between a strong and a weak model. Design effect is a property of
the contrast, not of the dataset alone.

## Artifacts

| Path | Contents |
|---|---|
| `outputs/representation_arms_v1/representation_arms_report.json` | four-arm run |
| `outputs/representation_arms_v1/validation_predictions.parquet` | per-arm predictions |
| `outputs/representation_arms_v1/strong_arm_contrasts.json` | direct strong-arm contrasts |
| `baseline/step7/run_representation_arms.py` | experiment |
| `baseline/step7/run_strong_arm_contrasts.py` | strong-arm contrasts |

Gitignored and local-only.

## Exposure

This is the SECOND development use of the validation partition, after the
constraint re-measurement. Both are legitimate development uses, but the
exposure ledger must be extended to record them as `tuning` before validation
is treated as unexposed in any future claim.
