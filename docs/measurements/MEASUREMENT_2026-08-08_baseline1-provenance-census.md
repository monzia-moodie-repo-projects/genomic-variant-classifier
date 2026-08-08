# MEASUREMENT 2026-08-08 — BASELINE-1: the provenance of `0.9847`, and what can be sealed

**Author: Monzia Moodie**
**Base commit: `0856fd7`. Method: repository probes only; no run was executed.**

BASELINE-1 asks which experiment produced the figure `0.9847`, cited across
this repository under several incompatible descriptions. This census answers
it, and the answer bounds what Commit C's `SealedEvaluation` can honestly
contain.

**Result: `0.9847` is UNATTRIBUTABLE from this repository.** Its earliest
appearance is a commit subject line; no committed artefact establishes it. The
cohort size published beside it, `154,404`, **is** attributable — and the
measured area under the receiver operating characteristic curve for that exact
cohort is `0.9974`, not `0.9847`.

---

## 1. What was claimed

`api/main.py` carried, from `ae1853b` (2026-03-25) until PROD-1 removed it on
2026-08-07:

```python
HOLDOUT_AUROC = 0.9847   # gene-stratified, 154 K variants
```

and the application programming interface description served at `/docs`:

> Holdout AUROC 0.9847 on 154,404 gene-stratified expert-reviewed variants.

The same figure appeared in the image label of every container ever built, in
`scripts/run_benchmark.py` as a comparison baseline, in `connector_1kgp.py` as
the in-distribution side of a temporal-drift comparison, and in the README as
both *"publication snapshot"* and *"Run 8 baseline"*.

## 2. Where it actually came from

`git log --all -S "0.9847" --reverse` gives the earliest introductions:

```
2133bd0  feat(phase2): mark Phase 2 complete — holdout AUROC 0.9847
497133a  feat(phase2): mark Phase 2 complete — holdout AUROC 0.9847
169a19c  feat(phase3): REST API + Docker + InferencePipeline
ae1853b  feat(phase3): REST API + Docker + InferencePipeline   (2026-03-25)
```

**The earliest appearance is a commit SUBJECT LINE.** `ae1853b` then carried
the value into source. The paired hashes are the same two changes appearing
twice in history, presumably from a rewrite; that duplication is noted and not
pursued here.

There is no Phase 2 artefact in the repository. `git ls-files outputs` returns
eighteen files across four directories:

| directory | what it holds |
|---|---|
| `run10b_final/` | partial metrics, per-model CSV, inventory, two logs |
| `run14/` | metrics, per-model CSV (test and validation), feature importance, data-quality audit, two manifests, pinned environment, log |
| `run14_observability/` | one JSON, one Markdown |
| `temporal_validation/` | one metrics JSON |

**Nothing from Phase 2. Nothing from Run 8.**

## 3. The four incompatible descriptions

| where | what it calls `0.9847` |
|---|---|
| `README_AUDIT_2026-07-14.md:273` | Holdout AUROC, **Run 8 baseline** |
| `ROADMAP.md:165` | a **2026-06-08 v2 re-baseline**, 64 features |
| `ROADMAP.md:194`, `api/main.py` | Holdout AUROC, **gene-stratified, 154 K variants** |
| `outputs/temporal_validation*/metrics.json` | `model_val_auroc` — a **validation** figure |

And a fifth quantity that merely shares the digits — see §5.

## 4. The audit already asked this, on 2026-07-14

`docs/audits/README_AUDIT_2026-07-14.md` §3.4:

> **Run-8 holdout AUROC:** line 26 says **0.9863**; line 273 says **0.9847**.
> Both cannot be right. **UNRESOLVED — neither has been checked against the
> Run-8 artifacts.**

and §6, *"UNRESOLVED — must be checked, not guessed"*, item 1:

> Run-8 holdout AUROC: **0.9847 or 0.9863?**

**This has been an open question for three and a half weeks**, during which the
figure continued to be served by the application programming interface, baked
into every container image, and cited as a benchmark baseline. BASELINE-1 is
therefore not a discovery. It is a known unanswered question that was treated
as settled everywhere except in the document that asked it.

The audit's question cannot be answered from this repository, because the
Run-8 artifacts it asks to check against **were never committed**.

## 5. Run 15's `0.9847` is a coincidence of four digits

`docs/sessions/SESSION_2026-06-06.md:44-48` carries the column headers:

```
| | AUROC | AUPRC | F1_macro | F1_weighted | MCC | Brier |
| Unseen-gene holdout | 0.9988 | 0.9945 | 0.9847 | 0.9927 | 0.9695 | 0.0059 |
```

So Run 15's `0.9847` is **F1_macro**, confirmed independently by
`docs/METRICS.md:127`.

**But `ae1853b` is dated 2026-03-25 and Run 15 ran on 2026-06-06.** The
constant predates the run by ten weeks, so Run 15 cannot be its source. Earlier
records — including this author's own PROD-1 commit message — noted the
resemblance in a way that invites reading it as a lineage. It is four digits
landing in the same place, and the dates settle it.

## 6. `model_val_auroc` is an echoed input, not a measurement

`outputs/temporal_validation/metrics.json` (tracked) and
`outputs/temporal_validation_phase4/metrics.json` (**untracked**) both record:

```json
"model_val_auroc": 0.9847
```

identically — while their own measured `auroc` values differ, `0.8153` and
`0.8190` respectively. **A figure identical across two different models is a
parameter the script was handed, not something it computed.** It is a fourth
copy of the unverified number, and cannot serve as evidence for it.

## 7. `154,404` IS attributable, and it convicts the claim

`outputs/run14/full/metrics.json`, committed:

```json
{
  "auroc": 0.9975, "auprc": 0.9914, "f1": 0.9775, "mcc": 0.955,
  "brier": 0.013,
  "val_auroc": 0.9974, "val_auprc": 0.9903, "val_f1": 0.9785,
  "val_mcc": 0.9569, "val_brier": 0.0111,
  "n_train": 1197216, "n_val": 154404, "n_test": 349067, "n_features": 78
}
```

**`n_val` is exactly 154,404.** The cohort the service advertised is **Run 14's
validation split**, and the measured area under the curve for that split is
**`0.9974`**.

So the published claim was not merely stale, and not merely two runs fused. Its
number and its denominator have different origins, and **the denominator's own
committed figure was four lines away from the number that was published against
it.**

## 8. `TEMPORALCITE-1` — a second finding

`connector_1kgp.py:29-30` motivates the whole 1000 Genomes Project connector
with:

> The phase-4 model shows a 0.166-point AUROC drop on temporal holdout variants
> (0.9847 in-distribution → 0.8191 on 2023+ variants).

The arithmetic is internally consistent: 0.9847 − 0.8191 = 0.1656. **But both
sides fail inspection.** The left is unattributable (§2). The right, `0.8191`,
is `temporal_validation_phase4`'s AUROC — and that file is **untracked**; the
tracked temporal-validation output reports `0.8153`.

This author previously ruled the citation untouchable because its arithmetic
checked out. That was correct arithmetic and insufficient scrutiny: **a
consistent difference between two quantities of unknown identity is not a
measured comparison.** Filed as `TEMPORALCITE-1`; the connector's scientific
motivation may still be sound, but the numbers quoted for it are not evidence.

## 9. What Commit C can honestly seal

**Run 14 — sealable.** `metrics.json` carries test and validation metrics, all
three split sizes and the feature count; `reproducibility_manifest.json` and
`pip_freeze_vm.txt` pin the environment; `ensemble.manifest.json` and
`scaler.manifest.json` describe the artefacts; per-model CSVs exist for both
test and validation. This is the worked example `SealedEvaluation` should be
built around.

**Run 10b — NOT sealable, and it says so.**

```json
"status": "partial — Run 10b instance destroyed mid-pipeline at ~06:00 UTC",
"lost": ["deep_ensemble.joblib", "GNN", "cloud-computed test AUROC"]
```

An artefact recording its own incompleteness is exactly what the type must be
able to represent **without pretending otherwise**. Run 10b is the test case
for a sealed record that is honestly partial.

**`0.9847` — unattributable, permanently.** No `SealedEvaluation` can be
composed for it. The correct outcome is to record that, not to reconstruct a
figure from four disagreeing descriptions.

## 10. Consequences for the register

* **BASELINE-1** remains open, with its answer now known: the figure cannot be
  attributed, and the audit's `0.9847 or 0.9863` cannot be resolved from this
  repository. It closes when that fact is recorded in the documents that still
  cite the number as established — the README and the roadmap, which are not
  touched here.
* **TEMPORALCITE-1** is filed by this census.
* Commit C's scope is bounded by §9: seal Run 14, represent Run 10b's
  partiality honestly, and attribute nothing else.

## 11. Method, and what was not done

Every statement above comes from a repository probe run on 2026-08-08 against
commit `0856fd7`: `git log -S`, `git ls-files`, and direct reads of committed
artefacts. **No run was executed and no figure was recomputed.** This census
establishes what the repository can and cannot support; it does not establish
what any model's performance actually is.

One probe defect is recorded rather than hidden: an earlier attempt in this
census read `docs/README_AUDIT_2026-07-14.md` while the file is at
`docs/audits/`, with `-ErrorAction SilentlyContinue` suppressing the miss. A
silent nothing read as "no citations found". The command was corrected and
re-run; the finding in §4 comes from the corrected read.
