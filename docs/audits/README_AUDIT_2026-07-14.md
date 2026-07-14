# README audit — claim by claim, against the code, 2026-07-14

**Auditor:** assistant, at Monzia Moodie's instruction.
**Method:** every checkable assertion in `README.md` was tested against the repository, the
committed data manifest, the Run-15 artifacts, or a direct measurement. Nothing below is
inferred from another document. Where a claim could not be settled, it is listed as
UNRESOLVED rather than guessed.
**Scope:** `README.md` at commit `4528414` (2026-07-14). 439 lines.
**Not done here:** no rewriting. This is the diagnosis. The decision about what the README
should *say* is the owner's, and several of the findings below are choices, not typos.

---

## 0. The headline

The README describes a system with **80 features, 862 passing tests, empirically calibrated
ACMG thresholds, a live monthly drift monitor, and connectors hardened against silent zeros.**

Every one of those five statements is false.

It has **95** features and **1,926** passing tests; its ACMG tier boundaries are **uncalibrated
hard-coded defaults** served behind a bare `except Exception: pass`; its drift monitor **had
never performed a single check** until 2026-07-13; and its connectors **silently returned 0.0
for 36 of the 78 features in the last completed run**.

The five-tier architecture itself is **sound and correctly described** — an earlier draft of
this audit wrongly attacked it and that finding is retracted in §1.1. What is wrong is the
claim that its boundaries are calibrated. They are not.

None of this is fraud. It is what happens to a document that is written once and never
re-derived — the same defect this project has now found in a schema baseline, a preflight gate,
a methods document, and a suite-size floor. **The README is the largest surviving instance of
root pattern (a).**

---

## 1. FALSE — refuted by direct measurement

### 1.1 "Empirically calibrated probability thresholds" — they are UNCALIBRATED DEFAULTS, silently

> line 102–103: *"Five-tier ACMG/AMP classification (Pathogenic to Benign) with **empirically
> calibrated probability thresholds**"*

**RETRACTION FIRST.** The first draft of this audit (2026-07-14, same day) filed the README's
*"five-tier clinical classification"* claim under FALSE, on the grounds that the model trains
on binary labels. **That conclusion was wrong and is withdrawn.** It was reached by reading a
single threshold band and generalising from it — the exact skim-and-conclude failure this
project's doctrine exists to prevent, committed by the auditor, in the audit.

The system **is** five-tier, by deliberate design. `api/schemas.py:343-353`:

```python
# ACMGish five-tier mapping based on calibrated probability.
# These defaults apply when no calibrated threshold file is found.
_DEFAULT_THRESHOLDS = {
    "Pathogenic":             (0.90, 1.01),
    "Likely pathogenic":      (0.70, 0.90),
    "Uncertain significance": (0.30, 0.70),
    "Likely benign":          (0.10, 0.30),
    "Benign":                 (-0.01, 0.10),
}
```

Training on confident labels and recovering the ordinal scale from a calibrated probability is
standard, defensible practice: `PATHOGENIC_TERMS` collapses Pathogenic **and** Likely
Pathogenic into label 1, `BENIGN_TERMS` collapses Benign and Likely Benign into 0, and the
tiers are reconstructed at inference. Excluding variants of uncertain significance from
*training* is the correct way to avoid label noise — it is not a defect, and the README does
not misdescribe the system's output.

**THE REAL DEFECT, which the first draft missed by not reading the whole function:**

`models/classification_thresholds.json` **DOES NOT EXIST** (verified 2026-07-14). So every
tier boundary the API applies is a **hard-coded default**, not a calibrated one. And
`schemas.py:356-382`:

```python
def _load_thresholds() -> dict[str, tuple[float, float]]:
    """
    Attempt to load calibrated thresholds from models/classification_thresholds.json.
    Falls back to _DEFAULT_THRESHOLDS SILENTLY if the file is absent or malformed.
    """
    ...
            except Exception:
                pass   # malformed file — use defaults
```

A bare `except Exception: pass`. If the calibration file existed but were malformed — a
truncated write, a bad key, a string where a float belongs — the loader would swallow it
without a word and serve the defaults. The docstring says *"silently"* out loud.

`CLAUDE.md` section 4: *"Nothing fails silently. A bare `except Exception` that logs and
continues is a defect, not robustness — it is exactly what erased a base model from the
ensemble."*

So the boundaries between **Pathogenic** and **Likely pathogenic**, and between **Uncertain
significance** and **Likely benign**, in a clinical variant classifier, are currently:

* uncalibrated,
* undeclared as uncalibrated, and
* protected by an exception handler that would hide a corrupt calibration file.

The README meanwhile asserts they are *"empirically calibrated"*.

**REQUIRED:**
1. `_load_thresholds()` must FAIL LOUD on a malformed file (raise, or log at ERROR and record
   the fallback in the response metadata) — never `except Exception: pass`.
2. It must LOG, at WARNING, when it is serving defaults because no calibration file was found,
   so that "these thresholds are uncalibrated" is a fact in the run record rather than a
   discovery someone makes later.
3. The README must not say "empirically calibrated" until
   `models/classification_thresholds.json` exists and is loaded.
4. `scripts/calibrate_thresholds.py` should be run against Run 17 and its output committed.

**OPEN QUESTION FOR THE OWNER (raised 2026-07-14, unresolved):** the owner describes the middle
tier as **"wild type"**. The code and the README both name it **"Uncertain significance"**. A
search of `src/` and the docs finds *"wild type"* used **only** as a biological term (`wt_aa`,
the wild-type residue, in the ESM-2 / EVE / AlphaFold connectors) — **never as a classification
tier**. These are different categories: *wild type* is the reference allele (no variant);
*uncertain significance* is a variant of unknown effect. If the intended taxonomy includes a
wild-type tier, the implementation does not have one.

### 1.2 "Connector silent-zero hardening — regression tests assert that connector fallbacks fail loud, not silently return 0.0"

> line 222–223

**This is the exact opposite of the truth, and it was refuted by measurement on 2026-07-13.**

`omim.py:105`:

```python
gene_table = self._get_gene_table()
result = df.copy()
if gene_table.empty:
    result["omim_n_diseases"] = DEFAULT_N_DISEASES   # 0
    return result                                    # no log, no warning, no raise
```

**Run 15 trained, evaluated and published with 36 of its 78 features CONSTANT ZERO** — 46% of
the feature space, across 1,038,974 variants (roadmap 6.21). GTEx, 1000 Genomes, FinnGen,
AlphaFold/protein structure, MaxEntScan, UniProt, ESM-2, EVE, dbSNP, PhyloP, OMIM, ClinGen —
every one silently stubbed to 0.0.

The README claims the precise safeguard whose absence caused the project's largest scientific
defect. (The guard now exists — `feature_census()` + the zero-variance gate, 2026-07-13 — so
this sentence could become true. It was not true when written.)

### 1.3 "A dedicated continual learning pipeline runs on every ClinVar monthly release … adaptive retraining is triggered automatically"

> line 108–114. Also line 20 (*"a committed drift-detection suite"*), line 203 (*"PSI …
> **runs on every data source update**"*), line 132 (*"a scheduled GitHub Actions
> drift-monitoring workflow"*).

**The scheduled drift monitor had never performed a single check** (roadmap 6.20). It fired on
the first of every month, created an empty directory, observed that the directory was empty,
and reported `drift_level=none` with a green tick. Nothing was ever detected; nothing was ever
triggered.

Separately (roadmap 6.19), the drift *library* stack was dead: `nannyml` could not import at
all, `evidently` was called through an API deleted two major versions earlier, and
`verify_drift_libs.py` — the script whose only job was to verify the drift libraries — could
not itself run.

### 1.4 The feature count — stated **nine** times, with **four** different values, all wrong

| line | claim |
|---|---|
| 7 (badge) | `Tabular features 80` |
| 26 | *"now a **80-feature** matrix"* |
| 40 | *"Input features span **80 dimensions**"* |
| 64 (diagram) | ***78**-feature engineering (engineer_features)* |
| 170 | *"## Feature set (**80** features)"* — and the table below it sums to exactly 80 |
| 254 | *"GET /info — Model metadata, **80 features**"* |
| 298 | Run 15 row: *"**79 features**"* |
| 338 | *"features/ — engineer_features (**80-column** pipeline)"* |
| 393 | *"# Train (full ensemble, **80 features**)"* |

**Truth: 95.** `EXPECTED_TABULAR_FEATURE_COUNT = 95`, enforced by
`tests/unit/test_feature_count_contract.py` and `test_schema_baseline_matches_contract.py`.

The README's own feature table (lines 174–194) contains **`HGMD | 2`** — features that never
held a non-zero value and were removed from the contract on 2026-07-13 — and **`Reserved (Deep
Ensemble) | 2 | uncertainty_epistemic, uncertainty_aleatoric`**, which live in
`PHASE_4_FEATURES`, **not** in `TABULAR_FEATURES`. They are counted in the 80.

### 1.5 The test count — stated **three** times, with **three** different values, all wrong

| line | claim | truth (2026-07-14) |
|---|---|---|
| 9 (badge) | `Tests — 862 passing` | **1,926 passed, 7 skipped, 1,933 collected** |
| 154 | *"**501/501** unit tests and integration suite green at HEAD"* | as above |
| 323 | *"Test depth — **501/501** unit tests + integration tests"* | as above |

### 1.6 Python 3.14.3 — the version that broke the dependency file

> line 140–141: *"a typed inter-agent message bus … (**34/34 tests passing on Python 3.14.3**)"*
> line 248: *"exercises the bus (**34/34 passing on Python 3.12.10**)"*

The README asserts the same suite passes on **two different Python versions**, in two places,
and one of them is **3.14.3** — the version under which `requirements.txt` was mis-compiled,
silently omitting `torch`, `torch-geometric`, `networkx`, `numba`, `pandera`, `pyspark` and
`river` because torch has no 3.14 wheels (roadmap 6.18). The project runs **3.11 and 3.12**.

### 1.7 The quickstart cannot be run

| line | command | status |
|---|---|---|
| 376 | `MODEL_PATH=models/phase2_pipeline.joblib uvicorn …` | **`models/phase2_pipeline.joblib` does not exist** |
| 394–398 | `python scripts/run_phase2_eval.py --parquet …` | **`--parquet` IS NOT A FLAG.** The script takes `--clinvar`. Verified: `grep -c '"--parquet"' scripts/run_phase2_eval.py` → **0** |
| 386–391 | `run_drift_monitor.py --reference-splits outputs/phase2_with_gnomad/splits/ … --auto-retrain` | path is stale; and `--auto-retrain` now **refuses** to run from the drift environment (it would unpickle a LightGBM 4.6.0 booster into a 4.5.0 runtime) |

The training quickstart is not stale — **it is wrong**. Anyone who copies it gets an argparse
error.

### 1.8 Files the README points at that do not exist

* `scripts/benchmark.py` (line 353) — **ABSENT** (`benchmark_polars.py` exists)
* `models/phase2_pipeline.joblib` (lines 365, 376) — **ABSENT**
* `models/registry.json` (line 364) — **ABSENT**. Note `drift_monitor.yml`'s registry smoke
  test looks for exactly this file.
* `models/drift_reference.pkl` (line 366) — **ABSENT**, and superseded by the committed
  aggregate profile (`data/reference/drift/run15_reference_profile.json`, 2026-07-13)

### 1.9 HGMD Professional listed as an integrated data source — three times

> line 60 (source diagram), line 120 (*"gene-disease knowledge bases (OMIM, ClinGen, LOVD,
> **HGMD**)"*), line 184 (feature table: *"HGMD | 2 | hgmd_is_disease_mutation,
> hgmd_n_reports"*)

The licence was never obtained (`ROADMAP.md`: *"HGMD | hgmd_* (2) | PAID, blocked"*), the
connector was never wired, and both columns were **constant zero for the life of the project**.
Removed from the contract 2026-07-13 (roadmap 6.21a) — and, independently of the licence, they
must never return as variant-level features: HGMD "DM" is the ClinVar-Pathogenic training label
under a different vendor's name.

`METHODS.md` carried the identical false claim and was corrected on 2026-07-13.

---

## 2. STALE — was true once; superseded

| line | claim | truth |
|---|---|---|
| 59, 121 | GTEx **v10** | `data_manifest.yaml:51` — **v11** gene median TPM |
| 59 | dbNSFP **v4.7** | manifest records **4.x**; version UNRESOLVED |
| 163 | KAN — *"pykan / efficient-kan; MLP fallback"* | actual: **`imodelsx` 1.0.13**, requiring an in-process repair shim (`kan.py::_repair_imodelsx_kan_bare_names`) |
| 152, 358 | CHANGELOG *"1,500+ lines"* | **4,133 lines** |
| 317, 360 | *"**ten** root-cause records"* | **49** files in `docs/incidents/` |
| 336 | *"**18** database connectors"* | **44** `.py` files in `data/` (exact connector count UNRESOLVED — needs a definition) |
| 349 | `run_drift_monitor.py` — *"exit 0/1/2/3"* | now **0/1/2/3/4** (4 = NOT CHECKED, added 2026-07-13) |
| 338 | *"features/ — engineer_features"* | `engineer_features()` lives in **`models/variant_ensemble.py`**. (A `features/` directory does exist — so this is a pointer to the wrong place, not to a missing one.) |
| 168 | ESM-2 *"full-cohort scoring after Run-16 coord-sync"* | Run 16 is long past; status UNRESOLVED |

---

## 3. INTERNALLY CONTRADICTORY — the README disagrees with itself

1. **Feature count:** 80 (×6) vs 78 (line 64) vs 79 (line 298). See §1.4.
2. **Test count:** 862 (badge) vs 501 (×2). See §1.5.
3. **Python version for the message-bus suite:** 3.14.3 (line 141) vs 3.12.10 (line 248).
4. **Run-8 holdout AUROC:** line 26 says **0.9863** (*"Earlier Run-8: holdout 0.9863 / test
   0.9833"*); line 273 says **0.9847** (*"Holdout AUROC (Run 8 baseline) — 0.9847"*). Both
   cannot be right. **UNRESOLVED — neither has been checked against the Run-8 artifacts.**
5. **Evaluation set size:** line 24 says Test **n = 304,711** (Run 15); line 267/277 says
   **349,067 held-out variants**. Different runs, presented without distinction, in a document
   whose reader has no way to tell them apart.

---

## 4. MISLEADING — technically defensible, materially oversells

### 4.1 The headline AUROC

> badge (line 5): `Holdout AUROC 0.9984` · line 24: *"Run 15 … Test AUROC 0.9984 / Val 0.9983 /
> unseen-gene-holdout 0.9988"*

The number is real: Run 15 did report it. **But Run 15 produced it with 36 of its 78 features
constant zero** (roadmap 6.21). The headline metric of this project was achieved on **38 real
features**, and the README presents it as the performance of an 80-feature system.

Worse, the surviving live features include `cadd_phred`, `revel_score`, `sift_score`,
`polyphen2_score` and `n_tools_pathogenic` — in-silico predictors **trained on ClinVar and
HGMD-DM**. That is the classic circularity problem, and with HGMD/LOVD/ClinGen refuted as the
leakage explanation (`docs/audits/LEAKAGE_METRIC_ANALYSIS_2026-07-08.md`), it is now the
**leading candidate by elimination** for why an AUROC of 0.998 is achievable at all.

**No number in this README should be quoted anywhere until Run 17 replaces it.**

### 4.2 "Autonomously maintained" / "13 specialised agents"

> lines 19, 135–143, 225

Thirteen agents are named. `src/genomic_variant_classifier/agent_layer/agents/` holds **41**
`.py` files. Whether all thirteen named agents are *scheduled and running* — as opposed to
merely *defined* — is **UNRESOLVED**. `tests/unit/test_check_agents_active.py` exists and
passes (12 tests), which is evidence, but the README's claim is that they *"continuously
monitor … without manual intervention"*, and the drift monitor next door made exactly that
claim while doing nothing at all. **This claim deserves the same scepticism, and it has not yet
been earned back.**

### 4.3 "Operationally hardened" — dual-layer preflight

> lines 145–153, 305–310

`scripts/preflight_check.py` and `scripts/preflight_vm.sh` **exist** (verified). But the
project's *current* gates are `Run_Preflight_Local.ps1` (G1) and `scripts/preflight_run17.py`,
neither of which the README mentions — and G1's own floors had **rotted five times in two
days** (roadmap 6.3), while `preflight_run17.py` carried a hard-coded `EXPECTED_SCHEMA_COLS = 87`
that was stale and would have **failed a correct baseline** (roadmap 6.22, fixed 2026-07-13).

"Operationally hardened" is the claim; "the gates were rotting faster than anyone was reading
them" is the record.

---

## 5. HONEST — credit where due

These are correct, clearly caveated, and should be preserved:

* **The histopathology branch is repeatedly and unambiguously marked `PLANNED`** (lines 52–55,
  77–79, 125–128). It would have been easy to imply otherwise. It does not.
* The ASCII architecture diagram marks the ResNet-50 branch `[PLANNED - see ROADMAP]`.
* `esm2_llr` is described precisely and correctly (lines 44–50), including the subtle and
  important point that the score is **signed** and that *"even benign variants score negative,
  so the model learns the threshold"* — and it names the silent-zero regression test.
* Line 133 correctly says publishing to a container registry and a full CI pipeline are
  *"roadmap items"* rather than claiming them.

---

## 6. UNRESOLVED — must be checked, not guessed

1. Run-8 holdout AUROC: **0.9847 or 0.9863?**
2. Are the ACMG thresholds in `api/schemas.py` (0.30 / 0.70) the output of
   `calibrate_thresholds.py`, or hand-chosen round numbers?
3. Base-model roster: the README says **twelve** (lines 36, 68); roadmap 6.6a describes a
   **thirteen**-model ensemble. Which is it today?
4. Are all thirteen named agents actually scheduled and running, or merely defined?
5. dbNSFP version: 4.7, or 4.x?
6. Is the *"runtime assertion at the bottom of the engineering function"* (line 196–197) still
   present? `scripts/maintenance/patch_drop_pipeline_assert_and_fix_comment.py` exists and its
   stated purpose is to **remove** an import-time assert.
7. Exact connector count (the "18" claim needs a definition before it can be counted).

---

## 7. Recommendation

**Do not hand-edit the numbers.** That is what produced this document. The README has now
carried a wrong feature count in nine places, a wrong test count in three, and two different
Python versions for the same test suite — precisely because every one of those numbers was
transcribed rather than derived.

The same fix that closed roadmap 6.22 applies here:

1. **Bind the checkable claims to the code with a test** — `tests/unit/test_readme_claims.py`,
   modelled on `test_methods_feature_count.py`: the feature count, the test count, the exit-code
   range, and the absence of HGMD as a source. Then the README cannot go stale silently; it goes
   **red**.
2. **Correct the false claims** (§1) — these are not stale, they are wrong, and two of them
   (§1.1 five-tier, §1.2 silent-zero hardening) misdescribe the science.
3. **Quarantine every performance number** behind an explicit note that Run 15 was produced with
   46% of its feature space non-existent, until Run 17 lands.
4. **Resolve §6** by measurement.

**The single most important line in this document is §1.1.** Everything else is a number that
went stale. That one is a description of what the model *is*, and it is not what the model is.
