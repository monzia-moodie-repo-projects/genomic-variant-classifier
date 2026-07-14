# CLAUDE.md — operating doctrine for this repository

**Read this first, every session. It is not background reading; it is the operating manual.**

Owner: Monzia Moodie. Repository: `github.com/monzia-moodie-repo-projects/genomic-variant-classifier`.
Created 2026-07-13, after a single session in which the assistant's own failures — every one of
them a failure to *read* — cost a destroyed roadmap, a broken virtual environment, a false claim
pushed into a commit message, and very nearly a silent-corruption bug in a clinical variant
classifier.

---

## 0. THE PRIME DIRECTIVE

> **This project's entire method is: make it fail loudly, then READ WHAT IT SAYS.**
>
> Every real finding here came from the second half of that sentence. Every assistant failure
> came from skipping it.

The findings of 2026-07-13 — a non-converged model, a silently-erased base model, a library that
returns wrong answers on mis-ordered columns, a headline model absent from every Continuous
Integration run for two months, an untested graph-neural-network branch, a conformal check that
had never executed — were **all printing in plain output, in every run, for weeks.** Nothing was
discovered by being clever. It was discovered by reading.

---

## 1. OUTPUT DISCIPLINE — NON-NEGOTIABLE, MECHANICAL

The assistant repeatedly drew confident, wrong conclusions from **output it had truncated
itself**. This is not a knowledge gap and it is not fixed by resolve. It is fixed by changing
how commands are constructed.

### 1.1 NEVER pre-truncate output you intend to reason about

| ❌ BANNED | ✅ REQUIRED |
|---|---|
| `... \| Select-Object -Last 15` | `... \| Tee-Object -FilePath $log` **then** `Select-String -Path $log` |
| `... \| Select-Object -First 45` | capture the whole thing; search the *file* |
| `... \| head -20`, `\| tail -n 30` | capture the whole thing; grep the *file* |
| **a result cap on a SEARCH TOOL** (`head_limit`, `max_results`, `limit`, "first N matches") | **no cap.** If the output is large, narrow the *pattern* or the *path* — never the *number of answers*. |

**Why:** you choose the window *before* you know where the signal is. It is a guess wearing the
costume of a filter. Documented casualties, all on 2026-07-13:

* `-First 45` cut an `AttributeError` off the bottom of a traceback → wrong root cause.
* `-Last 15` on a `pytest --co` run put the `ERROR` block **above** the window (pytest prints the
  item list last) → the assistant declared its own working guard "inert" and wrote paragraphs of
  self-criticism about a failure that never happened.
* `158 deletions(-)` in a commit summary, read past → a corrupted `docs/ROADMAP.md` was committed
  and pushed.
* **`head_limit: 40` on a search for `hgmd`** → the results hit the cap, and `tests/unit/test_hgmd.py`
  and `tests/unit/test_splice_ai_promotion.py` **fell off the bottom.** The assistant then stated,
  as measured fact, that no other HGMD references existed, removed the feature, and shipped **six
  red tests**. There is a file *literally named* `test_hgmd.py` and the search never showed it.

> **THE LAST ONE IS THE IMPORTANT ONE. It went straight through this rule.**
>
> The three rows above it ban *shell pipes*. A `head_limit` parameter on a search tool is not a
> pipe, so it did not *look* like the banned thing — but it is **exactly** the banned thing: a
> window chosen before the signal is known. **A truncation is a truncation whatever the syntax.**
> The rule is not about `head`. It is about never deciding how much of the truth you are willing
> to see.
>
> A search that is *capped* is indistinguishable, from the inside, from a search that **found
> everything**. That is what makes it lethal: it does not look like missing data, it looks like an
> answer. See also §1.4 — a search returning nothing may be malformed. **A search returning
> exactly N things, where N is the cap you set, has told you NOTHING except that there are at
> least N.**

A file keeps everything. A window does not. **Capturing to a file costs nothing and truncates
nothing.**

### 1.2 ALWAYS read the exit code

Print `$LASTEXITCODE` (PowerShell) / `echo $?` (bash) after any command whose result you will
reason about. On 2026-07-13 the machine said `4`, said `1`, and said `158 deletions` — plainly,
in three separate places — and each was looked past.

### 1.3 Before asserting "X is broken" or "X is fine", state what would FALSIFY it

Then confirm you have actually *seen* that evidence. "The ratchet didn't fire" is falsified by a
non-zero exit code. It was `4`. It was on screen.

### 1.4 A search that returns nothing is not a negative result

It may be a malformed search. On 2026-07-13 a bad glob returned no matches for `nannyml` /
`evidently`; the assistant asserted "not imported anywhere in `src/`" and **pushed that falsehood
into a commit message.** They are imported in five files. **Validate that the search looked
before believing that it found nothing.**

### 1.5 Read the FULL file before concluding what code does

"Dead code" was asserted about `kan.py:180-183` from the `__init__` **signature** without reading
the `__init__` **body** three lines below — which would have shown the parameters are accepted and
discarded. The lines were load-bearing. Deleting them broke the local path instantly.

**The signature says a parameter exists. Only the body says whether it is ever stored.**

---

## 2. THE FOUR ROOT PATTERNS

Every defect this project has found is one of four shapes. They are stated in full in
`docs/ROADMAP.md` §7; this is the operating summary.

### (a) A number written down once and never re-derived becomes a lie on a schedule

`KNOWN_ZERO_DEFAULT` commented 27 while the literal held 25. `variant_ensemble.py` saying
"65 features" against a 97-feature contract. `RUN_17_PLAN` asserting 91. A G1 pytest floor of
**1485** against a suite passing **1815** — 330 tests could have vanished and the gate would have
said PASS.

That floor then rotted **five more times in two days** (1485 → 1805 → 1842 → 1850 → 1853), *every
single time* beneath an emphatic, all-capitals comment ordering the next person to raise it.

> **A COMMENT DOES NOT ENFORCE ITSELF, AND NO VOLUME OF EMPHASIS WILL MAKE IT.**
> **If a rule can be forgotten, it will be. Make forgetting FAIL.**

**Fix:** derive it at gate time, or put it behind a ratchet. See `tests/EXPECTED_SUITE_SIZE` +
`tests/conftest.py::pytest_collection_modifyitems` (roadmap 6.14), and
`EXPECTED_TABULAR_FEATURE_COUNT` guarding `TABULAR_FEATURES`.

### (b) A library that hard-codes a CWD-relative writable path makes the suite a function of the developer's disk

The AlphaMissense fallback (12 tests red on a populated box, green on a clean one).
`ESM2Connector._DEFAULT_CACHE`. `ProteinStructurePipeline` downloading a structure **into the
checkout**. Invisible locally; visible only on a cold clone.

### (c) A gate that checks a PROXY instead of the thing it protects is not a gate

* G1 §13c checked `RUN_17_PLAN.md` for **completeness** (unfilled `<DECISION>` markers) and never
  for **truth** — so it green-lit a paid run against a document that misstated the very feature
  contract under test.
* `vm_bootstrap_run.sh` §E checked that `KANClassifier` **imports**. The bug was in `fit()`. It
  imports perfectly. **Run 17 would have passed every pre-flight, trained for eleven hours, and
  published a twelve-model algorithm comparison with a headline model silently missing.**
* The correctness harness imported `engineer_features` while the training pipeline ran a second,
  drifted copy — so the gate validated a code path the run never executed.

> **A model that imports is not a model that trains.**
> **A document that is complete is not a document that is correct.**
> **Gates must assert the thing they protect.**

### (d) A green result from a mutated environment is evidence about the ENVIRONMENT, not the code

The developer's `.venv312` held a `sed`-patched `imodelsx` from 2026-05 to 2026-07-13. Locally the
Kolmogorov-Arnold Network trained and every test passed — and that green was used, **in writing, in
a remediation document**, to conclude *"no historical run has ever lost a model."* On Linux, where
the science actually runs, KAN had been raising `NameError`, being silently swallowed, and vanishing
from the ensemble **in every Continuous Integration run for two months.**

> **When a local suite is green, ask what the green is evidence OF.**

### The meta-rule under all four

> **A finding in a log is a comment. A finding in a document is a comment.
> A finding that fails a test is a gate.**

`INCIDENT_2026-06-14` had already recorded the `data/` pollution. Nothing happened for four weeks,
because nothing *failed*. The 41 warnings printed in every run for weeks and were scrolled past;
**every single one turned out to be a real defect or the visible edge of one.**

---

## 3. HARD ENVIRONMENT RULES — violating these has already caused damage

### 3.1 NEVER write to a tracked file through the sandbox shell

The Linux sandbox mount serves **stale/torn reads**. On 2026-07-12 it produced a phantom
`SyntaxError`, a truncated `real_data_prep.py`, and a fabricated content-loss diff. On 2026-07-13
a `python` read-modify-write through that mount **silently reverted `docs/ROADMAP.md` to a
pre-`f377659` copy**, destroying the four-week catch-up delta and §6 (the open register) — 158
deletions, 0 insertions — and it was committed and pushed.

> **Tracked files are edited ONLY with the Windows-side file tools (Read / Edit / Write).**
> **The sandbox shell runs code. It never writes into the repository, and it never runs `git`.**

### 3.2 NEVER `pip install -e .` inside a clone that shares `.venv312`

It repoints the editable install at a temporary directory and breaks the developer's environment.
(Done once, 2026-07-11.)

### 3.3 NEVER `sed` into `site-packages`

Mutating an installed library on *some* machines and not others is what let the Kolmogorov-Arnold
Network be silently dropped from every Continuous Integration run for two months, and left the
developer's virtual environment holding a library **no clean machine had**. Repair in-process, in
code that ships. See `kan.py::_repair_imodelsx_kan_bare_names()`.

### 3.4 `git` runs on Windows, never in the sandbox.

### 3.5 When installing anything into `.venv312`, use `--no-deps` unless you have checked the resolution

`imodelsx` drags `pandas` 3.0 and `transformers` 5.13 **over the pinned stack**, and the
`transformers` 5.x break is silent until deep in data prep — it killed a Run 17 smoke test.
Verify the stack afterwards: `pandas` 2.3.3, `transformers` 4.46.3, `scikit-learn` 1.8.0,
`numpy` 2.4.4, `torch` 2.11.0.

### 3.6 `-s` hangs the suite. `Select-Object` buffers and looks like a hang. Neither is a bug.

---

## 4. PROJECT CONVENTIONS

* `from __future__ import annotations` at the top of every module.
* Module-level `logger = logging.getLogger(__name__)`. **No logging config in library modules.**
* **Never** `nx.read_gpickle` (removed in NetworkX 3.x).
* New **real** features → `TABULAR_FEATURES`, and **bump `EXPECTED_TABULAR_FEATURE_COUNT`** (currently
  **97**). It is a fail-loud guard. Only genuinely not-yet-computed placeholders go in
  `PHASE_2_FEATURES`.
* Installers are **guarded and reversible**.
* **Nothing fails silently.** A bare `except Exception` that logs and continues is a defect, not
  robustness — it is exactly what erased a base model from the ensemble.
* Verify against the **real code** before editing. The roadmap and run logs are authoritative;
  a stale snapshot is not.
* **Spell out every acronym on first use.** KAN = Kolmogorov-Arnold Network. GNN = Graph Neural
  Network. OOF = out-of-fold. LAC = Least Ambiguous set-valued Classifier. APS = Adaptive
  Prediction Sets. CBPE = Confidence-Based Performance Estimation. ESM-2 = Evolutionary Scale
  Modeling 2. LLR = Log-Likelihood Ratio.

---

## 5. THE GATES, AND WHAT EACH ACTUALLY ASSERTS

| gate | asserts |
|---|---|
| `tests/EXPECTED_SUITE_SIZE` + `conftest.py` (`--assert-suite-size`) | the **collected** count, exactly, in both directions. Fewer = tests VANISHED. More = ratchet not bumped. **Never lower the number to make it pass.** |
| `.github/workflows/ci.yml` → *Assert the coverage-critical dependencies are present* | `torch`, `torch_geometric`, `pandera`, `pyspark`, `river`, `mapie`, … actually import. A missing dependency silently skips its tests and the suite reports green. |
| `scripts/vm_bootstrap_run.sh` §E | every base model **FITS**, not merely imports. |
| `VariantEnsemble.ensemble_completeness_` | roster / trained / dropped / complete, written into the run artifacts. "The ensemble was complete" is a **recorded fact**, not an assumption. |
| `EnsembleConfig.allow_base_model_dropout = False` | a base model whose out-of-fold step fails **raises**. It can no longer vanish. |
| `tests/unit/test_feature_name_contract.py` | LightGBM's measured column-order behaviour. **Library-upgrade tripwire.** |
| G1 `Run_Preflight_Local.ps1` | full `pytest tests/` with the ratchet armed. `-SkipPytest` is an escape hatch on a gate that protects **paid compute**. |

**LightGBM is the sole outlier in the roster: it maps columns POSITIONALLY and returns silently
wrong predictions on mis-ordered input (measured delta 0.855, no error, no warning, even under
`-W error`). scikit-learn and XGBoost raise; CatBoost reorders by name. The `X_tab.values`
dispatch in `VariantEnsemble.fit` is therefore LOAD-BEARING. Do not "clean it up" to pass
DataFrames.**

---

## 6. BEFORE YOU CHANGE ANYTHING

1. **Establish live status first.** Read `docs/ROADMAP.md` (§6 open register, §7 the four
   patterns) and the latest `docs/status/REMEDIATION_*.md`. Never work from a remembered state.
2. **Measure the blast radius before touching a shared signature.** This rule has already
   prevented two real disasters: scaling `cnn_1d` (which consumes a **one-hot DNA encoding** —
   a `StandardScaler` would have destroyed it) and passing DataFrames to LightGBM (which would
   have armed a 0.855-probability silent-corruption bug in a pathogenicity classifier).
3. **Instrument, don't infer.** The LightGBM warning was mis-diagnosed **three times** from its
   text before anyone printed what the code was actually doing.
4. **A guard is not real until you have watched it FAIL.** Negative-test in every direction.
