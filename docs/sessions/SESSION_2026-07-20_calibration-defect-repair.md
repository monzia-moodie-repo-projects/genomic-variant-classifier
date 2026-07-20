# Session record -- 2026-07-20, part three: the calibration surface

**Commits:** `5615cd0`, `fd85f0f`, `44511fa` (this part). Continuous Integration #548, #549,
#550, all green. Suite 2017 -> 2055 -> 2060.

---

## 1. What started this

Commit `5615cd0` hardened `src/genomic_variant_classifier/evaluation/metrics.py` against six
defects raised by an independent audit. Before deprecating the legacy interface inside it, a
precondition had been set in that commit's own record: **a measured call-site census**.

The census (`census_metric_callsites_2026-07-20.py`, SHA 83994ebe) parsed all **813** Python
files under `src/`, `scripts/` and `tests/` -- by abstract syntax tree, not by name matching,
because an earlier name-based audit of the same surface had understated what `metrics.py`
contains. Zero parse failures.

It returned three things, in ascending order of consequence.

**The legacy interface has no external callers.** `compute_classification_metrics` and
`ModelEvaluator` are called at `metrics.py:73`, `metrics.py:721` (its own `__main__`), and
`tests/unit/test_evaluation_metrics.py:226,230`. Nowhere else. Removal is a live option
rather than a risk, and that is a scope decision.

**No module under `src/` imports the canonical kernel.** Only four modules import
`evaluation.metrics` at all, and two are tests: `scripts/evaluate_predictions.py` and
`scripts/probe_run14_univariate_leakage.py`. The module hardened in `5615cd0` has, to this
day, **no production caller**. That reframes the metric programme: a canonical registry is
not a tidying exercise, it is the thing that would give the kernel a purpose.

**Ten independent implementations of expected calibration error.** Plus three of bootstrap
confidence intervals, two of rank-based area under the receiver operating characteristic
curve, and three of coverage. Reading them could not settle whether they agree.

---

## 2. Building an instrument that does not lie

`probe_calibration_divergence_*.py` evaluates each implementation on identical fixtures. It
took five versions. Each version's failure determined the next design, and the sequence is
recorded because the failures are the useful part.

| version | SHA | loaded | what its failure taught |
|---|---|---|---|
| v1 | `5d69ed63` | 1 / 10 | function-level isolation is too tight |
| v2 | `bfb18d94` | 6 / 10 | a sanitised module needs a dependency fixed point |
| v3 | `9541e229` | 6 / 10 | relative imports need real package identity |
| v4 | `481ede66` | 6 / 10 | print the traceback; stop theorising about the error |
| v5 | `59a9265e` | 9 / 10 | `@dataclass` reads `sys.modules`; register the module |

The v4 traceback ended the guessing in one line:

```
File "...\Lib\dataclasses.py", line 749, in _is_type
    ns = sys.modules.get(cls.__module__).__dict__
AttributeError: 'NoneType' object has no attribute '__dict__'
```

All four remaining failures were one cause. An `exec`'d module is not in `sys.modules`, and
`@dataclass` looks its own class's module up there while the class body runs.

**Four defects in the probe itself, none found by reading it.**

1. **Arguments bound by position.** `_binned_calibration(probs, labels)` takes probabilities
   first; called as `f(y, p)` it accepted them silently and returned a plausible number,
   producing a **false 20x disagreement** that was entirely the probe's calling convention.
   Binding is now derived from parameter names via `inspect.signature`.
2. **`eps` classified as a probability** because it *contains* "p". One-character words now
   require an exact match. Tested against 23 realistic parameter names.
3. **A 1e-9 tolerance** declared `validate_external.py` -- which returns `round(ece, 4)` -- to
   be "computing a different quantity". Matching now searches for the decimal place at which
   the value equals the rounded reference, and reports it.
4. **A single 10-bin reference** reported the 10-bin and 15-bin implementations as disagreeing
   on fixtures where both were correct. **A parameter choice reported as a defect buries real
   findings under noise.** Each implementation is now judged against a reference computed at
   its own default bin count.

A fifth defect appeared in the validation of the fix: the check confirming each replacement
block was valid Python re-indented text that was already indented, and reported **six false
failures**. That is the twenty-eighth occurrence this session of a checker firing on
something that merely resembles its target.

---

## 3. The two defects

### Defect 1 -- the open top bin, three implementations

```
scripts/calibrate_thresholds.py:167   mask = (scores >= lo) & (scores < hi)
scripts/validate_external.py:88       mask = (y_prob >= lo) & (y_prob < hi)
scripts/calibration_analysis.py:75    mask = (y_prob >= lo) & (y_prob < hi)
```

With `hi == 1.0` the final bin is half-open, so every prediction of **exactly 1.0** -- a pure
decision-tree or ensemble leaf -- falls into no bin and is silently excluded. The rows the
model is most confident about contribute nothing to its measured calibration.

Measured under-report on a fixture with 20% of rows at 1.0: **86.7%**, independently
reproducing the **86.5%** first measured on 2026-07-08. `_calibration_summary` inherited it
by delegating to `calibration_bins`.

`scripts/calibrate_thresholds.py` **selects operating thresholds**.

### Defect 2 -- counts misaligned with the bins they weight, three more, undocumented

```
scripts/run_benchmark.py:65-74
scripts/validate_clinvar_temporal.py:235-242
src/genomic_variant_classifier/evaluation/benchmark.py:119-132

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, ...)
    counts = np.histogram(y_proba, bins=bins)[0]
    sum((c / n) * abs(fp - mp) for fp, mp, c in zip(frac_pos, mean_pred, counts))
```

`calibration_curve` returns statistics for **only the non-empty bins**. `np.histogram`
returns counts for **all** of them. `zip` truncates to the shorter and pairs the *k*-th
non-empty bin's statistics with the *k*-th **bin's** count.

Correct whenever every bin is occupied -- which is exactly why it survived review, and why it
looked right on ordinary fixtures.

| fixture | as written | correctly aligned | empty bins |
|---|---|---|---|
| perfectly calibrated | 0.024937 | 0.024937 | 0 |
| overconfident | 0.162413 | 0.162413 | 0 |
| 20% at p == 1.0 | 0.118207 | 0.118207 | 0 |
| saturated | 0.125000 | **0.250000** | 13 |
| sparse + saturated | 0.004825 | **0.309788** | 9 |

**Not merely halved. Sixty-four times under-reported on sparse saturated data** -- which is
what a well-separated classifier's output looks like.

### Why this was invisible

Before the repair, all six returned **exactly 0.125000** on the saturated fixture against a
true 0.250000. Three because they dropped the top bin; three because they misweighted it.
**Two different bugs reached the same wrong number by different routes**, and a unanimous
wrong answer reads as agreement, and agreement reads as correctness.

---

## 4. A retraction

`src/genomic_variant_classifier/evaluation/evaluator.py:305` was cited throughout this work
as carrying defect 1. **It does not.** It was repaired on **2026-07-10** and carries its own
dated comment at lines 321-323. It returned the closed-top reference exactly on all four
fixtures.

The claim came from `metrics.py`'s own docstring, which stated in the **present tense** that
`evaluator.py` *"uses `(p >= lo) & (p < hi)`"* -- true when written on 2026-07-08, false from
2026-07-10, never updated. **A document was quoted instead of the code being read**, and the
false claim was then repeated for an entire session.

That docstring is the **seventh recorded instance** of one failure mode: a fact stated twice
where only one copy was maintained. The others are `WindowAttachment.__iter__`'s
self-maintained todo list; the four ratchet numbers that went stale before the ratchet was
armed; the README badge at ratchet entry 1962; the LOVD classification map; the
`variant_ensemble.py` module header at entry 1985; and the counts a `WindowAttachment` stored
alongside the mask that defines them at entry 1999.

It is **corrected with its dates**, not deleted, so the ten days it was wrong are part of the
record.

---

## 5. The repair and its verification

Six files repaired, one docstring corrected, five tests added
(`tests/unit/test_calibration_implementations_agree.py`).

Every expectation in those tests is **derived** from a reference implemented in the test file,
at each implementation's own default bin count. **Zero hardcoded values**, so a change of bin
count is not mistaken for a defect and a change of definition cannot be waved through by
editing a constant. One test is a **control** asserting the pure-leaf fixture actually
separates the two definitions, so the others cannot pass vacuously. One asserts the fail-loud
length guard survives, so a future edit cannot quietly restore silent misweighting.

Verified by re-running the probe that found the defects:

- Saturated fixture: **all nine implementations return 0.250000, spread 0.000000.**
- Pure-leaf fixture: spread collapsed from **0.099541 to 0.000545**, the residue being bin
  count and `round(x, 4)`.
- `matches its own OPEN-top reference` reads **none** on all four fixtures.
- The two fixtures that were already correct are **unchanged** at 0.017068 and 0.022297. A
  repair that moves a correct number is not a repair.

Ratchet 2055 -> 2060, measured on the **staged** tree per ratchet entry 1962, then verified
under `--assert-suite-size` -- the flag Continuous Integration passes and a plain `pytest`
run does not. 2053 passed, 7 skipped. Continuous Integration #550 green on both Python 3.11
and 3.12.

---

## 6. Open items

1. **`src/.../agent_layer/agents/calibration_drift_agent.py:45` is the tenth implementation
   and remains unevaluated.** `CalibrationDriftAgent.__init__` requires `classes`,
   `baseline_ece` and `output_dir`. It is the agent that **monitors calibration drift in
   production**, it computes its own binning, and which of these two defects it carries is
   **unknown**. It is not assumed clean.
2. **`benchmark.py:125` computes `bin_midpoints` and never uses it.** Dead code.
3. **The legacy metrics interface has zero external callers**, measured. Removal is a live
   scope decision.
4. **Three independent bootstrap implementations remain unreconciled**: `evaluator.py:284`,
   `report_generator.py:85`, and the kernel's. The gene-cluster design effect measured in
   `5615cd0` was **2.935x**; whether the other two share the variant-level independence
   assumption has not been measured.
5. **METHODS.md section 3.1 remains stale** -- carried forward from the `fd85f0f` record.
   Line 139 describes four tabular base models against a roster of thirteen; line 152 says the
   sequence convolutional network is excluded from inference, written before its 2026-07-05
   Tier-1 re-architecture; line 164 states a STRING combined-score threshold of 500 while the
   registry caches `string_graph_700.pkl` -- flagged UNVERIFIED, not asserted.

---

## 7. Method note

Every defect in this session -- in the codebase and in the instruments built to find it -- was
caught by **running something against a fixture**, never by review. The five probe defects,
the six false failures in the installer validation, and both calibration defects themselves
were all invisible to careful reading and obvious to execution.

The durable lesson is not "read more carefully". It is that a check which asserts an
**outcome** finds what a check which asserts the **presence of text** cannot.
