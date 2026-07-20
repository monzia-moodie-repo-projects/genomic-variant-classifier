# Session record -- 2026-07-20

## WindowAttachment derivation, the tier-1 inversion, and the sequence-provenance gate

Two commits landed on `origin/main`:

| commit | subject | ratchet |
|---|---|---|
| `fb23543` | `WindowAttachment` derives its counts from two masks; tier 1 stops inverting them | 1985 -> 1999 |
| `106d107` | sequence-provenance gate -- refuse windows the ensemble cannot vouch for | 1999 -> 2017 |

The suite moved from 1,985 to 2,017 collected tests across the day (`1978 + 7 + 14 + 18 = 2017`,
counting the docstring pin at `84c6c54` from the preceding session span). Every number in this
document was recomputed on 2026-07-20 before being written down.

**Continuous Integration run #545, for `106d107`, has not been observed and is not asserted
green here.** The same is true of #541 and #544 from earlier spans. Those remain open items.

---

## 1. Commit `fb23543` -- the counts stop being independently stored

### 1.1 The defect

`seq_window_join.py`, tier 1 of `attach_delta_windows`, line 179 read:

```python
att = WindowAttachment(out, usable, n, int((~usable).sum()), 0, prov)
#                                        |_ n_unmapped ___|  |_ n_placeholder
```

In the `rows+ok` branch, `usable = ref.notna() & alt.notna() & ok`. A row that **is present**
but carries `ok=False` is a *builder placeholder*, not an unmapped row -- yet every unusable
row was attributed to `n_unmapped`, and `n_placeholder` was hardcoded to zero.

`data/processed/clinvar_grch38_clean_seq.parquet` carries `fasta_seq_ref`, `fasta_seq_alt` and
`ok`, so it resolves through precisely this tier. Its `summary()` line -- the string
`scripts/train.py:514` writes to the run log for every split -- would have read:

```
windows[rows+ok]: 4398366/4399089 usable (99.984%), 723 unmapped, 0 builder-placeholder
```

Exactly inverted. Nothing is unmapped, because every row is present on the frame. All 723 are
builder placeholders, measured on 2026-07-19 as 668 non-ACGT alleles, 53 reference mismatches
and 2 fetch failures.

Tier 2 (lines 207-215) was always correct, because it kept `mapped` and `ok` separate. Tier 3
is trivially correct. Only tier 1 conflated them.

### 1.2 Why it survived

**No test covered tier 1 with an `ok` column.** Every existing tier-1 test supplied windows
without one (`test_seq_window_join.py:185`, `test_train_cnn_activation.py:47`, `:70`, `:101`);
everything else exercised tier 2 or tier 3. The defect lived in the one shape nothing tested,
and it is the shape the production cohort uses.

### 1.3 The fix is structural, not arithmetic

Correcting line 179 would have fixed the symptom and left the next tier free to repeat it. The
class stored a *derived* value (`usable`) plus three *summaries* of masks it did not keep, so
nothing could enforce agreement between them -- nothing even prevented `n_rows` from
disagreeing with `len(windows)`.

`WindowAttachment` now stores four fields:

```python
windows: pd.DataFrame
key_found: np.ndarray     # a window was located for this row
builder_ok: np.ndarray    # the builder's verdict -- or fabricated; provenance says which
provenance: str
```

and derives `usable`, `n_rows`, `n_usable`, `n_unmapped`, `n_placeholder`, `usable_fraction`
and `provenance_is_verified` as properties. A tier must now state **both** masks; conflating
them is unrepresentable rather than merely discouraged. The installer asserts by parsing that
each of those seven is a property and not a field.

The field is named `key_found` rather than `mapped` because `genomic_lm.py:386` and `:432`
bind `att.usable` to a **local** called `mapped`. One word with two meanings in one codebase is
how a mask gets read wrong.

### 1.4 `subset()`

`WindowAttachment.subset(idx)` slices both masks and the frame, resets the index, and carries
`provenance` through unchanged -- selecting rows neither improves nor degrades the builder's
verdict about the rows selected.

**This method could not have been written honestly under the old layout.** Given only a
combined `usable` mask, a slice's unmapped-versus-placeholder breakdown is unrecoverable, so
any subset would have had to report a stale or invented figure.

It has two production callers, both discovered by measurement rather than assumed:

- `prediction_artifacts.py:365`, in `save_permutation_importance`
- `run_phase2_eval.py:1056` and `:1059`, carving the unseen-gene holdout

### 1.5 Verification

Fourteen new tests in `tests/unit/test_window_attachment_derives_its_counts.py`, all passing
on Monzia's machine in 5.21 seconds. Two are negative controls: one recomputes the **old**
formula and proves it disagrees on exactly this input (a regression test that would also pass
against the bug pins nothing), and one shows an unverified `parquet` attachment still reports
usable rows -- the trap `provenance_is_verified` exists to expose.

Before delivery the patch was applied to a copy of `seq_window_join.py` reconstructed verbatim
from a collector report and executed against a replica of the production cohort's shape:

```
summary() before : windows[rows+ok]: 9998/10000 usable (99.980%), 2 unmapped, 0 builder-placeholder
summary() now    : windows[rows+ok]: 9998/10000 usable (99.980%), 0 unmapped, 2 builder-placeholder
```

Tiers 2 and 3 were confirmed unchanged on frames carrying both failure modes.

Three reader files were then run on Monzia's machine -- `test_seq_window_join.py`,
`test_train_cnn_activation.py`, `test_no_content_based_poly_detection.py` -- because a property
that raises reads exactly like a field that is absent. **17 passed.**

---

## 2. Commit `106d107` -- the sequence-provenance gate

### 2.1 What `_require_x_seq` could not tell

The predecessor asked one question -- *"is `X_seq` None?"* -- of one parameter, and could not
distinguish:

1. **A checked input from an unchecked one.** `fit()` declares `X_seq_cal_ext` and the gate was
   called with `X_seq` alone, so a run with real train windows and a placeholder calibration
   partition passed cleanly: `cnn_1d` **fitted** on real sequence and **calibrated** on
   fabricated sequence, silently. That hole was opened by `ff97c34` -- the commit intended to
   close this very class of defect.

2. **A real window from an invented one.** `run9_ablations.py:677` handed it
   `pd.Series([placeholder] * n)` and it passed, because a `Series` is not `None`. Only the
   roster pop at `:702` kept `cnn_1d` away from it.

3. **A verified attachment from an unverified one.** `seq_window_join.py:211` sets
   `ok = np.ones(n, dtype=bool)` when the window source carries no `ok` column, so `n_usable`
   there counts rows **nobody checked**. That number is indistinguishable from a real one; only
   `provenance` tells them apart.

### 2.2 The design

One method, `_require_sequence_windows`, ranging over a **mapping** of every sequence parameter
the call received rather than a single argument. The point of the mapping is that omitting a
parameter becomes visible at the call site instead of invisible inside a helper -- which is
exactly how `X_seq_cal_ext` went unchecked.

Four refusals:

| refusal | reason |
|---|---|
| `None`, with a `SEQUENCE_MODELS` member active | fabricating a placeholder is roadmap 6.28 restated |
| a bare `DataFrame` or `Series` | carries no provenance at all -- weaker than `provenance="rows"` |
| unverified provenance | the `ok` mask was assumed, not read |
| too few usable rows, by fraction or absolute count | see thresholds below |

A `SequenceWindows` Protocol (`@runtime_checkable`) declares the contract **structurally**, so
`variant_ensemble.py` imports nothing from `genomic_variant_classifier.data`. That is asserted
by walking `Import` and `ImportFrom` nodes, both in the installer and in the test file. The
models layer keeps no dependency on the data layer.

A bare `DataFrame` is **not** a legacy special case. Carrying no provenance makes it the
*least*-verified thing that can arrive.

### 2.3 The thresholds, and why they ship marked unvalidated

```python
seq_min_usable_fraction: float = 0.95           # UNVALIDATED
seq_min_usable_rows: int = 100                  # UNVALIDATED
seq_require_verified_provenance: bool = True
```

All three are per-run overridable, following the `EnsembleConfig.zero_variance_min_rows` idiom,
so a `--max-train 3000` smoke run is not refused by the gate meant to protect it.

The statistical-power question -- how many usable 101-base-pair windows a convolutional network
needs before its output means anything -- requires a learning curve and a graphics-processing
unit, and **has not been run**. Marking that plainly in the configuration is better than
implying a basis that does not exist.

What *has* been measured (2026-07-19,
`docs/measurements/MEASUREMENT_2026-07-19_seq-window-selection-bias.md`) is the selection bias:

| quantity | value |
|---|---|
| pathogenic prevalence, rows without usable windows | 65.01% |
| pathogenic prevalence, rows with usable windows | 8.70% |
| risk ratio | 7.469, 95% confidence interval [7.080, 7.880] |
| truncating consequences, unusable vs usable | 62.38% vs 8.04%, ratio 7.758 |
| unusable rows in the trained cohort | 723 of 4,399,089 (0.0164%) |
| cohort pathogenic prevalence shift if all 723 are dropped | 0.009254 percentage points |

Rows without usable windows are a biological class, not a random sample -- so a coverage
threshold here is doing **bias control**, not power control. Yet the cohort-level impact is a
rounding error. Both are true at once, and that is what sets the shape: the **fraction** is the
primary guard and is set high, because the danger is not this cohort but a future run against a
stale, partial or mis-keyed artifact. An absolute floor of 100 rows would pass happily while
40% of a cohort trained on fabricated sequence.

### 2.4 The migration

Forty edits across nine files, plus the old method replaced by parsed span rather than by a
37-line text match -- a single stray space anywhere in 37 lines would otherwise have aborted a
40-edit commit for no good reason.

| file | change |
|---|---|
| `variant_ensemble.py` | Protocol; three config thresholds; the gate; `fit`, `predict_proba`, `evaluate` |
| `train.py` | three `.windows` shed; inclusion threshold reads the same config field the gate does |
| `run_phase2_eval.py` | three `.windows` shed; `len(seq_tr)` -> `_att_tr.n_rows`; two `subset()` calls |
| `run9_ablations.py` | placeholder `Series` deleted; five call sites pass `None`; dead `cnn_1d` branch removed |
| `prediction_artifacts.py` | false `pd.Series` annotation corrected; `.iloc[idx]` -> `subset(idx)` |
| `test_catboost.py` | two fixtures wrapped in real attachments |
| `test_variant_ensemble_save_load.py` | fixture wrapped; row floor lowered for a pickling test |
| `test_level2_leakfree_oof.py` | empty-string `Series` -> `None` |
| `test_x_seq_refusal_contract.py` | eleven tests migrated to the new gate |

`test_train_cnn_activation.py` needed no change: its only `.fit` is on `CNN1DClassifier`
directly, never on `VariantEnsemble`. That was established by measurement, not assumed.

Two details worth preserving:

- **`evaluate` keeps the attachment** in a local for its nested `predict_proba` call, because
  the inner gate would refuse an already-resolved frame as provenance-less.
- **`train.py:523` still pops `cnn_1d`** when the train split has no usable windows. That is
  the caller explicitly and verbosely choosing to run without the model -- not the silent
  fabrication the gate exists to stop. The two coexist, and now read the same threshold, so
  they cannot disagree about a run.

- **`prediction_artifacts.py:343`** carried `X_seq_test: pd.Series` -- the *same* false
  annotation `VariantEnsemble.fit` shed on 2026-07-15 and documented at length in its own
  docstring. It was fixed in one place and left standing in another for five days. `.iloc`
  accepts both shapes, so nothing complained.

---

## 3. Five defects caught before commit -- none by reading code

| # | defect | caught by |
|---|---|---|
| 1 | The gate was written and **never wired into the installer's edit list**. `_require_x_seq` would have survived; the gate would never have existed. | a post-check asserting the *outcome*, not the edit count |
| 2 | An installer check string-matched `"seq_window_join"` and fired on the **refusal message** telling the reader to call it. | dry run |
| 3 | An installer check string-matched `"_require_x_seq"` and fired on **the new gate's own docstring** explaining what its predecessor did. | dry run |
| 4 | An import check matched `torch.utils.data` because the module name ends in `.data`, reporting an architectural violation that did not exist. | dry run |
| 5 | **The gate demanded `X_seq_cal_ext` unconditionally.** `None` is the *normal* case -- it means no external calibration partition, which is what `train.py:561` and `run_phase2_eval.py:590` both do. It would have refused **every non-v2 run**. | `test_catboost`, not review |

Defect 5 is the one that mattered, and it carries a symmetry worth keeping: this gate exists
because the old check inspected **too few** inputs, and its first version failed by demanding
one that was **not in play**. Same root -- deciding what to check without asking what the caller
actually supplied. The rule is now conditional on `X_tab_cal_ext`, which is the thing that says
whether the partition exists, with tests pinning both directions.

---

## 4. The recurring failure mode -- fifteen occurrences in one day

**A checker that string-matches fires on the text written to describe its own rule.**

Instances recorded across 2026-07-19 and 2026-07-20:

1. A docstring-staleness regex matched the digits `19` inside the date `2026-07-19`.
2. A timing ratio went stale inside the section describing staleness -- four times.
3. A feature census needed parsing, not `grep`.
4. The gate installer's import check matched `torch.utils.data`.
5. The same check matched the refusal message's prose.
6. The survivor check matched the new gate's own docstring.
7. The structural-typing **test** asserted `"import WindowAttachment" not in src` and fired on
   the Protocol docstring saying *"This module does not import WindowAttachment"*.
8. The post-check verifying that fix fired on the replacement's docstring, which **quotes the
   old assertion** to explain what was wrong with it -- the fourteenth instance, inside the fix
   for the thirteenth.
9. A delivery check counting bash line-continuations flagged a legitimate **Python**
   continuation in a script's own source -- the fifteenth, and a false alarm rather than a
   defect.

Every one is now parsed. An import is an `ast` node; a definition and a call are `ast` nodes; a
mention is not.

**The durable lesson is not "remember to parse."** It is that outcome-asserting checks catch
what careful reading does not. Nine of the fifteen were found by a machine check that had been
written to assert a result rather than to confirm an action.

---

## 5. Arithmetic reconciliation -- and a reporting error in the installer

`git diff --cached --stat` for `106d107` reported **12 files changed, 668 insertions(+), 86
deletions(-)**. Splitting the per-file totals using each installer's reported net-line figure
gave **623 insertions**, a 45-line discrepancy.

The cause: the gate installer reported `variant_ensemble.py +81 lines`, computed **after** the
method replacement had already been applied to its working text. It excluded the +90 from
swapping 37 lines for 127. The true net is `90 + 81 + 10 = +181`, taking the file from 2,841 to
**3,022 lines**.

With that correction every file reconciles exactly:

| file | changed | net | ins | del |
|---|---:|---:|---:|---:|
| `README.md` | 2 | +0 | 1 | 1 |
| `scripts/run9_ablations.py` | 23 | +3 | 13 | 10 |
| `scripts/run_phase2_eval.py` | 8 | +0 | 4 | 4 |
| `scripts/train.py` | 10 | +2 | 6 | 4 |
| `evaluation/prediction_artifacts.py` | 7 | +3 | 5 | 2 |
| `models/variant_ensemble.py` | 241 | **+181** | 211 | 30 |
| `tests/EXPECTED_SUITE_SIZE` | 107 | +105 | 106 | 1 |
| `tests/unit/test_catboost.py` | 18 | +10 | 14 | 4 |
| `tests/unit/test_level2_leakfree_oof.py` | 11 | +1 | 6 | 5 |
| `tests/unit/test_sequence_provenance_gate.py` | 240 | +240 | 240 | 0 |
| `tests/unit/test_variant_ensemble_save_load.py` | 15 | +9 | 12 | 3 |
| `tests/unit/test_x_seq_refusal_contract.py` | 72 | +28 | 50 | 22 |
| **total** | **754** | | **668** | **86** |

`test_sequence_provenance_gate.py` at 240 lines is exactly 186 from the gate installer plus 54
from the fix installer -- an independent confirmation that both applied in full.

The line-delta report is cosmetic and affects nothing that runs. It was nonetheless **wrong**,
and a number that only looks right is precisely what this project keeps finding. Recorded here
rather than corrected silently.

---

## 6. Full-suite timings

Fifteen readings, 2026-07-19 and 2026-07-20, in seconds, sorted:

```
594.62  609.52  619.65  637.60  637.80  639.01  650.17  652.09  652.23
657.75  709.49  711.90  727.99  798.38  1026.25
```

Minimum 594.62, median 652.09, maximum 1026.25, mean 688.30. **Thirteen of fifteen fall between
594.62 and 727.99**; the two outliers are 798.38 and 1026.25, the latter on commit `ea8d6e8`,
which *removed* a test.

The distribution is stated raw and scoped to the reading count deliberately. An earlier version
of this section expressed the same information as a "percentage above median" ratio, which went
stale four times in a single day (58%, 51%, 51%, 56%) inside the very section describing
staleness. Machine load is the likeliest explanation for the spread and remains **unmeasured**.

---

## 7. Open items

**Carried forward, unchanged:**

- Statistical-power learning curve for both sequence thresholds. Requires a graphics-processing
  unit. Until then both ship marked unvalidated.
- Per-split `provenance` and `usable_fraction` are not recorded in the run artifact, though
  `summary()` already produces the string.
- Continuous Integration runs #541, #544 and #545 have never been observed green.
- `TABULAR_FEATURES` and `EXPECTED_TABULAR_FEATURE_COUNT` have no agreement pin.
- `train.py:98` defaults `--clinvar` to `data/processed/clinvar_grch38.parquet`, the
  4,420,180-row March artifact **with** structural rows. The canonical cohort for the
  2026-07-19 measurement is `clinvar_grch38_clean_seq.parquet`. The pipeline default is
  unresolved and is a trap.
- Three newline-corrupted `consequence` values in the cohort (`"missense_variant\n"`).
- Five LOVD remedies from `INCIDENT_2026-07-19` section 6; LOVD coverage on the 4.4-million-row
  cohort is unmeasured.
- `--seq-windows` resolver directory-versus-file ambiguity (`train.py` is internally consistent
  as a directory).
- Roadmap section 6C stale as of 2026-07-18.
- `tests/EXPECTED_SUITE_SIZE.bak_2026-07-18_ratchet`, a 50,237-byte stale duplicate.
- `v4.0.0` is a lightweight tag; there is no `v3.0.0`.
- `gh auth login` token is dead.
- Session records missing for 2026-05-08, 07-13, 07-14 and 07-15.
- Living metrics glossary and per-model algorithm comparison remain standing debts.

**New today:**

- The mask-loss and base-rate-at-inference `cnn_1d` build (Phase-4 fusion, which dissolves
  out-of-fold stacking) is recorded as a roadmap closing move, not a vague deferral.
