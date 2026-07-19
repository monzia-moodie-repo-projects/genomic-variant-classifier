# SEQUENCE-BRANCH AUDIT AND THE DESIGN FOR PART 3 — 2026-07-18

Status: **findings measured, design proposed, no code written.**
Evidence: `docs/measurements/SEQ_BRANCH_STATE_2026-07-18.txt`,
`docs/measurements/WINDOW_MIGRATION_2026-07-18.txt`,
`docs/measurements/WINDOW_SITES_2026-07-18.txt`.
Tree audited: `9cb8241`. Suite 1,968 collected, 1,961 passed, 7 skipped, Continuous
Integration green (#526).

This document exists because ten findings about the sequence branch were established on
2026-07-18 and none of them was written down. It records what was measured, what was measured
WRONGLY and then corrected, and what follows from both. No source file is modified by this
document.

---

## 1. THE HEADLINE: THE SELF-DELETING TODO LIST IS STALE ON ALL THREE ENTRIES

`WindowAttachment.__iter__` in `src/genomic_variant_classifier/data/seq_window_join.py`
carries a hand-maintained list that ends: *"This docstring is the todo list. When the list
above is empty, delete this method and the `usable`-less path stops existing."*

It claims three files still unpack the deprecated lossy two-tuple. Re-derived from the abstract
syntax tree on 2026-07-18:

| Claimed "STILL UNPACKING" | Measured |
|---|---|
| `scripts/train.py:441,458,480` | **MIGRATED** — binds at 439/455/477; reads `n_usable@516,521,530`, `usable_fraction@537,542` |
| `scripts/rekey_seq_windows_v2.py:145` | **MIGRATED** — reads `n_unmapped@146`, `usable@172` |
| `scripts/run_phase2_eval.py:425,426,427` | **MIGRATED** — binds at **436/437/438**; the claimed line numbers are also wrong |

**Not one of the three unpacks a tuple.** The document that decides when the lossy path may be
deleted has overstated the remaining work since 2026-07-15, and its line numbers for
`run_phase2_eval.py` are off by eleven.

This is roadmap section 7, root pattern (a) — a fact written down once and never re-derived —
occurring inside the mechanism built to close a root-pattern-(a) defect. The list was accurate
when written; `train.py` (+68/-27) and `run_phase2_eval.py` (+48/-14) both changed the same day
and landed in `e57835e`, and nothing re-checked it.

**The correction belongs in the same change as Part 3, and it must be DERIVED from the audit
rather than retyped, or it will be stale again by the next commit.**

## 2. THE AUDIT'S OWN BLIND SPOT, AND THE TWO FALSE POSITIVES IT PRODUCED

`audit_window_migration_2026-07-18.py` classifies a call site by asking whether the result is
bound to a tuple target or a single name, then looks for `<bound_name>.usable` and its
siblings. **It cannot follow a provenance read through a loop variable, a comprehension
variable, an alias, or a helper function.** It reports the absence of a DIRECT read, which is
not the absence of a read.

That produced two false positives, both resolved by reading the source rather than by
reasoning about it:

**(a) `scripts/run_phase2_eval.py` reported PARTIAL.** It is not merely migrated; it is the
most rigorous consumer in the repository. Lines 442–448:

```python
for _split_name, _a in _atts:
    logger.info("seq windows [%-5s]: %s", _split_name, _a.summary())
_n_tot           = sum(a.n_rows        for _, a in _atts)
_unmapped_tot    = sum(a.n_unmapped    for _, a in _atts)
_placeholder_tot = sum(a.n_placeholder for _, a in _atts)
_unusable_tot    = sum(a.n_rows - a.n_usable for _, a in _atts)
```

and it gates at 466: `if not skip_cnn and _unusable_tot > 0.005 * _n_tot: return 2`. Its own
comment explains why the previous measure was wrong — `n_unmapped` counts only key-join misses
and is blind to builder placeholders, so the gate "was measuring the smaller of the two failure
modes and calling it coverage."

**(b) `_att_tune` in `scripts/train.py` reported "provenance read: NONE".** Line 512 loops over
all three attachments and calls `_a.summary()`, which internally reads `n_usable`,
`usable_fraction`, `n_unmapped` and `n_placeholder`. Same blind spot.

**A third reporting flaw, in `collect_seq_branch_state_2026-07-18.py`:** its signature printer
omits decorators, so `n_usable` and `usable_fraction` — both `@property` — were rendered as
`def n_usable(self) -> int`. A reader would conclude that `a.n_rows - a.n_usable` at
`run_phase2_eval.py:448` subtracts a bound method from an integer and raises `TypeError`. It
does not; they are properties. Confirmed by reading `seq_window_join.py`, whose SHA-256
(`aa236d739d863a60`, 273 lines) is byte-identical between the audited tree and the reference
copy.

Recorded because the pattern is the lesson: **three tools built this session to check other
people's claims each produced a claim of their own that needed checking.** Each was caught by
reading source, none by re-reasoning.

## 3. THE ONE GENUINE GAP: THE TUNE SPLIT'S MASK IS COMPUTED, LOGGED, AND DISCARDED

`scripts/train.py`, measured 2026-07-18:

| Attachment | Provenance | Gated? |
|---|---|---|
| `_att_train` (455) | `n_usable@516,521,530`, `usable_fraction@537,542` | **YES** — `has_sequences = _att_train.n_usable > 100`; below it, `cnn_1d` is popped |
| `_att_test` (439) | `n_usable@531`, `usable_fraction@542` | **YES** — divergence warning at 537 |
| `_att_tune` (477) | logged at 514 via `summary()` | **NO** |

`X_seq_tune = _att_tune.windows` at line 478 takes the **unmasked** frame, and line 556 passes
it as `X_seq_cal_ext=X_seq_tune` into `ensemble.fit`. That is the gene-disjoint tune partition
used to fit the tree models' isotonic calibration step.

So if the tune split carries builder placeholders, **the calibrator is fitted partly on
fabricated sequence, and nothing warns.** The asymmetry is the tell: the code one screen above,
at lines 533–536, argues explicitly that coverage-dependent inputs make a model's *"measured
contribution an artefact of coverage rather than of biology"* — and then applies that reasoning
to train-versus-test only. Calibration shapes the probability outputs directly, so the argument
applies at least as strongly there.

**This is not currently causing harm on the configured path** — the artifact measures 723
placeholders in 4,399,089 rows (0.0164%), so the tune split's contamination is small. It is a
missing gate, not an active fire. But "small on today's artifact" is exactly the reasoning the
0.5% razor comment at `run_phase2_eval.py:455–465` warns against.

## 4. PART 3'S JUSTIFICATION IS ALREADY WRITTEN IN THE CODE

`scripts/train.py:523–525`, when `cnn_1d` is removed for lack of usable sequence:

```python
ensemble.base_estimators.pop("cnn_1d", None)
# X_seq_train / X_seq_test remain valid two-column placeholder DataFrames; with
# cnn_1d removed they satisfy the seq-aware signatures but are unused.
```

That is the author documenting a workaround for a signature demanding a value it will not use.
Fabricated data is constructed and passed for no reason other than to satisfy a type contract.
Part 3 removes the reason.

## 5. THE REMAINING CONSUMERS OF THE LOSSY SHIM

Measured, and smaller than the todo list implies. **Four tuple-unpack sites, all in tests:**

| Site | Nature |
|---|---|
| `tests/unit/test_seq_window_join.py:36` | incidental — inside `test_keyjoin_alignment_survives_shuffle_and_filter`, testing ROW ALIGNMENT |
| `tests/unit/test_seq_window_join.py:46` | incidental — same file, same pattern |
| `tests/unit/test_train_cnn_activation.py:101` | incidental — inside an alignment test: `w, n_unmapped = attach_delta_windows(mt)` |
| `tests/unit/test_seq_window_join.py:152` | **deliberate** — `test_back_compat_iter_still_unpacks_for_unmigrated_callers`, the shim's own test |

The first three reach for the shim out of convenience and would be clearer as `att.windows` /
`att.n_unmapped`. The fourth exists to pin `__iter__` and must be deleted in the same change
that deletes `__iter__`, not before.

`src/genomic_variant_classifier/data/genomic_lm.py:352` is an inline `return
attach_delta_windows(...)` inside `_resolve_windows`, whose docstring states *"returns the
WindowAttachment itself … Callers must mask on `.usable`."* A passthrough; provenance survives.
Correctly migrated.

## 6. TWO SURVIVING INSTANCES OF THE 6.28 ANNOTATION DEFECT

Roadmap 6.28 records that `X_seq: pd.Series` on `fit`/`predict_proba`/`evaluate` was FALSE —
`train.py` has always passed a DataFrame — and that every test passed a Series *because the
signature said so*, leaving the suite green on a code path the run never executes.

The annotation was corrected in `variant_ensemble.py` (now `X_seq: pd.DataFrame`, with the
docstring *"A 2-COLUMN [fasta_seq_ref, fasta_seq_alt] DataFrame, row-aligned to X_tab"*). It
was **not** corrected in two other places:

**(a) `src/genomic_variant_classifier/evaluation/prediction_artifacts.py:343`** still declares
`X_seq_test: pd.Series`, and at lines 371 and 381 passes it to
`ensemble.predict_proba(X_tab_sub, X_seq_sub)` — the permutation-importance path. The fix
landed in the callee and not the caller.

**(b) Three tests still construct a Series and pass it to `fit`:**

```
tests/unit/test_api.py:459                    X_seq = pd.Series(["A" * 101] * n)
tests/unit/test_base_model_dropout_is_loud.py:113   X_seq = pd.Series(["ACGT" * 8] * n)
tests/unit/test_ensemble_persistence.py:30    X_seq = pd.Series(["A" * 101] * n)  # inert: cnn_1d is excluded below
```

They are green only because `cnn_1d` is excluded in each. The comment on the third says
`# inert` — the author knew the value was fake. Under Part 3 these become `X_seq=None`, which
is the honest expression of what they already mean.

## 7. A SCRIPT THAT WOULD REINSTALL THE FABRICATION

`scripts/patch_train_cnn_activation.py` carries string payloads that write back:

```
"        X_seq_train = pd.Series([\"A\" * 101] * len(y_train))\n"
"        X_seq_test  = pd.Series([\"A\" * 101] * len(y_test))\n"
```

It is an already-applied patch script. Running it again would restore the fabrication that
Phase 3b removed. It does **not** trip the poly ban, correctly: the abstract syntax tree sees a
string constant containing that text, not a `Constant * Constant` expression. The hazard is
real but inert, and it is the same retirement class as
`scripts/download_finngen_R10_DEPRECATED.py`.

Related, and worth knowing before editing `train.py`:
`scripts/forensics/verify_w2b2.py:45` asserts the exact source substring
`"        ensemble.fit(X_train, X_seq_train, y_train)"`. Part 3 changes that line, and that
check will break.

## 8. `--seq-windows` — §6.29a's PROPOSED FIX IS NOW THE WRONG DIRECTION

Roadmap 6.29a proposed converging on **directory** semantics and repointing four launchers plus
`smoke_all_models.py` and `preflight_gate.py` at `data/processed/seq_windows`. Measured today:

| Caller | Names | Semantics |
|---|---|---|
| `scripts/train.py:103` | `data/processed/seq_windows` | **directory** |
| `scripts/run_phase2_eval.py:50` | `clinvar_grch38_clean_seq.parquet` | file |
| `scripts/smoke_all_models.py:31` | `clinvar_grch38_clean_seq.parquet` | file |
| `scripts/preflight_gate.py:27` | `clinvar_grch38_clean_seq.parquet` | file |
| `scripts/launch_run17_baseline.sh:181` | `clinvar_grch38_clean_seq.parquet` | file |

**One caller uses directory semantics; five use file semantics.** And 6.29a chose against the
file only because it lacked an `ok` column — disproved on 2026-07-18, when the artifact measured
21 columns WITH `ok`, 4,399,089 rows, 4,398,366 usable, 723 placeholder.

`train.py` has a real reason for the directory: lines 289–296 run
`verify_seq_windows(raw_df, _seq_win_dir, args.reference).raise_if_failed()`, which needs the
manifest beside the parquet. So neither semantics is simply wrong.

**Proposed direction, to be designed separately from Part 3:** one resolver — something of the
shape `resolve_seq_window_source(path) -> SeqWindowSource(parquet, manifest_or_None,
provenance)` — called by both scripts. A path that is a directory resolves to the parquet and
manifest inside it; a path that is a file resolves to itself with `manifest=None` and the
manifest gate explicitly SKIPPED WITH A RECORDED REASON rather than silently. That is one
meaning for the flag, resolved by inspection, with the resolution stated in the run record. It
is a smaller change than repointing five callers, and it preserves `train.py`'s coherence gate
instead of discarding it.

Also still true, from 6.29a: `preflight_gate.py:91–92` fails only when `--seq-windows` is
EMPTY. A non-empty path to a stale artifact passes a check named *"silent CNN degradation"*.

## 9. THE DESIGN FOR PART 3

**Goal:** remove the CLASS of defect in which fabricated sequence is manufactured solely to
satisfy a signature.

**Contract change.** In `VariantEnsemble`:

```
fit(X_tab, X_seq, y, ...)               ->  fit(X_tab, X_seq=None, y=..., ...)
predict_proba(X_tab, X_seq)             ->  predict_proba(X_tab, X_seq=None)
predict(X_tab, X_seq)                   ->  predict(X_tab, X_seq=None)
evaluate(X_tab, X_seq, y)               ->  evaluate(X_tab, X_seq=None, y=...)
```

Keyword ordering must be handled carefully: `y` is currently positional third. Making `X_seq`
optional while keeping `y` positional requires either keyword-only `y` or a sentinel. The
sentinel approach is rejected — it reintroduces "a value that means absent" — so `y` becomes
keyword-or-positional with `X_seq` defaulting to `None` only where the signature permits.
**This is the one part of the design that must be settled against the real call sites before
implementation, and there are 85 `X_seq` references across `src/`, `scripts/` and `tests/`.**

**The loud failure.** If `cnn_1d` is in `base_estimators` and `X_seq is None`, raise — naming
the model, the flag that would remove it (`--skip-cnn`), and the reason. Never warn, never
substitute a placeholder. A model that needs sequence and is handed none is a configuration
error, not a degradation to absorb.

**What this deletes.** The `train.py:523–525` workaround; the three tests' fake Series; and the
need for any caller to invent a two-column frame it will not use.

**Tests, each with a negative control:**

1. `fit` with `X_seq=None` and `cnn_1d` absent — succeeds, and no sequence-shaped object is
   constructed anywhere in the call.
2. `fit` with `X_seq=None` and `cnn_1d` PRESENT — raises, and the message names `cnn_1d`.
3. The raise happens BEFORE any model is fitted, so the failure costs no compute.
4. `predict_proba` / `evaluate` with `X_seq=None` on an ensemble fitted without sequence.
5. Passing a `pd.Series` where a two-column DataFrame is required still raises (the existing
   strict `_encode_batch` behaviour must not regress).
6. A control proving test 2 can fail: with `cnn_1d` present and a valid `X_seq`, no raise.

**Sequencing, and what must NOT be bundled:**

- Part 3 proper (contract + loud failure + tests).
- The three tests migrated from `pd.Series([...])` to `X_seq=None`.
- `prediction_artifacts.py:343` annotation corrected, with its call sites checked.
- The three incidental tuple unpacks migrated to attribute access.
- `WindowAttachment.__iter__` and `test_back_compat_iter_still_unpacks_for_unmigrated_callers`
  deleted together, once the three above are migrated — the shim's own docstring authorises
  exactly this.
- The stale todo list corrected, derived from the audit.
- `verify_w2b2.py:45`'s source-substring assertion updated in the same commit as the
  `train.py` line it pins.

**Deliberately NOT bundled:** the `_att_tune` calibration gate (§3) and the `--seq-windows`
resolver (§8). Both are real, both are separate concerns, and bundling three architectural
changes into one commit is how a reviewable change becomes an unreviewable one.

**Suite-size impact:** Part 3 adds tests, so `tests/EXPECTED_SUITE_SIZE` moves. The count must
be MEASURED on the staged tree and the README badge re-derived in the same commit — the
2026-07-18 Continuous Integration failure (#522, #523) was exactly that omission.

## 10. OPEN, CARRIED FORWARD

- `_att_tune` calibration gate (§3) — missing, not broken.
- `--seq-windows` resolver (§8) — 6.29a's remedy needs re-deriving.
- `preflight_gate.py:91` fails only on an EMPTY value.
- `scripts/patch_train_cnn_activation.py` — retirement candidate (§7).
- `data/primateai3d.py` — a connector with no feature in `TABULAR_FEATURES` (roadmap 6A,
  2026-07-15), still undispositioned.
- Run 9 ablation coverage gap: six external annotation families with no mask — ribonucleic acid
  sequencing (5 features), COSMIC (2), KEGG (2), GenomicLM / Nucleotide Transformer (2),
  Reactome (1), heterogeneous graph neural network (1).
- Session-document gap: `docs/sessions/` runs 2026-07-06 → 2026-07-18; 07-13, 07-14 and 07-15
  have no session record.
