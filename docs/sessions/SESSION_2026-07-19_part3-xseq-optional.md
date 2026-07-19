# SESSION 2026-07-19 (afternoon) — Part 3 of the hybrid: X_seq becomes optional

**Tree at start:** `0ebed11` · **at end:** `c78e1ec` · Continuous Integration #538 GREEN.
**Suite:** 1968 → 1967 → 1978 collected. Ratchet moved twice, measured both times.
**Companion record:** `docs/sessions/SESSION_2026-07-19_repo-identity-tags-stash-lovd.md`
covers the same calendar day, 00:00–02:47, and closed before any of this existed. Its section 8
lists Part 3 as "the next substantive engineering work"; this document is that work.

---

## 1. WHAT PART 3 WAS FOR

`scripts/train.py:523-525`, in the branch that removes the convolutional neural network when no
usable sequence windows exist:

```
ensemble.base_estimators.pop("cnn_1d", None)
# X_seq_train / X_seq_test remain valid two-column placeholder DataFrames; with
# cnn_1d removed they satisfy the seq-aware signatures but are unused.
```

That comment is an author documenting a workaround for a signature that demanded a value it
would not use. `VariantEnsemble.fit` took `X_seq: pd.DataFrame` as a required positional
argument, so a caller with no sequence had no way to say so and manufactured one instead. Three
tests did the same — `tests/unit/test_api.py:459`,
`tests/unit/test_base_model_dropout_is_loud.py:113`,
`tests/unit/test_ensemble_persistence.py:30` — the last of them annotated
`# inert: cnn_1d is excluded below`. The fixture docstring in the second was explicit: *"the
signature still requires it, so it is supplied and ignored."*

Roadmap 6.28 had already established why fabricated sequence is not harmless. The annotation
`X_seq: pd.Series` was FALSE — production has always passed a two-column
`[fasta_seq_ref, fasta_seq_alt]` frame — and on a Series `oh_alt - oh_ref` is identically zero,
so four of thirteen channels die and eight duplicate. The sequence model degenerates to a
one-hot classifier carrying no variant information, fits happily, and reports a number.

Part 3 removes the reason to fabricate.

## 2. THE DESIGN DECISION, AND THE MEASUREMENT THAT MADE IT

`y` was the third POSITIONAL parameter of `fit`, which appeared to force a choice between two
bad options. `audit_ensemble_call_sites_2026-07-19.py` measured all four candidate shapes
against reality — 152 `fit`/`evaluate` calls across `src/`, `scripts/` and `tests/`:

```
y passed POSITIONALLY : 22   (18 genuine VariantEnsemble; 4 GroupedConformalClassifier)
y passed BY KEYWORD   :  0
2 positional, no y kw : 108  (scikit-learn estimators, scalers)
OTHER                 : 22
accounted for         : 152 of 152
```

Three shapes were considered and a fourth was added only after reading that output:

- **(a) reorder** to `fit(X_tab, y, X_seq=None)` — REJECTED on sight. `fit(X, X_seq, y)` would
  silently bind `X_seq` to `y`. A change that misbinds without erroring is worse than the
  problem.
- **(b) keyword-only `y`** — would break 18 real call sites LOUDLY, which is survivable, but
  buys only the ability to OMIT the argument.
- **(c) signature UNCHANGED, `X_seq` simply ACCEPTS `None`** — zero breakage. A caller with no
  sequence writes `ensemble.fit(X_train, None, y_train)`.
- **sentinel object** — REJECTED. A distinguished value meaning "absent" is the exact defect
  class Part 3 exists to remove.

**(c) was installed.** Zero keyword-`y` callers meant it cost nothing, and it has a property
(b) does not: absence becomes VISIBLE at the call site rather than implied by an omitted
argument. For a change whose entire purpose is to stop expressing absence implicitly, that is
the stronger form. It also left `scripts/forensics/verify_w2b2.py:45` — which asserts the exact
source substring `        ensemble.fit(X_train, X_seq_train, y_train)` — untouched.

Option (c) was not in the original design. It emerged from reading the census rather than from
reasoning about the signature.

## 3. THE COMMITS

| commit | CI | change |
|---|---|---|
| `ff97c34` | #530 | `X_seq` accepts `None` in fit/predict_proba/predict/evaluate; module constant `SEQUENCE_MODELS`; helper `_require_x_seq` refuses loudly. +74 −7 |
| `11ee5d0` | #531 | `docs/measurements/ENSEMBLE_CALLS_2026-07-19.txt`, the census above |
| `58c9433` | #532 | three tests stop fabricating sequence (Part 3a). +26 −10 |
| `fe4ea94` | #533 | the last four tuple-unpacks become attribute access (Part 3b). +0 −0 lines, four call sites |
| `ea8d6e8` | #534 | `WindowAttachment.__iter__` and its pinning test deleted (Part 3c); ratchet 1968 → 1967. +83 −48 |
| `c78e1ec` | #538 | `tests/unit/test_x_seq_refusal_contract.py`, 11 tests (Part 3d); ratchet 1967 → 1978. +303 −2 |

Superseded by amend, preserved here because their Continuous Integration runs are in the
permanent record: `67a2df4` (#535), `123c7af` (#536), `881282c` (#537).

### 3A. The refusal

```python
SEQUENCE_MODELS: frozenset[str] = frozenset({"cnn_1d"})
```

One module constant, one helper `_require_x_seq`, three call sites — `fit:2338` against
`base_estimators`, `predict_proba:2622` and `evaluate:2648` against `trained_models_` — plus a
defensive assertion at `_leakfree_oof:2156`. Declared once so that adding a second sequence
model is one edit rather than a hunt through dispatch sites.

Optional does not mean tolerated. If a `SEQUENCE_MODELS` member is active and `X_seq` is
`None`, it raises before any estimator is fitted, naming the model, `--skip-cnn`, the
`base_estimators.pop` equivalent, the column names of the frame it wants, and the fact that no
compute has been spent.

### 3B. The shim, and its self-set deletion condition

`WindowAttachment.__iter__` (`seq_window_join.py:136-164`) let
`wins, n_unmapped = attach_delta_windows(...)` keep unpacking after 2026-07-15 changed the
return type to an object. It was deliberately lossy — it dropped `usable`, the provenance mask
that is the entire reason the object exists. Its docstring ended:

> This docstring is the todo list. When the list above is empty, delete this method and the
> `usable`-less path stops existing.

**All four of its claims were stale**, verified individually before the deletion:

| claim | reality |
|---|---|
| `train.py:441,458,480` still unpack | `439/455/477` bind the object. Migrated 2026-07-15. |
| `run_phase2_eval.py:425,426,427` still unpack | reads provenance via loop var `_atts` at 436-448 and GATES on it (`return 2` at 466) |
| `rekey_seq_windows_v2.py:145` still unpacks | migrated 2026-07-15 |
| `train.py:485`'s `_POLY_WIN` detector must be migrated before Run 17 | the detector is GONE. `train.py:480-511` records both its defects; `:516` now reads `has_sequences = _att_train.n_usable > 100` |

That last one was checked before anything was deleted, precisely because a docstring is the
kind of artifact that can hold the only record of a live defect — the same shape as the
seventy-two-day stash recovered twelve hours earlier. It did not. The detector's replacement is
provenance-based and cannot rot when a filler changes, because it never inspects the filler.

The docstring insisted it was *"accurate as written, not aspirational."* It was stale anyway.
A claim about a document's own freshness cannot make the document fresh, which is why it was
deleted rather than corrected.

### 3C. The eleven tests

`_require_x_seq` had no direct test. It was exercised only through its SILENT branch — calls
passing `X_seq=None` with a roster that happens to exclude `cnn_1d` — so nothing asserted the
loud branch was reachable at all. **A guard whose firing path is never executed is
indistinguishable from a guard that has been disarmed**, and three of those are on record: the
SpliceAI silent zero (`9ba3127`), `rekey_seq_windows_v2`'s gate, and `train.py`'s `_POLY_WIN`
detector, which went unconditionally true when `PLACEHOLDER_BASE` changed from `"A"` to `"N"`
while the full suite stayed green.

Five tests expect the refusal, **five are negative controls that must stay silent**, one checks
that `SEQUENCE_MODELS` names only models the default roster actually builds. The pairing is the
point: a test that only ever sees the raise cannot distinguish a correct guard from one that
raises always.

`test_fit_refuses_before_training_anything` asserts BY STATE — `trained_models_` empty AND no
artifact on disk. On the 4.4-million-row cohort that property is the difference between a
second and hours of paid compute.

`X_tab` is built from real `TABULAR_FEATURES` columns because `_assert_no_dead_features:2334`
runs BEFORE the refusal at `:2338`. With `f0..f19` columns the census would raise first and
every `pytest.raises` in the file would pass for the wrong reason.

All eleven passed on the first run, with no adjustment.

## 4. THREE RED RUNS, NONE OF THEM A CODE DEFECT

Continuous Integration #535, #536 and #537 all failed identically:

```
SUITE-SIZE RATCHET FAILED (roadmap 6.14)
  expected (tests/EXPECTED_SUITE_SIZE): 1967
  actually collected:                   1978
  11 MORE test(s) than expected.
```

Cause, in order:

1. **#535** — the tests were committed and pushed without the ratchet bump. The commit command
   supplied to the operator contained an unsubstituted placeholder,
   `-m "... ratchet <measured>"`, which PowerShell passed through verbatim because it sat
   inside a quoted string. That text became `67a2df4`'s permanent message. The same block was
   printed directly beneath an instruction to stop and report the count first — and an
   instruction to stop, printed above five runnable commands, is not an instruction to stop.
2. **#536 and #537** — the bump script had not been downloaded.
   `Get-FileHash` reported `Cannot find path`, `python` reported `[Errno 2]`, and the amend
   sequence that followed ran anyway because it had no gate on the script having succeeded.
   Two amends and two force-pushes later, `881282c`'s message claimed "ratchet 1978" while
   `tests/EXPECTED_SUITE_SIZE` still read 1967 — a message now false rather than merely
   unfilled.
3. Resolved at `c78e1ec`: bump applied, ratchet and badge both 1978, suite 1971 + 7 = 1978,
   one final amend so the message matches its diff.

**The gate was correct every time.** It cost three red runs of about three minutes each and
prevented a ratchet that silently stops describing the suite — which this file's own history
records happening four times. The failure mode being defended against is not a red build; it is
a green build that means nothing.

The ratchet moved twice this span and was MEASURED both times: 1967 by
`pytest --collect-only -q` on the staged tree, 1978 by the ratchet's own failure output. Never
computed. The README badge was DERIVED from the ratchet by parsing it back out, never typed —
the direct fix for the 2026-07-18 failure where the ratchet said 1968 and the badge said 1962.

## 5. DEFECTS IN MY OWN WORK THIS SPAN

Eight. Three of them are the same mistake.

**5.1 — the unsubstituted placeholder.** `-m "... ratchet <measured>"`, supplied and run.
Already in my own PowerShell notes as a rule; violated anyway.

**5.2 — a runnable command block under an instruction to stop.** #535 follows directly.

**5.3 — an amend sequence with no precondition gate.** Steps 4-7 assumed step 2 had succeeded.
It had not, because the file was never downloaded. #536 and #537 follow directly.

**5.4, 5.5, 5.6 — three checkers counted PROSE as CODE.** Part 3a's splatted-call census
counted its own new docstring's `` `ens.fit(*_tiny_inputs())` `` as an eighth call site. Part
3b's first draft counted `test_train_cnn_activation.py:26-41`, a docstring quoting the old
unpacking form, as a live call site. The structural audit of the new test file counted the
module docstring's quoted `pd.Series` as live construction. All three now parse with `ast` and
strip docstrings and comments before counting.

**5.7 — the blank-line repair chose its basis from the wrong side.** `cut()` took the blank
count from the DELETED block's indentation; `__iter__` is indented, so it kept one blank line
where a following top-level `def` needs two. Eleven post-checks passed on that output because
none of them looked at whitespace.

**5.8 — and the check added to catch 5.7 was itself wrong twice.** It treated `@` and `class`
as separate definition starts, so for `@dataclass` / `class WindowAttachment:` it demanded two
blank lines before the `class` line that sits directly beneath its own decorator — a
PRE-EXISTING property of `seq_window_join.py:91`. And it asserted GLOBAL PEP 8 cleanliness when
the only question a patch can answer is whether the edit made spacing worse. It aborted a
correct deletion. Now: decorated definitions are attributed to their decorator, and the result
is diffed against the original so only a NEW violation fails.

**The pattern.** Every one was caught by reading output or running a control — none by
re-reasoning. And the three prose-as-code miscounts are notable because each was fixed
narrowly, and the flaw carried into the next script anyway. The durable fix was structural
(parse, then count), not another regular expression.

## 6. AN UNEXPLAINED MEASUREMENT

Full-suite wall-clock across today's runs, in order:

```
650.17  711.90  639.01  652.09  652.23  709.49  1026.25  727.99
```

The 1026-second run was `ea8d6e8` — a commit that REMOVED a test. The median of these eight is
680.86s, so that run sits **51% above it**. Machine load is the likeliest explanation and this
is a laptop, but likeliest is not measured. The following run returned to 727.99s, which is
consistent with load and does not establish it. Recorded rather than dismissed; a suite that
slows without a code reason is the kind of thing that gets normalised until it is ten minutes
worse.

(An earlier draft of this paragraph said 58%, computed from the first six timings before the
last two existed. The figure was recomputed against all eight before this document was
committed. A percentage quoted from a superseded sample is the same defect class as every
stale docstring in section 3B, at a much smaller scale.)

## 7. OPEN, CARRIED FORWARD

- **`_att_tune`'s missing gate** (`scripts/train.py:477`). All three splits are LOGGED at
  512-514, but `has_sequences` reads TRAIN alone. `_att_tune` feeds `X_seq_cal_ext`, the
  calibration split. The comment at 493-495 supplies the argument against its own omission —
  *"the two sides share `_seq_win_arg` so they fail together in practice, but 'in practice' is
  not an invariant, and the gate must assert the thing it protects"* — and that reasoning
  applies identically to tune. Production, not test hygiene; its own commit.
- **The five LOVD remedies** in `docs/incidents/INCIDENT_2026-07-19_lovd-classification-map-silent-zero.md`
  section 6. Scope decision required: parenthetical normalisation is a bug fix recovering 36
  discarded pathogenic calls; a presence indicator and a functional-effect feature both move
  the feature contract.
- **LOVD coverage on the current cohort** — every figure measures the Run 9-era
  `models/v1/clinvar_enriched.parquet` (1,700,687 rows). The live cohort is 4,399,089 rows.
- **`--seq-windows` resolver** — one flag, two contracts (directory in `train.py:103`, file in
  four other places).
- **Roadmap 6C** is a 2026-07-18 snapshot and predates everything in this document.
- `tests/EXPECTED_SUITE_SIZE.bak_2026-07-18_ratchet`, 50,237 bytes — gitignored, never
  committed, but a stale duplicate of the project's safety-critical count file sitting beside
  the live one.
- `v4.0.0` is a lightweight tag on a Dockerfile build fix; no `v3.0.0` exists.
- `git gc` is safe (the three tag orphans are unreferenced); `gh auth login` (token dead).
- Session-record gaps: 2026-05-08, 07-13, 07-14, 07-15.
- Standing documentation debts: living metrics glossary, per-model algorithm comparison.
