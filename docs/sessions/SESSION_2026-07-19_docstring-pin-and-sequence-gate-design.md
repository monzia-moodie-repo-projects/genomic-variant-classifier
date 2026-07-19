# SESSION 2026-07-19 (evening) — the module header stops restating the roster, and the sequence-provenance gate is designed

**Tree at start:** `2f44ae3` · **at end:** `84c6c54` · Continuous Integration #540 GREEN.
**Suite:** 1978 → 1985 collected. Ratchet measured, never computed.
**Preceding record:** `docs/sessions/SESSION_2026-07-19_part3-xseq-optional.md` (`2f44ae3`),
which closed with Part 3 complete and named the `_att_tune` gate as the next substantive work.
This document covers one shipped repair and the design that precedes that gate.

---

## 1. SHIPPED — the module header of variant_ensemble.py (`84c6c54`)

### 1.1 The defect

`src/genomic_variant_classifier/models/variant_ensemble.py` opened with a header that:

* stated a fixed count of base classifiers — a count the roster had long outgrown;
* enumerated them by name, as a numbered list;
* attributed two of them to a machine-learning framework the module does not import
  (the import block at lines 35-60 contains no such import);
* carried a per-issue changelog section from a project phase left long ago, with entries
  describing edits like a removed parameter and a removed feature column.

That header is the FIRST TEXT anyone reading the project's central module encounters. It was
wrong on the count, wrong on the framework, and headed by a section whose vintage was visible
in its own title.

The stale wording is deliberately not reproduced in this document. It survives in git, and in
`tests/unit/test_module_docstring_is_not_a_stale_roster.py`, where it serves as the negative
control proving each new check would have caught it. A quoted stale claim reads as a live one
to anyone skimming — a lesson learned twice while writing the replacement (§1.4).

### 1.2 The fix is REMOVAL, not correction

Rewriting the number would fix the file on 2026-07-19 and reintroduce the defect the next time
a model is added. The enumeration is a COPY of what `_build_estimators` defines thirty lines
below it, and nothing forces a copy to move when the original does.

The header now states where the fact lives and stops restating the fact:

    the roster            VariantEnsemble._build_estimators
    at runtime            VariantEnsemble(config).base_estimators
    which need sequence   SEQUENCE_MODELS
    the feature contract  TABULAR_FEATURES, EXPECTED_TABULAR_FEATURE_COUNT

The changelog block is deleted outright. Git records that history permanently, with dates and
authorship, which a hand-maintained list in a docstring cannot.

### 1.3 It is pinned, because a rule nobody enforces is the design that just failed

`tests/unit/test_module_docstring_is_not_a_stale_roster.py`, 139 lines, seven tests:

| test | kind |
|---|---|
| header does not claim a model count | substantive |
| the count check rejects the old header | **negative control** |
| header does not enumerate models | substantive |
| the enumeration check rejects the old header | **negative control** |
| header does not attribute a framework to any model | substantive |
| the framework check rejects the old header | **negative control** |
| header points at where the roster actually lives | substantive |

Three of the seven exist solely to prove the checks can FAIL. A checker that has never
rejected anything has not been shown to work — and three checkers written earlier the same day
passed every one of their own controls while counting prose as code.

The last test is the one that keeps the fix honest: removing the copy is only half the job, and
a header that says nothing leaves the reader with nothing. It asserts the three anchors above
are present.

### 1.4 Three defects caught in the replacement before it shipped

Each is the same shape as the defect being repaired.

**The new header quoted the old claim.** The first draft explained the history by reproducing
the exact stale sentence and the exact framework name — putting both back into the first text a
reader sees. The installer's own stale-marker post-check flagged it and would have aborted.

**It then quoted the changelog section name**, one paragraph later, for the same reason and
with the same effect.

**The count-detecting regular expression matched a date.** `2026-07-19` contains `19`, which
sat within the forty-character window the expression allowed before the phrase "base
classifiers" later in the same sentence. A date is not a count claim. The expression now
requires the number to MODIFY the noun — at most two words may intervene — and is verified
against six probes, four that must match and two that must not. Shown here with SUBSTITUTED
numbers, so that this document does not itself restate the count it describes; the literal
probes live in the test file:

    "Implements 9 base classifiers plus a stacking meta-learner."   -> match
    "The 12 base models are permanent."                            -> match
    "seven base classifiers"                                       -> match
    "fourteen permanent base models"                               -> match
    "Until 2026-07-19 this header stated a fixed count of base classifiers"  -> NO match
    "see _build_estimators for the base-model roster"               -> NO match

That third defect is the sixth occasion in this session on which a checker of mine fired on the
prose written to describe the rule. The durable form, applied everywhere now: parse with `ast`,
strip docstrings and comments, then count.

### 1.5 The measurements

```
git add -A ; git status --short   ->  "M  src/.../variant_ensemble.py"
                                      "A  tests/unit/test_module_docstring_is_not_a_stale_roster.py"
                                      (no '??' lines)
pytest tests/ --collect-only -q    ->  "1985 tests collected in 20.27s"
pytest tests/ -q --assert-suite-size -> 1978 passed, 7 skipped in 798.38s
```

Ratchet 1978 → 1985, README badge derived from the ratchet by parsing it back out with the same
function `conftest.py` uses. Four files in one commit — source, test, ratchet, badge — because
splitting them is the failure recorded at ratchet entries 1962 and 1978.

Diff reconciliation, recomputed rather than accepted: README 1 insertion / 1 deletion; ratchet
57 / 1; test file 139 / 0; therefore `variant_ensemble.py` 55 / 32, which sums to the 87
reported and nets to the +23 the dry run predicted. The old docstring was 33 lines and the new
is 56; both end in `"""`, so git matches that line and counts 32 and 55. Nothing is
unaccounted for.

### 1.6 What this commit does NOT fix

Nothing yet asserts that `TABULAR_FEATURES` and `EXPECTED_TABULAR_FEATURE_COUNT` agree with
each other in the way the header is now pinned to the roster. That is a separate pin, of the
same genus, and it is not installed. Recorded in the ratchet entry as well as here.

---

## 2. THE FIFTH INSTANCE OF ONE FAILURE MODE

This project has now recorded the same defect five times. It is worth stating as one pattern
rather than five incidents, because the remedy is identical in every case.

| where | the two copies | how it was found |
|---|---|---|
| `WindowAttachment.__iter__` docstring | a hand-maintained migration list vs. the actual call sites | stale on ALL FOUR entries; deleted in `ea8d6e8` |
| `tests/EXPECTED_SUITE_SIZE` | the recorded number vs. the collected count | four numbers stale before the ratchet was armed (entries 1882, 1891, 1932, 1966) |
| README test badge | the badge vs. the ratchet | badge 1962 against ratchet 1968; Continuous Integration red on both Python 3.11 and 3.12 |
| LOVD classification map | the map's vocabulary vs. the artifact's | zeroed 91.8% of its own artifact |
| `variant_ensemble.py` module header | the enumeration vs. `_build_estimators` | this session |

**A fact stated twice, where only one copy is maintained.** The remedy in every case is to
state it once and point at it — and where a second statement is genuinely useful (a badge, a
ratchet), to DERIVE it mechanically rather than type it, and to fail loudly when the two
disagree.

---

## 3. DESIGNED, NOT YET BUILT — the sequence-provenance gate

### 3.1 The live defect

`scripts/train.py`, as of `84c6c54`:

```
439  _att_test  = attach_delta_windows(meta_test,  seq_windows_path=_seq_win_arg)
455  _att_train = attach_delta_windows(_meta_train, seq_windows_path=_seq_win_arg)
477  _att_tune  = attach_delta_windows(_meta_tune,  seq_windows_path=_seq_win_arg)   # _v2 only
512  for _split_name, _a in (("train", _att_train), ("test", _att_test), ("tune", _att_tune)):
514      logger.info("seq windows [%-5s]: %s", _split_name, _a.summary())
516  has_sequences = _att_train.n_usable > 100
537  if abs(_att_train.usable_fraction - _att_test.usable_fraction) > 0.05:   # warn
556      X_seq_cal_ext=X_seq_tune,
```

All three splits are LOGGED. Only train GATES. The divergence warning compares train against
test. **Nothing anywhere involves tune** — and tune is what feeds `X_seq_cal_ext`, the external
calibration partition.

Reachable state: real train windows, so `cnn_1d` stays in the ensemble; `X_seq_cal_ext` arrives
as a frame of placeholders; the model is FITTED on real sequence and CALIBRATED on fabricated
sequence. Silently.

That is worse than the divergence the file already warns about. Divergent coverage makes an
ablation delta suspect — the reader is told to distrust a number. Calibration on placeholders
produces probabilities that look entirely ordinary and are confidently wrong.

The argument against the omission is already in the file, written at lines 493-495 about a
different split: *the two sides share `_seq_win_arg` so they fail together in practice — but
"in practice" is not an invariant, and the gate must assert the thing it protects.*

### 3.2 A hole in Part 3, which was mine

`_require_x_seq` is called with `X_seq` at `fit:2338`, `predict_proba:2622` and
`evaluate:2648`. It is NEVER called with `X_seq_cal_ext`, which `fit` declares at 2290.

So `cnn_1d` active with `X_seq_cal_ext=None` passes the Part 3 refusal cleanly. Same defect as
§3.1, one layer up, introduced by the commit that was supposed to close this class of problem.

It also settles a design question: a per-parameter check that must be REMEMBERED for each new
parameter is the pattern that keeps failing. The gate must range over everything the method
received.

### 3.3 What reading `seq_window_join.py` established

Collected at `2f44ae3`, SHA-256 `534EFC20…`, 243 lines, and read in full — the file lost 30
lines to the `__iter__` deletion in `ea8d6e8`, so every line number recorded before that commit
is stale.

`WindowAttachment` is `@dataclass(frozen=True)` at lines 90-91:

```
windows: pd.DataFrame     usable: np.ndarray     n_rows: int
n_unmapped: int           n_placeholder: int     provenance: str
@property n_usable -> int   @property usable_fraction -> float   def summary() -> str
```

**`usable` is `np.ndarray`, not `pd.Series`.** My draft protocol had it wrong, and had omitted
`n_placeholder` and `provenance` entirely. This is exactly why the file was read rather than
recalled.

**FIVE PROVENANCE TIERS, AND TWO OF THEM FABRICATE THE MASK:**

| provenance | `usable` derived from | trustworthy |
|---|---|---|
| `rows+ok` | presence AND the builder's `ok` column | yes |
| `rows` | presence only — no `ok` travels with a pre-attached frame | **no** |
| `parquet+ok` | join AND `ok` | yes |
| `parquet` | join only; line 211 sets `ok = np.ones(n, dtype=bool)` | **no** |
| `none` | all `False` | trivially |

Line 211, when the parquet carries no `ok` column, asserts EVERY ROW USABLE. It warns at
195-200 — *builder-placeholder rows CANNOT be identified and will be treated as usable* — but a
warning in a log is not a gate, and `n_usable` afterwards reports a number indistinguishable
from a verified one.

**So `usable=True` means two different things**: "the builder verified this row" or "we could
not check, so we assumed yes." That is the same sentinel collision as the LOVD zero, as
`PLACEHOLDER_BASE`, and as the `X_seq` placeholder frames — the fourth instance.

**Consequence for the design:** a refusal reading `n_usable` alone can be fully satisfied by an
attachment whose mask was invented. The gate must carry `provenance` and treat any tier not
ending in `+ok` as unverified.

### 3.4 The class asks for masking; every consumer uses a threshold

`WindowAttachment`'s own docstring, lines 99-100:

> True iff the row carries sequence that came from the reference genome.
> **Consumers must mask on this and nothing else.**

And at `PLACEHOLDER_BASE`, lines 82-84: *its VALUE is deliberately uninteresting: nothing may
branch on it.*

`train.py` does not mask. It COUNTS — `_att_train.n_usable > 100` — and then passes every row,
placeholders included, to `fit`. The convolutional network trains on the unusable rows too; the
gate decides only whether it trains at all.

That gap is wider than the missing tune gate, and it opens a question that is not an
engineering detail.

### 3.5 [DECISION] Refuse-only, and why not masking

Four shapes were put to Monzia. He chose refusal (option 4), then pushed on whether a superior
build avoids the problem entirely. The exchange is recorded because the answer shapes Phase 4.

**My stated objection to refusal was WRONG and is retracted.** I said it "kills a long job."
The gate sits at `train.py:516`; `PHASE 3: Training` begins at 546. Refusing costs seconds and
happens before a single estimator is built.

**If the ensemble masks rows instead**, `cnn_1d`'s out-of-fold predictions no longer cover every
row. The Logistic Regression stacking meta-learner consumes one out-of-fold column per base
model, aligned across all rows. A column with gaps needs a defined policy — impute, drop those
rows for every model, or let `cnn_1d` abstain with a sentinel, which reintroduces exactly the
sentinel problem Part 3 removed.

**The masking problem cannot be engineered away.** Some variants have no constructible
reference window: structural variants, non-ACGT alleles, contigs the builder cannot resolve.
That is reference-genome reality, not a code smell. Any multi-modal design over a whole-genome
cohort faces unequal modality coverage, and every design either restricts the cohort to
complete cases, represents absence explicitly, or fuses at a level that tolerates it.

**[RECORDED AS THE CLOSING MOVE, not a vague intention]** The superior build: train `cnn_1d`
with unusable rows MASKED OUT OF THE LOSS, and emit the training-set base rate for those rows
at inference. The out-of-fold column is then complete and rectangular — no gaps, no sentinel,
no imputation — and every value is a legitimate probability. The meta-learner learns that a
base-rate output from `cnn_1d` carries no information, which is what is true. That dissolves
the stacking problem rather than managing it, and belongs with Phase 4 fusion work, where it
can be designed rather than squeezed alongside a defect repair.

Refuse-only is therefore the FLOOR, not the design. It is correct precisely because it does not
pretend to answer how the ensemble represents an absent modality.

### 3.6 [DECISION] The gate goes at the type level

The first proposal was a derived set of splits in `train.py`:
`required = {"train","test"} | ({"tune"} if _v2 else set())`.

**I oversold it.** I claimed a future split "cannot be silently omitted, because consuming it
means adding it." Nothing enforces that; the expression is still hand-written from someone's
reading of the file. It is one hand-written thing instead of three, which is better, but it is
not the property I described.

Monzia's objection — *"if it can be done later, it can be done now"* — was correct, and all
three of my reasons for deferring were weak: the defect has been unguarded since `_v2` landed
on 2026-07-11 with no run in flight, so the urgency was manufactured; Part 3 was six commits,
so "not one commit" was never an argument; and the signature having moved three commits ago
argues the other way, since the call sites are fresh and the 152-call census is current.

**The design:** a `SequenceWindows` `Protocol` (`@runtime_checkable`) declaring `windows`,
`usable: np.ndarray`, `n_rows`, `n_unmapped`, `n_placeholder`, `provenance`, `n_usable`,
`usable_fraction`. The ensemble never imports `WindowAttachment` — structural typing, no layer
crossing, and `run_phase2_eval.py` can satisfy the same contract without inheriting anything.

One gate inside `VariantEnsemble`, ranging over EVERY sequence input including
`X_seq_cal_ext`, refusing on: `None`; a bare `DataFrame`; insufficient usable rows; and
unverified provenance. The message names the split, the tier and the counts.

**A bare `DataFrame` is not a legacy special case.** It genuinely carries no provenance, so it
is simply the least-verified tier — equivalent to `provenance="rows"`. One policy applies
uniformly to all six inputs, and no shim with a todo list exists to go stale.

Implementation note from reading `predict_proba:2626` and `evaluate:2658`: both do
`X_input = X_seq` and pass it straight to `model.predict_proba`. The attachment must be
resolved to `.windows` ONCE at the top of each method into a local, rather than teaching three
dispatch sites about the protocol. `predict:2643` delegates to `predict_proba` and needs no
gate. `evaluate` gates at 2648 and then calls `predict_proba` at 2675, which gates again —
harmless today, wasteful once the check does real work.

### 3.7 [OPEN] The threshold, which is a science question

`n_usable > 100` has no stated basis, and applying it uniformly to three splits propagates an
unexamined constant rather than making it principled. It is TWO questions wearing one name:

**Statistical power.** Below how many usable windows can a convolutional network over
101-base-pair windows learn anything? Measurable by a learning curve — subsample usable rows to
100, 300, 1k, 3k, 10k, 30k, train `cnn_1d` at each, find where the confidence interval stops
excluding 0.5. Needs graphics-processing-unit time.

**Selection bias, which is the more dangerous one.** Unusable rows are NOT missing at random —
they are structural variants, non-ACGT alleles, unresolvable contigs. Systematically different
variants. A usable fraction of 0.6 does not mean "40% less data"; it means `cnn_1d` saw a
biased slice and its ablation delta is partly a measurement of variant type rather than of
sequence signal.

**That test costs nothing and needs no training**: compare usable against unusable rows on
label prevalence and on the existing tabular features. If pathogenic prevalence differs
materially, no threshold repairs it — the model is confounded at any coverage, and the honest
remedy is to report `cnn_1d`'s contribution conditional on variant class rather than as one
number.

**Recommendation: run the bias test BEFORE finalising the threshold.** It risks nothing and its
answer could change the design. If the populations are indistinguishable, a fraction threshold
is a straightforward power argument; if they diverge, the fraction is doing bias control and
must be far stricter.

**The precedent for the mechanism already exists.** `EnsembleConfig.zero_variance_min_rows: int
= 10_000`, lines 566-571: *"Below this row count, a constant column is plausibly just sampling…
The guard WARNS there and RAISES above it… This threshold is what lets the guard be armed by
default in the real run without turning every unit-test fixture red."* The sequence gate should
follow that idiom — an absolute floor and a fraction, both per-run overridable — so a
`--max-train 3000` smoke run is not refused by the very gate meant to protect it.

Until measured, the gate ships marked UNVALIDATED, with the per-split numbers in the run
artifact so a reader can check, rather than implying a basis it does not have.

### 3.8 [OPEN, cheap, available now] Provenance in the run artifact

Record each split's `provenance` and `usable_fraction` in the RUN ARTIFACT, not only the log.
`WindowAttachment.summary()` already produces exactly that string. Without it, an ablation
delta for `cnn_1d` cannot be interpreted after the fact — the reader has no way to know whether
the model saw real sequence. This is provenance in the output rather than a check, and it costs
almost nothing.

---

## 4. THE TIMING SERIES, STILL UNEXPLAINED

Full-suite wall-clock, every reading of 2026-07-19, in the order the runs happened:

```
650.17  711.90  639.01  652.09  652.23  709.49  1026.25  727.99  637.60  798.38  657.75
```

Eleven runs. **Nine sit between 637.60 and 727.99 seconds.** Two do not: 798.38s, and 1026.25s
on commit `ea8d6e8` — a commit that REMOVED a test. The slowest run is a little over one and a
half times the fastest.

The suite grew from 1968 to 1985 collected across this span, seventeen more tests. That cannot
account for a run three hundred seconds above the cluster, and both slow runs were followed by
runs inside it.

**What is not claimed here.** Machine load is the obvious explanation and this is a laptop, but
no measurement was taken of it. Two subsequent runs returning to normal is CONSISTENT with load
and does not establish it. The question stays open. A suite that slows without a code reason is
the kind of thing that gets normalised until it is ten minutes worse, so the readings are kept
rather than the conclusion.

**A methodological note, recorded because it recurred four times in one day.** This paragraph
previously expressed the anomaly as a percentage above the median. That figure was quoted as
58% from six timings, corrected to 51% against eight, restated as 51% against ten, and would
now read 56% against eleven — four different numbers, one unchanging reality, and no arithmetic
error in any of them. The computation was right every time and the FORM was wrong every time: a
ratio against a live median goes stale by construction, because the denominator moves whenever
another run happens.

The preceding paragraph diagnosed exactly this and then published the ratio anyway. That is the
failure section 2 catalogues — a fact stated in a form that must be maintained, where nothing
maintains it — appearing inside the section describing it. The form above does not go stale: a
raw series only ever gets longer, and appending to it falsifies nothing already written.

---

## 5. OPEN, CARRIED FORWARD

**Immediately next, in order:**

1. The selection-bias test (§3.7) — no graphics-processing unit, could change the design.
2. `SequenceWindows` protocol and the single gate (§3.6), including `X_seq_cal_ext`.
3. Migrate callers: `scripts/train.py`, `scripts/run_phase2_eval.py`, `scripts/run9_ablations.py`,
   then tests. Delete bare-`DataFrame` acceptance once nothing passes one.
4. `train.py:516` sheds its gate; the divergence warning at 537 reconsidered on its own merits,
   since a RELATIVE comparison measures something different once every split must clear a floor.

**Standing:**

* `TABULAR_FEATURES` vs `EXPECTED_TABULAR_FEATURE_COUNT` has no agreement pin (§1.6).
* The five LOVD remedies in `INCIDENT_2026-07-19_lovd-classification-map-silent-zero.md` §6 —
  scope decision required.
* LOVD coverage on the current cohort is unmeasured; every figure describes the Run 9-era
  1,700,687-row artifact, and the live cohort is 4,399,089 rows.
* `--seq-windows` resolver: one flag, two contracts (directory in `train.py:103`, file in four
  other places).
* Roadmap 6C is a 2026-07-18 snapshot and predates Part 3, the docstring pin, and all of §3.
* `tests/EXPECTED_SUITE_SIZE.bak_2026-07-18_ratchet`, 50,237 bytes — gitignored, never
  committed, a stale duplicate of the project's safety-critical count file beside the live one.
* `v4.0.0` is a lightweight tag on a Dockerfile build fix; no `v3.0.0` exists.
* `git gc` is safe; `gh auth login` needed (token dead).
* Session-record gaps: 2026-05-08, 07-13, 07-14, 07-15.
* Standing documentation debts: living metrics glossary, per-model algorithm comparison.
