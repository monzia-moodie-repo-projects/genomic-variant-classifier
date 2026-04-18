# INCIDENT 2026-04-17: ESM-2 silent-zero caused by missing HGVSp parser

**Discovered**: 2026-04-17 afternoon, during Run 9 preflight construction.
**Severity**: Low (no crash, no data corruption; a feature that was
supposed to contribute has contributed 0 for three consecutive runs).
**Status**: Open. Documented for Run 10 remediation.
**Affected runs**: Run 6, Run 7, Run 8 (ESM-2 contributed 0 in all three).
**Blocker for Run 9**: No. ESM-2 will continue to be stub in Run 9, but
this is now explicit rather than silent.

## Summary

The ESM-2 connector (`src/data/esm2.py`) requires four columns in its
input DataFrame: `gene_symbol`, `protein_pos`, `wt_aa`, `mut_aa`. When
any are missing, it logs an INFO-level message and returns zeros. The
training pipeline never populates `protein_pos`, `wt_aa`, or `mut_aa` --
grep across `src/` shows those columns are READ only by `esm2.py` and
`eve.py`, and WRITTEN by neither. As a result, every training run since
the connector was added has produced `esm2_delta_norm = 0.0` for every
variant.

## Detection

Built `tests/unit/test_esm2_activation.py` during Run 9 prep. Test
failed on `assert (deltas > 0).any()` with `deltas = [0.0, 0.0, 0.0]`
despite `_BACKEND == "transformers"`, `transformers 5.5.4` installed,
and `from transformers import AutoTokenizer, EsmModel` succeeding.

Investigation sequence:
1. Confirmed backend initialization: `python -c "from src.data import
   esm2; print(esm2._BACKEND)"` -> `transformers`. Real mode was live.
2. Read connector's `annotate_dataframe` source: line 400-408 returns
   zeros and logs if `required - set(df.columns)` is non-empty.
3. Grep'd pipeline for column writers:
   ```
   Get-ChildItem -Path src -Recurse -Filter "*.py" |
       Select-String -Pattern "wt_aa|mut_aa|protein_pos" |
       Group-Object Path
   ```
   Result: only `src/data/esm2.py` (17 hits) and `src/data/eve.py` (4
   hits). No writers anywhere in the pipeline. ESM-2 and EVE have both
   been silently inert.

## Root cause

VEP HGVSp annotation produces strings like `p.Arg1699Gln` but never
parses them into `(1699, 'R', 'Q')`. The pipeline has no HGVSp parser
to bridge the gap between what VEP emits and what `esm2.py` expects.
The connector's own docstring flags this on line 26: *"If protein_pos
/ wt_aa / mut_aa are absent (common when VEP HGVSp is not parsed yet),
returns 0.0 with an INFO log."* The INFO log is not loud enough to
surface during production runs -- nothing was grepping for it, and
feature-importance analysis couldn't distinguish "signal is truly
weak" from "feature was never populated".

This is the same failure class as the SpliceAI silent-zero that was
fixed in commit 9ba3127, but in a different layer: SpliceAI's fix was
connector-level (path defaulting), this one is pipeline-level (missing
upstream parser).

## Why this wasn't caught sooner

Run 8 feature-importance ranking put AlphaMissense 7th of 78 features,
with ESM-2 absent from the top features entirely. This looked like
"ESM-2 is weaker than AlphaMissense at predicting pathogenicity",
which is an entirely believable result. The hypothesis that ESM-2's
importance was zero-because-feature-was-zero was never tested.

General lesson: "this feature contributes less than expected" and
"this feature is literally zero" produce identical feature-importance
artifacts. Distinguishing them requires a pipeline-level audit of
NaN/zero fractions per feature, not just SHAP ranking. Worth adding
to the agent_layer routine checks.

## Verification (per 2026-04-17 gcloud-storage rule)

N/A for this INCIDENT -- no GCS artifacts involved. Verification is
the reproducible test output:

```
python -m pytest tests/unit/test_esm2_activation.py::test_esm2_not_in_stub_mode -v
# AssertionError: all esm2_delta_norm values are 0.0: [0.0, 0.0, 0.0]
```

and the grep:

```
Get-ChildItem -Path src -Recurse -Filter "*.py" |
    Select-String -Pattern "wt_aa|mut_aa|protein_pos" |
    Group-Object Path
# Only esm2.py (17) and eve.py (4). No writers.
```

## Remediation plan

**For Run 9 (this week)**: Accept ESM-2 stub mode as a known state.

- `tests/unit/test_esm2_activation.py` now has three tests:
  1. `test_esm2_emits_delta_norm_column` -- gates API drift
  2. `test_esm2_not_in_stub_mode` -- passes when input has all four
     required columns; gates the real-mode path for when Run 10 lands
     the parser
  3. `test_esm2_stub_mode_expected_when_columns_missing` -- documents
     current pipeline state; fails if the connector starts silently
     inferring the parsed columns (which would mask future bugs)

- Run 9 training log will contain `"ESM-2 stub mode -- all
  esm2_delta_norm values = 0.0"` WARNING. The launch runbook grep
  checklist now expects this and does NOT treat it as failure. When
  Run 10 lands the parser, remove that line from the expected-log set.

**For Run 10 (next)**: Add `src/data/hgvsp_parser.py`.

Input: a DataFrame with `hgvsp` (string like `ENSP00000350283.3:p.Arg1699Gln`)
or `protein_change` (`p.Arg1699Gln`).

Output: same DataFrame plus three new columns `protein_pos` (int),
`wt_aa` (single-letter code), `mut_aa` (single-letter code).

Must handle:
- 3-letter codes (`Arg`, `Gln`, `Ter`) and 1-letter codes (`R`, `Q`, `*`)
- Missense (`p.Arg175His`), nonsense (`p.Arg175Ter`), silent (`p.Arg175=`)
- Extension (`p.Ter175Cys_ext*5`) -- return NaN
- `p.?` (predicted, unknown) -- return NaN
- Indels at protein level (`p.Gly10_Ala12del`) -- return NaN
- Synonymous shorthand (`p.=`) -- return NaN

Non-goals: do NOT try to parse cDNA-level HGVS (`c.123A>G`); assume
upstream VEP already produced `p.` notation. NaN outputs are fine --
ESM-2 will naturally skip those via `candidates = df[... .notna()]`
on line 412.

Testing: unit test against a curated fixture of ~50 HGVSp strings
covering all the above cases. No external dependencies.

Wire into pipeline: call from `real_data_prep.py` between VEP
annotation and ESM-2/EVE annotation steps. ~1 line change plus the
new module.

**Estimated effort**: 1-2 days. Not suitable for cramming into a
session tail.

## Post-remediation validation

Once the parser lands:
1. Re-run `test_esm2_not_in_stub_mode` on the current-HEAD venv and
   confirm it PASSES with non-zero `esm2_delta_norm` values.
2. Delete `test_esm2_stub_mode_expected_when_columns_missing` (its
   premise no longer holds -- pipeline now populates the columns).
3. Run a mini eval on a 10K-variant subset to confirm ESM-2 makes the
   top-20 feature list.
4. THEN launch Run 10.

## Related INCIDENTs / commits

- `9ba3127` (2026-04-17 morning): SpliceAI silent-zero fix. Same
  failure class (feature contributes 0 across multiple runs, detected
  during preflight construction).
- `INCIDENT_2026-04-09_spliceai-vcf-size.md`: same pattern -- a
  feature-level bug that hid because nothing was auditing zero-fractions.

## Updates

- 2026-04-17: INCIDENT filed. Run 9 proceeds with known ESM-2 stub.