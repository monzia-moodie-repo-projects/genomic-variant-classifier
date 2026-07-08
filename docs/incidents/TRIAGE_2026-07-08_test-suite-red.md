# TRIAGE 2026-07-08 — Test suite is RED (24 of 1,616), and has been for some time

**Status:** OPEN. Root causes localised for all four clusters; two require one file read each.
**Discovered:** 2026-07-08, while validating `clean_cohort.py` v2 (commit `2f6dcde`).
**Author:** Monzia Moodie
**Reproduce:** `python -m pytest -q` → `24 failed, 1585 passed, 7 skipped in 365.51s`

---

## 0. Executive summary

Running the full suite for the first time in this session revealed **24 failures across six files**.
None is caused by `clean_cohort.py` v2 — proved by isolation experiment (§1). They fall into four
clusters:

| # | cluster | count | root cause | status |
|---|---|---:|---|---|
| **A** | protein-coord coverage gate raises on unit fixtures | **12** | The gate's precondition keys off **filesystem presence**, not declared configuration. Green on a clean box, red on this one. | root cause ESTABLISHED |
| **B** | `AttributeError: 'ESM2Connector' object has no attribute 'cache_path'` | **6** | The `conn` fixture constructs the object **without running `__init__`**. Code-vs-fixture drift. | localised; fixture unread |
| **C** | correctness-harness allowlist stale | **1** | `KNOWN_ZERO_DEFAULT` predates the 91→97 feature work. | ESTABLISHED |
| **D** | run-17 audit fixtures stale | **5** | Split fixtures never write `genomiclm_*` / `kegg_*`; the audit marks two of them FAIL-severity. | ESTABLISHED |

**The meta-finding is larger than any cluster.** The standing project brief records *"596 tests pass."*
The suite now collects **1,616** and **24 fail**. Six of those (C and D) were introduced by this
project's own 91→97 feature work. Nobody noticed, because **no gate runs the test suite**:
`Run_Preflight_VM.sh` checks GPU, CUDA, VRAM, deps, disk, RAM and git HEAD; `vm_bootstrap_run.sh`
checks the environment and the models. Neither runs `pytest`. On 2026-07-06 we shipped this code to
a rented GPU with 24 red tests and no mechanism capable of saying so.

This is `ORCHESTRATOR_CANARY_SPEC.md` §3 — *assertion liveness* — described by its absence.

---

## 1. Isolation experiment: `clean_cohort.py` v2 is exonerated

Logical argument (weak): none of the six failing files imports `clean_cohort`; every traceback
terminates in `real_data_prep.py`, `esm2.py`, or an audit script.

Experiment (decisive):

```
git stash push -- scripts/clean_cohort.py            # restore v1
python -m pytest tests/unit/test_core.py tests/unit/test_correctness_harness.py \
  tests/unit/test_esm2_llr.py tests/unit/test_lovd_annotation_reaches_training_matrix.py \
  tests/unit/test_run17_audit_persplit.py tests/unit/test_run17_fullflag_smoke.py -q
→ 24 failed, 226 passed
git stash pop
```

Identical failure set under v1. **`clean_cohort.py` v2 caused none of the 24.**

---

## 2. Cluster A (12) — the test verdict depends on untracked filesystem state

```python
# real_data_prep.py:65-74
def _protein_coord_source_present(cache_path: Path, am_path: object) -> bool:
    """True iff a protein-coord SOURCE is available ... Stub mode (no source) is a
    valid path -- unit tests and boxes without the 613 MB TSV -- and must never raise."""
    if am_path is not None and Path(str(am_path)).exists():
        return True
    return Path(str(cache_path)).exists()          # ← the defect

# real_data_prep.py:923
if _protein_coord_source_present(pc.cache_path, _am_tsv):
    _coord_cov = _assert_protein_coord_coverage(df, ac.min_protein_coord_coverage)   # raises
```

**Established.** The failing tests construct `AnnotationConfig()` with no AlphaMissense path, so
`_am_tsv` is `None`. The gate can only return `True` through `Path(str(pc.cache_path)).exists()` —
the connector's *default* cache path, which exists on this machine. Fixtures of 2 and 5 rows then
present `is_missense ≥ 1` with `protein_pos` all-NA, coverage computes to `0.0000 < 0.5`, and the
gate raises. Its own docstring says stub mode "must never raise".

**Consequence.** These twelve tests are **green on a clean machine and red on this one**. A test
suite whose verdict is a function of which data files happen to sit on the box is not a test suite.

**Classification.** Doctrine §3.1: an **uninventoried member of the trusted base**. Identical in
species to the NT stale-cache bug (`INCIDENT_2026-07-08`, §3), where an untracked
`modeling_esm.py` in `~/.cache/huggingface/modules/` silently satisfied a dependency and made
"works on my machine" literally true and generally false.

**The principled fix.** *Source presence is a property of the declared configuration, not of the
filesystem.* A cache built by a previous run against a previous cohort is not a source wired into
this run. The gate should fire iff the caller explicitly wired a protein-coord source; existence is
then checked only for the path that was explicitly named. Note the error message already concedes
the point: it warns the index may be "stale or mismatched for THIS cohort/box."

**To confirm the mechanism (not yet run):**
```powershell
Test-Path data\external\alphamissense\alphamissense_protein_index.parquet
Select-String -Path src\genomic_variant_classifier\data\real_data_prep.py -Pattern '_am_tsv\s*=|pc\s*=' -Context 2,4
Select-String -Path src\genomic_variant_classifier\data\*.py -Pattern 'class ProteinCoord' -Context 0,20
```

---

## 3. Cluster B (6) — the fixture constructs an object whose `__init__` never ran

```python
# esm2.py:571  (inside __init__)
self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
# esm2.py:643
base = Path(self.cache_path).parent if self.cache_path else Path("data/raw/cache")
#   → AttributeError: 'ESM2Connector' object has no attribute 'cache_path'
# reached from esm2.py:947  annotate_llr → self._score_cache_load() → self._score_cache_path()
```

**Established.** `__init__` sets the attribute unconditionally, so an instance lacking it never ran
`__init__`. Line 643 already guards `if self.cache_path`, i.e. the author anticipated *falsy*, never
*absent*. Production is unaffected: ESM-2 scored 176/181 variants on the VM on 2026-07-06.

**Hypothesis (unverified).** The `conn` fixture uses `ESM2Connector.__new__(...)`, or monkeypatches
`__init__`. The failure appeared when `annotate_llr` acquired a score cache (`_score_cache_load`) that
the fixture's construction path never anticipated.

**Not prescribing `getattr(self, "cache_path", None)`** — that would paper over a fixture defect that
has not been read. Required: `tests/unit/test_esm2_llr.py`.

---

## 4. Clusters C (1) and D (5) — 91→97 feature fallout

**C.** `tests/unit/test_correctness_harness.py:162` asserts stage-5 findings are a subset of
`KNOWN_ZERO_DEFAULT`. Six new features fall outside it:
`cosmic_recurrence, cosmic_sig_tier, genomiclm_delta_norm, genomiclm_llr,
kegg_disease_pathway_flag, kegg_pathway_count`.
The allowlist and `build_reference_slice()` predate the connectors.

**D.** `test_run17_audit_persplit.py` (3) and `test_run17_fullflag_smoke.py` (2). The split fixtures
`_write` / `_write_splits` never emit the new columns; `audit_smoke_feature_population.py --run17`
grades `genomiclm_delta_norm` and `kegg_pathway_count` as **FAIL-severity when ABSENT**, so the audit
returns 1 and the tests assert 0.

Both are stale-fixture problems, not production defects. Both are trivially fixable. Both have been
red since the connectors landed, which is the point.

---

## 5. Adjacent bugs found while reading

**5.1 The retention guidance is inverted.** `real_data_prep.py:388` and `:1684` advise
*"Lower min_review_tier or increase dataset size"* when a split is missing a class. The filter is
`df[df["review_tier"] <= self.config.min_review_tier]`, so **lowering** it keeps *fewer* rows.
Recovering a missing class requires **raising** it. The message sends the operator the wrong way.

**5.2 A latent `TypeError` on the metadata path.** `real_data_prep.py:474-481`:
```python
df["ReviewStatus"].str.lower().map(lambda s: next((v for k, v in REVIEW_STATUS_TIER.items() if k in s), 5))
```
`augment_reviewstatus.py:64` writes `""` for join misses, so no NaN reaches this today. Source
`metadata.review_status` contains real nulls. **If Fix (a) of the deletion incident lands without
normalisation, `k in s` on `NaN` raises.** Blocker for the source decision, recorded here.

**5.3 Duplicate test basename.** `tests/test_clean_cohort.py` (added `2f6dcde`) and
`tests/unit/test_clean_cohort.py` (pre-existing) share a basename. Collection currently succeeds, but
this is fragile under pytest's default import mode, and two suites now test one module. A pre-existing
`tests/unit/test_cohort_guard.py` and `test_cohort_guard_resilience.py` also exist and were **not read**
before the v2 rewrite. They pass against v2, but that is luck, not method. Consolidate after reading.

---

## 6. Fix plan — ordered, gated

1. **Read before writing.** `tests/unit/test_clean_cohort.py`, `tests/unit/test_cohort_guard.py`,
   `tests/unit/test_cohort_guard_resilience.py`, `tests/conftest.py`, `tests/unit/test_esm2_llr.py`.
   No further guard modules until these are read; a "cohort guard" already exists under that name.
2. **Cluster A** — make source presence a function of configuration, not filesystem. Complete
   rewritten function with negative tests: *a stale cache on disk must not make a stub-mode run raise.*
3. **Cluster B** — fix the fixture (or the constructor contract), whichever the fixture reveals.
4. **Clusters C, D** — update `KNOWN_ZERO_DEFAULT`, `build_reference_slice()`, and the split fixtures
   to the 97-feature contract. Derive the expected feature set from one source of truth rather than
   restating it in three places.
5. **Close the meta-gap.** Add a test gate:
   - `pytest -q` at commit time (pre-commit hook or CI), so guard-rot is caught at PR time;
   - a fast subset in `vm_bootstrap_run.sh` Phase E, before any paid compute. The full suite is
     365 s — too slow for a boot gate, fast enough for a commit gate.
6. **Fix 5.1** (inverted guidance) and **5.2** (NaN normalisation) before the deletion-incident source
   decision.
7. **Consolidate 5.3.**

---

## 7. Required inputs (none of these can be inferred from a traceback)

| # | file / command | answers |
|---|---|---|
| 1 | `scripts/augment_reviewstatus.py` (4,509 B) | §3 of `INCIDENT_2026-07-08` — the deletion join key. **Requested three times.** |
| 2 | `tests/unit/test_esm2_llr.py` | Cluster B fixture |
| 3 | `tests/unit/test_clean_cohort.py`, `tests/conftest.py`, `tests/unit/test_cohort_guard*.py` | §5.3 consolidation; prevents a second duplicated module |
| 4 | `git log -1 --format='%h %ci %s' f24bfc6` | Whether Run 14 predates `ReviewStatus`, hence whether its metrics are comparable to Runs 15–17 |
| 5 | `Select-String -Path outputs\**\*.log -Pattern 'Review tier filter'` | **Ground truth**: did the tier filter actually execute in each run? Supersedes all inference. |

---

## 8. What this costs, and what it bought

Nothing here endangers data. The cohort is intact at MD5 `7C5E107C…`, verified by the very guard
shipped in `2f6dcde`. No run consumed a regression.

What it revealed is that the project's assertions have been rotting unobserved: twelve tests whose
outcome depends on which files sit on the developer's disk, six that fail on a construction path the
production code never takes, six left behind by a feature expansion, and a headline count — *596 tests
pass* — that has been wrong for long enough that nobody knows when it stopped being true.

The doctrine's answer is not more tests. It is a **gate that runs them**, and a canary that tries to
trip each guard on synthetic input at startup. A guard nobody exercises is a comment.
