# INCIDENT 2026-06-12 -- CI red on flaky ESM-2 HuggingFace Hub download

## Status
RESOLVED. Test-level fix fee2e63 (skip-guard) + CI workflow hardening (HF offline
env; pytest -x -> --maxfail=5).

## Summary
GitHub Actions CI was red on runs #316 (35b59ef, a DOCS-ONLY commit) and #317
(985d42b, preflight) while the local unit suite was green. A docs-only commit cannot
break a test -- that was the tell: a flaky live-network dependency, not a regression.

## Symptom
tests/unit/test_esm2_llr_windowing.py::test_llr_long_protein_scores_finite_without_oom
failed in CI with 429 Too Many Requests for
huggingface.co/facebook/esm2_t6_8M_UR50D/resolve/main/config.json -> OSError
("We couldn't connect ... not a directory containing config.json"). CI runs
pytest tests/unit/ -x, so -x halted at this first failure (~58%), reported 499 passed,
and hid every test after it.

## Root cause
The test calls conn.annotate_llr(df), which lazily loads the real ESM-2 8M weights
from HF Hub on first use. CI runners have no local cache and share IPs that HF
rate-limits (429). Locally the 8M weights are cached, so the load hits the cache and
the test passes -- the local green MASKED the CI red. The flake was intermittent:
runs #311-#315 won the download lottery (green); #316/#317 hit the 429 (red).

## Why it went unnoticed
- Local-suite-green was treated as a proxy for CI-green; it cannot see model-cache
  availability.
- CI's pytest -x reports only the first failure and silently skips the remainder, so
  the red looked like one isolated failure with no signal about the rest of the suite.

## Fix
1. fee2e63 (test): wrap the single live-load in try/except OSError -> pytest.skip, so
   the test runs fully wherever the model loads (local; Vast.ai where 650M is cached)
   and skips only on the HF-offline/429 condition. Windowing index math stays covered
   offline by the sibling test_windowed_logit_row_reads_correct_residue (mocked model).
2. CI workflow (this close): add HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 to the
   unit-test step so CI never reaches HF Hub (429 becomes structurally impossible;
   model-needing tests fail fast -> skip). Replace pytest -x with --maxfail=5 so a
   future break surfaces several failures instead of halting at the first.

## Verification
- Reproduced the CI condition locally (empty HF_HOME + HF_HUB_OFFLINE): the test now
  SKIPS with the exact OSError message instead of erroring.
- With weights present (default cache, online): both windowing tests PASS -- local
  coverage intact.
- Whole-suite offline (empty HF_HOME + offline): 898 passed / 2 skipped / exit 0,
  proving no OTHER unguarded live-loader exists past the old -x cutoff (the annotate
  calls in test_esm2_llr.py / test_esm2_batched_equivalence.py are mocked or hit the
  0.0-default path).

## Prevention (new standing gates)
- Before relying on "suite green", run the unit suite under an empty offline HF cache:
  HF_HOME=<empty> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest tests/unit/ -q .
  Anything that ERRORS (vs passes/skips) is a network-coupled test to guard.
- CI no longer uses -x (uses --maxfail=5): a break surfaces broadly, not one-at-a-time.
- The 0xc0000139 torch_scatter/torch_sparse dumps during collection remain the
  known-benign importorskip path (missing PyG C-extensions on the CPU box); exit stays 0.
