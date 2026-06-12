#!/usr/bin/env python3
"""install_docs_incident_ci_esm2.py -- record the 2026-06-12 CI/ESM-2-Hub-flake
incident: create docs/incidents/INCIDENT_2026-06-12_ci-esm2-hub-flake.md, append
docs/CHANGELOG.md and docs/ROADMAP.md. Idempotent (marker-guarded), backup-first,
no-BOM, newline-preserving, ASCII. After running, regenerate the docx:
python scripts\\make_roadmap_docx.py . Author: Monzia Moodie."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

CHANGELOG = Path("docs/CHANGELOG.md")
ROADMAP = Path("docs/ROADMAP.md")
INCIDENT = Path("docs/incidents/INCIDENT_2026-06-12_ci-esm2-hub-flake.md")

CHANGELOG_MARKER = "<!-- docs-close: ci-esm2-hub-flake 2026-06-12 -->"
ROADMAP_MARKER = "## ROADMAP delta -- 2026-06-12 (CI ESM-2 Hub flake resolved)"

INCIDENT_DOC = """# INCIDENT 2026-06-12 -- CI red on flaky ESM-2 HuggingFace Hub download

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
"""

CHANGELOG_ENTRY = """<!-- docs-close: ci-esm2-hub-flake 2026-06-12 -->
## 2026-06-12 -- CI red resolved: flaky ESM-2 HuggingFace Hub download

### Fixed
- CI was red on runs #316 (docs-only) / #317 while local was green:
  test_llr_long_protein_scores_finite_without_oom loads the real ESM-2 8M from HF Hub;
  CI runners (no cache, rate-limited 429) erred, local (cached weights) passed.
  fee2e63 wraps the live load in try/except OSError -> pytest.skip; the test still runs
  fully wherever the model loads and skips only on HF-offline.

### Changed
- .github/workflows/ci.yml: HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 on the unit-test
  step (CI never reaches HF Hub -> 429 impossible); pytest -x -> --maxfail=5 (a break
  surfaces several failures instead of halting at and hiding everything after the first).

### Verification
- Reproduced offline (empty HF_HOME): test SKIPS, not errors. With weights: both pass.
- Whole offline suite: 898 passed / 2 skipped / exit 0 -- no other unguarded live-loader.

### Learned
- Local-suite-green is NOT a proxy for CI-green where a test loads an ESM-2 model: the
  local cache hides a hard network dependency. New gate: run the suite under an empty
  offline HF cache before trusting green.

### Commits
- fee2e63 (test skip-guard, already on origin/main); this close (ci.yml hardening + docs).
- See docs/incidents/INCIDENT_2026-06-12_ci-esm2-hub-flake.md.
"""

ROADMAP_DELTA = """## ROADMAP delta -- 2026-06-12 (CI ESM-2 Hub flake resolved)

### Done
- [x] CI restored to green. test_llr_long_protein_scores_finite_without_oom was loading
  the real ESM-2 8M from HF Hub; CI (no cache, 429) flaked red while local (cached) passed.
  fee2e63 skip-guards the live load; ci.yml now forces HF offline and uses --maxfail=5.
  See docs/incidents/INCIDENT_2026-06-12_ci-esm2-hub-flake.md.

### Standing disciplines -- ADDITIONS
- Offline-suite gate: before trusting "suite green", run tests/unit under an empty offline
  HF cache (HF_HOME=<empty>, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1). Anything that ERRORS
  (vs passes/skips) is a network-coupled test to guard. Local-green != CI-green for any test
  that loads an ESM-2 model (the local cache hides the dependency).
- CI surfaces failures broadly: pytest -x replaced by --maxfail=5 so a break is not reported
  as a single isolated failure with the rest of the suite hidden.

### Note (unchanged, benign)
- 0xc0000139 torch_scatter/torch_sparse dumps during collection are the known-benign
  importorskip path (missing PyG C-extensions on the CPU box); suite exit stays 0.
"""


def _read(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write(path, text):
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _nl(raw):
    return "\r\n" if "\r\n" in raw else "\n"


def _append(path, marker, body):
    if not path.exists():
        return f"MISSING: {path} (not appended)"
    raw = _read(path)
    if marker in raw:
        return f"already present, skipped: {path.name}"
    nl = _nl(raw)
    text = raw.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    text = text + "\n" + body.rstrip("\n") + "\n"
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    _write(path, text.replace("\n", nl))
    return f"appended: {path.name}"


def _create(path, body):
    if path.exists():
        return f"already present, skipped: {path.name}"
    nl = "\r\n"
    if ROADMAP.exists():
        nl = _nl(_read(ROADMAP))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, body.replace("\n", nl))
    return f"created: {path.name}"


def main():
    print(_create(INCIDENT, INCIDENT_DOC))
    print(_append(CHANGELOG, CHANGELOG_MARKER, CHANGELOG_ENTRY))
    print(_append(ROADMAP, ROADMAP_MARKER, ROADMAP_DELTA))
    print("NEXT: regenerate the docx (run make_roadmap_docx.py in scripts/), then review git diff.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
