#!/usr/bin/env python
"""audit_training_entrypoint.py (2026-07-10)

Map the training orchestration so the re-baseline (item 4) can be built to reuse the real entry
point rather than a guessed one. Read-only. ASCII-safe on every printed line (scanned files
contain non-ASCII, e.g. the subset symbol, which crashes a Windows cp1252 console); stdout is also
forced to replace un-encodable bytes.

Reports, for the training-related scripts and modules:
  - which files define a training entry point (main, argparse, if __name__),
  - the 13-model roster and where models are constructed,
  - meta-learner / stacking / ensemble fitting sites,
  - probability calibration sites,
  - where oof_predictions / test_predictions are written (the artifacts the conformal layer needs),
  - where the split is invoked (so split_protocol_v2 can be wired in),
  - the cohort/parquet inputs loaded.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(".")


def _ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def line(c="-", n=78):
    print(c * n)


# Candidate training files: scripts/ + src training modules. We DISCOVER them by name pattern
# and by content, rather than hardcoding, so nothing is missed.
NAME_HINTS = re.compile(
    r"(run_phase|train|baseline|ensemble|stack|pipeline|orchestr|run_stage|run15|rebaseline|"
    r"re_baseline|phase2|phase3|phase4)", re.IGNORECASE)

CONTENT_PATTERNS = {
    "ENTRYPOINT": re.compile(r'if __name__ ==|def main\(|argparse|add_argument|ArgumentParser'),
    "MODEL_CTOR": re.compile(
        r"RandomForest|XGB|LGBM|LightGBM|CatBoost|SVC|LogisticRegression|GradientBoosting|"
        r"tabular_nn|cnn_1d|\bkan\b|mc_dropout|deep_ensemble|GNN|gnn", re.IGNORECASE),
    "META_STACK": re.compile(r"meta_learner|stack|StackingClassifier|blend|ensemble_prob|oof", re.IGNORECASE),
    "CALIBRATION": re.compile(r"CalibratedClassifier|isotonic|sigmoid|platt|calibrat", re.IGNORECASE),
    "ARTIFACT_WRITE": re.compile(r"to_parquet|oof_predictions|test_predictions|predictions\.parquet"),
    "SPLIT_CALL": re.compile(r"_gene_aware_split|gene_stratified_split|GroupShuffleSplit|train_test_split|split_protocol"),
    "COHORT_LOAD": re.compile(r"read_parquet|clinvar_grch38|pathfix|cohort|\.parquet"),
}
EXCLUDE = (".venv312", ".venv", ".git", "site-packages", "__pycache__", "node_modules")


def discover_files():
    out = []
    for base in ("scripts", "src"):
        bp = ROOT / base
        if not bp.exists():
            continue
        for p in bp.rglob("*.py"):
            if any(e in str(p) for e in EXCLUDE):
                continue
            out.append(p)
    return sorted(out)


def main() -> int:
    print("=" * 78)
    print("TRAINING ENTRYPOINT AUDIT (map the pipeline for the re-baseline; reuse, do not guess)")
    print("=" * 78)
    files = discover_files()
    print(f"scanned {len(files)} python files under scripts/ and src/")
    line()

    # Rank files by how many training-relevant categories they hit, to find the orchestrators.
    scored = []
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        cats = {name: [] for name in CONTENT_PATTERNS}
        for i, ln in enumerate(txt.splitlines(), 1):
            for name, rx in CONTENT_PATTERNS.items():
                if rx.search(ln):
                    cats[name].append((i, ln.strip()))
        hitcats = [c for c, v in cats.items() if v]
        namehit = bool(NAME_HINTS.search(p.name))
        if hitcats and (namehit or len(hitcats) >= 3):
            scored.append((len(hitcats), p, cats))
    scored.sort(key=lambda t: -t[0])

    print("TOP TRAINING-RELATED FILES (by category coverage):")
    for ncat, p, cats in scored[:12]:
        rel = p.relative_to(ROOT)
        present = [c for c, v in cats.items() if v]
        print(_ascii_safe(f"  [{ncat} cats] {rel}"))
        print(_ascii_safe(f"       {', '.join(present)}"))
    line("=")

    # For the very top files, print the KEY lines per category (capped).
    for ncat, p, cats in scored[:6]:
        rel = p.relative_to(ROOT)
        print(_ascii_safe(f"=== {rel} ==="))
        for name in ["ENTRYPOINT", "SPLIT_CALL", "MODEL_CTOR", "META_STACK", "CALIBRATION",
                     "ARTIFACT_WRITE", "COHORT_LOAD"]:
            v = cats[name]
            if not v:
                continue
            print(_ascii_safe(f"  [{name}] {len(v)} hit(s):"))
            for i, ln in v[:6]:
                print(_ascii_safe(f"    L{i}: {ln[:140]}"))
            if len(v) > 6:
                print(f"    ... {len(v) - 6} more")
        line()
    print("AUDIT COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
