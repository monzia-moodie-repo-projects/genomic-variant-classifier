#!/usr/bin/env python3
r"""patch_variant_ensemble_finngen_r13.py

Stage 2 of the FinnGen R12+R13 dual-release experiment: extend the tabular feature
contract 88 -> 91 by adding the three R13 columns.

Three edits to src/genomic_variant_classifier/models/variant_ensemble.py:
  1. TABULAR_FEATURES: add finngen_r13_af_fin / finngen_r13_af_nfsee / finngen_r13_enrichment
     immediately AFTER the R12 trio (anchor: the three R12 lines at L252-254).
  2. EXPECTED_TABULAR_FEATURE_COUNT: 88 -> 91 (anchor: the literal at L157).
  3. Defaults block: add the three R13 defaults AFTER the R12 defaults (anchor: L616-618),
     mirroring R12 value semantics -- af_fin/af_nfsee = 0.0, enrichment = 1.0 (ratio).

The R13 connector (column_prefix="r13_") already emits exactly these names (Stage 1, ca76482).
INFERENCE_FEATURE_COLUMNS is derived as list(TABULAR_FEATURES) (test_feature_count_contract.py:38),
so it auto-tracks; no second list to edit. The contract test gates all four invariants
(len==91, inference len==91, inference==list(TABULAR_FEATURES), uniqueness).

ANCHOR-BASED, IDEMPOTENT, LF. Validates each anchor occurs exactly once before applying.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/models/variant_ensemble.py")
MARKER = "finngen_r13_af_fin"  # idempotency sentinel

# ---- Edit 1: TABULAR_FEATURES -- add R13 trio after R12 trio ----
FEAT_OLD = '''    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",'''
FEAT_NEW = '''    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
    "finngen_r13_af_fin",
    "finngen_r13_af_nfsee",
    "finngen_r13_enrichment",'''

# ---- Edit 2: count 88 -> 91 ----
COUNT_OLD = "EXPECTED_TABULAR_FEATURE_COUNT = 88"
COUNT_NEW = "EXPECTED_TABULAR_FEATURE_COUNT = 91"

# ---- Edit 3: defaults -- add R13 trio after R12 defaults ----
DEF_OLD = '''        ("finngen_af_fin", 0.0),
        ("finngen_af_nfsee", 0.0),
        ("finngen_enrichment", 1.0),'''
DEF_NEW = '''        ("finngen_af_fin", 0.0),
        ("finngen_af_nfsee", 0.0),
        ("finngen_enrichment", 1.0),
        ("finngen_r13_af_fin", 0.0),
        ("finngen_r13_af_nfsee", 0.0),
        ("finngen_r13_enrichment", 1.0),'''

EDITS = [
    ("TABULAR_FEATURES +R13 trio", FEAT_OLD, FEAT_NEW),
    ("count 88->91", COUNT_OLD, COUNT_NEW),
    ("defaults +R13 trio", DEF_OLD, DEF_NEW),
]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): finngen_r13 columns already present."); return 0

    problems = []
    for name, old, _new in EDITS:
        c = src.count(old)
        if c != 1:
            problems.append(f"  {name}: anchor occurs {c}x (need 1)")
    # collision guard: R13 names must NOT already exist anywhere
    for col in ("finngen_r13_af_fin", "finngen_r13_af_nfsee", "finngen_r13_enrichment"):
        if col in src:
            problems.append(f"  COLLISION: {col} already present pre-patch")
    if problems:
        print("FAIL: anchor validation:\n" + "\n".join(problems)); return 3
    if ns.check:
        print("CHECK: all 3 anchors found once; no R13 collision."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".py.pre_finngen_r13.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")

    new = src
    for name, old, repl in EDITS:
        new = new.replace(old, repl, 1)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = {
        "count is 91": "EXPECTED_TABULAR_FEATURE_COUNT = 91" in after,
        "R13 af_fin in features": '"finngen_r13_af_fin",' in after,
        "R13 af_nfsee in features": '"finngen_r13_af_nfsee",' in after,
        "R13 enrichment in features": '"finngen_r13_enrichment",' in after,
        "R13 af_fin default 0.0": '("finngen_r13_af_fin", 0.0),' in after,
        "R13 af_nfsee default 0.0": '("finngen_r13_af_nfsee", 0.0),' in after,
        "R13 enrichment default 1.0": '("finngen_r13_enrichment", 1.0),' in after,
        "R12 trio still present (not clobbered)": '"finngen_enrichment",' in after,
        "each R13 name appears exactly twice (feature + default)":
            all(after.count(c) == 2 for c in ("finngen_r13_af_fin", "finngen_r13_af_nfsee", "finngen_r13_enrichment")),
    }
    try:
        ast.parse(after); checks["compiles"] = True
    except SyntaxError as e:
        checks["compiles"] = False; print("  SYNTAX ERROR:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
