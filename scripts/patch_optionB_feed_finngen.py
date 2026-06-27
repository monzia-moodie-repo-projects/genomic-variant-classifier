#!/usr/bin/env python3
r"""patch_optionB_feed_finngen.py

Option B for the FinnGen R12+R13 dual-release harness: FEED finngen in the synthetic
reference slice (so the zero-audit actively checks it) instead of allowlisting it.

This restores the project's PRE-EXISTING decision: test_allowlist_unchanged_size
(committed earlier) asserts the allowlist must NOT grow, with the comment "Option B
feeds the fixture, does not allowlist". The Stage-3 commit (752335c) wrongly took
Option A (allowlist 27->29); this reverts that AND removes the two R12 AF columns
(now fed), landing the allowlist at 25.

Edits across TWO files:

  correctness_harness.py:
  (1) build_reference_slice: add the 6 finngen columns (R12+R13) with nonzero synthetic
      values. VERIFIED direct passthrough -- engineer_features does
      feats[col]=df.get(col, default) (lines 618-630); feeding the output names
      populates them (empirically nonzero_rate=1.00). AF ~ uniform(0,0.5);
      enrichment ~ uniform(0.5,5).
  (2) KNOWN_ZERO_DEFAULT: remove finngen_af_fin + finngen_af_nfsee (R12; now fed) AND
      the two R13 AF names added by 752335c. Net 29 -> 25.

  test_harness_fixture_omim_molecular.py:
  (3) test_allowlist_unchanged_size: 27 -> 25.

ANCHOR-BASED, IDEMPOTENT, LF.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

HARNESS = Path("src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py")
TEST = Path("tests/unit/test_harness_fixture_omim_molecular.py")

# ---- Edit 1: feed 6 finngen columns in build_reference_slice (anchor: gnn_score line) ----
FEED_OLD = '''        "alphafold_plddt": rng.uniform(20, 95, n), "gnn_score": rng.uniform(0.1, 0.9, n),'''
FEED_NEW = '''        "alphafold_plddt": rng.uniform(20, 95, n), "gnn_score": rng.uniform(0.1, 0.9, n),
        # FinnGen R12 + R13 population AF -- FED (Option B): zero-audit actively checks
        # these (direct passthrough via df.get in engineer_features). NOT allowlisted.
        "finngen_af_fin": rng.uniform(0, 0.5, n), "finngen_af_nfsee": rng.uniform(0, 0.5, n),
        "finngen_enrichment": rng.uniform(0.5, 5, n),
        "finngen_r13_af_fin": rng.uniform(0, 0.5, n), "finngen_r13_af_nfsee": rng.uniform(0, 0.5, n),
        "finngen_r13_enrichment": rng.uniform(0.5, 5, n),'''

# ---- Edit 2a: strip R12 finngen pair from the cadd_high line ----
ZERO_R12_OLD = '''    "cadd_high", "finngen_af_fin", "finngen_af_nfsee", "gene_is_constrained",'''
ZERO_R12_NEW = '''    "cadd_high", "gene_is_constrained",'''

# ---- Edit 2b: remove the R13 block added by 752335c ----
ZERO_R13_OLD = '''    # R13 dual-release: same fixture-zero status as the R12 AF twins above
    # (build_reference_slice synthesizes no finngen_* column). enrichment omitted
    # (defaults to 1.0, never zeros -- like finngen_enrichment).
    "finngen_r13_af_fin", "finngen_r13_af_nfsee",
'''
ZERO_R13_NEW = ''

# ---- Edit 3: test 27 -> 25 ----
TEST_OLD = '''def test_allowlist_unchanged_size():
    # Option B must NOT grow the allowlist (that would be Option A).
    assert len(KNOWN_ZERO_DEFAULT) == 27, (
        f"KNOWN_ZERO_DEFAULT must stay 27 (Option B feeds the fixture, does not allowlist); "
        f"got {len(KNOWN_ZERO_DEFAULT)}.")'''
TEST_NEW = '''def test_allowlist_unchanged_size():
    # Option B feeds finngen in the fixture; the two R12 AF columns
    # (finngen_af_fin, finngen_af_nfsee) were therefore REMOVED from the allowlist
    # (27 -> 25). R13 AF is also fed, never allowlisted. enrichment never zeros.
    assert len(KNOWN_ZERO_DEFAULT) == 25, (
        f"KNOWN_ZERO_DEFAULT must be 25 (Option B feeds finngen R12+R13 AF, does not "
        f"allowlist them); got {len(KNOWN_ZERO_DEFAULT)}.")'''

H_MARK = '"finngen_af_fin": rng.uniform'
T_MARK = "must be 25 (Option B feeds finngen"


def _validate(path, edits, mark, label):
    src = path.read_text(encoding="utf-8")
    if mark in src:
        return ("idem", src)
    problems = []
    for name, old, _new in edits:
        c = src.count(old)
        if c != 1:
            problems.append(f"  {label}/{name}: anchor occurs {c}x (need 1)")
    if problems:
        print(f"FAIL: {label}:\n" + "\n".join(problems)); return ("fail", src)
    return ("ok", src)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not HARNESS.exists() or not TEST.exists():
        print("FAIL: target file(s) missing."); return 2

    harness_edits = [
        ("feed-finngen", FEED_OLD, FEED_NEW),
        ("zero-r12", ZERO_R12_OLD, ZERO_R12_NEW),
        ("zero-r13", ZERO_R13_OLD, ZERO_R13_NEW),
    ]
    test_edits = [("allowlist-25", TEST_OLD, TEST_NEW)]

    sh, srcH = _validate(HARNESS, harness_edits, H_MARK, "harness")
    st, srcT = _validate(TEST, test_edits, T_MARK, "test")
    if sh == "fail" or st == "fail":
        print("RESULT: FAIL (anchor validation)"); return 3
    if ns.check:
        print(f"CHECK: harness={sh}, test={st} (anchors found / idempotent)."); print("RESULT: PASS (check)"); return 0

    if sh == "ok":
        bH = HARNESS.with_suffix(".py.pre_optB.bak")
        if not bH.exists(): bH.write_text(srcH, encoding="utf-8", newline="")
        newH = srcH
        for _n, old, repl in harness_edits:
            newH = newH.replace(old, repl, 1)
        HARNESS.write_text(newH, encoding="utf-8", newline="\n")
    if st == "ok":
        bT = TEST.with_suffix(".py.pre_optB.bak")
        if not bT.exists(): bT.write_text(srcT, encoding="utf-8", newline="")
        newT = srcT.replace(TEST_OLD, TEST_NEW, 1)
        TEST.write_text(newT, encoding="utf-8", newline="\n")

    aH = HARNESS.read_text(encoding="utf-8")
    aT = TEST.read_text(encoding="utf-8")
    checks = {
        "harness feeds finngen_af_fin": '"finngen_af_fin": rng.uniform' in aH,
        "harness feeds r13_af_fin": '"finngen_r13_af_fin": rng.uniform' in aH,
        "harness feeds enrichment": '"finngen_enrichment": rng.uniform' in aH,
        "R12 stripped from allowlist line": '"cadd_high", "gene_is_constrained",' in aH,
        "R13 block removed from allowlist": "R13 dual-release: same fixture-zero" not in aH,
        "test expects 25": "== 25" in aT,
        "test no longer expects 27": "== 27" not in aT,
    }
    try:
        ast.parse(aH); checks["harness compiles"] = True
    except SyntaxError as e:
        checks["harness compiles"] = False; print("  HARNESS SYNTAX:", e)
    try:
        ast.parse(aT); checks["test compiles"] = True
    except SyntaxError as e:
        checks["test compiles"] = False; print("  TEST SYNTAX:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
