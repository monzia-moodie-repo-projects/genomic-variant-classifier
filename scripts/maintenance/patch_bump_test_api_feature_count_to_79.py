#!/usr/bin/env python3
"""
patch_bump_test_api_feature_count_to_79.py
==========================================
Bring tests/unit/test_api.py into line with the 79-feature contract
(reactome_pathway_count was appended to TABULAR_FEATURES / INFERENCE_FEATURE_COLUMNS).

Mechanism (confirmed from the session that built /info):
  * /info["n_features"]    is a HARDCODED literal carried by the _make_pipeline
    mock fixture (historically bumped 64 -> 72 -> 73 on prior feature additions).
  * /info["feature_names"] tracks LIVE INFERENCE_FEATURE_COLUMNS (now 79).
That is why, in the failing run, `n_features == 78` PASSED (stale mock literal)
while `len(feature_names) == 78` FAILED at 79. Syncing the mock literal to 79
restores n_features == feature_names == 79.

Three regex edits (all whitespace-tolerant, counts reported):
  A. mock literal     n_features = 78   ->  79
  B. assertion        body["n_features"]        == 78   ->  == 79
  C. assertion        len(body["feature_names"]) == 78  ->  == 79

Safety contract (cannot silently do the wrong thing):
  * If A, B, C all match ZERO times  -> already applied -> SKIP (idempotent).
  * Else REQUIRE A>=1 AND B>=1 AND C>=1. If any is 0 while others matched,
    ABORT and print the per-anchor counts -- this means the structure drifted
    (e.g. n_features literal now lives in api/main.py, not the fixture), and
    must be reviewed by hand rather than half-patched.
  * .bak backup, AST-verify, BOM-free UTF-8.

Run from the repo root. Re-runnable.
"""
from __future__ import annotations

import ast
import datetime as _dt
import re
import sys
from pathlib import Path

TARGET = Path("tests/unit/test_api.py")

# A: the mock fixture's n_features literal (single '=', not the '==' assertions).
RE_LITERAL = re.compile(r"(\bn_features\s*=\s*)78\b")
# B: the n_features assertion.
RE_ASSERT_NF = re.compile(r'(body\["n_features"\]\s*==\s*)78\b')
# C: the feature_names length assertion.
RE_ASSERT_FN = re.compile(r'(len\(body\["feature_names"\]\)\s*==\s*)78\b')


def _show(label: str, rx: re.Pattern, text: str) -> int:
    hits = list(rx.finditer(text))
    for m in hits:
        line = text.count("\n", 0, m.start()) + 1
        print(f"    {label} match @ line {line}: ...{m.group(0)}...")
    return len(hits)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root.")
        sys.exit(2)
    original = TARGET.read_text(encoding="utf-8")

    a = _show("A literal", RE_LITERAL, original)
    b = _show("B assert n_features", RE_ASSERT_NF, original)
    c = _show("C assert feature_names", RE_ASSERT_FN, original)
    print(f"  counts: A(literal)={a}  B(n_features==)={b}  C(feature_names==)={c}")

    if a == 0 and b == 0 and c == 0:
        # Sanity: make sure that's because it's already at 79, not because the
        # anchors vanished entirely.
        if re.search(r"\bn_features\s*=\s*79\b", original) or \
           re.search(r'body\["n_features"\]\s*==\s*79\b', original):
            print("  SKIP  test_api.py already at 79. No changes.")
            return
        print("  ABORT: none of the 78-anchors found AND no 79 present -- "
              "structure unrecognized. Manual review required.")
        sys.exit(2)

    if not (a >= 1 and b >= 1 and c >= 1):
        print("  ABORT: partial match -- the three anchors did not all appear.")
        print("         This usually means the n_features literal moved out of")
        print("         the fixture (e.g. into src/.../api/main.py). Patch was")
        print("         NOT applied; review the counts above by hand.")
        sys.exit(2)

    text = RE_LITERAL.sub(r"\g<1>79", original)
    text = RE_ASSERT_NF.sub(r"\g<1>79", text)
    text = RE_ASSERT_FN.sub(r"\g<1>79", text)

    if text == original:
        print("  ABORT: no substitutions made despite matches (unexpected).")
        sys.exit(3)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}")
        sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"  OK    bumped A={a}, B={b}, C={c} occurrence(s): 78 -> 79")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
