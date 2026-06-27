#!/usr/bin/env python3
r"""patch_harness_known_zero_finngen_r13.py

Stage 3 of the FinnGen R12+R13 dual-release experiment: extend the harness
KNOWN_ZERO_DEFAULT allowlist 27 -> 29 by adding the two R13 allele-frequency columns
(finngen_r13_af_fin, finngen_r13_af_nfsee), mirroring their R12 twins
(finngen_af_fin, finngen_af_nfsee) which are already in the set.

WHY allowlist (not feed in the fixture):
  build_reference_slice() does NOT synthesize ANY finngen_* column (verified: the
  column dict has no finngen key). So R12's two AF columns zero out in the synthetic
  slice and are allowlisted as fixture-zero (NOT a claim the connector is dead --
  the real FinnGenConnector produces live values; the comment block says the set is
  "empirically derived by running engineer_features on build_reference_slice").
  R13 is structurally identical: its two AF columns zero out in the same slice for
  the same reason, so they join the allowlist symmetrically with R12.
  finngen_r13_enrichment is NOT added -- it defaults to 1.0 (a ratio), never zeros,
  exactly like finngen_enrichment which is also absent from the set.

The test imports KNOWN_ZERO_DEFAULT from the harness (single source of truth), so
this one edit updates both the gate and the test.

ANCHOR-BASED, IDEMPOTENT, LF.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py")
MARKER = "finngen_r13_af_fin"

# Anchor: the line containing the R12 finngen AF columns (line 320). Add R13's two right after,
# on their own line, with a brief comment. Anchor must match exactly + once.
OLD = '''    "cadd_high", "finngen_af_fin", "finngen_af_nfsee", "gene_is_constrained",'''
NEW = '''    "cadd_high", "finngen_af_fin", "finngen_af_nfsee", "gene_is_constrained",
    # R13 dual-release: same fixture-zero status as the R12 AF twins above
    # (build_reference_slice synthesizes no finngen_* column). enrichment omitted
    # (defaults to 1.0, never zeros -- like finngen_enrichment).
    "finngen_r13_af_fin", "finngen_r13_af_nfsee",'''


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): finngen_r13 AF columns already in KNOWN_ZERO_DEFAULT."); return 0
    c = src.count(OLD)
    if c != 1:
        print(f"FAIL: anchor occurs {c}x (need 1)."); return 3
    # guard: R13 enrichment must NOT be added (it never zeros); ensure we're not about to add it
    if "finngen_r13_enrichment" in NEW:
        print("FAIL: patch would add finngen_r13_enrichment (must NOT -- defaults 1.0)."); return 4
    if ns.check:
        print("CHECK: anchor found once; enrichment correctly excluded."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".py.pre_known_zero_r13.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    new = src.replace(OLD, NEW, 1)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = {
        "R13 af_fin added": '"finngen_r13_af_fin"' in after,
        "R13 af_nfsee added": '"finngen_r13_af_nfsee"' in after,
        "R13 enrichment NOT added": "finngen_r13_enrichment" not in after,
        "R12 twins still present": '"finngen_af_fin", "finngen_af_nfsee"' in after,
        "compiles": False,  # set below
    }
    try:
        ast.parse(after); checks["compiles"] = True
    except SyntaxError as e:
        print("  SYNTAX ERROR:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
