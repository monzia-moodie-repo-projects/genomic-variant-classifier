#!/usr/bin/env python3
r"""patch_harness_fixture_omim_molecular.py

Add omim_n_diseases_molecular as a populated INPUT column to build_reference_slice()
in src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py, RESPECTING
the real-world invariant molecular_count <= total_disease_count.

WHY (Option B, not A): build_reference_slice's docstring contract is "fully-populated...
the only ~all-zero columns are exactly KNOWN_ZERO_DEFAULT." omim_n_diseases_molecular is
feature #88 (added this session), consumed by engineer_features as pure pass-through
(df.get("omim_n_diseases_molecular", 0)) exactly like its sibling omim_n_diseases which the
fixture DOES feed. The fixture forgot the new twin -> it falls to zero default -> harness
flags it. Connector is LIVE (71.68% real cohort), so allowlisting (Option A) would wrongly
mark a live feature expected-zero. Correct fix: feed it, keeping KNOWN_ZERO_DEFAULT at 27.

INVARIANT (verified by test): molecular <= total. To enforce, we HOIST omim_n_diseases to a
local `omim_nd` before the DataFrame literal, then reference it in BOTH the n_diseases entry
and the new molecular entry as np.minimum(omim_nd, rng.integers(0,10,n)). This guarantees
molecular <= total for every row (independent draws do NOT, as the test proved).

TWO coordinated edits, both anchor-based + idempotent. LF line endings.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py")

# Edit 1: hoist a local just before the `return pd.DataFrame({` line.
RET_ANCHOR = "    return pd.DataFrame({"
HOIST = (
    "    # feature #88: molecular-basis disease count, bounded by total (molecular <= total).\n"
    "    omim_nd = rng.integers(0, 10, n)\n"
)

# Edit 2: the inline omim_n_diseases entry -> use the hoisted local + add the molecular twin.
OLD_OMIM = '        "omim_n_diseases": rng.integers(0, 10, n), "omim_is_autosomal_dominant": rng.integers(0, 2, n),'
NEW_OMIM = (
    '        "omim_n_diseases": omim_nd, "omim_is_autosomal_dominant": rng.integers(0, 2, n),\n'
    '        "omim_n_diseases_molecular": np.minimum(omim_nd, rng.integers(0, 10, n)),  # feature #88; molecular<=total; keeps fixture "fully-populated"'
)

MARKER = '"omim_n_diseases_molecular":'
HOIST_MARKER = "omim_nd = rng.integers(0, 10, n)"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src and HOIST_MARKER in src:
        print("OK (idempotent): fixture already feeds omim_n_diseases_molecular (bounded)."); return 0

    # Validate both anchors exist exactly once.
    cr = src.count(RET_ANCHOR); co = src.count(OLD_OMIM)
    if cr != 1:
        print(f"FAIL: return-anchor occurs {cr}x (need 1)."); return 3
    if co != 1:
        print(f"FAIL: omim_n_diseases entry occurs {co}x (need 1)."); return 4

    if ns.check:
        print("CHECK: both anchors found once (return literal + omim_n_diseases entry).")
        print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".py.pre_omim_molecular.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")

    # Apply Edit 1 (hoist) then Edit 2 (entry rewrite). Order: hoist first (insert before return),
    # then replace the omim entry. Both target distinct, unique anchors.
    new = src.replace(RET_ANCHOR, HOIST + RET_ANCHOR, 1)
    new = new.replace(OLD_OMIM, NEW_OMIM, 1)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    ok_marker = MARKER in after
    ok_hoist = HOIST_MARKER in after
    ok_uses_local = '"omim_n_diseases": omim_nd' in after
    ok_min = "np.minimum(omim_nd, rng.integers(0, 10, n))" in after
    try:
        ast.parse(after); compiles = True
    except SyntaxError as e:
        compiles = False; print("  SYNTAX ERROR:", e)
    print(f"  {'OK' if ok_hoist else 'MISSING'}  hoisted local omim_nd")
    print(f"  {'OK' if ok_uses_local else 'MISSING'}  omim_n_diseases entry uses local omim_nd")
    print(f"  {'OK' if ok_marker else 'MISSING'}  molecular twin fed")
    print(f"  {'OK' if ok_min else 'MISSING'}  molecular bounded by np.minimum(omim_nd, ...) (invariant molecular<=total)")
    print(f"  {'OK' if compiles else 'FAIL'}  module still compiles")
    ok = ok_marker and ok_hoist and ok_uses_local and ok_min and compiles
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
