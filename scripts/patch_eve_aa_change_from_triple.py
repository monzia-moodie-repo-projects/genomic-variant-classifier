#!/usr/bin/env python3
r"""patch_eve_aa_change_from_triple.py

THE final EVE silent-zero fix. EVEConnector._annotate built its variant-side
lookup key (`_aa_change`) ONLY from the `protein_change` (HGVSp) column:

    protein_change = result.get("protein_change", ...).fillna("")
    result["_aa_change"] = protein_change.map(_hgvsp_to_eve_key)

In the whole-genome ClinVar cohort, `protein_change` is 100% NULL, so every
`_aa_change` was None -> the gene+aa_change join matched nothing -> eve_score
0.5 everywhere, EVEN THOUGH AlphaMissense (step 10b) populates wt_aa/protein_pos/
mut_aa for ~130k missense variants and 2,149 of their genes have EVE models.

The EVE lookup table itself keys aa_change as wt_aa + position + mt_aa
(eve.py:357-360). This patch makes the VARIANT side build the same key from the
populated coordinate triple (wt_aa/protein_pos/mut_aa) FIRST, falling back to the
HGVSp parse of protein_change for cohorts that carry it instead. Purely additive:
a protein_change-only cohort behaves exactly as before; a coordinate cohort now
matches. Mirrors the coordinate-first / HGVSp-fallback pattern already used at
real_data_prep 10b/10c.

Target (anchor verified against the live file):
  src/genomic_variant_classifier/data/eve.py  (the _annotate key block, ~402-407)

ANCHOR-BASED, IDEMPOTENT. Written LF-preserving (Python tolerates either; we keep
the file's existing newline style by writing text unchanged elsewhere).

  python scripts/patch_eve_aa_change_from_triple.py --check
  python scripts/patch_eve_aa_change_from_triple.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

EVE = Path("src/genomic_variant_classifier/data/eve.py")

ANCHOR = (
    "        # Derive aa_change from protein_change\n"
    "        protein_change = result.get(\n"
    '            "protein_change",\n'
    "            pd.Series([\"\"] * len(result), index=result.index),\n"
    "        ).fillna(\"\")\n"
    "        result[\"_aa_change\"] = protein_change.map(_hgvsp_to_eve_key)\n"
)

INSERT = (
    "        # Derive aa_change for the EVE join. The lookup side builds its key as\n"
    "        # wt_aa + position + mt_aa (see _parse_single_csv); build the SAME key on\n"
    "        # the variant side from the populated coordinate triple\n"
    "        # (wt_aa / protein_pos / mut_aa, filled by AlphaMissense step 10b) FIRST,\n"
    "        # then fall back to parsing protein_change (HGVSp) for cohorts that carry\n"
    "        # it instead. Coordinate-first is load-bearing: the whole-genome ClinVar\n"
    "        # cohort has protein_change 100% null, so a protein_change-only key left\n"
    "        # eve_score at 0.5 for every variant despite coords being present.\n"
    "        def _eve_key_from_triple(_wt: object, _pos: object, _mut: object) -> Optional[str]:\n"
    "            if _wt is None or _mut is None or pd.isna(_pos):\n"
    "                return None\n"
    "            _wt_s = str(_wt).strip()\n"
    "            _mut_s = str(_mut).strip()\n"
    "            if not _wt_s or not _mut_s:\n"
    "                return None\n"
    "            try:\n"
    "                return f\"{_wt_s}{int(_pos)}{_mut_s}\"\n"
    "            except (TypeError, ValueError):\n"
    "                return None\n"
    "\n"
    "        if {\"wt_aa\", \"protein_pos\", \"mut_aa\"}.issubset(result.columns):\n"
    "            _triple_key = [\n"
    "                _eve_key_from_triple(_w, _p, _m)\n"
    "                for _w, _p, _m in zip(\n"
    "                    result[\"wt_aa\"], result[\"protein_pos\"], result[\"mut_aa\"]\n"
    "                )\n"
    "            ]\n"
    "        else:\n"
    "            _triple_key = [None] * len(result)\n"
    "\n"
    "        protein_change = result.get(\n"
    '            "protein_change",\n'
    "            pd.Series([\"\"] * len(result), index=result.index),\n"
    "        ).fillna(\"\")\n"
    "        _hgvsp_key = protein_change.map(_hgvsp_to_eve_key)\n"
    "\n"
    "        # Coordinate triple wins where present; HGVSp fills the remaining rows.\n"
    "        result[\"_aa_change\"] = [\n"
    "            _t if _t is not None else _h\n"
    "            for _t, _h in zip(_triple_key, _hgvsp_key)\n"
    "        ]\n"
)

MARKER = "# Derive aa_change for the EVE join."  # idempotency sentinel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not EVE.exists():
        print(f"FAIL: {EVE} not found.")
        return 2
    src = EVE.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): eve.py already patched.")
        return 0

    n = src.count(ANCHOR)
    if n != 1:
        print(f"FAIL: anchor occurs {n}x (need exactly 1). The _annotate aa_change "
              "block differs from expected; not patching blind.")
        return 3

    # Optional integrity checks: confirm the symbols the insert relies on exist.
    needs = ["_hgvsp_to_eve_key", "Optional", "import pandas as pd", "def _annotate"]
    missing = [s for s in needs if s not in src]
    if missing:
        print(f"FAIL: eve.py missing expected symbols {missing}; aborting to avoid a broken patch.")
        return 4

    if ns.check:
        print("CHECK: anchor found exactly once; required symbols present (_hgvsp_to_eve_key, Optional, pandas).")
        print("RESULT: PASS (check)")
        return 0

    patched = src.replace(ANCHOR, INSERT, 1)
    backup = EVE.with_suffix(".py.pre_aa_triple.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="")
        print(f"OK: backup -> {backup}")
    EVE.write_text(patched, encoding="utf-8", newline="")

    # Post-checks
    after = EVE.read_text(encoding="utf-8")
    checks = [
        ("triple key helper present", "_eve_key_from_triple" in after),
        ("triple-first construction", '{"wt_aa", "protein_pos", "mut_aa"}.issubset' in after),
        ("hgvsp fallback retained", "_hgvsp_key = protein_change.map(_hgvsp_to_eve_key)" in after),
        ("coalesce triple|hgvsp", "_t if _t is not None else _h" in after),
        ("old protein_change-only line removed",
         'result["_aa_change"] = protein_change.map(_hgvsp_to_eve_key)' not in after),
    ]
    ok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}")
        ok &= present
    # Compile check
    import py_compile
    try:
        py_compile.compile(str(EVE), doraise=True)
        print("  OK  eve.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL  eve.py does not compile: {exc}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
