#!/usr/bin/env python3
r"""patch_api_omim_molecular.py

Two coupled edits so the API knows about omim_n_diseases_molecular:
  EDIT A — schemas.py VariantRequest model: add the field (Optional[int], default
           None, ge=0), mirroring omim_n_diseases. Default => backward compatible
           (existing callers who omit it don't get a 422).
  EDIT B — main.py _variant_to_row dict: add the entry so the value flows into
           the feature row. (Patching only one of these breaks the API.)

Anchors verified against reads 19a (schemas) and 16b (main dict). IDEMPOTENT.
"""
from __future__ import annotations
import argparse, py_compile
from pathlib import Path

SCHEMAS = Path("src/genomic_variant_classifier/api/schemas.py")
MAIN    = Path("src/genomic_variant_classifier/api/main.py")

SCHEMA_OLD = '''    omim_n_diseases: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of OMIM disease phenotypes for this gene.",
    )
    omim_is_autosomal_dominant: Optional[int] = Field('''

SCHEMA_NEW = '''    omim_n_diseases: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of OMIM disease phenotypes for this gene.",
    )
    omim_n_diseases_molecular: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of OMIM phenotypes with a confirmed molecular basis (mapping key '(3)').",
    )
    omim_is_autosomal_dominant: Optional[int] = Field('''

MAIN_OLD = '''        "omim_n_diseases":            req.omim_n_diseases,
        "omim_is_autosomal_dominant": req.omim_is_autosomal_dominant,'''

MAIN_NEW = '''        "omim_n_diseases":            req.omim_n_diseases,
        "omim_n_diseases_molecular":  req.omim_n_diseases_molecular,
        "omim_is_autosomal_dominant": req.omim_is_autosomal_dominant,'''


def _patch(path, old, new, marker, label, check):
    src = path.read_text(encoding="utf-8")
    if marker in src:
        print(f"OK (idempotent): {label} already patched."); return True, False
    c = src.count(old)
    if c != 1:
        print(f"FAIL: {label} anchor occurs {c}x (need 1)."); return False, False
    if check:
        print(f"CHECK: {label} anchor found once."); return True, False
    backup = path.with_suffix(path.suffix + ".pre_omim_molecular.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    path.write_text(src.replace(old, new, 1), encoding="utf-8", newline="\n")
    ok = marker in path.read_text(encoding="utf-8")
    print(f"  {'OK' if ok else 'MISSING'}  {label}")
    return ok, True


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    for p in (SCHEMAS, MAIN):
        if not p.exists():
            print(f"FAIL: {p} not found."); return 2
    a_ok, _ = _patch(SCHEMAS, SCHEMA_OLD, SCHEMA_NEW, "omim_n_diseases_molecular", "schemas VariantRequest", ns.check)
    b_ok, _ = _patch(MAIN, MAIN_OLD, MAIN_NEW, '"omim_n_diseases_molecular"', "main _variant_to_row", ns.check)
    if ns.check:
        print("RESULT:", "PASS (check)" if (a_ok and b_ok) else "FAIL (check)")
        return 0 if (a_ok and b_ok) else 3
    ok = a_ok and b_ok
    for p, label in ((SCHEMAS, "schemas.py"), (MAIN, "main.py")):
        try:
            py_compile.compile(str(p), doraise=True); print(f"  OK  {label} compiles")
        except py_compile.PyCompileError as exc:
            print(f"  FAIL compile {label}: {exc}"); ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
