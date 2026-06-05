#!/usr/bin/env python3
"""
test_patch_imodelsx.py - validate the imodelsx KAN patcher logic in .venv312.

Run from the repo root:
    python scripts/test_patch_imodelsx.py

Exercises patch_imodelsx_kan.patch_file on a fixture reproducing the v1.0.13
bare-name bug: all three refs rewritten to self.<attr>, patched file compiles,
idempotent re-run, clean file untouched. Does NOT touch the real imodelsx
(that is done explicitly by the installer's final step). Exit 0 = pass.
"""
from __future__ import annotations

import importlib.util
import py_compile
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "patch_imodelsx_kan", HERE / "patch_imodelsx_kan.py")
pk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pk)

BUGGY = (
    "class KANClassifier:\n"
    "    def fit(self, X, y):\n"
    "        a, b, c, d = train_test_split(\n"
    "            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)\n"
    "        return self\n"
)
CLEAN = "class KANClassifier:\n    def fit(self, X, y):\n        return self\n"


def main() -> int:
    fails = []
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "kan_sklearn.py"
        p.write_text(BUGGY, encoding="utf-8")
        rc = pk.patch_file(p)
        after = p.read_text(encoding="utf-8")
        if rc != 0:
            fails.append(f"patch_file returned {rc}")
        for o in ("test_size=test_size", "random_state=random_state", "shuffle=shuffle"):
            if o in after:
                fails.append(f"bare ref still present: {o}")
        for n in ("test_size=self.test_size", "random_state=self.random_state", "shuffle=self.shuffle"):
            if n not in after:
                fails.append(f"missing rewritten ref: {n}")
        try:
            py_compile.compile(str(p), doraise=True)
            print("[ok] patched fixture rewrites 3 refs and compiles")
        except py_compile.PyCompileError as e:
            fails.append(f"patched fixture does not compile: {e}")

        if pk.patch_file(p) != 0 or p.read_text(encoding="utf-8") != after:
            fails.append("not idempotent")
        else:
            print("[ok] idempotent re-run")

        c = Path(d) / "clean.py"
        c.write_text(CLEAN, encoding="utf-8")
        if pk.patch_file(c) != 0 or c.read_text(encoding="utf-8") != CLEAN:
            fails.append("clean file was modified")
        else:
            print("[ok] clean file untouched")

    print()
    if fails:
        for f in fails:
            print("[FAIL] " + f)
        return 1
    print("ALL PATCH-IMODELSX CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
