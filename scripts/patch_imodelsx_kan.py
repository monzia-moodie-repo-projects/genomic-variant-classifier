#!/usr/bin/env python3
"""
scripts/patch_imodelsx_kan.py - fix the imodelsx v1.0.13 KANClassifier.fit bug.

imodelsx's KANClassifier.fit references bare `test_size`/`random_state`/`shuffle`
(they should be `self.<attr>`), so fitting raises NameError
("name 'test_size' is not defined") and KAN drops out of BOTH ensemble fits
(this is the Run 15 KAN_FAIL). kan.py already sets the matching instance
attributes; this rewrites the three bare-name references in the INSTALLED package
source to `self.<attr>` (the same fix the old launcher sed applied), so any FRESH
python process that imports imodelsx afterwards gets the corrected method.

Because it edits the file, run it ONCE per environment BEFORE the process that
imports imodelsx: the smoke gate invokes it first, and the launcher should too.
Idempotent (no-op if already correct), guarded (only the three known refs),
and non-destructive (verifies the result before returning success).

    python scripts/patch_imodelsx_kan.py
Exit 0 = patched or already-correct; 1 = imodelsx absent / unreadable / unwritable.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPLACEMENTS = [
    ("test_size=test_size", "test_size=self.test_size"),
    ("random_state=random_state", "random_state=self.random_state"),
    ("shuffle=shuffle", "shuffle=self.shuffle"),
]


def locate() -> Path | None:
    try:
        import imodelsx.kan.kan_sklearn as m  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[patch-imodelsx] imodelsx not importable: {e}")
        return None
    return Path(m.__file__)


def patch_file(p: Path) -> int:
    """Patch one source file in place. Returns process-style exit code."""
    try:
        src = p.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[patch-imodelsx] cannot read {p}: {e}")
        return 1
    needed = [(o, n) for o, n in REPLACEMENTS if o in src]
    if not needed:
        print(f"[patch-imodelsx] already correct (no bare-name refs) in {p.name}")
        return 0
    patched = src
    for o, n in needed:
        patched = patched.replace(o, n)
    try:
        p.write_text(patched, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"[patch-imodelsx] cannot write {p}: {e}")
        return 1
    remaining = [o for o, _ in REPLACEMENTS if o in p.read_text(encoding="utf-8")]
    if remaining:
        print(f"[patch-imodelsx] WARNING: bare refs still present after write: {remaining}")
        return 1
    print(f"[patch-imodelsx] fixed {len(needed)} bare-name ref(s) in {p}: "
          + ", ".join(o for o, _ in needed))
    return 0


def main() -> int:
    p = locate()
    if p is None:
        return 1
    return patch_file(p)


if __name__ == "__main__":
    sys.exit(main())
