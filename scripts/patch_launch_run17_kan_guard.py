#!/usr/bin/env python3
"""
patch_launch_run17_kan_guard.py  --  Monzia Moodie

Insert an idempotent imodelsx KAN package-patch step into scripts/launch_run17_baseline.sh, between the
smoke-import step and the .pyc cleanup. patch_imodelsx_kan.py's own docstring says the launcher "should
too" -- without it, imodelsx KANClassifier.fit raises NameError and KAN silently drops from BOTH ensemble
fits (invalid model-comparison run). The full launcher otherwise has NO KAN guard (only the smoke gate does),
so a launch on a VM where smoke did not run would lose KAN unnoticed. Exact-anchor, idempotent, EOL/BOM-safe.
Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("scripts/launch_run17_baseline.sh")
MARKER = "patch_imodelsx_kan.py"
ANCHOR = (
    'echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"\n\n'
    '# -- 3. Stale .pyc cleanup ----------------------------------------------------\n'
)
INSERT = (
    'echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"\n\n'
    '# -- 2b. imodelsx KAN package patch (idempotent) ------------------------------\n'
    '# patch_imodelsx_kan.py docstring: "the smoke gate invokes it first, and the launcher should too."\n'
    '# Unpatched, imodelsx KANClassifier.fit raises NameError -> KAN drops from BOTH ensemble fits, an\n'
    '# invalid model-comparison run. The full launcher has no other KAN guard (only the smoke gate does).\n'
    'echo "==> [2b/6] imodelsx KAN patch" | tee -a "$LOG"\n'
    'if ! python scripts/patch_imodelsx_kan.py 2>&1 | tee -a "$LOG"; then\n'
    '    echo "==> ABORT (exit 3): imodelsx KAN patch failed (KAN would drop from the ensemble)" | tee -a "$LOG"; exit 3\n'
    'fi\n\n'
    '# -- 3. Stale .pyc cleanup ----------------------------------------------------\n'
)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if MARKER in text:
        print("[skip] launcher already applies patch_imodelsx_kan.py"); return 0
    if text.count(ANCHOR) != 1:
        print(f"ERROR: anchor found {text.count(ANCHOR)}x (expected 1); not patching", file=sys.stderr); return 3
    text = text.replace(ANCHOR, INSERT)
    data = text.replace("\n", eol).encode("utf-8")
    if had_bom:
        data = b"\xef\xbb\xbf" + data
    TARGET.write_bytes(data)
    print(f"[patched] launch_run17_baseline.sh += [2b] imodelsx KAN patch (eol={'CRLF' if eol!=chr(10) else 'LF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
