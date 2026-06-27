#!/usr/bin/env python3
r"""patch_requirements_pybigtools.py

Add pybigtools (PhyloP's BigWig reader) to BOTH dependency files:
  - requirements.in   : the pip-compile SOURCE (under '# Annotation connectors')
  - requirements.txt  : the file the VM actually installs from (hand-maintained tail)

WHY BOTH: requirements.in is correct-practice (survives recompilation); requirements.txt
is what the VM's `pip install -r requirements.txt` actually reads NOW (run16-style install;
launch_run17 activates a prebuilt /venv/main and does NOT pip-install, so the dep must be
present in the file used at provisioning time). Adding only to .in without recompiling =>
VM still misses it. Adding only to .txt => lost on next pip-compile. Both covers both.

PIN: pybigtools>=0.3.0 (NOT ==). PyPI confirms 0.3.0 ships manylinux wheels for
cp39/310/311/312/313, so any VM Python gets a binary wheel (no Rust source build).
Without this, PhyloPConnector hits ImportError -> silent phylop_score=0.0 across the cohort.

Anchors verified against reads 23a/23b. ANCHOR-BASED, IDEMPOTENT, CRLF-safe.
"""
from __future__ import annotations
import argparse
from pathlib import Path

REQ_IN = Path("requirements.in")
REQ_TXT = Path("requirements.txt")
MARKER = "pybigtools"

# requirements.in: insert under "# Annotation connectors" (head shows this section).
# Anchor on that comment line; insert the dep right after it.
IN_ANCHOR = "# Annotation connectors\n"
IN_INSERT = "# Annotation connectors\npybigtools>=0.3.0  # PhyloP BigWig reader (cp39-cp313 manylinux wheels; no pyBigWig on Windows)\n"

# requirements.txt: append to the hand-maintained tail, after transformers==4.46.3 (the last line).
TXT_ANCHOR = "transformers==4.46.3\n"
TXT_INSERT = "transformers==4.46.3\npybigtools>=0.3.0  # PhyloP BigWig reader (added Run-17; cp39-cp313 wheels)\n"
# Fallback if the file does not end with a newline after transformers:
TXT_ANCHOR_NONL = "transformers==4.46.3"


def _patch_file(path: Path, anchor: str, insert: str, check: bool, label: str,
                anchor_alt: str | None = None) -> tuple[bool, bool]:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"OK (idempotent): {label} already has pybigtools."); return True, False
    use_anchor = anchor
    cnt = src.count(anchor)
    if cnt != 1 and anchor_alt is not None:
        # try alt (no trailing newline) — append a newline + line
        cnt_alt = src.count(anchor_alt)
        if cnt_alt == 1 and not src.endswith("\n"):
            use_anchor = anchor_alt
            insert = anchor_alt + "\n" + insert.split("\n",1)[1]
            cnt = 1
    if cnt != 1:
        print(f"FAIL: {label} anchor occurs {cnt}x (need 1)."); return False, False
    if check:
        print(f"CHECK: {label} anchor found once."); return True, False
    backup = path.with_suffix(path.suffix + ".pre_pybigtools.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")
    path.write_text(src.replace(use_anchor, insert, 1), encoding="utf-8", newline="\n")
    ok = MARKER in path.read_text(encoding="utf-8")
    print(f"  {'OK' if ok else 'MISSING'}  {label}: pybigtools added")
    return ok, True


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    for p in (REQ_IN, REQ_TXT):
        if not p.exists():
            print(f"FAIL: {p} not found."); return 2
    a_ok, _ = _patch_file(REQ_IN, IN_ANCHOR, IN_INSERT, ns.check, "requirements.in")
    b_ok, _ = _patch_file(REQ_TXT, TXT_ANCHOR, TXT_INSERT, ns.check, "requirements.txt",
                          anchor_alt=TXT_ANCHOR_NONL)
    ok = a_ok and b_ok
    if ns.check:
        print("RESULT:", "PASS (check)" if ok else "FAIL (check)")
        return 0 if ok else 3
    # post-check: both files contain pybigtools exactly once
    for p, label in ((REQ_IN, "requirements.in"), (REQ_TXT, "requirements.txt")):
        c = p.read_text(encoding="utf-8").count("pybigtools")
        print(f"  {label}: pybigtools count = {c}")
        ok &= (c >= 1)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
