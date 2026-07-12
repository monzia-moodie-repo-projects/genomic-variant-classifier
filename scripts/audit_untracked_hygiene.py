"""audit_untracked_hygiene.py -- evidence before deletion. v2, 2026-07-11.

READ-ONLY. Deletes nothing, moves nothing, stages nothing.

WHAT v1 GOT WRONG (fixed here; recorded so the mistake is not repeated)
----------------------------------------------------------------------
v1 classified each untracked script by "is its name mentioned anywhere?". Three flaws:

  1. SELF-POLLUTION. v1's own docstring named several files as examples, so those files
     came back "REFERENCED by scripts/audit_untracked_hygiene.py". The audit was citing
     itself as evidence. v2 excludes this file from the corpus.

  2. TEST MODULES MISCLASSIFIED. A test file is a pytest ENTRY POINT -- it is collected,
     not imported. Nothing referencing it does not make it an orphan. v2 treats every
     file under tests/ as a ROOT.

  3. NO TRANSITIVE CLOSURE, and it cut both ways:
       * scripts/diagnose_identity_join.py was called "REFERENCED" -- but it is referenced
         BY scripts/probe_identity_first_recovery.py, which a test imports. It is therefore
         TRANSITIVELY REQUIRED, and deleting it would break the suite.
       * scripts/verify_oof_alignment.py and scripts/fix_ece_binning.py were also called
         "REFERENCED" -- but only by scripts that are THEMSELVES orphans. They are
         TRANSITIVELY ORPHANED.
     A one-hop reference count cannot tell these apart. v2 computes reachability to a
     fixed point from the roots.

THE FINDING THAT PROMPTED ALL THIS
----------------------------------
Four TRACKED test files import UNTRACKED scripts:

    tests/test_diagnose_and_recover_alleleless.py -> scripts/diagnose_and_recover_alleleless.py
    tests/test_recover_alleleless_provenance.py   -> scripts/recover_alleleless_provenance.py
    tests/test_resolve_alleleless_ncbi.py         -> scripts/resolve_alleleless_ncbi.py
    tests/test_classify_alleleless_by_type.py     -> scripts/classify_alleleless_by_type.py

A FRESH CLONE OF THIS REPOSITORY HAS A RED SUITE: the tests are committed, the modules
they import are not. The 1,814 passing tests pass only because those scripts happen to
sit in this working tree -- "works on my machine, and generally false". Same species as
TRIAGE_2026-07-08 cluster A (the coverage gate whose verdict depended on disk contents),
and what that triage called "an uninventoried member of the trusted base".

Committing the REQUIRED scripts is therefore a BUG FIX, not housekeeping.

CLASSIFICATION (v2)
-------------------
  TEST-MODULE : lives under tests/. A pytest entry point -> COMMIT (it is already being
                collected and is inside your green count).
  REQUIRED    : reachable, transitively, from tests/ or src/ -> COMMIT. Deleting it
                breaks the suite or the package.
  DOC-ONLY    : named ONLY in docs/. Not needed to run anything, but it reproduces a
                documented investigation -> keep for provenance, or archive.
  ORPHAN      : reachable from nothing. Spent forensics -> delete, or move to
                scripts/forensics/ if the investigation is worth preserving.

    python scripts\\audit_untracked_hygiene.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SELF = Path(__file__).resolve()
RULE = "=" * 78

CODE_SUFFIXES = {".py", ".ps1", ".sh", ".toml", ".cfg", ".yml", ".yaml"}
DOC_SUFFIXES = {".md", ".rst", ".txt"}


def untracked() -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [ln[3:].strip() for ln in out.splitlines() if ln.startswith("??")]


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main() -> int:
    files = [Path(f) for f in untracked()]

    # Every script in the repo (tracked or not) -- the universe we resolve names against.
    all_scripts = {p.stem: p for p in Path("scripts").rglob("*.py") if p.resolve() != SELF}

    # ---- ROOTS: tests/ (pytest entry points) and src/ (the package) -----------
    roots: list[Path] = []
    for root_dir in ("tests", "src"):
        for p in Path(root_dir).rglob("*"):
            if p.is_file() and p.suffix in CODE_SUFFIXES and p.resolve() != SELF:
                roots.append(p)
    # pyproject can pin console entry points too
    if Path("pyproject.toml").is_file():
        roots.append(Path("pyproject.toml"))

    # ---- transitive closure to a fixed point ---------------------------------
    required: set[str] = set()
    frontier = list(roots)
    seen: set[Path] = set()

    while frontier:
        cur = frontier.pop()
        if cur in seen or cur.resolve() == SELF:
            continue
        seen.add(cur)
        text = read(cur)
        for stem, path in all_scripts.items():
            if stem in required:
                continue
            if re.search(rf"\b{re.escape(stem)}\b", text):
                required.add(stem)
                frontier.append(path)          # follow it -- transitive

    # ---- docs-only ------------------------------------------------------------
    doc_text = ""
    for p in Path("docs").rglob("*"):
        if p.is_file() and p.suffix in DOC_SUFFIXES:
            doc_text += read(p)

    buckets: dict[str, list[str]] = {"TEST-MODULE": [], "REQUIRED": [], "DOC-ONLY": [], "ORPHAN": [], "OTHER": []}

    for f in files:
        s = str(f).replace("\\", "/")
        if s.startswith("tests/"):
            buckets["TEST-MODULE"].append(s)
        elif f.suffix != ".py":
            buckets["OTHER"].append(s)
        elif f.stem in required:
            buckets["REQUIRED"].append(s)
        elif re.search(rf"\b{re.escape(f.stem)}\b", doc_text):
            buckets["DOC-ONLY"].append(s)
        else:
            buckets["ORPHAN"].append(s)

    print(RULE)
    print(f"UNTRACKED HYGIENE AUDIT v2 -- {len(files)} untracked file(s)")
    print("READ-ONLY. Transitive closure from tests/ + src/. Self excluded.")
    print(RULE)

    verdict = {
        "TEST-MODULE": "COMMIT -- pytest already collects these; they are inside your green count.",
        "REQUIRED":    "COMMIT -- transitively reachable from tests/ or src/. Deleting breaks the suite.",
        "DOC-ONLY":    "KEEP for provenance, or archive to scripts/forensics/. Not needed to run anything.",
        "ORPHAN":      "SPENT FORENSICS -- reachable from nothing. Delete, or archive to scripts/forensics/.",
        "OTHER":       "REVIEW individually.",
    }
    for b in ("TEST-MODULE", "REQUIRED", "DOC-ONLY", "ORPHAN", "OTHER"):
        items = sorted(buckets[b])
        print(f"\n{RULE}\n{b}  ({len(items)})\n  -> {verdict[b]}\n{RULE}")
        for s in items:
            print(f"  {s}")

    print(f"\n{RULE}\nSUMMARY")
    for b in ("TEST-MODULE", "REQUIRED", "DOC-ONLY", "ORPHAN", "OTHER"):
        print(f"  {b:12s}: {len(buckets[b]):3d}")
    print(RULE)
    print("\nCOMMIT bucket 1 + 2 FIRST -- that is the clean-clone bug fix, not housekeeping.")
    print("Then decide DOC-ONLY and ORPHAN deliberately. Never `git add -A`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
