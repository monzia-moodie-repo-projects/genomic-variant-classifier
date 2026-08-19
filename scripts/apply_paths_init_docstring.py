#!/usr/bin/env python3
"""apply_paths_init_docstring.py -- Author: Monzia Moodie

The paths package docstring enumerated THREE roots after a fourth had landed.

THE DEFECT, AND IT IS MINE
`cache_root` was added at 05f1a72 (INSTALLER-TRANSACTION-1 step 2). The package
docstring at src/genomic_variant_classifier/paths/__init__.py opened with "one
authority, three roots" and listed exactly three. I added the field and did not
re-derive the enumeration two directories away.

That is the same defect as the ".gitattributes carries 31 rules" error
corrected at 320e9cf: a count stated once and carried forward while the thing
it counts changed.

CORRECTED IN PLACE, NOT SUPERSEDED
REQUIRED_PROVENANCE_CORRECTION governs RECORDS -- a changelog entry, a session
document, a statement about what was believed at a past moment. Those get
corrections beside them so the original survives.

This is not a record. It is a LIVE DESCRIPTION OF CURRENT STRUCTURE, and a
module docstring that describes the module wrongly is simply wrong. The
distinction was drawn at ed10e41 and applies in the opposite direction here.

MEASURED 2026-08-19 BEFORE COMPOSING THE EDIT
    __init__.py           585 bytes, 15 lines, LF-only, pure ASCII
    non-docstring code    ONE statement: `from __future__ import annotations`
    imports FROM the package (not the module) : ZERO

So the file is documentation only, and this edit carries no functional risk.

WHY THE ANCHOR IS THE DOCSTRING ALONE
A reconstruction of the whole file came to 586 bytes against the real 585 --
one byte of blank-line structure I had assumed wrongly. Anchoring on the
docstring rather than the file means the surrounding structure is preserved
exactly as it is, whatever it is.

AND WHY THE REPLACEMENT DOES NOT CONTAIN THE STALE PHRASE
A first draft explained the correction with the words "three roots", which
would make any verifier searching for the stale phrase report failure on the
EXPLANATION of the fix. Same shape as the ast.dump check that matched the
docstring describing the defect it guarded against. The replacement now says
"undercounted the domains" instead.

Usage:  python scripts/apply_paths_init_docstring.py --repo-root . --check
        python scripts/apply_paths_init_docstring.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

TARGET = "src/genomic_variant_classifier/paths/__init__.py"
SPEC = "paths_init_spec.json"

#: The four domains the docstring must enumerate, in resolution order.
DOMAINS = ("project_root", "artifact_root", "state_root", "cache_root")


def _verify(source: str) -> tuple:
    """By AST, on the DOCSTRING -- not by substring over the file."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)
    doc = ast.get_docstring(tree)
    if not doc:
        return False, "the module docstring is gone"
    missing = [d for d in DOMAINS if d not in doc]
    if missing:
        return False, "the docstring omits {}".format(missing)
    if "three roots" in doc:
        return False, ("the docstring still says 'three roots'; there are "
                       "{}".format(len(DOMAINS)))
    if "four roots" not in doc:
        return False, "the docstring does not state the count"
    body = [n for n in tree.body if not (isinstance(n, ast.Expr)
            and isinstance(getattr(n, "value", None), ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.ImportFrom):
        return False, ("the module gained or lost code: {} non-docstring "
                       "statement(s)".format(len(body)))
    return True, "{} domains enumerated; the single import is intact".format(
        len(DOMAINS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    spec_path = Path(args.spec) if args.spec else Path(__file__).with_name(SPEC)
    if not spec_path.exists():
        print("  ERROR: the edit spec is missing: {}".format(spec_path))
        return 2
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    p = Path(args.repo_root) / TARGET
    if not p.exists():
        print("  ERROR: not found: {}".format(TARGET))
        return 2
    raw = p.read_bytes()
    src = raw.decode("utf-8")
    print("  target: {} bytes, {} lines, CRLF {}, non-ASCII {}".format(
        len(raw), len(src.splitlines()), raw.count(b"\r\n"),
        sum(1 for b in raw if b > 0x7F)))

    ok, _ = _verify(src)
    if ok:
        print("  already applied")
        return 0

    n = src.count(spec["old"])
    if n != 1:
        print("  ERROR: the docstring anchor occurs {} time(s), expected 1; "
              "NOTHING written.".format(n))
        return 1
    print("  anchor OK  (1 occurrence)")

    patched = src.replace(spec["old"], spec["new"], 1)
    ok, msg = _verify(patched)
    if not ok:
        print("  ERROR: verification failed BEFORE writing ({}); "
              "NOTHING written.".format(msg))
        return 1
    print("  pre-write  {}".format(msg))

    if args.check:
        print("\n  --check: 1 edit pending. Nothing written.")
        return 0

    backup = p.with_suffix(p.suffix + ".pre_docstring.bak")
    if not backup.exists():
        backup.write_bytes(raw)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(patched)
    after = p.read_bytes()
    if after.count(b"\r\n") != raw.count(b"\r\n"):
        p.write_bytes(raw)
        print("  ERROR: line endings changed; ROLLED BACK.")
        return 1
    print("  wrote {}  ({} bytes, CRLF {})".format(
        TARGET, len(after), after.count(b"\r\n")))

    ok, msg = _verify(after.decode("utf-8"))
    if not ok:
        p.write_bytes(raw)
        print("  ERROR: POST-WRITE failed ({}); ROLLED BACK.".format(msg))
        return 1
    print("  post-write {}".format(msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
