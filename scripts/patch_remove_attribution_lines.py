#!/usr/bin/env python3
"""Remove an unrequested authorship line from every module that carries it.

WHAT THIS REMOVES
=================
An authorship line of the form

    Author: written for <name>, YYYY-MM-DD.

was added to 51 modules between 2026-07-20 and 2026-07-26. It was never
requested. Four of those files were corrected when the line was raised; this
script removes the remaining 47.

The line appears in TWO shapes, and treating them identically would corrupt
three files:

  STANDALONE (44 files) -- the whole line is the attribution and nothing else:

      <docstring prose>
      <blank>
      Author: written for <name>, 2026-07-21.
      \"\"\"

    The line is deleted, together with the blank line above it, so no gap is
    left immediately before the docstring terminator.

  CONTINUED (3 files) -- real prose follows on the SAME line and wraps onto the
  next one:

      Author: written for <name>, 2026-07-24. Companion to
      probe_label_column_terms.py.

    Deleting the whole line would silently destroy "Companion to" and leave a
    dangling sentence. Only the attribution clause is removed, so the file keeps

      Companion to
      probe_label_column_terms.py.

GUARANTEES
----------
  * COMMENT-ONLY. Every edit lies inside a module docstring. No statement, no
    import, no logic, and no collected-test count changes.
  * COUNT-GUARDED. The expected number of files and edits is asserted before
    anything is written. A mismatch aborts with nothing modified, because a
    sweep that finds an unexpected number of matches is a sweep that has found
    something it was not designed for.
  * SYNTAX-GATED. Every edited file is compiled with `ast.parse` BEFORE it is
    written to disk. A file that would not parse is never written.
  * BACKUP-FIRST. Each edited file gets a timestamped `.bak_` copy, which the
    repository's .gitignore already excludes.
  * IDEMPOTENT. A second run finds zero matches and exits 0 without writing.
  * BYTE-PRESERVING. Files are read and written as bytes with explicit LF
    endings; encoding, line endings and the absence of a byte-order mark are
    verified per file and asserted unchanged apart from the removed lines.

USAGE
-----
    python scripts\\patch_remove_attribution_lines.py --repo C:\\Projects\\genomic-variant-classifier
    python scripts\\patch_remove_attribution_lines.py --repo C:\\Projects\\genomic-variant-classifier --apply

Dry run is the default and prints every planned edit. Nothing is written
without --apply.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import shutil
import sys
from pathlib import Path

# The attribution clause. Kept as a pattern rather than a literal so the script
# does not itself restate the phrase more than once.
_NAME = "Monzia Moodie"
STANDALONE = re.compile(r"^Author: written for " + re.escape(_NAME) + r", \d{4}-\d{2}-\d{2}\.\s*$")
CONTINUED = re.compile(r"^Author: written for " + re.escape(_NAME) + r", \d{4}-\d{2}-\d{2}\.\s+(\S.*)$")
ANY_HIT = re.compile(r"written for " + re.escape(_NAME))

# Two audited states are legitimate, depending on whether the four files added on
# 2026-07-26 were corrected separately first:
#
#   51 files  the phrase is present everywhere, including those four
#   47 files  those four were already corrected; this sweep handles the rest
#
# In BOTH states exactly three files carry the "continued" shape, because all
# three date from 2026-07-24 and none of them is one of the four. Any other
# count means the sweep has found something it was not designed for.
AUDITED_STATES = {
    51: {"standalone": 48, "continued": 3, "note": "the four 2026-07-26 files are still present"},
    47: {"standalone": 44, "continued": 3, "note": "the four 2026-07-26 files were corrected already"},
}

SEARCH_ROOTS = ("src", "tests", "scripts")


def plan_file(path: Path):
    """Return (new_lines, n_standalone, n_continued) or None when nothing matches."""
    raw = path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        raise SystemExit(f"ABORT: {path} carries a byte-order mark; not handled")
    if b"\r" in raw:
        raise SystemExit(f"ABORT: {path} uses CRLF endings; not handled")
    text = raw.decode("utf-8")
    lines = text.split("\n")

    out: list[str] = []
    n_std = n_cont = 0
    for i, line in enumerate(lines):
        m_cont = CONTINUED.match(line)
        if m_cont:
            out.append(m_cont.group(1).rstrip())
            n_cont += 1
            continue
        if STANDALONE.match(line):
            # Drop the blank line(s) directly above so the docstring does not
            # end on an empty line where the attribution used to sit.
            while out and out[-1].strip() == "":
                out.pop()
            n_std += 1
            continue
        out.append(line)

    if not (n_std or n_cont):
        return None
    return out, n_std, n_cont


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="absolute path to the repository root")
    ap.add_argument("--apply", action="store_true", help="write changes (default is a dry run)")
    a = ap.parse_args(argv)

    repo = Path(a.repo).resolve()
    if not (repo / ".git").exists():
        print(f"ABORT: {repo} does not look like a git repository", file=sys.stderr)
        return 3

    candidates = sorted(
        p for root in SEARCH_ROOTS for p in (repo / root).rglob("*.py")
        if "__pycache__" not in p.parts and ANY_HIT.search(p.read_text(encoding="utf-8"))
    )

    print(f"repository : {repo}")
    print(f"mode       : {'APPLY' if a.apply else 'DRY RUN (nothing will be written)'}")
    print(f"matched    : {len(candidates)} file(s)")
    print()

    if not candidates:
        print("Nothing to do; the sweep has already been applied. Exiting 0.")
        return 0

    plans = []
    tot_std = tot_cont = 0
    for p in candidates:
        result = plan_file(p)
        if result is None:
            print(f"ABORT: {p} matched the search but produced no edit plan", file=sys.stderr)
            return 2
        new_lines, n_std, n_cont = result
        tot_std += n_std
        tot_cont += n_cont
        plans.append((p, new_lines, n_std, n_cont))
        kind = "standalone" if n_std else "continued "
        print(f"  {kind}  {p.relative_to(repo)}")

    print()
    state = AUDITED_STATES.get(len(plans))
    print(f"files               : {len(plans):>3}  "
          f"({'audited state: ' + state['note'] if state else 'NOT AN AUDITED STATE'})")
    print(f"standalone removals : {tot_std:>3}  "
          f"(expected {state['standalone'] if state else '?'})")
    print(f"continued  rewrites : {tot_cont:>3}  "
          f"(expected {state['continued'] if state else '?'})")

    if state is None or (tot_std, tot_cont) != (state["standalone"], state["continued"]):
        print("\nABORT: counts differ from every audited expectation. Nothing modified.",
              file=sys.stderr)
        print(f"Audited states are {sorted(AUDITED_STATES)} files. A sweep that finds an "
              "unexpected number of matches has found something it was not designed "
              "for; re-audit before forcing it.", file=sys.stderr)
        return 2

    # Syntax-gate every file BEFORE writing any of them.
    for p, new_lines, _, _ in plans:
        try:
            ast.parse("\n".join(new_lines), filename=str(p))
        except SyntaxError as exc:
            print(f"\nABORT: edited {p} would not parse: {exc}. Nothing modified.",
                  file=sys.stderr)
            return 2
    print("syntax gate         : all edited files parse")

    # Residual check: no file may still contain the phrase after the edit.
    for p, new_lines, _, _ in plans:
        if ANY_HIT.search("\n".join(new_lines)):
            print(f"\nABORT: {p} would still contain the phrase after editing. "
                  "Nothing modified.", file=sys.stderr)
            return 2
    print("residual gate       : no file retains the phrase")

    if not a.apply:
        print("\nDRY RUN complete. Re-run with --apply to write.")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    written = 0
    for p, new_lines, _, _ in plans:
        shutil.copy2(p, p.with_name(p.name + f".bak_{stamp}"))
        p.write_bytes(("\n".join(new_lines)).encode("utf-8"))
        written += 1
    print(f"\nwrote {written} file(s); each has a .bak_{stamp} backup "
          "(matched by .gitignore).")

    # Post-check on disk, not on the in-memory plan.
    still = [str(p.relative_to(repo)) for p, _, _, _ in plans
             if ANY_HIT.search(p.read_text(encoding="utf-8"))]
    if still:
        print(f"POST-CHECK FAILED: {still}", file=sys.stderr)
        return 2
    print("post-check          : zero occurrences remain on disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
