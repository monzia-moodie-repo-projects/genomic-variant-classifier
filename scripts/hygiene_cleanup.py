#!/usr/bin/env python3
"""hygiene_cleanup.py -- Author: Monzia Moodie

Repo hygiene for the EVE/provisioning session leftovers. DRY-RUN by default: prints
exactly what it would change. Re-run with --apply to act. It NEVER commits -- you
review `git status` and commit yourself.

Actions:
  1. Append (idempotently, BOM-free) ignore patterns for runtime state + scratch:
       offers.txt
       src/genomic_variant_classifier/agent_layer/data/
       src/genomic_variant_classifier/agent_layer/logs/
       *.diff
  2. `git rm --cached` any of those paths that are already TRACKED (so runtime state
     stops being versioned) -- file stays on disk, leaves the index.
  3. Delete the leftover diff artifacts data_readiness_agent.diff / orchestrator.diff.
  4. Print `git status --short` so you can stage the intended files and commit.

Run from the repo root with the venv active:  python scripts/hygiene_cleanup.py
then, to act:                                  python scripts/hygiene_cleanup.py --apply
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IGNORE_PATTERNS = [
    "offers.txt",
    "src/genomic_variant_classifier/agent_layer/data/",
    "src/genomic_variant_classifier/agent_layer/logs/",
    "*.diff",
]
RUNTIME_PATHS = [
    "offers.txt",
    "src/genomic_variant_classifier/agent_layer/data",
    "src/genomic_variant_classifier/agent_layer/logs",
]
DIFF_ARTIFACTS = ["data_readiness_agent.diff", "orchestrator.diff"]


def _git(repo: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def _is_tracked(repo: Path, path: str) -> bool:
    rc, _ = _git(repo, "ls-files", "--error-unmatch", path)
    return rc == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry-run)")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    gi = repo / ".gitignore"
    if not (repo / ".git").exists():
        print(f"not a git repo: {repo}")
        return 2
    act = args.apply
    tag = "APPLY" if act else "DRY-RUN"
    print(f"[{tag}] repo: {repo}")

    # 1. .gitignore additions (idempotent, BOM-free)
    existing = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    existing_set = {ln.strip() for ln in existing}
    to_add = [p for p in IGNORE_PATTERNS if p not in existing_set]
    if to_add:
        print(f"[{tag}] .gitignore += {to_add}")
        if act:
            block = ("\n# session hygiene: runtime state + scratch (do not version)\n"
                     + "\n".join(to_add) + "\n")
            with open(gi, "a", encoding="utf-8", newline="\n") as fh:  # no BOM
                fh.write(block)
    else:
        print(f"[{tag}] .gitignore already covers all patterns (no-op)")

    # 2. untrack runtime paths that are currently tracked
    for path in RUNTIME_PATHS:
        if (repo / path).exists() and _is_tracked(repo, path):
            print(f"[{tag}] git rm --cached -r {path}  (tracked runtime state -> untrack)")
            if act:
                _git(repo, "rm", "-r", "--cached", path)
        else:
            print(f"[{tag}] {path}: not tracked or absent (no untrack needed)")

    # 3. delete leftover diff artifacts
    for d in DIFF_ARTIFACTS:
        f = repo / d
        if f.exists():
            print(f"[{tag}] delete artifact {d}")
            if act:
                if _is_tracked(repo, d):
                    _git(repo, "rm", "-f", d)
                else:
                    f.unlink()
        else:
            print(f"[{tag}] {d}: absent (nothing to delete)")

    # 4. show status
    print(f"\n[{tag}] git status --short:")
    _, out = _git(repo, "status", "--short")
    print(out or "  (clean)")
    if not act:
        print("\nDRY-RUN only -- re-run with --apply to perform the above, then review "
              "`git status`, stage the intended files (tests, docs), and commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
