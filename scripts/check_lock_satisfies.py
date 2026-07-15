#!/usr/bin/env python3
"""Assert requirements-api.lock SATISFIES requirements-api.txt. Deterministic. No recompile.

Created 2026-07-14 (roadmap 6.25), replacing a gate that could not stop failing.

WHY THE PREVIOUS GATE WAS WRONG
-------------------------------
On 2026-07-13 the lockfile gate was changed (roadmap 6.19) from "does the lock install?" to
"RECOMPILE the lock and fail if the result differs from what is committed". The reasoning was
sound -- checking that a lock installs is a PROXY; it never checks the lock is the RIGHT lock.

The implementation was not. It ran:

    pip-compile --quiet --output-file=/tmp/regenerated.lock requirements-api.txt
    diff committed regenerated

and that comparison is **guaranteed to fail on a schedule, with nothing wrong.** Five
independent reasons, every one sufficient on its own:

1. PIP-COMPILE PRESERVES EXISTING PINS. Given an output file that already exists, pip-compile
   treats its pins as constraints and changes only what it must. Compiling to a NEW path
   (/tmp/regenerated.lock) has no pins to preserve, so it resolves the whole tree to the
   latest on the index. The gate therefore compared "the committed lock" against "the newest
   possible resolution of the universe, today". Those are different questions.

   MEASURED 2026-07-14, on one machine, minutes apart:
       pip-compile requirements-api.txt -o requirements-api.lock   -> slowapi 0.1.9  (preserved)
       pip-compile requirements-api.txt -o /tmp/fresh.lock         -> slowapi 0.1.10 (latest)
   Same input. Same second. Opposite answers.

2. UPSTREAM RELEASES. 22 transitive packages drifted between the commit and a fresh compile
   MINUTES later -- anyio, certifi, charset-normalizer, click, fonttools, httptools, idna,
   matplotlib, narwhals, packaging, pillow, plotly, prometheus-client, pytz, scipy, slowapi,
   typing-extensions, tzdata, urllib3, watchfiles, websockets, wrapt -- while
   `git diff HEAD -- requirements-api.txt` was EMPTY. Nobody changed anything. Time passed.

3. PIP-TOOLS ITSELF WAS UNPINNED. The step ran `pip install -q pip-tools`, so the tool
   resolving the tree drifted too. Its `--strip-extras` default changed between versions,
   which alone flips `uvicorn[standard]` to `uvicorn` in the output and produces a diff.

4. ENVIRONMENT MARKERS MAKE THE OUTPUT PLATFORM-SPECIFIC. See below -- this is the real
   defect the broken gate stumbled into.

5. PYTHON VERSION. The committed lock's header says Python 3.12; the gate's runner is 3.11.
   Markers keyed on python_version resolve differently.

A GATE THAT GOES RED BECAUSE A STRANGER PUBLISHED A WHEEL IS NOT A GATE. It is a scheduled
false alarm, and a scheduled false alarm gets muted -- taking the real signal with it. That is
root pattern (a) wearing a gate's uniform: a check written once that becomes a lie on a
schedule.

WHAT THIS CHECKS INSTEAD
------------------------
The thing the lock actually has to be true about:

    EVERY DIRECT REQUIREMENT IN requirements-api.txt IS PRESENT IN THE LOCK, PINNED TO A
    VERSION THAT SATISFIES THE DECLARED SPECIFIER.

That is deterministic. It does not consult the network. It does not depend on the day, the
platform, the Python version, or the pip-tools version. It cannot drift.

And it catches the defect that MATTERS -- the one roadmap 6.19 was actually trying to catch:
somebody edits requirements-api.txt and forgets to regenerate the lock, so the Docker API
image keeps installing the old pin. If `requirements-api.txt` says `fastapi==0.135.2` and the
lock says `fastapi==0.119.1`, this fails, loudly, forever, until it is fixed.

WHAT IT DELIBERATELY DOES *NOT* CHECK
-------------------------------------
That the lock's transitive dependencies are the newest available. **THAT IS NOT A DEFECT.**
A lock's entire purpose is to freeze a resolution so builds are reproducible. Upgrading is a
DELIBERATE, REVIEWED ACT -- `pip-compile --upgrade`, read the diff, run the suite, commit --
not something a gate should extort from you every time a transitive dependency ships a patch.

Conflating "old" with "wrong" is how you end up upgrading 22 packages at 3 a.m. to make a red
tick go green, three days before a paid training run.

USAGE
-----
    python scripts/check_lock_satisfies.py                 # exit 0 = satisfied
    python scripts/check_lock_satisfies.py --verbose       # print every checked requirement

Called by BOTH .github/workflows/ci.yml AND tests/unit/test_requirements_files_agree.py, so
the gate and the test can never disagree about what the rule is. (The console-ASCII work on
2026-07-14 shipped a gate and a repair tool with two different detectors; the gate flagged 21
things the tool could not fix. One detector, shared. -- roadmap 6.24)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQS = Path("requirements-api.txt")
LOCK = Path("requirements-api.lock")


def _canonical(name: str) -> str:
    """PEP 503 canonical form: lowercase, runs of -_. collapsed to a single -."""
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path) -> list:
    """Direct requirements from a requirements .txt (skips comments, blanks, -r/-c/flags)."""
    from packaging.requirements import Requirement

    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        out.append(Requirement(line))
    return out


def parse_lock(path: Path) -> dict[str, str]:
    """name -> pinned version, from a pip-compile lock. Ignores comments and `# via` lines."""
    from packaging.requirements import Requirement

    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        req = Requirement(line)
        # A lock line is always `name[extras]==X.Y.Z`; take the one pinned version.
        pinned = [s.version for s in req.specifier if s.operator == "=="]
        if pinned:
            pins[_canonical(req.name)] = pinned[0]
    return pins


def check(verbose: bool = False) -> list[str]:
    """Return a list of human-readable problems. Empty list = the lock satisfies the .txt."""
    problems: list[str] = []

    if not REQS.is_file():
        return [f"{REQS} does not exist"]
    if not LOCK.is_file():
        return [
            f"{LOCK} does not exist. The Docker API image installs it (Dockerfile:68-70) and "
            f"falls back to the loose pins in {REQS} without it. Generate it:\n"
            f"    pip-compile {REQS} -o {LOCK}"
        ]

    direct = parse_requirements(REQS)
    pins = parse_lock(LOCK)

    if not direct:
        return [f"parsed ZERO direct requirements out of {REQS} -- the parser is broken, and a "
                f"check that checks nothing passes for free"]
    if not pins:
        return [f"parsed ZERO pins out of {LOCK} -- the parser is broken"]

    for req in direct:
        name = _canonical(req.name)
        if name not in pins:
            problems.append(
                f"{req.name!r} is required by {REQS} but is ABSENT from {LOCK}. "
                f"The lock is stale: regenerate it."
            )
            continue

        version = pins[name]
        # prereleases=True: a pinned prerelease in the lock must still be judged against the
        # specifier rather than silently skipped.
        if not req.specifier.contains(version, prereleases=True):
            problems.append(
                f"{req.name}: {LOCK} pins {version}, which does NOT satisfy "
                f"{REQS}'s {req.name}{req.specifier}.\n"
                f"        This is the real staleness: the .txt was changed and the lock was "
                f"not regenerated, so the Docker API image is still installing {version}."
            )
        elif verbose:
            print(f"  OK  {req.name}{req.specifier}  -> lock pins {version}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--verbose", action="store_true", help="print every satisfied requirement")
    args = ap.parse_args()

    problems = check(verbose=args.verbose)

    if problems:
        print()
        print("=" * 78)
        print(f"  {LOCK} DOES NOT SATISFY {REQS}")
        print("=" * 78)
        for p in problems:
            print(f"    - {p}")
        print()
        print("  Regenerate the lock and commit the result:")
        print(f"      pip-compile {REQS} -o {LOCK}")
        print()
        print("  NOTE: this check is deterministic. It does NOT care whether the lock's")
        print("  transitive dependencies are the newest available -- being old is not a")
        print("  defect, it is the point of a lock. It fails only when the lock genuinely")
        print("  contradicts the requirements file.")
        return 1

    print(f"{LOCK} satisfies every direct requirement in {REQS} "
          f"({len(parse_requirements(REQS))} checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
