"""Test bootstrap (tests/conftest.py).

Makes the repo-local `scripts/` directory importable during tests regardless of CWD or
environment, so tests can `import clean_cohort` (and other script-level modules) the same way
locally and in CI.

Root cause this fixes: CI runs `pytest tests/unit/ -x -q` from the repo root, where `scripts/` is
not a package and not on sys.path; `from clean_cohort import run_clean` raised ModuleNotFoundError
on the GitHub runner (CI run #233) even though it imported in the local dev environment. Walking
ancestors (rather than hardcoding depth) keeps this correct if the tests tree is ever moved.
"""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _anc in (_here.parent, *_here.parents):
    _scripts = _anc / "scripts"
    if _scripts.is_dir() and (_scripts / "clean_cohort.py").is_file():
        for _p in (str(_scripts), str(_anc)):  # scripts dir + repo root
            if _p not in sys.path:
                sys.path.insert(0, _p)
        break
