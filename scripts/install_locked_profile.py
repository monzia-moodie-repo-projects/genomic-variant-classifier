"""Install one dependency profile EXACTLY from its lock into a NEW virtual environment, then prove the result IS the lock.

    python scripts/install_locked_profile.py --profile drift --target linux-3.12 --venv /path/to/NEW_ENV

ONE procedure for every place a locked profile is installed -- the 'pytest' and 'drift monitor' jobs, the monthly drift monitor and
a person's own machine -- so they cannot drift apart (DEPENDENCY-PROFILES-1, DRIFT-PROFILE-LOCK). Standard library only: it runs on
the BASE interpreter, before any environment exists. In order, each step refusing on failure:

    0. refuse an existing --venv or observation (an existing environment is never modified), a target the profile has no lock for,
       and a missing input -- before anything is created;
    1. `python -m venv` with THIS interpreter;
    2. pip, from the generator toolchain's hashed record, with --require-hashes --no-deps;
    3. the target's lock, with --require-hashes --no-deps --no-cache-dir and the profile's own flags (deps/profiles.py):
       --only-binary=:all: for drift, --build-constraint build-constraints-train-test.txt for train-test;
    4. this project, editable, with --no-deps --no-build-isolation (its build backend is the locked setuptools);
    5. an observation of the new environment (deps/environment_observation.py, run by the NEW interpreter with -I -B);
    6. the environment contract (scripts/check_dependency_profiles.py --environment), run by the NEW interpreter: inventory,
       dependency closure, target, pip check and the profile's forbidden distributions.
Every Python it starts runs ISOLATED (-I): no PYTHONPATH, no user site-packages, no current directory on the import path, so
nothing outside the new environment and this checkout can change what is installed or what is judged.

Exit 0: installed and proven. Exit 1: a step failed (the output says which). Exit 2: refused before anything was created.
The observation is written next to the environment as NEW_ENV.observation.json and kept: it is the evidence of what was installed.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The declaration is imported from the checkout by the BASE interpreter: without this, that import leaves __pycache__ folders in the
# checkout (measured 2026-11-02 in an emulation of the drift job: deps/__pycache__ with two .pyc files).
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from genomic_variant_classifier.deps.profiles import PIP, PROFILES, TARGETS, TOOLCHAIN_LOCK  # noqa: E402

logger = logging.getLogger(__name__)


def pip_record(toolchain_text: str) -> str:
    """The pip record of the toolchain lock: its pin line and every continuation line after it, exactly as written."""
    lines = toolchain_text.splitlines()
    head = "{}=={} \\".format(*PIP)
    starts = [i for i, line in enumerate(lines) if line == head]
    if len(starts) != 1:
        raise ValueError("{} must hold exactly one line {!r}; found {}".format(TOOLCHAIN_LOCK, head, len(starts)))
    record = [lines[starts[0]]]
    for line in lines[starts[0] + 1:]:
        record.append(line)
        if not line.endswith("\\"):
            break
    if not all(r.strip().startswith("--hash=sha256:") for r in record[1:]) or len(record) < 2:
        raise ValueError("the pip record of {} is not a pin followed only by hashes: {}".format(TOOLCHAIN_LOCK, record))
    return "\n".join(record) + "\n"


def _run(step: str, command: list, env: dict) -> None:
    print("--- {}: {}".format(step, " ".join(str(c) for c in command)), flush=True)
    done = subprocess.run([str(c) for c in command], env=env)
    if done.returncode != 0:
        raise RuntimeError("{} failed with exit {}".format(step, done.returncode))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--venv", required=True, type=Path, help="a NEW folder for the environment")
    args = ap.parse_args(argv)
    p = PROFILES[args.profile]
    venv = args.venv.resolve()
    observation = venv.parent / (venv.name + ".observation.json")
    if args.target not in p.targets:
        print("REFUSED: profile {} has no lock for target {}; its targets are {}".format(p.name, args.target, list(p.targets)))
        return 2
    for path in (venv, observation):
        if path.exists():
            print("REFUSED: {} already exists; an existing environment is never modified".format(path))
            return 2
    lock = ROOT / p.lock_for(args.target)
    build = ["--build-constraint", ROOT / p.build_constraints] if p.build_constraints else []
    for needed in (lock, ROOT / TOOLCHAIN_LOCK, *build[1:]):
        if not needed.is_file():
            print("REFUSED: {} is missing".format(needed))
            return 2
    try:
        record = pip_record((ROOT / TOOLCHAIN_LOCK).read_text(encoding="utf-8"))
    except ValueError as exc:
        print("REFUSED: {}".format(exc))
        return 2
    env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print("profile {}, target {}, lock {} (sha256 {})".format(p.name, args.target, lock.name, hashlib.sha256(lock.read_bytes()).hexdigest()))
    print("base interpreter {} ({})".format(sys.executable, sys.version.split()[0]))
    try:
        _run("1/6 virtual environment", [sys.executable, "-I", "-m", "venv", venv], env)
        record_file = venv / "gvc-pip-record.txt"
        record_file.write_text(record, encoding="ascii", newline="\n")
        try:
            _run("2/6 pip from the toolchain record", [python, "-I", "-m", "pip", "install", "--require-hashes", "--no-deps", "--no-cache-dir",
                                                      "-r", record_file], env)
        finally:
            record_file.unlink()
        _run("3/6 the lock", [python, "-I", "-m", "pip", "install", "--require-hashes", "--no-deps", "--no-cache-dir", *p.install_flags, *build,
                               "-r", lock], env)
        _run("4/6 the project, editable", [python, "-I", "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-cache-dir", "-e", ROOT],
             env)
        _run("5/6 observation", [python, "-I", "-B", "-m", "genomic_variant_classifier.deps.environment_observation", observation], env)
        digest = hashlib.sha256(observation.read_bytes()).hexdigest()
        _run("6/6 the environment IS the lock", [python, "-I", "-B", ROOT / "scripts" / "check_dependency_profiles.py", "--environment", observation,
                                                 "--profile", p.name, "--target", args.target, "--observation-sha256", digest,
                                                 "--root", ROOT], env)
    except (RuntimeError, OSError) as exc:
        print("FAILED: {}".format(exc))
        return 1
    print("INSTALLED AND PROVEN: {} ({} on {}); observation {} sha256 {}".format(venv, p.name, args.target, observation, digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
