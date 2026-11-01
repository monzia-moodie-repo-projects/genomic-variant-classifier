"""Check the training-and-test dependency profile: its locks offline, or one installed environment against its lock.

DEPENDENCY-PROFILES-1 (owner rulings 2026-10-31). Two modes, both deterministic and both refusing on the first malformed input:

    --offline
        Both locks are exact, hash-bearing and fully accounted for by the repository's parser; requirements.txt and
        requirements-dev.txt are satisfied on every named target of each lock; every locked distribution that has a measured
        constraint is locked at exactly that version; the generator toolchain's generated file pins exactly what its source names.
        No network, no installation. Run by the 'lockfile drift check' job.

    --environment OBSERVATION.json --target NAME --observation-sha256 HEX
        One environment, observed by deps/environment_observation.py, judged by deps/environment_contract.verify() against the lock
        of its target, with pip and this project as the only auxiliary distributions. Run by the 'pytest' job BEFORE the suite.

Exit 0: satisfied. Exit 1: issues (every one printed). Exit 2: refused (the inputs could not be judged).

TARGETS are written out, not taken from whichever host runs the check: a lock describes a target, and a host's own fields are only
evidence about that host. Each target is the MEASURED marker environment of the interpreter that resolved its lock, because
pip-compile evaluates markers on that interpreter:
    linux-3.11    /usr/bin/python3.11 (3.11.17) of the development sandbox, the interpreter of the Linux lock's 3.11 resolution
                  (packaging.markers.default_environment(), read 2026-11-01);
    linux-3.12    /usr/bin/python3.12 (3.12.3) of the same sandbox, the interpreter of the 3.12 resolution (the same values are in
                  the observation of the locked environment built with it, 2026-11-01);
    windows-3.12  the owner's .venv312 (3.12.10), where the Windows lock was resolved (observation a4fc5829, 2026-10-31).
A runner's own kernel release, kernel version and Python patch release differ from these: verify() RECORDS them as host provenance
and judges only the stable target family, the Python policy and every marker outcome.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from genomic_variant_classifier.deps import requirements_parse
from genomic_variant_classifier.deps.environment_contract import (
    ContractRefusal, constraint_issues, read_verified_json, source_issues, verify,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("requirements.txt", "requirements-dev.txt")
CONSTRAINTS = "constraints-observed-venv312.txt"
TOOLCHAIN_SOURCE, TOOLCHAIN_LOCK = "requirements-generator-toolchain.in", "requirements-generator-toolchain.txt"
BUILD_CONSTRAINTS = "build-constraints-train-test.txt"
PROJECT = ("genomic-variant-classifier", "0.1.0")
PIP = ("pip", "26.1.2")

_LINUX = {"implementation_name": "cpython", "os_name": "posix", "platform_machine": "x86_64", "platform_release": "6.18.44-fc-v185",
          "platform_system": "Linux", "platform_version": "#1 SMP PREEMPT_DYNAMIC @0", "platform_python_implementation": "CPython",
          "sys_platform": "linux"}
TARGETS = {
    "linux-3.11": dict(_LINUX, implementation_version="3.11.17", python_full_version="3.11.17", python_version="3.11"),
    "linux-3.12": dict(_LINUX, implementation_version="3.12.3", python_full_version="3.12.3", python_version="3.12"),
    "windows-3.12": {"implementation_name": "cpython", "implementation_version": "3.12.10", "os_name": "nt", "platform_machine": "AMD64",
                     "platform_release": "11", "platform_system": "Windows", "platform_version": "10.0.29698",
                     "platform_python_implementation": "CPython", "python_full_version": "3.12.10", "python_version": "3.12",
                     "sys_platform": "win32"},
}
LOCKS = {"requirements-train-test-linux.lock": ("linux-3.11", "linux-3.12"), "requirements-train-test-windows.lock": ("windows-3.12",)}
POLICY = {"linux-3.11": "==3.11.*", "linux-3.12": "==3.12.*", "windows-3.12": "==3.12.*"}


def _requirements(path: Path, *, locked: bool) -> list:
    """Clauses through the repository's own parser; a lock must reconcile, be exact and carry hashes on every record.

    A lock clause MAY carry a marker: pip-compile copies a declaration's marker into the lock (measured 2026-11-01:
    'pybigwig==0.3.29 ; sys_platform != "win32"' in the Linux lock, from requirements.txt). Which clauses apply is decided per target
    by `_pins`, never by refusing the marker.
    """
    try:
        parsed = requirements_parse.parse_requirements_file(path)
    except (requirements_parse.MeasurementIntegrityError, OSError) as exc:
        raise ContractRefusal("{} cannot be read as requirements: {}".format(path.name, exc)) from exc
    audit = parsed.audit.as_dict()
    if not audit["reconciles"] or audit["parse_failures"]:
        raise ContractRefusal("{} does not parse cleanly: {}".format(path.name, audit))
    if parsed.directives:
        raise ContractRefusal("{} holds directives the profile does not admit: {}".format(path.name, [d.raw for d in parsed.directives]))
    out = []
    for clauses in parsed.clauses.values():
        for clause in clauses:
            try:
                req = Requirement(re.sub(r"\s*--hash=\S+", "", clause.raw).strip())
            except InvalidRequirement as exc:
                raise ContractRefusal("{}: {}".format(path.name, exc)) from exc
            if locked:
                spec = list(req.specifier)
                if len(spec) != 1 or spec[0].operator != "==" or req.url:
                    raise ContractRefusal("{}: not an exact pin: {}".format(path.name, req))
                if not clause.hashes:
                    raise ContractRefusal("{}: {} carries no hash".format(path.name, req.name))
            out.append(req)
    if not out:
        raise ContractRefusal("{} holds no requirement".format(path.name))
    return out


def _pins(path: Path, target: str) -> dict:
    """The lock's pins ACTIVE on `target`; two active clauses for one name are refused."""
    env = dict(TARGETS[target], extra="")
    pins: dict = {}
    for r in _requirements(path, locked=True):
        if r.marker is None or r.marker.evaluate(env):
            if r.name.lower() in {k.lower() for k in pins}:
                raise ContractRefusal("{}: two clauses for {} are active on {}".format(path.name, r.name, target))
            pins[r.name] = list(r.specifier)[0].version
    return pins


def offline(root: Path) -> list:
    roots = [r for name in SOURCES for r in _requirements(root / name, locked=False)]
    constraints = {r.name: list(r.specifier)[0].version for r in _requirements(root / CONSTRAINTS, locked=False)}
    lines = []
    for lock, targets in LOCKS.items():
        for t in targets:
            pins = _pins(root / lock, t)
            for i in source_issues(roots, pins, TARGETS[t]) + constraint_issues(pins, constraints):
                lines.append("{} [{}] {}: {} {}".format(lock, t, i.code.value, i.distribution, i.detail))
    tool_src = {r.name.lower(): str(r.specifier) for r in _requirements(root / TOOLCHAIN_SOURCE, locked=False)}
    tool_pins = {k.lower(): "==" + v for k, v in _pins(root / TOOLCHAIN_LOCK, "windows-3.12").items()}
    for r in _requirements(root / BUILD_CONSTRAINTS, locked=False):
        want = str(r.specifier)
        if tool_pins.get(r.name.lower()) != want:
            lines.append("{}: {}{} differs from the generator toolchain's {}".format(BUILD_CONSTRAINTS, r.name, want, tool_pins.get(r.name.lower())))
        for lock, targets in LOCKS.items():
            locked = {k.lower(): v for k, v in _pins(root / lock, targets[0]).items()}.get(r.name.lower())
            if locked is not None and "==" + locked != want:
                lines.append("{}: {}{} differs from {}'s {}".format(BUILD_CONSTRAINTS, r.name, want, lock, locked))
    if tool_src != tool_pins:
        lines.append("{} pins {} but {} names {}".format(TOOLCHAIN_LOCK, sorted(tool_pins.items()), TOOLCHAIN_SOURCE, sorted(tool_src.items())))
    return lines


def environment(root: Path, observation: Path, sha256: str, target: str) -> list:
    lock = next(name for name, targets in LOCKS.items() if target in targets)
    roots = [r for name in SOURCES for r in _requirements(root / name, locked=False)]
    verdict = verify(read_verified_json(observation, sha256), _pins(root / lock, target), auxiliary=dict([PIP, PROJECT]), roots=roots,
                     locked_target=TARGETS[target], python_policy=POLICY[target])
    for name, (was, now) in verdict.provenance_changes.items():
        print("  host differs from the written target (recorded, not judged): {}: {} -> {}".format(name, was, now))
    return ["{}: {} {}".format(i.code.value, i.distribution, i.detail) for i in verdict.issues]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--environment", type=Path, metavar="OBSERVATION_JSON")
    ap.add_argument("--observation-sha256")
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    try:
        if args.offline:
            problems = offline(args.root)
            label = "both locks and their sources"
        else:
            if not args.target or not args.observation_sha256:
                ap.error("--environment needs --target and --observation-sha256")
            problems = environment(args.root, args.environment, args.observation_sha256, args.target)
            label = "the environment against the {} lock".format(args.target)
    except ContractRefusal as exc:
        print("REFUSED: {}".format(exc))
        return 2
    for line in problems:
        print("  " + line)
    print("{}: {}".format(label, "SATISFIED" if not problems else "{} issue(s)".format(len(problems))))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
