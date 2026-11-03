"""Check the dependency profiles: their locks offline, one installed environment against its lock, or derive an alignment constraint.

DEPENDENCY-PROFILES-1 (owner rulings 2026-10-31) and DRIFT-PROFILE-LOCK (2026-11-02). The profiles themselves -- sources, locks,
targets, install flags, departures, forbidden distributions -- are declared ONCE, in src/genomic_variant_classifier/deps/profiles.py.
Three modes, all deterministic, all refusing on the first malformed input:

    --offline
        For EVERY profile: each lock is exact, hash-bearing and fully accounted for by the repository's parser; the profile's sources are
        satisfied on every named target of each lock; every locked distribution that has a constraint is locked at exactly that
        version; no forbidden distribution is locked. A profile ALIGNED with another: its constraint file is exactly the derivation of
        the other profile's locks, and each declared departure still departs on every target. Then the generator toolchain's
        generated file pins exactly what its source names, and each build constraint equals the toolchain and the locks. No network, no
        installation. Run by the 'lockfile drift check' job.

    --environment OBSERVATION.json --profile NAME --target NAME --observation-sha256 HEX
        One environment, observed by deps/environment_observation.py, judged by deps/environment_contract.verify() against the lock
        of its profile and target, with pip and this project as the only auxiliary distributions and the profile's forbidden
        distributions refused. Run by scripts/install_locked_profile.py, right after it installs.

    --derive-constraints NAME
        Write the constraint file of an aligned profile (for 'drift': constraints-drift.txt) from the locks it is aligned with. Run
        before resolving that profile's locks; --offline then refuses any other content.

Exit 0: satisfied (or written). Exit 1: issues (every one printed). Exit 2: refused (the inputs could not be judged).

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from genomic_variant_classifier.deps import requirements_parse
from genomic_variant_classifier.deps.environment_contract import (
    ContractRefusal, constraint_issues, read_verified_json, source_issues, verify,
)
from genomic_variant_classifier.deps.model import DistributionName
from genomic_variant_classifier.deps.profiles import (
    PIP, POLICY, PROFILES, PROJECT, TARGETS, TOOLCHAIN_LOCK, TOOLCHAIN_SOURCE, Profile, profile,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


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


def _canonical(pins: dict) -> dict:
    return {str(DistributionName(k)): v for k, v in pins.items()}


def _exact(path: Path) -> dict:
    """A constraint file: every clause an exact pin without a marker, one per name."""
    out: dict = {}
    for r in _requirements(path, locked=False):
        spec = list(r.specifier)
        if len(spec) != 1 or spec[0].operator != "==" or r.url or r.marker is not None:
            raise ContractRefusal("{}: a constraint must be an exact pin without a marker: {}".format(path.name, r))
        key = str(DistributionName(r.name))
        if key in out:
            raise ContractRefusal("{}: two constraints for {}".format(path.name, key))
        out[key] = spec[0].version
    return out


def derive_constraints(root: Path, p: Profile) -> str:
    """The constraint file of an aligned profile: every pin of the aligned profile's locks active on any of their targets, less the
    declared departures. One version per distribution across every lock and target, or refused."""
    if not p.aligned_with:
        raise ContractRefusal("profile {} is not aligned with another profile; its constraints are measured, not derived".format(p.name))
    aligned = profile(p.aligned_with)
    versions: dict = {}
    for lock, targets in aligned.locks:
        for t in targets:
            for key, version in _canonical(_pins(root / lock, t)).items():
                if versions.setdefault(key, version) != version:
                    raise ContractRefusal("{} differs across the {} locks: {} and {}".format(key, aligned.name, versions[key], version))
    departed = {str(DistributionName(n)) for n, _ in p.departures}
    lines = ["# GENERATED by `python scripts/check_dependency_profiles.py --derive-constraints {}`. Do not edit.".format(p.name),
             "# Every pin active on any target of the {} locks ({}),".format(aligned.name, ", ".join(lock for lock, _ in aligned.locks)),
             "# less this profile's declared departures ({}; reasons in deps/profiles.py).".format(", ".join(sorted(departed))),
             "# The lockfile job refuses any other content. Author: Monzia Moodie"]
    lines += ["{}=={}".format(k, versions[k]) for k in sorted(versions) if k not in departed]
    return "\n".join(lines) + "\n"


def _profile_issues(root: Path, p: Profile) -> list:
    roots = [r for name in p.sources for r in _requirements(root / name, locked=False)]
    constraints = _exact(root / p.constraints) if p.aligned_with else {
        str(DistributionName(r.name)): list(r.specifier)[0].version for r in _requirements(root / p.constraints, locked=False)}
    forbidden = {str(DistributionName(n)): why for n, why in p.forbidden}
    lines = []
    for lock, targets in p.locks:
        for t in targets:
            pins = _pins(root / lock, t)
            for i in source_issues(roots, pins, TARGETS[t]) + constraint_issues(pins, constraints):
                lines.append("{} [{}] {}: {} {}".format(lock, t, i.code.value, i.distribution, i.detail))
            for key in sorted(forbidden.keys() & _canonical(pins).keys()):
                lines.append("{} [{}] forbidden: {} ({})".format(lock, t, key, forbidden[key]))
    if p.aligned_with:
        if (root / p.constraints).read_text(encoding="utf-8") != derive_constraints(root, p):
            lines.append("{} is not the derivation of the {} locks; run: python scripts/check_dependency_profiles.py "
                         "--derive-constraints {}".format(p.constraints, p.aligned_with, p.name))
        aligned = profile(p.aligned_with)
        for name, _ in p.departures:
            key = str(DistributionName(name))
            for lock, targets in p.locks:
                for t in targets:
                    mine = _canonical(_pins(root / lock, t)).get(key)
                    theirs = _canonical(_pins(root / aligned.lock_for(t), t)).get(key)
                    if mine is None or theirs is None or mine == theirs:
                        lines.append("{} [{}] departure {} does not depart: {} here, {} in {}".format(
                            lock, t, key, mine, theirs, aligned.lock_for(t)))
    return lines


def _toolchain_issues(root: Path) -> list:
    lines = []
    tool_src = {r.name.lower(): str(r.specifier) for r in _requirements(root / TOOLCHAIN_SOURCE, locked=False)}
    tool_pins = {k.lower(): "==" + v for k, v in _pins(root / TOOLCHAIN_LOCK, "windows-3.12").items()}
    if tool_src != tool_pins:
        lines.append("{} pins {} but {} names {}".format(TOOLCHAIN_LOCK, sorted(tool_pins.items()), TOOLCHAIN_SOURCE, sorted(tool_src.items())))
    for p in PROFILES.values():
        if not p.build_constraints:
            continue
        for r in _requirements(root / p.build_constraints, locked=False):
            want = str(r.specifier)
            if tool_pins.get(r.name.lower()) != want:
                lines.append("{}: {}{} differs from the generator toolchain's {}".format(p.build_constraints, r.name, want, tool_pins.get(r.name.lower())))
            for lock, targets in p.locks:
                locked = {k.lower(): v for k, v in _pins(root / lock, targets[0]).items()}.get(r.name.lower())
                if locked is not None and "==" + locked != want:
                    lines.append("{}: {}{} differs from {}'s {}".format(p.build_constraints, r.name, want, lock, locked))
    return lines


def offline(root: Path) -> list:
    lines = []
    for p in PROFILES.values():
        lines += _profile_issues(root, p)
    return lines + _toolchain_issues(root)


def environment(root: Path, profile_name: str, observation: Path, sha256: str, target: str) -> list:
    p = profile(profile_name)
    if target not in p.targets:
        raise ContractRefusal("profile {} has no lock for target {}".format(p.name, target))
    roots = [r for name in p.sources for r in _requirements(root / name, locked=False)]
    verdict = verify(read_verified_json(observation, sha256), _pins(root / p.lock_for(target), target), auxiliary=dict([PIP, PROJECT]),
                     roots=roots, locked_target=TARGETS[target], python_policy=POLICY[target], forbidden=[n for n, _ in p.forbidden])
    for name, (was, now) in verdict.provenance_changes.items():
        print("  host differs from the written target (recorded, not judged): {}: {} -> {}".format(name, was, now))
    return ["{}: {} {}".format(i.code.value, i.distribution, i.detail) for i in verdict.issues]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--environment", type=Path, metavar="OBSERVATION_JSON")
    mode.add_argument("--derive-constraints", choices=sorted(n for n, p in PROFILES.items() if p.aligned_with), metavar="PROFILE")
    ap.add_argument("--profile", choices=sorted(PROFILES))
    ap.add_argument("--observation-sha256")
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    try:
        if args.offline:
            problems = offline(args.root)
            label = "every profile's locks and sources ({})".format(", ".join(sorted(PROFILES)))
        elif args.derive_constraints:
            p = profile(args.derive_constraints)
            text = derive_constraints(args.root, p)
            out = args.root / p.constraints
            with out.open("w", encoding="ascii", newline="\n") as handle:
                handle.write(text)
            print("written {} ({} constraints) sha256 {}".format(
                out, sum(1 for line in text.splitlines() if not line.startswith("#")), hashlib.sha256(text.encode("ascii")).hexdigest()))
            return 0
        else:
            if not args.profile or not args.target or not args.observation_sha256:
                ap.error("--environment needs --profile, --target and --observation-sha256")
            problems = environment(args.root, args.profile, args.environment, args.observation_sha256, args.target)
            label = "the environment against the {} lock for {}".format(args.profile, args.target)
    except (ContractRefusal, KeyError) as exc:
        print("REFUSED: {}".format(exc))
        return 2
    for line in problems:
        print("  " + line)
    print("{}: {}".format(label, "SATISFIED" if not problems else "{} issue(s)".format(len(problems))))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
