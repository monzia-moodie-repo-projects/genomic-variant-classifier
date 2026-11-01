"""Does an installed Python environment match its lock EXACTLY? One verdict, with every reason.

DEPENDENCY-PROFILES-1 (owner rulings 2026-10-31)
================================================
A lock is a claim about what WILL be installed; an environment observation is a record of what IS installed. This module compares
the two and refuses to call them equal unless every distribution is present exactly once at exactly the locked version, nothing is
present that the lock and the declared auxiliary tools do not name, every installed distribution's own requirements (including the
extras a root requirement selects) are satisfied by what is installed, and the observed interpreter is the target the lock was
generated for.

Four separate checks, because each catches what the others cannot:

    inventory     missing, wrong version, unexpected, and DUPLICATE metadata. A same-version duplicate is refused too: two records for
                  one name is an ambiguous installation even when both say the same version. MEASURED 2026-10-31 on the owner's
                  .venv312: genomic-variant-classifier 0.1.0 twice -- the editable install's dist-info AND src/...egg-info, which
                  the editable install's path entry makes visible (EGG-INFO-SHADOW-1). `pip check` passed on that environment.
    closure       every installed distribution's Requires-Dist, evaluated with the extras actually selected, is satisfied. A
                  --no-deps install of an incomplete lock passes a version comparison and fails here.
    target        three different questions, never one dictionary comparison (owner ruling 2026-10-31):
                      provenance  the host's own fields (an operating-system build) are RECORDED, never a reason to refuse;
                      eligibility the stable target family (system, machine, implementation, Python minor) must match, and the
                                  Python version must satisfy an EXPLICIT policy;
                      semantics   every requirement marker in the closure is evaluated on the locked target AND on the observed
                                  host; a different outcome refuses, because then the lock no longer describes this host.
    evidence      an observation is read only if its bytes have the SHA-256 the caller expects, with duplicate JSON keys and
                  non-finite numbers refused.

WHAT THIS DOES NOT ESTABLISH: the integrity of wheel bytes (pip's --require-hashes does that at installation), compatibility of
compiled wheels with the host's processor, libc or drivers, or any behaviour of the scientific code. A satisfied contract is a
precondition for those claims, not one of them.

Author: Monzia Moodie
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from genomic_variant_classifier.deps.model import DistributionName

logger = logging.getLogger(__name__)

OBSERVATION_SCHEMA = "gvc.environment-observation/1"

#: Fields whose change means a DIFFERENT target, not a patched host of the same one.
STABLE_TARGET_FIELDS = ("implementation_name", "os_name", "platform_machine", "platform_system",
                        "platform_python_implementation", "python_version", "sys_platform")


class ContractRefusal(ValueError):
    """The inputs cannot be judged (malformed, incomplete, or not what the caller vouched for). Never a verdict."""


class IssueCode(str, Enum):
    DUPLICATE_METADATA = "duplicate_metadata"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    UNEXPECTED = "unexpected"
    FORBIDDEN = "forbidden"
    DEPENDENCY_MISSING = "dependency_missing"
    DEPENDENCY_VERSION = "dependency_version"
    UNDECLARED_EXTRA = "undeclared_extra"
    REQUIRES_PYTHON = "requires_python"
    DIRECT_URL_DEPENDENCY = "direct_url_dependency"
    TARGET_FAMILY_CHANGED = "target_family_changed"
    PYTHON_OUTSIDE_POLICY = "python_outside_policy"
    MARKER_OUTCOME_CHANGED = "marker_outcome_changed"
    NOT_A_VIRTUAL_ENVIRONMENT = "not_a_virtual_environment"
    SYSTEM_SITE_PACKAGES = "system_site_packages"
    PIP_CHECK_FAILED = "pip_check_failed"
    SOURCE_MISSING = "source_missing"
    SOURCE_CONFLICT = "source_conflict"
    CONSTRAINT_VIOLATED = "constraint_violated"


@dataclass(frozen=True)
class Issue:
    code: IssueCode
    distribution: str
    detail: str = ""


@dataclass(frozen=True)
class ContractVerdict:
    """All issues, in a stable order, and the host facts recorded but not judged."""
    issues: tuple
    provenance_changes: dict = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"satisfied": self.satisfied,
                "issues": [{"code": i.code.value, "distribution": i.distribution, "detail": i.detail} for i in self.issues],
                "provenance_changes": dict(self.provenance_changes),
                "scope": "inventory, dependency closure, target and marker semantics; NOT wheel integrity, binary "
                         "compatibility or scientific behaviour"}


def _refuse(condition: bool, message: str) -> None:
    if not condition:
        raise ContractRefusal(message)


def _version(text, what: str) -> Version:
    _refuse(isinstance(text, str) and bool(text), "{}: version must be non-empty text".format(what))
    try:
        return Version(text)
    except InvalidVersion as exc:
        raise ContractRefusal("{}: invalid version {!r}".format(what, text)) from exc


def read_verified_json(path, expected_sha256: str):
    """Read evidence only when its bytes have exactly the SHA-256 the caller vouches for."""
    _refuse(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_sha256), "a full lowercase SHA-256 is required")
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    _refuse(actual == expected_sha256, "evidence digest mismatch: {} is {}, not {}".format(path, actual, expected_sha256))

    def pairs(items):
        out = {}
        for key, value in items:
            _refuse(key not in out, "duplicate JSON key {!r}".format(key))
            out[key] = value
        return out

    def constant(value):
        raise ContractRefusal("non-finite JSON constant {}".format(value))

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def complete_marker_environment(env) -> dict:
    """A target named in full; the running host is never used to fill a gap."""
    _refuse(isinstance(env, dict) and set(env) == set(default_environment()),
            "a complete explicit marker environment is required (keys: {})".format(sorted(default_environment())))
    _refuse(all(isinstance(v, str) and v for v in env.values()), "every marker value must be non-empty text")
    full = _version(env["python_full_version"], "python_full_version")
    _refuse(env["python_version"] == "{}.{}".format(full.major, full.minor), "python_version disagrees with python_full_version")
    return dict(env)


def _groups(distributions) -> dict:
    _refuse(isinstance(distributions, list) and distributions, "the observation lists no distributions")
    groups: dict = {}
    for row in distributions:
        _refuse(isinstance(row, dict), "a distribution row must be an object")
        name = row.get("name")
        _refuse(isinstance(name, str) and bool(name), "a distribution row has no name")
        _version(row.get("version"), name)
        groups.setdefault(str(DistributionName(name)), []).append(row)
    return groups


def inventory_issues(distributions, expected: dict, *, forbidden=()) -> list:
    """Exact equality of the installed set with `expected` (name -> version). Duplicates are refused even at equal versions."""
    groups = _groups(distributions)
    pins: dict = {}
    for raw, version in expected.items():
        key = str(DistributionName(raw))
        _refuse(key not in pins, "two expected entries for {}".format(key))
        pins[key] = _version(version, key)
    issues = []
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) > 1:
            issues.append(Issue(IssueCode.DUPLICATE_METADATA, key,
                                "; ".join(str(r.get("metadata_path")) for r in rows)))
        if key not in pins:
            issues.append(Issue(IssueCode.UNEXPECTED, key, ", ".join(r["version"] for r in rows)))
        elif any(Version(r["version"]) != pins[key] for r in rows):
            issues.append(Issue(IssueCode.VERSION_MISMATCH, key,
                                "expected {}, installed {}".format(pins[key], ", ".join(r["version"] for r in rows))))
    for key in sorted(pins.keys() - groups.keys()):
        issues.append(Issue(IssueCode.MISSING, key, "expected {}".format(pins[key])))
    for key in sorted({str(DistributionName(f)) for f in forbidden} & groups.keys()):
        issues.append(Issue(IssueCode.FORBIDDEN, key))
    return issues


def closure_issues(distributions, roots, target) -> tuple:
    """(issues, evaluated): every installed distribution's requirements, with the extras its requirers select, are met.

    `evaluated` is every (requirement text, extra) pair whose marker this walk evaluated -- exactly the semantics the target
    check must re-evaluate on the host, no more (the first version re-evaluated every marker under every extra name in the
    environment: 14 s on the owner's 251 distributions) and no less.
    """
    target = complete_marker_environment(target)
    groups = _groups(distributions)
    installed = {k: v[0] for k, v in groups.items()}       # duplicates are the inventory check's verdict, not this one's
    contexts = {k: {""} for k in installed}
    pending = [(k, "") for k in installed]
    found: set = set()
    evaluated: set = set()

    def visit(req: Requirement, parent: str) -> None:
        key = str(DistributionName(req.name))
        if req.url:
            found.add((IssueCode.DIRECT_URL_DEPENDENCY, parent, str(req)))
            return
        if key not in installed:
            found.add((IssueCode.DEPENDENCY_MISSING, parent, str(req)))
            return
        if not req.specifier.contains(installed[key]["version"], prereleases=True):
            found.add((IssueCode.DEPENDENCY_VERSION, parent, "{} (installed {})".format(req, installed[key]["version"])))
        declared = {str(DistributionName(e)) for e in installed[key].get("provides_extra") or []}
        for extra in req.extras:
            normal = str(DistributionName(extra))
            if normal not in declared:
                found.add((IssueCode.UNDECLARED_EXTRA, key, normal))
            if normal not in contexts[key]:
                contexts[key].add(normal)
                pending.append((key, normal))

    for raw in roots:
        req = raw if isinstance(raw, Requirement) else Requirement(raw)
        evaluated.add((str(req), ""))
        if req.marker is None or req.marker.evaluate(dict(target, extra="")):
            visit(req, "<root>")
    seen: set = set()
    while pending:
        key, extra = pending.pop()
        if (key, extra) in seen:
            continue
        seen.add((key, extra))
        row = installed[key]
        if row.get("requires_python") and not SpecifierSet(row["requires_python"]).contains(
                target["python_full_version"], prereleases=True):
            found.add((IssueCode.REQUIRES_PYTHON, key, row["requires_python"]))
        for text in row.get("requires_dist") or []:
            req = Requirement(text)
            evaluated.add((text, extra))
            if req.marker is None or req.marker.evaluate(dict(target, extra=extra)):
                visit(req, key)
    return [Issue(c, p, d) for c, p, d in sorted(found, key=lambda t: (t[0].value, t[1], t[2]))], evaluated


def target_issues(locked_target, observed, evaluated, *, python_policy: str) -> tuple:
    """(issues, provenance_changes): eligibility and marker semantics judged; host-only differences recorded.

    `evaluated`: the (requirement text, extra) pairs the closure walk evaluated on the locked target.
    """
    locked = complete_marker_environment(locked_target)
    host = complete_marker_environment(observed)
    policy = SpecifierSet(python_policy)
    _refuse(bool(str(policy)), "an explicit Python version policy is required")
    issues = [Issue(IssueCode.TARGET_FAMILY_CHANGED, f, "{} -> {}".format(locked[f], host[f]))
              for f in STABLE_TARGET_FIELDS if locked[f] != host[f]]
    if not policy.contains(host["python_full_version"], prereleases=False):
        issues.append(Issue(IssueCode.PYTHON_OUTSIDE_POLICY, "python", "{} not in {}".format(host["python_full_version"], policy)))
    for text, extra in sorted(evaluated):
        req = Requirement(text)
        if req.marker is not None and req.marker.evaluate(dict(locked, extra=extra)) != req.marker.evaluate(dict(host, extra=extra)):
            issues.append(Issue(IssueCode.MARKER_OUTCOME_CHANGED, str(DistributionName(req.name)), "{} [extra={!r}]".format(text, extra)))
    changes = {k: [locked[k], host[k]] for k in sorted(locked) if locked[k] != host[k]}
    return issues, changes


def source_issues(roots, lock_pins: dict, target) -> list:
    """Every declared requirement active on `target` is in the lock, at a version the declaration allows."""
    target = complete_marker_environment(target)
    pins = {str(DistributionName(k)): _version(v, k) for k, v in lock_pins.items()}
    issues = []
    for raw in roots:
        req = raw if isinstance(raw, Requirement) else Requirement(raw)
        _refuse(req.url is None, "a direct-URL declaration needs its own artifact admission: {}".format(req))
        if req.marker is not None and not req.marker.evaluate(dict(target, extra="")):
            continue
        key = str(DistributionName(req.name))
        if key not in pins:
            issues.append(Issue(IssueCode.SOURCE_MISSING, key, str(req)))
        elif not req.specifier.contains(pins[key], prereleases=True):
            issues.append(Issue(IssueCode.SOURCE_CONFLICT, key, "{} but locked {}".format(req, pins[key])))
    return issues


def constraint_issues(lock_pins: dict, constraints: dict) -> list:
    """Every locked distribution that has a measured constraint is locked at exactly that version (none silently substituted)."""
    want = {str(DistributionName(k)): _version(v, k) for k, v in constraints.items()}
    return [Issue(IssueCode.CONSTRAINT_VIOLATED, key, "constrained {} but locked {}".format(want[key], v))
            for key, v in sorted((str(DistributionName(k)), _version(v, k)) for k, v in lock_pins.items())
            if key in want and v != want[key]]


def verify(observation: dict, lock_pins: dict, *, auxiliary: dict, roots, locked_target: dict, python_policy: str,
           forbidden=()) -> ContractVerdict:
    """The whole contract for one observed environment against one lock (plus explicitly versioned auxiliary tools)."""
    _refuse(isinstance(observation, dict) and observation.get("schema") == OBSERVATION_SCHEMA,
            "not a {} observation".format(OBSERVATION_SCHEMA))
    expected = dict(lock_pins)
    for name, version in auxiliary.items():
        _refuse(str(DistributionName(name)) not in {str(DistributionName(k)) for k in expected},
                "auxiliary {} is already locked; it cannot be both".format(name))
        expected[name] = version
    dists = observation["distributions"]
    issues = list(inventory_issues(dists, expected, forbidden=forbidden))
    interp = observation.get("interpreter") or {}
    if not interp.get("is_virtual_environment"):
        issues.append(Issue(IssueCode.NOT_A_VIRTUAL_ENVIRONMENT, "interpreter", str(interp.get("prefix"))))
    cfg = (interp.get("pyvenv_cfg") or "").lower().replace(" ", "")
    if "include-system-site-packages=false" not in cfg:
        issues.append(Issue(IssueCode.SYSTEM_SITE_PACKAGES, "interpreter", "pyvenv.cfg does not exclude system site-packages"))
    pip_check = observation.get("pip_check") or {}
    if pip_check.get("exit_code") != 0:
        issues.append(Issue(IssueCode.PIP_CHECK_FAILED, "pip", str(pip_check.get("stdout", "")).strip()[:2000]))
    c_issues, evaluated = closure_issues(dists, roots, locked_target)
    issues.extend(c_issues)
    t_issues, changes = target_issues(locked_target, observation["marker_environment"], evaluated, python_policy=python_policy)
    issues.extend(t_issues)
    verdict = ContractVerdict(issues=tuple(issues), provenance_changes=changes)
    logger.info("environment contract: %s, %d issue(s), %d host-only difference(s)",
                "satisfied" if verdict.satisfied else "NOT satisfied", len(verdict.issues), len(changes))
    return verdict
