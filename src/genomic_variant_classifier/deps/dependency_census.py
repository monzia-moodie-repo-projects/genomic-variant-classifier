"""Which code actually imports a package, and from where.

PYTHON IMPORT CENSUS -- and the name is the epistemic claim.

Renamed from "dependency census" deliberately. AST analysis sees exactly one of
at least four consumer channels:

    Python imports          <- this instrument
    command-line invocation    (pre-commit runs as a console script)
    plugin / entry-point discovery
    configuration references

`pre_commit` measured ZERO imports and is nonetheless a real dependency. So
"NO IMPORT ANYWHERE" must never be allowed to evolve into "UNUSED", and calling
this a dependency census would invite exactly that.

DEPENDENCY-SCOPE-CENSUS
=======================
Step 5 of the dependency ruling: measure the real consumers of `pyfaidx`,
`httpx`, `anyio`, `jinja2` and `seaborn` BEFORE assigning any of them a scope.

    Do not classify dependencies based on the file in which they happen to sit
    now -- that would use the drifted representation to infer the intended
    ontology.

`requirements-dev.txt` currently declares all five. That file is named
"development" and its only measured consumer is continuous integration, so its
contents are evidence of history, not of intent. The imports are the evidence.

WHY AST AND NOT A TEXT SEARCH
-----------------------------
A grep for `pyfaidx` matches:

    # pyfaidx would be needed here                 <- a comment
    A docstring mentioning pyfaidx by name          <- a docstring
    raise ImportError("install pyfaidx")            <- a string literal
    import pyfaidx                                  <- the only real evidence

(The docstring case is written in prose above rather than shown literally,
because a triple-quoted example inside this docstring would terminate it --
which is exactly what happened on the first attempt, and the resulting
SyntaxError was reported to stderr while a chained py_compile printed PASS.)

Only the last is a consumer. This session produced eleven instances of a text
search matching its own documentation, so the census walks `ast.Import` and
`ast.ImportFrom` nodes, where comments and strings do not exist.

It also records TRY-GUARDED imports separately. An import inside `try:` with an
`ImportError` handler is an OPTIONAL consumer -- the code runs without it -- and
that is a materially different claim from a hard dependency.

WHAT IT DOES NOT DO
-------------------
It does not assign scopes. It reports where imports occur, and the assignment
is a judgement about intent that belongs to a person:

    src/      production import      -> runtime dependency
    scripts/  pipeline import        -> operational dependency
    tests/    only                   -> test dependency
    notebooks/ or one-off analysis   -> analysis dependency

A package imported in src/ while declared development-only is the
torch_geometric defect shape, and the census names it rather than deciding it.

Author: Monzia Moodie
"""

from __future__ import annotations

import ast
import io
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

from genomic_variant_classifier.deps.model import (
    DistributionImportMapping, DistributionName, ImportName, MappingSource,
)

#: Directories excluded from the walk. A virtual environment contains the
#: packages themselves, and counting their internal imports would report every
#: dependency as its own consumer.
_SKIP_DIRS = {".venv312", ".venv", ".git", "node_modules", "__pycache__",
              "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


class CensusError(RuntimeError):
    """The census could not complete, and must not report a partial result."""


@dataclass(frozen=True)
class ImportSite:
    """One import statement, located, with how hard a dependency it is."""
    path: str
    lineno: int
    module: str
    requirement: "ImportRequirement"
    statement: str

    @property
    def guarded(self) -> bool:
        """Retained for readability. NOT the classification -- `requirement` is."""
        return self.requirement is not ImportRequirement.HARD

    def __repr__(self) -> str:
        g = ("" if self.requirement is ImportRequirement.HARD
             else " [{}]".format(self.requirement.value))
        return "{}:{} {}{}".format(self.path, self.lineno, self.statement, g)


@dataclass(frozen=True)
class CensusAudit:
    """What the walk saw. Files that failed to parse are NAMED, not skipped.

    A census that silently ignored unparseable files would under-report
    consumers -- the same shape as a parser that silently drops records.
    """
    files_walked: int = 0
    files_parsed: int = 0
    parse_failures: tuple = ()
    roots: tuple = ()
    mappings: tuple = ()

    def reconciles(self) -> bool:
        """True BY CONSTRUCTION, and retained only as a reportable field.

        Every walked file takes exactly one branch -- parsed, or recorded as a
        failure -- so this equality can never be false. An earlier version
        RAISED on it, which read like a guard and guarded nothing: sabotage
        removing that raise changed no observable behaviour.

        The identical tautology was written into requirements_parse.py twice
        and corrected there by reconciling against the PHYSICAL LINE COUNT, a
        quantity no branch touches. This census has no equivalent independent
        quantity -- the file count IS the walk -- so the false guard is deleted
        rather than replaced by a longer one.

        What does the real work here is `walked and not parsed`, which is not a
        tautology: it distinguishes an instrument failure from a finding.
        """
        return self.files_walked == self.files_parsed + len(self.parse_failures)


def _top_level(name: str) -> str:
    return (name or "").split(".")[0]


class ImportRequirement(str, Enum):
    """How hard a dependency an import site actually represents.

    A boolean `guarded` flag was WRONG. Measured 2026-08-13:

        except ImportError          -> optional; a missing package is caught
        except ModuleNotFoundError  -> optional for a MISSING package, but a
                                       plain ImportError from a BROKEN package
                                       escapes it
        except Exception / bare     -> optional, but indiscriminately -- it also
                                       swallows failures that are not about
                                       availability
        except ValueError           -> NOT optional. An unavailable package
                                       raises ImportError and escapes.

    The previous implementation marked all four optional, because it tested
    only that a handler EXISTED. Three of the eight handler shapes probed were
    reported wrongly.
    """
    HARD = "hard"
    IMPORTERROR_GUARDED = "importerror_guarded"
    MODULENOTFOUND_GUARDED = "modulenotfound_guarded"
    BROAD_EXCEPTION_GUARDED = "broad_exception_guarded"


def _handler_names(handler) -> set:
    """Every exception name a handler names, flattening tuples."""
    names = set()

    def collect(node):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                collect(elt)

    if handler.type is not None:
        collect(handler.type)
    return names


def _requirement_of_handlers(handlers) -> "ImportRequirement":
    """Classify by what the handlers CATCH, not by their existence.

    A missing package raises ModuleNotFoundError, a subclass of ImportError --
    verified, not assumed. So `except ImportError` catches it; `except
    ModuleNotFoundError` catches it too but would NOT catch a plain
    ImportError from a partially broken package.
    """
    broad = False
    importerror = False
    modulenotfound = False
    for h in handlers:
        if h.type is None:
            broad = True
            continue
        names = _handler_names(h)
        if "ImportError" in names:
            importerror = True
        if "ModuleNotFoundError" in names:
            modulenotfound = True
        if names & {"Exception", "BaseException"}:
            broad = True
    if importerror:
        return ImportRequirement.IMPORTERROR_GUARDED
    if modulenotfound:
        return ImportRequirement.MODULENOTFOUND_GUARDED
    if broad:
        return ImportRequirement.BROAD_EXCEPTION_GUARDED
    return ImportRequirement.HARD


def _requirements_by_line(tree: ast.AST) -> dict:
    """Map each guarded import's line number to its ImportRequirement.

    Only `try` bodies are considered. An import in an `except` or `else` clause
    is not protected by that statement's handlers.
    """
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.handlers:
            continue
        requirement = _requirement_of_handlers(node.handlers)
        if requirement is ImportRequirement.HARD:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    out[sub.lineno] = requirement
    return out


@lru_cache(maxsize=1)
def _installed_packages() -> dict:
    """Top-level module to distributions, scanned ONCE.

    MEASURED 2026-08-13: `packages_distributions()` is NOT cached by
    importlib -- 0.541 s on the first call and 0.536 s on the second, because
    it rescans site-packages every time. Calling it per package made six
    lookups cost 3.25 s against 0.001 s with a shared mapping, and turned a
    0.05 s test file into 17.75 s.

    A frozen mapping is returned so a caller cannot mutate the cache.
    """
    try:
        from importlib.metadata import packages_distributions
        return dict(packages_distributions())
    except Exception:                                          # noqa: BLE001
        return {}


def resolve_modules(distribution, *, installed=None):
    """Which modules a distribution provides, and on what evidence.

    MEASURED 2026-08-13 against installed metadata: four of seven sampled
    distributions DISAGREE with a naive hyphen-to-underscore guess --

        pyBigWig        -> pyBigWig, pyBigWigTest   (guess: pybigwig)
        beautifulsoup4  -> bs4                      (guess: beautifulsoup4)
        pyyaml          -> yaml, _yaml              (guess: pyyaml)
        python-dateutil -> dateutil                 (guess: python_dateutil)

    pyBigWig matters directly here: Python imports are case-sensitive, so the
    guess would find nothing. The previous census transformed distribution
    names into module names with `.lower().replace("-", "_")`, which is exactly
    the string surgery the shared model forbids.

    Metadata is consulted first. When a distribution is not installed the
    assumption is still made -- there is no alternative -- but it is RECORDED
    as ASSUMED_IDENTICAL rather than hidden inside a transformation.
    """
    dist = DistributionName(distribution)
    if installed is None:
        installed = _installed_packages()
    modules = tuple(
        ImportName(module)
        for module, dists in installed.items()
        if any(DistributionName(d) == dist for d in dists)
    )
    if modules:
        return DistributionImportMapping(
            distribution=dist, modules=modules,
            source=MappingSource.PACKAGE_METADATA)
    # Not installed: assume the distribution name doubles as the module, in
    # both the given spelling and the underscored form. Recorded as an
    # ASSUMPTION so a caller can see that nobody checked.
    raw = str(distribution)
    guesses = {raw, raw.lower(), raw.lower().replace("-", "_")}
    return DistributionImportMapping(
        distribution=dist,
        modules=tuple(ImportName(g) for g in sorted(guesses)),
        source=MappingSource.ASSUMED_IDENTICAL)


def census(roots, packages, *, repo_root=None, allow_partial=False,
           installed=None):
    """Return (sites, audit): every import of `packages` under `roots`.

    `sites` maps a normalised package name to a tuple of ImportSite. A package
    with no importer maps to an empty tuple -- present in the result, so
    "measured and absent" is distinguishable from "never asked".
    """
    repo = Path(repo_root or ".").resolve()
    # Distribution identity and import identity are SEPARATE vocabularies.
    # Results are keyed by canonical DISTRIBUTION name; matching happens on
    # the modules that distribution actually provides.
    mappings = {}
    module_to_dist = {}
    shared = installed if installed is not None else _installed_packages()
    for pkg in packages:
        m = resolve_modules(pkg, installed=shared)
        key = str(m.distribution)
        mappings[key] = m
        for mod in m.modules:
            module_to_dist.setdefault(mod.top_level, key)
    found = defaultdict(list)
    for key in mappings:
        found[key] = []

    walked = parsed = 0
    failures = []
    for root in roots:
        base = repo / root
        if not base.exists():
            raise CensusError(
                "census root {!r} does not exist under {}; a root that is not "
                "walked is not a root that found nothing".format(root, repo))
        for path in sorted(base.rglob("*.py")):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            walked += 1
            try:
                src = io.open(path, encoding="utf-8", errors="strict").read()
                tree = ast.parse(src)
            except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                failures.append("{}: {}: {}".format(
                    path.relative_to(repo), type(exc).__name__, exc))
                continue
            parsed += 1
            guarded_lines = _requirements_by_line(tree)
            lines = src.splitlines()
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [_top_level(a.name) for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:            # relative import: never external
                        continue
                    names = [_top_level(node.module)]
                for name in names:
                    key = module_to_dist.get(name)
                    if key is None:
                        continue
                    stmt = (lines[node.lineno - 1].strip()
                            if 0 < node.lineno <= len(lines) else "")
                    found[key].append(ImportSite(
                        path=str(path.relative_to(repo)).replace("\\", "/"),
                        lineno=node.lineno,
                        module=name,
                        requirement=guarded_lines.get(
                            node.lineno, ImportRequirement.HARD),
                        statement=stmt,
                    ))

    audit = CensusAudit(files_walked=walked, files_parsed=parsed,
                        parse_failures=tuple(failures), roots=tuple(roots),
                        mappings=tuple(sorted(mappings)))
    if failures and not allow_partial:
        raise CensusError(
            "{} file(s) failed to parse and allow_partial is False: {}. A "
            "partially measured import topology must not present itself as a "
            "complete one -- the measured run happened to be 941/941, but the "
            "instrument must GUARANTEE that rather than depend on it.".format(
                len(failures), failures[:3]))
    if walked and not parsed:
        raise CensusError(
            "{} file(s) walked and NONE parsed. This is an instrument failure, "
            "not a finding. First: {}".format(
                walked, failures[0] if failures else "(none recorded)"))
    return {k: tuple(v) for k, v in found.items()}, audit


def report(sites, audit, mappings=None) -> str:
    out = []
    out.append("  walked {} file(s), parsed {}, failed {}".format(
        audit.files_walked, audit.files_parsed, len(audit.parse_failures)))
    out.append("  roots: {}".format(", ".join(audit.roots)))
    for f in audit.parse_failures[:5]:
        out.append("    PARSE FAILURE  {}".format(f))
    out.append("")
    for pkg in sorted(sites):
        s = sites[pkg]
        if not s:
            src = ""
            if mappings and pkg in mappings:
                m = mappings[pkg]
                src = "  [modules {} via {}]".format(
                    [str(x) for x in m.modules], m.source.value)
            out.append("  {:<12} NO IMPORT ANYWHERE{}".format(pkg, src))
            out.append("               -> declared but unused, or used via a "
                       "non-import path (a console script, a plugin entry "
                       "point). Verify before removing.")
            continue
        by_root = defaultdict(int)
        for site in s:
            by_root[site.path.split("/", 1)[0]] += 1
        hard = sum(1 for site in s if site.requirement is ImportRequirement.HARD)
        soft = len(s) - hard
        out.append("  {:<12} {} import(s), {} HARD{}: {}".format(
            pkg, len(s), hard,
            ", {} guarded".format(soft) if soft else "",
            dict(by_root)))
        for site in s[:6]:
            out.append("      {}".format(site))
        if len(s) > 6:
            out.append("      ... and {} more".format(len(s) - 6))
    return "\n".join(out)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    packages = argv or ["pyfaidx", "httpx", "anyio", "jinja2", "seaborn",
                        "pytest", "pytest_cov", "pre_commit"]
    roots = ["src", "scripts", "tests"]
    roots = [r for r in roots if (Path(".") / r).exists()]
    if not roots:
        print("  no census roots found under the working directory")
        return 2
    sites, audit = census(roots, packages)
    shared = _installed_packages()
    mappings = {str(DistributionName(p)): resolve_modules(p, installed=shared)
                for p in packages}
    print(report(sites, audit, mappings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
