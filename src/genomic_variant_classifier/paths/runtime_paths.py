"""One authority for where the repository, its artifacts and its state live.

RUNTIME-PATHS-1
===============
This project accumulated FIVE independent conventions for locating things:

    root: str = "."                     five agents, ambient working directory
    PROJECT_ROOT                        config.py, a hard-coded Windows literal
    ADAPTATION_PROJECT_ROOT             adaptation_agent, its own variable
    Path(__file__).parent               shared_state.py, the only correct one
    Path("data/agent_state.json")       version_monitor_agent, cwd-relative

Each was defensible alone. Together they mean no single answer exists to "where
does this project keep things", and every new component invents a sixth.

MEASURED 2026-08-14, and each measurement changed the design:

    GVC_PROJECT_ROOT is set NOWHERE -- not in continuous integration, not in
    the Dockerfile, not in any script, not in the shell. So config.py:17's
    fallback, a literal C:\\Projects\\genomic-variant-classifier, is the value
    every consumer receives. On the Linux runner it names a path that cannot
    exist.

    Two of its derived constants point at directories that do not exist even on
    the workstation: AUDIT_LOG_DIR and SHARED_STATE_PATH resolve to
    PROJECT_ROOT / "agent_layer", but agent_layer lives under
    src/genomic_variant_classifier/. Nothing reads either -- they are dead
    declarations that look authoritative.

    Two files named agent_state.json hold UNRELATED schemas: a flat
    literature-scout key-value store, and the orchestrator's structured
    SharedState. Filenames alone do not encode ownership, and reasoning from
    the name nearly merged them.

WHY THREE ROOTS AND NOT ONE
    project_root    where the source lives. Identity, not destination.
    artifact_root   where generated output goes.
    state_root      where mutable operational state persists.

Conflating the first two is OUTPUT-ROOT-CONFLATION-1 -- an agent computing
`Path(self._root) / "reports"` depends on repository layout to answer a
question about where its output belongs. Conflating the first and third is why
mutable state ended up inside the source tree twice.

DISCOVERY VERIFIES IDENTITY, NOT MERE EXISTENCE
`(candidate / "src").exists()` is a comfort assertion -- any directory can
contain `src/`. Discovery here requires a conjunction of sentinels AND the
declared project name from pyproject.toml, measured to be
"genomic-variant-classifier" under [project].

AND THERE IS NO DEVELOPER PATH FALLBACK
Explicit argument, then GVC_PROJECT_ROOT, then discovery, then RAISE. A
resolver that guesses on failure is how a machine-specific literal became the
value every consumer received.

Author: Monzia Moodie
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: Measured from pyproject.toml on 2026-08-14: [project] name, hyphenated.
PROJECT_NAME = "genomic-variant-classifier"

#: A directory is this repository only if it holds ALL of these AND declares
#: the project name above. Each alone is common; the conjunction is not.
#:
#: RUNTIME-SENTINEL-TEST-ARTEFACT-1 (2026-08-17). This tuple once included
#: "tests/EXPECTED_SUITE_SIZE". That is a TEST-SUITE ARTEFACT being used to
#: identify a DEPLOYMENT root, and every correct deployment excludes tests --
#: so the conjunction could not hold in a container by construction.
#:
#: MEASURED against this repository's own Dockerfile and .dockerignore:
#:
#:     pyproject.toml                  copied into the trainer image : True
#:     src/genomic_variant_classifier  copied into the trainer image : True
#:     tests/EXPECTED_SUITE_SIZE       copied into the trainer image : FALSE
#:                                     (excluded by `tests/` at .dockerignore)
#:
#: The trainer image runs `COPY . .`, so discovery would have RAISED on import
#: of any module reaching agent_layer.config -- which is where cloud training
#: runs. The defect was latent from 69a9597 only because nothing imported this
#: module yet; PROJECT-ROOT-HARDCODED-1 is its first consumer.
#:
#: Two sentinels plus the declared name lose NO discrimination. Measured:
#:
#:     this repository : True
#:     C:/Users/monzi  : False      C:/Projects : False
#:     C:/Windows      : False      C:/         : False
#:
#: A sentinel must be present in EVERY environment where the root must be
#: found. Test artefacts, documentation and build outputs are not.
PROJECT_SENTINELS = (
    "pyproject.toml",
    "src/genomic_variant_classifier",
)

#: Environment overrides. Read at RESOLUTION time, not at import time, so a
#: test can control them in-process without reimporting the module.
ENV_PROJECT_ROOT = "GVC_PROJECT_ROOT"
ENV_ARTIFACT_ROOT = "GVC_ARTIFACT_ROOT"
ENV_STATE_ROOT = "GVC_STATE_ROOT"

#: The transaction journal lives OUTSIDE the repository, so an interrupted
#: installer survives a working-tree reset and a successful one leaves the
#: repository with zero rollback artefacts.
#:
#: This is a FIFTH path domain, not a synonym for state_root. state_root
#: defaults to <project>/.gvc-state -- correct for agent state, which belongs
#: to THIS checkout. A transaction journal does not: it must outlive the
#: checkout it is repairing.
ENV_CACHE_ROOT = "GVC_CACHE_ROOT"


class RuntimePathError(RuntimeError):
    """A location could not be determined, and no guess was made.

    Raised INSTEAD of falling back to a developer's absolute path. The previous
    behaviour returned C:\\Projects\\genomic-variant-classifier on every machine
    in the world, which is worse than failing because it fails LATER and
    somewhere else.
    """


@dataclass(frozen=True)
class RuntimePaths:
    """Where this project keeps things. One authority, three distinct roots."""

    project_root: Path
    artifact_root: Path
    state_root: Path
    cache_root: Path

    @property
    def reports_root(self) -> Path:
        """Generated reports. Five agents write here.

        Derived from artifact_root, NOT project_root: where output goes is a
        deployment decision, not a fact about repository layout.
        """
        return self.artifact_root / "reports"

    @property
    def literature_scout_state(self) -> Path:
        """The literature-scout store: a flat key-value change-detection log.

        NAMED FOR ITS OWNER. Two files called agent_state.json held unrelated
        schemas, and that ambiguity nearly caused a merge that would have
        destroyed both.
        """
        return self.state_root / "literature_scout" / "state.json"

    @property
    def orchestrator_state(self) -> Path:
        """The orchestrator's SharedState: structured, keyed by agent."""
        return self.state_root / "orchestrator" / "state.json"

    @property
    def transaction_journal(self) -> Path:
        """Where an in-flight installer transaction records its preimages.

        INSTALLER-TRANSACTION-1. Under cache_root, never under project_root:
        a successful installer must leave NO rollback artefact in the
        repository, and an interrupted one must still be recoverable.

        MEASURED 2026-08-19: 148 `.bak_<timestamp>` siblings had accumulated
        inside the repository across eight days, 17,640,928 bytes, invisible
        to `git status` because .gitignore carries `*.bak_*`. What was
        designed as a rollback implementation detail had become a permanent
        archival system by omission.
        """
        return self.cache_root / "transactions"

    def describe(self) -> dict:
        """A serialisable record, for provenance in run artifacts."""
        return {
            "project_root": str(self.project_root),
            "artifact_root": str(self.artifact_root),
            "state_root": str(self.state_root),
            "cache_root": str(self.cache_root),
            "reports_root": str(self.reports_root),
            "literature_scout_state": str(self.literature_scout_state),
            "orchestrator_state": str(self.orchestrator_state),
            "transaction_journal": str(self.transaction_journal),
        }


def _has_sentinels(path: Path) -> bool:
    return all((path / s).exists() for s in PROJECT_SENTINELS)


def _declares_project_name(path: Path) -> bool:
    """Read the declared name from pyproject.toml.

    A structural check: a directory that merely contains src/ and a
    pyproject.toml belonging to a DIFFERENT project cannot satisfy it.
    """
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        with pyproject.open("rb") as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return doc.get("project", {}).get("name") == PROJECT_NAME


def looks_like_project_root(path: Path) -> bool:
    """Both conditions. Neither alone is sufficient."""
    return _has_sentinels(path) and _declares_project_name(path)


def discover_project_root(origin: Path = None) -> Path:
    """Walk upward from `origin` looking for this repository. None if absent.

    Anchored to `__file__` by default, so the answer does not depend on the
    working directory -- which is the defect this module exists to end.
    """
    start = (Path(__file__) if origin is None else Path(origin)).resolve()
    for candidate in (start, *start.parents):
        if candidate.is_dir() and looks_like_project_root(candidate):
            return candidate
    return None


def _require_project_root(path: Path, source: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimePathError(
            "{} names {!r}, which is not a directory. A project root that does "
            "not exist is not a usable default -- config.py's literal "
            "C:\\\\Projects\\\\... was exactly this, on every machine that is "
            "not the author's workstation.".format(source, str(resolved)))
    if not looks_like_project_root(resolved):
        missing = [s for s in PROJECT_SENTINELS if not (resolved / s).exists()]
        raise RuntimePathError(
            "{} names {!r}, which is not this repository. Missing sentinel(s): "
            "{}; declares project name {!r} (expected {!r}). Existence alone is "
            "not identity.".format(
                source, str(resolved), missing or "none",
                _declared_name(resolved), PROJECT_NAME))
    return resolved


def _declared_name(path: Path):
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        with pyproject.open("rb") as fh:
            return tomllib.load(fh).get("project", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        return None


def resolve_project_root(*, explicit=None, environ=None) -> Path:
    """Explicit, then environment, then discovery, then RAISE.

    Every branch that returns a path has VERIFIED it is this repository. There
    is deliberately no final fallback.
    """
    env = os.environ if environ is None else environ
    if explicit is not None:
        return _require_project_root(explicit, "the explicit project_root argument")
    configured = env.get(ENV_PROJECT_ROOT)
    if configured:
        return _require_project_root(configured, ENV_PROJECT_ROOT)
    discovered = discover_project_root()
    if discovered is not None:
        return discovered
    raise RuntimePathError(
        "Cannot determine the project root. Pass project_root=..., set {}, or "
        "run from within the repository. NO developer-specific path is used as "
        "a fallback: that is PROJECT-ROOT-HARDCODED-1, where config.py:17 "
        "returned one author's absolute Windows path on every machine."
        .format(ENV_PROJECT_ROOT))


def _resolve_secondary(explicit, environment, default: Path, label: str) -> Path:
    """Artifact and state roots need not exist yet -- they are DESTINATIONS.

    Unlike the project root, which must be an existing repository, these are
    created on first write. Requiring them to exist would make a fresh clone
    fail before it could produce anything.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if environment:
        return Path(environment).expanduser().resolve()
    return default


def _default_cache_root(env) -> Path:
    """A user-scoped location OUTSIDE any repository.

    Order, and why each link exists:

        LOCALAPPDATA    on Windows. MEASURED 2026-08-19: set, and
                        AppData/Local resolves outside the repository.
        XDG_STATE_HOME  the POSIX convention for state that should persist
                        but is not configuration.
        home/.local/state
                        the fallback that ALWAYS resolves. MEASURED: with
                        HOME unset on Windows, Path.home() still returned
                        C:/Users/monzi via USERPROFILE. On POSIX it falls
                        back to the password database.

    A NOTE ON WHAT CANNOT BE TESTED FROM WINDOWS. Passing a fake environment
    with XDG_STATE_HOME="/home/runner/.local/state" selects the right BRANCH
    but produces "C:/home/runner/..." -- because path flavour is baked into
    the platform, not into the environment. So tests here assert
    RELATIONSHIPS (outside the repository, beneath the chosen base) rather
    than literal paths, and the literal POSIX form is verified on the runner.
    """
    if os.name == "nt" and env.get("LOCALAPPDATA"):
        base = Path(env["LOCALAPPDATA"])
    elif env.get("XDG_STATE_HOME"):
        base = Path(env["XDG_STATE_HOME"])
    else:
        base = Path.home() / ".local" / "state"
    return (base / "GenomicVariantClassifier").expanduser().resolve()


def resolve_runtime_paths(*, project_root=None, artifact_root=None,
                          state_root=None, cache_root=None,
                          environ=None) -> RuntimePaths:
    """The single entry point. Resolves all FOUR roots, or raises."""
    env = os.environ if environ is None else environ
    project = resolve_project_root(explicit=project_root, environ=env)
    artifacts = _resolve_secondary(
        artifact_root, env.get(ENV_ARTIFACT_ROOT), project, "artifact_root")
    state = _resolve_secondary(
        state_root, env.get(ENV_STATE_ROOT), project / ".gvc-state", "state_root")
    cache = _resolve_secondary(
        cache_root, env.get(ENV_CACHE_ROOT), _default_cache_root(env),
        "cache_root")
    return RuntimePaths(project_root=project, artifact_root=artifacts,
                        state_root=state, cache_root=cache)
