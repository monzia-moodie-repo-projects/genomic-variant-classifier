"""The dependency profiles: what each one declares, which lock serves which target, how it is installed, and what it must agree with.

DEPENDENCY-PROFILES-1 (owner rulings 2026-10-31) created the training-and-test profile; DRIFT-PROFILE-LOCK (2026-11-02) adds the
isolated drift-monitoring profile. ONE declaration for both, read by the check (scripts/check_dependency_profiles.py), by the
installer (scripts/install_locked_profile.py) and by the tests, so the three cannot disagree about a file name, a target or a flag.

Standard library only: the installer imports this module with the BASE interpreter, before any environment exists.

TARGETS are written out, not taken from whichever host runs the check: a lock describes a target, and a host's own fields are only
evidence about that host. Each target is the MEASURED marker environment of the interpreter that resolved its lock, because
pip-compile evaluates markers on that interpreter:
    linux-3.11    /usr/bin/python3.11 (3.11.17) of the development sandbox (packaging.markers.default_environment(), 2026-11-01);
    linux-3.12    /usr/bin/python3.12 (3.12.3) of the same sandbox;
    windows-3.12  the owner's .venv312 (3.12.10), where the Windows locks are resolved (observation a4fc5829, 2026-10-31).
A runner's own kernel release, kernel version and Python patch release differ from these: the environment contract RECORDS them as
host provenance and judges only the stable target family, the Python policy and every marker outcome.

ALIGNMENT (DRIFT-PROFILE-LOCK). The drift monitor reads the parquet files the training pipeline writes and computes on the same
numerical stack, so wherever the two profiles share a distribution they must hold the SAME version -- except where a distribution
the drift profile needs makes that impossible. Each such DEPARTURE is named here with the measurement that proves it necessary: on
2026-11-02, a resolution of the drift sources constrained to the training version of that one distribution was impossible (pip-tools
7.5.3 under Python 3.12.3; the logs are in the session record). Every other shared distribution is constrained to the training
locks' version by constraints-drift.txt, which is DERIVED from those locks and checked to be exactly that derivation.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PIP = ("pip", "26.1.2")
PROJECT = ("genomic-variant-classifier", "0.1.0")
TOOLCHAIN_SOURCE, TOOLCHAIN_LOCK = "requirements-generator-toolchain.in", "requirements-generator-toolchain.txt"

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
POLICY = {"linux-3.11": "==3.11.*", "linux-3.12": "==3.12.*", "windows-3.12": "==3.12.*"}


@dataclass(frozen=True)
class Profile:
    """One dependency profile. Tuples, not dicts, so the declaration is immutable."""
    name: str
    sources: tuple                      # authored declarations; every lock must satisfy them on each of its targets
    constraints: str                    # the versions the resolution was constrained to
    locks: tuple                        # ((lock file, (target, ...)), ...)
    install_flags: tuple                # pip flags for installing the lock, besides --require-hashes --no-deps --no-cache-dir
    build_constraints: str = ""         # passed as --build-constraint: pins the build backend of source-only distributions
    aligned_with: str = ""              # the profile whose locks the constraints are DERIVED from ("" when measured, not derived)
    departures: tuple = ()              # ((distribution, measured reason), ...) allowed to differ from `aligned_with`
    forbidden: tuple = ()               # distributions that must never be in this profile, with the reason

    def lock_for(self, target: str) -> str:
        for lock, targets in self.locks:
            if target in targets:
                return lock
        raise KeyError("profile {} has no lock for target {}".format(self.name, target))

    @property
    def targets(self) -> tuple:
        return tuple(t for _, targets in self.locks for t in targets)


TRAIN_TEST = Profile(
    name="train-test",
    sources=("requirements.txt", "requirements-dev.txt"),
    constraints="constraints-observed-venv312.txt",
    locks=(("requirements-train-test-linux.lock", ("linux-3.11", "linux-3.12")),
           ("requirements-train-test-windows.lock", ("windows-3.12",))),
    install_flags=(),
    # deflate-dict 1.2.2, dict-hash 1.3.7 and pyspark 3.5.8 exist only as source archives (measured 2026-11-01).
    build_constraints="build-constraints-train-test.txt",
)

DRIFT = Profile(
    name="drift",
    sources=("requirements-drift.txt", "requirements-drift-test.txt"),
    constraints="constraints-drift.txt",
    locks=(("requirements-drift-linux.lock", ("linux-3.11", "linux-3.12")),
           ("requirements-drift-windows.lock", ("windows-3.12",))),
    # Every distribution of the drift locks has a wheel for each target (measured 2026-11-02 with --only-binary=:all: dry runs), so
    # nothing is ever built: a source archive appearing later is refused at installation instead of being built unpinned.
    install_flags=("--only-binary=:all:",),
    aligned_with="train-test",
    departures=(
        ("lightgbm", "nannyml 0.13.1 requires lightgbm>=3.3,<4.6; the training locks hold 4.6.0, a base model of the ensemble."),
        ("plotly", "nannyml 0.13.1 requires plotly>=5.6,<6.0 and evidently 0.7.6 plotly>=5.10.0,<6; the training locks hold 6.6.0."),
        ("xgboost", "nannyml 0.13.1 requires FLAML[automl], whose 'automl' extra requires xgboost>=0.90,<3.0.0 in every release "
                    "2.3.3 to 2.7.0; the training locks hold 3.2.0."),
    ),
    forbidden=(
        ("pyspark", "nannyml imports it when present; pyspark 3.5.x calls np.NaN, removed in NumPy 2.0, so its presence breaks "
                    "`import nannyml` (measured 2026-07-13)."),
        ("torch", "training only; the drift environment never loads a model artifact."),
        ("torch-geometric", "training only."),
    ),
)

PROFILES = {p.name: p for p in (TRAIN_TEST, DRIFT)}


def install_command(profile_name: str) -> str:
    """The one way to build a profile's environment, as a person types it. Every message that tells someone how to build the drift
    environment cites THIS, never a copy: four hand-kept copies of the old `pip install -r requirements-drift.txt` recipe outlived
    the change that retired it (measured 2026-11-02)."""
    p = profile(profile_name)
    return ("python scripts/install_locked_profile.py --profile {} --target TARGET --venv NEW_FOLDER\n"
            "  (TARGET: {}; it refuses an existing folder, and proves the result IS the lock)").format(p.name, ", ".join(p.targets))


def profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError("unknown profile {!r}; the profiles are {}".format(name, sorted(PROFILES))) from None
