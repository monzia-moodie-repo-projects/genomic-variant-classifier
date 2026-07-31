#!/usr/bin/env python3
"""scripts/maintenance/preflight_data_guard.py -- Monzia Moodie

Fail-loud guard: verify data/ is a usable REAL local directory (or a junction
that currently resolves) with the canonical subtrees present, AND that the
volume has enough free space to run, BEFORE any run or test touches it.

Returns 0 if usable, 1 if the data/ path is broken or free space is below the
hard floor. Importable: assert_data_usable(), check_free_space(), storage_rows().

WHAT CHANGED ON 2026-07-21, AND WHY
====================================
Two defects, both of omission.

FIRST: THIS GUARD NEVER CHECKED FREE SPACE. It verified structure -- not a
dangling junction, not shadowed by a file, canonical subtrees present -- and
stopped there. So the guard whose entire purpose is catching storage problems
before a run could not catch the storage problem the machine actually had. On
2026-07-20 the volume reached 0.716 per cent free and the discovery was made by
a run failing, not by a guard refusing.

SECOND, AND WORSE: NOTHING EVER CALLED IT. A repository-wide search across
every .py, .sh, .ps1 and .yaml on 2026-07-21 found zero invocations of
assert_data_usable or of this file. Its own docstring said it was "importable
... to wire into run scripts / conftest", and it never was. A guard that is not
invoked is not a guard; it is a comment that happens to be executable. It is
now wired into preflight_run17.run_all() via storage_gate().

THE MEASUREMENTS THAT MOTIVATED IT (2026-07-21)
------------------------------------------------
    volume            935.59 GiB, and C: is the ONLY fixed volume
    free               83.50 GiB (8.925 per cent)
    data/              98.75 GiB -- LARGER than the free space, so there is no
                       headroom to stage, duplicate or rebuild in place
    suite wall-clock   605.08 s to 1131.67 s over nine runs on identical code,
                       a 1.87x range consistent with a host under storage
                       pressure

THREE SEVERITIES, NOT ONE THRESHOLD
------------------------------------
A single threshold must choose between crying wolf and staying silent until it
is too late. This uses three bands, all from configs/data_manifest.yaml:

    free >= required                 OK
    hard_floor <= free < required    WARN -- the run proceeds; the margin is gone
    free < hard_floor                FAIL -- refuse; this is where runs corrupt

    required = working_cache_gib + max(headroom_fraction * volume,
                                       headroom_min_gib)

POLICY LIVES IN THE MANIFEST, NOT HERE
---------------------------------------
docs/standards/DATA_LAYOUT_STANDARD.md already declares
configs/data_manifest.yaml the single source of truth that "the auditor, setup,
and sync scripts all read". These numbers previously existed only as three
constants inside scripts/forensics/audit_disk_census.py. That script keeps its
own copies deliberately, because it must run standalone from any directory
without the repository importable -- and tests/unit/test_storage_guard.py fails
if the two ever disagree. A number written in two places is wrong in one of them
eventually; a test is what makes "eventually" arrive on purpose.

If the manifest is missing or malformed the guard uses documented defaults and
SAYS SO on stderr. It does not fail closed on a missing policy file, because
refusing every run because a configuration file moved would be a worse failure
than the one being guarded against -- but a silent fallback would be worse
still.

Usage:
  python scripts/maintenance/preflight_data_guard.py
  python scripts/maintenance/preflight_data_guard.py data --manifest configs/data_manifest.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3
_REQUIRED = ["external", "raw", "processed"]

# Documented defaults, used only when the manifest cannot be read. Kept
# numerically identical to the manifest's storage: block; the drift test pins
# all three sources together.
DEFAULT_POLICY = {
    "working_cache_gib": 14.7,
    # ADDED 2026-07-30. The JEPA embedding cache is a ONE-TIME artifact and
    # is NOT part of required_free_bytes; the run gate reads
    # working_cache_gib alone. This exists so the figure has one home and
    # scripts/forensics/audit_disk_census.py can be pinned against it.
    "jepa_embedding_cache_gib": 55.2,
    "headroom_fraction": 0.05,
    "headroom_min_gib": 20.0,
    "hard_floor_gib": 25.0,
    "warn_below_percent_free": 10.0,
}


@dataclass(frozen=True)
class StoragePolicy:
    """How much free space a run requires, and when to refuse."""

    working_cache_gib: float
    jepa_embedding_cache_gib: float
    headroom_fraction: float
    headroom_min_gib: float
    hard_floor_gib: float
    warn_below_percent_free: float
    source: str = "defaults"

    def __post_init__(self) -> None:
        if not 0.0 <= self.headroom_fraction < 1.0:
            raise ValueError(
                f"headroom_fraction must be in [0, 1), got {self.headroom_fraction}")
        for name in ("working_cache_gib", "jepa_embedding_cache_gib",
                     "headroom_min_gib", "hard_floor_gib"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")
        if self.hard_floor_gib > self.headroom_min_gib + self.working_cache_gib:
            raise ValueError(
                f"hard_floor_gib ({self.hard_floor_gib}) exceeds the required "
                f"total at its own floor; the FAIL band would swallow the WARN "
                "band and the gate would go straight from OK to refusing.")

    @classmethod
    def load(cls, manifest: str | Path = "configs/data_manifest.yaml") -> "StoragePolicy":
        p = Path(manifest)
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            block = data.get("storage")
            if not isinstance(block, dict):
                raise KeyError("no 'storage' section")
            missing = [k for k in DEFAULT_POLICY if k not in block]
            if missing:
                raise KeyError(f"storage section missing {missing}")
            return cls(**{k: float(block[k]) for k in DEFAULT_POLICY},
                       source=str(p))
        except Exception as e:
            print(f"[data-guard] WARNING: could not read storage policy from {p} "
                  f"({type(e).__name__}: {e}); using documented defaults.",
                  file=sys.stderr)
            return cls(**DEFAULT_POLICY, source=f"defaults (could not read {p})")

    def required_free_bytes(self, total_bytes: int) -> float:
        headroom = max(self.headroom_fraction * total_bytes,
                       self.headroom_min_gib * GIB)
        return self.working_cache_gib * GIB + headroom


@dataclass(frozen=True)
class StorageVerdict:
    """The outcome of a free-space check, with every number that produced it."""

    severity: str            # "OK" | "WARN" | "FAIL"
    total_bytes: int
    free_bytes: int
    required_bytes: float
    hard_floor_bytes: float
    percent_free: float
    message: str
    policy_source: str

    @property
    def ok(self) -> bool:
        return self.severity == "OK"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["total_gib"] = self.total_bytes / GIB
        d["free_gib"] = self.free_bytes / GIB
        d["required_gib"] = self.required_bytes / GIB
        return d


def check_free_space(path: str | Path = ".",
                     policy: StoragePolicy | None = None) -> StorageVerdict:
    """Measure free space at `path` and grade it against the policy.

    Never raises on a healthy filesystem: the verdict is the return value, so a
    caller decides what to do with WARN. Raises only if the path cannot be
    stat'ed at all, which is a different problem.
    """
    policy = policy or StoragePolicy.load()
    total, _used, free = shutil.disk_usage(str(path))
    required = policy.required_free_bytes(total)
    floor = policy.hard_floor_gib * GIB
    pct = 100.0 * free / total if total else 0.0

    if free < floor:
        sev = "FAIL"
        msg = (f"free space {free/GIB:.2f} GiB is below the hard floor "
               f"{floor/GIB:.2f} GiB ({pct:.2f} per cent of {total/GIB:.2f} GiB). "
               "Refusing: runs at this level corrupt artifacts and fail late. "
               "See docs/sessions/SESSION_2026-07-21 Part Seven for the measured "
               "consumers, and scripts/forensics/audit_disk_census.py to re-measure.")
    elif free < required:
        sev = "WARN"
        msg = (f"free space {free/GIB:.2f} GiB is below the required "
               f"{required/GIB:.2f} GiB ({pct:.2f} per cent of {total/GIB:.2f} GiB) "
               "but above the hard floor. The run may proceed with no margin: a "
               "full run builds a working cache of "
               f"{policy.working_cache_gib:.1f} GiB.")
    else:
        sev = "OK"
        msg = (f"free space {free/GIB:.2f} GiB meets the required "
               f"{required/GIB:.2f} GiB ({pct:.2f} per cent of {total/GIB:.2f} GiB).")

    if sev != "FAIL" and pct < policy.warn_below_percent_free:
        msg += (f" NOTE: below {policy.warn_below_percent_free:.0f} per cent free, "
                "Windows competes with the workload -- antivirus scanning, "
                "temporary-file allocation and shadow copies. A 1.87x suite "
                "wall-clock range was measured in this band on 2026-07-21.")

    return StorageVerdict(severity=sev, total_bytes=total, free_bytes=free,
                          required_bytes=required, hard_floor_bytes=floor,
                          percent_free=pct, message=msg,
                          policy_source=policy.source)


def assert_data_usable(data_dir: str | Path = "data") -> None:
    """Structural checks only. Unchanged in behaviour since 2026-06-14."""
    p = Path(data_dir)
    isjunction = getattr(os.path, "isjunction", lambda _p: False)
    if os.path.lexists(p) and not p.exists():
        raise SystemExit(
            f"[data-guard] '{p}' is a DANGLING junction/symlink (target gone -- e.g. Google "
            "Drive G: not mounted/synced). Mount/sync the target or re-point data/, then retry."
        )
    if p.exists() and not p.is_dir():
        raise SystemExit(f"[data-guard] '{p}' exists but is NOT a directory (a stray file shadows it). "
                         "Remove/rename it and restore data/ (git or setup_data_tree.py).")
    if not p.exists():
        raise SystemExit(f"[data-guard] '{p}' is missing. Run scripts/maintenance/setup_data_tree.py.")
    missing = [s for s in _REQUIRED if not (p / s).is_dir()]
    if missing:
        raise SystemExit(f"[data-guard] '{p}' is missing canonical subtrees {missing}. "
                         "Run scripts/maintenance/setup_data_tree.py.")
    kind = "junction(resolves)" if (isjunction(p) or os.path.islink(p)) else "real dir"
    print(f"[data-guard] OK -- '{p}' usable ({kind}); subtrees present.")


def storage_rows(data_dir: str | Path = "data",
                 manifest: str | Path = "configs/data_manifest.yaml"
                 ) -> list[tuple[str, str]]:
    """(severity, message) rows in preflight_run17's gate convention.

    Structural failure is a FAIL row rather than a raised SystemExit, so one
    broken gate cannot prevent the others from reporting -- a preflight that
    dies on its first problem hides the second.
    """
    rows: list[tuple[str, str]] = []
    p = Path(data_dir)
    try:
        assert_data_usable(p)
        rows.append(("OK", f"data-guard: '{p}' usable; canonical subtrees present"))
    except SystemExit as e:
        rows.append(("FAIL", f"data-guard: {e}"))
        return rows

    v = check_free_space(p, StoragePolicy.load(manifest))
    rows.append((v.severity, f"storage: {v.message}"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="data/ structural + free-space guard")
    ap.add_argument("data_dir", nargs="?", default="data")
    ap.add_argument("--manifest", default="configs/data_manifest.yaml")
    ap.add_argument("--skip-space", action="store_true",
                    help="structural checks only (the pre-2026-07-21 behaviour)")
    args = ap.parse_args()

    try:
        assert_data_usable(args.data_dir)
    except SystemExit as e:
        print(e)
        return 1

    if args.skip_space:
        print("[data-guard] free-space check SKIPPED by --skip-space.")
        return 0

    v = check_free_space(args.data_dir, StoragePolicy.load(args.manifest))
    print(f"[data-guard] policy source: {v.policy_source}")
    print(f"[data-guard] {v.severity} -- {v.message}")
    return 1 if v.severity == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
