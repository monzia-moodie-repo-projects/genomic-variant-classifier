#!/usr/bin/env python3
"""install_no_detritus.py -- Author: Monzia Moodie

INSTALLER-TRANSACTION-1, step 4: the first payload installed by a transaction.

    An installer is one atomic repository state transition. There are no
    transactional "payload files" and nontransactional "bookkeeping files".

WHAT IS DIFFERENT ABOUT THIS INSTALLER
Every installer before it wrote files directly and backed each one up to a
`.bak_<timestamp>` sibling that it never removed. This one:

    declares its COMPLETE write set before touching the repository
    opens a RepositoryTransaction whose rollback state lives OUTSIDE the tree
    proves the actual write set equals the declared one
    runs the acceptance gate while ROLLBACK IS STILL AVAILABLE
    verifies prospective hygiene BEFORE commit, and attests it afterwards
    commits, destroying the journal
    leaves nothing beside any target

THE ORDERING IS THE POINT
A hygiene check that runs only AFTER commit can diagnose a violation but not
repair one -- the journal is already destroyed. So the check runs twice:

    prospective   inside the transaction, where a failure rolls everything back
    attested      after commit, proving the committed state is clean

    transaction active -> payload installed -> write set proven ->
    acceptance gate -> prospective hygiene -> COMMIT -> journal destroyed ->
    attested hygiene

THE RATCHET IS DERIVED, NEVER TRANSCRIBED
Hand-carried counts produced repeated stale-count incidents on 2026-08-20
alone: a file expected to collect 52 collected 54; another expected 38
collected 44. So this runner MEASURES collection before and after, and renders
both the ratchet number and the README badge from that one measurement.

MEASURED 2026-08-20, before this was written:

    a full `tests/unit` collection            exit 0, 35.1s, 4,960 cases
    a single file                              exit 0,  3.7s
    a file with an import error                exit 2, NO parseable count

The error case is why the exit code gates the parse rather than the parse
gating itself: pytest reports "no tests collected, 1 error" and the regex finds
nothing, but a partial collection must not be trusted either.

    Two full collections cost roughly 70 seconds of the install budget.

WHAT THIS DOES NOT IMPLEMENT
No backup, rollback, path-ownership, recovery, preimage or journal semantics.
Those belong to RepositoryTransaction. This knows only what should change, what
must be true afterward, and which gate to run.

Usage:
    python scripts/install_no_detritus.py --repo-root . --payload <dir> --check
    python scripts/install_no_detritus.py --repo-root . --payload <dir> --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genomic_variant_classifier.paths.runtime_paths import (  # noqa: E402
    resolve_runtime_paths,
)
from genomic_variant_classifier.repository_hygiene import (  # noqa: E402
    backup_artifacts as H,
)
from genomic_variant_classifier.transactions.install_plan import (  # noqa: E402
    DerivedCount, InstallPlan, PlanError, PlannedTarget, TargetAction,
    WriteSetViolation, render_ratchet, render_readme,
)
from genomic_variant_classifier.transactions.repository_transaction import (  # noqa: E402
    RepositoryTransaction, TransactionError, incomplete_transactions,
)

UNIT = "INSTALLER-TRANSACTION-1-step-4"

#: THE RATCHET COUNTS `tests/`, NOT `tests/unit`.
#:
#: MEASURED 2026-08-21: tests/ collects 5207 and tests/unit collects 4982,
#: while the ratchet reads 5207 and the README badge reads tests-5207-. The
#: first version of this runner measured tests/unit and then tried to render
#: the badge from 4982 -- render_readme REFUSED rather than guess, which is how
#: the mismatch was found.
#:
#: The ratchet's own header calls itself "the single source of truth for how
#: many tests exist", tests/conftest.py enforces it against the COLLECTED total
#: of tests/, and tests/unit/test_readme_claims.py requires the badge to EQUAL
#: it with no tolerance. So the counter scope is not a choice.
#:
#: The ACCEPTANCE scope is different and deliberately narrower: every gate in
#: this project's history has run tests/unit. Two scopes, two names, neither
#: standing in for the other.
COUNTER_SCOPE = "tests"
ACCEPTANCE_SCOPE = "tests/unit"

#: The payload: one new test file, installed transactionally.
PAYLOAD = (
    ("test_no_detritus.py", "tests/unit/test_no_detritus.py"),
)

RATCHET_ENTRY = """
# {date} -- {before} -> {after} (+{delta}). INSTALLER-TRANSACTION-1 step 4.
#
#   The FIRST payload installed by a RepositoryTransaction rather than by a
#   script writing files directly. The ratchet and README are ordinary
#   transaction targets, not bookkeeping handled outside it: a failure at any
#   point restores all of them.
#
#   The write set is DECLARED before the repository is touched and proven
#   afterwards, so an installer that unexpectedly touches one more file fails
#   even when every test passes.
#
#   This number was MEASURED, not transcribed: pytest collected {before} before
#   and {after} after, in the same run that wrote it.
#
#   COUNTER SCOPE: tests/. This file certifies COLLECTION and nothing
#   else. Acceptance evidence is owned by the install attestation; the
#   commit message summarises it. RUNNER-GATE-METADATA-ORDER-1,
#   2026-08-21: an acceptance line rendered before the gate ran recorded
#   zeroes permanently.

"""


class RunnerError(RuntimeError):
    """A refusal by the runner, distinct from a plan or transaction refusal."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect(repo: Path, target: str) -> int:
    """Measure pytest collection, gating on the EXIT CODE.

    MEASURED 2026-08-20: a file with an import error produces exit 2 and "no
    tests collected, 1 error", so no count is parseable. A partial collection
    would be worse -- a plausible number from an incomplete measurement -- and
    that is what the exit-code gate refuses.
    """
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", target, "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(repo), capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        tail = "\n".join(l for l in proc.stdout.splitlines()[-6:] if l.strip())
        raise RunnerError(
            "collection of {} exited {}; refusing to derive a count from an "
            "incomplete measurement.\n{}".format(target, proc.returncode, tail))
    count = None
    for line in reversed(proc.stdout.splitlines()):
        match = re.search(r"^(\d+)\s+tests?\s+collected", line.strip())
        if match:
            count = int(match.group(1))
            break
    if count is None:
        raise RunnerError(
            "collection of {} produced no parseable count".format(target))
    print("  collected {:>6} case(s) from {:<12} in {:.1f}s".format(
        count, target, elapsed))
    return count


def _run_suite(repo: Path, target: str, active_transaction=None) -> tuple:
    """Run the acceptance gate and return (passed, skipped, failed).

    INVARIANT-SELF-REFERENCE-1. The gate runs INSIDE the apply transaction, so
    that transaction's journal is legitimately non-terminal while the suite
    looks at it. The identifier is exported so the no-detritus invariant can
    exclude exactly that one journal -- never a blanket tolerance.

    Exported into the CHILD environment only, not this process's, so nothing
    outside the gate can see it.
    """
    started = time.perf_counter()
    env = dict(os.environ)
    if active_transaction:
        env["GVC_ACTIVE_TRANSACTION"] = active_transaction
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", target, "-q", "--tb=short",
         "-p", "no:cacheprovider"],
        cwd=str(repo), capture_output=True, text=True, timeout=5400, env=env)
    elapsed = time.perf_counter() - started
    summary = None
    for line in reversed(proc.stdout.splitlines()):
        if re.search(r"\d+ passed", line):
            summary = line
            break
    if summary is None:
        tail = "\n".join(l for l in proc.stdout.splitlines()[-8:] if l.strip())
        raise RunnerError("no pytest summary line found.\n{}".format(tail))
    passed = int(re.search(r"(\d+) passed", summary).group(1))
    skipped_m = re.search(r"(\d+) skipped", summary)
    failed_m = re.search(r"(\d+) failed", summary)
    skipped = int(skipped_m.group(1)) if skipped_m else 0
    failed = int(failed_m.group(1)) if failed_m else 0
    print("  gate: {} passed, {} skipped, {} failed in {:.1f}s".format(
        passed, skipped, failed, elapsed))
    if failed or proc.returncode not in (0,):
        for line in proc.stdout.splitlines():
            if line.startswith("FAILED "):
                print("      {}".format(line[:100]))
        raise RunnerError(
            "the acceptance gate failed: {} failed, pytest exited {}".format(
                failed, proc.returncode))
    return passed, skipped, failed


def _assert_hygiene(repo: Path, phase: str) -> None:
    """The no-detritus invariant, as a transaction gate.

    Run PROSPECTIVELY inside the transaction, where a violation still has a
    rollback, and ATTESTED after commit, where it proves the committed state.
    """
    detritus = sorted(H.iter_repository_detritus(repo))
    if detritus:
        raise RunnerError(
            "{} hygiene: {} backup-shaped file(s) present: {}".format(
                phase, len(detritus), detritus))
    print("  {} hygiene: no detritus".format(phase))


def build_plan(repo: Path, payload_dir: Path, before: int, after: int,
               pre_exists: dict) -> InstallPlan:
    """Assemble the COMPLETE declared transition, including both counters.

    `pre_exists` is captured BEFORE any mutation. An earlier version of this
    function asked the live filesystem, which by then held the staged payload,
    so a freshly CREATED file was described in the plan as a PATCH. A plan that
    misdescribes its own action is not a description of the transition.
    """
    count = DerivedCount(before=before, after=after)
    if count.delta <= 0:
        raise RunnerError(
            "the payload adds {} case(s); a test-adding unit must add at least "
            "one".format(count.delta))

    targets = []
    for source_name, dest in PAYLOAD:
        source = payload_dir / source_name
        if not source.is_file():
            raise RunnerError("payload file missing: {}".format(source))
        data = source.read_bytes()
        non_ascii = sum(1 for b in data if b > 0x7F)
        if non_ascii:
            raise RunnerError("{} carries {} non-ASCII byte(s)".format(
                source_name, non_ascii))
        if dest not in pre_exists:
            raise RunnerError(
                "no pre-mutation existence was captured for {}".format(dest))
        action = (TargetAction.PATCH if pre_exists[dest]
                  else TargetAction.CREATE)
        targets.append(PlannedTarget(dest, action, data, "payload"))

    ratchet_path = repo / "tests" / "EXPECTED_SUITE_SIZE"
    entry = RATCHET_ENTRY.format(
        date=time.strftime("%Y-%m-%d"), before=before, after=after,
        delta=count.delta)
    targets.append(PlannedTarget(
        "tests/EXPECTED_SUITE_SIZE", TargetAction.PATCH,
        render_ratchet(ratchet_path.read_bytes(), entry, count.after),
        "ratchet, rendered from the measured count"))

    readme_path = repo / "README.md"
    targets.append(PlannedTarget(
        "README.md", TargetAction.PATCH,
        render_readme(readme_path.read_bytes(), count.before, count.after),
        "badge, rendered from the SAME measured count"))

    return InstallPlan(unit=UNIT, targets=tuple(targets),
                       expected_delta=count.delta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--payload", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    repo = Path(args.repo_root).resolve()
    payload_dir = Path(args.payload).resolve()
    paths = resolve_runtime_paths(project_root=repo)
    journal = paths.transaction_journal

    print("  repository : {}".format(repo))
    print("  journal    : {}  (inside repo: {})".format(
        journal, str(journal).startswith(str(repo))))
    if str(journal).startswith(str(repo)):
        raise RunnerError(
            "the transaction journal resolves INSIDE the repository; the "
            "hygiene invariant cannot then distinguish residue from live "
            "recovery state")

    pending = incomplete_transactions(journal)
    if pending:
        raise RunnerError(
            "{} unresolved transaction journal(s); reconcile before "
            "installing: {}".format(
                len(pending), [p.get("transaction_id", "?") for p in pending]))

    _assert_hygiene(repo, "pre-install")

    # PRE-MUTATION STATE, captured while the tree is pristine. Every action in
    # the plan is decided from this, never from a filesystem that a staging
    # step has already changed.
    pre_exists = {dest: (repo / dest).exists() for _, dest in PAYLOAD}
    print("  pre-state  : {}".format(
        {k: ("exists" if v else "absent") for k, v in pre_exists.items()}))

    print("\n--- phase 1: measurement ---")
    before = _collect(repo, COUNTER_SCOPE)

    # THE MEASURED SCOPE MUST BE THE COUNTER'S SCOPE.
    #
    # If these disagree, the runner is measuring something the ratchet does not
    # describe, and every number it derives afterwards is wrong in a way that
    # renders cleanly. Refusing here costs one collection; not refusing writes
    # a false counter into the repository inside a transaction.
    ratchet_path = repo / "tests" / "EXPECTED_SUITE_SIZE"
    bare = [l.strip() for l in ratchet_path.read_text(encoding="utf-8").split("\n")
            if l.strip().isdigit()]
    if len(bare) != 1:
        raise RunnerError(
            "the ratchet holds {} bare number(s); expected exactly 1".format(len(bare)))
    if int(bare[0]) != before:
        raise RunnerError(
            "the ratchet reads {} but {} collects {}. The runner would be "
            "deriving counters from a scope the ratchet does not describe."
            .format(bare[0], COUNTER_SCOPE, before))
    print("  the ratchet ({}) agrees with {}".format(bare[0], COUNTER_SCOPE))

    # The AFTER count can only be measured with the payload staged, and staging
    # is a repository write. So it happens in a MEASUREMENT transaction that is
    # always rolled back -- the tree is pristine again before the plan is
    # validated, which is what lets validate_against() run against the state
    # the plan actually describes.
    with RepositoryTransaction(repo, journal) as probe:
        print("  measurement transaction {}".format(probe.transaction_id[:12]))
        for source_name, dest in PAYLOAD:
            data = (payload_dir / source_name).read_bytes()
            if pre_exists[dest]:
                probe.patch(dest, data)
            else:
                probe.create(dest, data)
        after = _collect(repo, COUNTER_SCOPE)
        probe.rollback(reason="measurement complete")
    print("  measurement rolled back; tree pristine again")
    _assert_hygiene(repo, "post-measurement")

    print("\n--- phase 2: plan ---")
    plan = build_plan(repo, payload_dir, before, after, pre_exists)
    plan.validate_against(repo)
    print("  plan digest: {}".format(plan.digest[:16]))
    print("  validated against the pristine tree")
    for t in plan.targets:
        print("    {:<6} {:<38} {}".format(
            t.action.value, t.relpath[:38], t.reason))

    if not args.apply:
        print("\n  --check: nothing written.")
        print(json.dumps(plan.describe(), indent=2, sort_keys=True))
        _assert_hygiene(repo, "post-check")
        return 0

    print("\n--- phase 3: apply ---")
    with RepositoryTransaction(repo, journal) as tx:
        print("  apply transaction {}".format(tx.transaction_id[:12]))
        for target in plan.targets:
            if target.action is TargetAction.PATCH:
                tx.patch(target.relpath, target.payload)
            else:
                tx.create(target.relpath, target.payload)
        plan.assert_write_set(tx)
        print("  write set  : proven equal to the declared set")

        print("\n--- acceptance gate (rollback still available) ---")
        passed, skipped, _failed = _run_suite(
            repo, ACCEPTANCE_SCOPE, active_transaction=tx.transaction_id)
        _assert_hygiene(repo, "prospective")
        tx.commit()
        print("  committed; journal destroyed: {}".format(
            not tx.directory.exists()))

    print("\n--- attestation ---")
    _assert_hygiene(repo, "attested")
    remaining = incomplete_transactions(journal)
    if remaining:
        raise RunnerError("journals remain after commit: {}".format(remaining))
    print("  no journals remain")
    print("\n  ratchet {} -> {}, gate {} passed / {} skipped".format(
        before, after, passed, skipped))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RunnerError, PlanError, WriteSetViolation, TransactionError) as exc:
        print("\n  REFUSED: {}".format(exc))
        sys.exit(1)
