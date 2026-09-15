#!/usr/bin/env python3
"""Evidence crosses the store. Supervision reads what was recovered, not what was held.

Author: Monzia Moodie

WHY THIS IS A REBUILD
=====================
MEASURED 2026-09-14, against the previous runner:

    main() never called assess_failure_record()
        -- a reason catalog was built, tested, and never consulted.
    supervision consumed the in-memory TargetResult objects
        -- so persistence was a side effect, not a boundary anything crossed.
    nothing called record_delivery()
        -- every finding stayed pending forever; the retry queue only grew.
    the success heartbeat was sent BEFORE the report write could fail
        -- external success could precede the required output.

Each is the same defect in a different place: a component existed, was
correct, and was not on the path.

THE ORDER, AND EVERY STEP IS LOAD-BEARING
=========================================
    1. begin_attempt          an attempt STARTED -- not a success
    2. run each check          producing an observation
    3. commit_finding          durably, BEFORE any delivery
    4. RECOVER through the store's read interface
    5. ASSESS the recovered record under the approved profile
    6. supervise               required targets from POLICY, never from results
    7. write the report        the required output
    8. record_delivery         only after the report is on disk
    9. heartbeat               only after everything above
   10. exit                    the projection

Step 4 is not ceremony. A record that cannot be recovered and assessed is not
evidence, and discovering that AFTER reporting success is the failure this
ordering prevents. A separate-process recovery test is the qualification
control; a same-process recovery is the routine check.

Step 7 before step 9: a heartbeat that says "the run succeeded" must not be
sent while the report that proves it is still unwritten.

WHAT THIS DOES NOT ESTABLISH
============================
    * evidence sufficiency. Request and response bytes are NOT yet retained,
      so the report cannot independently establish request fidelity, response
      validity, or traversal completeness. A verifier needs those captures.
    * that anyone read the alarm.
    * crash durability. Process recovery is tested; power loss is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from genomic_variant_classifier.source_monitor.finding_store import (
    FindingStore, StoreError)
from genomic_variant_classifier.source_monitor.heartbeat import (
    signal_outcome, signal_start)
from genomic_variant_classifier.source_monitor.monitor_supervisor import (
    Health, TargetResult, supervise)
from genomic_variant_classifier.source_monitor.reason_catalog import (
    ContractFinding, Reason, ReasonProfile, assess_failure_record,
    make_failure_record)

# ---------------------------------------------------------------------------
# THE APPROVED POLICY. Required targets live here, NOT in the check registry.
#
# If these came from the registry, deleting a check would delete its
# obligation and the supervisor would agree that nothing is missing.
# ---------------------------------------------------------------------------
REQUIRED_TARGETS = (
    "gnomad-public-releases",
)

#: RECOGNITION uses the whole shared catalog; PERMISSION is the subset.
#: MEASURED 2026-09-14: a profile whose known_codes held a hand-picked subset
#: classified a globally defined code as UNKNOWN rather than
#: KNOWN-BUT-DISALLOWED -- different findings with different causes. The
#: previous profile also forbade TRAVERSAL_TOKEN_CYCLE, which the check can
#: emit, so a recognized source defect crashed the reporting path.
RELEASE_PROFILE = ReasonProfile(
    profile_id="release-listing",
    revision="r2",
    known_codes=frozenset(r.value for r in Reason),
    allowed_codes=frozenset({
        Reason.REQUEST_ENDPOINT.value,
        Reason.REQUEST_QUERY_MISMATCH.value,
        Reason.RESPONSE_JSON_DUPLICATE.value,
        Reason.RESPONSE_JSON_SYNTAX.value,
        Reason.RESPONSE_JSON_VALUE.value,
        Reason.RESPONSE_UNEXPECTED_SHAPE.value,
        Reason.TRAVERSAL_TRUNCATED.value,
        Reason.TRAVERSAL_TOKEN_MALFORMED.value,
        Reason.TRAVERSAL_TOKEN_CYCLE.value,
        Reason.TRAVERSAL_BUDGET_EXHAUSTED.value,
        Reason.TRANSPORT_UNREACHABLE.value,
        Reason.TRANSPORT_TIMEOUT.value,
        Reason.CONFIG_INVALID_BASELINE.value,
    }),
)

def default_store_path():
    """An ANCHORED store location. Never resolved against the caller's cwd.

    Prefers the project's own runtime-path resolver when it is importable, so
    this subsystem lands where every other piece of durable state does. Falls
    back to a per-user state directory -- not to a relative path, which is the
    defect this function exists to prevent.
    """
    try:
        from genomic_variant_classifier.paths.runtime_paths import (
            resolve_runtime_paths)
        base = Path(resolve_runtime_paths().state_dir)
    except Exception:
        base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("XDG_STATE_HOME")
            or (Path.home() / ".local" / "state")) / "GenomicVariantClassifier"
    return (base / "source_monitor" / "findings.sqlite3").resolve()


CHECKS: dict = {}


def register(target):
    def wrap(fn):
        if target in CHECKS:
            raise RuntimeError("duplicate check registration: {}".format(target))
        CHECKS[target] = fn
        return fn
    return wrap


def validate_configuration() -> None:
    """Refuse BEFORE activation. Not at registration -- that would make
    registration depend on import order."""
    missing = sorted(set(REQUIRED_TARGETS) - set(CHECKS))
    if missing:
        raise RuntimeError(
            "the approved policy requires targets with no implementation: "
            "{}".format(missing))
    if len(set(REQUIRED_TARGETS)) != len(REQUIRED_TARGETS):
        # MEASURED: duplicate required targets were accepted, so the
        # obligation set itself was never validated.
        raise RuntimeError("the approved policy lists a target twice")


@register("gnomad-public-releases")
def check_gnomad_releases(*, transport=None) -> TargetResult:
    from genomic_variant_classifier.source_monitor.gnomad_release_check import (
        observe_releases)
    return observe_releases(profile=RELEASE_PROFILE, transport=transport)


def _persist_and_recover(store, attempt, target, reason, detail):
    """Commit a failure, RECOVER it, and ASSESS it. Returns the assessment.

    The recovery is the point. A record that cannot be read back and
    recognized is not evidence, and finding that out after reporting success
    is exactly the ordering this prevents.
    """
    record = make_failure_record(RELEASE_PROFILE, reason, detail)
    committed = store.commit_finding(attempt_id=attempt, subject=target,
                                     record=record)
    recovered = store.get_finding(committed.event_id)["record"]
    assessment = assess_failure_record(recovered, RELEASE_PROFILE)
    return committed, assessment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default=None,
                   help="the finding store. Defaults to an ANCHORED runtime "
                        "state directory, NEVER a repository-relative path. "
                        "MEASURED 2026-09-14: an earlier default of "
                        "var/monitor/findings.sqlite3 wrote INSIDE the "
                        "repository, left it dirty, and the next installer "
                        "REFUSED at its working-tree precondition. A "
                        "monitoring subsystem must not dirty the repository it "
                        "monitors. This is LITERATURE-STATE-CWD-RELATIVE-1 in "
                        "a new subsystem: version_monitor_agent.py had "
                        "Path(\"data/agent_state.json\"), resolved against the "
                        "process working directory, and ONE logical store came "
                        "to exist at TWO depths with divergent contents.")
    p.add_argument("--report", default=None)
    p.add_argument("--heartbeat-url", default=os.environ.get("GVC_HEARTBEAT_URL"),
                   help="an external heartbeat endpoint. A scheduled job CANNOT "
                        "supervise its own absence. Defaults to "
                        "GVC_HEARTBEAT_URL so the value never appears in a "
                        "command line -- it is a BEARER CAPABILITY and a "
                        "holder can silence the alarm.")
    args = p.parse_args(argv)

    validate_configuration()
    started = signal_start(args.heartbeat_url)

    store = FindingStore(args.store or default_store_path())
    attempt = store.begin_attempt("monitor-run")

    results = []
    assessments = []
    committed_ids = []

    for target in REQUIRED_TARGETS:
        try:
            result = CHECKS[target]()
        except Exception as exc:
            # An unexpected exception is NOT evidence about transport. It is
            # an unhandled implementation fault, and calling it
            # transport.unreachable -- as the previous runner did -- asserts a
            # cause nobody measured.
            committed, assessment = _persist_and_recover(
                store, attempt, target, Reason.RESPONSE_UNEXPECTED_SHAPE,
                "unhandled check fault: {}: {}".format(type(exc).__name__, exc))
            committed_ids.append(committed.event_id)
            assessments.append((target, assessment))
            results.append(TargetResult(target, Health.FAILED,
                                        reason=Reason.RESPONSE_UNEXPECTED_SHAPE))
            continue

        if result.reason is not None:
            committed, assessment = _persist_and_recover(
                store, attempt, target, result.reason,
                "; ".join(result.findings) or result.health.value)
            committed_ids.append(committed.event_id)
            assessments.append((target, assessment))
            if not assessment.profile_failure_recognized:
                # The record did not survive its own contract. The observation
                # stands, but the reporting path is not qualified to speak for
                # it, so the run is unqualified.
                results.append(TargetResult(
                    target, Health.FAILED,
                    reason=Reason.RESPONSE_UNEXPECTED_SHAPE))
                continue
        results.append(result)

    report = supervise(REQUIRED_TARGETS, results)
    document = report.as_document()
    document["attempt_id"] = attempt
    document["store"] = str(Path(args.store).resolve())
    document["assessments"] = [
        {"target": t, "reason": a.reported_reason,
         "recognized": a.profile_failure_recognized,
         "contract_health": list(a.contract_health)}
        for t, a in assessments]
    document["profile"] = {"id": RELEASE_PROFILE.profile_id,
                           "revision": RELEASE_PROFILE.revision}
    document["does_not_establish"].append(
        "source authenticity: a response digest is integrity relative to bytes "
        "THIS PROCESS received and self-reported. It authenticates nothing "
        "about the source, and a verifier must hold the approved request plan "
        "independently to check the retained request_url against it")

    # THE REPORT IS THE REQUIRED OUTPUT. It is written BEFORE delivery is
    # recorded and BEFORE any heartbeat, so external success cannot precede it.
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
    if args.report:
        target_path = Path(args.report)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(raw + "\n", encoding="utf-8", newline="\n")

    # DELIVERY IS RECORDED ONLY AFTER THE REPORT EXISTS. The report is this
    # run's delivery channel: a finding a reader can open. Recording it before
    # the write would mark delivered a finding nobody can see.
    delivered = []
    if args.report:
        for event_id in committed_ids:
            try:
                store.record_delivery(event_id, "report:{}".format(target_path))
                delivered.append(event_id)
            except StoreError as exc:
                document.setdefault("delivery_errors", []).append(str(exc))

    store.finish_attempt(
        attempt,
        "complete" if report.exit_code == 0 else
        ("incomplete" if report.exit_code == 1 else "failed"))

    finished = signal_outcome(args.heartbeat_url, report.exit_code)
    document["heartbeat"] = {"start": started.as_document(),
                             "outcome": finished.as_document()}
    document["delivered"] = len(delivered)
    document["pending_deliveries"] = len(store.pending_deliveries())
    if args.report:
        target_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True))
    print("\nexit {}  ({})".format(
        report.exit_code,
        {0: "qualified; no action", 1: "qualified; review required",
         2: "NOT qualified"}[report.exit_code]), file=sys.stderr)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
