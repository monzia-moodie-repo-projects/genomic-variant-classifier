"""Account for every required target. Guess nothing from a missing result.

Author: Monzia Moodie

WHY THIS EXISTS
===============
MEASURED 2026-09-14 across this project's monitoring layer, every instrument
that could have reported a broken agent reported success instead:

    check_agents_active.py   22 of 22 agents STALE at 84.59 days -> "OK", exit 0
    audit_agent_operational  structure only; never asks whether anything ran
    VersionMonitorAgent      status "ok" was a LITERAL, set unconditionally
    run_pipeline             printed "[OK]" beside action=error
    run_data_freshness.py    returned 0 whether or not a change was detected
    _record_run_telemetry    ignored result["status"], so degraded became ok

A monitoring layer that cannot fail cannot report. This supervisor's only
purpose is to fail when it should.

REQUIRED TARGETS COME FROM THE POLICY, NOT THE RESULTS
======================================================
The single most important rule here. If the required-target list were derived
from the results that arrived, a target that never ran would simply not be
required, and the supervisor would agree with itself that nothing is missing.

    required   <- approved policy
    arrived    <- this run's results
    missing    = required - arrived        ALWAYS a failure

THE THREE-WAY PROJECTION
========================
    0  every required target qualified; no actionable finding
    1  every required target qualified; review required
    2  required evidence, policy, or execution could not be qualified

The exit code is a PROJECTION of the structured report, not its storage.
Three numbers cannot encode the report, and the report is what is persisted.

A PARTIAL OBSERVATION MAY CARRY A VALID POSITIVE WITNESS
========================================================
Page one showing a newer release and page two timing out are two
simultaneously true statements:

    a newer release WAS observed          -> a finding worth keeping
    the inventory was NOT established     -> the observation is incomplete

Exit 2 AND retain the finding. Discarding the witness because coverage failed
loses real information; letting the witness make the observation complete
asserts coverage that was not achieved.

WHAT THIS DOES NOT ESTABLISH
============================
    * that a result belongs to the attempt, subject or policy it names --
      receipt binding happens before this;
    * that a reported diagnosis is true -- recognizing a vocabulary is not
      agreeing with its content;
    * that anything was delivered -- delivery is recorded separately, and a
      committed-but-undelivered finding is a PENDING DELIVERY, not a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Health(str, Enum):
    """Observation health. Independent of any scientific disposition."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    ABSENT = "absent"          # no result arrived for a required target


class SupervisorFinding(str, Enum):
    """Supervisor-owned. A producer may never emit one of these."""

    TARGET_ABSENT = "supervisor.target_absent"
    TARGET_UNDECLARED = "supervisor.target_undeclared"
    DUPLICATE_RESULT = "supervisor.duplicate_result"
    NO_REQUIRED_TARGETS = "supervisor.no_required_targets"
    DUPLICATE_REQUIRED = "supervisor.duplicate_required_target"
    RESULT_MALFORMED = "supervisor.result_malformed"


@dataclass(frozen=True)
class TargetResult:
    """One target's outcome, as reported by a qualified producer."""

    target: str
    health: Health
    findings: tuple = ()          # valid witnesses, kept even when incomplete
    reason: Any = None            # the producer's reason code, if it failed
    #: What a VERIFIER would need to check the claim. Retained bytes and the
    #: request that produced them -- not a demonstration of activity. A report
    #: that names a malformed page and retains nothing to check is
    #: unfalsifiable.
    captures: tuple = ()

    def __post_init__(self) -> None:
        if type(self.target) is not str or not self.target:
            raise ValueError("a target result must name its target")
        if type(self.health) is not Health:
            raise ValueError("health must be a Health member, not {!r}".format(
                self.health))
        # A CONTRADICTORY RESULT MUST NOT BE CONSTRUCTIBLE.
        #
        # MEASURED 2026-09-14: TargetResult("x", Health.COMPLETE,
        # reason=Reason.TRANSPORT_TIMEOUT) was accepted, and with no review
        # finding the supervisor returned EXIT 0. A completed observation that
        # names why it failed is not a completed observation.
        #
        # This closes one contradiction. It does NOT establish evidence
        # sufficiency -- a worker can still assert COMPLETE having examined
        # nothing. Qualification belongs to a verifier holding the approved
        # request plan and the retained captures.
        if self.health is Health.COMPLETE and self.reason is not None:
            raise ValueError(
                "a COMPLETE result cannot carry a failure reason: "
                "{!r}".format(self.reason))
        if self.health in (Health.INCOMPLETE, Health.FAILED) and \
                self.reason is None:
            raise ValueError(
                "a {} result must name its reason".format(self.health.value))


@dataclass(frozen=True)
class RunReport:
    """The structured report. The exit code is derived from this, not stored."""

    required: tuple
    results: tuple
    supervisor_findings: tuple = ()

    @property
    def unqualified(self) -> tuple:
        return tuple(r.target for r in self.results
                     if r.health in (Health.INCOMPLETE, Health.FAILED,
                                     Health.ABSENT))

    @property
    def review_findings(self) -> tuple:
        out = []
        for r in self.results:
            for f in r.findings:
                out.append((r.target, f))
        return tuple(out)

    @property
    def exit_code(self) -> int:
        if self.supervisor_findings or self.unqualified:
            return 2
        if self.review_findings:
            return 1
        return 0

    def as_document(self) -> dict:
        return {
            "schema": "gvc.monitor-run-report",
            "schema_version": 1,
            "required": list(self.required),
            "results": [
                {"target": r.target, "health": r.health.value,
                 "findings": list(r.findings),
                 "captures": list(r.captures),
                 "reason": (r.reason.value if hasattr(r.reason, "value")
                            else r.reason)}
                for r in self.results],
            "supervisor_findings": list(self.supervisor_findings),
            "unqualified": list(self.unqualified),
            "exit_code": self.exit_code,
            "does_not_establish": [
                "that any result belongs to the attempt, subject or policy it "
                "names; receipt binding happens before supervision",
                "that a reported diagnosis is true; recognizing a vocabulary "
                "is not agreeing with its content",
                "that anything was delivered; a committed-but-undelivered "
                "finding is a PENDING DELIVERY, not a success",
            ],
        }


def supervise(required_targets, results) -> RunReport:
    """Account for every required target against the results that arrived.

    `required_targets` comes from the APPROVED POLICY. It is never derived
    from `results` -- that inversion is what lets a system agree with itself
    that a target which never ran was never needed.
    """
    required = tuple(required_targets)
    supervisor_findings = []

    if not required:
        # A policy requiring nothing cannot fail, so it cannot report.
        supervisor_findings.append(SupervisorFinding.NO_REQUIRED_TARGETS.value)
    if len(set(required)) != len(required):
        # MEASURED 2026-09-15: supervise(("a","a"), [one result]) returned
        # EXIT 0 with no findings. The obligation set silently shrank from two
        # to one while reporting success.
        #
        # run_monitor.validate_configuration refuses a duplicated policy, but a
        # caller that does not go through the runner had no protection. The
        # guard belongs where the INVARIANT lives, not only in one caller.
        supervisor_findings.append(SupervisorFinding.DUPLICATE_REQUIRED.value)

    seen = {}
    ordered = []
    for r in results:
        if type(r) is not TargetResult:
            supervisor_findings.append(SupervisorFinding.RESULT_MALFORMED.value)
            continue
        if r.target in seen:
            supervisor_findings.append(SupervisorFinding.DUPLICATE_RESULT.value)
            continue
        seen[r.target] = r
        ordered.append(r)

    for target in required:
        if target not in seen:
            supervisor_findings.append(SupervisorFinding.TARGET_ABSENT.value)
            ordered.append(TargetResult(target, Health.ABSENT))

    for target in seen:
        if target not in required:
            supervisor_findings.append(
                SupervisorFinding.TARGET_UNDECLARED.value)

    return RunReport(required=required, results=tuple(ordered),
                     supervisor_findings=tuple(supervisor_findings))
