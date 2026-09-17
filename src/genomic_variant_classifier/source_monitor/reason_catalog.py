"""Stable failure identities that survive every reporting boundary.

Author: Monzia Moodie

WHY THIS EXISTS
===============
MEASURED 2026-09-14, across this project's monitoring layer:

    VersionMonitorAgent      status "ok" was a LITERAL, set unconditionally
    check_agents_active.py   22 of 22 agents STALE at 84.59 days -> "OK", exit 0
    run_pipeline             printed "[OK]" beside action=error
    run_data_freshness.py    returned 0 whether or not a change was detected
    _record_run_telemetry    ignored result["status"], so degraded became ok

And in code written the same day to repair it:

    _check_gnomad_release    an INVALID expected version read as "current"
                             a PAGINATED listing read as complete
                             the request itself omitted nextPageToken, so
                             the evidence of truncation could not exist

Every one is the same defect: a failure with no stable machine-readable
identity, so a boundary somewhere -- a string search, a literal, a lossy
translation, a runner's return value -- turned it into success.

THE THREE-LAYER RULE
====================
    Shared reason definitions   stable identity and meaning, defined ONCE
    Protocol profile            which reasons a producer MAY emit
    Supervisor                  contract findings about the records it receives

A profile references shared definitions; it never redefines them. A record
does not choose the catalog, profile, or allowed set that judges it.

    reason  -- read by machines; behaviour depends on it
    detail  -- read by people; changing it must change NOTHING

WHAT RECOGNITION MEANS
======================
A recognized failure record conforms to its expected vocabulary. It does NOT
mean the source check succeeded, and it does NOT mean the reported diagnosis
is true. A recognized failure exits 2. Always.

WHAT THIS DOES NOT ESTABLISH
============================
    * that a record belongs to the correct attempt, subject or policy -- the
      surrounding receipt must be bound before this assessor runs;
    * who emitted the record or at what stage -- role/stage authorization is
      a separate, later check;
    * crash durability or notification delivery -- persistence and delivery
      are tested at their own boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Shared reason definitions. ONE meaning per code, defined here and nowhere
# else. Names describe OBSERVED CONDITIONS: an endpoint mismatch does not say
# whether its cause was configuration, middleware, or interference.
# ---------------------------------------------------------------------------

class Reason(str, Enum):
    # request boundary
    REQUEST_ENDPOINT = "request.endpoint"
    REQUEST_FRAGMENT = "request.fragment"
    REQUEST_URL_SYNTAX = "request.url_syntax"
    REQUEST_QUERY_SYNTAX = "request.query_syntax"
    REQUEST_QUERY_DUPLICATE = "request.query_duplicate"
    REQUEST_QUERY_MISMATCH = "request.query_mismatch"
    # response boundary
    RESPONSE_JSON_DUPLICATE = "response.json_duplicate"
    RESPONSE_JSON_SYNTAX = "response.json_syntax"
    RESPONSE_JSON_VALUE = "response.json_value"
    RESPONSE_UNEXPECTED_SHAPE = "response.unexpected_shape"
    # traversal
    TRAVERSAL_TRUNCATED = "traversal.truncated"
    TRAVERSAL_TOKEN_MALFORMED = "traversal.token_malformed"
    TRAVERSAL_TOKEN_CYCLE = "traversal.token_cycle"
    TRAVERSAL_BUDGET_EXHAUSTED = "traversal.budget_exhausted"
    # transport
    TRANSPORT_UNREACHABLE = "transport.unreachable"
    TRANSPORT_TIMEOUT = "transport.timeout"
    # configuration
    CONFIG_INVALID_BASELINE = "config.invalid_baseline"
    # artifact
    ARTIFACT_DIGEST_MISMATCH = "artifact.digest_mismatch"
    ARTIFACT_MISSING = "artifact.missing"
    # retained evidence
    #
    # A GAP IS NOT TRUNCATION. Truncation is a known stopping point the
    # traversal reports; a gap in the retained capture sequence is a page that
    # was retained and then LOST between the adapter and the verifier, with
    # nothing saying so. Reusing TRAVERSAL_TRUNCATED would conflate a declared
    # limit with a silent loss, and a shared definition carries ONE meaning.
    #
    # MEASURED 2026-09-15: the verifier READ `sequence` for labelling and never
    # checked it. Captures numbered [1, 3] verified clean; two captures both
    # claiming sequence 1 produced findings a reader could not tell apart; and
    # "one" and -5 travelled into the report unchallenged.
    EVIDENCE_CAPTURE_SEQUENCE_INVALID = "evidence.capture_sequence_invalid"

    # MEASURED 2026-09-16 by an independent forensic probe run against this
    # exact catalog: five adversarial cases produced ZERO findings from the
    # verifier then in place. Each below closes one, and each is a DISTINCT
    # condition from its nearest neighbour -- conflating them would repeat the
    # "a gap is not truncation" error one probe cycle later.
    #
    #   ARTIFACT_DIGEST_MISMATCH   already means: the digest STRING is not a
    #                              64-character lowercase hex value (a FORMAT
    #                              check, no bytes involved).
    #   EVIDENCE_INTEGRITY_MISMATCH  the digest IS well-formed and does NOT
    #                              match sha256(retained body). Probe case
    #                              "arbitrary_valid_length_digest": a
    #                              well-formed fabrication passed every
    #                              existing check.
    EVIDENCE_INTEGRITY_MISMATCH = "evidence.integrity_mismatch"

    #: The retained body was never captured (over the per-page or per-run
    #: retention budget) or was stripped before reaching the verifier. Neither
    #: CONFIRMS nor DENIES structure or acceptance for that page -- it is
    #: unestablished, not failed. Conflating this with a structural failure
    #: would report a resource-management decision as a data defect.
    EVIDENCE_BODY_UNAVAILABLE = "evidence.body_unavailable"

    #: The producer's own `accepted` flag disagrees with the verifier's
    #: INDEPENDENT structural recomputation from the retained body. Probe case
    #: "rejected_flag_not_checked": `accepted` was recorded and never read.
    #: RULING 2026-09-16: "Do not make accepted authoritative. It is the
    #: producer's diagnostic. The verifier should recompute acceptance."
    #: This is the finding that recomputation disagreed; it is NOT a
    #: structural defect in the bytes themselves (those get their own
    #: RESPONSE_* code from the independent recomputation).
    EVIDENCE_ACCEPTANCE_DISAGREEMENT = "evidence.acceptance_disagreement"

    #: A later request's pageToken does not equal the value the PRECEDING
    #: capture's own response body declared. Distinct from
    #: TRAVERSAL_TOKEN_MALFORMED, which is a syntax defect in one token
    #: considered alone. Probe case "arbitrary_continuation": a syntactically
    #: valid token that does not match what was actually promised.
    EVIDENCE_TOKEN_CHAIN_MISMATCH = "evidence.token_chain_mismatch"

    #: MEASURED 2026-09-16: qualify() independently re-derives witnesses from
    #: the retained body; nothing compared them against the producer's own
    #: self-reported `findings` text before this code existed. A producer
    #: that retained honest bytes but LIED about what it found in them would
    #: have passed silently -- the same class of gap EVIDENCE_ACCEPTANCE_
    #: DISAGREEMENT closed for the `accepted` flag, applied here to findings.
    EVIDENCE_WITNESS_DISAGREEMENT = "evidence.witness_disagreement"

    #: MEASURED 2026-09-17, from an external ruling's own injected-producer
    #: probes: a producer reporting Health.COMPLETE with zero captures and
    #: zero self-reported findings produced exit 0, even though qualify()
    #: itself correctly returned traversal_completeness="unestablished" and
    #: both eligibility flags False. Confirmed directly: RunReport.exit_code
    #: derives entirely from the adapter's own health/findings, never from
    #: qualify()'s output -- Q3 wired the verifier into the FAILURE path
    #: (outcome.findings, witness disagreement) but never made its assessment
    #: authoritative for what counts as a CLEAN result. This code closes
    #: that: qualification that does not establish enough evidence forces an
    #: operational finding, regardless of what the producer itself claimed.
    EVIDENCE_QUALIFICATION_UNESTABLISHED = "evidence.qualification_unestablished"


#: Supervisor-owned findings. NEVER in a producer's allowed set. A worker that
#: emits one of these is asserting a health judgment it is not authorized to
#: make -- the same self-certification the transaction owner refuses.
class ContractFinding(str, Enum):
    INVALID_FAILURE_RECORD = "contract.invalid_failure_record"
    UNSUPPORTED_FAILURE_SCHEMA = "contract.unsupported_failure_schema"
    PROFILE_MISMATCH = "contract.profile_mismatch"
    UNKNOWN_REASON = "contract.unknown_reason"
    REASON_NOT_ALLOWED = "contract.reason_not_allowed"


FAILURE_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = frozenset({
    "schema_version", "kind", "profile_id", "profile_revision",
    "reason", "detail",
})


class ProfileError(ValueError):
    """The profile itself is malformed. Refused BEFORE ACTIVATION."""


@dataclass(frozen=True)
class ReasonProfile:
    """Which shared reasons a producer under this profile may emit.

    `known_codes` is the catalog the supervisor consults. `allowed_codes` is
    the subset this profile permits. A known-but-disallowed reason and an
    unknown reason are DIFFERENT findings.
    """

    profile_id: str
    revision: str
    known_codes: frozenset
    allowed_codes: frozenset

    def __post_init__(self) -> None:
        for name in ("profile_id", "revision"):
            v = getattr(self, name)
            if type(v) is not str or not v or v != v.strip():
                raise ProfileError("{} must be a non-empty stripped string".format(name))
        if type(self.known_codes) is not frozenset or \
                type(self.allowed_codes) is not frozenset:
            raise ProfileError("reason sets must be frozensets")
        catalog = {r.value for r in Reason}
        undefined = self.known_codes - catalog
        if undefined:
            raise ProfileError(
                "profile knows codes the shared catalog does not define: "
                "{}".format(sorted(undefined)))
        if not self.allowed_codes <= self.known_codes:
            raise ProfileError(
                "profile names undefined reasons: {}".format(
                    sorted(self.allowed_codes - self.known_codes)))
        supervisor_owned = {c.value for c in ContractFinding}
        leaked = self.allowed_codes & supervisor_owned
        if leaked:
            raise ProfileError(
                "profile permits SUPERVISOR-OWNED findings to a producer: "
                "{}".format(sorted(leaked)))


@dataclass(frozen=True)
class FailureAssessment:
    """What the supervisor concluded about ONE failure record."""

    reported_reason: Any
    profile_failure_recognized: bool
    contract_health: tuple

    @property
    def exit_code(self) -> int:
        # RECOGNIZING A FAILURE NEVER MAKES THE CHECK SUCCESSFUL.
        return 2


def make_failure_record(profile: ReasonProfile, reason: Reason,
                        detail: str) -> dict:
    """Build a record a producer may emit. Refuses what the profile forbids.

    This is the producer-side gate. The supervisor re-checks everything, so
    a producer that bypasses this function gains nothing.
    """
    if type(reason) is not Reason:
        raise TypeError("reason must be a Reason member, not {!r}".format(reason))
    if reason.value not in profile.allowed_codes:
        raise ProfileError(
            "{} does not permit {}".format(profile.profile_id, reason.value))
    if type(detail) is not str:
        raise TypeError("detail must be a string")
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "kind": "failure",
        "profile_id": profile.profile_id,
        "profile_revision": profile.revision,
        "reason": reason.value,
        "detail": detail,
    }


def assess_failure_record(record: Any, profile: ReasonProfile) -> FailureAssessment:
    """Assess a failure subrecord against an INDEPENDENTLY approved profile.

    The surrounding receipt must already satisfy decoding, authentication,
    attempt, subject, policy and catalog-binding requirements. Raw received
    evidence remains retained separately. The returned contract finding is the
    PRIMARY refusal, not an exhaustive fault inventory.

    The original reported reason is PRESERVED on every path, including the
    unknown-reason path. It is never replaced by the contract finding.
    """
    code = record.get("reason") if type(record) is dict else None
    code = code if type(code) is str else None

    def refused(finding: ContractFinding) -> FailureAssessment:
        return FailureAssessment(code, False, (finding.value,))

    if type(record) is not dict or set(record) != _REQUIRED_FIELDS:
        return refused(ContractFinding.INVALID_FAILURE_RECORD)
    sv = record["schema_version"]
    if type(sv) is not int or sv != FAILURE_SCHEMA_VERSION:
        # `type(sv) is not int` refuses True, which == 1 would accept.
        return refused(ContractFinding.UNSUPPORTED_FAILURE_SCHEMA)
    if record["kind"] != "failure":
        return refused(ContractFinding.INVALID_FAILURE_RECORD)
    for key in ("profile_id", "profile_revision", "reason"):
        v = record[key]
        if type(v) is not str or not v or v != v.strip():
            return refused(ContractFinding.INVALID_FAILURE_RECORD)
    if type(record["detail"]) is not str:
        return refused(ContractFinding.INVALID_FAILURE_RECORD)
    if (record["profile_id"], record["profile_revision"]) != \
            (profile.profile_id, profile.revision):
        return refused(ContractFinding.PROFILE_MISMATCH)
    if code not in profile.known_codes:
        return refused(ContractFinding.UNKNOWN_REASON)
    if code not in profile.allowed_codes:
        return refused(ContractFinding.REASON_NOT_ALLOWED)
    return FailureAssessment(code, True, ())
