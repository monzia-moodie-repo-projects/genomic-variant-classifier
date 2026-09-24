"""Hold the approved plan INDEPENDENTLY. Recompute what the bytes establish.

Author: Monzia Moodie

WHY THIS IS A REBUILD, NOT AN EXTENSION
========================================
An INDEPENDENT forensic probe was run against this exact module on
2026-09-16, using the SAME conventions this project already applies to
itself: run the adversarial case, do not read the code and reason about it.
Six probes produced ZERO findings against real gaps:

    arbitrary_valid_length_digest    a fabricated digest, well-formed,
                                      PASSED -- nothing recomputed it
    rejected_flag_not_checked        `accepted` was recorded and NEVER READ
    empty_continuation               (closed already; retained as a case)
    arbitrary_continuation           a syntactically valid but WRONG token
                                      passed, because the chain was checked
                                      for SHAPE only -- "the adapter retains
                                      the request, not the response body"
    out_of_order_sequences           [1, 3] and [2, 1] both verified CLEAN;
                                      the check was `sorted(seen) == range`,
                                      a MULTISET check, not an ORDER check
    complete_without_captures        a COMPLETE claim with zero captures
                                      exited 0 with verified: []

Each is the SAME root defect the previous rebuild exists to name: a claim
with no independent recomputation behind it is unfalsifiable. The fix here
is structural, following the ruling of 2026-09-16: retain the actual bytes,
recompute everything from them, and never let the producer's own claims --
`accepted`, `findings`, `health` -- stand in for that recomputation.

THE FOUR-OBJECT DIVISION
=========================
    Raw observation      the adapter's TargetResult + captures: what the
                          producer CLAIMED, preserved even when wrong
    Verifier (HERE)       what the retained captures ESTABLISH under the
                          approved plan, recomputed from bytes
    Supervisor            whether required evidence obligations were met
    Delivery subsystem    publication and acknowledgement status

A raw observation must remain deserializable even when self-contradictory --
`TargetResult(health=COMPLETE, captures=())` is preserved AS EVIDENCE OF A
DEFECT, not refused at construction. Refusing it there would make the defect
disappear instead of being recorded. Qualification is this module's job, not
the producer object's.

THREE SEPARATE QUESTIONS, NEVER COLLAPSED
==========================================
    traversal_completeness    did the walk reach a valid terminal page,
                               with an intact, in-order, value-checked chain?
    eligible_for_existence    can "a newer release EXISTS" be claimed from
                               at least one independently valid witness?
    eligible_for_absence      can "no newer release exists" be claimed --
                               which needs the FULL completed traversal?

A partial traversal can support an existence claim. A completed traversal
that finds nothing supports an absence claim. These are not the same
question, and collapsing them is how "the assay completed" comes to mean
"the assay answered the question" -- which it does not.

INDEPENDENCE IS THE WHOLE POINT, INCLUDING FOR THE GRAMMAR
============================================================
This module does not import BASE_QUERY, EXPECTED_KIND, APPROVED_BASELINE,
`parse_version`, or `_validate_page` from the adapter. It re-declares the
plan and reimplements structural validation and release-grammar parsing on
its own. A verifier that shares code with the thing it verifies cannot
detect a change to that shared code. Agreement between the two independent
declarations is a TESTED PROPERTY (see test_monitoring.py), not an
assumption -- the same pattern already used for the endpoint, field mask,
and budgets.

WHAT A PASS ESTABLISHES, AND WHAT IT DOES NOT
=============================================
    ESTABLISHES  every retained request matches the approved plan; the
                 retained capture sequence is 1..n IN OBSERVED ORDER; each
                 request's continuation token equals the ACTUAL token the
                 PRECEDING capture's own body declared; the retained body's
                 digest matches its declared value; structure and acceptance
                 are independently recomputed, never trusted from the
                 producer; and a completed traversal reached a genuine
                 terminal page by this exact chain.

    DOES NOT     that the bytes were received from the named source. A
                 digest matching its own declared value proves the retained
                 bytes are SELF-CONSISTENT, not that they came from gnomAD.
                 Authenticating the source needs a transport that signs what
                 it saw, which does not exist here.

    DOES NOT     that a completed traversal is a point-in-time snapshot.
                 Cloud Storage's own listing documentation describes objects
                 created during pagination as possibly absent once their
                 namespace position has already been traversed. This
                 establishes completion of THIS traversal under that API's
                 semantics, not snapshot consistency.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from urllib.parse import parse_qsl, urlsplit

from .reason_catalog import Reason

# ---------------------------------------------------------------------------
# THE APPROVED PLAN, DECLARED HERE. Not imported. See the module docstring.
# ---------------------------------------------------------------------------
APPROVED_ENDPOINT = ("https://storage.googleapis.com/storage/v1/b/"
                     "gcp-public-data--gnomad/o")
APPROVED_QUERY = {"prefix": "release/", "delimiter": "/", "maxResults": "1000",
                  "fields": "kind,prefixes,nextPageToken"}
MAX_BYTES_PER_PAGE = 4 * 1024 * 1024
MAX_TOKEN_CHARS = 8192

#: Independently declared release grammar and baseline. See
#: test_the_verifier_and_the_adapter_declare_the_SAME_baseline_and_kind for
#: the agreement test.
EXPECTED_KIND = "storage#objects"
#: APPROVED 2026-09-24 (owner; first recorded in the rulings preserved 2026-09-22). APPROVAL IS NOT ADOPTION:
#: every gnomAD artifact the project USES is still v4.1, and a release change for constraint does not by
#: itself establish one for every frequency product. See
#: docs/measurements/DECISION_2026-09-24_gnomad-4.1.1-approval.md. Anything NEWER than this still alerts.
APPROVED_BASELINE = "4.1.1"


def _independent_parse_release_version(prefix):
    """A SEPARATE implementation of the release grammar, not imported.

    Written independently of gnomad_release_check.parse_version so the two
    can drift-detect each other. See the differential test in
    test_monitoring.py, which runs both against one battery of inputs.
    """
    name = str(prefix).strip("/").split("/")[-1]
    if name.startswith("v"):
        name = name[1:]
    if not name:
        return None
    parts = []
    for chunk in name.split("."):
        if not chunk.isdigit():
            return None
        parts.append(int(chunk))
    return tuple(parts)


class StructuralDisposition(str, Enum):
    """The verifier's OWN determination. Never the producer's `accepted`."""

    VALID = "valid"
    INVALID = "invalid"
    UNESTABLISHED = "unestablished"    # no retained body to examine


@dataclass(frozen=True)
class PlanFinding:
    """One reason a capture, or the run as a whole, does not match the plan
    or does not survive independent recomputation."""

    sequence: int
    reason: object
    detail: str

    def as_document(self):
        return {"sequence": self.sequence,
                "reason": self.reason.value if hasattr(self.reason, "value")
                else self.reason,
                "detail": self.detail}


class TraversalCompleteness(str, Enum):
    """Did the walk reach a valid terminal page by an intact chain?

    UNESTABLISHED: zero usable captures -- nothing to reason about.
    INCOMPLETE:    captures exist but the chain is broken, out of order, or
                   never reaches a genuine terminal page.
    COMPLETE:      every capture independently valid, in observed order, the
                   token chain matches value-for-value, and the last capture
                   in the list independently declares no continuation.
    """

    UNESTABLISHED = "unestablished"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


@dataclass(frozen=True)
class QualificationOutcome:
    """What the retained captures establish. Bound to one target, one plan.

    RULING 2026-09-16, requirement 8: "Bind the assessment to the attempt,
    target, approved plan, and captured evidence." `plan_fingerprint` is a
    digest of the approved endpoint, query AND baseline, so two outcomes
    computed under different plan revisions are never silently compared as if
    under one. The baseline is part of the plan: it decides what counts as a
    witness, so an outcome before the 4.1.1 approval and one after it are
    different plans (added 2026-09-24).
    """

    target: str
    plan_fingerprint: str
    traversal_completeness: TraversalCompleteness
    eligible_for_existence_claim: bool
    eligible_for_absence_claim: bool
    positive_witnesses: tuple
    findings: tuple
    captures_examined: int

    def as_document(self):
        return {
            "target": self.target,
            "plan_fingerprint": self.plan_fingerprint,
            "traversal_completeness": self.traversal_completeness.value,
            "eligible_for_existence_claim": self.eligible_for_existence_claim,
            "eligible_for_absence_claim": self.eligible_for_absence_claim,
            "positive_witnesses": list(self.positive_witnesses),
            "findings": [f.as_document() for f in self.findings],
            "captures_examined": self.captures_examined,
            "does_not_establish": [
                "source authenticity: a digest matching its own declared "
                "value proves the retained bytes are self-consistent, not "
                "that they came from the named source",
                "that a completed traversal is a point-in-time snapshot; "
                "Cloud Storage's own listing contract permits objects "
                "created during pagination to be absent",
                "that a newer release PREFIX names a usable product",
            ],
        }


def _plan_fingerprint():
    blob = json.dumps({"endpoint": APPROVED_ENDPOINT, "query": APPROVED_QUERY,
                       "approved_baseline": APPROVED_BASELINE},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _independently_validate_structure(raw):
    """Reimplemented, not imported. Returns (disposition, prefixes, token,
    reason, detail). `prefixes` and `token` are None unless VALID."""
    def _no_duplicates(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError("duplicate JSON key: {}".format(key))
            seen.add(key)
        return dict(pairs)

    try:
        page = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except ValueError as exc:
        reason = (Reason.RESPONSE_JSON_DUPLICATE if "duplicate" in str(exc)
                  else Reason.RESPONSE_JSON_SYNTAX)
        return (StructuralDisposition.INVALID, None, None, reason, str(exc))
    except Exception as exc:
        return (StructuralDisposition.INVALID, None, None,
                Reason.RESPONSE_JSON_SYNTAX,
                "{}: {}".format(type(exc).__name__, exc))

    if type(page) is not dict:
        return (StructuralDisposition.INVALID, None, None,
                Reason.RESPONSE_UNEXPECTED_SHAPE, "page is not an object")
    if page.get("kind") != EXPECTED_KIND:
        return (StructuralDisposition.INVALID, None, None,
                Reason.RESPONSE_UNEXPECTED_SHAPE,
                "kind is {!r}, expected {!r}".format(
                    page.get("kind"), EXPECTED_KIND))
    prefixes = page.get("prefixes", [])
    if type(prefixes) is not list or any(type(p) is not str for p in prefixes):
        return (StructuralDisposition.INVALID, None, None,
                Reason.RESPONSE_UNEXPECTED_SHAPE,
                "prefixes is not a list of strings")
    token = page.get("nextPageToken")
    if token is not None:
        if type(token) is not str or not token or len(token) > MAX_TOKEN_CHARS:
            return (StructuralDisposition.INVALID, None, None,
                    Reason.TRAVERSAL_TOKEN_MALFORMED,
                    "nextPageToken is {!r}".format(token))
    return (StructuralDisposition.VALID, tuple(prefixes), token, None, "")


def verify_plan_conformance(captures):
    """Request-plan checks: endpoint, query, fragment, and per-capture
    payload shape (size, digest FORMAT only -- integrity is `qualify`'s
    job). Sequence ORDER is checked here too, since it is a property of the
    retained list regardless of body content.

    Returns a tuple of PlanFinding. EVERY independent check runs for every
    capture -- see the module docstring on why a first-failure return is
    itself a defect class.
    """
    findings = []

    observed_order = []
    type_valid = True
    for cap in captures:
        v = cap.get("sequence") if type(cap) is dict else None
        # `True == 1` in Python, so a LIST-EQUALITY order check alone would
        # accept a bool where an int belongs. The type is checked separately
        # from the order comparison for exactly that reason.
        if type(v) is not int or type(v) is bool:
            type_valid = False
        observed_order.append(v)
    expected_order = list(range(1, len(captures) + 1))
    if not type_valid or observed_order != expected_order:
        # ORDER, not membership. MEASURED: sorted(seen) == range(1, n+1)
        # passed [1, 3] and passed a REVERSED [3, 2, 1] list once sorted.
        findings.append(PlanFinding(
            0, Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID,
            "observed sequence order {} is not {} -- gaps, repeats, "
            "non-integers, wrong types, or a violated ORDER all land "
            "here".format(observed_order, expected_order)))

    for index, cap in enumerate(captures, start=1):
        if type(cap) is not dict:
            findings.append(PlanFinding(index, Reason.REQUEST_URL_SYNTAX,
                                        "capture is not an object"))
            continue
        seq = cap.get("sequence", index)
        _check_payload_shape(findings, seq, cap)

        url = cap.get("request_url")
        if type(url) is not str or not url:
            findings.append(PlanFinding(seq, Reason.REQUEST_URL_SYNTAX,
                                        "no request_url retained"))
            continue
        try:
            parts = urlsplit(url)
        except ValueError as exc:
            findings.append(PlanFinding(seq, Reason.REQUEST_URL_SYNTAX, str(exc)))
            continue
        approved = urlsplit(APPROVED_ENDPOINT)
        if (parts.scheme, parts.netloc, parts.path) != \
                (approved.scheme, approved.netloc, approved.path):
            findings.append(PlanFinding(
                seq, Reason.REQUEST_ENDPOINT,
                "endpoint is {}://{}{}".format(parts.scheme, parts.netloc,
                                               parts.path)))
        if "#" in url:
            findings.append(PlanFinding(seq, Reason.REQUEST_FRAGMENT,
                                        "request carries a fragment"))
        try:
            pairs = parse_qsl(parts.query, keep_blank_values=True,
                              strict_parsing=True, max_num_fields=12)
        except ValueError as exc:
            findings.append(PlanFinding(seq, Reason.REQUEST_QUERY_SYNTAX, str(exc)))
            continue
        if len({k for k, _ in pairs}) != len(pairs):
            findings.append(PlanFinding(seq, Reason.REQUEST_QUERY_DUPLICATE,
                                        "duplicate query parameter"))
            continue
        got = dict(pairs)
        got.pop("pageToken", None)
        if got != APPROVED_QUERY:
            differing = sorted(set(got.items()) ^ set(APPROVED_QUERY.items()))
            findings.append(PlanFinding(
                seq, Reason.REQUEST_QUERY_MISMATCH,
                "query differs from the approved plan: {}".format(differing[:4])))

    return tuple(findings)


def _check_payload_shape(findings, seq, cap):
    size = cap.get("response_bytes")
    if type(size) is not int or type(size) is bool or size < 0:
        findings.append(PlanFinding(seq, Reason.RESPONSE_UNEXPECTED_SHAPE,
                                    "response_bytes is {!r}".format(size)))
    elif size > MAX_BYTES_PER_PAGE:
        findings.append(PlanFinding(seq, Reason.TRAVERSAL_BUDGET_EXHAUSTED,
                                    "page exceeded the declared budget"))
    digest = cap.get("response_sha256")
    if type(digest) is not str or len(digest) != 64 or \
            any(c not in "0123456789abcdef" for c in digest):
        findings.append(PlanFinding(
            seq, Reason.ARTIFACT_DIGEST_MISMATCH,
            "response_sha256 is not a 64-character lowercase digest"))


def _request_token(url):
    """The pageToken this request carries, or None. Best-effort: a syntax
    defect here is already reported by verify_plan_conformance."""
    try:
        pairs = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        return pairs.get("pageToken")
    except Exception:
        return None


def qualify(target, captures):
    """Recompute what the retained captures establish. THE central function.

    Ignores the producer's `health`, `accepted`, and `findings` entirely as
    inputs to the outcome -- they are read only to detect and report
    DISAGREEMENT with what is independently recomputed here.
    """
    plan_findings = list(verify_plan_conformance(captures))
    order_broken = any(f.reason is Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID
                       for f in plan_findings)
    endpoint_bad = {f.sequence for f in plan_findings
                    if f.reason in (Reason.REQUEST_ENDPOINT,
                                    Reason.REQUEST_QUERY_MISMATCH)}

    baseline = _independent_parse_release_version(
        "release/{}/".format(APPROVED_BASELINE))

    witnesses = []
    seen_witness_versions = set()
    own_tokens = {}            # sequence -> the token THIS capture's own
                               # body independently declares (None if none
                               # or unavailable)
    terminal_sequences = set()
    all_valid_in_order = not order_broken

    for index, cap in enumerate(captures, start=1):
        if type(cap) is not dict:
            all_valid_in_order = False
            continue
        seq = cap.get("sequence", index)

        body_retained = bool(cap.get("body_retained"))
        body_b64 = cap.get("response_body_b64") or ""
        declared_digest = cap.get("response_sha256")

        if not body_retained or not body_b64:
            plan_findings.append(PlanFinding(
                seq, Reason.EVIDENCE_BODY_UNAVAILABLE,
                "no retained body for capture {}; structure and acceptance "
                "are unestablished, not confirmed or denied".format(seq)))
            all_valid_in_order = False
            continue

        try:
            raw = base64.b64decode(body_b64, validate=True)
        except Exception as exc:
            plan_findings.append(PlanFinding(
                seq, Reason.EVIDENCE_BODY_UNAVAILABLE,
                "retained body is not valid base64: {}".format(exc)))
            all_valid_in_order = False
            continue

        # 1. INTEGRITY: recompute the digest from the bytes we actually hold.
        recomputed = hashlib.sha256(raw).hexdigest()
        if type(declared_digest) is str and recomputed != declared_digest:
            plan_findings.append(PlanFinding(
                seq, Reason.EVIDENCE_INTEGRITY_MISMATCH,
                "declared digest {} does not match sha256(retained body) "
                "{}".format(declared_digest[:16], recomputed[:16])))
            # NOT fatal to structural validation: we validate the bytes we
            # actually hold, which is the only evidence that exists.

        # 2. STRUCTURE + ACCEPTANCE, recomputed independently.
        disposition, prefixes, token, bad_reason, bad_detail = \
            _independently_validate_structure(raw)
        ind_accepted = disposition is StructuralDisposition.VALID

        declared_accepted = cap.get("accepted")
        if type(declared_accepted) is bool and declared_accepted != ind_accepted:
            plan_findings.append(PlanFinding(
                seq, Reason.EVIDENCE_ACCEPTANCE_DISAGREEMENT,
                "producer claimed accepted={!r}; independent structural "
                "recomputation says {!r}".format(declared_accepted,
                                                 ind_accepted)))

        if not ind_accepted:
            plan_findings.append(PlanFinding(seq, bad_reason, bad_detail))
            all_valid_in_order = False
            own_tokens[seq] = None
            continue

        own_tokens[seq] = token
        if token is None:
            terminal_sequences.add(seq)

        if seq not in endpoint_bad:
            for p in prefixes:
                v = _independent_parse_release_version(p)
                if v is not None and baseline is not None and v > baseline:
                    if v not in seen_witness_versions:
                        seen_witness_versions.add(v)
                        witnesses.append(".".join(str(x) for x in v))

    # 3. TOKEN CHAIN, BY VALUE. Capture i's request token must equal
    # capture (i-1)'s OWN body's declared token, in OBSERVED LIST ORDER.
    chain_intact = True
    if not order_broken:
        for i, cap in enumerate(captures):
            seq = cap.get("sequence", i + 1) if type(cap) is dict else i + 1
            request_token = _request_token(cap.get("request_url", "")) \
                if type(cap) is dict else None
            if i == 0:
                if request_token is not None:
                    plan_findings.append(PlanFinding(
                        seq, Reason.EVIDENCE_TOKEN_CHAIN_MISMATCH,
                        "the first request carries a pageToken"))
                    chain_intact = False
                continue
            prev_seq = captures[i - 1].get("sequence", i) \
                if type(captures[i - 1]) is dict else i
            if prev_seq in terminal_sequences:
                plan_findings.append(PlanFinding(
                    seq, Reason.EVIDENCE_TOKEN_CHAIN_MISMATCH,
                    "a capture follows a terminal page"))
                chain_intact = False
                continue
            prev_token = own_tokens.get(prev_seq)
            if prev_token is None:
                # The preceding page was unusable (no body, or rejected) --
                # the chain cannot be confirmed from here. Not a MATCH claim.
                chain_intact = False
                continue
            if request_token != prev_token:
                plan_findings.append(PlanFinding(
                    seq, Reason.EVIDENCE_TOKEN_CHAIN_MISMATCH,
                    "request token does not equal the preceding capture's "
                    "own declared continuation token"))
                chain_intact = False
    else:
        chain_intact = False

    last_is_terminal = bool(captures) and \
        (captures[-1].get("sequence") if type(captures[-1]) is dict
         else None) in terminal_sequences

    if not captures:
        completeness = TraversalCompleteness.UNESTABLISHED
    elif (all_valid_in_order and chain_intact and last_is_terminal
          and not endpoint_bad):
        completeness = TraversalCompleteness.COMPLETE
    else:
        completeness = TraversalCompleteness.INCOMPLETE

    # Existence is supported the moment ANY witness was independently
    # extracted from a usable page -- regardless of overall completeness.
    # RULING 2026-09-16: "A partial traversal can support a valid existence
    # claim." COMPLETE with zero witnesses does NOT retroactively manufacture
    # one; existence and completeness are read from different evidence.
    eligible_for_existence = len(witnesses) > 0
    eligible_for_absence = completeness is TraversalCompleteness.COMPLETE

    return QualificationOutcome(
        target=target,
        plan_fingerprint=_plan_fingerprint(),
        traversal_completeness=completeness,
        eligible_for_existence_claim=eligible_for_existence,
        eligible_for_absence_claim=eligible_for_absence,
        positive_witnesses=tuple(witnesses),
        findings=tuple(plan_findings),
        captures_examined=len(captures),
    )


def verify_captures(captures):
    """RETAINED for the callers built against the previous interface: the
    plan-conformance subset of `qualify`'s findings. Prefer `qualify` for
    anything deciding qualification -- this does not recompute digests,
    structure, acceptance, or the token chain by value."""
    return verify_plan_conformance(captures)
