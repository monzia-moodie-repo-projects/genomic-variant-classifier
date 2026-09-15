"""Hold the approved plan INDEPENDENTLY. Check the captures against it.

Author: Monzia Moodie

WHY THIS EXISTS
===============
The adapter retains `request_url`, `response_sha256` and `response_bytes` for
every page. Nothing checks them. A report that carries evidence nobody
verifies is the same unfalsifiable shape as a status field nobody reads --
which is what `status: "ok"` was in VersionMonitorAgent, and what
`[OK] action=error` was in run_pipeline.

The evidence exists so a verifier can REFUSE. This is that verifier.

INDEPENDENCE IS THE WHOLE POINT
===============================
This module builds the approved query from its OWN declaration. It does not
import the adapter's BASE_QUERY, because a verifier that derives its
expectation from the subject cannot detect a subject that changed.

MEASURED 2026-09-14: an earlier request carried `fields=prefixes`, which OMITS
nextPageToken. Its absence therefore meant nothing, and the evidence of
truncation could not exist. A verifier holding the mask independently refuses
that request; a verifier reading the mask from the adapter would have approved
it, because it would have been comparing the adapter to itself.

WHAT A PASS ESTABLISHES, AND WHAT IT DOES NOT
=============================================
    ESTABLISHES  every retained request matches the approved plan, page by
                 page; the retained capture sequence is 1..n with no gaps or
                 repeats; the continuation chain is consistent with the tokens
                 the captures record; and no page exceeded its declared budget.

    DOES NOT     that the bytes were received from the named source. The
                 digest is self-reported: this process computed it over bytes
                 it already held. Authenticating the source needs a transport
                 that signs what it saw, which does not exist here.

    DOES NOT     that a completed traversal is a point-in-time snapshot.
                 Objects created in an already-traversed portion of the
                 namespace can be missed.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from .reason_catalog import Reason

#: DECLARED HERE, not imported from the adapter. See the module docstring.
APPROVED_ENDPOINT = ("https://storage.googleapis.com/storage/v1/b/"
                     "gcp-public-data--gnomad/o")
APPROVED_QUERY = {"prefix": "release/", "delimiter": "/", "maxResults": "1000",
                  "fields": "kind,prefixes,nextPageToken"}
MAX_BYTES_PER_PAGE = 4 * 1024 * 1024
MAX_TOKEN_CHARS = 8192


@dataclass(frozen=True)
class PlanFinding:
    """One reason a capture does not match the approved plan."""

    sequence: int
    reason: object
    detail: str

    def as_document(self):
        return {"sequence": self.sequence,
                "reason": self.reason.value if hasattr(self.reason, "value")
                else self.reason,
                "detail": self.detail}


def _check_payload(findings, seq, cap) -> None:
    """Size and digest checks. Independent of the request, so they run even
    when the URL or query could not be parsed."""
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


def verify_captures(captures) -> tuple:
    """Return a tuple of PlanFinding. EMPTY means every capture matched.

    EVERY INDEPENDENT CHECK RUNS for every capture, so the result is a
    COMPLETE inventory of what did not match -- NOT a first failure. MEASURED
    2026-09-15: an earlier version returned on the first failure, so a capture
    with a wrong query, a truncated digest AND an over-budget size produced ONE
    finding, and a caller reading a one-element tuple took it for a complete
    inventory. Only a check whose own input is unavailable is skipped; those
    are marked FATAL in the body.

    An empty capture list produces NO findings and establishes NOTHING -- the
    caller must not read "no findings" as "verified". That distinction is the
    reason this returns findings rather than a boolean.
    """
    findings = []

    # THE SEQUENCE IS A PROPERTY OF THE LIST, not of any single capture, so it
    # is checked once, before the loop. MEASURED 2026-09-15: captures numbered
    # [1, 3] verified CLEAN -- a page retained and then lost between the
    # adapter and the verifier, with nothing saying so.
    seen = []
    for index, cap in enumerate(captures, start=1):
        raw = cap.get("sequence") if type(cap) is dict else None
        if type(raw) is not int or type(raw) is bool or raw < 1:
            findings.append(PlanFinding(
                index, Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID,
                "capture {} has sequence {!r}".format(index, raw)))
        else:
            seen.append(raw)
    if seen and sorted(seen) != list(range(1, len(captures) + 1)):
        findings.append(PlanFinding(
            0, Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID,
            "retained sequences {} are not 1..{} without gaps or "
            "repeats".format(sorted(seen), len(captures))))

    expected_token = None          # the first request carries no pageToken

    for index, cap in enumerate(captures, start=1):
        if type(cap) is not dict:
            findings.append(PlanFinding(index, Reason.REQUEST_URL_SYNTAX,
                                        "capture is not an object"))
            continue
        seq = cap.get("sequence", index)
        url = cap.get("request_url")
        if type(url) is not str or not url:
            findings.append(PlanFinding(seq, Reason.REQUEST_URL_SYNTAX,
                                        "no request_url retained"))
            _check_payload(findings, seq, cap)
            continue                          # FATAL: no URL to parse

        try:
            parts = urlsplit(url)
        except ValueError as exc:
            findings.append(PlanFinding(seq, Reason.REQUEST_URL_SYNTAX, str(exc)))
            _check_payload(findings, seq, cap)
            continue                          # FATAL: URL unparseable
        approved = urlsplit(APPROVED_ENDPOINT)
        if (parts.scheme, parts.netloc, parts.path) != \
                (approved.scheme, approved.netloc, approved.path):
            findings.append(PlanFinding(
                seq, Reason.REQUEST_ENDPOINT,
                "endpoint is {}://{}{}".format(parts.scheme, parts.netloc,
                                               parts.path)))
            # NOT fatal: the query and the payload remain inspectable.
        if "#" in url:
            # A trailing '#' parses to an EMPTY fragment, which `or
            # parts.fragment` would miss. Checking the raw string catches it.
            findings.append(PlanFinding(seq, Reason.REQUEST_FRAGMENT,
                                        "request carries a fragment"))

        try:
            pairs = parse_qsl(parts.query, keep_blank_values=True,
                              strict_parsing=True, max_num_fields=12)
        except ValueError as exc:
            findings.append(PlanFinding(seq, Reason.REQUEST_QUERY_SYNTAX, str(exc)))
            _check_payload(findings, seq, cap)
            continue                          # FATAL: query unparseable
        if len({k for k, _ in pairs}) != len(pairs):
            findings.append(PlanFinding(seq, Reason.REQUEST_QUERY_DUPLICATE,
                                        "duplicate query parameter"))
            _check_payload(findings, seq, cap)
            continue                          # FATAL: the query is ambiguous

        got = dict(pairs)
        token = got.pop("pageToken", None)
        if got != APPROVED_QUERY:
            differing = sorted(
                set(got.items()) ^ set(APPROVED_QUERY.items()))
            findings.append(PlanFinding(
                seq, Reason.REQUEST_QUERY_MISMATCH,
                "query differs from the approved plan: {}".format(differing[:4])))
            # NOT fatal: the token chain and the payload still apply.

        # The continuation chain: page one carries no token, and each later
        # page must carry the token the PREVIOUS page reported.
        if expected_token is None:
            if token is not None:
                findings.append(PlanFinding(
                    seq, Reason.TRAVERSAL_TOKEN_MALFORMED,
                    "first request carries a pageToken"))
        else:
            if token is None:
                findings.append(PlanFinding(
                    seq, Reason.TRAVERSAL_TOKEN_MALFORMED,
                    "continuation request carries no pageToken"))
            elif len(token) > MAX_TOKEN_CHARS:
                findings.append(PlanFinding(
                    seq, Reason.TRAVERSAL_TOKEN_MALFORMED,
                    "pageToken exceeds {} characters".format(MAX_TOKEN_CHARS)))

        _check_payload(findings, seq, cap)

        # The NEXT page's expected token cannot be derived from this capture:
        # the adapter retains the request, not the response body. The chain is
        # therefore checked for SHAPE -- first page tokenless, later pages
        # tokened -- and not for VALUE. Saying so matters: a reader must not
        # take this for a stronger check than it is.
        expected_token = token if token is not None else "present"

    return tuple(findings)
