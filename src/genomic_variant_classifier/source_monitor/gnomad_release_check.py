"""Accumulate validated pages. An interruption must not erase what was proven.

Author: Monzia Moodie

WHY THIS IS A REBUILD, NOT AN EDIT
==================================
The previous implementation claimed, in three module docstrings and a test
named for it, that a partial observation keeps its valid witness. MEASURED
2026-09-14 against the LIVE TRANSPORT BRANCH:

    page one: release/4.1.1/ + nextPageToken
    page two: TimeoutError
    ->  health   FAILED                 (not INCOMPLETE)
        reason   TRANSPORT_UNREACHABLE  (not TRANSPORT_TIMEOUT)
        findings ('TimeoutError: second page',)     4.1.1 WAS DISCARDED

The guarantee failed on the only path that matters. It passed its test because
THE FIXTURE BRANCH BYPASSED THE REQUEST LOOP ENTIRELY: fixtures were loaded as
a ready-made list of pages, so the test never reached the code that lost the
accumulator.

A fixture that substitutes for a whole subsystem tests the fixture.

THE REPAIR IS STRUCTURAL
========================
Transport is injected BENEATH the adapter. Fixtures and live execution
traverse the SAME request builder, decoder, page validator, token-chain
validator and accumulator. There is no second path to diverge.

Each page is VALIDATED and ACCUMULATED before the next is requested, and every
exit preserves what was already proven alongside the reason it stopped.

WHAT A COMPLETED TRAVERSAL ESTABLISHES
======================================
Google Cloud Storage documents that objects created in an already-traversed
portion of the namespace can be missed during pagination. A validated terminal
traversal therefore establishes COMPLETION OF THAT TRAVERSAL, not an atomic
snapshot of the collection.

WHAT THIS MODULE DOES NOT ESTABLISH
===================================
    * that a newer release PREFIX means a usable product;
    * evidence sufficiency -- this produces observations and diagnostics, and
      a verifier qualifies them against the approved plan and retained bytes;
    * source authenticity -- a digest is integrity relative to retained bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

from .monitor_supervisor import Health, TargetResult
from .reason_catalog import Reason

ENDPOINT = ("https://storage.googleapis.com/storage/v1/b/"
            "gcp-public-data--gnomad/o")

#: nextPageToken is in the mask DELIBERATELY. Without it the response cannot
#: report its own incompleteness, and its absence would mean nothing.
BASE_QUERY = {"prefix": "release/", "delimiter": "/", "maxResults": "1000",
              "fields": "kind,prefixes,nextPageToken"}

EXPECTED_KIND = "storage#objects"
#: APPROVED 2026-09-24 (owner; first recorded in the rulings preserved 2026-09-22). APPROVAL IS NOT ADOPTION:
#: every gnomAD input of the PRODUCTION pipeline is still v4.1 (exploratory work used 4.1.1),
#: and a release change for constraint does not by itself establish one for every
#: frequency product. See
#: docs/measurements/DECISION_2026-09-24_gnomad-4.1.1-approval.md. Anything NEWER than this still alerts.
APPROVED_BASELINE = "4.1.1"

#: Declared budgets. Each is a policy statement, not a claim about the service.
MAX_PAGES = 20
MAX_BYTES_PER_PAGE = 4 * 1024 * 1024
MAX_TOKEN_CHARS = 8192
MAX_ELAPSED_SECONDS = 120          # a per-request timeout does not bound total

#: RULING 2026-09-16, requirement 3.1/3.2: "Preserve the exact response body
#: presented to the decoder. Recompute its digest during recovery and
#: verification." A digest with no retained body is a claim nothing can
#: check -- probe case "arbitrary_valid_length_digest" passed every existing
#: check because the verifier had only the digest to compare against itself.
#:
#: Bounded independently of MAX_BYTES_PER_PAGE, which governs PROCESSING. A
#: page within budget is still not retained once the RUN'S total would exceed
#: this -- "Capture under bounded resource use... including limits on
#: decompressed data." The measured live response is 305 bytes; twenty pages
#: at that size is nowhere near this ceiling, so ordinary operation retains
#: everything and only a pathological run degrades to digest-only retention.
MAX_TOTAL_RETAINED_BYTES = 8 * 1024 * 1024


class PageInvalid(Exception):
    """A page failed validation. Carries the reason code naming why."""

    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class PageCapture:
    """The bytes a verifier needs. Retained whether the page validated or not.

    A REJECTED page is captured too. Without it the report can say a page was
    malformed and offer nothing to check that against -- which is the same
    unfalsifiable shape as a status field nobody verifies.

    `request_url` is retained because the absence of a continuation token is
    only meaningful if the request ASKED for one. MEASURED 2026-09-14: an
    earlier request carried `fields=prefixes`, which omits nextPageToken, so
    its absence meant nothing and the evidence of truncation could not exist.
    """

    sequence: int
    request_url: str
    response_sha256: str
    response_bytes: int
    accepted: bool
    rejected_because: str = ""

    #: A TRANSPORT ATTEMPT identifier, distinct from `sequence` (the LOGICAL
    #: page ordinal). RULING 2026-09-16: "Retries mean there may be several
    #: transport attempts for one logical page. Give them different
    #: identifiers. Otherwise, introducing retries later will break the
    #: meaning of sequence." No retry logic exists yet -- this field exists so
    #: adding one is additive, not a silent redefinition of `sequence`.
    attempt_number: int = 1

    #: Base64 of the EXACT bytes handed to the decoder. Empty when retention
    #: was bounded away (see MAX_TOTAL_RETAINED_BYTES); `body_retained` says
    #: which case this is, so an empty string is never mistaken for "the body
    #: was empty".
    response_body_b64: str = ""
    body_retained: bool = False

    def as_document(self):
        return {"sequence": self.sequence, "request_url": self.request_url,
                "response_sha256": self.response_sha256,
                "response_bytes": self.response_bytes,
                "accepted": self.accepted,
                "rejected_because": self.rejected_because,
                "attempt_number": self.attempt_number,
                "response_body_b64": self.response_body_b64,
                "body_retained": self.body_retained}


@dataclass
class Traversal:
    """What has been PROVEN so far. Survives any interruption."""

    prefixes: list = field(default_factory=list)
    captures: list = field(default_factory=list)
    pages_read: int = 0
    bytes_read: int = 0
    retained_bytes_total: int = 0
    stopped_because: str = "not started"
    reason: object = None


def parse_version(prefix):
    """('release/4.1.1/') -> (4, 1, 1). None if outside the release grammar.

    A tuple, so ordering is numeric: as strings "4.10" < "4.9" is True, which
    would call 4.10 the older release. A leading "v" is stripped -- the bucket
    holds BOTH release/4.0/ and release/v4.0/, measured 2026-09-14.

    SIGNED COMPONENTS ARE REFUSED. MEASURED: the previous parser read
    'release/-1/' as (-1,), outside the intended grammar, sorting below every
    real release while silently participating in comparisons.
    """
    name = prefix.strip("/").split("/")[-1]
    if name.startswith("v"):
        name = name[1:]
    if not name:
        return None
    out = []
    for part in name.split("."):
        if not part.isdigit():        # rejects "", "-1", "+2", "4a", " 4"
            return None
        out.append(int(part))
    return tuple(out)


def _refuse_constant(name):
    raise PageInvalid(Reason.RESPONSE_JSON_VALUE,
                      "JSON constant {!r} is not permitted".format(name))


def _validate_page(raw):
    """Decode and validate ONE page. Raises PageInvalid with a reason code."""
    if len(raw) > MAX_BYTES_PER_PAGE:
        raise PageInvalid(Reason.TRAVERSAL_BUDGET_EXHAUSTED,
                          "page exceeded {} bytes".format(MAX_BYTES_PER_PAGE))

    def _no_duplicates(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                raise PageInvalid(Reason.RESPONSE_JSON_DUPLICATE,
                                  "duplicate JSON key: {}".format(key))
            seen.add(key)
        return dict(pairs)

    try:
        page = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
                          parse_constant=_refuse_constant)
    except PageInvalid:
        raise
    except Exception as exc:
        raise PageInvalid(Reason.RESPONSE_JSON_SYNTAX,
                          "{}: {}".format(type(exc).__name__, exc)) from exc

    if type(page) is not dict:
        raise PageInvalid(Reason.RESPONSE_UNEXPECTED_SHAPE,
                          "page is not an object")
    kind = page.get("kind")
    if kind != EXPECTED_KIND:
        # MEASURED: any kind was accepted, so a response from a different
        # collection read as a release listing.
        raise PageInvalid(Reason.RESPONSE_UNEXPECTED_SHAPE,
                          "kind is {!r}, expected {!r}".format(kind, EXPECTED_KIND))
    prefixes = page.get("prefixes", [])
    if type(prefixes) is not list:
        # MEASURED: a dict was accepted, and iterating its keys disguised it.
        raise PageInvalid(
            Reason.RESPONSE_UNEXPECTED_SHAPE,
            "prefixes is {}, expected a list".format(type(prefixes).__name__))
    for p in prefixes:
        if type(p) is not str:
            raise PageInvalid(Reason.RESPONSE_UNEXPECTED_SHAPE,
                              "a prefix is not a string")

    token = page.get("nextPageToken")
    if token is not None:
        if type(token) is not str:
            raise PageInvalid(Reason.TRAVERSAL_TOKEN_MALFORMED,
                              "token is {}".format(type(token).__name__))
        if not token:
            # MEASURED: an empty string was accepted as terminal, turning
            # malformed completion metadata into completeness.
            raise PageInvalid(Reason.TRAVERSAL_TOKEN_MALFORMED,
                              "token is an empty string")
        if len(token) > MAX_TOKEN_CHARS:
            raise PageInvalid(Reason.TRAVERSAL_TOKEN_MALFORMED,
                              "token exceeds {} characters".format(MAX_TOKEN_CHARS))
    return prefixes, token


def traverse(transport, *, max_pages=MAX_PAGES,
             max_elapsed=MAX_ELAPSED_SECONDS, clock=time.monotonic,
             max_retained_bytes=MAX_TOTAL_RETAINED_BYTES):
    """Walk the listing, accumulating VALIDATED pages.

    `transport(url) -> bytes` is injected. Fixtures and live execution use the
    same loop, so there is no second path that can diverge from this one.

    EVERY exit preserves the accumulator: an interruption on page two must not
    erase page one.
    """
    state = Traversal()
    seen_tokens = set()
    token = None
    started = clock()

    while True:
        if state.pages_read >= max_pages:
            state.stopped_because = "page budget exhausted"
            state.reason = Reason.TRAVERSAL_BUDGET_EXHAUSTED
            return state
        if clock() - started > max_elapsed:
            state.stopped_because = "elapsed budget exhausted"
            state.reason = Reason.TRAVERSAL_BUDGET_EXHAUSTED
            return state

        query = dict(BASE_QUERY)
        if token:
            query["pageToken"] = token
        url = ENDPOINT + "?" + urlencode(query)

        try:
            raw = transport(url)
        except TimeoutError as exc:
            state.stopped_because = "transport timeout: {}".format(exc)
            state.reason = Reason.TRANSPORT_TIMEOUT
            return state                       # the accumulator SURVIVES
        except Exception as exc:
            state.stopped_because = "{}: {}".format(type(exc).__name__, exc)
            state.reason = Reason.TRANSPORT_UNREACHABLE
            return state                       # the accumulator SURVIVES

        digest = hashlib.sha256(raw).hexdigest()

        # RETAIN THE BODY, bounded per-page AND across the whole run. A page
        # that is itself within MAX_BYTES_PER_PAGE can still be dropped from
        # RETENTION once the cumulative total would exceed the run budget --
        # processing and retention are governed by separate limits on
        # purpose, so a long clean run degrades to digest-only capture rather
        # than growing memory without bound.
        retain_body = (len(raw) <= MAX_BYTES_PER_PAGE
                      and state.retained_bytes_total + len(raw)
                      <= max_retained_bytes)
        body_b64 = base64.b64encode(raw).decode("ascii") if retain_body else ""
        if retain_body:
            state.retained_bytes_total += len(raw)

        try:
            prefixes, token = _validate_page(raw)
        except PageInvalid as exc:
            # CAPTURE THE REJECTED PAGE. A report that names a malformed page
            # and retains nothing to check is unfalsifiable.
            state.captures.append(PageCapture(
                len(state.captures) + 1, url, digest, len(raw), False,
                exc.detail, 1, body_b64, retain_body))
            state.stopped_because = exc.detail
            state.reason = exc.reason
            return state                       # the accumulator SURVIVES

        state.captures.append(PageCapture(
            len(state.captures) + 1, url, digest, len(raw), True, "",
            1, body_b64, retain_body))

        # Only a VALIDATED page contributes. A malformed page never adds
        # unverified findings.
        state.prefixes.extend(prefixes)
        state.pages_read += 1
        state.bytes_read += len(raw)

        if not token:
            state.stopped_because = "terminal page"
            state.reason = None
            return state
        if token in seen_tokens:
            state.stopped_because = "continuation token repeated"
            state.reason = Reason.TRAVERSAL_TOKEN_CYCLE
            return state
        seen_tokens.add(token)


def observe_releases(*, profile=None, transport=None):
    """Produce an observation. Qualification belongs to a verifier."""
    target = "gnomad-public-releases"

    baseline = parse_version("release/{}/".format(APPROVED_BASELINE))
    if baseline is None:
        return TargetResult(target, Health.FAILED,
                            findings=("approved baseline {!r} is outside the "
                                      "release grammar".format(APPROVED_BASELINE),),
                            reason=Reason.CONFIG_INVALID_BASELINE)

    state = traverse(transport or _live_transport)

    parsed = [v for v in (parse_version(p) for p in state.prefixes)
              if v is not None]
    newer = sorted({v for v in parsed if v > baseline})
    findings = tuple(
        "release {} is newer than the approved {}".format(
            ".".join(str(x) for x in v), APPROVED_BASELINE) for v in newer)

    if state.reason is not None:
        # INCOMPLETE, and the witness is KEPT. Both statements are true: a
        # newer release WAS observed, and the inventory was NOT established.
        return TargetResult(target, Health.INCOMPLETE, findings=findings,
                            reason=state.reason, captures=tuple(
                                c.as_document() for c in state.captures))
    if not state.prefixes:
        return TargetResult(target, Health.FAILED,
                            findings=("no release prefixes were returned",),
                            reason=Reason.RESPONSE_UNEXPECTED_SHAPE,
                            captures=tuple(c.as_document()
                                           for c in state.captures))
    if not parsed:
        # MEASURED: a collection holding only release/latest/ read as complete
        # with no finding -- no parseable versions became a clean comparison.
        return TargetResult(target, Health.FAILED,
                            findings=("no prefix parsed as a release version",),
                            reason=Reason.RESPONSE_UNEXPECTED_SHAPE,
                            captures=tuple(c.as_document()
                                           for c in state.captures))
    return TargetResult(target, Health.COMPLETE, findings=findings,
                        captures=tuple(c.as_document()
                                       for c in state.captures))


def _live_transport(url):
    import urllib.request
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read(MAX_BYTES_PER_PAGE + 1)
