"""C2 delivery protocol (owner ruling 2026-09-29, integrated 2026-09-30). Standard library only; no credentials or live HTTP.

Integrated from the owner's reviewed reference (GVC_C2_delivery_review_2026-09-29, c2_protocol.py sha256 dd48c907...), with the
owner's ADOPTED timing policy (2026-09-30) replacing the reference's illustrative five-minute allowance:
    MAX_RECEIPT_AGE_SECONDS = 900   a NEW post needs a receipt at most 900 s old ("receipt.needs_reverification" beyond)
    MAX_FUTURE_SKEW_SECONDS = 60    a receipt at most 60 s ahead of the writer's clock ("receipt.future_timestamp" beyond)
Accepted: -60 <= age_seconds <= +900, inclusive; the allowance never extends the stale limit. These are explicit initial policy
choices, not empirical limits; each Result records the observed age so any later adjustment can be justified by measurement.
An existing matching acknowledgement is recognised after expiry (the age check follows the acknowledgement search); a reversed
evaluation interval is always refused ("time.reversed"); re-verification must RERUN the checker (the receipt binds the current
evaluation run and attempt), so editing a timestamp never restores eligibility.

The caller authenticates GitHub context and acquires the writer concurrency
group BEFORE constructing Bindings/DispatchHistory or calling deliver().
These Python objects are contracts, not authentication tokens.
See README.md for the required adapter and the limitations of comment storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import re
from typing import Protocol

FLAGS = (
    "execution_authenticated", "configuration_bound", "observation_complete",
    "claims_reconciled", "review_required",
    "current_monitoring_obligation_satisfied",
)
REASONS = frozenset({
    "execution.invalid", "configuration.invalid", "observation.incomplete",
    "observation.invalid", "claims.disagree", "freshness.expired",
    "freshness.future", "policy.changed", "checker.unavailable",
    "artifact.missing", "artifact.invalid", "policy.unsupported",
})
LIMIT = 65536
#: The writer's VERSIONED delivery policy (owner decision 2026-09-30).
DELIVERY_POLICY_VERSION = 1
MAX_RECEIPT_AGE_SECONDS = 900
MAX_FUTURE_SKEW_SECONDS = 60
HEX = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


class Refusal(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise Refusal(code)


def exact_keys(value, expected, code):
    require(type(value) is dict and set(value) == set(expected), code)


def positive(value):
    return type(value) is int and 0 < value <= 2**53 - 1


def _tree(value, depth=0):
    require(depth <= 24, "json.too_deep")
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise Refusal("json.unicode") from exc
        return
    if type(value) is int:
        require(abs(value) <= 2**53 - 1, "json.integer_range")
        return
    if type(value) is list:
        for item in value:
            _tree(item, depth + 1)
        return
    if type(value) is dict:
        require(all(type(k) is str for k in value), "json.key_type")
        for key, item in value.items():
            _tree(key, depth + 1)
            _tree(item, depth + 1)
        return
    raise Refusal("json.type")


def canonical(value):
    """Named project codec: ASCII, sorted keys, compact; not RFC 8785."""
    _tree(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(domain, value):
    return hashlib.sha256(domain.encode("ascii") + b"\0" + canonical(value)).hexdigest()


def strict_load(raw):
    require(type(raw) is bytes and 0 < len(raw) <= LIMIT, "receipt.size")
    require(not raw.startswith(b"\xef\xbb\xbf"), "json.bom")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "json.duplicate_key")
            result[key] = value
        return result

    def no_number(token):
        raise Refusal("json.non_integer_number")

    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                         parse_float=no_number, parse_constant=no_number)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, Refusal):
            raise
        raise Refusal("json.invalid") from exc
    _tree(doc)
    return doc


def timestamp(value):
    require(type(value) is str, "time.type")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise Refusal("time.format") from exc
    require(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value, "time.noncanonical")
    return parsed


def checked_hash(value):
    return type(value) is str and HEX.fullmatch(value) is not None


def validate_payload(p):
    exact_keys(p, ("schema", "schema_version", "issuer_role", "subject", "evidence", "checker",
                   "evaluation", "decision", "diagnostics"), "receipt.fields")
    require(p["schema"] == "gvc.monitor-receipt", "receipt.schema")
    require(type(p["schema_version"]) is int and p["schema_version"] == 1,
            "receipt.version")
    require(p["issuer_role"] in ("checker", "coordinator"), "receipt.issuer")
    s = p["subject"]
    exact_keys(s, ("repository", "repository_id", "workflow_id", "workflow_path",
                   "run_id", "run_number", "attempt", "commit"), "subject.fields")
    require(type(s["repository"]) is str and
            re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", s["repository"]),
            "subject.repository")
    require(all(positive(s[k]) for k in
                ("repository_id", "workflow_id", "run_id", "run_number", "attempt")),
            "subject.integer")
    require(s["workflow_path"] == ".github/workflows/source_monitor.yml",
            "subject.workflow")
    require(type(s["commit"]) is str and COMMIT.fullmatch(s["commit"]), "subject.commit")
    e = p["evidence"]
    exact_keys(e, ("state", "artifact_id", "archive_sha256", "report_sha256"),
               "evidence.fields")
    require(e["state"] in ("complete", "missing", "invalid", "unavailable"), "evidence.state")
    require(e["artifact_id"] is None or positive(e["artifact_id"]), "evidence.id")
    require(all(e[k] is None or checked_hash(e[k])
                for k in ("archive_sha256", "report_sha256")), "evidence.digest")
    if e["state"] == "complete":
        require(all(e[k] is not None for k in
                    ("artifact_id", "archive_sha256", "report_sha256")), "evidence.incomplete")
    c = p["checker"]
    exact_keys(c, ("commit", "code_manifest_sha256", "policy_sha256"), "checker.fields")
    require(type(c["commit"]) is str and COMMIT.fullmatch(c["commit"]), "checker.commit")
    require(checked_hash(c["code_manifest_sha256"]) and checked_hash(c["policy_sha256"]),
            "checker.digest")
    ev = p["evaluation"]
    exact_keys(ev, ("run_id", "attempt", "started_at", "finished_at"), "evaluation.fields")
    require(positive(ev["run_id"]) and positive(ev["attempt"]), "evaluation.integer")
    require(timestamp(ev["started_at"]) <= timestamp(ev["finished_at"]), "time.reversed")
    d = p["decision"]
    exact_keys(d, ("status", "verified", "flags", "reviews", "reasons"), "decision.fields")
    require(d["status"] in ("completed", "unavailable"), "decision.status")
    require(type(d["reviews"]) is list and type(d["reasons"]) is list, "decision.lists")
    for item in d["reviews"]:
        exact_keys(item, ("target", "kind", "raw_prefix"), "review.fields")
        require(all(type(item[k]) is str and 0 < len(item[k]) <= 512 for k in item),
                "review.values")
        require(item["kind"] in ("newer", "unsupported"), "review.kind")
    for reason in d["reasons"]:
        exact_keys(reason, ("code", "target"), "reason.fields")
        require(type(reason["code"]) is str and reason["code"] in REASONS, "reason.code")
        require(type(reason["target"]) is str and len(reason["target"]) <= 512,
                "reason.target")
    if d["status"] == "unavailable":
        require(d["verified"] is None and d["flags"] is None and not d["reviews"],
                "decision.unavailable_shape")
        require(any(r["code"] == "checker.unavailable" for r in d["reasons"]),
                "decision.unavailable_reason")
    else:
        require(p["issuer_role"] == "checker", "decision.coordinator_cannot_verify")
        f = d["flags"]
        exact_keys(f, FLAGS, "flags.fields")
        require(all(type(v) is bool for v in f.values()) and
                type(d["verified"]) is bool, "flags.type")
        require(d["verified"] == all(f[k] for k in FLAGS[:4]), "flags.verified")
        require(not f["execution_authenticated"] or e["state"] == "complete",
                "flags.evidence")
        require(not d["reviews"] or
                (f["execution_authenticated"] and f["configuration_bound"]),
                "review.unbound")
        require(f["review_required"] == bool(d["reviews"]), "flags.review")
        require(not f[FLAGS[-1]] or d["verified"], "flags.current")
    require(type(p["diagnostics"]) is list and
            all(type(x) is str for x in p["diagnostics"]), "diagnostics.shape")
    require(len(canonical(p)) <= LIMIT - 256, "receipt.size")
    return p


def seal(payload):
    validate_payload(payload)
    return canonical({"payload": payload,
                      "receipt_sha256": digest("gvc.receipt/v1", payload)})


@dataclass(frozen=True)
class Bindings:
    """Independently established by the trusted GitHub adapter, never the receipt."""
    subject: dict
    checker: dict
    evaluation_run_id: int
    evaluation_attempt: int
    issuer_role: str = "checker"


def open_receipt(raw, bindings: Bindings, now: datetime):
    doc = strict_load(raw)
    exact_keys(doc, ("payload", "receipt_sha256"), "envelope.fields")
    p = validate_payload(doc["payload"])
    require(checked_hash(doc["receipt_sha256"]) and
            doc["receipt_sha256"] == digest("gvc.receipt/v1", p), "receipt.checksum")
    # Canonical bytes give type-sensitive comparisons (True must not equal 1).
    require(canonical(p["subject"]) == canonical(bindings.subject), "binding.subject")
    require(canonical(p["checker"]) == canonical(bindings.checker), "binding.checker")
    require(p["issuer_role"] == bindings.issuer_role, "binding.issuer")
    require(positive(bindings.evaluation_run_id) and positive(bindings.evaluation_attempt),
            "binding.evaluation_type")
    ev = p["evaluation"]
    require((ev["run_id"], ev["attempt"]) ==
            (bindings.evaluation_run_id, bindings.evaluation_attempt), "binding.evaluation")
    require(isinstance(now, datetime) and now.utcoffset() is not None, "time.clock")
    require(timestamp(ev["finished_at"]) <= now + timedelta(seconds=MAX_FUTURE_SKEW_SECONDS), "receipt.future_timestamp")
    return p


def decision_material(p):
    validate_payload(p)
    d = p["decision"]
    # Sorting preserves multiplicity; it does NOT turn an evidence multiset into a set.
    normalized = dict(d, reviews=sorted(d["reviews"], key=canonical),
                      reasons=sorted(d["reasons"], key=canonical))
    return {"issuer_role": p["issuer_role"],
            "subject": p["subject"], "evidence": p["evidence"],
            "checker": p["checker"], "decision": normalized}


def decision_id(p):
    return digest("gvc.decision/v1", decision_material(p))


@dataclass(frozen=True)
class Destination:
    repository_id: int
    issue_id: int
    number: int


def select_destination(open_labelled_issues, configured: Destination):
    """Input must include ALL pages, excluding pull requests, from the trusted API."""
    require(all(positive(v) for v in
                (configured.repository_id, configured.issue_id, configured.number)),
            "destination.config")
    require(type(open_labelled_issues) is list, "destination.list")
    require(len(open_labelled_issues) != 0, "destination.missing")
    require(len(open_labelled_issues) == 1, "destination.ambiguous")
    issue = open_labelled_issues[0]
    require(type(issue) is dict and
            canonical(issue) == canonical({"repository_id": configured.repository_id,
                                           "id": configured.issue_id,
                                           "number": configured.number}),
            "destination.changed")
    return configured


def delivery_id(p, destination):
    require(p["subject"]["repository_id"] == destination.repository_id, "destination.repository")
    return digest("gvc.delivery/v1", {"decision_id": decision_id(p),
        "repository_id": destination.repository_id, "issue_id": destination.issue_id})


def render_comment(p, destination):
    key = delivery_id(p, destination)
    s = p["subject"]
    # Indented JSON treats all finding strings as data, including Markdown, @mentions
    # and fake hidden markers. Escaping prevents raw "<!--" even inside data strings.
    material = json.dumps(decision_material(p), ensure_ascii=True, sort_keys=True, indent=2)
    material = material.replace("<", "\\u003c").replace(">", "\\u003e").replace("@", "\\u0040")
    body = (
        "Source-monitor verifier decision\n\n"
        f"Run: https://github.com/{s['repository']}/actions/runs/{s['run_id']}"
        f"/attempts/{s['attempt']}\n\n"
        "These are the verifier's results about this run. Freshness is assessed at "
        "verification; this comment is not a live health indicator. Review items "
        "remain visible when observation or reconciliation failed. No automatic closure.\n\n"
        + "**Review items**\n\n"
        + ("\n".join("    " + json.dumps(item, ensure_ascii=True, sort_keys=True)
                    .replace("<", "\\u003c").replace(">", "\\u003e").replace("@", "\\u0040")
                    for item in p["decision"]["reviews"]) or "No bound review items in this decision.")
        + "\n\n<details><summary>Full decision and provenance</summary>\n\n"
        + "\n".join("    " + line for line in material.splitlines())
        + "\n\n</details>"
        + f"\n\n<!-- gvc:c2:v1:{key} -->\n"
    )
    require(len(body.encode("utf-8")) <= 48000, "comment.too_large")
    return body


@dataclass(frozen=True)
class Comment:
    id: int
    issue_id: int
    author_id: int
    body: str


@dataclass(frozen=True)
class Page:
    comments: tuple[Comment, ...]
    next_cursor: str | None


class PostNotAttempted(Refusal):
    """A local precondition failed BEFORE the POST transport was entered (owner ruling 2026-10-01): a definite
    no-attempt result, never an uncertain delivery."""

    def __init__(self, code, age_seconds=None):
        super().__init__(code)
        self.age_seconds = age_seconds


class Channel(Protocol):
    def page(self, cursor: str | None) -> Page: ...
    def create_once(self, body: str, *, before_send) -> Comment:
        """Prepare the request, call before_send() IMMEDIATELY before invoking the transport, then exactly ONE HTTP POST;
        no client/library/proxy retry loop. If before_send raises, nothing is sent."""
        ...


def find_ack(channel, p, destination, author_id):
    """Complete direct listing; never search-index results or first-page absence."""
    require(positive(author_id), "comment.author_config")
    key = delivery_id(p, destination)
    expected = render_comment(p, destination)
    marker = f"<!-- gvc:c2:v1:{key} -->"
    cursor, cursors, ids, matches = None, set(), set(), []
    for _ in range(200):
        require(cursor not in cursors, "comments.cursor_cycle")
        cursors.add(cursor)
        page = channel.page(cursor)
        require(type(page) is Page and type(page.comments) is tuple, "comments.page")
        for c in page.comments:
            require(type(c) is Comment and positive(c.id) and type(c.body) is str,
                    "comments.shape")
            require(c.id not in ids, "comments.unstable_listing")
            ids.add(c.id)
            require(type(c.issue_id) is int and c.issue_id == destination.issue_id,
                    "comments.destination")
            if type(c.author_id) is int and c.author_id == author_id and marker in c.body:
                require(c.body == expected, "comments.modified_ack")
                matches.append(c)
        if page.next_cursor is None:
            require(len(matches) <= 1, "comments.duplicate_ack")
            return matches[0] if matches else None
        require(type(page.next_cursor) is str and page.next_cursor, "comments.cursor")
        cursor = page.next_cursor
    raise Refusal("comments.scan_limit")


class History(Enum):
    NO_PRIOR_DISPATCH = "no_prior_dispatch"
    PRIOR_DISPATCH = "prior_dispatch"
    UNKNOWN = "history_unknown"


@dataclass(frozen=True)
class DispatchHistory:
    state: History
    evidence_ref: str


#: The ONE delivery-action vocabulary (C2 repairs 4): every Result is constructed from it, and the journal reader validates
#: recorded outcomes against the same set -- writer and reader cannot drift.
ACTIONS = frozenset({"acknowledged", "preview", "no_op", "archive", "blocked", "unknown"})


@dataclass(frozen=True)
class Result:
    action: str
    reason: str
    comment_id: int | None = None
    age_seconds: float | None = None      # the OBSERVED receipt age when the delivery policy was applied

    def __post_init__(self):
        require(type(self.action) is str and self.action in ACTIONS, "result.action")
        require(type(self.reason) is str and 0 < len(self.reason) <= 256, "result.reason")


def event_kind(p):
    d = p["decision"]
    if d["status"] == "unavailable":
        return "verification_unavailable"
    if not d["verified"]:
        return "verification_failed_with_review" if d["reviews"] else "verification_failed"
    if d["reviews"]:
        return "review_required"
    return "current_clean" if d["flags"][FLAGS[-1]] else "historical_clean"


def deliver(channel, p, destination, *, author_id, history: DispatchHistory,
            clock, attempt=None, automatic=True, simulation=False, cancelled=False):
    """Caller holds the shared writer queue for the entire invocation.

    FRESHNESS AT THE POST BOUNDARY (owner ruling 2026-10-01): `clock` is read ONCE, inside the channel's before_send --
    after the acknowledgement search, the history and the request preparation, immediately before the transport. The
    guarantee is exactly that: freshness is enforced immediately before the client initiates the POST. `attempt` (a
    caller-owned dict) records post_issued ONLY once that gate has passed, plus the observed age.

    If any prior writer could have sent this decision, history MUST be PRIOR or
    UNKNOWN, reconstructed across process/run restarts. A default 'new' value is
    unsafe. No absence result turns PRIOR/UNKNOWN back into NO_PRIOR_DISPATCH.
    """
    validate_payload(p)
    if simulation or not automatic:
        return Result("preview", "simulation" if simulation else "manual_verification")
    if cancelled:
        return Result("no_op", "cancelled")
    try:
        found = find_ack(channel, p, destination, author_id)
    except Exception as exc:
        return Result("blocked", exc.code if isinstance(exc, Refusal) else "comments.read_failed")
    if found:
        return Result("acknowledged", "matching_comment", found.id)
    require(type(history) is DispatchHistory and type(history.state) is History and
            type(history.evidence_ref) is str and bool(history.evidence_ref), "history.evidence")
    if history.state is not History.NO_PRIOR_DISPATCH:
        return Result("unknown", "prior_dispatch_unresolved")
    # Historical failures and positive witnesses must still surface.
    if event_kind(p) == "historical_clean":
        return Result("archive", "historical_clean_no_new_review")
    state = attempt if attempt is not None else {}
    state["post_issued"] = False
    finished = timestamp(p["evaluation"]["finished_at"])
    body = render_comment(p, destination)

    def before_send():
        now = clock()
        if not (isinstance(now, datetime) and now.utcoffset() is not None):
            raise PostNotAttempted("time.clock")
        age_seconds = (now - finished).total_seconds()
        if age_seconds < -MAX_FUTURE_SKEW_SECONDS:
            raise PostNotAttempted("receipt.future_timestamp", age_seconds)
        if age_seconds > MAX_RECEIPT_AGE_SECONDS:
            raise PostNotAttempted("receipt.needs_reverification", age_seconds)
        state["post_issued"], state["age_seconds"] = True, age_seconds   # conservative: the transport is being entered

    try:
        c = channel.create_once(body, before_send=before_send)
        if not state["post_issued"]:
            state["post_issued"] = True        # a comment came back WITHOUT the gate: it posted UNGATED -- conservative
            raise Refusal("post.gate_bypassed")
        require(type(c) is Comment and positive(c.id) and
                type(c.issue_id) is int and c.issue_id == destination.issue_id and
                type(c.author_id) is int and c.author_id == author_id and c.body == body,
                "post.response_mismatch")
        return Result("acknowledged", "created", c.id, state["age_seconds"])
    except PostNotAttempted as exc:
        return Result("blocked", exc.code, age_seconds=exc.age_seconds)     # DEFINITE no-attempt, never "unknown"
    except Exception as exc:
        if isinstance(exc, Refusal) and exc.code == "post.gate_bypassed":
            raise
        if not state["post_issued"]:
            return Result("blocked", "post.not_attempted")                 # failed BEFORE the gate: nothing was sent
    # The gate passed and the transport was entered: a lost response, a server error or an invalid success response is
    # AMBIGUOUS. One bounded reconciliation attempt is safe; a second POST is not.
    try:
        found = find_ack(channel, p, destination, author_id)
    except Exception:
        found = None
    if found:
        return Result("acknowledged", "reconciled_after_post_error", found.id, state["age_seconds"])
    return Result("unknown", "post_outcome_unknown", age_seconds=state["age_seconds"])


def acknowledge_pending(pending_key, received_key):
    """The reference's previously unreachable 'nothing pending' reason comes first."""
    require(pending_key is not None, "nothing_pending")
    require(pending_key == received_key, "wrong_pending_key")
    return received_key
