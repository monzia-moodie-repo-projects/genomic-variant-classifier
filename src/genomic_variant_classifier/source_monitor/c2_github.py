"""C2 GitHub adapter (owner ruling 2026-09-29; README boundaries 5-10). The ONLY module that talks to GitHub for delivery.

Every GitHub fact used here was taken from GitHub's documentation (recorded 2026-09-30):
  * issue comments: GET /repos/{o}/{r}/issues/{n}/comments, ascending id, per_page <= 100; a comment has id, body,
    user.id and issue_url -- NO numeric issue id, so the channel DERIVES it by an EXACT issue_url match;
  * the issues endpoints treat every pull request as an issue (a "pull_request" key) -- refused as a destination;
  * jobs: GET /repos/{o}/{r}/actions/runs/{id}/attempts/{n}/jobs; each job has name, status, conclusion, steps[]
    (name, status, conclusion);
  * artifacts (upload-artifact v4) are immutable and names are UNIQUE within a run -- so the per-attempt outcome record
    is named OUTCOME_ARTIFACT_PREFIX + attempt.

TRANSPORT (measured 2026-09-30 on Python 3.12.3): the default opener did not follow a 307 on POST, but Python's default
handler turns a POST redirected by 301/302/303 into a GET and follows it. This adapter refuses EVERY redirect for API
calls, and attaches the token unredirected (never forwarded to another host). One call = one HTTP request; there is
no retry loop anywhere in this module.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass

from genomic_variant_classifier.source_monitor import c2_protocol as c2

API_ROOT = "https://api.github.com"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_PAGES = 200
OUTCOME_ARTIFACT_PREFIX = "c2-delivery-outcome-attempt-"
OUTCOME_SCHEMA = "gvc.c2-delivery-outcome"
PUBLISH_JOB = "publish"
DELIVERY_STEP = "Deliver the verification receipt"


class Transport:
    """request(method, url, body_bytes_or_None, *, allow_redirect=False) -> (status, headers, body). Live HTTPS only."""

    def __init__(self, token: str, limit: int = MAX_JSON_BYTES):
        self.token, self.limit = token, limit

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        self._no_redirect = urllib.request.build_opener(_NoRedirect)
        self._redirect = urllib.request.build_opener()

    def __call__(self, method, url, body=None, *, allow_redirect=False):
        if not url.startswith(API_ROOT + "/"):
            raise c2.Refusal("transport.host")
        req = urllib.request.Request(url, data=body, method=method, headers={
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
            **({"Content-Type": "application/json"} if body is not None else {})})
        req.add_unredirected_header("Authorization", "Bearer " + self.token)
        opener = self._redirect if allow_redirect else self._no_redirect
        try:
            with opener.open(req, timeout=60) as resp:
                if not resp.geturl().startswith("https://"):
                    raise c2.Refusal("transport.scheme")
                data = resp.read(self.limit + 1)
                status, headers = resp.status, dict(resp.headers)
        except urllib.error.HTTPError as exc:
            with exc:                                       # an HTTPError holds the response: always close it
                return exc.code, dict(exc.headers or {}), exc.read(self.limit + 1)
        if len(data) > self.limit:
            raise c2.Refusal("transport.size")
        return status, headers, data


def _json(status, body, expected=200):
    c2.require(status == expected, "http.status_{}".format(status))
    return json.loads(body.decode("utf-8"))


_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def _prefixes(repository, repository_id, suffix):
    """The two DOCUMENTED forms of one repository resource: GitHub's pagination guide shows a request to
    /repos/{owner}/{repo}/issues answered with Link URLs under /repositories/{numeric id}/issues (verified 2026-09-30)."""
    return ("{}/repos/{}{}".format(API_ROOT, repository, suffix), "{}/repositories/{}{}".format(API_ROOT, repository_id, suffix))


def _next_link(headers, prefixes):
    """The Link header's rel="next" URL, or None. Refused unless it is one of the documented forms of the SAME resource."""
    link = next((v for k, v in headers.items() if k.lower() == "link"), None)
    if not link:
        return None
    m = _NEXT.search(link)
    if m is None:
        return None
    c2.require(any(m.group(1).startswith(p + "?") for p in prefixes), "pagination.foreign_link")
    return m.group(1)


def _all_pages(request, first_url, prefixes, key=None):
    """Every item from every page; refuses a repeated cursor and more than MAX_PAGES. When the listing is an object with
    `key`, its total_count must EQUAL the items collected -- GitHub has been reported to return EMPTY pages beyond a
    ceiling while total_count shows more, and a silently truncated history would look like "no prior dispatch"."""
    url, seen, items, total = first_url, set(), [], None
    for _ in range(MAX_PAGES):
        c2.require(url not in seen, "pagination.cycle")
        seen.add(url)
        status, headers, body = request("GET", url)
        doc = _json(status, body)
        page = doc if key is None else doc.get(key) if isinstance(doc, dict) else None
        c2.require(type(page) is list, "pagination.page")
        if key is not None:
            c2.require(type(doc.get("total_count")) is int and total in (None, doc["total_count"]), "pagination.total_count")
            total = doc["total_count"]
        items.extend(page)
        url = _next_link(headers, prefixes)
        if url is None:
            c2.require(total is None or total == len(items), "pagination.incomplete")
            return items
    raise c2.Refusal("pagination.limit")


def subject_from_attempt(attempt) -> dict:
    """The receipt subject from GitHub's OWN attempt record -- never from a report. The ONE definition: the checker
    (scripts/verify_monitor_run.py) and the publisher (scripts/publish_monitor_receipt.py) must derive it identically."""
    repo = attempt.get("repository") if isinstance(attempt, dict) else None
    if not isinstance(repo, dict):
        raise ValueError("GitHub's attempt record has no repository object")
    return {"repository": repo.get("full_name"), "repository_id": repo.get("id"), "workflow_id": attempt.get("workflow_id"),
            "workflow_path": attempt.get("path"), "run_id": attempt.get("id"), "run_number": attempt.get("run_number"),
            "attempt": attempt.get("run_attempt"), "commit": attempt.get("head_sha")}


@dataclass(frozen=True)
class Pinned:
    """The committed destination (owner ruling: pin the standing issue; the label is a consistency check)."""
    repository: str
    repository_id: int
    issue_id: int
    number: int
    author_id: int                   # the numeric identity whose acknowledgements are trusted (revalidate at activation)
    label: str                       # the consistency label, from the DEPLOYMENT configuration (2026-10-01)


def select_destination(request, pinned: Pinned) -> c2.Destination:
    """The open labelled issue, validated against the pinned identity. Refuses missing, multiple, closed, transferred,
    pull-request or unexpected destinations (README boundary 5)."""
    base = "{}/repos/{}".format(API_ROOT, pinned.repository)
    repo = _json(*request("GET", base)[::2])
    c2.require(type(repo) is dict and repo.get("id") == pinned.repository_id and repo.get("full_name") == pinned.repository,
               "destination.repository")
    # Percent-encoded: the label comes from configuration; unencoded, a space, "&" or "#" would corrupt the query.
    issues = _all_pages(request, base + "/issues?labels={}&state=open&per_page=100".format(urllib.parse.quote(pinned.label, safe="")),
                        _prefixes(pinned.repository, pinned.repository_id, "/issues"))
    listed = [{"repository_id": pinned.repository_id, "id": i.get("id"), "number": i.get("number")}
              for i in issues if isinstance(i, dict) and "pull_request" not in i]
    configured = c2.Destination(pinned.repository_id, pinned.issue_id, pinned.number)
    c2.select_destination(listed, configured)
    issue = _json(*request("GET", base + "/issues/{}".format(pinned.number))[::2])
    c2.require(type(issue) is dict and "pull_request" not in issue, "destination.pull_request")
    c2.require(issue.get("id") == pinned.issue_id and issue.get("state") == "open"
               and issue.get("repository_url") == base
               and pinned.label in [x.get("name") for x in issue.get("labels") or [] if isinstance(x, dict)], "destination.changed")
    return configured


class CommentChannel:
    """c2_protocol.Channel over the DIRECT issue-comments endpoint (never search results)."""

    def __init__(self, request, pinned: Pinned):
        self.request, self.pinned = request, pinned
        self.base = "{}/repos/{}/issues/{}".format(API_ROOT, pinned.repository, pinned.number)
        self.comments_url = self.base + "/comments"
        self.prefixes = _prefixes(pinned.repository, pinned.repository_id, "/issues/{}/comments".format(pinned.number))

    def _comment(self, raw) -> c2.Comment:
        c2.require(type(raw) is dict and type(raw.get("body")) is str, "comments.shape")
        user = raw.get("user")
        author = user.get("id") if isinstance(user, dict) else None
        issue_id = self.pinned.issue_id if raw.get("issue_url") == self.base else -1      # EXACT url or foreign
        return c2.Comment(raw.get("id"), issue_id, author, raw["body"])

    def page(self, cursor):
        url = cursor if cursor is not None else self.comments_url + "?per_page=100"
        if cursor is not None:
            c2.require(any(cursor.startswith(p + "?") for p in self.prefixes), "comments.cursor")
        status, headers, body = self.request("GET", url)
        rows = _json(status, body)
        c2.require(type(rows) is list, "comments.page")
        return c2.Page(tuple(self._comment(r) for r in rows), _next_link(headers, self.prefixes))

    def create_once(self, body: str, *, before_send) -> c2.Comment:
        """Prepare the request bytes, call before_send() -- the final freshness gate (owner ruling 2026-10-01) -- and only
        then EXACTLY one HTTP POST. If before_send raises, nothing is sent. Any non-201 answer or transport error AFTER the
        gate is raised (the caller treats it as ambiguous)."""
        data = json.dumps({"body": body}).encode("utf-8")
        before_send()
        status, _, raw = self.request("POST", self.comments_url, data)
        return self._comment(_json(status, raw, expected=201))


def outcome_record(*, delivery_id: str, post_issued: bool, result) -> dict:
    """The per-attempt outcome record the delivery step uploads as OUTCOME_ARTIFACT_PREFIX + attempt."""
    return {"schema": OUTCOME_SCHEMA, "schema_version": 1, "delivery_id": delivery_id, "post_issued": post_issued,
            "action": result.action, "reason": result.reason}


def _read_outcome(raw_zip: bytes, delivery_id: str):
    """-> True (a POST was issued for THIS delivery), False (none was), or None (unreadable / not for this delivery)."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
            names = zf.namelist()
            if names != ["outcome.json"]:
                return None
            doc = c2.strict_load(zf.read("outcome.json"))
    except (zipfile.BadZipFile, c2.Refusal, KeyError, ValueError):
        return None
    if not (type(doc) is dict and set(doc) == {"schema", "schema_version", "delivery_id", "post_issued", "action", "reason"}
            and doc["schema"] == OUTCOME_SCHEMA and type(doc["schema_version"]) is int and doc["schema_version"] == 1
            and type(doc["post_issued"]) is bool):
        return None
    if doc["delivery_id"] != delivery_id:
        return False        # deliver() issues AT MOST ONE POST per invocation: a POST for ANOTHER delivery is not ours
    return doc["post_issued"]


_NOT_STARTED = frozenset({"queued", "pending", "waiting", "requested"})


def dispatch_history(request, *, repository: str, repository_id: int, workflow_id: int, run_name: str, current_run_id: int,
                     current_attempt: int, delivery_id: str) -> c2.DispatchHistory:
    """Prior-dispatch knowledge from AUTHENTICATED GitHub execution history (README boundaries 9-10). Never
    NO_PRIOR_DISPATCH by default: every prior execution indexed by run_name, and every earlier attempt of the CURRENT run
    (examined EXPLICITLY -- never dependent on the run listing, which can lag), must be proven not to have issued a POST
    for delivery_id. The workflow is authenticated by its NUMERIC id (taken by the caller from GitHub's record of the
    current run), never by a path string whose format may carry a ref suffix."""
    base = "{}/repos/{}".format(API_ROOT, repository)
    evidence, unknown, prior = [], [], []

    def examine(run_id, n):
        jobs = _all_pages(request, base + "/actions/runs/{}/attempts/{}/jobs?per_page=100".format(run_id, n),
                          _prefixes(repository, repository_id, "/actions/runs/{}/attempts/{}/jobs".format(run_id, n)), key="jobs")
        steps = [s for j in jobs if isinstance(j, dict) and j.get("name") == PUBLISH_JOB
                 for s in j.get("steps") or [] if isinstance(s, dict) and s.get("name") == DELIVERY_STEP]
        tag = "run {} attempt {}".format(run_id, n)
        if not [s for s in steps if s.get("status") not in _NOT_STARTED and s.get("conclusion") != "skipped"]:
            evidence.append(tag + ": delivery step never started")
            return
        name = OUTCOME_ARTIFACT_PREFIX + str(n)
        arts = _all_pages(request, base + "/actions/runs/{}/artifacts?per_page=100".format(run_id),
                          _prefixes(repository, repository_id, "/actions/runs/{}/artifacts".format(run_id)), key="artifacts")
        named = [a for a in arts if isinstance(a, dict) and a.get("name") == name]
        if len(named) != 1 or named[0].get("expired") is not False:
            unknown.append(tag + ": delivery step started, outcome record missing or expired")
            return
        status, _, raw = request("GET", base + "/actions/artifacts/{}/zip".format(named[0].get("id")), allow_redirect=True)
        posted = _read_outcome(raw, delivery_id) if status == 200 else None
        if posted is True:
            prior.append(tag + ": a POST was issued for this delivery")
        elif posted is None:
            unknown.append(tag + ": outcome record unreadable")
        else:
            evidence.append(tag + ": delivery step started, no POST for this delivery")

    try:
        c2.require(c2.positive(workflow_id) and c2.positive(repository_id), "history.identity")
        c2.require(c2.positive(current_run_id) and c2.positive(current_attempt), "history.current")
        runs = _all_pages(request, base + "/actions/workflows/{}/runs?per_page=100".format(workflow_id),
                          _prefixes(repository, repository_id, "/actions/workflows/{}/runs".format(workflow_id)),
                          key="workflow_runs")
        for run in runs:
            c2.require(isinstance(run, dict) and run.get("workflow_id") == workflow_id, "history.workflow")
            if run.get("display_title") != run_name or run.get("id") == current_run_id:
                continue
            c2.require(c2.positive(run.get("id")) and c2.positive(run.get("run_attempt")), "history.run")
            for n in range(1, run["run_attempt"] + 1):
                examine(run["id"], n)
        for n in range(1, current_attempt):              # the CURRENT run's earlier attempts, always
            examine(current_run_id, n)
    except Exception as exc:
        return c2.DispatchHistory(c2.History.UNKNOWN, "history unreadable: {}: {}".format(
            type(exc).__name__, getattr(exc, "code", exc)))
    if prior:
        return c2.DispatchHistory(c2.History.PRIOR_DISPATCH, "; ".join(prior))
    if unknown:
        return c2.DispatchHistory(c2.History.UNKNOWN, "; ".join(unknown))
    return c2.DispatchHistory(c2.History.NO_PRIOR_DISPATCH, "; ".join(evidence) or "no prior execution for " + run_name)
