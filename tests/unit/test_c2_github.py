"""The C2 GitHub adapter against a fake GitHub that emits the DOCUMENTED response forms (verified 2026-09-30):
page-two Link URLs under /repositories/{numeric id}/...; comments carrying only an issue_url; pull requests inside the
issues listing; total_count on every object listing. Every refusal pins its exact reason; every delivery counts POSTs.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from datetime import datetime, timezone

import pytest

from genomic_variant_classifier.source_monitor import c2_github as gh
from genomic_variant_classifier.source_monitor import c2_protocol as c2
from tests.unit.test_c2_protocol import fixture

REPO, REPO_ID, ISSUE_ID, NUMBER, BOT = "o/r", 10, 200, 27, 41898282
PIN = gh.Pinned(REPO, REPO_ID, ISSUE_ID, NUMBER, BOT)
API = gh.API_ROOT + "/repos/" + REPO
ISSUE_URL = API + "/issues/27"
WORKFLOW_ID, RUN_NAME, CURRENT_RUN = 555, "verify 123/1", 900


class FakeGitHub:
    """request(method, url, body, *, allow_redirect) over a route table; records every call; POSTs mutate state."""

    def __init__(self, routes=None, comments=(), post="created"):
        self.routes, self.comments, self.post, self.calls = dict(routes or {}), list(comments), post, []

    def __call__(self, method, url, body=None, *, allow_redirect=False):
        self.calls.append((method, url))
        if method == "POST":
            text = json.loads(body)["body"]
            if self.post in ("created", "lost_after_commit"):
                row = {"id": 1000 + len(self.comments), "user": {"id": BOT}, "body": text, "issue_url": ISSUE_URL}
                self.comments.append(row)
                return (201, {}, json.dumps(row).encode()) if self.post == "created" else (502, {}, b"bad gateway")
            return 502, {}, b"bad gateway"
        if url.startswith((ISSUE_URL + "/comments", "{}/repositories/{}/issues/27/comments".format(gh.API_ROOT, REPO_ID))):
            return self._comment_page(url)               # BOTH documented forms of the comment resource
        answer = self.routes[(method, url)]
        return answer() if callable(answer) else answer

    def _comment_page(self, url):
        page = 2 if url.endswith("&page=2") else 1
        rows = self.comments[:2] if page == 1 and len(self.comments) > 2 else self.comments[2:] if page == 2 else self.comments
        link = {"Link": '<{}/repositories/{}/issues/27/comments?per_page=100&page=2>; rel="next"'.format(gh.API_ROOT, REPO_ID)} \
            if page == 1 and len(self.comments) > 2 else {}
        return 200, link, json.dumps(rows).encode()

    @property
    def posts(self):
        return sum(1 for m, _ in self.calls if m == "POST")


def ok(doc, headers=None):
    return 200, headers or {}, json.dumps(doc).encode()


def destination_routes(issues=None, issue=None, repo=None):
    return {("GET", API): ok(repo or {"id": REPO_ID, "full_name": REPO}),
            ("GET", API + "/issues?labels=source-monitor-alert&state=open&per_page=100"):
                ok(issues if issues is not None else [{"id": ISSUE_ID, "number": NUMBER},
                                                       {"id": 7, "number": 8, "pull_request": {}}]),
            ("GET", ISSUE_URL): ok(issue or {"id": ISSUE_ID, "number": NUMBER, "state": "open", "repository_url": API,
                                             "labels": [{"name": "source-monitor-alert"}]})}


def refused(code, fn, *args, **kwargs):
    with pytest.raises(c2.Refusal) as exc:
        fn(*args, **kwargs)
    assert exc.value.code == code


# ------------------------------------------------------------------ destination
def test_the_pinned_open_labelled_issue_is_selected_and_a_listed_pull_request_is_ignored():
    assert gh.select_destination(FakeGitHub(destination_routes()), PIN) == c2.Destination(REPO_ID, ISSUE_ID, NUMBER)


@pytest.mark.parametrize("routes, code", [
    (destination_routes(repo={"id": 11, "full_name": REPO}), "destination.repository"),
    (destination_routes(issues=[]), "destination.missing"),
    (destination_routes(issues=[{"id": ISSUE_ID, "number": NUMBER}, {"id": 201, "number": 28}]), "destination.ambiguous"),
    (destination_routes(issues=[{"id": 999, "number": NUMBER}]), "destination.changed"),
    (destination_routes(issue={"id": ISSUE_ID, "state": "closed", "repository_url": API,
                               "labels": [{"name": "source-monitor-alert"}]}), "destination.changed"),
    (destination_routes(issue={"id": ISSUE_ID, "state": "open", "repository_url": API, "labels": []}), "destination.changed"),
    (destination_routes(issue={"id": ISSUE_ID, "state": "open", "repository_url": gh.API_ROOT + "/repos/x/y",
                               "labels": [{"name": "source-monitor-alert"}]}), "destination.changed"),
    (destination_routes(issue={"id": ISSUE_ID, "pull_request": {}}), "destination.pull_request"),
])
def test_an_unexpected_destination_is_refused_with_its_reason(routes, code):
    refused(code, gh.select_destination, FakeGitHub(routes), PIN)


# ------------------------------------------------------------------ comment channel
def _comment(i, body="x", author=999, issue_url=ISSUE_URL):
    return {"id": i, "user": {"id": author}, "body": body, "issue_url": issue_url}


def test_pagination_follows_the_documented_numeric_repository_link_form():
    fake = FakeGitHub(comments=[_comment(1), _comment(2), _comment(3)])
    channel = gh.CommentChannel(fake, PIN)
    first = channel.page(None)
    second = channel.page(first.next_cursor)
    assert [c.id for c in first.comments + second.comments] == [1, 2, 3] and second.next_cursor is None
    assert first.next_cursor.startswith(gh.API_ROOT + "/repositories/10/issues/27/comments?")


def test_a_link_to_another_resource_is_refused():
    fake = FakeGitHub(comments=[_comment(1)])
    fake._comment_page = lambda url: (200, {"Link": '<https://api.github.com/repositories/10/issues/28/comments?page=2>; rel="next"'}, b"[]")
    refused("pagination.foreign_link", gh.CommentChannel(fake, PIN).page, None)


def test_a_comment_whose_issue_url_is_not_the_pinned_issue_blocks_the_scan():
    fake = FakeGitHub(comments=[_comment(1, issue_url=API + "/issues/28")])
    result = c2.deliver(gh.CommentChannel(fake, PIN), fixture(), c2.Destination(REPO_ID, ISSUE_ID, NUMBER), author_id=BOT,
                        history=c2.DispatchHistory(c2.History.NO_PRIOR_DISPATCH, "t"),
                        now=datetime(2026, 9, 29, 11, 59, 30, tzinfo=timezone.utc))
    assert (result.action, result.reason, fake.posts) == ("blocked", "comments.destination", 0)


# ------------------------------------------------------------------ one POST through the real deliver()
NOW = datetime(2026, 9, 29, 11, 59, 30, tzinfo=timezone.utc)
DEST = c2.Destination(REPO_ID, ISSUE_ID, NUMBER)
NEW = c2.DispatchHistory(c2.History.NO_PRIOR_DISPATCH, "test")


@pytest.mark.parametrize("mode, action, reason", [
    ("created", "acknowledged", "created"),
    ("lost_after_commit", "acknowledged", "reconciled_after_post_error"),
    ("lost_before_commit", "unknown", "post_outcome_unknown"),
])
def test_exactly_one_post_whatever_the_response(mode, action, reason):
    fake = FakeGitHub(comments=[_comment(1), _comment(2), _comment(3)], post=mode)
    result = c2.deliver(gh.CommentChannel(fake, PIN), fixture(), DEST, author_id=BOT, history=NEW, now=NOW)
    assert (result.action, result.reason, fake.posts) == (action, reason, 1)


def test_an_existing_acknowledgement_on_page_two_prevents_any_post():
    body = c2.render_comment(fixture(), DEST)
    fake = FakeGitHub(comments=[_comment(1), _comment(2), _comment(3, body=body, author=BOT)])
    result = c2.deliver(gh.CommentChannel(fake, PIN), fixture(), DEST, author_id=BOT, history=NEW, now=NOW)
    assert (result.action, result.reason, result.comment_id, fake.posts) == ("acknowledged", "matching_comment", 3, 0)


# ------------------------------------------------------------------ dispatch history
def outcome_zip(delivery_id, post_issued, names=("outcome.json",)):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, json.dumps({"schema": gh.OUTCOME_SCHEMA, "schema_version": 1, "delivery_id": delivery_id,
                                          "post_issued": post_issued, "action": "x", "reason": "y"}))
    return buf.getvalue()


KEY = "k" * 64


def history_routes(runs, attempts):
    """runs: listing entries; attempts: {(run_id, n): (step_status, step_conclusion, artifact_zip_or_None, expired)}."""
    routes = {("GET", API + "/actions/workflows/{}/runs?per_page=100".format(WORKFLOW_ID)):
              ok({"total_count": len(runs), "workflow_runs": runs})}
    for (run_id, n), (status, conclusion, archive, expired) in attempts.items():
        steps = [] if status is None else [{"name": gh.DELIVERY_STEP, "status": status, "conclusion": conclusion}]
        routes[("GET", API + "/actions/runs/{}/attempts/{}/jobs?per_page=100".format(run_id, n))] = \
            ok({"total_count": 1, "jobs": [{"name": gh.PUBLISH_JOB, "steps": steps}]})
    by_run = {}
    for (run_id, n), (_, _, archive, expired) in attempts.items():
        if archive is not None:
            art_id = run_id * 10 + n
            by_run.setdefault(run_id, []).append({"id": art_id, "name": gh.OUTCOME_ARTIFACT_PREFIX + str(n), "expired": expired})
            routes[("GET", API + "/actions/artifacts/{}/zip".format(art_id))] = (200, {}, archive)
    for run_id in {r for r, _ in attempts}:
        arts = by_run.get(run_id, [])
        routes[("GET", API + "/actions/runs/{}/artifacts?per_page=100".format(run_id))] = \
            ok({"total_count": len(arts), "artifacts": arts})
    return routes


def history(routes, current_attempt=2):
    return gh.dispatch_history(FakeGitHub(routes), repository=REPO, repository_id=REPO_ID, workflow_id=WORKFLOW_ID,
                               run_name=RUN_NAME, current_run_id=CURRENT_RUN, current_attempt=current_attempt, delivery_id=KEY)


def run(run_id, attempts, name=RUN_NAME, workflow_id=WORKFLOW_ID):
    return {"id": run_id, "run_attempt": attempts, "display_title": name, "workflow_id": workflow_id}


def test_no_prior_execution_at_all_is_no_prior_dispatch():
    h = history(history_routes([run(CURRENT_RUN, 1)], {}), current_attempt=1)
    assert h.state is c2.History.NO_PRIOR_DISPATCH and h.evidence_ref == "no prior execution for " + RUN_NAME


@pytest.mark.parametrize("attempt, state", [
    (("completed", "success", outcome_zip(KEY, True), False), c2.History.PRIOR_DISPATCH),
    (("completed", "success", outcome_zip(KEY, False), False), c2.History.NO_PRIOR_DISPATCH),
    (("completed", "success", outcome_zip("other" * 12 + "abcd", True), False), c2.History.NO_PRIOR_DISPATCH),
    (("completed", "failure", None, False), c2.History.UNKNOWN),                           # started, no record
    (("completed", "success", outcome_zip(KEY, False), True), c2.History.UNKNOWN),          # record expired
    (("completed", "success", outcome_zip(KEY, False, names=("a", "b")), False), c2.History.UNKNOWN),
    (("completed", "skipped", None, False), c2.History.NO_PRIOR_DISPATCH),
    (("queued", None, None, False), c2.History.NO_PRIOR_DISPATCH),
    ((None, None, None, False), c2.History.NO_PRIOR_DISPATCH),                               # the step never existed
])
def test_the_current_runs_earlier_attempt_is_classified(attempt, state):
    assert history(history_routes([run(CURRENT_RUN, 2)], {(CURRENT_RUN, 1): attempt})).state is state


def test_the_current_run_is_examined_even_when_the_listing_lags():
    """Defect found by re-reading 2026-09-30: the current run's earlier attempts were examined only if it was listed."""
    h = history(history_routes([], {(CURRENT_RUN, 1): ("completed", "failure", None, False)}))
    assert h.state is c2.History.UNKNOWN and "run 900 attempt 1" in h.evidence_ref


def test_another_run_with_the_same_run_name_is_examined_in_every_attempt():
    routes = history_routes([run(CURRENT_RUN, 1), run(800, 2)],
                            {(800, 1): ("completed", "success", outcome_zip(KEY, False), False),
                             (800, 2): ("completed", "success", outcome_zip(KEY, True), False)})
    h = history(routes, current_attempt=1)
    assert h.state is c2.History.PRIOR_DISPATCH and h.evidence_ref == "run 800 attempt 2: a POST was issued for this delivery"


@pytest.mark.parametrize("routes, needle", [
    (lambda: {("GET", API + "/actions/workflows/555/runs?per_page=100"): ok({"total_count": 5, "workflow_runs": []})},
     "pagination.incomplete"),
    (lambda: history_routes([run(1, 1, workflow_id=556)], {}), "history.workflow"),
    (lambda: {}, "KeyError"),
])
def test_an_incomplete_or_foreign_history_is_unknown_never_new(routes, needle):
    h = history(routes(), current_attempt=1)
    assert h.state is c2.History.UNKNOWN and needle in h.evidence_ref


# ------------------------------------------------------------------ transport
def test_the_transport_refuses_any_host_but_githubs_api():
    refused("transport.host", gh.Transport("T"), "GET", "https://example.com/repos/o/r")


def test_an_http_error_response_is_returned_and_closed():
    t, holder = gh.Transport("T"), {}

    class Opener:
        def open(self, req, timeout):
            holder["headers"] = dict(req.unredirected_hdrs)
            holder["error"] = urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", {}, io.BytesIO(b"bad gateway"))
            raise holder["error"]
    t._no_redirect = Opener()
    assert t("POST", API + "/issues/27/comments", b"{}") == (502, {}, b"bad gateway")
    assert holder["error"].fp.closed and holder["headers"] == {"Authorization": "Bearer T"}
