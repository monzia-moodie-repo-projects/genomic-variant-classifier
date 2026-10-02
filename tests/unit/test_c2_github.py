"""The C2 GitHub adapter against a fake GitHub that emits the DOCUMENTED response forms (verified 2026-09-30):
page-two Link URLs under /repositories/{numeric id}/...; comments carrying only an issue_url; pull requests inside the
issues listing; total_count on every object listing. Every refusal pins its exact reason; every delivery counts POSTs.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor import c2_github as gh
from genomic_variant_classifier.source_monitor import c2_protocol as c2
from tests.unit.test_c2_protocol import fixture

REPO, REPO_ID, ISSUE_ID, NUMBER, BOT = "o/r", 10, 200, 27, 41898282
PIN = gh.Pinned(REPO, REPO_ID, ISSUE_ID, NUMBER, BOT, "source-monitor-alert")
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
                        clock=lambda: datetime(2026, 9, 29, 11, 59, 30, tzinfo=timezone.utc))
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
    result = c2.deliver(gh.CommentChannel(fake, PIN), fixture(), DEST, author_id=BOT, history=NEW, clock=lambda: NOW)
    assert (result.action, result.reason, fake.posts) == (action, reason, 1)


def test_an_existing_acknowledgement_on_page_two_prevents_any_post():
    body = c2.render_comment(fixture(), DEST)
    fake = FakeGitHub(comments=[_comment(1), _comment(2), _comment(3, body=body, author=BOT)])
    result = c2.deliver(gh.CommentChannel(fake, PIN), fixture(), DEST, author_id=BOT, history=NEW, clock=lambda: NOW)
    assert (result.action, result.reason, result.comment_id, fake.posts) == ("acknowledged", "matching_comment", 3, 0)


# ------------------------------------------------------------------ dispatch history: the JOB-LOG journal (C2 repairs 3)
# MEASURED 2026-10-01: after a re-run the earlier attempt's ARTIFACTS vanish from every listing, while its job log stays
# retrievable by job id. REST log lines are "<ISO-8601>Z <text>"; the real attempt-1 log of qualification run 36897111869 is
# the fixture below (it predates journaling: a started delivery step with NO journal line).
# Named *.txt, not *.log: .gitignore L51 ignores *.log, which once kept this fixture out of the patch (2026-10-01).
REAL_LOG = (Path(__file__).resolve().parents[1] / "fixtures" / "source_monitor_runs" / "qualification_publish_job_attempt1_log.txt").read_bytes()
KEY = "a" * 64                    # delivery keys are SHA-256 hex digests (C2 repairs 4 validates the syntax)
OTHER = "b" * 64
TS = "2026-10-01T17:08:22.9474469Z "


def journal_log(*journal):
    """A log in the measured format: real runner lines around the given journal lines."""
    lines = ["2026-10-01T17:08:14.4910484Z Current runner version: '2.337.0'", "2026-10-01T17:08:22.9474000Z receipt: issuer checker event kind review_required"]
    lines += [TS + j for j in journal] + ["2026-10-01T17:08:24.8517866Z Cleaning up orphan processes"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def attempt_line(key=KEY):
    return gh.journal_attempt_line(key)


def outcome_line(key=KEY, posted=False, **override):
    doc = {"schema": gh.OUTCOME_SCHEMA, "schema_version": 1, "delivery_id": key, "post_issued": posted,
           "action": "acknowledged" if posted else "preview", "reason": "created" if posted else "manual_verification"}
    doc.update(override)
    return gh.journal_outcome_line(doc)


def history_routes(runs, attempts):
    """attempts: {(run_id, n): (step_status, step_conclusion, log_bytes_or_None, publish_jobs)}; each publish job is completed."""
    routes = {("GET", API + "/actions/workflows/{}/runs?per_page=100".format(WORKFLOW_ID)):
              ok({"total_count": len(runs), "workflow_runs": runs})}
    for (run_id, n), (status, conclusion, log, publish_jobs) in attempts.items():
        steps = [] if status is None else [{"name": gh.DELIVERY_STEP, "status": status, "conclusion": conclusion}]
        jobs = [{"id": run_id * 100 + n * 10 + k, "name": gh.PUBLISH_JOB, "status": "completed", "steps": steps} for k in range(publish_jobs)]
        routes[("GET", API + "/actions/runs/{}/attempts/{}/jobs?per_page=100".format(run_id, n))] = ok({"total_count": len(jobs), "jobs": jobs})
        for job in jobs:
            routes[("GET", API + "/actions/jobs/{}/logs".format(job["id"]))] = (200, {}, log) if log is not None else (404, {}, b"not found")
    return routes


def history(routes, current_attempt=2):
    return gh.dispatch_history(FakeGitHub(routes), repository=REPO, repository_id=REPO_ID, workflow_id=WORKFLOW_ID,
                               run_name=RUN_NAME, current_run_id=CURRENT_RUN, current_attempt=current_attempt, delivery_id=KEY)


def run(run_id, attempts, name=RUN_NAME, workflow_id=WORKFLOW_ID):
    return {"id": run_id, "run_attempt": attempts, "display_title": name, "workflow_id": workflow_id}


def test_no_prior_execution_at_all_is_no_prior_dispatch():
    h = history(history_routes([run(CURRENT_RUN, 1)], {}), current_attempt=1)
    assert h.state is c2.History.NO_PRIOR_DISPATCH and h.evidence_ref == "no prior execution for " + RUN_NAME


P, N, U = c2.History.PRIOR_DISPATCH, c2.History.NO_PRIOR_DISPATCH, c2.History.UNKNOWN


@pytest.mark.parametrize("attempt, state", [
    (("completed", "success", journal_log(attempt_line(), outcome_line(posted=True)), 1), P),
    (("completed", "failure", journal_log(attempt_line()), 1), P),      # a recovered intent: a POST may have been issued
    (("completed", "success", journal_log(attempt_line(), outcome_line(posted=False, action="unknown", reason="post_outcome_unknown")), 1), P),  # a false outcome never erases an intent
    (("completed", "success", journal_log(outcome_line(posted=False)), 1), N),
    (("completed", "success", journal_log(attempt_line(OTHER), outcome_line(OTHER, True)), 1), N),
    (("completed", "success", journal_log(attempt_line(OTHER)), 1), N),     # its ONE POST was another delivery's
    (("completed", "skipped", None, 1), N),                                 # ONLY an explicit skip is no dispatch
    (("completed", "failure", journal_log(), 1), U),                        # started, no journal line
    (("completed", "success", REAL_LOG, 1), U),                             # the REAL pre-journal log
    (("completed", "success", journal_log("C2-ATTEMPT {not json"), 1), U),
    (("completed", "success", journal_log(attempt_line().replace('"schema_version":1', '"schema_version":1,"x":1')), 1), U),
    (("completed", "success", journal_log(outcome_line(posted=False)).replace(b"\n2026", b"\n" + TS.encode() + b"C2-OUTCOME\n2026", 1), 1), U),
    (("completed", "success", None, 1), U),                                 # log not retrievable
    (("completed", "success", b"\xff\xfe not utf-8 " + attempt_line().encode(), 1), U),
    (("completed", "success", journal_log(outcome_line(posted=False)), 2), U),   # publish job not unique
    (("queued", None, None, 1), U),                                         # non-terminal proves nothing
    ((None, None, None, 1), U),                                             # the delivery step is absent
    # the owner's counterexamples (2026-10-02) and the related probes, each REPRODUCED on 7657b38b before this repair:
    (("completed", "success", journal_log(outcome_line(posted=False, action=42, reason=[])), 1), U),
    (("completed", "success", journal_log(outcome_line(posted=False, reason="")), 1), U),
    (("completed", "success", journal_log(outcome_line(posted=False, action="sent")), 1), U),   # outside the vocabulary
    (("completed", "success", journal_log(attempt_line(OTHER), attempt_line("c" * 64)), 1), U),   # two intents
    (("completed", "success", journal_log(outcome_line(OTHER), outcome_line("c" * 64)), 1), U),    # two outcomes
    (("completed", "success", journal_log(attempt_line("x")), 1), U),                         # not a delivery key
    (("completed", "success", journal_log(outcome_line(KEY), attempt_line(KEY)), 1), U),      # outcome before intent
    (("completed", "success", journal_log(attempt_line(KEY), outcome_line(OTHER)), 1), U),    # conflicting identities
    (("completed", "success", journal_log(outcome_line("", True)), 1), U),                    # a POST with no key
    (("completed", "success", journal_log(outcome_line(posted=False).replace('"schema_version":1', '"schema_version":1.0')), 1), U),
])
def test_the_current_runs_earlier_attempt_is_classified_from_its_job_log(attempt, state):
    assert history(history_routes([run(CURRENT_RUN, 2)], {(CURRENT_RUN, 1): attempt})).state is state


@pytest.mark.parametrize("jobs", [
    [],                                                                      # the owner's empty-job-list counterexample
    [{"id": 9001, "name": "verify", "status": "completed", "steps": []}],    # no publish job at all
    [{"id": 9001, "name": gh.PUBLISH_JOB, "status": "in_progress", "steps": [{"name": gh.DELIVERY_STEP, "status": "completed", "conclusion": "skipped"}]}],
    [{"id": 9001, "name": gh.PUBLISH_JOB, "status": "completed", "steps": [{"name": gh.DELIVERY_STEP, "status": "completed", "conclusion": "skipped"}] * 2}],
    [{"id": 9001, "name": gh.PUBLISH_JOB, "status": "completed", "steps": "not a list"}],
])
def test_missing_or_ambiguous_job_structure_is_unknown_never_no_dispatch(jobs):
    routes = history_routes([run(CURRENT_RUN, 2)], {})
    routes[("GET", API + "/actions/runs/{}/attempts/1/jobs?per_page=100".format(CURRENT_RUN))] = ok({"total_count": len(jobs), "jobs": jobs})
    assert history(routes).state is c2.History.UNKNOWN


def test_the_journal_parser_accepts_the_measured_format_and_a_leading_byte_order_mark():
    assert gh.read_journal("\ufeff" + TS + attempt_line(), KEY) is True
    assert gh.read_journal(REAL_LOG.decode("utf-8"), KEY) is None
    assert gh.read_journal(REAL_LOG.decode("utf-8") + TS + outcome_line(posted=False) + "\n", KEY) is False
    assert gh.read_journal(TS + attempt_line(), "not-a-key") is None


def test_one_delivery_action_vocabulary_owned_by_the_protocol():
    """Every Result constructed anywhere uses c2.ACTIONS, and Result itself refuses anything else."""
    import ast
    root = Path(__file__).resolve().parents[2]
    used = set()
    for rel in ("src/genomic_variant_classifier/source_monitor/c2_protocol.py", "scripts/publish_monitor_receipt.py"):
        for node in ast.walk(ast.parse((root / rel).read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "Result"
                    and node.args and isinstance(node.args[0], ast.Constant)):
                used.add(node.args[0].value)
    assert used and used <= c2.ACTIONS
    with pytest.raises(c2.Refusal):
        c2.Result("sent", "x")


def test_the_current_run_is_examined_even_when_the_listing_lags():
    """Defect found by re-reading 2026-09-30: the current run's earlier attempts were examined only if it was listed."""
    h = history(history_routes([], {(CURRENT_RUN, 1): ("completed", "failure", journal_log(), 1)}))
    assert h.state is c2.History.UNKNOWN and "run 900 attempt 1" in h.evidence_ref


def test_another_run_with_the_same_run_name_is_examined_in_every_attempt():
    routes = history_routes([run(CURRENT_RUN, 1), run(800, 2)],
                            {(800, 1): ("completed", "success", journal_log(outcome_line(posted=False)), 1),
                             (800, 2): ("completed", "success", journal_log(attempt_line(), outcome_line(posted=True)), 1)})
    h = history(routes, current_attempt=1)
    assert h.state is c2.History.PRIOR_DISPATCH
    assert h.evidence_ref == "run 800 attempt 2: the journal shows a POST may have been issued for this delivery"


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



def test_the_comment_channel_gates_after_preparing_and_before_the_request():
    fake, order = FakeGitHub(), []
    channel = gh.CommentChannel(lambda *a, **k: (order.append("request"), fake(*a, **k))[1], PIN)
    channel.create_once("x", before_send=lambda: order.append("gate"))
    assert order == ["gate", "request"]


def test_a_refusing_gate_means_no_request_at_all():
    fake = FakeGitHub()

    def refuse():
        raise c2.PostNotAttempted("receipt.needs_reverification", 901.0)
    with pytest.raises(c2.PostNotAttempted):
        gh.CommentChannel(fake, PIN).create_once("x", before_send=refuse)
    assert fake.calls == []


def test_a_configured_label_is_percent_encoded_in_the_query():
    """The label comes from configuration since 2026-10-01; unencoded, a space, '&' or '#' would corrupt the query."""
    pin = gh.Pinned(REPO, REPO_ID, ISSUE_ID, NUMBER, BOT, "a b&c#d")
    routes = destination_routes(issue={"id": ISSUE_ID, "number": NUMBER, "state": "open", "repository_url": API,
                                       "labels": [{"name": "a b&c#d"}]})
    routes[("GET", API + "/issues?labels=a%20b%26c%23d&state=open&per_page=100")] = \
        routes.pop(("GET", API + "/issues?labels=source-monitor-alert&state=open&per_page=100"))
    fake = FakeGitHub(routes)
    try:
        result = gh.select_destination(fake, pin)
    except KeyError:                     # an unrouted (malformed) URL: judged by the request ACTUALLY made, below
        result = None
    listing = [u for m, u in fake.calls if "/issues?labels=" in u]
    assert listing == [API + "/issues?labels=a%20b%26c%23d&state=open&per_page=100"], listing
    assert result == c2.Destination(REPO_ID, ISSUE_ID, NUMBER)
