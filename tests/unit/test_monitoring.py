"""A monitoring layer that cannot fail cannot report.

MEASURED 2026-09-14, at repository head 5cb1a330, every instrument that could
have revealed a broken agent reported success instead:

    check_agents_active.py   22 of 22 agents STALE at 84.59 days -> "OK", exit 0
    audit_agent_operational  structure only; never asks whether anything ran
    VersionMonitorAgent      status "ok" was a LITERAL, set unconditionally
    run_pipeline             printed "[OK]" beside action=error
    run_data_freshness.py    returned 0 whether or not a change was detected
    _record_run_telemetry    ignored result["status"], so degraded became ok

gnomAD -- Genome Aggregation Database -- version 4.1.1 was released 2026-03-30
and found by hand on 2026-09-12.

Every test below exists because one of those silences was possible. They are
written as refusals: the subject must FAIL when it should, because passing
when it should not is the defect.

Author: Monzia Moodie
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor.finding_store import (
    FindingStore, StoreError)
from genomic_variant_classifier.source_monitor.heartbeat import (
    SignalOutcome, signal_outcome, signal_start)
from genomic_variant_classifier.source_monitor.monitor_supervisor import (
    Health, TargetResult, supervise)
from genomic_variant_classifier.source_monitor.reason_catalog import (
    ContractFinding, ProfileError, Reason, ReasonProfile,
    assess_failure_record, make_failure_record)
from genomic_variant_classifier.source_monitor import gnomad_release_check as grc


PROFILE = ReasonProfile(
    profile_id="release-listing",
    revision="r1",
    # RECOGNITION uses the whole shared catalog. MEASURED 2026-09-14: a
    # profile whose known_codes held a hand-picked subset classified a
    # globally defined code as UNKNOWN rather than KNOWN-BUT-DISALLOWED,
    # which are different findings with different causes.
    known_codes=frozenset(r.value for r in Reason),
    allowed_codes=frozenset({Reason.REQUEST_ENDPOINT.value,
                             Reason.TRAVERSAL_TRUNCATED.value}),
)

#: MEASURED 2026-09-14 from the live listing endpoint. Twelve prefixes, one of
#: them carrying a "v" that a strict numeric parse raises on.
MEASURED_PREFIXES = [
    "release/2.1.1/", "release/2.1/", "release/3.0.1/", "release/3.0/",
    "release/3.1.1/", "release/3.1.2/", "release/3.1.3/", "release/3.1/",
    "release/4.0/", "release/4.1.1/", "release/4.1/", "release/v4.0/",
]


# --------------------------------------------------------------------------
# 1. Reason identities survive every boundary
# --------------------------------------------------------------------------

def _record():
    return make_failure_record(PROFILE, Reason.REQUEST_ENDPOINT, "wrong endpoint")


def _roundtrip(record):
    return json.loads(json.dumps(record))


def test_a_conforming_record_is_recognized_after_persistence():
    a = assess_failure_record(_roundtrip(_record()), PROFILE)
    assert a.profile_failure_recognized and a.contract_health == ()


def test_recognizing_a_failure_never_makes_the_check_successful():
    assert assess_failure_record(_roundtrip(_record()), PROFILE).exit_code == 2


def test_a_deleted_reason_is_not_recovered_from_free_text():
    broken = {k: v for k, v in _record().items() if k != "reason"}
    broken["detail"] = "reason=request.endpoint"
    a = assess_failure_record(_roundtrip(broken), PROFILE)
    assert a.contract_health == (ContractFinding.INVALID_FAILURE_RECORD.value,)


def test_a_record_flattened_to_a_string_is_refused():
    a = assess_failure_record("failure: request.endpoint", PROFILE)
    assert a.contract_health == (ContractFinding.INVALID_FAILURE_RECORD.value,)


def test_an_unknown_reason_preserves_the_original_alongside_the_finding():
    a = assess_failure_record(_roundtrip(dict(_record(), reason="future.x")),
                              PROFILE)
    assert a.contract_health == (ContractFinding.UNKNOWN_REASON.value,)
    assert a.reported_reason == "future.x"


def test_a_known_but_disallowed_reason_is_a_different_finding():
    a = assess_failure_record(
        _roundtrip(dict(_record(), reason=Reason.ARTIFACT_DIGEST_MISMATCH.value)),
        PROFILE)
    assert a.contract_health == (ContractFinding.REASON_NOT_ALLOWED.value,)


def test_changing_only_human_wording_does_not_change_the_disposition():
    a = assess_failure_record(_roundtrip(dict(_record(), detail="other words")),
                              PROFILE)
    assert a.profile_failure_recognized


def test_a_boolean_schema_version_is_refused():
    """True == 1, so an `== 1` check would accept it."""
    a = assess_failure_record(_roundtrip(dict(_record(), schema_version=True)),
                              PROFILE)
    assert a.contract_health == (
        ContractFinding.UNSUPPORTED_FAILURE_SCHEMA.value,)


def test_a_profile_naming_an_undefined_reason_is_refused_before_activation():
    with pytest.raises(ProfileError, match="undefined"):
        ReasonProfile("x", "r1", frozenset({Reason.REQUEST_ENDPOINT.value}),
                      frozenset({"made.up"}))


def test_a_producer_cannot_emit_a_reason_its_profile_forbids():
    with pytest.raises(ProfileError, match="does not permit"):
        make_failure_record(PROFILE, Reason.ARTIFACT_DIGEST_MISMATCH, "x")


# --------------------------------------------------------------------------
# 2. Commit precedes delivery, and survives the process
# --------------------------------------------------------------------------

def test_a_committed_finding_is_pending_until_a_channel_accepts(tmp_path):
    store = FindingStore(tmp_path / "f.sqlite3")
    attempt = store.begin_attempt("s")
    c = store.commit_finding(attempt_id=attempt, subject="s", record=_record())
    assert len(store.pending_deliveries()) == 1
    assert store.get_finding(c.event_id)["delivered_at"] is None


def test_a_pending_delivery_survives_a_fresh_process(tmp_path):
    """The store is written by one interpreter and read by ANOTHER. A same-
    process reopen would not establish recovery."""
    db = tmp_path / "f.sqlite3"
    store = FindingStore(db)
    attempt = store.begin_attempt("s")
    c = store.commit_finding(attempt_id=attempt, subject="s", record=_record())
    code = (
        "import sys, json;"
        "sys.path.insert(0, {src!r});"
        "from genomic_variant_classifier.source_monitor.finding_store "
        "import FindingStore;"
        "s=FindingStore({db!r});"
        "print(json.dumps([p['event_id'] for p in s.pending_deliveries()]))"
    ).format(src=str(Path(__file__).resolve().parents[2] / "src"), db=str(db))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip()) == [c.event_id]


def test_a_finding_cannot_be_delivered_twice(tmp_path):
    store = FindingStore(tmp_path / "f.sqlite3")
    attempt = store.begin_attempt("s")
    c = store.commit_finding(attempt_id=attempt, subject="s", record=_record())
    store.record_delivery(c.event_id, "ref-1")
    with pytest.raises(StoreError, match="no undelivered finding"):
        store.record_delivery(c.event_id, "ref-2")


def test_an_unknown_attempt_outcome_is_refused(tmp_path):
    store = FindingStore(tmp_path / "f.sqlite3")
    with pytest.raises(StoreError, match="unknown attempt outcome"):
        store.finish_attempt(store.begin_attempt("s"), "fine")


# --------------------------------------------------------------------------
# 3. The supervisor accounts for every REQUIRED target
# --------------------------------------------------------------------------

REQ = ("a", "b")


def test_a_required_target_that_never_reported_is_a_failure():
    r = supervise(REQ, [TargetResult("a", Health.COMPLETE)])
    assert r.exit_code == 2
    assert any(x.health is Health.ABSENT and x.target == "b" for x in r.results)


def test_a_policy_requiring_nothing_cannot_pass_silently():
    """A monitor that cannot fail cannot report."""
    assert supervise([], []).exit_code == 2


def test_a_partial_observation_keeps_its_valid_witness():
    r = supervise(REQ, [TargetResult("a", Health.INCOMPLETE,
                                     findings=("4.1.1 seen on page 1",),
                                     reason=Reason.TRAVERSAL_TRUNCATED),
                        TargetResult("b", Health.COMPLETE)])
    assert r.exit_code == 2
    assert r.review_findings == (("a", "4.1.1 seen on page 1"),)


def test_a_finding_without_failure_is_review_not_failure():
    r = supervise(REQ, [TargetResult("a", Health.COMPLETE, findings=("x",)),
                        TargetResult("b", Health.COMPLETE)])
    assert r.exit_code == 1


def test_all_complete_with_no_findings_is_zero():
    assert supervise(REQ, [TargetResult(t, Health.COMPLETE)
                           for t in REQ]).exit_code == 0


def test_a_duplicate_or_undeclared_result_is_a_supervisor_finding():
    dup = supervise(REQ, [TargetResult("a", Health.COMPLETE),
                          TargetResult("a", Health.COMPLETE),
                          TargetResult("b", Health.COMPLETE)])
    assert dup.exit_code == 2
    extra = supervise(REQ, [TargetResult(t, Health.COMPLETE) for t in REQ]
                      + [TargetResult("c", Health.COMPLETE)])
    assert extra.exit_code == 2


def test_health_must_be_the_enum_not_a_string():
    with pytest.raises(ValueError, match="Health member"):
        TargetResult("a", "complete")


# --------------------------------------------------------------------------
# 4. The release check reports what it could not examine
#
# EVERY test here injects TRANSPORT, the same seam live execution uses. The
# previous tests passed a pre-built list of pages and never entered the
# request loop, which is why a guarantee asserted in three docstrings passed
# 35 tests and failed in reality.
# --------------------------------------------------------------------------

def _transport(*pages):
    """Serve pages, or raise an exception, through the injected seam."""
    seq = iter(pages)

    def send(url):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item).encode("utf-8")

    return send


_PAGE_ONE = {"kind": "storage#objects", "prefixes": ["release/4.1.1/"],
             "nextPageToken": "page-2"}

_PAGE_ONE_TERMINAL = {"kind": "storage#objects", "prefixes": ["release/4.1.1/"]}


def test_a_later_timeout_preserves_the_earlier_release():
    """THE DEFECT MEASURED 2026-09-14. Page one held 4.1.1, page two timed
    out, and the finding was DISCARDED -- health FAILED, reason
    TRANSPORT_UNREACHABLE. The accumulator must survive every exit."""
    r = grc.observe_releases(
        transport=_transport(_PAGE_ONE, TimeoutError("second page")))
    assert r.health is Health.INCOMPLETE
    assert r.reason is Reason.TRANSPORT_TIMEOUT
    assert any("4.1.1" in f for f in r.findings)


def test_the_measured_listing_reports_the_newer_release():
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    assert r.health is Health.COMPLETE
    assert any("4.1.1" in f for f in r.findings)


@pytest.mark.parametrize("page, reason", [
    pytest.param({"kind": "storage#objects", "prefixes": ["release/4.1/"],
                  "nextPageToken": ""},
                 Reason.TRAVERSAL_TOKEN_MALFORMED, id="empty_token_is_not_terminal"),
    pytest.param({"kind": "storage#buckets", "prefixes": ["release/4.1/"]},
                 Reason.RESPONSE_UNEXPECTED_SHAPE, id="wrong_kind"),
    pytest.param({"kind": "storage#objects", "prefixes": {"a": 1}},
                 Reason.RESPONSE_UNEXPECTED_SHAPE, id="prefixes_as_a_mapping"),
    pytest.param({"kind": "storage#objects", "prefixes": [42]},
                 Reason.RESPONSE_UNEXPECTED_SHAPE, id="a_prefix_that_is_not_a_string"),
])
def test_a_malformed_page_is_named_not_accepted(page, reason):
    r = grc.observe_releases(transport=_transport(page))
    assert r.health is Health.INCOMPLETE
    assert r.reason is reason


def test_no_parseable_version_is_not_a_clean_comparison():
    """MEASURED: a collection holding only release/latest/ read as COMPLETE
    with no finding."""
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": ["release/latest/"]}))
    assert r.health is Health.FAILED
    assert r.reason is Reason.RESPONSE_UNEXPECTED_SHAPE


def test_a_repeated_continuation_token_is_a_cycle():
    r = grc.observe_releases(transport=_transport(_PAGE_ONE, _PAGE_ONE))
    assert r.health is Health.INCOMPLETE
    assert r.reason is Reason.TRAVERSAL_TOKEN_CYCLE
    assert any("4.1.1" in f for f in r.findings)      # witness still kept


def test_a_page_budget_stops_the_traversal_without_losing_pages():
    page = dict(_PAGE_ONE)
    state = grc.traverse(lambda url: json.dumps(
        {"kind": "storage#objects", "prefixes": ["release/4.1.1/"],
         "nextPageToken": "t{}".format(len(url))}).encode(), max_pages=2)
    assert state.reason is Reason.TRAVERSAL_BUDGET_EXHAUSTED
    assert state.pages_read == 2
    assert state.prefixes                              # what was read SURVIVES


def test_duplicate_json_keys_are_refused():
    def send(url):
        return b'{"kind":"storage#objects","kind":"other","prefixes":[]}'
    r = grc.observe_releases(transport=send)
    assert r.reason is Reason.RESPONSE_JSON_DUPLICATE


def test_the_request_asks_for_its_own_completion_evidence():
    """fields=prefixes OMITS nextPageToken, so its absence would mean nothing."""
    assert "nextPageToken" in grc.BASE_QUERY["fields"]


def test_the_v_prefixed_release_parses():
    assert grc.parse_version("release/v4.0/") == (4, 0)


def test_version_ordering_is_numeric_not_lexicographic():
    assert "4.10" < "4.9"                       # what strings would do
    assert grc.parse_version("release/4.10/") > grc.parse_version("release/4.9/")


@pytest.mark.parametrize("bad", ["release/", "release/latest/", "release/4.x/",
                                 "release/-1/", "release/+2/", "release/4 1/"])
def test_a_value_outside_the_release_grammar_is_refused(bad):
    """MEASURED: 'release/-1/' parsed as (-1,), sorting below every real
    release while silently participating in comparisons."""
    assert grc.parse_version(bad) is None


# --------------------------------------------------------------------------
# 5. The heartbeat reports EXECUTION, not findings
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, code):
        self._code = code

    def getcode(self):
        return self._code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("exit_code, suffix", [
    pytest.param(0, "", id="qualified_no_action_is_success"),
    pytest.param(1, "", id="review_required_is_ALSO_success"),
    pytest.param(2, "/fail", id="not_qualified_is_failure"),
])
def test_the_exit_code_decides_the_signal(monkeypatch, exit_code, suffix):
    """Exit 1 signals SUCCESS. A monitor that FINDS something has worked;
    signalling failure there trains the reader to ignore alarms on exactly the
    runs that matter most."""
    sent = []
    from genomic_variant_classifier.source_monitor import heartbeat as hb
    monkeypatch.setattr(hb.urllib.request, "urlopen",
                        lambda req, timeout=None: (sent.append(req.full_url),
                                                   _Resp(200))[1])
    signal_outcome("https://hc.example/tok", exit_code)
    assert sent[0] == "https://hc.example/tok" + suffix


def test_an_unrecognised_exit_code_does_not_default_to_success(monkeypatch):
    sent = []
    from genomic_variant_classifier.source_monitor import heartbeat as hb
    monkeypatch.setattr(hb.urllib.request, "urlopen",
                        lambda req, timeout=None: (sent.append(req.full_url),
                                                   _Resp(200))[1])
    signal_outcome("https://hc.example/tok", 7)
    assert sent[0].endswith("/fail")


def test_a_boolean_exit_code_is_refused():
    with pytest.raises(TypeError, match="must be an int"):
        signal_outcome("https://hc.example/tok", True)


def test_an_unreachable_heartbeat_is_visible_and_does_not_raise(monkeypatch):
    """The monitoring verdict STANDS. A watchdog that can veto its subject is
    a second point of failure, not a safeguard."""
    from genomic_variant_classifier.source_monitor import heartbeat as hb

    def boom(req, timeout=None):
        raise OSError("unreachable")

    monkeypatch.setattr(hb.urllib.request, "urlopen", boom)
    out = signal_outcome("https://hc.example/tok", 0)
    # The detail is the exception's CLASS NAME only -- see
    # test_a_synthetic_credential_in_the_exception_message_does_not_leak
    # for why the raw message is never recorded here.
    assert out.attempted and not out.delivered and out.detail == "OSError"


def test_a_synthetic_credential_in_the_exception_message_does_not_leak(monkeypatch):
    """MEASURED 2026-09-16, from an external ruling's reading of this module:
    str(exc) was serialized directly into the report. Proven empirically: an
    exception whose OWN message embeds the request's full URL -- ordinary
    behaviour for several urllib and SSL failure paths -- put the endpoint
    straight into the published report."""
    from genomic_variant_classifier.source_monitor import heartbeat as hb

    secret = "https://hc-ping.com/synthetic-credential-do-not-leak-me"

    def boom(req, timeout=None):
        raise ValueError("unable to open {}: refused".format(req.full_url))

    monkeypatch.setattr(hb.urllib.request, "urlopen", boom)
    out = signal_outcome(secret, 0)
    assert secret not in out.detail
    assert out.detail == "ValueError"


def test_a_malformed_url_is_reported_not_raised(monkeypatch):
    """MEASURED 2026-09-16: urlsplit() sat OUTSIDE every exception handler.
    urlsplit("https://[::1/malformed") raises ValueError: Invalid IPv6 URL --
    confirmed directly -- so a malformed secret crashed the run instead of
    being reported as a configuration problem."""
    from genomic_variant_classifier.source_monitor.heartbeat import _send
    out = _send("https://[::1/malformed", "/fail")
    assert not out.attempted
    assert out.endpoint_configured
    assert "does not parse as a URL" in out.detail


def test_an_http_error_reports_only_the_status_code(monkeypatch):
    """The status code is useful and safe; the exception's own message is
    neither guaranteed safe nor needed once the code is captured."""
    import urllib.error
    from genomic_variant_classifier.source_monitor import heartbeat as hb

    secret = "https://hc-ping.com/another-secret-token"

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(hb.urllib.request, "urlopen", boom)
    out = signal_outcome(secret, 0)
    assert out.detail == "http 404"
    assert secret not in out.detail


def test_a_non_2xx_response_is_not_delivered(monkeypatch):
    from genomic_variant_classifier.source_monitor import heartbeat as hb
    monkeypatch.setattr(hb.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(503))
    assert not signal_outcome("https://hc.example/tok", 0).delivered


def test_an_unconfigured_endpoint_is_a_state_not_a_pretended_signal():
    out = signal_outcome(None, 0)
    assert not out.attempted and not out.endpoint_configured
    assert not signal_start(None).attempted


# --------------------------------------------------------------------------
# 6. A contradictory result must not be constructible
# --------------------------------------------------------------------------

def test_a_complete_result_cannot_carry_a_failure_reason():
    """MEASURED 2026-09-14: COMPLETE with TRANSPORT_TIMEOUT was accepted and,
    with no review finding, the supervisor returned EXIT 0."""
    with pytest.raises(ValueError, match="cannot carry a failure reason"):
        TargetResult("x", Health.COMPLETE, reason=Reason.TRANSPORT_TIMEOUT)


@pytest.mark.parametrize("health", [
    pytest.param(Health.INCOMPLETE, id="incomplete"),
    pytest.param(Health.FAILED, id="failed"),
])
def test_an_unqualified_result_must_name_its_reason(health):
    with pytest.raises(ValueError, match="must name its reason"):
        TargetResult("x", health)


def test_absent_is_constructed_by_the_supervisor_without_a_reason():
    """ABSENT is the supervisor's own finding about a target that never
    reported; the producer supplied nothing to name."""
    r = supervise(("a",), [])
    assert r.results[0].health is Health.ABSENT
    assert r.exit_code == 2


# --------------------------------------------------------------------------
# 7. The runner: evidence must CROSS the store
#
# MEASURED 2026-09-14 against the previous runner: main() never called
# assess_failure_record(), supervision consumed the in-memory TargetResult,
# nothing called record_delivery(), and the success heartbeat was sent BEFORE
# the report write could fail. Each was a correct component that was not on
# the path.
# --------------------------------------------------------------------------

from genomic_variant_classifier.source_monitor import run_monitor as rm


def _stub_check(monkeypatch, *pages):
    """Point the registered check at an injected transport."""
    def run(**_kw):
        return grc.observe_releases(profile=rm.RELEASE_PROFILE,
                                    transport=_transport(*pages))
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)


def test_a_failure_is_persisted_recovered_and_assessed(monkeypatch, tmp_path):
    """The assessment must come from the RECOVERED record, so a finding that
    cannot survive its own store is caught before success is reported."""
    _stub_check(monkeypatch, _PAGE_ONE, TimeoutError("second page"))
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"),
                    "--report", str(report)])
    assert code == 2
    doc = json.loads(report.read_text())
    assert doc["assessments"] == [{
        "target": "gnomad-public-releases",
        "reason": Reason.TRANSPORT_TIMEOUT.value,
        "recognized": True,
        "contract_health": [],
    }]


def test_the_witness_survives_into_the_runners_report(monkeypatch, tmp_path):
    _stub_check(monkeypatch, _PAGE_ONE, TimeoutError("second page"))
    report = tmp_path / "r.json"
    rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    assert any("4.1.1" in f for f in doc["results"][0]["findings"])


def test_delivery_is_recorded_only_when_a_report_was_written(monkeypatch,
                                                             tmp_path):
    """The report IS the delivery channel. With no report there is nothing a
    reader can open, so the finding must stay PENDING."""
    _stub_check(monkeypatch, _PAGE_ONE, TimeoutError("second page"))
    code = rm.main(["--store", str(tmp_path / "f.sqlite3")])
    assert code == 2
    store = FindingStore(tmp_path / "f.sqlite3")
    assert len(store.pending_deliveries()) == 1


def test_a_written_report_marks_the_finding_delivered(monkeypatch, tmp_path):
    _stub_check(monkeypatch, _PAGE_ONE, TimeoutError("second page"))
    report = tmp_path / "r.json"
    rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    assert doc["delivered"] == 1 and doc["pending_deliveries"] == 0


def test_the_heartbeat_is_signalled_after_the_report_exists(monkeypatch,
                                                            tmp_path):
    """MEASURED: a success heartbeat preceded a failing report write, so
    external success could precede the required output."""
    report = tmp_path / "r.json"
    order = []
    from genomic_variant_classifier.source_monitor import heartbeat as hb

    def urlopen(req, timeout=None):
        order.append(("heartbeat", report.exists()))
        return _Resp(200)

    monkeypatch.setattr(hb.urllib.request, "urlopen", urlopen)
    _stub_check(monkeypatch, {"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES})
    rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report),
             "--heartbeat-url", "https://hc.example/tok"])
    # the START signal precedes the report; the OUTCOME signal must not.
    assert order[0] == ("heartbeat", False)
    assert order[-1] == ("heartbeat", True)


def test_a_policy_target_with_no_implementation_is_refused(monkeypatch):
    monkeypatch.setattr(rm, "REQUIRED_TARGETS",
                        rm.REQUIRED_TARGETS + ("never-implemented",))
    with pytest.raises(RuntimeError, match="no implementation"):
        rm.validate_configuration()


def test_a_duplicated_required_target_is_refused(monkeypatch):
    """MEASURED: duplicate required targets were accepted, so the obligation
    set itself was never validated."""
    monkeypatch.setattr(rm, "REQUIRED_TARGETS",
                        ("gnomad-public-releases", "gnomad-public-releases"))
    with pytest.raises(RuntimeError, match="twice"):
        rm.validate_configuration()


def _reasons_emitted_by(module, exclude_assignment=None):
    """Every Reason.X referenced in a module's source.

    DERIVED, not hand-listed. MEASURED 2026-09-15: a hand-enumerated control
    covered the ADAPTER's codes and passed while SIX codes the verifier can
    emit were unpermitted -- five of its ten refusal paths would have crashed
    the reporting path the first time they fired.

    `exclude_assignment` skips a named module-level assignment, so the profile
    DECLARATION does not count as an emission.
    """
    import ast as _ast
    import inspect

    tree = _ast.parse(inspect.getsource(module))
    skip = set()
    if exclude_assignment:
        for node in tree.body:
            if isinstance(node, _ast.Assign) and any(
                    getattr(t, "id", None) == exclude_assignment
                    for t in node.targets):
                skip = set(range(node.lineno, node.end_lineno + 1))
    names = {
        node.attr
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Attribute)
        and isinstance(node.value, _ast.Name)
        and node.value.id == "Reason"
        and node.lineno not in skip
    }
    assert names, "the extractor found no Reason references -- it is broken"
    return {getattr(Reason, n).value for n in names}


@pytest.mark.parametrize("producer", [
    pytest.param("verifier", id="request_verifier"),
    pytest.param("adapter", id="gnomad_release_check"),
    pytest.param("runner", id="run_monitor"),
])
def test_the_profile_permits_every_reason_EACH_PRODUCER_can_emit(producer):
    """THREE producers emit reason codes: the adapter, the verifier and the
    runner. MEASURED 2026-09-15: the control covered the adapter only, then
    the adapter and the verifier. The runner's own codes -- it emits
    RESPONSE_UNEXPECTED_SHAPE for an unhandled check fault -- were never
    checked by anything."""
    from genomic_variant_classifier.source_monitor import request_verifier as rv
    module, exclude = {
        "verifier": (rv, None),
        "adapter": (grc, None),
        "runner": (rm, "RELEASE_PROFILE"),
    }[producer]
    emitted = _reasons_emitted_by(module, exclude)
    missing = sorted(emitted - set(rm.RELEASE_PROFILE.allowed_codes))
    assert missing == [], missing


def test_the_runtime_profile_permits_every_reason_the_VERIFIER_can_emit():
    """MEASURED 2026-09-15: EVIDENCE_CAPTURE_SEQUENCE_INVALID was added to the
    shared catalog and never to allowed_codes. The producer gate refused it on
    the first run after the runner stopped flattening plan findings -- and
    while the flattening was in place the omission was INVISIBLE.

    The enumeration below is derived from the verifier's SOURCE rather than
    hand-listed, so a new plan-finding code cannot be forgotten the way this
    one was."""
    import ast as _ast
    import inspect
    from genomic_variant_classifier.source_monitor import request_verifier as rv

    tree = _ast.parse(inspect.getsource(rv))
    emitted = {
        node.attr
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Attribute)
        and isinstance(node.value, _ast.Name)
        and node.value.id == "Reason"
    }
    assert emitted, "the extractor found no Reason references -- it is broken"
    missing = sorted(
        getattr(Reason, name).value for name in emitted
        if getattr(Reason, name).value not in rm.RELEASE_PROFILE.allowed_codes)
    assert missing == [], missing


def test_the_runtime_profile_permits_every_reason_the_check_can_emit():
    """MEASURED: the check emitted TRAVERSAL_TOKEN_CYCLE while the runner's
    profile forbade it, so a recognized source defect crashed the reporting
    path with ProfileError."""
    emitted = {Reason.TRANSPORT_TIMEOUT, Reason.TRANSPORT_UNREACHABLE,
               Reason.TRAVERSAL_TOKEN_CYCLE, Reason.TRAVERSAL_TOKEN_MALFORMED,
               Reason.TRAVERSAL_BUDGET_EXHAUSTED, Reason.RESPONSE_JSON_SYNTAX,
               Reason.RESPONSE_JSON_DUPLICATE, Reason.RESPONSE_JSON_VALUE,
               Reason.RESPONSE_UNEXPECTED_SHAPE,
               Reason.CONFIG_INVALID_BASELINE}
    missing = sorted(r.value for r in emitted
                     if r.value not in rm.RELEASE_PROFILE.allowed_codes)
    assert missing == []


def test_recognition_uses_the_whole_catalog_not_a_subset():
    """A globally defined code omitted from known_codes would be classified
    UNKNOWN rather than KNOWN-BUT-DISALLOWED -- different causes."""
    assert rm.RELEASE_PROFILE.known_codes == frozenset(r.value for r in Reason)


# --------------------------------------------------------------------------
# 8. Retained captures: what a VERIFIER would check
#
# A report that names a malformed page and retains nothing to check is
# unfalsifiable -- the same shape as a status field nobody verifies. These
# tests cover the retention itself, which was previously verified only by a
# manual probe. A passing manual probe is not a regression test.
# --------------------------------------------------------------------------

def test_the_retained_digest_matches_the_bytes_served():
    page = {"kind": "storage#objects", "prefixes": MEASURED_PREFIXES}
    served = json.dumps(page).encode("utf-8")
    r = grc.observe_releases(transport=_transport(page))
    assert r.captures[0]["response_sha256"] == hashlib.sha256(served).hexdigest()
    assert r.captures[0]["response_bytes"] == len(served)


def test_the_request_is_retained_so_a_missing_token_is_meaningful():
    """MEASURED 2026-09-14: an earlier request carried `fields=prefixes`,
    which OMITS nextPageToken. Its absence therefore meant nothing, and the
    evidence of truncation could not exist. A verifier must be able to see
    what the request ASKED for."""
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    assert "nextPageToken" in r.captures[0]["request_url"]


def test_a_rejected_page_is_captured_not_discarded():
    bad = {"kind": "storage#buckets", "prefixes": ["release/4.1/"]}
    served = json.dumps(bad).encode("utf-8")
    r = grc.observe_releases(transport=_transport(bad))
    assert len(r.captures) == 1
    assert r.captures[0]["accepted"] is False
    assert "storage#buckets" in r.captures[0]["rejected_because"]
    assert r.captures[0]["response_sha256"] == hashlib.sha256(served).hexdigest()


def test_every_page_of_a_multi_page_traversal_is_captured():
    r = grc.observe_releases(
        transport=_transport(_PAGE_ONE,
                             {"kind": "storage#objects",
                              "prefixes": ["release/4.2/"]}))
    assert [c["sequence"] for c in r.captures] == [1, 2]
    assert all(c["accepted"] for c in r.captures)


def test_captures_survive_an_interruption_with_the_witness():
    r = grc.observe_releases(
        transport=_transport(_PAGE_ONE, TimeoutError("second page")))
    assert len(r.captures) == 1          # page one was captured
    assert r.captures[0]["accepted"] is True
    assert any("4.1.1" in f for f in r.findings)


def test_a_transport_failure_before_any_page_captures_nothing():
    """There are no bytes to retain, and the report must not imply otherwise."""
    r = grc.observe_releases(transport=_transport(TimeoutError("first page")))
    assert r.captures == ()
    assert r.reason is Reason.TRANSPORT_TIMEOUT


def test_captures_reach_the_report_document_and_round_trip():
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    doc = supervise(("gnomad-public-releases",), [r]).as_document()
    assert len(doc["results"][0]["captures"]) == 1
    assert json.loads(json.dumps(doc)) == doc


def test_the_default_store_is_anchored_not_repository_relative():
    """MEASURED 2026-09-14: the default was var/monitor/findings.sqlite3. The
    live run wrote it INSIDE the checkout, left the tree dirty, and the next
    installer REFUSED at its working-tree precondition.

    This is LITERATURE-STATE-CWD-RELATIVE-1 in a new subsystem:
    version_monitor_agent.py had Path("data/agent_state.json"), resolved
    against the process working directory, and ONE logical store came to exist
    at TWO depths with divergent contents."""
    path = rm.default_store_path()
    assert path.is_absolute(), path
    assert "source_monitor" in str(path)


def test_the_default_store_is_outside_any_checkout(tmp_path, monkeypatch):
    """Resolving from a different working directory must not move it."""
    monkeypatch.chdir(tmp_path)
    assert rm.default_store_path() == rm.default_store_path()
    assert not str(rm.default_store_path()).startswith(str(tmp_path))


def test_main_runs_with_NO_store_argument(monkeypatch, tmp_path):
    """MEASURED 2026-09-15, the first run of the INSTALLED code:

        TypeError: argument should be a str or an os.PathLike object ...
                   not 'NoneType'

    One site resolved the default and another used args.store directly. EVERY
    one of the seventy tests before this one passed --store EXPLICITLY, so the
    default was covered only in isolation, never through main(). A suite that
    never exercises the default cannot catch a defect in the default."""
    _stub_check(monkeypatch, {"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES})
    monkeypatch.setattr(rm, "default_store_path",
                        lambda: tmp_path / "anchored" / "findings.sqlite3")
    report = tmp_path / "r.json"
    code = rm.main(["--report", str(report)])
    assert code == 1
    doc = json.loads(report.read_text())
    assert doc["store"].endswith("findings.sqlite3")
    assert (tmp_path / "anchored" / "findings.sqlite3").exists()


def test_the_reported_store_is_the_store_actually_used(monkeypatch, tmp_path):
    """The two sites must not diverge again: the path in the report is the
    path the findings were written to."""
    _stub_check(monkeypatch, _PAGE_ONE, TimeoutError("second page"))
    explicit = tmp_path / "chosen.sqlite3"
    report = tmp_path / "r.json"
    rm.main(["--store", str(explicit), "--report", str(report)])
    doc = json.loads(report.read_text())
    assert Path(doc["store"]) == explicit.resolve()
    assert FindingStore(explicit).get_finding is not None


# --------------------------------------------------------------------------
# 9. The independent verifier, and the runner that consults it
# --------------------------------------------------------------------------

from genomic_variant_classifier.source_monitor.request_verifier import (
    APPROVED_QUERY, verify_captures)


def test_a_real_traversal_verifies_against_the_approved_plan():
    r = grc.observe_releases(
        transport=_transport(_PAGE_ONE, {"kind": "storage#objects",
                                         "prefixes": ["release/4.2/"]}))
    assert len(r.captures) == 2
    assert verify_captures(r.captures) == ()


def test_the_verifier_holds_the_field_mask_INDEPENDENTLY():
    """MEASURED 2026-09-14: `fields=prefixes` OMITS nextPageToken, so its
    absence meant nothing. A verifier reading the mask FROM THE ADAPTER would
    approve that request -- it would be comparing the adapter to itself."""
    assert "nextPageToken" in APPROVED_QUERY["fields"]
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    tampered = [dict(r.captures[0],
                     request_url=r.captures[0]["request_url"].replace(
                         "fields=kind%2Cprefixes%2CnextPageToken",
                         "fields=prefixes"))]
    findings = verify_captures(tampered)
    assert [f.reason for f in findings] == [Reason.REQUEST_QUERY_MISMATCH]


@pytest.mark.parametrize("mutation, reason", [
    pytest.param({"request_url": "https://evil/x?prefix=release%2F"},
                 Reason.REQUEST_ENDPOINT, id="wrong_endpoint"),
    pytest.param({"response_sha256": "abc"},
                 Reason.ARTIFACT_DIGEST_MISMATCH, id="truncated_digest"),
    pytest.param({"response_bytes": True},
                 Reason.RESPONSE_UNEXPECTED_SHAPE, id="bytes_as_bool"),
    pytest.param({"response_bytes": 5 * 1024 * 1024},
                 Reason.TRAVERSAL_BUDGET_EXHAUSTED, id="over_budget"),
])
def test_the_verifier_refuses_a_tampered_capture(mutation, reason):
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    findings = verify_captures([dict(r.captures[0], **mutation)])
    assert any(f.reason is reason for f in findings), \
        [f.as_document() for f in findings]


def test_an_empty_capture_list_yields_no_findings_and_proves_nothing():
    """Returning findings rather than a boolean is what stops 'no findings'
    being read as 'verified'."""
    assert verify_captures([]) == ()


def test_the_runner_refuses_a_result_whose_request_did_not_match(monkeypatch,
                                                                 tmp_path):
    """A report carrying evidence nobody verifies is the same unfalsifiable
    shape as `status: "ok"` in VersionMonitorAgent."""
    def run(**_kw):
        r = grc.observe_releases(
            profile=rm.RELEASE_PROFILE,
            transport=_transport({"kind": "storage#objects",
                                  "prefixes": MEASURED_PREFIXES}))
        bad = tuple(dict(c, request_url="https://evil/x?prefix=release%2F")
                    for c in r.captures)
        return TargetResult(r.target, r.health, findings=r.findings,
                            captures=bad)

    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"),
                    "--report", str(report)])
    assert code == 2
    doc = json.loads(report.read_text())
    assert doc["plan_verification"][0]["target"] == "gnomad-public-releases"
    # The result now carries the FIRST finding's OWN reason, not a blanket
    # REQUEST_QUERY_MISMATCH. MEASURED 2026-09-15: the runner previously
    # committed every plan finding under that one code, so a LOST PAGE was
    # recorded in the durable store as a query mismatch and the catalog
    # distinction died one layer above the verifier.
    assert doc["results"][0]["reason"] == Reason.REQUEST_ENDPOINT.value
    # EVERY finding is committed individually, each keeping its own reason.
    assert {a["reason"] for a in doc["assessments"]} == {
        f["reason"] for f in doc["plan_verification"][0]["findings"]}
    assert all(a["recognized"] for a in doc["assessments"])


def test_a_clean_run_records_no_plan_findings(monkeypatch, tmp_path):
    _stub_check(monkeypatch, {"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES})
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"),
                    "--report", str(report)])
    assert code == 1
    doc = json.loads(report.read_text())
    assert doc["plan_verification"] == []
    assert doc["plan_verified_targets"] == ["gnomad-public-releases"]


def test_the_verifier_reports_EVERY_defect_not_the_first():
    """MEASURED 2026-09-15: an earlier version returned on the first failure,
    so a capture with a wrong query, a truncated digest AND an over-budget size
    produced ONE finding. A caller reading a one-element tuple takes it for a
    complete inventory -- the same error as reporting 'lof.obs 150' from an
    assertion that raises on its first failing metric."""
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    bad = dict(r.captures[0],
               request_url=r.captures[0]["request_url"].replace(
                   "fields=kind%2Cprefixes%2CnextPageToken", "fields=prefixes"),
               response_sha256="abc",
               response_bytes=5 * 1024 * 1024)
    reasons = {f.reason for f in verify_captures([bad])}
    assert reasons == {Reason.REQUEST_QUERY_MISMATCH,
                       Reason.ARTIFACT_DIGEST_MISMATCH,
                       Reason.TRAVERSAL_BUDGET_EXHAUSTED}


def test_payload_defects_are_reported_even_when_the_url_is_unparseable():
    """Size and digest do not depend on the request, so a FATAL request defect
    must not suppress them."""
    r = grc.observe_releases(
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    bad = dict(r.captures[0], request_url="http://[", response_sha256="x")
    reasons = {f.reason for f in verify_captures([bad])}
    assert Reason.ARTIFACT_DIGEST_MISMATCH in reasons
    assert len(reasons) >= 2


def test_the_verifier_and_the_adapter_declare_the_SAME_budgets():
    """The verifier declares its plan INDEPENDENTLY -- that is what lets it
    detect an adapter that changed. The cost is that the two declarations can
    drift, and a verifier whose budget silently exceeded the adapter's would
    approve pages the adapter refused.

    This makes divergence a FAILURE rather than a silence, without coupling
    either module to the other."""
    from genomic_variant_classifier.source_monitor import request_verifier as rv
    assert rv.MAX_BYTES_PER_PAGE == grc.MAX_BYTES_PER_PAGE
    assert rv.MAX_TOKEN_CHARS == grc.MAX_TOKEN_CHARS


def test_the_verifier_and_the_adapter_declare_the_SAME_endpoint_and_mask():
    """Same reasoning as the budgets: independent declarations, checked for
    agreement rather than shared."""
    from genomic_variant_classifier.source_monitor import request_verifier as rv
    assert rv.APPROVED_ENDPOINT == grc.ENDPOINT
    assert rv.APPROVED_QUERY == grc.BASE_QUERY


@pytest.mark.parametrize("captures, why", [
    pytest.param([{"sequence": 1}, {"sequence": 3}], "a gap", id="gap_1_3"),
    pytest.param([{"sequence": 1}, {"sequence": 1}], "a repeat", id="repeat_1_1"),
    pytest.param([{"sequence": "one"}], "a string", id="not_an_int"),
    pytest.param([{"sequence": -5}], "negative", id="negative"),
    pytest.param([{"sequence": True}], "a bool", id="bool_is_not_an_int"),
    pytest.param([{}], "absent", id="absent"),
])
def test_an_invalid_capture_sequence_is_refused(captures, why):
    """MEASURED 2026-09-15: the verifier READ `sequence` for labelling and
    never checked it. Captures numbered [1, 3] verified CLEAN -- a page
    retained and then LOST between the adapter and the verifier, with nothing
    saying so.

    A GAP IS NOT TRUNCATION. Truncation is a known stopping point the
    traversal reports; a gap is a silent loss. Reusing TRAVERSAL_TRUNCATED
    would conflate a declared limit with a lost page, and a shared definition
    carries one meaning."""
    findings = verify_captures(captures)
    assert any(f.reason is Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID
               for f in findings), why


def test_a_well_formed_capture_sequence_is_accepted():
    r = grc.observe_releases(
        transport=_transport(_PAGE_ONE, {"kind": "storage#objects",
                                         "prefixes": ["release/4.2/"]}))
    assert [c["sequence"] for c in r.captures] == [1, 2]
    assert verify_captures(r.captures) == ()


def test_the_sequence_code_is_in_the_shared_catalog():
    """A code invented inside the verifier would be unknown to the assessor,
    which classifies an unrecognised reason as a CONTRACT failure rather than
    a source finding."""
    assert Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID.value in {
        r.value for r in Reason}
    assert (Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID.value
            in rm.RELEASE_PROFILE.known_codes)


def test_a_finding_must_name_an_attempt_that_began(tmp_path):
    """MEASURED 2026-09-15: attempt_id was a plain TEXT column with no
    constraint, so a finding could name an attempt that NEVER BEGAN and
    pending_deliveries returned it as though it were provenanced.

    SQLite disables foreign keys BY DEFAULT and the setting is
    PER-CONNECTION, so declaring the constraint without the pragma enforces
    nothing."""
    store = FindingStore(tmp_path / "f.sqlite3")
    with pytest.raises(StoreError, match="no such attempt"):
        store.commit_finding(attempt_id="never-began", subject="s",
                             record=_record())
    assert store.pending_deliveries() == []


def test_the_foreign_key_pragma_is_actually_on(tmp_path):
    """The constraint and the pragma are separate facts. Asserting the schema
    alone would pass while enforcement was off."""
    import sqlite3
    store = FindingStore(tmp_path / "f.sqlite3")
    with store._connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_a_record_with_a_non_string_key_is_refused(tmp_path):
    """MEASURED 2026-09-15: {1: "x"} was ACCEPTED and json.dumps coerced the
    integer key to "1". The record RECOVERED was not the record COMMITTED."""
    store = FindingStore(tmp_path / "f.sqlite3")
    attempt = store.begin_attempt("s")
    with pytest.raises(StoreError, match="string keys"):
        store.commit_finding(attempt_id=attempt, subject="s",
                             record={1: "x"})


def test_a_refused_commit_leaves_no_row(tmp_path):
    """A rolled-back insert must not leave a partial row behind."""
    import sqlite3
    path = tmp_path / "f.sqlite3"
    store = FindingStore(path)
    with pytest.raises(StoreError):
        store.commit_finding(attempt_id="never-began", subject="s",
                             record=_record())
    with sqlite3.connect(str(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0


@pytest.mark.parametrize("endpoint, why", [
    pytest.param("file:///etc/passwd", "urllib OPENS local files, so a "
                 "misconfigured endpoint made the heartbeat READ FROM DISK "
                 "and report a delivery", id="file_scheme"),
    pytest.param("http://hc.example/tok", "the URL is a BEARER CAPABILITY; "
                 "signalling over plaintext leaks it", id="plaintext"),
    pytest.param("https://hc.example/tok?x=1", "the suffix appended AFTER the "
                 "query, producing a URL the operator never wrote",
                 id="query_string"),
    pytest.param("https://hc.example/tok#f", "same, with a fragment",
                 id="fragment"),
    pytest.param("https:///tok", "no host", id="no_host"),
    pytest.param(12345, "letting urllib raise about a string IT built hides "
                 "the real fault", id="not_a_string"),
])
def test_a_malformed_heartbeat_endpoint_is_refused(monkeypatch, endpoint, why):
    """MEASURED 2026-09-15. Every one of these was ACCEPTED."""
    from genomic_variant_classifier.source_monitor import heartbeat as hb
    sent = []
    monkeypatch.setattr(hb.urllib.request, "urlopen",
                        lambda req, timeout=None: (sent.append(req.full_url),
                                                   _Resp(200))[1])
    out = signal_outcome(endpoint, 2)
    assert not out.delivered, why
    assert sent == [], "a refused endpoint must not be contacted"


def test_a_valid_https_endpoint_is_still_signalled(monkeypatch):
    """The refusals above must not have made every endpoint unreachable."""
    from genomic_variant_classifier.source_monitor import heartbeat as hb
    sent = []
    monkeypatch.setattr(hb.urllib.request, "urlopen",
                        lambda req, timeout=None: (sent.append(req.full_url),
                                                   _Resp(200))[1])
    assert signal_outcome("https://hc.example/tok", 2).delivered
    assert sent == ["https://hc.example/tok/fail"]


def test_the_supervisor_refuses_a_duplicated_required_target():
    """MEASURED 2026-09-15: supervise(("a","a"), [one result]) returned EXIT 0
    with no findings. The obligation set silently shrank from two to one while
    reporting success.

    run_monitor.validate_configuration refuses a duplicated policy, but a
    caller that does not go through the runner had no protection. The guard
    belongs where the INVARIANT lives, not only in one of its callers."""
    from genomic_variant_classifier.source_monitor.monitor_supervisor import (
        SupervisorFinding)
    r = supervise(("a", "a"), [TargetResult("a", Health.COMPLETE)])
    assert r.exit_code == 2
    assert any(f.kind is SupervisorFinding.DUPLICATE_REQUIRED
              for f in r.supervisor_findings)


def test_a_policy_without_duplicates_is_unaffected():
    r = supervise(("a", "b"), [TargetResult(t, Health.COMPLETE)
                               for t in ("a", "b")])
    assert r.exit_code == 0 and r.supervisor_findings == ()


def test_the_registered_check_runs_through_the_real_transport_seam():
    """MEASURED 2026-09-15 by a public-surface census: check_gnomad_releases
    was the ONE public name no test referenced. Every test replaces it via
    monkeypatch.setitem(rm.CHECKS, ...), so the REGISTERED function itself
    never ran in the suite.

    It is three lines, and it carries the only binding between the registry
    key and the adapter."""
    r = rm.CHECKS["gnomad-public-releases"](
        transport=_transport({"kind": "storage#objects",
                              "prefixes": MEASURED_PREFIXES}))
    assert r.target == "gnomad-public-releases"
    assert r.health is Health.COMPLETE
    assert any("4.1.1" in f for f in r.findings)


def test_the_registered_check_passes_the_RUNNERS_profile():
    """A second profile here would produce records the runner's own assessor
    judges under DIFFERENT permissions -- recognized by one and refused by the
    other, with nothing naming the mismatch."""
    import ast as _ast
    import inspect

    tree = _ast.parse(inspect.getsource(rm.check_gnomad_releases))
    passed = {
        kw.value.id
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call)
        for kw in node.keywords
        if kw.arg == "profile" and isinstance(kw.value, _ast.Name)
    }
    assert passed == {"RELEASE_PROFILE"}, passed


# --------------------------------------------------------------------------
# 10. Independent recomputation, per the ruling of 2026-09-16
#
# An INDEPENDENT forensic probe found six adversarial cases producing ZERO
# findings from the verifier as it stood: a fabricated digest, an ignored
# `accepted` flag, a syntactically valid but WRONG continuation token, an
# out-of-order capture list (checked as a multiset, not a sequence), and a
# COMPLETE claim with no captures at all. Every test below reproduces one of
# those cases through the REAL transport seam and asserts the fix.
# --------------------------------------------------------------------------

import base64 as _b64

from genomic_variant_classifier.source_monitor.request_verifier import (
    qualify, TraversalCompleteness, _independent_parse_release_version)


def _paged(prefixes, token=None, kind="storage#objects"):
    """A raw page DICT, matching what the file's own `_transport(*pages)`
    fixture expects (it does the json.dumps().encode() itself -- see line
    248). For the byte-level mutation tests below that need actual encoded
    bytes to tamper with, call json.dumps(_paged(...)).encode() explicitly at
    the call site instead of here."""
    d = {"kind": kind, "prefixes": prefixes}
    if token is not None:
        d["nextPageToken"] = token
    return d


def test_a_fabricated_digest_is_caught_by_recomputation():
    """Probe case arbitrary_valid_length_digest: a well-formed fabrication
    passed every check that existed before this repair."""
    r = grc.observe_releases(transport=_transport(_paged(["release/4.1.1/"])))
    tampered = tuple(dict(c, response_sha256="a" * 64) for c in r.captures)
    out = qualify(r.target, tampered)
    assert any(f.reason is Reason.EVIDENCE_INTEGRITY_MISMATCH
              for f in out.findings)


def test_a_consistent_body_and_digest_swap_raises_no_integrity_finding():
    """Matching bytes and digest establish internal consistency, not origin.
    A consistently swapped pair must NOT be flagged as tampered -- and the
    outcome document must still disclaim authenticity regardless."""
    r = grc.observe_releases(transport=_transport(_paged(["release/4.1.1/"])))
    new_body = json.dumps(_paged(["release/9.9.9/"])).encode()
    swapped = tuple(dict(c, response_body_b64=_b64.b64encode(new_body).decode(),
                         response_sha256=hashlib.sha256(new_body).hexdigest())
                    for c in r.captures)
    out = qualify(r.target, swapped)
    assert not any(f.reason is Reason.EVIDENCE_INTEGRITY_MISMATCH
                  for f in out.findings)
    assert "9.9.9" in out.positive_witnesses
    assert any("authenticity" in d for d in out.as_document()["does_not_establish"])


def test_the_producers_accepted_flag_is_not_authoritative():
    """Probe case rejected_flag_not_checked: `accepted` was recorded and
    never read. A valid page marked rejected must still yield its witness,
    with the disagreement flagged separately."""
    r = grc.observe_releases(transport=_transport(_paged(["release/4.1.1/"])))
    lied = tuple(dict(c, accepted=False, rejected_because="fabricated")
                for c in r.captures)
    out = qualify(r.target, lied)
    assert any(f.reason is Reason.EVIDENCE_ACCEPTANCE_DISAGREEMENT
              for f in out.findings)
    assert "4.1.1" in out.positive_witnesses


def test_a_wrong_but_well_formed_continuation_token_breaks_the_chain():
    """Probe case arbitrary_continuation: the previous chain check compared
    SHAPE only ('does page 2 carry a token'), because it had no access to
    what page 1's own body actually declared. It now does."""
    r = grc.observe_releases(transport=_transport(
        _paged(["release/4.1.1/"], token="tok-A"),
        _paged(["release/4.2/"])))
    wrong_url = r.captures[1]["request_url"].replace(
        "pageToken=tok-A", "pageToken=tok-DIFFERENT")
    tampered = (r.captures[0], dict(r.captures[1], request_url=wrong_url))
    out = qualify(r.target, tampered)
    assert any(f.reason is Reason.EVIDENCE_TOKEN_CHAIN_MISMATCH
              for f in out.findings)


def test_a_capture_following_a_terminal_page_breaks_the_chain():
    r = grc.observe_releases(transport=_transport(_paged(["release/4.1.1/"])))
    ghost = dict(r.captures[0], sequence=2,
                request_url=r.captures[0]["request_url"] + "&pageToken=ghost")
    out = qualify(r.target, r.captures + (ghost,))
    assert any(f.reason is Reason.EVIDENCE_TOKEN_CHAIN_MISMATCH
              for f in out.findings)


def test_reversed_captures_fail_the_order_check_not_just_the_set_check():
    """Probe case out_of_order_sequences: `sorted(seen) == range(1, n+1)` is a
    MULTISET check and passed a reversed list once sorted."""
    r = grc.observe_releases(transport=_transport(
        _paged(["release/4.1.1/"], token="tok-A"),
        _paged(["release/4.2/"])))
    out = qualify(r.target, (r.captures[1], r.captures[0]))
    assert any(f.reason is Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID
              for f in out.findings)


def test_a_bool_sequence_is_refused_even_though_True_equals_one():
    """`True == 1` in Python, so a bare list-equality order check would
    accept a bool where an int belongs."""
    out = qualify("x", ({"sequence": True, "request_url": "",
                        "response_sha256": "a" * 64, "response_bytes": 0,
                        "accepted": True},))
    assert any(f.reason is Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID
              for f in out.findings)


def test_removing_the_final_capture_leaves_traversal_incomplete_but_keeps_the_witness():
    r = grc.observe_releases(transport=_transport(
        _paged(["release/4.1.1/"], token="tok-A"),
        _paged(["release/4.2/"])))
    out = qualify(r.target, (r.captures[0],))
    assert out.traversal_completeness is TraversalCompleteness.INCOMPLETE
    assert "4.1.1" in out.positive_witnesses


def test_a_timeout_after_a_witness_keeps_the_witness_and_marks_incomplete():
    r = grc.observe_releases(transport=_transport(
        _paged(["release/4.1.1/"], token="tok-A"), TimeoutError("second page")))
    out = qualify(r.target, r.captures)
    assert "4.1.1" in out.positive_witnesses
    assert out.traversal_completeness is TraversalCompleteness.INCOMPLETE


def test_zero_captures_qualifies_for_neither_claim_whatever_health_claims():
    """Probe case complete_without_captures: exit 0, verified: [].
    Qualification must not be inferred from the producer's claimed health."""
    out = qualify("gnomad-public-releases", ())
    assert out.traversal_completeness is TraversalCompleteness.UNESTABLISHED
    assert not out.eligible_for_existence_claim
    assert not out.eligible_for_absence_claim


def test_duplicate_json_keys_are_refused_by_independent_structural_validation():
    dup = b'{"kind":"storage#objects","kind":"other","prefixes":[]}'
    r = grc.observe_releases(transport=_transport(_paged(["release/4.1.1/"])))
    bad = tuple(dict(c, response_body_b64=_b64.b64encode(dup).decode(),
                    response_sha256=hashlib.sha256(dup).hexdigest())
               for c in r.captures)
    out = qualify(r.target, bad)
    assert any(f.reason is Reason.RESPONSE_JSON_DUPLICATE for f in out.findings)


def test_traversal_completeness_requires_zero_endpoint_or_query_mismatches():
    """A traversal against the WRONG plan is not a completed traversal of the
    approved subject, even if every page it fetched is internally valid."""
    r = grc.observe_releases(transport=_transport(_paged(["release/4.0/"])))
    wrong = tuple(dict(c, request_url=c["request_url"].replace(
        "gcp-public-data--gnomad", "some-other-bucket")) for c in r.captures)
    out = qualify(r.target, wrong)
    assert out.traversal_completeness is not TraversalCompleteness.COMPLETE


@pytest.mark.parametrize("prefix", [
    "release/4.1.1/", "release/4.1/", "release/v4.0/", "release/4.0/",
    "release/", "release/latest/", "release/4.x/", "release/-1/",
    "release/+2/", "release/4 1/", "release/4.10/", "release/4.9/",
    "release//", "release/0/", "release/v/", "release/4.1.1.1/",
])
def test_the_two_independent_release_grammars_agree(prefix):
    """TWO SEPARATE implementations of the same grammar -- not one imported
    into the other -- so they can drift-detect each other. This battery is
    the drift detector."""
    assert grc.parse_version(prefix) == _independent_parse_release_version(prefix)


def test_the_verifier_and_the_adapter_declare_the_SAME_baseline_and_kind():
    from genomic_variant_classifier.source_monitor import request_verifier as rv
    assert rv.APPROVED_BASELINE == grc.APPROVED_BASELINE
    assert rv.EXPECTED_KIND == grc.EXPECTED_KIND


def test_two_different_targets_each_duplicating_are_distinguishable():
    """MEASURED 2026-09-16: two DIFFERENT targets each producing a duplicate
    result yielded two BYTE-IDENTICAL bare strings -- no way to tell two
    targets were affected rather than one target twice, or which was which.
    The discarded result's own content survived nowhere else in the report."""
    from genomic_variant_classifier.source_monitor.monitor_supervisor import (
        SupervisorFinding)
    r = supervise(("target-A", "target-B"), [
        TargetResult("target-A", Health.COMPLETE),
        TargetResult("target-A", Health.FAILED, reason="x"),
        TargetResult("target-B", Health.COMPLETE),
        TargetResult("target-B", Health.INCOMPLETE, reason="y"),
    ])
    dupes = [f for f in r.supervisor_findings
            if f.kind is SupervisorFinding.DUPLICATE_RESULT]
    assert len(dupes) == 2
    details = {f.detail for f in dupes}
    assert len(details) == 2, "the two findings must be distinguishable"
    assert any("target-A" in d for d in details)
    assert any("target-B" in d for d in details)


def test_supervisor_finding_as_document_is_structured():
    from genomic_variant_classifier.source_monitor.monitor_supervisor import (
        SupervisorFinding, SupervisorFindingRecord)
    rec = SupervisorFindingRecord(SupervisorFinding.TARGET_ABSENT, "target 'x'")
    doc = rec.as_document()
    assert doc == {"kind": "supervisor.target_absent", "detail": "target 'x'"}


def test_pending_deliveries_preserves_true_commit_order_within_one_second(tmp_path):
    """MEASURED 2026-09-16: committed_at has SECOND precision; the fallback
    tiebreaker was a random UUID. Twenty findings committed within one
    second came back in an order unrelated to their true commit sequence."""
    store = FindingStore(tmp_path / "f.sqlite3")
    attempt = store.begin_attempt("s")
    ids = [store.commit_finding(attempt_id=attempt, subject="s",
                                record={"i": i}).event_id
          for i in range(20)]
    pending = store.pending_deliveries()
    assert [p["event_id"] for p in pending] == ids


def test_main_calls_qualify_not_the_legacy_verify_captures_shim(monkeypatch, tmp_path):
    """MEASURED 2026-09-16: this call site invoked verify_captures(), the
    plan-conformance-only subset kept for backward compatibility, not
    qualify(). Everything Q2 built -- digest integrity, independent
    structure, the token chain by value -- had never once run through
    main(). This is the regression test that closes the gap: it confirms
    the report the RUNNER produces carries qualify()'s three-axis outcome,
    not just plan-conformance findings."""
    _stub_check(monkeypatch, MEASURED_PREFIXES and
               {"kind": "storage#objects", "prefixes": MEASURED_PREFIXES})
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    assert "qualification" in doc, "the runner must expose qualify()'s output"
    q = doc["qualification"]["gnomad-public-releases"]
    assert q["traversal_completeness"] == "complete"
    assert q["eligible_for_absence_claim"] is True
    assert "4.1.1" in q["positive_witnesses"]
    assert code == 1


def test_a_fabricated_digest_is_caught_END_TO_END_through_main(monkeypatch, tmp_path):
    """The single most important regression test in this unit: the ORIGINAL
    probe case, exercised through main() itself rather than qualify() in
    isolation. The legacy verify_captures() path checked digest FORMAT only
    and would have missed this completely -- confirmed earlier this session
    by running the pre-fix module directly. This confirms the runner, not
    just the library function, now catches it."""
    def run(**_kw):
        r = grc.observe_releases(profile=rm.RELEASE_PROFILE,
                                 transport=_transport(_PAGE_ONE_TERMINAL))
        tampered = tuple(dict(c, response_sha256="a" * 64) for c in r.captures)
        return TargetResult(r.target, r.health, findings=r.findings,
                            captures=tampered)
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    findings = doc["plan_verification"][0]["findings"]
    assert any(f["reason"] == "evidence.integrity_mismatch" for f in findings)
    assert code == 2


def test_a_producers_claimed_finding_that_the_bytes_do_not_support_is_flagged(
        monkeypatch, tmp_path):
    """A producer that retains honest bytes but LIES about what it found in
    them -- claims a witness the retained body does not support -- must not
    pass silently. This is the new cross-check, exercised end to end."""
    def run(**_kw):
        r = grc.observe_releases(profile=rm.RELEASE_PROFILE,
                                 transport=_transport(_PAGE_ONE_TERMINAL))
        return TargetResult(r.target, r.health,
                            findings=("release 9.9.9 is newer than the approved 4.1",),
                            captures=r.captures)
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    reasons = {a["reason"] for a in doc["assessments"]}
    assert "evidence.witness_disagreement" in reasons
    assert code == 2


# ---------------------------------------------------------------------------
# qualification must be ESTABLISHED, not merely absent of findings -- 2026-09-17
#
# MEASURED, from an external ruling's own injected-producer probes: a
# producer reporting Health.COMPLETE with zero captures and zero self-
# reported findings produced exit 0, even though qualify() itself correctly
# returned traversal_completeness="unestablished" and both eligibility
# flags False. Confirmed directly before this fix existed: run_monitor.py
# wired qualify() into the FAILURE path (outcome.findings, witness
# disagreement) but never made its assessment authoritative for what counts
# as a CLEAN result -- RunReport.exit_code derives entirely from the
# producer's own health/findings.
# ---------------------------------------------------------------------------

def test_a_producer_claiming_complete_with_zero_evidence_is_not_qualified(
        monkeypatch, tmp_path):
    """The exact scenario the ruling's probe table names first: COMPLETE
    health, no captures, no findings. Before this fix: exit 0."""
    def run(**_kw):
        return TargetResult("gnomad-public-releases", Health.COMPLETE,
                            findings=(), captures=())
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    assert code == 2
    assert doc["results"][0]["health"] == "incomplete"
    assert doc["results"][0]["reason"] == "evidence.qualification_unestablished"
    assert doc["unqualified"] == ["gnomad-public-releases"]


def test_an_unsupported_claim_with_zero_evidence_is_not_a_review_finding(
        monkeypatch, tmp_path):
    """The ruling's second probe row: COMPLETE, no captures, but the
    producer CLAIMS a newer release anyway. Before this fix: exit 1 --
    treating a claim backed by nothing as a legitimate finding. The claim
    text itself is still preserved for diagnosis; the target is not."""
    def run(**_kw):
        return TargetResult(
            "gnomad-public-releases", Health.COMPLETE,
            findings=("release 9.9.9 is newer than the approved 4.1",),
            captures=())
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    assert code == 2
    assert doc["results"][0]["health"] == "incomplete"
    assert "release 9.9.9" in doc["results"][0]["findings"][0]


def test_a_genuine_witness_under_incomplete_traversal_is_still_an_operational_problem(
        monkeypatch, tmp_path):
    """MEASURED 2026-09-17: a THIRD ruling, reviewing R's own fix, found
    this exact scenario produced results[0].health == "complete" while
    qualification[...].traversal_completeness == "incomplete" --
    SIMULTANEOUSLY, in the same report, for the same target. Confirmed
    directly. "The approved traversal completed" and "a claim is
    supported" are different propositions; this test previously asserted
    the WRONG one satisfied the other, and that expectation is deliberately
    retired here, per the ruling's own explicit instruction. The useful
    requirement -- witness preservation -- remains: the finding is still
    visible in results[0].findings even though the target is correctly
    INCOMPLETE, not COMPLETE."""
    import base64, hashlib as hashlib_
    body = (b'{"kind": "storage#objects", "prefixes": ["release/4.1.1/"], '
            b'"nextPageToken": "tok-2"}')
    capture = {
        "sequence": 1, "attempt_number": 1, "accepted": True,
        "rejected_because": "", "body_retained": True,
        "request_url": ("https://storage.googleapis.com/storage/v1/b/"
                        "gcp-public-data--gnomad/o?prefix=release%2F&"
                        "delimiter=%2F&maxResults=1000&fields=kind%2C"
                        "prefixes%2CnextPageToken"),
        "response_body_b64": base64.b64encode(body).decode(),
        "response_bytes": len(body),
        "response_sha256": hashlib_.sha256(body).hexdigest(),
    }
    def run(**_kw):
        return TargetResult(
            "gnomad-public-releases", Health.COMPLETE,
            findings=("release 4.1.1 is newer than the approved 4.1",),
            captures=(capture,))
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    q = doc["qualification"]["gnomad-public-releases"]
    assert q["traversal_completeness"] == "incomplete"
    assert q["eligible_for_existence_claim"] is True
    assert code == 2
    assert doc["results"][0]["health"] == "incomplete"
    assert doc["unqualified"] == ["gnomad-public-releases"]
    # THE COEXISTENCE REQUIREMENT, PRESERVED: the witness is not erased
    # merely because the observation is operationally incomplete.
    assert doc["results"][0]["findings"] == [
        "release 4.1.1 is newer than the approved 4.1"]


def test_a_genuinely_complete_traversal_with_a_witness_still_exits_one(
        monkeypatch, tmp_path):
    """The counterpart the ruling itself specifies
    (test_complete_observation_with_a_witness_is_not_absence): a
    traversal that GENUINELY reaches a terminal page, carrying a real
    witness, must be unaffected by the fix above. This matches every real
    live run this session -- the actual gnomAD bucket has always returned
    a terminal page with no nextPageToken."""
    import base64, hashlib as hashlib_
    body = b'{"kind": "storage#objects", "prefixes": ["release/4.1.1/"]}'
    capture = {
        "sequence": 1, "attempt_number": 1, "accepted": True,
        "rejected_because": "", "body_retained": True,
        "request_url": ("https://storage.googleapis.com/storage/v1/b/"
                        "gcp-public-data--gnomad/o?prefix=release%2F&"
                        "delimiter=%2F&maxResults=1000&fields=kind%2C"
                        "prefixes%2CnextPageToken"),
        "response_body_b64": base64.b64encode(body).decode(),
        "response_bytes": len(body),
        "response_sha256": hashlib_.sha256(body).hexdigest(),
    }
    def run(**_kw):
        return TargetResult(
            "gnomad-public-releases", Health.COMPLETE,
            findings=("release 4.1.1 is newer than the approved 4.1",),
            captures=(capture,))
    monkeypatch.setitem(rm.CHECKS, "gnomad-public-releases", run)
    report = tmp_path / "r.json"
    code = rm.main(["--store", str(tmp_path / "f.sqlite3"), "--report", str(report)])
    doc = json.loads(report.read_text())
    q = doc["qualification"]["gnomad-public-releases"]
    assert q["traversal_completeness"] == "complete"
    assert code == 1
    assert doc["results"][0]["health"] == "complete"
    assert doc["unqualified"] == []
