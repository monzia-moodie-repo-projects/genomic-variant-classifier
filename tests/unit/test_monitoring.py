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
    assert out.attempted and not out.delivered and "unreachable" in out.detail


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
