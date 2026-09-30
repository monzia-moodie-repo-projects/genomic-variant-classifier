"""The C2 delivery protocol (owner ruling 2026-09-29; integrated 2026-09-30).

Ported from the owner's reviewed reference tests (tests/test_protocol.py, 62 tests; every refusal PINS its exact reason code), with
ONE deliberate change -- test_future_clock's reason, per the adopted policy -- and EXACT timing-boundary tests added for the adopted
policy (-60 s and +900 s accepted, -61 s and +901 s refused with distinct reasons, the observed age recorded).

Author: Monzia Moodie
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import unittest

from genomic_variant_classifier.source_monitor import c2_protocol as c

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
DEST = c.Destination(10, 200, 27)
BOT = 100
NEW = c.DispatchHistory(c.History.NO_PRIOR_DISPATCH, "authenticated-run-history:test")
PRIOR = c.DispatchHistory(c.History.PRIOR_DISPATCH, "previous-writer:test")


def fixture():
    return {
        "schema": "gvc.monitor-receipt", "schema_version": 1, "issuer_role": "checker",
        "subject": {"repository": "owner/project", "repository_id": 10,
                    "workflow_id": 20, "workflow_path": ".github/workflows/source_monitor.yml",
                    "run_id": 123, "run_number": 9, "attempt": 1, "commit": "a" * 40},
        "evidence": {"state": "complete", "artifact_id": 30,
                     "archive_sha256": "b" * 64, "report_sha256": "c" * 64},
        "checker": {"commit": "d" * 40, "code_manifest_sha256": "e" * 64,
                    "policy_sha256": "f" * 64},
        "evaluation": {"run_id": 456, "attempt": 1,
                       "started_at": "2026-09-29T11:58:00Z",
                       "finished_at": "2026-09-29T11:59:00Z"},
        "decision": {"status": "completed", "verified": True,
                     "flags": {name: True for name in c.FLAGS},
                     "reviews": [{"target": "gnomad-public-releases", "kind": "newer",
                                  "raw_prefix": "release/4.1.2/"}], "reasons": []},
        "diagnostics": [],
    }


def bindings(p):
    return c.Bindings(deepcopy(p["subject"]), deepcopy(p["checker"]),
                      p["evaluation"]["run_id"], p["evaluation"]["attempt"])


class FakeChannel:
    def __init__(self, rows=(), fault=None, page_size=2):
        self.rows = list(rows)
        self.fault = fault
        self.page_size = page_size
        self.posts = 0
        self.reads = 0
        self.delayed = None

    def page(self, cursor):
        self.reads += 1
        if self.fault == "read":
            raise OSError("GET failed")
        i = 0 if cursor is None else int(cursor)
        end = i + self.page_size
        return c.Page(tuple(self.rows[i:end]), str(end) if end < len(self.rows) else None)

    def create_once(self, body):
        self.posts += 1
        row = c.Comment(max([x.id for x in self.rows] + [0]) + 1, DEST.issue_id, BOT, body)
        if self.fault == "drop_before_commit":
            raise TimeoutError("not committed in this test; caller cannot know")
        if self.fault == "delayed_visibility":
            self.delayed = row
            raise TimeoutError("may commit later")
        self.rows.append(row)
        if self.fault == "drop_after_commit":
            raise TimeoutError("response lost after commit")
        if self.fault == "bad_response":
            return c.Comment(row.id, 999, BOT, body)
        return row

    def reveal(self):
        if self.delayed:
            self.rows.append(self.delayed)
            self.delayed = None


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = fixture()

    def refusal(self, code, fn, *args, **kwargs):
        with self.assertRaises(c.Refusal) as raised:
            fn(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)

    def send(self, channel, history=NEW, **kwargs):
        return c.deliver(channel, self.p, DEST, author_id=BOT, history=history,
                         now=kwargs.pop("now", NOW), **kwargs)

    def test_roundtrip(self):
        self.assertEqual(c.open_receipt(c.seal(self.p), bindings(self.p), NOW), self.p)

    def test_boolean_version(self):
        self.p["schema_version"] = True
        self.refusal("receipt.version", c.seal, self.p)

    def test_boolean_run_id(self):
        self.p["subject"]["run_id"] = True
        self.refusal("subject.integer", c.seal, self.p)

    def test_integer_flag(self):
        self.p["decision"]["flags"]["review_required"] = 1
        self.refusal("flags.type", c.seal, self.p)

    def test_duplicate_json_key(self):
        self.refusal("json.duplicate_key", c.strict_load, b'{"x":1,"x":2}')

    def test_nan(self):
        self.refusal("json.non_integer_number", c.strict_load, b'{"x":NaN}')

    def test_float(self):
        self.refusal("json.non_integer_number", c.strict_load, b'{"x":1.0}')

    def test_bom(self):
        self.refusal("json.bom", c.strict_load, b"\xef\xbb\xbf{}")

    def test_unknown_fields(self):
        self.p["extra"] = 1
        self.refusal("receipt.fields", c.seal, self.p)

    def test_oversize(self):
        self.refusal("receipt.size", c.strict_load, b" " * (c.LIMIT + 1))

    def test_checksum_changed(self):
        d = json.loads(c.seal(self.p))
        d["payload"]["diagnostics"] = ["changed"]
        self.refusal("receipt.checksum", c.open_receipt, c.canonical(d), bindings(self.p), NOW)

    def test_recomputed_checksum_cannot_change_subject_binding(self):
        expected = bindings(self.p)
        self.p["subject"]["attempt"] = 2
        self.refusal("binding.subject", c.open_receipt, c.seal(self.p), expected, NOW)

    def test_recomputed_checksum_cannot_change_checker(self):
        expected = bindings(self.p)
        self.p["checker"]["commit"] = "0" * 40
        self.refusal("binding.checker", c.open_receipt, c.seal(self.p), expected, NOW)

    def test_other_evaluation_run(self):
        expected = bindings(self.p)
        self.p["evaluation"]["run_id"] += 1
        self.refusal("binding.evaluation", c.open_receipt, c.seal(self.p), expected, NOW)

    def test_future_clock(self):
        # Owner policy 2026-09-30: the reference's "time.future" (5 min) became "receipt.future_timestamp" (60 s).
        self.refusal("receipt.future_timestamp", c.open_receipt, c.seal(self.p), bindings(self.p),
                     NOW - timedelta(hours=1))

    def test_reverse_evaluation_times(self):
        self.p["evaluation"]["started_at"] = "2026-09-29T12:00:00Z"
        self.refusal("time.reversed", c.seal, self.p)

    def test_verified_derived_from_four_flags(self):
        self.p["decision"]["flags"]["observation_complete"] = False
        self.refusal("flags.verified", c.seal, self.p)

    def test_unbound_review_refused(self):
        self.p["decision"]["flags"]["execution_authenticated"] = False
        self.p["decision"]["verified"] = False
        self.refusal("review.unbound", c.seal, self.p)

    def test_current_cannot_be_true_when_not_verified(self):
        self.p["decision"]["flags"]["observation_complete"] = False
        self.p["decision"]["verified"] = False
        self.refusal("flags.current", c.seal, self.p)

    def test_unavailable_is_not_all_false(self):
        self.p["decision"] = {"status": "unavailable", "verified": None, "flags": None,
                              "reviews": [], "reasons": [{"code": "checker.unavailable", "target": ""}]}
        self.assertEqual(c.event_kind(c.open_receipt(c.seal(self.p), bindings(self.p), NOW)),
                         "verification_unavailable")

    def test_unavailable_must_have_reason(self):
        self.p["decision"] = {"status": "unavailable", "verified": None, "flags": None,
                              "reviews": [], "reasons": []}
        self.refusal("decision.unavailable_reason", c.seal, self.p)

    def test_coordinator_cannot_claim_a_verification(self):
        self.p["issuer_role"] = "coordinator"
        self.refusal("decision.coordinator_cannot_verify", c.seal, self.p)

    def test_coordinator_fallback_is_distinct_and_bound(self):
        self.p["issuer_role"] = "coordinator"
        self.p["decision"] = {"status": "unavailable", "verified": None, "flags": None,
                              "reviews": [], "reasons": [{"code": "checker.unavailable", "target": ""}]}
        expected = c.Bindings(deepcopy(self.p["subject"]), deepcopy(self.p["checker"]),
                             456, 1, "coordinator")
        self.assertEqual(c.open_receipt(c.seal(self.p), expected, NOW), self.p)
        self.refusal("binding.issuer", c.open_receipt, c.seal(self.p), bindings(self.p), NOW)

    def test_receipt_changes_but_decision_does_not(self):
        old_receipt, old_decision = c.seal(self.p), c.decision_id(self.p)
        self.p["evaluation"]["finished_at"] = "2026-09-29T12:00:00Z"
        self.p["evaluation"]["run_id"] += 1
        self.p["diagnostics"] = ["age is now two minutes"]
        self.assertNotEqual(c.seal(self.p), old_receipt)
        self.assertEqual(c.decision_id(self.p), old_decision)

    def test_report_digest_is_in_decision(self):
        key = c.decision_id(self.p)
        self.p["evidence"]["report_sha256"] = "0" * 64
        self.assertNotEqual(key, c.decision_id(self.p))

    def test_archive_digest_is_distinct_and_bound(self):
        key = c.decision_id(self.p)
        self.p["evidence"]["archive_sha256"] = "1" * 64
        self.assertNotEqual(key, c.decision_id(self.p))

    def test_source_attempt_changes_decision(self):
        key = c.decision_id(self.p)
        self.p["subject"]["attempt"] += 1
        self.assertNotEqual(key, c.decision_id(self.p))

    def test_policy_changes_decision(self):
        key = c.decision_id(self.p)
        self.p["checker"]["policy_sha256"] = "0" * 64
        self.assertNotEqual(key, c.decision_id(self.p))

    def test_reason_code_changes_decision(self):
        key = c.decision_id(self.p)
        self.p["decision"]["reasons"] = [{"code": "claims.disagree", "target": "x"}]
        self.assertNotEqual(key, c.decision_id(self.p))

    def test_review_order_irrelevant_multiplicity_significant(self):
        r = self.p["decision"]["reviews"]
        r.append({"target": "x", "kind": "unsupported", "raw_prefix": "release/latest/"})
        key = c.decision_id(self.p)
        r.reverse()
        self.assertEqual(key, c.decision_id(self.p))
        r.append(deepcopy(r[0]))
        self.assertNotEqual(key, c.decision_id(self.p))

    def test_destination_missing(self):
        self.refusal("destination.missing", c.select_destination, [], DEST)

    def test_destination_ambiguous(self):
        self.refusal("destination.ambiguous", c.select_destination, [{}, {}], DEST)

    def test_destination_swapped(self):
        self.refusal("destination.changed", c.select_destination,
                     [{"repository_id": 10, "id": 201, "number": 27}], DEST)

    def test_destination_matches(self):
        self.assertEqual(c.select_destination(
            [{"repository_id": 10, "id": 200, "number": 27}], DEST), DEST)

    def test_first_post_and_repeat(self):
        channel = FakeChannel()
        self.assertEqual(self.send(channel).reason, "created")
        self.assertEqual(self.send(channel, PRIOR).reason, "matching_comment")
        self.assertEqual(channel.posts, 1)

    def test_dedup_later_receipt_with_same_decision(self):
        channel = FakeChannel()
        self.send(channel)
        self.p["evaluation"]["finished_at"] = "2026-09-29T12:00:00Z"
        self.p["diagnostics"] = ["changed prose"]
        self.assertEqual(self.send(channel, PRIOR).action, "acknowledged")
        self.assertEqual(channel.posts, 1)

    def test_lost_response_after_commit_reconciles(self):
        channel = FakeChannel(fault="drop_after_commit")
        self.assertEqual(self.send(channel).reason, "reconciled_after_post_error")
        self.assertEqual(channel.posts, 1)

    def test_lost_response_before_commit_stays_unknown(self):
        channel = FakeChannel(fault="drop_before_commit")
        self.assertEqual(self.send(channel).reason, "post_outcome_unknown")
        self.assertEqual(self.send(channel, PRIOR).reason, "prior_dispatch_unresolved")
        self.assertEqual(channel.posts, 1)

    def test_delayed_visibility_across_restart(self):
        channel = FakeChannel(fault="delayed_visibility")
        self.assertEqual(self.send(channel).action, "unknown")
        self.assertEqual(self.send(channel, PRIOR).action, "unknown")
        channel.reveal()
        self.assertEqual(self.send(channel, PRIOR).action, "acknowledged")
        self.assertEqual(channel.posts, 1)

    def test_unknown_history_never_posts(self):
        channel = FakeChannel()
        hist = c.DispatchHistory(c.History.UNKNOWN, "history-unavailable:test")
        self.assertEqual(self.send(channel, hist).action, "unknown")
        self.assertEqual(channel.posts, 0)

    def test_deleted_ack_does_not_authorize_repost(self):
        channel = FakeChannel()
        self.send(channel)
        channel.rows.clear()
        self.assertEqual(self.send(channel, PRIOR).action, "unknown")
        self.assertEqual(channel.posts, 1)

    def test_wrong_create_response_reconciles_actual_comment(self):
        channel = FakeChannel(fault="bad_response")
        self.assertEqual(self.send(channel).reason, "reconciled_after_post_error")
        self.assertEqual(channel.posts, 1)

    def test_later_page_ack(self):
        channel = FakeChannel([c.Comment(1, 200, 999, "unrelated"),
                               c.Comment(2, 200, 999, "unrelated"),
                               c.Comment(3, 200, BOT, c.render_comment(self.p, DEST))])
        self.assertEqual(self.send(channel).comment_id, 3)
        self.assertEqual(channel.reads, 2)
        self.assertEqual(channel.posts, 0)

    def test_forged_marker_cannot_suppress_delivery(self):
        channel = FakeChannel([c.Comment(1, 200, 999, c.render_comment(self.p, DEST))])
        self.assertEqual(self.send(channel).reason, "created")

    def test_modified_bot_ack_blocks(self):
        channel = FakeChannel([c.Comment(1, 200, BOT, c.render_comment(self.p, DEST) + "edited")])
        self.assertEqual(self.send(channel).reason, "comments.modified_ack")
        self.assertEqual(channel.posts, 0)

    def test_two_existing_acknowledgements_block(self):
        body = c.render_comment(self.p, DEST)
        channel = FakeChannel([c.Comment(1, 200, BOT, body), c.Comment(2, 200, BOT, body)])
        self.assertEqual(self.send(channel).reason, "comments.duplicate_ack")
        self.assertEqual(channel.posts, 0)

    def test_read_failure_is_not_absence(self):
        channel = FakeChannel(fault="read")
        self.assertEqual(self.send(channel).reason, "comments.read_failed")
        self.assertEqual(channel.posts, 0)

    def test_legacy_run_link_not_an_ack(self):
        channel = FakeChannel([c.Comment(1, 200, BOT, "https://github.com/owner/project/actions/runs/123")])
        self.assertEqual(self.send(channel).reason, "created")

    def test_simulation_cannot_post(self):
        channel = FakeChannel()
        self.assertEqual(self.send(channel, simulation=True).reason, "simulation")
        self.assertEqual((channel.posts, channel.reads), (0, 0))

    def test_manual_cannot_post(self):
        channel = FakeChannel()
        self.assertEqual(self.send(channel, automatic=False).reason, "manual_verification")
        self.assertEqual(channel.posts, 0)

    def test_cancelled_no_mutation(self):
        channel = FakeChannel()
        self.assertEqual(self.send(channel, cancelled=True).reason, "cancelled")
        self.assertEqual(channel.posts, 0)

    def test_partial_observation_keeps_real_review(self):
        self.p["decision"]["flags"]["observation_complete"] = False
        self.p["decision"]["flags"][c.FLAGS[-1]] = False
        self.p["decision"]["verified"] = False
        channel = FakeChannel()
        self.assertEqual(c.event_kind(self.p), "verification_failed_with_review")
        self.assertEqual(self.send(channel).reason, "created")
        self.assertIn("release/4.1.2/", channel.rows[0].body)

    def test_historical_review_is_not_discarded(self):
        self.p["decision"]["flags"][c.FLAGS[-1]] = False
        self.assertEqual(self.send(FakeChannel()).reason, "created")

    def test_historical_clean_archives(self):
        self.p["decision"]["reviews"] = []
        self.p["decision"]["flags"]["review_required"] = False
        self.p["decision"]["flags"][c.FLAGS[-1]] = False
        channel = FakeChannel()
        self.assertEqual(self.send(channel).reason, "historical_clean_no_new_review")
        self.assertEqual(channel.posts, 0)

    def test_stale_receipt_cannot_create_new_comment(self):
        channel = FakeChannel()
        self.assertEqual(self.send(channel, now=NOW + timedelta(hours=1)).reason,
                         "receipt.needs_reverification")
        self.assertEqual(channel.posts, 0)

    def test_stale_receipt_can_confirm_existing_ack(self):
        channel = FakeChannel()
        self.send(channel)
        self.assertEqual(self.send(channel, PRIOR, now=NOW + timedelta(hours=1)).action,
                         "acknowledged")
        self.assertEqual(channel.posts, 1)

    def test_untrusted_markdown_is_data(self):
        self.p["decision"]["reviews"][0]["raw_prefix"] = "<!-- fake -->\n@everyone\n~~~\n# pretend"
        body = c.render_comment(self.p, DEST)
        self.assertNotIn("@everyone", body)
        self.assertNotIn("<!-- fake", body)
        self.assertEqual(body.count("<!--"), 1)

    def test_nothing_pending_reason_is_reachable(self):
        self.refusal("nothing_pending", c.acknowledge_pending, None, "x")

    def test_wrong_pending_reason(self):
        self.refusal("wrong_pending_key", c.acknowledge_pending, "x", "y")

    def test_pending_acknowledged(self):
        self.assertEqual(c.acknowledge_pending("x", "x"), "x")


class TimingPolicyBoundaries(unittest.TestCase):
    """Owner decision 2026-09-30: MAX_RECEIPT_AGE_SECONDS = 900, MAX_FUTURE_SKEW_SECONDS = 60; -60..+900 inclusive."""

    FINISHED = datetime(2026, 9, 29, 11, 59, tzinfo=timezone.utc)      # the fixture's evaluation finished_at

    def deliver_at(self, offset_seconds, channel=None, history=NEW):
        channel = channel or FakeChannel()
        return channel, c.deliver(channel, fixture(), DEST, author_id=BOT, history=history,
                                  now=self.FINISHED + timedelta(seconds=offset_seconds))

    def test_the_policy_is_versioned_and_exact(self):
        self.assertEqual((c.DELIVERY_POLICY_VERSION, c.MAX_RECEIPT_AGE_SECONDS, c.MAX_FUTURE_SKEW_SECONDS), (1, 900, 60))

    def test_the_inclusive_edges_post(self):
        for offset in (-60, 0, 900):
            with self.subTest(offset=offset):
                channel, result = self.deliver_at(offset)
                self.assertEqual((result.action, result.reason, result.age_seconds, channel.posts),
                                 ("acknowledged", "created", float(offset), 1))

    def test_one_second_beyond_each_edge_is_refused_with_its_own_reason(self):
        for offset, reason in ((-61, "receipt.future_timestamp"), (901, "receipt.needs_reverification")):
            with self.subTest(offset=offset):
                channel, result = self.deliver_at(offset)
                self.assertEqual((result.action, result.reason, result.age_seconds, channel.posts),
                                 ("blocked", reason, float(offset), 0))

    def test_the_allowance_does_not_extend_the_stale_limit(self):
        channel, result = self.deliver_at(960)
        self.assertEqual((result.reason, channel.posts), ("receipt.needs_reverification", 0))

    def test_admission_allows_60_seconds_ahead_and_refuses_61(self):
        p = fixture()
        self.assertEqual(c.open_receipt(c.seal(p), bindings(p), self.FINISHED - timedelta(seconds=60)), p)
        with self.assertRaises(c.Refusal) as raised:
            c.open_receipt(c.seal(p), bindings(p), self.FINISHED - timedelta(seconds=61))
        self.assertEqual(raised.exception.code, "receipt.future_timestamp")

    def test_an_expired_receipt_still_confirms_an_existing_acknowledgement(self):
        channel, first = self.deliver_at(0)
        _, later = self.deliver_at(901, channel=channel, history=PRIOR)
        self.assertEqual((first.reason, later.action, later.reason, channel.posts), ("created", "acknowledged", "matching_comment", 1))
