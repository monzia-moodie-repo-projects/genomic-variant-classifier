"""Publish one source-monitor verification to the standing alert issue -- C2, the ONE issue writer (owner ruling 2026-09-29).

The publish job of source_monitor_verify.yml runs this with issues: write. It consumes the verify job's receipt (a job
output of the SAME run) and never the producer's report. Everything the receipt is bound against is obtained
INDEPENDENTLY: the subject from GitHub's record of the source attempt; the checker's commit (the workflow commit, which
both jobs check out and verify), its code manifest and its effective policy recomputed from this job's own checkout; the
evaluation run and attempt from this run's identity. When no checker receipt arrived, the COORDINATOR issues a distinct
"verification unavailable" receipt from authenticated metadata -- never six invented flags, never a guessed run.

A verifier run dispatched by hand is a PREVIEW: no destination lookup, no history, no POST. At most ONE POST otherwise
(c2_protocol.deliver); the per-attempt outcome record (did this attempt issue a POST, for which delivery) is written
whatever happens, so a later attempt can reconstruct prior dispatch. Exit 0: acknowledged / preview / no_op / archive;
exit 1: blocked / unknown -- a VISIBLE failure, never a blind retry.

Importable functions never read GITHUB_*; only __main__ reads the environment (the protection built after CI run #896).

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from genomic_variant_classifier.source_monitor import c2_github as gh  # noqa: E402
from genomic_variant_classifier.source_monitor import c2_protocol as c2  # noqa: E402
from genomic_variant_classifier.source_monitor import report_verifier as rv  # noqa: E402

#: The standing alert issue, pinned by stable identity (owner ruling 2026-09-29; measured 2026-09-29/30). The author id is
#: the identity whose acknowledgements are trusted -- REVALIDATE at activation.
PIN = gh.Pinned(repository=rv.EXPECTED_REPOSITORY, repository_id=1151261021, issue_id=5600463137, number=27,
                author_id=41898282)
SUCCESS = {"acknowledged", "preview", "no_op", "archive"}


def run_name(source_run_id: int, source_attempt: int) -> str:
    """The run-name index of this workflow (source_monitor_verify.yml's run-name must render EXACTLY this)."""
    return "verify {}/{}".format(source_run_id, source_attempt)


class CountingChannel:
    """Wraps a Channel and counts create_once() calls -- the outcome record's post_issued."""

    def __init__(self, channel):
        self.channel, self.posts = channel, 0

    def page(self, cursor):
        return self.channel.page(cursor)

    def create_once(self, body):
        self.posts += 1
        return self.channel.create_once(body)


def expected_checker(workflow_sha: str) -> dict:
    """The checker identity recomputed from THIS checkout (the same pinned workflow commit as the verify job)."""
    from genomic_variant_classifier.source_monitor.run_monitor import REQUIRED_TARGETS
    manifest = [[path, sha] for path, sha in sorted(rv.code_manifest(_ROOT).items())]
    policy = rv.effective_policy(rv.current_reconstruction(_ROOT), REQUIRED_TARGETS)
    return {"commit": workflow_sha, "code_manifest_sha256": c2.digest("gvc.checker-code/v1", manifest),
            "policy_sha256": c2.digest("gvc.verification-policy/v1", policy)}


def coordinator_payload(*, subject, checker, evaluation_run_id, evaluation_attempt, when, diagnostics):
    stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"schema": "gvc.monitor-receipt", "schema_version": 1, "issuer_role": "coordinator", "subject": subject,
            "evidence": {"state": "unavailable", "artifact_id": None, "archive_sha256": None, "report_sha256": None},
            "checker": checker, "evaluation": {"run_id": evaluation_run_id, "attempt": evaluation_attempt,
                                               "started_at": stamp, "finished_at": stamp},
            "decision": {"status": "unavailable", "verified": None, "flags": None, "reviews": [],
                         "reasons": [{"code": "checker.unavailable", "target": ""}]},
            "diagnostics": diagnostics}


def main(argv=None, *, request, receipt_b64: str, now=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event", required=True, choices=["workflow_run", "workflow_dispatch"])
    for name in ("--source-run-id", "--source-attempt", "--current-run-id", "--current-attempt"):
        parser.add_argument(name, type=int, required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--outcome", required=True, help="where to write this attempt's outcome record")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.workflow_sha) or not all(
            c2.positive(v) for v in (args.source_run_id, args.source_attempt, args.current_run_id, args.current_attempt)):
        parser.error("identifiers must be positive integers and a full 40-character workflow commit")
    now = now or datetime.now(timezone.utc)
    automatic = args.event == "workflow_run"
    base = "{}/repos/{}".format(gh.API_ROOT, PIN.repository)
    channel, result, delivery = None, c2.Result("unknown", "delivery_did_not_return"), ""
    try:
        # Independent bindings.
        attempt = gh._json(*request("GET", base + "/actions/runs/{}/attempts/{}".format(
            args.source_run_id, args.source_attempt))[::2])
        subject = gh.subject_from_attempt(attempt)
        checker = expected_checker(args.workflow_sha)
        raw = base64.b64decode(receipt_b64, validate=True) if receipt_b64.strip() else b""
        if raw:
            payload = c2.open_receipt(raw, c2.Bindings(subject, checker, args.current_run_id, args.current_attempt), now)
        else:
            payload = coordinator_payload(subject=subject, checker=checker, evaluation_run_id=args.current_run_id,
                                          evaluation_attempt=args.current_attempt, when=now,
                                          diagnostics=["no checker receipt arrived from the verify job"])
            payload = c2.open_receipt(c2.seal(payload), c2.Bindings(subject, checker, args.current_run_id,
                                                                    args.current_attempt, "coordinator"), now)
        print("receipt: issuer {} event kind {}".format(payload["issuer_role"], c2.event_kind(payload)))
        if not automatic:
            result = c2.deliver(None, payload, None, author_id=PIN.author_id, history=None, now=now, automatic=False)
        else:
            destination = gh.select_destination(request, PIN)
            delivery = c2.delivery_id(payload, destination)
            current = gh._json(*request("GET", base + "/actions/runs/{}".format(args.current_run_id))[::2])
            history = gh.dispatch_history(request, repository=PIN.repository, repository_id=PIN.repository_id,
                                          workflow_id=current.get("workflow_id"),
                                          run_name=run_name(args.source_run_id, args.source_attempt),
                                          current_run_id=args.current_run_id, current_attempt=args.current_attempt,
                                          delivery_id=delivery)
            print("history: {} -- {}".format(history.state.value, history.evidence_ref))
            channel = CountingChannel(gh.CommentChannel(request, PIN))
            result = c2.deliver(channel, payload, destination, author_id=PIN.author_id, history=history, now=now)
    finally:
        # EVERY path writes this attempt's outcome record -- a preview, an early refusal, an exception. A started delivery
        # step WITHOUT a record reads as UNKNOWN to every later attempt (c2_github.dispatch_history), so an unwritten record
        # after a harmless preview would block that source run's delivery forever (found designing the tests, 2026-09-30).
        record = gh.outcome_record(delivery_id=delivery, post_issued=channel is not None and channel.posts > 0, result=result)
        Path(args.outcome).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="ascii")
    print("RESULT {} {} comment {} age {}".format(result.action, result.reason, result.comment_id, result.age_seconds))
    return 0 if result.action in SUCCESS else 1


if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN") or ""
    sys.exit(main(request=gh.Transport(token), receipt_b64=os.environ.get("RECEIPT_B64", "")))
