"""Verify one source-monitor run attempt from GitHub's records and its report archive (change C1, 2026-09-27).

Writes a verdict, a step summary and -- with --receipt -- the C2 checker RECEIPT (owner ruling 2026-09-29), BEFORE
returning its exit code, so an ordinary verification failure still produces one. It writes NO issue. Trusted code only -- the report, its archive and
the files at the run's commit are data (report_verifier). Exit 0: verified (review items allowed -- a valid exit-1
monitor run is a SUCCESSFUL verification); exit 2: not verified, or verification could not be completed.

SECURITY (measured 2026-09-27 on this interpreter): urllib FORWARDS a header added with add_header to a redirected
host, and does NOT forward one added with add_unredirected_header. GitHub answers an artifact download with a
redirect to a storage host, so the token is attached UNREDIRECTED -- it never leaves api.github.com.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from genomic_variant_classifier.data import release_approval as ra  # noqa: E402
from genomic_variant_classifier.source_monitor import report_verifier as rv  # noqa: E402

MAX_JSON_BYTES = 4 * 1024 * 1024


def http_fetch(url, token, limit):
    """GET over HTTPS with a hard byte cap; the token is attached UNREDIRECTED."""
    if not url.startswith("https://"):
        raise ValueError("refusing a non-HTTPS URL: {}".format(url))
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "X-GitHub-Api-Version": "2022-11-28"})
    if token:
        req.add_unredirected_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if not resp.geturl().startswith("https://"):
            raise ValueError("redirected to a non-HTTPS URL")
        body = resp.read(limit + 1)
    if len(body) > limit:
        raise ValueError("response exceeds {} bytes: {}".format(limit, url))
    return body


def api_base(deployment):
    """The GitHub API base of the DEPLOYMENT's repository (from the trusted checkout's configuration)."""
    return "https://api.github.com/repos/" + deployment.repository


def _get_json(fetch, api, path):
    return rv.strict_json(fetch(api + path, MAX_JSON_BYTES), MAX_JSON_BYTES)


def collect_attempt(run_id, run_attempt, fetch, api):
    """GitHub's run and ATTEMPT records -- fetched FIRST and kept, so a later evidence failure still has an
    authenticated subject for a checker-issued "unavailable" receipt (C2, 2026-09-30)."""
    return (_get_json(fetch, api, "/actions/runs/{}".format(run_id)),
            _get_json(fetch, api, "/actions/runs/{}/attempts/{}".format(run_id, run_attempt)))


def collect_evidence(run_id, fetch, api):
    artifacts = _get_json(fetch, api, "/actions/runs/{}/artifacts?per_page=100".format(run_id))
    named = [a for a in artifacts.get("artifacts", []) if isinstance(a, dict) and a.get("name") == rv.ARTIFACT_NAME]
    archive = fetch(api + "/actions/artifacts/{}/zip".format(named[0]["id"]), rv.MAX_ARCHIVE_BYTES) if len(named) == 1 else b""
    return artifacts, archive


def summary(doc):
    lines = ["## Source-monitor run verification (the checker; the publish job delivers)", "",
             "Run {} attempt {}: **{}**".format(doc["run_id"], doc["run_attempt"], "VERIFIED" if doc["verified"] else "NOT VERIFIED"), "",
             "| Result | Value |", "|---|---|"]
    lines += ["| `{}` | {} |".format(k, v) for k, v in doc["flags"].items()]
    if doc["review_items"]:
        lines += ["", "### Review items"] + ["- {}".format(i) for i in doc["review_items"]]
    problems = [(k, p) for k, ps in doc["problems"].items() for p in ps]
    if problems:
        lines += ["", "### Problems"] + ["- `{}`: {}".format(k, p) for k, p in problems]
    return "\n".join(lines) + "\n"


def _stamp(moment):
    """The protocol's canonical second-precision UTC timestamp."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def receipt_subject(attempt):
    """The receipt subject from GitHub's OWN attempt record (one definition: c2_github.subject_from_attempt)."""
    from genomic_variant_classifier.source_monitor.c2_github import subject_from_attempt
    return subject_from_attempt(attempt)


def build_receipt_payload(*, subject, verdict, checker, evaluation, diagnostics):
    """A CHECKER-issued receipt payload. `verdict` is a report_verifier.Verdict (status completed), or None when the
    checker ran but could not complete (status unavailable, reason checker.unavailable -- never six invented flags)."""
    if verdict is None:
        decision = {"status": "unavailable", "verified": None, "flags": None, "reviews": [],
                    "reasons": [{"code": "checker.unavailable", "target": ""}]}
        evidence = {"state": "unavailable", "artifact_id": None, "archive_sha256": None, "report_sha256": None}
    else:
        decision = {"status": "completed", "verified": verdict.verified, "flags": dict(verdict.flags),
                    "reviews": list(verdict.reviews), "reasons": list(verdict.reasons)}
        evidence = dict(verdict.evidence)
    return {"schema": "gvc.monitor-receipt", "schema_version": 1, "issuer_role": "checker", "subject": subject,
            "evidence": evidence, "checker": checker, "evaluation": evaluation, "decision": decision,
            "diagnostics": diagnostics}


def main(argv=None, *, fetch=None, read_blob=None, now=None, current=None, step_summary=None, deployment_root=None):
    """`step_summary` is a path to append the Markdown summary to -- passed EXPLICITLY, read from the environment
    only by `__main__` (MEASURED 2026-09-27: reading $GITHUB_STEP_SUMMARY here let a TEST publish a fabricated
    verdict onto CI run #896's summary page)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--repo", default=str(_ROOT), help="a checkout holding the run's commit")
    parser.add_argument("--verdict", required=True, help="where to write the verdict JSON")
    parser.add_argument("--receipt", help="where to write the C2 checker receipt (requires the three arguments below)")
    parser.add_argument("--checker-commit", help="the TRUSTED checkout's full commit (verified by the workflow)")
    parser.add_argument("--evaluation-run-id", type=int, help="THIS verifier run's id")
    parser.add_argument("--evaluation-attempt", type=int, help="THIS verifier run's attempt")
    args = parser.parse_args(argv)
    if args.receipt and None in (args.checker_commit, args.evaluation_run_id, args.evaluation_attempt):
        parser.error("--receipt requires --checker-commit, --evaluation-run-id and --evaluation-attempt")
    started = now or datetime.now(timezone.utc)
    token = os.environ.get("GITHUB_TOKEN")
    fetch = fetch or (lambda url, limit: http_fetch(url, token, limit))
    doc = verdict = attempt = None
    from genomic_variant_classifier.source_monitor import deployment as dep
    from genomic_variant_classifier.source_monitor.run_monitor import REQUIRED_TARGETS
    # The DEPLOYMENT first, from THIS trusted checkout (owner ruling 2026-10-01): an unloadable or unresolved configuration
    # refuses verification -- stated, no receipt, exit 2.
    try:
        deployment = dep.load(deployment_root or _ROOT)      # tests inject a root; __main__ never does
    except dep.DeploymentError as exc:
        doc = rv.Verdict(run_id=args.run_id, run_attempt=args.run_attempt).as_document()
        doc["problems"]["execution_authenticated"].append("the deployment configuration refuses execution: {}".format(exc))
        Path(args.verdict).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.receipt:
            print("NO RECEIPT: DeploymentError: {} -- the coordinator reports verification unavailable".format(exc), file=sys.stderr)
        print("NOT VERIFIED run {} attempt {}".format(args.run_id, args.run_attempt))
        return 2
    api = api_base(deployment)
    # TODAY's policy and the checker's IDENTITY come from THIS trusted checkout, BEFORE any fallible remote evidence
    # collection (owner ruling 2026-10-01: "failure to obtain evidence changes the result, not that identity"). A
    # reconstruction failure makes the current obligation false with its cause recorded; it does not abort verification.
    current_problem = identity = identity_problem = None
    if current is None:
        try:
            current = rv.current_reconstruction(_ROOT)
        except Exception as exc:
            current_problem = "today's interpretation could not be reconstructed: {}: {}".format(type(exc).__name__, exc)
    if args.receipt:
        try:
            identity = rv.build_checker_identity(root=_ROOT, commit=args.checker_commit, current=current,
                                                 required_targets=REQUIRED_TARGETS, deployment=deployment)
        except Exception as exc:
            identity_problem = "the checker identity could not be reconstructed: {}: {}".format(type(exc).__name__, exc)
    try:
        run, attempt = collect_attempt(args.run_id, args.run_attempt, fetch, api)
        artifacts, archive = collect_evidence(args.run_id, fetch, api)
        if read_blob is None:
            # Only an EMPTY `git ls-tree` is "absent" (interpretation_contract.BlobAbsent); every other failure refuses.
            read_blob = rv.commit_blob_reader(ra.git_reader(args.repo))
        verdict = rv.verify(archive, attempt, artifacts, run_id=args.run_id, run_attempt=args.run_attempt,
                            latest_attempt=run.get("run_attempt") if isinstance(run, dict) else None, read_blob=read_blob,
                            now=now or datetime.now(timezone.utc), current=current,
                            required_targets=REQUIRED_TARGETS, deployment=deployment)
        doc = verdict.as_document()
        if current_problem:
            doc["problems"]["current_monitoring_obligation_satisfied"].append(current_problem)
    except Exception as exc:
        doc = rv.Verdict(run_id=args.run_id, run_attempt=args.run_attempt).as_document()
        doc["problems"]["execution_authenticated"].append(
            "verification could not be completed: {}: {}".format(type(exc).__name__, exc))
    Path(args.verdict).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.receipt:
        # Written BEFORE the exit code is returned (owner ruling 2026-09-29), so an ordinary failure still yields one.
        diagnostics = ["{}: {}".format(k, x) for k, xs in doc["problems"].items() for x in xs]
        if verdict is not None:
            diagnostics += ["diagnostic candidate (not a confirmed review): {}".format(json.dumps(c, sort_keys=True))
                            for c in verdict.candidates]
        try:
            if attempt is None:
                raise ValueError("GitHub's attempt record was not obtained; the subject is unknown, so no run is guessed")
            if identity is None:
                raise ValueError(identity_problem)        # never an ordinary checker receipt without an identity
            payload = build_receipt_payload(
                subject=receipt_subject(attempt), verdict=verdict, checker=identity, diagnostics=diagnostics,
                evaluation={"run_id": args.evaluation_run_id, "attempt": args.evaluation_attempt,
                            "started_at": _stamp(started), "finished_at": _stamp(now or datetime.now(timezone.utc))})
            from genomic_variant_classifier.source_monitor import c2_receipt_io
            c2_receipt_io.write_receipt(payload, args.receipt)
            print("RECEIPT written: {}".format(args.receipt))
        except Exception as exc:
            print("NO RECEIPT: {}: {} -- the coordinator reports verification unavailable".format(type(exc).__name__, exc),
                  file=sys.stderr)
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(summary(doc))
    print("{} run {} attempt {}".format("VERIFIED" if doc["verified"] else "NOT VERIFIED", args.run_id, args.run_attempt))
    return 0 if doc["verified"] else 2


if __name__ == "__main__":
    sys.exit(main(step_summary=os.environ.get("GITHUB_STEP_SUMMARY")))
