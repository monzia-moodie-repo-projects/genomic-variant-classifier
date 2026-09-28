"""Verify one source-monitor run attempt from GitHub's records and its report archive (change C1, 2026-09-27).

PREVIEW: writes a verdict and a step summary; it writes NO issue. Trusted code only -- the report, its archive and
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

API = "https://api.github.com/repos/" + rv.EXPECTED_REPOSITORY
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


def collect(run_id, run_attempt, fetch):
    def get_json(path):
        return rv.strict_json(fetch(API + path, MAX_JSON_BYTES), MAX_JSON_BYTES)
    run = get_json("/actions/runs/{}".format(run_id))
    attempt = get_json("/actions/runs/{}/attempts/{}".format(run_id, run_attempt))
    artifacts = get_json("/actions/runs/{}/artifacts?per_page=100".format(run_id))
    named = [a for a in artifacts.get("artifacts", []) if isinstance(a, dict) and a.get("name") == rv.ARTIFACT_NAME]
    archive = fetch(API + "/actions/artifacts/{}/zip".format(named[0]["id"]), rv.MAX_ARCHIVE_BYTES) if len(named) == 1 else b""
    return run, attempt, artifacts, archive


def summary(doc):
    lines = ["## Source-monitor run verification (PREVIEW -- writes no issue)", "",
             "Run {} attempt {}: **{}**".format(doc["run_id"], doc["run_attempt"], "VERIFIED" if doc["verified"] else "NOT VERIFIED"), "",
             "| Result | Value |", "|---|---|"]
    lines += ["| `{}` | {} |".format(k, v) for k, v in doc["flags"].items()]
    if doc["review_items"]:
        lines += ["", "### Review items"] + ["- {}".format(i) for i in doc["review_items"]]
    problems = [(k, p) for k, ps in doc["problems"].items() for p in ps]
    if problems:
        lines += ["", "### Problems"] + ["- `{}`: {}".format(k, p) for k, p in problems]
    return "\n".join(lines) + "\n"


def main(argv=None, *, fetch=None, read_blob=None, now=None, current_parts=None, step_summary=None):
    """`step_summary` is a path to append the Markdown summary to -- passed EXPLICITLY, read from the environment
    only by `__main__` (MEASURED 2026-09-27: reading $GITHUB_STEP_SUMMARY here let a TEST publish a fabricated
    verdict onto CI run #896's summary page)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--repo", default=str(_ROOT), help="a checkout holding the run's commit")
    parser.add_argument("--verdict", required=True, help="where to write the verdict JSON")
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    fetch = fetch or (lambda url, limit: http_fetch(url, token, limit))
    doc = None
    try:
        run, attempt, artifacts, archive = collect(args.run_id, args.run_attempt, fetch)
        if read_blob is None:
            git = ra.git_reader(args.repo)
            read_blob = lambda commit, path: ra.read_blob_at(git, commit, path, max_size=rv.MAX_REPORT_BYTES)[1]  # noqa: E731
        if current_parts is None:
            from genomic_variant_classifier.source_monitor import run_monitor as rm
            current_parts = rm._approval_and_fingerprint()[1]["parts"]
        from genomic_variant_classifier.source_monitor.run_monitor import REQUIRED_TARGETS
        doc = rv.verify(archive, attempt, artifacts, run_id=args.run_id, run_attempt=args.run_attempt,
                        latest_attempt=run.get("run_attempt") if isinstance(run, dict) else None, read_blob=read_blob,
                        now=now or datetime.now(timezone.utc), current_parts=current_parts,
                        required_targets=REQUIRED_TARGETS).as_document()
    except Exception as exc:
        doc = rv.Verdict(run_id=args.run_id, run_attempt=args.run_attempt).as_document()
        doc["problems"]["execution_authenticated"].append(
            "verification could not be completed: {}: {}".format(type(exc).__name__, exc))
    Path(args.verdict).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write(summary(doc))
    print("{} run {} attempt {}".format("VERIFIED" if doc["verified"] else "NOT VERIFIED", args.run_id, args.run_attempt))
    return 0 if doc["verified"] else 2


if __name__ == "__main__":
    sys.exit(main(step_summary=os.environ.get("GITHUB_STEP_SUMMARY")))
