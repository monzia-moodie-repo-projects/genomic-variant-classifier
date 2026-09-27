"""Verify a source-monitor run from GitHub's records and the run's own report -- as DATA.

Created 2026-09-27 (change C, stage C1: PREVIEW). Owner rulings of 2026-09-25/26: the checker lives in
this repository and runs as TRUSTED code; the report, its archive and the files at the run's commit are
read as data and never executed or imported. The existing alert remains the only issue writer until this
verifier has shown it reports failures as well as successes.

THE SIX RESULTS (the ruling's exact names; they answer different questions)
    execution_authenticated                  GitHub's records bind repository, workflow, event, branch,
                                             commit, run ID, attempt and exactly one report artifact whose
                                             archive digest this process recomputed
    configuration_bound                      the approval and interpretation dependencies AT THE RUN'S
                                             COMMIT (Git blobs) match the report's fingerprint parts
    observation_complete                     the retained responses replay, under the trusted verifier, to a
                                             complete traversal
    claims_reconciled                        the report's qualification equals the trusted replay EXACTLY, and
                                             the producer's claims reconcile with its witnesses and names
    review_required                          the run reports something to review (exit 1). A valid exit-1 run
                                             is a SUCCESSFUL verification with a review item -- verification
                                             never requires zero findings
    current_monitoring_obligation_satisfied  the run is recent enough AND its parts equal what the trusted
                                             code on main computes now. A historical run can be valid under its
                                             own policy and still not satisfy today's obligation.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

EXPECTED_REPOSITORY = "monzia-moodie-repo-projects/genomic-variant-classifier"
EXPECTED_WORKFLOW_PATH = ".github/workflows/source_monitor.yml"
EXPECTED_EVENTS = frozenset({"schedule", "workflow_dispatch"})
EXPECTED_BRANCH = "main"
ARTIFACT_NAME = "source-monitor-report"
REPORT_MEMBER = "report.json"
REPORT_SCHEMA, REPORT_SCHEMA_VERSION = "gvc.monitor-run-report", 1
MAX_ARCHIVE_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 1024 * 1024
#: The monitor is scheduled weekly; a run older than this cannot satisfy today's obligation.
MAX_AGE = timedelta(days=8)
_PART_FILES = {"adapter_code": "src/genomic_variant_classifier/source_monitor/gnomad_release_check.py",
               "verifier_code": "src/genomic_variant_classifier/source_monitor/request_verifier.py",
               "environment_lock": "requirements-source-monitor.txt"}
FLAGS = ("execution_authenticated", "configuration_bound", "observation_complete", "claims_reconciled",
         "review_required", "current_monitoring_obligation_satisfied")


@dataclass
class Verdict:
    run_id: int
    run_attempt: int
    flags: dict = field(default_factory=lambda: {f: False for f in FLAGS})
    problems: dict = field(default_factory=lambda: {f: [] for f in FLAGS})
    review_items: list = field(default_factory=list)

    @property
    def verified(self) -> bool:
        """The run is authentic, bound, complete and reconciled -- review items do NOT make this false."""
        return all(self.flags[f] for f in FLAGS[:4])

    def as_document(self) -> dict:
        return {"schema": "gvc.monitor-run-verdict", "schema_version": 1, "run_id": self.run_id,
                "run_attempt": self.run_attempt, "verified": self.verified, "flags": dict(self.flags),
                "problems": {k: list(v) for k, v in self.problems.items()}, "review_items": list(self.review_items)}


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: {}".format(key))
        out[key] = value
    return out


def _no_constant(value):
    raise ValueError("non-JSON numeric constant: {}".format(value))


def strict_json(raw: bytes, limit: int):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("expected 1..{} bytes".format(limit))
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("a byte-order mark is not permitted")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_no_constant)


def read_archive(archive: bytes, artifact: dict) -> bytes:
    """The artifact's archive -> report.json bytes. RAISES on any mismatch; a warning is not enough."""
    if type(archive) is not bytes or not 0 < len(archive) <= MAX_ARCHIVE_BYTES:
        raise ValueError("archive must be 1..{} bytes".format(MAX_ARCHIVE_BYTES))
    digest = artifact.get("digest")
    if type(digest) is not str or not digest.startswith("sha256:"):
        raise ValueError("the artifact has no sha256 digest to verify against")
    actual = "sha256:" + hashlib.sha256(archive).hexdigest()
    if actual != digest:
        raise ValueError("archive digest {} differs from GitHub's {}".format(actual, digest))
    if artifact.get("size_in_bytes") != len(archive):
        raise ValueError("archive is {} bytes; GitHub records {}".format(len(archive), artifact.get("size_in_bytes")))
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        infos = zf.infolist()
        if [i.filename for i in infos] != [REPORT_MEMBER]:
            raise ValueError("the archive must hold exactly {!r}; it holds {}".format(
                REPORT_MEMBER, [i.filename for i in infos]))
        info = infos[0]
        if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("report.json is not a regular file member")
        if info.flag_bits & 0x1:
            raise ValueError("encrypted archive members are refused")
        if not 0 < info.file_size <= MAX_REPORT_BYTES:
            raise ValueError("report.json declares {} bytes".format(info.file_size))
        raw = zf.read(info)
    if len(raw) != info.file_size:
        raise ValueError("report.json read {} bytes, declared {}".format(len(raw), info.file_size))
    return raw


def parse_report(raw: bytes, required_targets) -> dict:
    report = strict_json(raw, MAX_REPORT_BYTES)
    if type(report) is not dict:
        raise ValueError("the report is not a JSON object")
    if report.get("schema") != REPORT_SCHEMA or report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported report schema {!r} v{!r}".format(report.get("schema"), report.get("schema_version")))
    results = report.get("results")
    if type(results) is not list or any(type(r) is not dict or type(r.get("target")) is not str for r in results):
        raise ValueError("results must be a list of objects each naming a target")
    targets = [r["target"] for r in results]
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate targets in results: {}".format(sorted(targets)))
    if set(targets) != set(required_targets):
        raise ValueError("targets {} differ from the required {} (absent: {}; orphan: {})".format(
            sorted(targets), sorted(required_targets), sorted(set(required_targets) - set(targets)),
            sorted(set(targets) - set(required_targets))))
    if type(report.get("exit_code")) is not int or report["exit_code"] not in (0, 1, 2):
        raise ValueError("exit_code must be 0, 1 or 2")
    q = report.get("qualification")
    if type(q) is not dict or not set(q) <= set(required_targets):
        raise ValueError("qualification must be an object keyed only by required targets")
    return report


def _time(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def check_execution(run: dict, artifacts: dict, report: dict, *, run_id: int, run_attempt: int,
                    latest_attempt: int) -> list:
    """GitHub's records for THIS run and attempt. Returns problems.

    `run` is the ATTEMPT record (/actions/runs/{id}/attempts/{n}): its run_attempt is n itself and its time
    window is that attempt's. `latest_attempt` comes from the RUN record. MEASURED 2026-09-27: the two records
    share every field, so the attempt record alone cannot say whether the run was later rerun.
    """
    p = []
    expect = {"id": run_id, "path": EXPECTED_WORKFLOW_PATH, "head_branch": EXPECTED_BRANCH, "status": "completed"}
    for key, value in expect.items():
        if run.get(key) != value:
            p.append("run {} is {!r}, not {!r}".format(key, run.get(key), value))
    if run.get("event") not in EXPECTED_EVENTS:
        p.append("run event {!r} is not one of {}".format(run.get("event"), sorted(EXPECTED_EVENTS)))
    for key in ("repository", "head_repository"):
        name = (run.get(key) or {}).get("full_name")
        if name != EXPECTED_REPOSITORY:
            p.append("{} is {!r}, not {!r}".format(key, name, EXPECTED_REPOSITORY))
    if run.get("run_attempt") != run_attempt:
        p.append("the attempt record is for attempt {!r}, not {}".format(run.get("run_attempt"), run_attempt))
    if type(latest_attempt) is not int or not 1 <= run_attempt <= latest_attempt:
        p.append("attempt {} is not an attempt of this run (latest {!r})".format(run_attempt, latest_attempt))
    listed = artifacts.get("artifacts")
    if type(listed) is not list:
        # One-shot iterators and other non-lists are REFUSED, never consumed or coerced (acceptance: iterator inputs).
        return p + ["the artifact listing is not a list: {}".format(type(listed).__name__)]
    if artifacts.get("total_count") != len(listed):
        p.append("the artifact listing is incomplete (total_count {} vs {} listed)".format(artifacts.get("total_count"), len(listed)))
    named = [a for a in listed if a.get("name") == ARTIFACT_NAME]
    if len(named) != 1:
        p.append("expected exactly ONE {!r} artifact, found {} -- ambiguous selection is refused".format(ARTIFACT_NAME, len(named)))
    else:
        a = named[0]
        wr = a.get("workflow_run") or {}
        if a.get("expired") is not False:
            p.append("the artifact is expired or its state is unknown")
        if wr.get("id") != run_id or wr.get("head_sha") != run.get("head_sha"):
            p.append("the artifact belongs to run {!r} at {!r}, not this run".format(wr.get("id"), wr.get("head_sha")))
        try:
            made, start, end = _time(a["created_at"]), _time(run["run_started_at"]), _time(run["updated_at"])
            if not start <= made <= end:
                p.append("the artifact was created {} outside the attempt's window {}..{}".format(made, start, end))
        except (KeyError, TypeError, ValueError) as exc:
            p.append("the attempt's time window could not be established: {}".format(exc))
    declared = report.get("github_run")
    if declared is not None:
        want = {"repository": EXPECTED_REPOSITORY, "run_id": str(run_id), "run_attempt": str(run_attempt),
                "sha": run.get("head_sha")}
        for key, value in want.items():
            if (declared or {}).get(key) != value:
                p.append("the report declares {} {!r}, GitHub records {!r}".format(key, (declared or {}).get(key), value))
    elif latest_attempt != 1:
        p.append("the report does not declare its attempt and the run has {!r} attempts -- ambiguous".format(latest_attempt))
    return p


def check_configuration(report: dict, read_blob, head_sha: str) -> list:
    """Approval and interpretation dependencies AT THE RUN'S COMMIT, read as Git blobs -- never executed."""
    from genomic_variant_classifier.data import release_approval as ra
    from genomic_variant_classifier.data.source_registry import SourceRegistry

    p = []
    it = report.get("interpretation") or {}
    parts = it.get("parts")
    if type(parts) is not dict:
        return ["the report has no interpretation parts: {!r}".format(it)]
    try:
        registry = SourceRegistry.from_text(read_blob(head_sha, "configs/data_manifest.yaml").decode("utf-8"),
                                            "{}:configs/data_manifest.yaml".format(head_sha))
        ptr = registry.approval_pointer("gnomad-public-releases")
        approval = ra.load_approval(ptr.target, ptr.record, ptr.sha256, lambda path: read_blob(head_sha, path))
        if parts.get("approval") != approval.record_sha256:
            p.append("approval part {!r} is not the record selected at {} ({})".format(
                parts.get("approval"), head_sha, approval.record_sha256))
    except Exception as exc:
        p.append("the approval at {} could not be established: {}: {}".format(head_sha, type(exc).__name__, exc))
    for part, path in _PART_FILES.items():
        try:
            digest = hashlib.sha256(read_blob(head_sha, path)).hexdigest()
        except Exception as exc:
            p.append("{} at {} unreadable: {}".format(path, head_sha, exc))
            continue
        if parts.get(part) != digest:
            p.append("{} part {!r} is not the SHA-256 of {} at {} ({})".format(part, parts.get(part), path, head_sha, digest))
    try:
        if ra.interpretation_fingerprint(parts) != it.get("fingerprint"):
            p.append("the fingerprint does not recompute from its parts")
    except Exception as exc:
        p.append("the fingerprint could not be recomputed: {}".format(exc))
    return p


def check_observation(report: dict):
    """Replay the retained responses with the TRUSTED verifier. -> (complete, reconciled, review_items, problems)."""
    from genomic_variant_classifier.source_monitor import run_monitor as rm
    from genomic_variant_classifier.source_monitor.request_verifier import TraversalCompleteness, qualify

    complete, reconciled, review, problems = True, True, [], []
    for r in report["results"]:
        target, captures = r["target"], r.get("captures") or []
        try:
            outcome = qualify(target, captures)
        except Exception as exc:
            problems.append("{}: replay failed: {}: {}".format(target, type(exc).__name__, exc))
            complete = reconciled = False
            continue
        if outcome.traversal_completeness is not TraversalCompleteness.COMPLETE:
            complete = False
            problems.append("{}: replay is {}".format(target, outcome.traversal_completeness.value))
        # The report's qualification for this target must equal the trusted replay's document EXACTLY -- plan
        # fingerprint, completeness, both eligibility flags, witnesses, unsupported names, plan findings, count.
        claimed, mine = report["qualification"].get(target), outcome.as_document()
        if claimed != mine:
            reconciled = False
            differing = sorted(k for k in set(mine) | set(claimed or {}) if (claimed or {}).get(k) != mine.get(k))
            problems.append("{}: the report's qualification differs from the replay in {}".format(target, differing))
        issues = rm._reconcile_claims(r.get("findings") or [], outcome.positive_witnesses, outcome.unsupported_names)
        if issues:
            reconciled = False
            problems.extend("{}: {}".format(target, i) for i in issues)
        review.extend("{}: {}".format(target, f) for f in (r.get("findings") or []))
    return complete, reconciled, review, problems


def check_current(report: dict, run: dict, now: datetime, current: dict) -> list:
    p = []
    try:
        age = now - _time(run["created_at"])
        if age > MAX_AGE:
            p.append("the run is {} old; the monitoring obligation needs one within {}".format(age, MAX_AGE))
    except (KeyError, TypeError, ValueError) as exc:
        p.append("the run's age could not be established: {}".format(exc))
    parts = (report.get("interpretation") or {}).get("parts") or {}
    for key, value in sorted((current or {}).items()):
        if parts.get(key) != value:
            p.append("{} of this run {!r} is not today's {!r}".format(key, parts.get(key), value))
    return p


def verify(archive: bytes, run: dict, artifacts: dict, *, run_id: int, run_attempt: int, latest_attempt: int, read_blob,
           now: datetime, current_parts: dict, required_targets) -> Verdict:
    """`run` is the ATTEMPT record; `latest_attempt` the run record's run_attempt (see check_execution)."""
    v = Verdict(run_id=run_id, run_attempt=run_attempt)
    listed = artifacts.get("artifacts") if type(artifacts) is dict else None
    if type(listed) is not list or type(run) is not dict:
        v.problems["execution_authenticated"].append("the run record and artifact listing must be a dict and a list")
        return v
    named = [a for a in listed if type(a) is dict and a.get("name") == ARTIFACT_NAME]
    if len(named) != 1:
        # Refuse BEFORE reading any archive, naming the real reason: a verifier must never pick one of several
        # candidates (e.g. the most recent) -- and must never misstate why it refused.
        v.problems["execution_authenticated"].append(
            "expected exactly ONE {!r} artifact, found {} -- ambiguous selection is refused".format(ARTIFACT_NAME, len(named)))
        return v
    try:
        report = parse_report(read_archive(archive, named[0]), required_targets)
    except Exception as exc:
        v.problems["execution_authenticated"].append("the report could not be admitted: {}: {}".format(type(exc).__name__, exc))
        return v
    p = check_execution(run, artifacts, report, run_id=run_id, run_attempt=run_attempt, latest_attempt=latest_attempt)
    v.problems["execution_authenticated"] += p
    v.flags["execution_authenticated"] = not p
    p = check_configuration(report, read_blob, run.get("head_sha"))
    v.problems["configuration_bound"] += p
    v.flags["configuration_bound"] = not p
    complete, reconciled, review, p = check_observation(report)
    v.flags["observation_complete"], v.flags["claims_reconciled"] = complete, reconciled
    v.problems["observation_complete" if not complete else "claims_reconciled"] += p
    v.review_items = review
    v.flags["review_required"] = report["exit_code"] == 1 and bool(review)
    p = check_current(report, run, now, current_parts)
    v.problems["current_monitoring_obligation_satisfied"] += p
    v.flags["current_monitoring_obligation_satisfied"] = not p and v.verified
    return v
