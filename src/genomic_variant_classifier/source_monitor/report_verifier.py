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
    observation_complete                     the trusted replay (bound to the run's policy) completed its
                                             traversal for EVERY required target AND reported no validation
                                             finding (integrity, plan, transport, structure). Raw traversal
                                             completeness is kept separately in the problems
    claims_reconciled                        the report's qualification strict-equals the replay's (type-sensitive,
                                             nested); the producer's review claims equal the replay's as an exact
                                             multiset; the producer's exit code equals the replay's expectation
    review_required                          execution authenticated AND configuration bound AND the replay found a
                                             newer-release witness or an unsupported name. Never from the producer's
                                             exit code. A valid exit-1 run verifies with a review item
    current_monitoring_obligation_satisfied  verified, coherent attempt chronology, age from the ATTEMPT start within
                                             MAX_AGE, verifier clock not before completion beyond SKEW, and the
                                             run's bound interpretation EQUALS today's (version, policy digest, parts)

INTERPRETATION (owner rulings 2026-09-28, review revision 3): the policy comes from the committed policy file AT THE
RUN COMMIT, or -- when that file is confirmed absent -- from an ADMITTED legacy record (interpretation_contract);
EVERY ingredient is reconstructed independently; the report's interpretation must strict-equal the reconstruction
as a whole. The replay runs only under a policy its trusted handler implements; any other is UNSUPPORTED.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# The repository, source workflow (path AND numeric id), branch and events come from the DEPLOYMENT configuration
# (source_monitor/deployment.py, owner ruling 2026-10-01), passed explicitly -- never module globals.
ARTIFACT_NAME = "source-monitor-report"
REPORT_MEMBER = "report.json"
REPORT_SCHEMA, REPORT_SCHEMA_VERSION = "gvc.monitor-run-report", 1
MAX_ARCHIVE_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 1024 * 1024
#: The monitor is scheduled weekly; a run older than this cannot satisfy today's obligation. POLICY PARAMETERS
#: (owner ruling 2026-09-28: explicit example values, not empirical guarantees).
MAX_AGE = timedelta(days=8)
SKEW = timedelta(minutes=5)
APPROVAL_TARGET = "gnomad-public-releases"
FLAGS = ("execution_authenticated", "configuration_bound", "observation_complete", "claims_reconciled",
         "review_required", "current_monitoring_obligation_satisfied")


@dataclass
class Verdict:
    run_id: int
    run_attempt: int
    flags: dict = field(default_factory=lambda: {f: False for f in FLAGS})
    problems: dict = field(default_factory=lambda: {f: [] for f in FLAGS})
    review_items: list = field(default_factory=list)
    # C2 (owner ruling 2026-09-29): TYPED results, emitted where the checker DETECTS each condition -- the writer never
    # parses the English problems above. reviews: confirmed, from authenticated AND configuration-bound evidence only;
    # candidates: the same typed items when that binding failed -- diagnostic candidates, never confirmed claims.
    reviews: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    evidence: dict = field(default_factory=lambda: {"state": "unavailable", "artifact_id": None,
                                                    "archive_sha256": None, "report_sha256": None})

    def reason(self, code: str, target: str = "") -> None:
        """One stable (code, target) per detected condition -- independent of how many messages describe it."""
        item = {"code": code, "target": target}
        if item not in self.reasons:
            self.reasons.append(item)

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
    if not required_targets or len(set(required_targets)) != len(tuple(required_targets)):
        raise ValueError("the required target roster must be nonempty and duplicate-free")   # obligations come from policy
    report = strict_json(raw, MAX_REPORT_BYTES)
    if type(report) is not dict:
        raise ValueError("the report is not a JSON object")
    # EXACT types: `True == 1` in Python, so an equality test admitted a boolean schema version (reproduced 2026-09-28).
    if report.get("schema") != REPORT_SCHEMA or type(report.get("schema_version")) is not int \
            or report["schema_version"] != REPORT_SCHEMA_VERSION:
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


def _selection_problem(found: int) -> str:
    """ABSENT and AMBIGUOUS are different reasons (MEASURED 2026-09-27: preview run #2 against CI run 36320627013
    said "found 0 -- ambiguous selection is refused"; zero artifacts is absence, not ambiguity)."""
    if found == 0:
        return "no {!r} artifact exists for this run -- the report is absent".format(ARTIFACT_NAME)
    return "expected exactly ONE {!r} artifact, found {} -- ambiguous selection is refused".format(ARTIFACT_NAME, found)


def _time(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def check_execution(run: dict, artifacts: dict, report: dict, *, run_id: int, run_attempt: int,
                    latest_attempt: int, deployment) -> list:
    """GitHub's records for THIS run and attempt. Returns problems.

    `run` is the ATTEMPT record (/actions/runs/{id}/attempts/{n}): its run_attempt is n itself and its time
    window is that attempt's. `latest_attempt` comes from the RUN record. MEASURED 2026-09-27: the two records
    share every field, so the attempt record alone cannot say whether the run was later rerun.
    """
    p = []
    # The source workflow is authenticated by its NUMERIC id as well as its path (2026-10-01: path-only before).
    expect = {"id": run_id, "path": deployment.source_workflow_path, "workflow_id": deployment.source_workflow_id,
              "head_branch": deployment.branch, "status": "completed"}
    for key, value in expect.items():
        if run.get(key) != value:
            p.append("run {} is {!r}, not {!r}".format(key, run.get(key), value))
    if run.get("event") not in deployment.events:
        p.append("run event {!r} is not one of {}".format(run.get("event"), sorted(deployment.events)))
    for key in ("repository", "head_repository"):
        name = (run.get(key) or {}).get("full_name")
        if name != deployment.repository:
            p.append("{} is {!r}, not {!r}".format(key, name, deployment.repository))
    if (run.get("repository") or {}).get("id") != deployment.repository_id:
        p.append("repository id is {!r}, not {!r}".format((run.get("repository") or {}).get("id"), deployment.repository_id))
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
        p.append(_selection_problem(len(named)))
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
        want = {"repository": deployment.repository, "run_id": str(run_id), "run_attempt": str(run_attempt),
                "sha": run.get("head_sha")}
        for key, value in want.items():
            if (declared or {}).get(key) != value:
                p.append("the report declares {} {!r}, GitHub records {!r}".format(key, (declared or {}).get(key), value))
    elif latest_attempt != 1:
        p.append("the report does not declare its attempt and the run has {!r} attempts -- ambiguous".format(latest_attempt))
    return p


def commit_blob_reader(git):
    """read(commit, path) -> bytes. Raises interpretation_contract.BlobAbsent ONLY when `git ls-tree` lists NOTHING for
    the path at an existing commit; every other failure propagates (a refusal, never a downgrade). Regular-file mode,
    size and membership are enforced by release_approval.read_blob_at."""
    from genomic_variant_classifier.data import release_approval as ra
    from genomic_variant_classifier.source_monitor import interpretation_contract as ic

    def read(commit: str, path: str) -> bytes:
        if git("cat-file", "-t", commit).strip() != b"commit":
            raise ic.ContractError("{} is not a commit".format(commit))
        if git("ls-tree", "-z", commit, "--", path) == b"":
            raise ic.BlobAbsent("{}:{} has no tree entry".format(commit, path))
        return ra.read_blob_at(git, commit, path, max_size=MAX_REPORT_BYTES)[1]
    return read


def check_configuration(report: dict, read_blob, head_sha: str):
    """The run commit's policy (committed file, or an ADMITTED legacy record when confirmed absent), the approval it
    selects, and EVERY ingredient reconstructed from Git blobs. -> (problems, policy | None, bound | None)."""
    from genomic_variant_classifier.data import release_approval as ra
    from genomic_variant_classifier.data.source_registry import SourceRegistry
    from genomic_variant_classifier.source_monitor import interpretation_contract as ic

    def read(path):
        return read_blob(head_sha, path)
    try:
        policy = ic.select_policy(head_sha, read_blob)
        registry = SourceRegistry.from_text(read("configs/data_manifest.yaml").decode("utf-8"),
                                            "{}:configs/data_manifest.yaml".format(head_sha))
        ptr = registry.approval_pointer(APPROVAL_TARGET)
        approval = ra.load_approval(ptr.target, ptr.record, ptr.sha256, read)
        bound = ic.reconstruct(policy, read, approved_record_bytes=read(ptr.record),
                               approved_release=approval.approved_release)
    except Exception as exc:
        return (["the interpretation at {} could not be reconstructed: {}: {}".format(head_sha, type(exc).__name__, exc)],
                None, None)
    try:
        ic.bind_report(report.get("interpretation"), bound)
    except ic.ContractError as exc:
        return ["the report's interpretation does not bind to {}: {}".format(head_sha, exc)], policy, None
    return [], policy, bound


@dataclass(frozen=True, order=True)
class ReviewItem:
    target: str
    kind: str          # "newer" | "unsupported"
    raw_prefix: str


def _render(item: ReviewItem, baseline: str) -> str:
    if item.kind == "newer":
        return "{}: release prefix {} is newer than the approved {}".format(item.target, json.dumps(item.raw_prefix), baseline)
    return "{}: release prefix {} is outside the supported release grammar".format(item.target, json.dumps(item.raw_prefix))


def replay_handler(policy):
    """The reviewed replay handler for `policy`, or a ContractError. request_verifier.qualify embodies the verifier's
    OWN constants, so it replays a policy faithfully ONLY when that policy strict-equals the verifier's declaration;
    anything else is UNSUPPORTED rather than silently replayed under today's constants (review revision 3)."""
    from genomic_variant_classifier.source_monitor import interpretation_contract as ic
    from genomic_variant_classifier.source_monitor import request_verifier as rq

    declared = rq.declaration()
    if policy.semantics != ic.SEMANTICS or not ic.strict_equal(policy.rules, declared["release_rules"]) \
            or not ic.strict_equal(policy.plan, declared["request_plan"]):
        raise ic.ContractError("no reviewed replay handler implements this policy (semantics {!r}, baseline {!r})".format(
            policy.semantics, policy.plan.get("approved_baseline")))
    return rq.qualify


def check_observation(report: dict, policy):
    """Replay under the run's policy. -> (complete, reconciled, review_items[str], observation_problems,
    reconciliation_problems, typed ReviewItems, reason codes [(code, target)] emitted AT each detection site). The two problem lists are SEPARATE by construction: observation problems (unsupported or
    failed replay, incomplete traversal, blocking validation findings) and reconciliation problems (qualification,
    claims, exit code).

    Review items are DERIVED from the replay (typed, as an exact multiset); validation findings BLOCK; the producer's
    qualification must strict-equal the replay's; its exit code must equal the replay's expectation."""
    from genomic_variant_classifier.source_monitor import interpretation_contract as ic
    from genomic_variant_classifier.source_monitor import run_monitor as rm
    from genomic_variant_classifier.source_monitor.request_verifier import TraversalCompleteness

    if policy is None:
        return False, False, [], ["the replay cannot run: the run's policy was not established"], [], [], []
    try:
        qualify = replay_handler(policy)
    except ic.ContractError as exc:
        return False, False, [], ["unsupported replay: {}".format(exc)], [], [], [("policy.unsupported", "")]
    baseline = policy.plan["approved_baseline"]
    complete, reconciled = True, True
    replay_items, declared_items, observation, problems, codes = [], [], [], [], []
    for r in report["results"]:
        target, captures = r["target"], r.get("captures") or []
        try:
            outcome = qualify(target, captures)
        except Exception as exc:
            observation.append("{}: replay failed: {}: {}".format(target, type(exc).__name__, exc))
            codes.append(("observation.invalid", target))
            complete = reconciled = False
            continue
        if outcome.traversal_completeness is not TraversalCompleteness.COMPLETE:
            complete = False
            observation.append("{}: traversal is {}".format(target, outcome.traversal_completeness.value))
            codes.append(("observation.incomplete", target))
        for f in outcome.findings:
            doc = f.as_document()
            observation.append("{}: blocking validation finding {}: {}".format(target, doc["reason"], doc["detail"]))
            codes.append(("observation.invalid", target))
        claimed, mine = report["qualification"].get(target), outcome.as_document()
        if not ic.strict_equal(claimed, mine):
            reconciled = False
            differing = sorted(k for k in set(mine) | set(claimed if type(claimed) is dict else {})
                               if not ic.strict_equal((claimed if type(claimed) is dict else {}).get(k), mine.get(k)))
            problems.append("{}: the report's qualification differs from the replay (value or JSON type) in {}".format(
                target, differing))
            codes.append(("claims.disagree", target))
        replay_items += [ReviewItem(target, "newer", w) for w in outcome.positive_witnesses]
        replay_items += [ReviewItem(target, "unsupported", u) for u in outcome.unsupported_names]
        for claim in r.get("findings") or []:
            pair, problem = rm._parse_claim(claim, baseline)
            if problem:
                reconciled = False
                problems.append("{}: {}".format(target, problem))
                codes.append(("claims.disagree", target))
            else:
                declared_items.append(ReviewItem(target, pair[0], pair[1]))
    if Counter(declared_items) != Counter(replay_items):
        reconciled = False
        problems.append("the producer's review claims {} differ from the replay's {} as a multiset".format(
            sorted(Counter(declared_items).items()), sorted(Counter(replay_items).items())))
        codes.append(("claims.disagree", ""))
    qualified_observation = complete and not observation
    expected_exit = 2 if not qualified_observation else (1 if replay_items else 0)
    if report["exit_code"] != expected_exit:
        reconciled = False
        problems.append("the report's exit code {} conflicts with the trusted replay, which requires {}".format(
            report["exit_code"], expected_exit))
        codes.append(("claims.disagree", ""))
    return (qualified_observation, reconciled, [_render(i, baseline) for i in sorted(replay_items)], observation, problems,
            sorted(replay_items), codes)


def check_current(attempt: dict, artifact: dict, now: datetime, historical, current):
    """Today's obligation. Chronology run creation <= attempt start <= artifact creation <= attempt end (GitHub's own
    times); the verifier clock may not precede the attempt's end beyond SKEW; age is measured CONSERVATIVELY from the
    ATTEMPT start (a rerun does not inherit the run's age); the bound interpretation must EQUAL today's.
    -> (problems, reason codes [(code, target)] emitted at each detection site)."""
    p, codes = [], []
    try:
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("the verification time must be timezone-aware")
        created, started = _time(attempt["created_at"]), _time(attempt["run_started_at"])
        observed, ended = _time(artifact["created_at"]), _time(attempt["updated_at"])
        if not created <= started <= observed <= ended:
            p.append("the attempt chronology is inconsistent: created {}, started {}, artifact {}, ended {}".format(
                created, started, observed, ended))
            codes.append(("execution.invalid", ""))
        if ended > now + SKEW:
            p.append("the verifier clock {} precedes the attempt's end {} beyond the {} allowance".format(now, ended, SKEW))
            codes.append(("freshness.future", ""))
        if now - started > MAX_AGE:
            p.append("the attempt started {} ago; the monitoring obligation needs one within {}".format(now - started, MAX_AGE))
            codes.append(("freshness.expired", ""))
    except (KeyError, TypeError, ValueError) as exc:
        p.append("freshness could not be established: {}".format(exc))
        codes.append(("execution.invalid", ""))
    if historical is None or current is None:
        p.append("the run's and today's interpretations must both be reconstructed to compare them")
    elif historical != current:
        differing = sorted(k for k in set(historical.parts) | set(current.parts)
                           if historical.parts.get(k) != current.parts.get(k))
        p.append("this run's interpretation (version {}, policy {}) is not today's (version {}, policy {}); parts differ in "
                 "{}".format(historical.version, historical.contract_sha256, current.version, current.contract_sha256,
                             differing))
        codes.append(("policy.changed", ""))
    return p, codes


def verify(archive: bytes, run: dict, artifacts: dict, *, run_id: int, run_attempt: int, latest_attempt: int, read_blob,
           now: datetime, current, required_targets, deployment) -> Verdict:
    """`run` is the ATTEMPT record; `latest_attempt` the run record's run_attempt; `current` is TODAY's
    interpretation_contract.Bound from the trusted checkout (None if it could not be established)."""
    v = Verdict(run_id=run_id, run_attempt=run_attempt)
    listed = artifacts.get("artifacts") if type(artifacts) is dict else None
    if type(listed) is not list or type(run) is not dict:
        v.problems["execution_authenticated"].append("the run record and artifact listing must be a dict and a list")
        v.reason("execution.invalid")
        return v
    named = [a for a in listed if type(a) is dict and a.get("name") == ARTIFACT_NAME]
    if len(named) != 1:
        v.problems["execution_authenticated"].append(_selection_problem(len(named)))
        v.evidence["state"] = "missing" if not named else "invalid"
        v.reason("artifact.missing" if not named else "artifact.invalid")
        return v
    artifact_id = named[0].get("id")
    v.evidence.update(artifact_id=artifact_id if type(artifact_id) is int and artifact_id > 0 else None,
                      archive_sha256=hashlib.sha256(archive).hexdigest() if type(archive) is bytes else None)
    try:
        raw = read_archive(archive, named[0])
        v.evidence["report_sha256"] = hashlib.sha256(raw).hexdigest()   # the ADMITTED member's bytes, not the ZIP's
        report = parse_report(raw, required_targets)
    except Exception as exc:
        v.problems["execution_authenticated"].append("the report could not be admitted: {}: {}".format(type(exc).__name__, exc))
        v.evidence["state"] = "invalid"
        v.reason("artifact.invalid")
        return v
    if v.evidence["artifact_id"] is None:
        # STATED where detected (measured 2026-09-30: without this line the flag went false with NO problem message).
        v.problems["execution_authenticated"].append("the artifact has no usable identifier ({!r}); its evidence identity is incomplete"
                                                     .format(named[0].get("id")))
    v.evidence["state"] = "complete" if v.evidence["artifact_id"] is not None else "invalid"
    p = check_execution(run, artifacts, report, run_id=run_id, run_attempt=run_attempt, latest_attempt=latest_attempt,
                        deployment=deployment)
    v.problems["execution_authenticated"] += p
    v.flags["execution_authenticated"] = not p and v.evidence["state"] == "complete"
    if not v.flags["execution_authenticated"]:
        v.reason("execution.invalid")
    p, policy, historical = check_configuration(report, read_blob, run.get("head_sha"))
    v.problems["configuration_bound"] += p
    v.flags["configuration_bound"] = not p
    if p:
        v.reason("configuration.invalid")
    complete, reconciled, review, observation, reconciliation, items, codes = check_observation(report, policy)
    v.flags["observation_complete"], v.flags["claims_reconciled"] = complete, reconciled
    v.problems["observation_complete"] += observation
    v.problems["claims_reconciled"] += reconciliation
    for code, target in codes:
        v.reason(code, target)
    v.review_items = review
    typed = [{"target": i.target, "kind": i.kind, "raw_prefix": i.raw_prefix} for i in items]
    bound = v.flags["execution_authenticated"] and v.flags["configuration_bound"]
    v.reviews, v.candidates = (typed, []) if bound else ([], typed)
    v.flags["review_required"] = bound and bool(typed)
    p, codes = check_current(run, named[0], now, historical, current)
    v.problems["current_monitoring_obligation_satisfied"] += p
    for code, target in codes:
        v.reason(code, target)
    v.flags["current_monitoring_obligation_satisfied"] = not p and v.verified
    return v


#: C2 CHECKER IDENTITY (owner ruling 2026-09-29: "bind the executed dependency manifest and environment lock"). MEASURED
#: 2026-09-30: importing the checker loads 7 project modules, but a REAL verification loads 15 -- the 8 lazily imported
#: ones (the replay, the contract, the monitor's roster) decide the verdict. This is that measured set, plus the script
#: and the hash-locked environment the preview workflow installs. A test re-measures it after a real verification.
CHECKER_SOURCES = (
    "requirements-source-monitor.txt",
    "scripts/verify_monitor_run.py",
    "src/genomic_variant_classifier/__init__.py",
    "src/genomic_variant_classifier/data/__init__.py",
    "src/genomic_variant_classifier/data/release_approval.py",
    "src/genomic_variant_classifier/data/source_registry.py",
    "src/genomic_variant_classifier/source_monitor/__init__.py",
    "src/genomic_variant_classifier/source_monitor/c2_github.py",
    "src/genomic_variant_classifier/source_monitor/c2_protocol.py",
    "src/genomic_variant_classifier/source_monitor/c2_receipt_io.py",
    "src/genomic_variant_classifier/source_monitor/deployment.py",
    "src/genomic_variant_classifier/source_monitor/finding_store.py",
    "src/genomic_variant_classifier/source_monitor/heartbeat.py",
    "src/genomic_variant_classifier/source_monitor/interpretation_contract.py",
    "src/genomic_variant_classifier/source_monitor/monitor_supervisor.py",
    "src/genomic_variant_classifier/source_monitor/reason_catalog.py",
    "src/genomic_variant_classifier/source_monitor/report_verifier.py",
    "src/genomic_variant_classifier/source_monitor/request_verifier.py",
    "src/genomic_variant_classifier/source_monitor/run_monitor.py",
)


def code_manifest(repo_root) -> dict:
    """{path: SHA-256} of CHECKER_SOURCES, read from the TRUSTED checkout. A missing file refuses (raises)."""
    from pathlib import Path
    root = Path(repo_root)
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in CHECKER_SOURCES}


def effective_policy(current, required_targets, deployment) -> dict:
    """The COMPLETE effective verification policy (owner ruling 2026-09-29): today's bound interpretation, the required
    target roster, the freshness and skew limits, the admission rules and the supported semantics -- not one file.
    The WRITER's delivery policy (receipt age, clock allowance) is a different policy and is deliberately not here. The
    DEPLOYMENT configuration is bound by its exact-bytes digest: the qualification repository's policy digest therefore
    legitimately differs from production's while the code manifest is identical (owner ruling 2026-10-01)."""
    from genomic_variant_classifier.source_monitor import interpretation_contract as ic
    if current is None:
        raise ValueError("today's interpretation was not reconstructed, so no effective policy can be bound")
    return {"schema": "gvc.verification-policy", "schema_version": 1,
            "interpretation": current.as_document(),
            "required_targets": sorted(required_targets),
            "max_age_seconds": int(MAX_AGE.total_seconds()), "skew_seconds": int(SKEW.total_seconds()),
            "deployment_sha256": deployment.sha256,
            "admission": dict(deployment.admission(), artifact=ARTIFACT_NAME, member=REPORT_MEMBER,
                              report_schema=[REPORT_SCHEMA, REPORT_SCHEMA_VERSION],
                              max_archive_bytes=MAX_ARCHIVE_BYTES, max_report_bytes=MAX_REPORT_BYTES),
            "semantics": ic.SEMANTICS, "legacy_commits": sorted(r.commit for r in ic.LEGACY_RECORDS)}


def build_checker_identity(*, root, commit, current, required_targets, deployment) -> dict:
    """The checker's identity -- derived from the TRUSTED checkout, never from a receipt (owner ruling 2026-10-01).

    THE ONE DEFINITION: the checker computes it BEFORE any fallible remote evidence collection and keeps it for completed
    AND unavailable results ("failure to obtain evidence changes the result, not that identity"); the publisher computes
    it independently from its own checkout to bind the receipt. Raises when the policy or code identity cannot be
    reconstructed -- the caller must then issue NO ordinary checker receipt."""
    from genomic_variant_classifier.source_monitor import c2_protocol as c2
    manifest = code_manifest(root)
    return {"commit": commit,
            "code_manifest_sha256": c2.digest("gvc.checker-code/v1", [[path, sha] for path, sha in sorted(manifest.items())]),
            "policy_sha256": c2.digest("gvc.verification-policy/v1", effective_policy(current, required_targets, deployment))}


def current_reconstruction(repo_root):
    """TODAY's bound interpretation from the trusted checkout: its committed policy file (it must be present), the
    manifest-selected approval, and every ingredient from the checkout's bytes."""
    import hashlib
    from pathlib import Path

    from genomic_variant_classifier.data import release_approval as ra
    from genomic_variant_classifier.data.source_registry import SourceRegistry
    from genomic_variant_classifier.source_monitor import interpretation_contract as ic

    root = Path(repo_root)
    policy = ic.parse_policy((root / ic.CONFIG_PATH).read_bytes())
    ptr = SourceRegistry.load(root / "configs" / "data_manifest.yaml").approval_pointer(APPROVAL_TARGET)
    approval = ra.load_approval(ptr.target, ptr.record, ptr.sha256, ra.worktree_record_reader(root))
    record_bytes = (root / ptr.record).read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != approval.record_sha256:
        raise ra.PolicyError("the approval record's bytes changed after it was loaded")
    return ic.reconstruct(policy, lambda path: (root / path).read_bytes(), approved_record_bytes=record_bytes,
                          approved_release=approval.approved_release)
