#!/usr/bin/env python3
"""Evidence crosses the store. Supervision reads what was recovered, not what was held.

Author: Monzia Moodie

WHY THIS IS A REBUILD
=====================
MEASURED 2026-09-14, against the previous runner:

    main() never called assess_failure_record()
        -- a reason catalog was built, tested, and never consulted.
    supervision consumed the in-memory TargetResult objects
        -- so persistence was a side effect, not a boundary anything crossed.
    nothing called record_delivery()
        -- every finding stayed pending forever; the retry queue only grew.
    the success heartbeat was sent BEFORE the report write could fail
        -- external success could precede the required output.

Each is the same defect in a different place: a component existed, was
correct, and was not on the path.

THE ORDER, AND EVERY STEP IS LOAD-BEARING
=========================================
    1. begin_attempt          an attempt STARTED -- not a success
    2. run each check          producing an observation
    3. commit_finding          durably, BEFORE any delivery
    4. RECOVER through the store's read interface
    5. ASSESS the recovered record under the approved profile
    6. supervise               required targets from POLICY, never from results
    7. write the report        the required output
    8. record_delivery         only after the report is on disk
    9. heartbeat               only after everything above
   10. exit                    the projection

Step 4 is not ceremony. A record that cannot be recovered and assessed is not
evidence, and discovering that AFTER reporting success is the failure this
ordering prevents. A separate-process recovery test is the qualification
control; a same-process recovery is the routine check.

Step 7 before step 9: a heartbeat that says "the run succeeded" must not be
sent while the report that proves it is still unwritten.

WHAT THIS DOES NOT ESTABLISH
============================
    * that anyone read the alarm.
    * crash durability. Process recovery is tested; power loss is not.
    * source authenticity. A digest matching its own declared value proves
      the retained bytes are self-consistent, not that they came from the
      named source.

WHAT CHANGED 2026-09-16
========================
This docstring used to say "request and response bytes are NOT yet
retained, so the report cannot independently establish request fidelity,
response validity, or traversal completeness. A verifier needs those
captures." That was true when it was written and had been false since Q2
landed: bytes ARE retained, and qualify() independently recomputes digest
integrity, structure, acceptance, and the token chain by value from them.
The docstring was never updated, and this call site was calling
verify_captures() -- the plan-conformance-only subset kept for backward
compatibility -- not qualify(), the whole time. Everything Q2 built had
never once run through this path. It does now.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from genomic_variant_classifier.source_monitor.finding_store import (
    FindingStore, StoreError)
from genomic_variant_classifier.source_monitor.heartbeat import (
    signal_outcome, signal_start)
from genomic_variant_classifier.source_monitor.monitor_supervisor import (
    Health, TargetResult, supervise)
from genomic_variant_classifier.source_monitor.request_verifier import (
    qualify)
from genomic_variant_classifier.source_monitor.reason_catalog import (
    ContractFinding, Reason, ReasonProfile, assess_failure_record,
    make_failure_record)

# ---------------------------------------------------------------------------
# THE APPROVED POLICY. Required targets live here, NOT in the check registry.
#
# If these came from the registry, deleting a check would delete its
# obligation and the supervisor would agree that nothing is missing.
# ---------------------------------------------------------------------------
REQUIRED_TARGETS = (
    "gnomad-public-releases",
)

#: RECOGNITION uses the whole shared catalog; PERMISSION is the subset.
#: MEASURED 2026-09-14: a profile whose known_codes held a hand-picked subset
#: classified a globally defined code as UNKNOWN rather than
#: KNOWN-BUT-DISALLOWED -- different findings with different causes. The
#: previous profile also forbade TRAVERSAL_TOKEN_CYCLE, which the check can
#: emit, so a recognized source defect crashed the reporting path.
RELEASE_PROFILE = ReasonProfile(
    profile_id="release-listing",
    revision="r2",
    known_codes=frozenset(r.value for r in Reason),
    allowed_codes=frozenset({
        Reason.REQUEST_ENDPOINT.value,
        Reason.REQUEST_QUERY_MISMATCH.value,
        Reason.RESPONSE_JSON_DUPLICATE.value,
        Reason.RESPONSE_JSON_SYNTAX.value,
        Reason.RESPONSE_JSON_VALUE.value,
        Reason.RESPONSE_UNEXPECTED_SHAPE.value,
        Reason.TRAVERSAL_TRUNCATED.value,
        Reason.TRAVERSAL_TOKEN_MALFORMED.value,
        Reason.TRAVERSAL_TOKEN_CYCLE.value,
        Reason.TRAVERSAL_BUDGET_EXHAUSTED.value,
        Reason.TRANSPORT_UNREACHABLE.value,
        Reason.TRANSPORT_TIMEOUT.value,
        Reason.CONFIG_INVALID_BASELINE.value,
        # MEASURED 2026-09-15: this code was added to the SHARED CATALOG and
        # never to this permission list. The producer gate refused it --
        # "release-listing does not permit evidence.capture_sequence_invalid"
        # -- on the first run after the runner stopped flattening plan
        # findings. While the flattening was in place the omission was
        # INVISIBLE: every plan finding was committed as
        # REQUEST_QUERY_MISMATCH, so this code was never emitted.
        #
        # The flattening was HIDING the omission, exactly as this project's
        # fillna sweep "was dead code kept alive by the defects it would
        # otherwise have revealed".
        Reason.EVIDENCE_CAPTURE_SEQUENCE_INVALID.value,
        # THE OTHER FIVE THE VERIFIER CAN EMIT.
        #
        # MEASURED 2026-09-15: after adding the one code the producer gate had
        # just refused, I declared the gap closed. A test that DERIVES the
        # enumeration from the verifier's source found FIVE MORE -- so five of
        # the verifier's ten refusal paths would have crashed the reporting
        # path with ProfileError the first time they fired, and
        # artifact.digest_mismatch is the one most likely to fire on a real
        # corrupted response.
        #
        # The previous control hand-enumerated the ADAPTER's codes and passed
        # while all six verifier codes were unpermitted. A hand-maintained
        # enumeration is a claim someone must keep true.
        Reason.ARTIFACT_DIGEST_MISMATCH.value,
        Reason.REQUEST_FRAGMENT.value,
        Reason.REQUEST_QUERY_DUPLICATE.value,
        Reason.REQUEST_QUERY_SYNTAX.value,
        Reason.REQUEST_URL_SYNTAX.value,
        # MEASURED 2026-09-16: an independent forensic probe found the
        # verifier's own recomputation entirely absent. Repairing it added
        # FOUR codes the verifier can now emit -- test_the_profile_permits_
        # every_reason_EACH_PRODUCER_can_emit[request_verifier] refused to
        # pass until they were listed here, which is that test doing exactly
        # the job it was built for.
        Reason.EVIDENCE_INTEGRITY_MISMATCH.value,
        Reason.EVIDENCE_BODY_UNAVAILABLE.value,
        Reason.EVIDENCE_ACCEPTANCE_DISAGREEMENT.value,
        Reason.EVIDENCE_TOKEN_CHAIN_MISMATCH.value,
        # MEASURED 2026-09-16: the FIFTH occurrence of the same omission
        # today -- a code added to the catalog and not yet permitted here.
        # Caught this time by test_the_profile_permits_every_reason_EACH_
        # PRODUCER_can_emit[run_monitor] itself, before any live run, which
        # is exactly what that test was built for.
        Reason.EVIDENCE_WITNESS_DISAGREEMENT.value,
    }),
)

def default_store_path():
    """An ANCHORED store location. Never resolved against the caller's cwd.

    Prefers the project's own runtime-path resolver when it is importable, so
    this subsystem lands where every other piece of durable state does. Falls
    back to a per-user state directory -- not to a relative path, which is the
    defect this function exists to prevent.
    """
    try:
        from genomic_variant_classifier.paths.runtime_paths import (
            resolve_runtime_paths)
        base = Path(resolve_runtime_paths().state_dir)
    except Exception:
        base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("XDG_STATE_HOME")
            or (Path.home() / ".local" / "state")) / "GenomicVariantClassifier"
    return (base / "source_monitor" / "findings.sqlite3").resolve()


CHECKS: dict = {}


def register(target):
    def wrap(fn):
        if target in CHECKS:
            raise RuntimeError("duplicate check registration: {}".format(target))
        CHECKS[target] = fn
        return fn
    return wrap


def validate_configuration() -> None:
    """Refuse BEFORE activation. Not at registration -- that would make
    registration depend on import order."""
    missing = sorted(set(REQUIRED_TARGETS) - set(CHECKS))
    if missing:
        raise RuntimeError(
            "the approved policy requires targets with no implementation: "
            "{}".format(missing))
    if len(set(REQUIRED_TARGETS)) != len(REQUIRED_TARGETS):
        # MEASURED: duplicate required targets were accepted, so the
        # obligation set itself was never validated.
        raise RuntimeError("the approved policy lists a target twice")


@register("gnomad-public-releases")
def check_gnomad_releases(*, transport=None) -> TargetResult:
    from genomic_variant_classifier.source_monitor.gnomad_release_check import (
        observe_releases)
    return observe_releases(profile=RELEASE_PROFILE, transport=transport)


def _persist_and_recover(store, attempt, target, reason, detail):
    """Commit a failure, RECOVER it, and ASSESS it. Returns the assessment.

    The recovery is the point. A record that cannot be read back and
    recognized is not evidence, and finding that out after reporting success
    is exactly the ordering this prevents.
    """
    record = make_failure_record(RELEASE_PROFILE, reason, detail)
    committed = store.commit_finding(attempt_id=attempt, subject=target,
                                     record=record)
    recovered = store.get_finding(committed.event_id)["record"]
    assessment = assess_failure_record(recovered, RELEASE_PROFILE)
    return committed, assessment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", default=None,
                   help="the finding store. Defaults to an ANCHORED runtime "
                        "state directory, NEVER a repository-relative path. "
                        "MEASURED 2026-09-14: an earlier default of "
                        "var/monitor/findings.sqlite3 wrote INSIDE the "
                        "repository, left it dirty, and the next installer "
                        "REFUSED at its working-tree precondition. A "
                        "monitoring subsystem must not dirty the repository it "
                        "monitors. This is LITERATURE-STATE-CWD-RELATIVE-1 in "
                        "a new subsystem: version_monitor_agent.py had "
                        "Path(\"data/agent_state.json\"), resolved against the "
                        "process working directory, and ONE logical store came "
                        "to exist at TWO depths with divergent contents.")
    p.add_argument("--report", default=None)
    p.add_argument("--heartbeat-url", default=os.environ.get("GVC_HEARTBEAT_URL"),
                   help="an external heartbeat endpoint. A scheduled job CANNOT "
                        "supervise its own absence. Defaults to "
                        "GVC_HEARTBEAT_URL so the value never appears in a "
                        "command line -- it is a BEARER CAPABILITY and a "
                        "holder can silence the alarm.")
    args = p.parse_args(argv)

    validate_configuration()
    started = signal_start(args.heartbeat_url)

    # RESOLVE ONCE. MEASURED 2026-09-15, the first run of the INSTALLED code:
    # this site used `args.store or default_store_path()` while line 245 used
    # `args.store` directly, which is None when the default applies:
    #
    #     TypeError: argument should be a str or an os.PathLike object ...
    #                not 'NoneType'
    #
    # The store was created in the right place and the run then crashed
    # writing the report -- after commit_finding, so nothing was lost.
    #
    # Every one of the seventy tests passed --store EXPLICITLY, so the default
    # was covered only by default_store_path() in isolation, never through
    # main(). A suite that never exercises the default cannot catch a defect
    # in the default.
    store_path = Path(args.store) if args.store else default_store_path()
    store = FindingStore(store_path)
    attempt = store.begin_attempt("monitor-run")

    results = []
    assessments = []
    committed_ids = []
    verification = []
    qualification = {}

    for target in REQUIRED_TARGETS:
        try:
            result = CHECKS[target]()
        except Exception as exc:
            # An unexpected exception is NOT evidence about transport. It is
            # an unhandled implementation fault, and calling it
            # transport.unreachable -- as the previous runner did -- asserts a
            # cause nobody measured.
            committed, assessment = _persist_and_recover(
                store, attempt, target, Reason.RESPONSE_UNEXPECTED_SHAPE,
                "unhandled check fault: {}: {}".format(type(exc).__name__, exc))
            committed_ids.append(committed.event_id)
            assessments.append((target, assessment))
            results.append(TargetResult(target, Health.FAILED,
                                        reason=Reason.RESPONSE_UNEXPECTED_SHAPE))
            continue

        # QUALIFY THE RETAINED CAPTURES against an INDEPENDENTLY held plan.
        #
        # MEASURED 2026-09-16: this call site invoked verify_captures(), the
        # PLAN-CONFORMANCE-ONLY subset kept for backward compatibility, not
        # qualify() -- the function Q2 actually built. Digest integrity was
        # never recomputed, `accepted` was never independently re-validated,
        # the token chain was never checked by value, and every displayed
        # finding was the ADAPTER's own self-reported claim. This report's
        # OWN docstring still said "request and response bytes are NOT yet
        # retained" -- true before Q2, false after, and nobody had updated
        # it. Everything Q2 tested had never once run through this path.
        outcome = qualify(target, result.captures)
        if outcome.findings:
            verification.append(
                (target, [f.as_document() for f in outcome.findings]))
            # EACH FINDING KEEPS ITS OWN REASON.
            #
            # MEASURED 2026-09-15: an earlier version committed EVERY plan
            # finding as REQUEST_QUERY_MISMATCH with the details concatenated
            # into a string. A LOST PAGE -- evidence.capture_sequence_invalid,
            # added in this same unit precisely because a gap is not
            # truncation -- was recorded in the durable store as a QUERY
            # MISMATCH. The catalog distinction was destroyed one layer up.
            # qualify()'s findings now ALSO include integrity mismatches,
            # acceptance disagreements and token-chain mismatches -- the same
            # discipline applies to all of them, not only plan-conformance.
            for finding in outcome.findings:
                committed, assessment = _persist_and_recover(
                    store, attempt, target, finding.reason, finding.detail)
                committed_ids.append(committed.event_id)
                assessments.append((target, assessment))
            # The RESULT carries the first finding's reason, because a
            # TargetResult names ONE reason by construction. Every finding is
            # committed individually above, so nothing is lost -- the result's
            # reason is a summary, and the store holds the inventory.
            results.append(TargetResult(
                target, Health.FAILED, reason=outcome.findings[0].reason,
                captures=result.captures))
            continue

        # THE PRODUCER'S CLAIMED FINDINGS vs the INDEPENDENTLY DERIVED
        # WITNESSES. A producer that retained honest bytes but lied about
        # what it found in them -- claimed nothing where a witness exists,
        # or claimed a witness the retained body does not support -- passed
        # silently until this check existed, the same class of gap
        # EVIDENCE_ACCEPTANCE_DISAGREEMENT closed for the `accepted` flag.
        #
        # The comparison is substring-based, not exact-match: the adapter's
        # claim is a READABLE SENTENCE ("release 4.1.1 is newer than the
        # approved 4.1"); the independently derived witness is the bare
        # value ("4.1.1"). Every witness must appear somewhere in the
        # adapter's own claimed text, or the disagreement is real.
        witness_mismatch = any(
            not any(w in claim for claim in result.findings)
            for w in outcome.positive_witnesses)
        if witness_mismatch:
            committed, assessment = _persist_and_recover(
                store, attempt, target, Reason.EVIDENCE_WITNESS_DISAGREEMENT,
                "independently derived witnesses {!r} are not reflected in "
                "the producer's claimed findings {!r}".format(
                    outcome.positive_witnesses, result.findings))
            committed_ids.append(committed.event_id)
            assessments.append((target, assessment))
            results.append(TargetResult(
                target, Health.FAILED,
                reason=Reason.EVIDENCE_WITNESS_DISAGREEMENT,
                captures=result.captures))
            continue

        if result.reason is not None:
            committed, assessment = _persist_and_recover(
                store, attempt, target, result.reason,
                "; ".join(result.findings) or result.health.value)
            committed_ids.append(committed.event_id)
            assessments.append((target, assessment))
            if not assessment.profile_failure_recognized:
                # The record did not survive its own contract. The observation
                # stands, but the reporting path is not qualified to speak for
                # it, so the run is unqualified.
                results.append(TargetResult(
                    target, Health.FAILED,
                    reason=Reason.RESPONSE_UNEXPECTED_SHAPE))
                continue
        results.append(result)
        qualification[target] = outcome.as_document()

    report = supervise(REQUIRED_TARGETS, results)
    document = report.as_document()
    document["attempt_id"] = attempt
    document["store"] = str(store_path.resolve())
    document["assessments"] = [
        {"target": t, "reason": a.reported_reason,
         "recognized": a.profile_failure_recognized,
         "contract_health": list(a.contract_health)}
        for t, a in assessments]
    document["profile"] = {"id": RELEASE_PROFILE.profile_id,
                           "revision": RELEASE_PROFILE.revision}
    document["plan_verification"] = [
        {"target": t, "findings": f} for t, f in verification]
    document["plan_verified_targets"] = [
        r.target for r in report.results if r.captures]
    # THE THREE-AXIS OUTCOME, exposed in a production report for the first
    # time: traversal_completeness, eligible_for_existence_claim and
    # eligible_for_absence_claim answer three DIFFERENT questions that this
    # report previously could not distinguish at all.
    document["qualification"] = qualification
    document["does_not_establish"].append(
        "source authenticity: a response digest is integrity relative to bytes "
        "THIS PROCESS received and self-reported. It authenticates nothing "
        "about the source, and a verifier must hold the approved request plan "
        "independently to check the retained request_url against it")

    # THE REPORT IS THE REQUIRED OUTPUT. It is written BEFORE delivery is
    # recorded and BEFORE any heartbeat, so external success cannot precede it.
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
    if args.report:
        target_path = Path(args.report)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(raw + "\n", encoding="utf-8", newline="\n")

    # DELIVERY IS RECORDED ONLY AFTER THE REPORT EXISTS. The report is this
    # run's delivery channel: a finding a reader can open. Recording it before
    # the write would mark delivered a finding nobody can see.
    delivered = []
    if args.report:
        for event_id in committed_ids:
            try:
                store.record_delivery(event_id, "report:{}".format(target_path))
                delivered.append(event_id)
            except StoreError as exc:
                document.setdefault("delivery_errors", []).append(str(exc))

    store.finish_attempt(
        attempt,
        "complete" if report.exit_code == 0 else
        ("incomplete" if report.exit_code == 1 else "failed"))

    finished = signal_outcome(args.heartbeat_url, report.exit_code)
    document["heartbeat"] = {"start": started.as_document(),
                             "outcome": finished.as_document()}
    document["delivered"] = len(delivered)
    document["pending_deliveries"] = len(store.pending_deliveries())
    if args.report:
        target_path.write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True))
    print("\nexit {}  ({})".format(
        report.exit_code,
        {0: "qualified; no action", 1: "qualified; review required",
         2: "NOT qualified"}[report.exit_code]), file=sys.stderr)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
