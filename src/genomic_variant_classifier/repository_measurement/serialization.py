"""The wire contract, so an instrument need not import what it measures.

ADR-0005.

An external probe analysing this checkout must not do:

    sys.path.insert(...)
    from genomic_variant_classifier.repository_measurement import ...

merely to state "I scanned tracked Markdown files". Otherwise the thing under
measurement becomes a runtime dependency of the measuring instrument, and a
repository too broken to import is a repository that cannot be diagnosed.

So instruments emit plain dictionaries matching this schema, and
repository-side code parses them STRICTLY. Unknown keys are an error: a typo
such as `member_counts` beside `member_count` must not coexist silently with
the field it shadows.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .claims import MeasurementClaim
from .corpus import (CorpusKind, CorpusSnapshot, CorpusSpec,
                     corpus_membership_digest)
from .evidence import AnalysisCoverage, EvidenceItem, EvidenceStrength
from .report import MeasurementMode, MeasurementResult, Verdict

SCHEMA = "gvc.repository-measurement"
SCHEMA_VERSION = 1


class MeasurementSchemaError(ValueError):
    """A payload does not satisfy the wire contract."""


def require_keys(obj: Mapping[str, Any], *, required: frozenset,
                 optional: frozenset = frozenset(), label: str) -> None:
    """Strict key checking. Schema drift is an error, not tolerated entropy."""
    if not isinstance(obj, Mapping):
        raise MeasurementSchemaError(
            "{}: expected an object, got {}".format(label, type(obj).__name__))
    keys = frozenset(obj)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise MeasurementSchemaError(
            "{}: missing keys {!r}".format(label, sorted(missing)))
    if unknown:
        raise MeasurementSchemaError(
            "{}: unknown keys {!r}. A typo beside a real field must not "
            "coexist silently with it.".format(label, sorted(unknown)))


def serialize_measurement(result: MeasurementResult) -> str:
    """Canonical, deterministic bytes. Sorted keys, no incidental whitespace."""
    c = result.corpus
    payload: dict = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "corpus": {
            "kind": c.spec.kind.value,
            "selector": c.spec.selector,
            "enumerator": c.spec.enumerator,
            "includes_untracked": c.spec.includes_untracked,
            "includes_ignored": c.spec.includes_ignored,
            "minimum_members": c.spec.minimum_members,
            "member_count": c.n_members,
            "membership_sha256": c.membership_sha256,
            "members": list(c.members),
            "repository_head": c.repository_head,
            "worktree_dirty": c.worktree_dirty,
        },
        "mode": result.mode.value,
        "claim": {
            "proves": list(result.claim.proves),
            "does_not_prove": list(result.claim.does_not_prove),
            "method": result.claim.method,
        },
        "evidence": [
            {"statement": e.statement, "strength": e.strength.value,
             "basis": e.basis} for e in result.evidence
        ],
        "coverage": (None if result.coverage is None else {
            "selected": result.coverage.selected,
            "attempted": result.coverage.attempted,
            "succeeded": result.coverage.succeeded,
            "failed": result.coverage.failed,
        }),
        "verdict": None if result.verdict is None else result.verdict.value,
        "complete_census": result.complete_census,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


_CORPUS_REQUIRED = frozenset({
    "kind", "selector", "enumerator", "member_count", "membership_sha256"})
_CORPUS_OPTIONAL = frozenset({
    "includes_untracked", "includes_ignored", "minimum_members", "members",
    "repository_head", "worktree_dirty"})
_TOP_REQUIRED = frozenset({
    "schema", "schema_version", "corpus", "mode", "claim", "evidence"})
_TOP_OPTIONAL = frozenset({"coverage", "verdict", "complete_census"})


def parse_measurement(text: str) -> MeasurementResult:
    """Parse and VALIDATE a wire payload. Unknown schema versions are refused."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MeasurementSchemaError("payload is not JSON: {}".format(exc))
    require_keys(payload, required=_TOP_REQUIRED, optional=_TOP_OPTIONAL,
                 label="measurement")
    if payload["schema"] != SCHEMA:
        raise MeasurementSchemaError(
            "unknown schema {!r}; this parser owns {!r}".format(
                payload["schema"], SCHEMA))
    if payload["schema_version"] != SCHEMA_VERSION:
        raise MeasurementSchemaError(
            "unknown schema_version {!r}; this parser owns {}. A future "
            "version must be read by a parser that knows it, never guessed at."
            .format(payload["schema_version"], SCHEMA_VERSION))

    c = payload["corpus"]
    require_keys(c, required=_CORPUS_REQUIRED, optional=_CORPUS_OPTIONAL,
                 label="measurement.corpus")
    spec = CorpusSpec(
        kind=CorpusKind(c["kind"]),
        selector=c["selector"],
        enumerator=c["enumerator"],
        includes_untracked=bool(c.get("includes_untracked", False)),
        includes_ignored=bool(c.get("includes_ignored", False)),
        minimum_members=int(c.get("minimum_members", 0)),
    )
    members = tuple(c.get("members") or ())
    if c.get("members") is not None:
        if len(members) != c["member_count"]:
            raise MeasurementSchemaError(
                "corpus.members holds {} entries but member_count is {}"
                .format(len(members), c["member_count"]))
        if corpus_membership_digest(members) != c["membership_sha256"]:
            raise MeasurementSchemaError(
                "corpus.members does not hash to membership_sha256; the "
                "member list and its identity disagree")
    snapshot = CorpusSnapshot(
        spec=spec, members=members,
        repository_head=c.get("repository_head"),
        worktree_dirty=c.get("worktree_dirty"),
    )
    cl = payload["claim"]
    require_keys(cl, required=frozenset({"proves", "does_not_prove", "method"}),
                 label="measurement.claim")
    claim = MeasurementClaim(
        proves=tuple(cl["proves"]),
        does_not_prove=tuple(cl["does_not_prove"]),
        method=cl["method"])
    evidence = []
    for i, e in enumerate(payload["evidence"]):
        require_keys(e, required=frozenset({"statement", "strength", "basis"}),
                     label="measurement.evidence[{}]".format(i))
        evidence.append(EvidenceItem(
            statement=e["statement"],
            strength=EvidenceStrength(e["strength"]),
            basis=e["basis"]))
    cov = payload.get("coverage")
    coverage = None
    if cov is not None:
        require_keys(cov, required=frozenset(
            {"selected", "attempted", "succeeded", "failed"}),
            label="measurement.coverage")
        coverage = AnalysisCoverage(
            selected=cov["selected"], attempted=cov["attempted"],
            succeeded=cov["succeeded"], failed=cov["failed"])
    verdict = payload.get("verdict")
    return MeasurementResult(
        corpus=snapshot, mode=MeasurementMode(payload["mode"]), claim=claim,
        evidence=tuple(evidence), coverage=coverage,
        verdict=None if verdict is None else Verdict(verdict),
        complete_census=bool(payload.get("complete_census", False)))
