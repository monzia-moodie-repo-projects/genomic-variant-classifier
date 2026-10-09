"""Pre-execution run and evaluation intents (owner ruling 2026-10-09).

WHAT WAS AUTHORIZED BEFOREHAND, NOT WHAT HAPPENED AFTERWARD
===========================================================
A run record can hold what was authorized before execution or what happened after it; the analysis STAGE belongs to the former.

    sealed analysis contract    methods, eligibility, universes, endpoint definitions, exposure-failure policy (analysis_contract.py)
    RUN INTENT (this module)    stage, contract digest, plan digest, input identities, environment identity, implementation identity
                                and the RELEASE-POLICY identity -- written once, before the estimator runs or any label is opened
    completed run record        the intent digest, observed outcomes and evidence identities
    derived release decision    which outputs may be evaluated or released under that intent (evaluation_boundary.py)

MEASURED 2026-10-09 BEFORE THIS MODULE EXISTED: the repository had NO operation-intent mechanism. maintenance_channel.py said
"`operation_intent` already provides the root", but no RuntimePaths property or other definition of it exists, now or in main's history
(the string occurs only in the commit that added that sentence). environment_qualification.admission.RunRecord is a MUTABLE
execution-status record (started -> terminal; its "stage" field is the active EXECUTION STEP), so it cannot hold immutable pre-execution
intent. The ruling's alternative applies: a typed, versioned intent record.

RULES (ruling 2026-10-09)
=========================
    * One authoritative stage field; a missing or unknown stage refuses.
    * The intent binds the release-policy identity (exposure_outcomes.release_policy_sha256): a newly selected policy cannot authorize
      itself, and an intent bound to another policy is refused, so an old intent never silently acquires a new meaning.
    * Admission compares the persisted bytes with the digest recorded when the intent was sealed -- never a digest recomputed from a
      newly supplied intent and trusted. Changing ONLY the stage changes the digest, and admission refuses it.
    * A feasibility run stays feasibility; a confirmatory evaluation is a NEW EvaluationIntent naming the admitted computation and its
      unchanged score bytes. Nothing is relabelled.
    * Prior knowledge is DECLARED (software withholding does not blind a researcher who already knows prominent genes), and the earlier
      intents that informed this one are named in `informed_by`.

Canonical form: JSON with sorted keys, no whitespace, ASCII, one trailing LF -- parse() accepts ONLY bytes equal to that rendering.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path

from genomic_variant_classifier.inference.exact_confirmation import InferenceError
from genomic_variant_classifier.inference.exposure_outcomes import STAGES, release_policy_sha256

logger = logging.getLogger(__name__)

__all__ = ["RUN_INTENT_SCHEMA", "EVALUATION_INTENT_SCHEMA", "Stage", "InputIdentity", "RunIntent", "EvaluationIntent", "seal_intent",
           "admit_run_intent", "admit_evaluation_intent"]

RUN_INTENT_SCHEMA = "gvc.analysis-run-intent"
EVALUATION_INTENT_SCHEMA = "gvc.analysis-evaluation-intent"
_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_TREE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_TEXT = 4000


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise InferenceError(code, detail)


class Stage(str, Enum):
    """The DECLARED analysis stage -- not a certification that the evidence establishes a scientific claim."""

    FEASIBILITY = "feasibility"
    CONFIRMATORY = "confirmatory"


if tuple(sorted(s.value for s in Stage)) != tuple(sorted(STAGES)):        # one vocabulary with the release table, checked at import
    raise InferenceError("stage_vocabulary", "Stage and exposure_outcomes.STAGES differ")


def _digest(value, code: str) -> None:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None, code)


def _identifier(value, code: str) -> None:
    _require(type(value) is str and _IDENTIFIER.fullmatch(value) is not None, code)


def _text(value, code: str) -> None:
    _require(type(value) is str and bool(value.strip()) and len(value) <= _MAX_TEXT, code)


def _informed_by(value, code: str) -> None:
    _require(type(value) is tuple and list(value) == sorted(set(value)), code, "sorted, without duplicates")
    for item in value:
        _digest(item, code)


@dataclass(frozen=True)
class InputIdentity:
    """An exact input file: its name in the run, the SHA-256 of its bytes and its size (a RECORDED identity, not proof of upstream
    authenticity)."""

    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _identifier(self.name, "intent_input_name")
        _digest(self.sha256, "intent_input_sha256")
        _require(type(self.size_bytes) is int and self.size_bytes > 0, "intent_input_size")


def _strict_json(raw: bytes, code: str) -> dict:
    def pairs(items):
        out = {}
        for k, v in items:
            _require(k not in out, code, "duplicate key " + k)
            out[k] = v
        return out

    def refuse(token):
        raise InferenceError(code, "non-integer number " + token)
    _require(type(raw) is bytes, code, "bytes required")
    try:
        doc = json.loads(raw.decode("ascii"), object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceError(code, str(exc)[:200]) from None
    _require(type(doc) is dict, code, "not an object")
    return doc


def _canonical(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _stage(value, code: str) -> Stage:
    _require(type(value) is str and value in {s.value for s in Stage}, code, repr(value)[:40])
    return Stage(value)


@dataclass(frozen=True)
class RunIntent:
    """The pre-execution run specification of ONE computation (ruling 2026-10-09)."""

    run_id: str
    stage: Stage
    contract_sha256: str          # SealedContract.contract_id
    plan_sha256: str              # exposure_outcomes.PairPlan.sha256 -- eligibility frozen before execution
    inputs: tuple                 # InputIdentity, sorted by name, unique names
    environment_sha256: str       # the qualified environment identity
    implementation_tree: str      # the git tree of the checkout that runs
    release_policy_sha256: str    # exposure_outcomes.release_policy_sha256() of the implementation that admits it
    prior_knowledge: str          # declared prior knowledge relevant to the evaluation
    informed_by: tuple            # digests of earlier intents whose findings informed this one, sorted

    def __post_init__(self) -> None:
        _identifier(self.run_id, "intent_run_id")
        _require(type(self.stage) is Stage, "intent_stage")
        for name in ("contract_sha256", "plan_sha256", "environment_sha256", "release_policy_sha256"):
            _digest(getattr(self, name), "intent_" + name)
        _require(type(self.inputs) is tuple and len(self.inputs) > 0 and all(type(x) is InputIdentity for x in self.inputs), "intent_inputs")
        names = [x.name for x in self.inputs]
        _require(names == sorted(set(names)), "intent_inputs", "sorted by name, unique")
        _require(type(self.implementation_tree) is str and _GIT_TREE.fullmatch(self.implementation_tree) is not None, "intent_implementation_tree")
        _text(self.prior_knowledge, "intent_prior_knowledge")
        _informed_by(self.informed_by, "intent_informed_by")

    def render(self) -> bytes:
        return _canonical({"schema": RUN_INTENT_SCHEMA, "schema_version": _VERSION, "run_id": self.run_id, "stage": self.stage.value,
                           "contract_sha256": self.contract_sha256, "plan_sha256": self.plan_sha256,
                           "inputs": [{"name": x.name, "sha256": x.sha256, "size_bytes": x.size_bytes} for x in self.inputs],
                           "environment_sha256": self.environment_sha256, "implementation_tree": self.implementation_tree,
                           "release_policy_sha256": self.release_policy_sha256, "prior_knowledge": self.prior_knowledge,
                           "informed_by": list(self.informed_by)})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.render()).hexdigest()

    @classmethod
    def parse(cls, raw: bytes) -> "RunIntent":
        doc = _strict_json(raw, "intent_json")
        keys = {"schema", "schema_version"} | {f.name for f in fields(cls)}
        _require(set(doc) == keys, "intent_keys", repr(sorted(set(doc) ^ keys))[:200])
        _require(doc["schema"] == RUN_INTENT_SCHEMA and doc["schema_version"] == _VERSION and type(doc["schema_version"]) is int, "intent_schema")
        _require(type(doc["inputs"]) is list and all(type(x) is dict and set(x) == {"name", "sha256", "size_bytes"} for x in doc["inputs"]),
                 "intent_inputs")
        _require(type(doc["informed_by"]) is list, "intent_informed_by")
        intent = cls(doc["run_id"], _stage(doc["stage"], "intent_stage"), doc["contract_sha256"], doc["plan_sha256"],
                     tuple(InputIdentity(x["name"], x["sha256"], x["size_bytes"]) for x in doc["inputs"]), doc["environment_sha256"],
                     doc["implementation_tree"], doc["release_policy_sha256"], doc["prior_knowledge"], tuple(doc["informed_by"]))
        _require(intent.render() == raw, "intent_not_canonical", "only the canonical rendering is admitted")
        return intent


@dataclass(frozen=True)
class EvaluationIntent:
    """The pre-evaluation specification of ONE reference evaluation of ONE admitted computation (ruling 2026-10-09).

    A FEASIBILITY evaluation names no reference (its evaluator never opens one); a CONFIRMATORY evaluation must name the reference's
    digest. It may reuse the unchanged score artifact of an admitted feasibility computation -- a new identity, never a relabelling --
    and must then list that run intent in `informed_by` (the evaluation boundary checks it)."""

    evaluation_id: str
    stage: Stage
    run_intent_sha256: str        # the admitted computation
    score_artifact_sha256: str    # the exact score bytes it evaluates
    reference_sha256: str | None  # None exactly when FEASIBILITY
    release_policy_sha256: str
    prior_knowledge: str
    informed_by: tuple

    def __post_init__(self) -> None:
        _identifier(self.evaluation_id, "evaluation_intent_id")
        _require(type(self.stage) is Stage, "evaluation_intent_stage")
        for name in ("run_intent_sha256", "score_artifact_sha256", "release_policy_sha256"):
            _digest(getattr(self, name), "evaluation_intent_" + name)
        if self.stage is Stage.FEASIBILITY:
            _require(self.reference_sha256 is None, "evaluation_intent_feasibility_reference_forbidden")
        else:
            _digest(self.reference_sha256, "evaluation_intent_reference_sha256")
        _text(self.prior_knowledge, "evaluation_intent_prior_knowledge")
        _informed_by(self.informed_by, "evaluation_intent_informed_by")

    def render(self) -> bytes:
        return _canonical({"schema": EVALUATION_INTENT_SCHEMA, "schema_version": _VERSION, "evaluation_id": self.evaluation_id,
                           "stage": self.stage.value, "run_intent_sha256": self.run_intent_sha256,
                           "score_artifact_sha256": self.score_artifact_sha256, "reference_sha256": self.reference_sha256,
                           "release_policy_sha256": self.release_policy_sha256, "prior_knowledge": self.prior_knowledge,
                           "informed_by": list(self.informed_by)})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.render()).hexdigest()

    @classmethod
    def parse(cls, raw: bytes) -> "EvaluationIntent":
        doc = _strict_json(raw, "evaluation_intent_json")
        keys = {"schema", "schema_version"} | {f.name for f in fields(cls)}
        _require(set(doc) == keys, "evaluation_intent_keys", repr(sorted(set(doc) ^ keys))[:200])
        _require(doc["schema"] == EVALUATION_INTENT_SCHEMA and doc["schema_version"] == _VERSION and type(doc["schema_version"]) is int,
                 "evaluation_intent_schema")
        _require(type(doc["informed_by"]) is list, "evaluation_intent_informed_by")
        intent = cls(doc["evaluation_id"], _stage(doc["stage"], "evaluation_intent_stage"), doc["run_intent_sha256"],
                     doc["score_artifact_sha256"], doc["reference_sha256"], doc["release_policy_sha256"], doc["prior_knowledge"],
                     tuple(doc["informed_by"]))
        _require(intent.render() == raw, "evaluation_intent_not_canonical", "only the canonical rendering is admitted")
        return intent


def seal_intent(path, intent) -> str:
    """Persist an intent BEFORE execution: written exclusively (an existing file refuses -- an intent is never replaced), synchronised,
    read back and compared. Returns the SHA-256 of the persisted bytes: the digest admission later requires."""
    _require(type(intent) in (RunIntent, EvaluationIntent), "intent_type")
    path = Path(path)
    payload = intent.render()
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise InferenceError("intent_exists", "{} exists; an intent is never replaced -- a new intent is a new identity".format(path)) from None
    _require(path.read_bytes() == payload, "intent_write_mismatch", str(path))
    return hashlib.sha256(payload).hexdigest()


def _admit(raw: bytes, admitted_sha256: str, kind, code: str):
    _digest(admitted_sha256, code + "_admitted_digest")
    _require(type(raw) is bytes, code + "_bytes")
    _require(hashlib.sha256(raw).hexdigest() == admitted_sha256, code + "_binding_mismatch",
             "the bytes are not the ones admitted before execution")
    intent = kind.parse(raw)
    current = release_policy_sha256()
    _require(intent.release_policy_sha256 == current, code + "_release_policy_not_this_implementation",
             "bound {} / this implementation {}".format(intent.release_policy_sha256, current))
    return intent


def admit_run_intent(raw: bytes, admitted_sha256: str) -> RunIntent:
    """The run intent whose persisted bytes have the digest recorded when it was sealed, parsed strictly, and bound to the release
    policy of the implementation now running. Anything else refuses."""
    return _admit(raw, admitted_sha256, RunIntent, "intent")


def admit_evaluation_intent(raw: bytes, admitted_sha256: str) -> EvaluationIntent:
    return _admit(raw, admitted_sha256, EvaluationIntent, "evaluation_intent")
