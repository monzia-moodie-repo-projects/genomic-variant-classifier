"""The ONE authoritative interpretation contract of the source monitor (owner rulings 2026-09-28, review revision 3).

DEPENDENCY-LIGHT BY DESIGN (ruling): standard library only. No approval, producer or Git code is imported here; the
CALLER reads committed blobs and supplies approval bytes that it has already selected and authorized.

WHAT THIS OWNS: the versioned ingredient ROSTER (names and committed paths), the fingerprint protocol, the named JSON
codecs, the committed POLICY file format and its parser, the explicit LEGACY domain, independent reconstruction and
whole-document binding. It does NOT own replay or grammar semantics: the adapter and the verifier keep their independent
implementations, and each must DECLARE its parameters so they can be compared with the committed policy.

FINGERPRINT PROTOCOL
    version 1  SHA256(b"gvc-monitor-interpretation/v1\\0" + compact({approval, release_rules, request_plan, adapter_code,
               verifier_code, environment_lock})). HISTORICAL; accepted only for LEGACY_RECORDS.
    version 2  SHA256(b"gvc-monitor-interpretation/v2\\0" + compact({"contract_sha256": SHA256(policy file bytes),
               "parts": version-1 parts + orchestrator_code})). The committed policy file itself is bound.

POLICY SELECTION for an AUTHENTICATED commit (never from the report, never from a timestamp):
    present              -> parse the committed bytes (schema below);
    confirmed ABSENT     -> version 1 ONLY if the commit is a LEGACY_RECORD (its own rules/plan bytes), else REFUSED;
    unreadable / other   -> the caller's error propagates (a refusal, never a downgrade).
A legacy commit whose tree CONTAINS the file is refused ("legacy history contradicts the authenticated tree").

"Version 1" names a fingerprint FORMAT; it does not assert that every old run used identical constants. Each legacy
record therefore carries that commit's OWN rules and plan bytes (measured 2026-09-29: identical at all three).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Callable

CONFIG_PATH = "configs/source_monitor_interpretation.json"
POLICY_SCHEMA = "gvc.source-monitor-interpretation"
SEMANTICS = "gnomad-release-listing-v1"
COMPACT, SPACED = "json-ascii-compact-v1", "json-ascii-spaced-v1"
ENCODINGS = {"release_rules": COMPACT, "request_plan": SPACED}
MAX_POLICY_BYTES = 65536
MAX_DEPTH = 24
_SAFE_INT = 2 ** 53 - 1
_HEX64 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_RELEASE = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){1,2}")
_PREFIX = "src/genomic_variant_classifier/source_monitor/"
_V1_FILES = (("adapter_code", _PREFIX + "gnomad_release_check.py"),
             ("verifier_code", _PREFIX + "request_verifier.py"),
             ("environment_lock", "requirements-source-monitor.txt"))
_DATA_ROLES = ("approval", "release_rules", "request_plan")


class ContractError(ValueError):
    """Any violation of the interpretation contract. Never caught to produce a default."""


class BlobAbsent(LookupError):
    """Raised by a CALLER's reader ONLY after an authenticated, complete tree lookup proves the path absent."""


# ---------------------------------------------------------------------------------------------- strict primitives
def checked_digest(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ContractError("{} must be a complete lowercase SHA-256 digest, got {!r}".format(label, value))
    return value


def exact_int(value: object, lo: int, hi: int, label: str) -> int:
    if type(value) is not int or not lo <= value <= hi:
        raise ContractError("{} must be an integer in [{}, {}] (never a boolean), got {!r}".format(label, lo, hi, value))
    return value


def exact_keys(value: object, keys, label: str) -> dict:
    if type(value) is not dict or set(value) != set(keys):
        got = sorted(value) if type(value) is dict else type(value).__name__
        raise ContractError("{} must have exactly {}, got {}".format(label, sorted(keys), got))
    return value


def strict_equal(left, right) -> bool:
    """Type-sensitive equality: Python's == equates True, 1 and 1.0, including inside dicts and lists."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(strict_equal(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def _tree(value, depth=0):
    if depth > MAX_DEPTH:
        raise ContractError("JSON nesting deeper than {} is refused".format(MAX_DEPTH))
    if value is None or type(value) in (bool, str):
        if type(value) is str:
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise ContractError("invalid Unicode in a JSON string") from exc
        return
    if type(value) is int:
        exact_int(value, -_SAFE_INT, _SAFE_INT, "a JSON integer")
        return
    if type(value) is list:
        for item in value:
            _tree(item, depth + 1)
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for key, item in value.items():
            _tree(key, depth + 1)
            _tree(item, depth + 1)
        return
    raise ContractError("this protocol admits no floating-point or other non-JSON values, got {}".format(
        type(value).__name__))


def strict_json(raw: bytes, *, limit: int = MAX_POLICY_BYTES):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ContractError("expected 1..{} JSON bytes".format(limit))
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ContractError("a byte-order mark is refused")

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ContractError("duplicate JSON key {!r}".format(key))
            out[key] = value
        return out

    def no_float(token):
        raise ContractError("non-integer JSON number {} is refused".format(token))
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=no_float, parse_constant=no_float)
    except ContractError:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ContractError("malformed JSON: {}".format(exc)) from exc
    _tree(value)
    return value


def encode(value, encoding: str) -> bytes:
    """The project's NAMED codecs (sorted keys, ASCII). Neither is RFC 8785 (JSON Canonicalization Scheme)."""
    _tree(value)
    if encoding == COMPACT:
        separators = (",", ":")
    elif encoding == SPACED:
        separators = (", ", ": ")
    else:
        raise ContractError("unsupported codec {!r}".format(encoding))
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=separators).encode("ascii")


# ---------------------------------------------------------------------------------------------- roster and fingerprint
@dataclass(frozen=True)
class Spec:
    version: int
    files: tuple

    @property
    def names(self) -> frozenset:
        return frozenset(_DATA_ROLES + tuple(name for name, _ in self.files))


def spec(version: object) -> Spec:
    exact_int(version, 1, 2, "the fingerprint version")
    files = _V1_FILES + ((("orchestrator_code", _PREFIX + "run_monitor.py"),) if version == 2 else ())
    return Spec(version, files)


def validate_parts(parts, version) -> dict:
    exact_keys(parts, spec(version).names, "version-{} interpretation parts".format(version))
    return {name: checked_digest(value, name) for name, value in parts.items()}


def fingerprint(parts, *, version: object, contract_sha256=None) -> str:
    parts = validate_parts(parts, version)
    if version == 1:
        if contract_sha256 is not None:
            raise ContractError("a version-1 fingerprint cannot include a configuration digest")
        payload = parts
    else:
        payload = {"contract_sha256": checked_digest(contract_sha256, "contract_sha256"), "parts": parts}
    return hashlib.sha256("gvc-monitor-interpretation/v{}\0".format(version).encode("ascii")
                          + encode(payload, COMPACT)).hexdigest()


def legacy_fingerprint(parts) -> str:
    """The exact historical version-1 byte algorithm (compatibility wrapper)."""
    return fingerprint(parts, version=1)


# ---------------------------------------------------------------------------------------------- policy
#: The ONE semantics this code implements (gnomad-release-listing-v1). Policy values outside it need a reviewed handler.
IMPLEMENTED_RULES = {"grammar": "gnomad-stable-ascii-2-or-3-components-v1", "envelope": "release/NAME/",
                     "components": [2, 3], "missing_patch": 0, "max_prefix_chars": 256, "max_component_digits": 9,
                     "unsupported": "review finding, exit 1, blocks absence claims"}
IMPLEMENTED_ENDPOINT = "https://storage.googleapis.com/storage/v1/b/gcp-public-data--gnomad/o"
IMPLEMENTED_QUERY = {"prefix": "release/", "delimiter": "/", "maxResults": "1000",
                     "fields": "kind,prefixes,nextPageToken"}


@dataclass(frozen=True)
class Policy:
    version: int
    semantics: str
    raw_sha256: object          # SHA-256 of the committed policy file; None for a legacy (version-1) record
    rules_bytes: bytes
    plan_bytes: bytes

    @property
    def rules(self) -> dict:
        return strict_json(self.rules_bytes)

    @property
    def plan(self) -> dict:
        return strict_json(self.plan_bytes)


def parse_policy(raw: bytes) -> Policy:
    d = exact_keys(strict_json(raw), {"schema", "schema_version", "fingerprint_version", "semantics", "encodings",
                                      "release_rules", "request_plan"}, "the interpretation policy")
    if d["schema"] != POLICY_SCHEMA:
        raise ContractError("unknown policy schema {!r}".format(d["schema"]))
    exact_int(d["schema_version"], 1, 1, "the policy schema_version")
    exact_int(d["fingerprint_version"], 2, 2, "the policy fingerprint_version")
    if d["semantics"] != SEMANTICS:
        raise ContractError("unsupported semantics {!r}: needs a reviewed replay handler".format(d["semantics"]))
    if not strict_equal(d["encodings"], ENCODINGS):
        raise ContractError("unsupported codec mapping {!r}".format(d["encodings"]))
    rules = exact_keys(d["release_rules"], IMPLEMENTED_RULES, "release_rules")
    if not strict_equal(rules, IMPLEMENTED_RULES):
        raise ContractError("release_rules are outside the implemented semantics {}".format(SEMANTICS))
    plan = exact_keys(d["request_plan"], {"endpoint", "query", "approved_baseline"}, "request_plan")
    if plan["endpoint"] != IMPLEMENTED_ENDPOINT or not strict_equal(plan["query"], IMPLEMENTED_QUERY):
        raise ContractError("the request plan is outside the implemented semantics {}".format(SEMANTICS))
    if type(plan["approved_baseline"]) is not str or _RELEASE.fullmatch(plan["approved_baseline"]) is None:
        raise ContractError("approved_baseline must be a canonical stable release, got {!r}".format(plan["approved_baseline"]))
    return Policy(2, d["semantics"], hashlib.sha256(raw).hexdigest(), encode(rules, ENCODINGS["release_rules"]),
                  encode(plan, ENCODINGS["request_plan"]))


@dataclass(frozen=True)
class LegacyRecord:
    commit: str
    rules_bytes: bytes
    plan_bytes: bytes


def _legacy(commit: str) -> LegacyRecord:
    plan = {"endpoint": IMPLEMENTED_ENDPOINT, "query": IMPLEMENTED_QUERY, "approved_baseline": "4.1.1"}
    return LegacyRecord(commit, encode(IMPLEMENTED_RULES, COMPACT), encode(plan, SPACED))


#: The AUDITED legacy domain (measured 2026-09-29 from Git): every first-parent main commit whose producer emitted
#: version-1 interpretations -- the change-B merge, the C1 merge and the isolation merge. Request verifier, adapter
#: and lock blobs are identical at all three; run_monitor.py differs only by the github_run block, not the rules.
#: Commits BEFORE 8e7d762 emitted no interpretation (unsupported, not version 1).
LEGACY_RECORDS = (_legacy("8e7d762e5c154a9b8f55cd4d5050404243203e12"),
                  _legacy("f211e1955d0e8f7696f8a9bcf25ad43045d42686"),
                  _legacy("6f37d9b150012aab994ff403f1f0223a87902af7"))


def select_policy(commit: str, read_committed_blob: Callable[[str, str], bytes], *,
                  legacy_records=LEGACY_RECORDS) -> Policy:
    """The policy of an AUTHENTICATED commit (the caller authenticated it). No date or report fallback."""
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        raise ContractError("expected a full 40-character commit identifier, got {!r}".format(commit))
    records = {record.commit: record for record in legacy_records}
    if len(records) != len(legacy_records):
        raise ContractError("duplicate legacy commit")
    try:
        raw = read_committed_blob(commit, CONFIG_PATH)
    except BlobAbsent:
        if commit not in records:
            raise ContractError("{} is absent at {}, which is outside the admitted legacy history".format(
                CONFIG_PATH, commit))
        record = records[commit]
        return Policy(1, SEMANTICS, None, record.rules_bytes, record.plan_bytes)
    if commit in records:
        raise ContractError("legacy history contradicts the authenticated tree: {} is present at {}".format(
            CONFIG_PATH, commit))
    return parse_policy(raw)


# ---------------------------------------------------------------------------------------------- reconstruction, binding
@dataclass(frozen=True)
class Bound:
    version: int
    contract_sha256: object
    ingredients: tuple

    @property
    def parts(self) -> dict:
        return dict(self.ingredients)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.parts, version=self.version, contract_sha256=self.contract_sha256)

    def as_document(self) -> dict:
        d = {"parts": self.parts, "fingerprint": self.fingerprint}
        if self.version == 2:
            d.update(schema_version=2, contract_sha256=self.contract_sha256)
        return d


def reconstruct(policy: Policy, read_file: Callable[[str], bytes], *, approved_record_bytes: bytes,
                approved_release: str) -> Bound:
    """EVERY ingredient from bytes -- never from a report. The approval must already be selected and authorized."""
    if approved_release != policy.plan["approved_baseline"]:
        raise ContractError("the approval's release {!r} differs from the policy's approved_baseline {!r}".format(
            approved_release, policy.plan["approved_baseline"]))
    material = {"approval": approved_record_bytes, "release_rules": policy.rules_bytes,
                "request_plan": policy.plan_bytes}
    material.update((name, read_file(path)) for name, path in spec(policy.version).files)
    empty = sorted(k for k, v in material.items() if type(v) is not bytes or not v)
    if empty:
        raise ContractError("interpretation material is missing or empty: {}".format(empty))
    parts = {name: hashlib.sha256(raw).hexdigest() for name, raw in material.items()}
    validate_parts(parts, policy.version)
    return Bound(policy.version, policy.raw_sha256, tuple(sorted(parts.items())))


def bind_report(document: object, rebuilt: Bound) -> Bound:
    """The report's interpretation object must strict-equal the independent reconstruction as a WHOLE."""
    expected = rebuilt.as_document()
    if rebuilt.version == 1 and type(document) is dict and "schema_version" in document:
        exact_int(document["schema_version"], 1, 1, "a version-1 interpretation's schema_version")
        expected = dict(expected, schema_version=1)
    if type(document) is not dict:
        raise ContractError("the interpretation is not an object")
    if set(document) != set(expected):
        raise ContractError("the interpretation fields {} differ from the version-{} contract {}".format(
            sorted(document), rebuilt.version, sorted(expected)))
    differing = sorted(k for k in expected if not strict_equal(document[k], expected[k]))
    if differing == ["parts"] or "parts" in differing:
        parts = document["parts"] if type(document["parts"]) is dict else {}
        wrong = sorted(set(parts) ^ set(expected["parts"])) or sorted(
            k for k in expected["parts"] if not strict_equal(parts.get(k), expected["parts"][k]))
        raise ContractError("reported parts differ from the independent reconstruction in {}".format(wrong))
    if differing:
        raise ContractError("the interpretation differs from the independent reconstruction in {}".format(differing))
    return rebuilt


def require_producer_agreement(policy: Policy, *, adapter_rules, verifier_rules, adapter_plan, verifier_plan) -> None:
    """Before any network access: BOTH independent implementations must declare exactly the committed policy."""
    for who, got, want in (("adapter rules", adapter_rules, policy.rules), ("verifier rules", verifier_rules, policy.rules),
                           ("adapter plan", adapter_plan, policy.plan), ("verifier plan", verifier_plan, policy.plan)):
        if not strict_equal(got, want):
            raise ContractError("the {} declaration differs from the committed policy".format(who))
