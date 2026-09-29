"""The one interpretation contract (owner rulings 2026-09-28, review revision 3) -- its own guarantees, every refusal
reason PINNED (a wrong reason is a defect, not a pass).

Author: Monzia Moodie
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor import gnomad_release_check as grc
from genomic_variant_classifier.source_monitor import interpretation_contract as ic
from genomic_variant_classifier.source_monitor import report_verifier as rv
from genomic_variant_classifier.source_monitor import request_verifier as rq

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "source_monitor_runs"
POLICY = (ROOT / ic.CONFIG_PATH).read_bytes()
with open(FIX / "run8_commit_blobs.json", encoding="utf-8") as _fh:
    _BLOBS = json.load(_fh)
RUN8 = {p: base64.b64decode(b["base64"]) for p, b in _BLOBS["blobs"].items()}
RUN8_PARTS = {"approval": "b4396470053b3beb7527032de67a197458165cf73700a836908e8e877b35c250",
              "release_rules": "6125fb706c4928f83e938db6fc116de0fde0c82bb8e434bb67a5c8940c373de9", "request_plan": "7d500ab2a75258b75412b1e7d36b192c96bb8abae5108b5ede9bd5ca24ec8ba9"}   # MEASURED from the real run-8 report


# ------------------------------------------------------------------ dependency discipline
def test_the_contract_imports_only_the_standard_library():
    """Review revision 3: a DEPENDENCY-LIGHT registry; no approval, producer or Git code is imported into it."""
    import ast
    tree = ast.parse((ROOT / "src/genomic_variant_classifier/source_monitor/interpretation_contract.py").read_text("utf-8"))
    imported = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported == {"__future__", "hashlib", "json", "re", "dataclasses", "typing"}


# ------------------------------------------------------------------ strict primitives
@pytest.mark.parametrize("raw, reason", [
    (b'{"a": 1.5}', "non-integer JSON number 1.5 is refused"),
    (b'{"a": NaN}', "non-integer JSON number NaN is refused"),
    (b'{"a": 1, "a": 2}', "duplicate JSON key 'a'"),
    (b"\xef\xbb\xbf{}", "a byte-order mark is refused"),
    (b"", "expected 1..65536 JSON bytes"),
    (b'{"a": 9007199254740992}', "a JSON integer must be an integer in [-9007199254740991, 9007199254740991] (never a "
                                  "boolean), got 9007199254740992"),
    (b"[" * 26 + b"]" * 26, "JSON nesting deeper than 24 is refused"),
])
def test_strict_json_refuses_with_its_exact_reason(raw, reason):
    with pytest.raises(ic.ContractError) as exc:
        ic.strict_json(raw)
    assert str(exc.value) == reason


def test_strict_equal_separates_booleans_integers_and_floats_even_when_nested():
    assert not ic.strict_equal({"n": 1}, {"n": True}) and not ic.strict_equal([True], [1])
    assert not ic.strict_equal({"x": {"y": 0}}, {"x": {"y": False}}) and ic.strict_equal({"x": [1, "a"]}, {"x": [1, "a"]})


def test_the_named_codecs_reproduce_the_historical_bytes_and_differ_from_each_other():
    value = {"b": {"y": "1", "x": [2, 3]}, "a": "z"}
    assert ic.encode(value, ic.COMPACT) == b'{"a":"z","b":{"x":[2,3],"y":"1"}}'
    assert ic.encode(value, ic.SPACED) == json.dumps(value, sort_keys=True).encode("ascii")    # Python's default
    with pytest.raises(ic.ContractError, match=r"^unsupported codec 'rfc8785'$"):
        ic.encode(value, "rfc8785")


# ------------------------------------------------------------------ fingerprint protocol
def test_versions_are_exact_integers_and_booleans_are_refused():
    for bad in (True, 0, 3, "2", 2.0):
        with pytest.raises(ic.ContractError, match=r"^the fingerprint version must be an integer in \[1, 2\]"):
            ic.spec(bad)


def test_version_2_is_version_1_plus_the_orchestrator():
    assert ic.spec(2).names - ic.spec(1).names == {"orchestrator_code"}
    assert dict(ic.spec(2).files)["orchestrator_code"] == "src/genomic_variant_classifier/source_monitor/run_monitor.py"


def test_the_fingerprint_is_order_invariant_and_version_1_refuses_a_policy_digest():
    parts = {k: hashlib.sha256(k.encode()).hexdigest() for k in ic.spec(1).names}
    assert ic.fingerprint(parts, version=1) == ic.fingerprint(dict(reversed(list(parts.items()))), version=1)
    with pytest.raises(ic.ContractError, match=r"^a version-1 fingerprint cannot include a configuration digest$"):
        ic.fingerprint(parts, version=1, contract_sha256="0" * 64)


def test_the_version_2_envelope_binds_the_policy_digest():
    parts = {k: hashlib.sha256(k.encode()).hexdigest() for k in ic.spec(2).names}
    assert ic.fingerprint(parts, version=2, contract_sha256="a" * 64) != ic.fingerprint(parts, version=2,
                                                                                        contract_sha256="b" * 64)
    with pytest.raises(ic.ContractError, match=r"^contract_sha256 must be a complete lowercase SHA-256 digest, got None$"):
        ic.fingerprint(parts, version=2)


# ------------------------------------------------------------------ the committed policy
def test_the_committed_policy_is_the_rulings_file_and_parses_as_version_2():
    # The owner's review-revision-3 file, byte-for-byte (full digest MEASURED 2026-09-29).
    assert hashlib.sha256(POLICY).hexdigest() == "7c05e8028bef55e690ef31a6c49b8988d7848f64f6c5f08a562336c0ff8031c3"
    policy = ic.parse_policy(POLICY)
    assert (policy.version, policy.semantics, policy.raw_sha256) == (2, ic.SEMANTICS, hashlib.sha256(POLICY).hexdigest())


@pytest.mark.parametrize("change, reason", [
    (lambda d: d.update(schema="other"), "unknown policy schema 'other'"),
    (lambda d: d.update(fingerprint_version=True), "the policy fingerprint_version must be an integer in [2, 2] (never a "
                                                   "boolean), got True"),
    (lambda d: d.update(semantics="x"), "unsupported semantics 'x': needs a reviewed replay handler"),
    (lambda d: d["encodings"].update(release_rules="json-ascii-spaced-v1"),
     "unsupported codec mapping {'release_rules': 'json-ascii-spaced-v1', 'request_plan': 'json-ascii-spaced-v1'}"),
    (lambda d: d["release_rules"].update(max_component_digits=10),
     "release_rules are outside the implemented semantics gnomad-release-listing-v1"),
    (lambda d: d["request_plan"].update(approved_baseline="4.1"), None),          # a canonical 2-component release: allowed
    (lambda d: d["request_plan"].update(approved_baseline="04.1.1"),
     "approved_baseline must be a canonical stable release, got '04.1.1'"),
    (lambda d: d.update(extra=1), "the interpretation policy must have exactly ['encodings', 'fingerprint_version', "
                                  "'release_rules', 'request_plan', 'schema', 'schema_version', 'semantics'], got "
                                  "['encodings', 'extra', 'fingerprint_version', 'release_rules', 'request_plan', 'schema', "
                                  "'schema_version', 'semantics']"),
])
def test_a_policy_outside_the_contract_is_refused_with_its_exact_reason(change, reason):
    doc = json.loads(POLICY)
    change(doc)
    raw = json.dumps(doc).encode("ascii")
    if reason is None:
        assert ic.parse_policy(raw).plan["approved_baseline"] == "4.1"
        return
    with pytest.raises(ic.ContractError) as exc:
        ic.parse_policy(raw)
    assert str(exc.value) == reason


# ------------------------------------------------------------------ legacy domain and selection
def test_every_legacy_record_carries_run_8s_exact_historical_bytes():
    """Measured 2026-09-29: the rules/plan inputs are identical at all three legacy commits."""
    assert [r.commit[:7] for r in ic.LEGACY_RECORDS] == ["8e7d762", "f211e19", "6f37d9b"]
    for record in ic.LEGACY_RECORDS:
        assert hashlib.sha256(record.rules_bytes).hexdigest() == RUN8_PARTS["release_rules"]
        assert hashlib.sha256(record.plan_bytes).hexdigest() == RUN8_PARTS["request_plan"]


def _reader(files):
    def read(commit, path):
        if path not in files:
            raise ic.BlobAbsent(path)
        return files[path]
    return read


def test_selection_admits_version_1_only_for_legacy_and_refuses_contradictions():
    legacy = ic.LEGACY_RECORDS[0].commit
    assert ic.select_policy(legacy, _reader({})).version == 1
    with pytest.raises(ic.ContractError) as exc:
        ic.select_policy("1" * 40, _reader({}))
    assert str(exc.value) == ("configs/source_monitor_interpretation.json is absent at {}, which is outside the admitted "
                              "legacy history".format("1" * 40))
    with pytest.raises(ic.ContractError) as exc:
        ic.select_policy(legacy, _reader({ic.CONFIG_PATH: POLICY}))
    assert str(exc.value) == ("legacy history contradicts the authenticated tree: configs/source_monitor_interpretation."
                              "json is present at {}".format(legacy))
    with pytest.raises(ic.ContractError, match=r"^expected a full 40-character commit identifier, got '8e7d762'$"):
        ic.select_policy("8e7d762", _reader({}))


def test_a_reader_failure_that_is_not_confirmed_absence_propagates_never_downgrades():
    def broken(commit, path):
        raise OSError("disk error")
    with pytest.raises(OSError, match="disk error"):
        ic.select_policy(ic.LEGACY_RECORDS[0].commit, broken)


# ------------------------------------------------------------------ reconstruction, binding, agreement
def test_run_8_reconstructs_to_its_real_fingerprint_and_binds_as_a_whole():
    policy = ic.select_policy(_BLOBS["commit"], _reader(RUN8))
    bound = ic.reconstruct(policy, lambda p: RUN8[p], approved_record_bytes=RUN8[
        "docs/approvals/APPROVAL_2026-09-24_gnomad-4.1.1.json"], approved_release="4.1.1")
    assert bound.fingerprint == "4a39c1620367495fd98e8076be1c2527b755a05b43e8fe5dd82902deee15fbe5"   # the REAL run-8 report's
    assert bound.parts["approval"] == RUN8_PARTS["approval"]
    assert ic.bind_report(bound.as_document(), bound) is bound


def test_an_approval_that_disagrees_with_the_declared_baseline_is_refused():
    policy = ic.parse_policy(POLICY)
    with pytest.raises(ic.ContractError) as exc:
        ic.reconstruct(policy, lambda p: b"x", approved_record_bytes=b"{}", approved_release="4.1.2")
    assert str(exc.value) == "the approval's release '4.1.2' differs from the policy's approved_baseline '4.1.1'"


def test_both_independent_declarations_must_equal_the_committed_policy():
    policy = ic.parse_policy(POLICY)
    adapter, verifier = grc.declaration(ROOT), rq.declaration()
    ic.require_producer_agreement(policy, adapter_rules=adapter["release_rules"], verifier_rules=verifier["release_rules"],
                                  adapter_plan=adapter["request_plan"], verifier_plan=verifier["request_plan"])
    drifted = dict(verifier["release_rules"], max_component_digits=True)
    with pytest.raises(ic.ContractError, match=r"^the verifier rules declaration differs from the committed policy$"):
        ic.require_producer_agreement(policy, adapter_rules=adapter["release_rules"], verifier_rules=drifted,
                                      adapter_plan=adapter["request_plan"], verifier_plan=verifier["request_plan"])


@pytest.mark.parametrize("classify", [grc.classify_prefix, rq.independent_classify], ids=["adapter", "verifier"])
def test_the_declared_component_digit_limit_is_what_each_classifier_does(classify):
    assert grc.MAX_COMPONENT_DIGITS == rq.MAX_COMPONENT_DIGITS == 9
    assert classify("release/4.1.123456789/") == ("stable", (4, 1, 123456789))
    assert classify("release/4.1.1234567890/") == ("unsupported", None)


# ------------------------------------------------------------------ commit_blob_reader against REAL Git
@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.fail("git is required for the commit-reader tests")
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True).stdout
    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    (tmp_path / "present.json").write_bytes(b"{}")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.txt").write_bytes(b"x")
    git("add", ".")
    git("commit", "-q", "-m", "c")
    return tmp_path, git("rev-parse", "HEAD").decode().strip()


def test_the_commit_reader_signals_absence_ONLY_for_an_empty_tree_listing(repo):
    from genomic_variant_classifier.data import release_approval as ra
    path, commit = repo
    read = rv.commit_blob_reader(ra.git_reader(str(path)))
    assert read(commit, "present.json") == b"{}"
    with pytest.raises(ic.BlobAbsent):
        read(commit, "configs/source_monitor_interpretation.json")
    for not_absent in (lambda: read(commit, "sub"), lambda: read("0" * 40, "present.json")):
        with pytest.raises(Exception) as exc:
            not_absent()
        assert not isinstance(exc.value, ic.BlobAbsent), type(exc.value).__name__
