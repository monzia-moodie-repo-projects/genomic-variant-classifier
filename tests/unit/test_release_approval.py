"""Release approval: an immutable record SELECTED by the manifest (change A, 2026-09-26).

Ports the owner's reference kernels' cases (approval_contract, git_evidence) to the project
module, and adds what the reference packages could not: REAL Git objects of this repository,
the real record and manifest, the verifier's runtime pin, and the append-only snapshot in
temporary repositories. Real-Git tests need Git 2.45+ (--no-lazy-fetch) and skip ONLY when
this tree is not a Git working tree (the convention of test_gitattributes_contract.py).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from genomic_variant_classifier.data import release_approval as ra
from genomic_variant_classifier.data.source_registry import SourceRegistry

_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = dict(role="historical_approval_record", kind="git_blob",
                 repository="monzia-moodie-repo-projects/genomic-variant-classifier",
                 commit="38987f54f88e13c4bdd394eb9bb4eb2b192ccdb6",
                 path="docs/measurements/DECISION_2026-09-24_gnomad-4.1.1-approval.md",
                 git_blob_oid="67f2a773fbdf3b38c22cec3cfd3b4dcbda2082a0",
                 sha256="f59f0355a1057accb54614ee273099b48bf9601a579c12037d1dc2b886234d17",
                 bytes=7372)
_PATH = "docs/approvals/EXAMPLE_approval.json"
_IS_GIT = (_ROOT / ".git").exists()


def record(**overrides) -> bytes:
    d = {"schema_version": 1, "target": "gnomad-public-releases", "approved_release": "4.1.1",
         "scope": "release_monitoring_baseline", "recorded_at_utc": "2026-09-25T12:00:00Z",
         "approval_documented_on": "2026-09-24", "approval_granted_at_utc": None,
         "evidence": [dict(_EVIDENCE)], "supersedes_sha256": None}
    d.update(overrides)
    return (json.dumps(d, sort_keys=True, indent=2) + "\n").encode("utf-8")


def load(raw: bytes, target: str = "gnomad-public-releases") -> ra.Approval:
    return ra.load_approval(target, _PATH, ra.sha256(raw), {_PATH: raw}.__getitem__)


# ------------------------------------------------------------- 1. approval contract
def test_a_valid_record_with_an_unknown_grant_date():
    a = load(record())
    assert (a.approved_release, a.scope, a.approval_granted_at_utc) == ("4.1.1", "release_monitoring_baseline", None)
    assert a.evidence[0].role == "historical_approval_record"


def test_the_record_is_read_once_and_hashed_and_parsed_from_the_same_bytes():
    calls = []
    def read(path):
        calls.append(path)
        return record()
    ra.load_approval("gnomad-public-releases", _PATH, ra.sha256(record()), read)
    assert calls == [_PATH]


def test_changed_bytes_including_crlf_are_refused():
    with pytest.raises(ra.PolicyError, match="digest mismatch"):
        ra.load_approval("gnomad-public-releases", _PATH, ra.sha256(record()),
                         lambda _: record().replace(b"\n", b"\r\n"))


@pytest.mark.parametrize("path", ["../other.json", "/docs/approvals/a.json", "docs/approvals/../a.json",
                                  "docs\\approvals\\a.json", "https://example.org/a.json",
                                  "docs/approvals/sub/a.json", "docs/approvals/a.yaml"])
def test_record_paths_outside_the_direct_directory_are_refused(path):
    with pytest.raises(ra.PolicyError, match="direct"):
        ra.load_approval("gnomad-public-releases", path, ra.sha256(record()), lambda _: record())


def test_short_digest_missing_record_and_wrong_target_are_refused():
    with pytest.raises(ra.PolicyError, match="complete lowercase"):
        ra.load_approval("gnomad-public-releases", _PATH, ra.sha256(record())[:16], lambda _: record())
    with pytest.raises(ra.PolicyError, match="unavailable"):
        ra.load_approval("gnomad-public-releases", _PATH, ra.sha256(record()), {}.__getitem__)
    with pytest.raises(ra.PolicyError, match="target mismatch"):
        load(record(target="another-product"))


@pytest.mark.parametrize("value", [True, "1", 2])
def test_an_unknown_schema_is_refused(value):
    with pytest.raises(ra.PolicyError, match="schema"):
        ra.parse_approval(record(schema_version=value))


def test_a_scope_beyond_release_monitoring_is_refused():
    with pytest.raises(ra.PolicyError, match="scope"):
        ra.parse_approval(record(scope="production_adoption"))


@pytest.mark.parametrize("kind", ["missing", "unknown"])
def test_missing_and_unknown_record_keys_are_refused(kind):
    d = json.loads(record())
    if kind == "missing":
        del d["approval_granted_at_utc"]
    else:
        d["adopted"] = True
    with pytest.raises(ra.PolicyError, match="missing="):
        ra.parse_approval(json.dumps(d).encode())


@pytest.mark.parametrize("raw", [b"", b"x" * 65537, b"\xff", b"\xef\xbb\xbf" + b"{}",
                                 b'{"x": NaN}', b'{"x": Infinity}',
                                 b'{"schema_version": 1, "schema_version": 1}'],
                         ids=["empty", "oversize", "not-utf8", "bom", "nan", "infinity", "duplicate-key"])
def test_malformed_record_bytes_are_refused(raw):
    with pytest.raises(ra.PolicyError):
        ra.parse_approval(raw)


@pytest.mark.parametrize("value", ["2026-09-25T12:00:00", "2026-02-31T12:00:00Z", "yesterday"])
def test_a_bad_creation_time_is_refused(value):
    with pytest.raises(ra.PolicyError):
        ra.parse_approval(record(recorded_at_utc=value))


def test_dates_must_be_consistent_and_a_known_grant_is_accepted():
    with pytest.raises(ra.PolicyError, match="after record creation"):
        ra.parse_approval(record(approval_documented_on="2026-09-26"))
    with pytest.raises(ra.PolicyError, match="grant date contradicts"):
        ra.parse_approval(record(approval_granted_at_utc="2026-09-25T10:00:00Z"))
    assert ra.parse_approval(record(approval_granted_at_utc="2026-09-22T10:00:00Z")).approval_granted_at_utc == "2026-09-22T10:00:00Z"


@pytest.mark.parametrize("evidence", [
    [], "text", [dict(_EVIDENCE, sha256="123")], [dict(_EVIDENCE)] * 2,
    [dict(_EVIDENCE, role="owner_authorization")], [dict(_EVIDENCE, fallback="main")],
    [{k: v for k, v in _EVIDENCE.items() if k != "role"}]],
    ids=["empty", "text", "short-sha", "duplicate", "unknown-role", "extra-field", "no-role"])
def test_evidence_must_be_nonempty_typed_unique_and_complete(evidence):
    with pytest.raises(ra.PolicyError):
        ra.parse_approval(record(evidence=evidence))


def test_the_verifier_pin_is_checked_at_runtime_on_every_field():
    a = load(record())
    pin = {"target": a.target, "approved_release": a.approved_release, "record_sha256": a.record_sha256}
    ra.require_verifier_pin(a, **pin)
    for field, value in (("target", "other"), ("approved_release", "4.1"), ("record_sha256", ra.sha256(b"x"))):
        with pytest.raises(ra.PolicyError, match="disagree"):
            ra.require_verifier_pin(a, **dict(pin, **{field: value}))


def test_append_only_allows_additions_and_refuses_edit_removal_and_rename():
    ra.require_append_only({_PATH: record()}, {_PATH: record(), "docs/approvals/new.json": b"new"})
    for after in ({}, {_PATH: record() + b"\n"}, {"docs/approvals/renamed.json": record()}):
        with pytest.raises(ra.PolicyError, match="modified or removed"):
            ra.require_append_only({_PATH: record()}, after)


def test_a_successor_must_link_keep_target_and_scope_and_come_later():
    a = load(record())
    ra.require_successor(a, a)
    good = dict(supersedes_sha256=a.record_sha256, recorded_at_utc="2026-09-26T12:00:00Z")
    ra.require_successor(a, load(record(**good)))
    for override in ({"supersedes_sha256": None}, {"target": "other"}, {"recorded_at_utc": a.recorded_at_utc}):
        with pytest.raises(ra.PolicyError):
            ra.require_successor(a, ra.parse_approval(record(**(good | override))))


def test_the_interpretation_fingerprint_is_order_invariant_and_dependency_sensitive():
    parts = {n: ra.sha256(n.encode()) for n in ("approval", "release_rules", "request_plan",
                                                "adapter_code", "verifier_code", "environment_lock")}
    first = ra.interpretation_fingerprint(parts)
    assert first == ra.interpretation_fingerprint(dict(reversed(list(parts.items()))))
    for field in parts:
        assert first != ra.interpretation_fingerprint(parts | {field: ra.sha256(b"new")})
    with pytest.raises(ra.PolicyError, match="every named"):
        ra.interpretation_fingerprint({"approval": ra.sha256(b"x")})


# ------------------------------------------------------------- 2. evidence descriptor
def _ref(**overrides) -> ra.GitEvidence:
    return ra.GitEvidence(**dict(_EVIDENCE, **overrides))


@pytest.mark.parametrize("oid", ["main", "HEAD", "a" * 8, "a" * 39, "g" * 40])
def test_no_branch_or_abbreviated_object_identifier(oid):
    with pytest.raises(ra.PolicyError, match="full Git"):
        _ref(commit=oid)


def test_object_formats_must_match_and_digests_must_be_complete():
    with pytest.raises(ra.PolicyError, match="formats differ"):
        _ref(git_blob_oid="b" * 64)
    with pytest.raises(ra.PolicyError, match="complete lowercase"):
        _ref(sha256=_EVIDENCE["sha256"][:16])


@pytest.mark.parametrize("path", ["../x", "docs/../x", "/tmp/x", "docs//x", "docs\\x", "docs/",
                                  ":(glob)*", "docs/x\n", "docs/\ud800"])
def test_only_literal_utf8_repository_paths(path):
    with pytest.raises(ra.PolicyError):
        _ref(path=path)


@pytest.mark.parametrize("size", [True, 0, -1, "7372", 1048577])
def test_evidence_size_type_and_limit(size):
    with pytest.raises(ra.PolicyError, match="bytes"):
        _ref(bytes=size)


@pytest.mark.parametrize("change", [{"kind": "mutable_url"}, {"repository": "https://example.org"}])
def test_unknown_kind_and_repository_form(change):
    with pytest.raises(ra.PolicyError):
        _ref(**change)


def _synthetic(raw=b"Public decision.\n", n=40):
    framed = b"blob " + str(len(raw)).encode() + b"\0" + raw
    alg = hashlib.sha1 if n == 40 else hashlib.sha256
    return _ref(commit="a" * n, path="docs/decision.md", git_blob_oid=alg(framed).hexdigest(),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)), raw


def test_byte_verification_both_object_formats_and_its_three_refusals():
    for n in (40, 64):
        ref, raw = _synthetic(n=n)
        ra.verify_evidence_bytes(ref, raw)
    ref, raw = _synthetic()
    with pytest.raises(ra.PolicyError, match="byte count"):
        ra.verify_evidence_bytes(ref, raw.replace(b"\n", b"\r\n"))
    with pytest.raises(ra.PolicyError, match="SHA-256 differs"):
        ra.verify_evidence_bytes(ref, b"X" + raw[1:])
    with pytest.raises(ra.PolicyError, match="Git blob identifier"):
        ra.verify_evidence_bytes(replace(ref, git_blob_oid="b" * 40), raw)


# ------------------------------------------------------------- 3. reader contract (scripted Git)
class FakeGit:
    def __init__(self, ref, raw):
        self.ref, self.raw, self.calls = ref, raw, []
        self.kind, self.mode, self.blob_kind = b"commit\n", b"100644", b"blob"
        self.path, self.oid, self.size, self.extra = ref.path.encode(), ref.git_blob_oid.encode(), str(len(raw)).encode(), b""

    def __call__(self, *args):
        self.calls.append(args)
        if args == ("cat-file", "-t", self.ref.commit):
            return self.kind
        if args == ("ls-tree", "-z", self.ref.commit, "--", self.ref.path):
            return self.mode + b" " + self.blob_kind + b" " + self.oid + b"\t" + self.path + b"\0" + self.extra
        if args == ("cat-file", "-s", self.oid.decode()):
            return self.size
        if args == ("cat-file", "blob", self.oid.decode()):
            return self.raw
        raise AssertionError(f"unexpected Git command: {args!r}")


def test_membership_then_size_then_exact_bytes_in_four_calls():
    ref, raw = _synthetic(); git = FakeGit(ref, raw)
    assert ra.verify_git_evidence(ref, git) == raw
    assert [c[:2] for c in git.calls] == [("cat-file", "-t"), ("ls-tree", "-z"), ("cat-file", "-s"), ("cat-file", "blob")]


def test_an_executable_regular_file_is_allowed():
    ref, raw = _synthetic(); git = FakeGit(ref, raw); git.mode = b"100755"
    assert ra.verify_git_evidence(ref, git) == raw


@pytest.mark.parametrize("attr,value,message", [
    ("kind", b"tag\n", "not a commit"), ("extra", b"other\0", "exactly one"),
    ("path", b"different.md", "names a different path"), ("oid", b"b" * 40, "pinned blob")])
def test_membership_failures_are_refused(attr, value, message):
    ref, raw = _synthetic(); git = FakeGit(ref, raw); setattr(git, attr, value)
    with pytest.raises(ra.PolicyError, match=message):
        ra.verify_git_evidence(ref, git)


@pytest.mark.parametrize("mode,kind", [(b"120000", b"blob"), (b"160000", b"commit")], ids=["symlink", "submodule"])
def test_a_symlink_or_submodule_is_refused(mode, kind):
    ref, raw = _synthetic(); git = FakeGit(ref, raw); git.mode, git.blob_kind = mode, kind
    with pytest.raises(ra.PolicyError, match="regular file"):
        ra.verify_git_evidence(ref, git)


@pytest.mark.parametrize("size", [b"999999999", b"invalid"])
def test_the_size_is_checked_BEFORE_the_contents_are_read(size):
    ref, raw = _synthetic(); git = FakeGit(ref, raw); git.size = size
    with pytest.raises(ra.PolicyError):
        ra.verify_git_evidence(ref, git)
    assert not any(c[:2] == ("cat-file", "blob") for c in git.calls)


def test_a_record_over_its_limit_is_refused_before_reading():
    ref, raw = _synthetic(); git = FakeGit(ref, raw); git.size = str(ra.MAX_RECORD_BYTES + 1).encode()
    with pytest.raises(ra.PolicyError, match="exceeds"):
        ra.read_blob_at(git, ref.commit, ref.path, max_size=ra.MAX_RECORD_BYTES)
    assert not any(c[:2] == ("cat-file", "blob") for c in git.calls)


def test_the_subprocess_contract_without_invoking_git(tmp_path):
    with patch("genomic_variant_classifier.data.release_approval.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, stdout=b"raw\r\n", stderr=b"")
        assert ra.git_reader(tmp_path)("cat-file", "-t", "a" * 40) == b"raw\r\n"
        args, kwargs = run.call_args
        for flag in ("--no-replace-objects", "--no-lazy-fetch", "--literal-pathspecs"):
            assert flag in args[0]
        assert not kwargs.get("shell", False) and not kwargs.get("text", False)
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        run.return_value = subprocess.CompletedProcess([], 128, stdout=b"", stderr=b"absent")
        with pytest.raises(ra.PolicyError, match="exit 128"):
            ra.git_reader(tmp_path)("cat-file", "-t", "a" * 40)


# ------------------------------------------------------------- 4. the real record, manifest and objects
_needs_git = pytest.mark.skipif(not _IS_GIT, reason=f"{_ROOT} is not a Git working tree; objects cannot be read")


def _active():
    ptr = SourceRegistry.load(_ROOT / "configs/data_manifest.yaml").approval_pointer("gnomad-public-releases")
    return ptr, ra.load_approval(ptr.target, ptr.record, ptr.sha256, ra.worktree_record_reader(_ROOT))


def test_the_manifest_selects_the_real_4_1_1_approval_with_exact_facts():
    ptr, a = _active()
    assert (a.target, a.approved_release, a.scope) == ("gnomad-public-releases", "4.1.1", "release_monitoring_baseline")
    assert (a.approval_documented_on, a.approval_granted_at_utc, a.supersedes_sha256) == ("2026-09-24", None, None)
    assert [asdict(e) for e in a.evidence] == [_EVIDENCE]
    assert a.record_sha256 == ptr.sha256


def test_the_verifier_pin_agrees_with_the_manifest_selected_approval_at_runtime():
    from genomic_variant_classifier.source_monitor import gnomad_release_check, request_verifier as rv
    _, a = _active()
    ra.require_verifier_pin(a, target=rv.APPROVAL_TARGET, approved_release=rv.APPROVED_BASELINE,
                            record_sha256=rv.APPROVED_RECORD_SHA256)
    assert gnomad_release_check.APPROVED_BASELINE == a.approved_release


@_needs_git
def test_the_real_evidence_verifies_from_this_repositorys_git_objects():
    _, a = _active()
    raw = ra.verify_git_evidence(a.evidence[0], ra.git_reader(_ROOT))
    assert len(raw) == 7372


@_needs_git
def test_a_falsified_reference_is_refused_by_real_git():
    git = ra.git_reader(_ROOT)
    readme_oid = git("rev-parse", _EVIDENCE["commit"] + ":README.md").strip().decode()
    with pytest.raises(ra.PolicyError, match="pinned blob"):
        ra.verify_git_evidence(_ref(git_blob_oid=readme_oid), git)
    with pytest.raises(ra.PolicyError, match="exactly one tree entry"):
        ra.verify_git_evidence(_ref(path=_EVIDENCE["path"].replace(".md", "X.md")), git)


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs elevated rights on Windows")
def test_the_worktree_reader_refuses_a_symlinked_record(tmp_path):
    (tmp_path / "docs/approvals").mkdir(parents=True)
    (tmp_path / "real.json").write_bytes(record())
    (tmp_path / _PATH).symlink_to(tmp_path / "real.json")
    with pytest.raises(ra.PolicyError, match="not a regular file"):
        ra.worktree_record_reader(tmp_path)(_PATH)


# ------------------------------------------------------------- 5. append-only snapshot, temporary repositories
def _script():
    spec = importlib.util.spec_from_file_location("check_release_approvals", _ROOT / "scripts/check_release_approvals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@x")
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, env=env).stdout
    git("init", "-q")
    (tmp_path / "docs/approvals").mkdir(parents=True)
    (tmp_path / "docs/approvals/A.json").write_bytes(b"first\n")
    git("add", "-A"); git("commit", "-q", "-m", "base")
    return git, git("rev-parse", "HEAD").strip().decode()


def _commit(git, tmp_path, message):
    git("add", "-A"); git("commit", "-q", "-m", message)
    return git("rev-parse", "HEAD").strip().decode()


@pytest.mark.parametrize("change", ["edit", "delete", "rename"])
def test_the_script_snapshot_refuses_edit_deletion_and_rename(tmp_path, change):
    git, base = _repo(tmp_path)
    a = tmp_path / "docs/approvals/A.json"
    if change == "edit":
        a.write_bytes(b"first, edited\n")
    elif change == "delete":
        a.unlink(); (tmp_path / "docs/approvals/keep.txt").write_bytes(b"x")
    else:
        a.rename(tmp_path / "docs/approvals/B.json")
    head = _commit(git, tmp_path, change)
    s, reader = _script(), ra.git_reader(tmp_path)
    with pytest.raises(ra.PolicyError, match="modified or removed"):
        ra.require_append_only(s._approvals_at(reader, base), s._approvals_at(reader, head))


def test_the_script_snapshot_allows_an_addition(tmp_path):
    git, base = _repo(tmp_path)
    (tmp_path / "docs/approvals/B.json").write_bytes(b"second\n")
    head = _commit(git, tmp_path, "add")
    s, reader = _script(), ra.git_reader(tmp_path)
    before, after = s._approvals_at(reader, base), s._approvals_at(reader, head)
    ra.require_append_only(before, after)
    assert sorted(after) == ["docs/approvals/A.json", "docs/approvals/B.json"]


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs elevated rights on Windows")
def test_the_script_snapshot_refuses_a_non_regular_entry(tmp_path):
    git, _ = _repo(tmp_path)
    (tmp_path / "docs/approvals/L.json").symlink_to("A.json")
    head = _commit(git, tmp_path, "symlink")
    with pytest.raises(ra.PolicyError, match="not a regular file"):
        _script()._approvals_at(ra.git_reader(tmp_path), head)


# ------------------------------------------------------------- 6. the CI step
def test_ci_verifies_approvals_before_the_suite_with_explicit_fetches():
    import yaml
    with open(_ROOT / ".github/workflows/ci.yml", encoding="utf-8") as fh:
        steps = yaml.safe_load(fh)["jobs"]["test"]["steps"]
    names = [s.get("name") for s in steps]
    i = names.index("Verify release approvals (append-only history, pinned evidence)")
    assert names[i + 1] == "Run the test suite"
    run = steps[i]["run"]
    for fragment in ("git --version", "--list-evidence-commits", "git fetch --no-tags --depth=1 origin $(printf '%s\\n' $base $evidence | sort -u)",
                     'check_release_approvals.py --base "$base"', "NOT evaluated"):
        assert fragment in run, fragment
