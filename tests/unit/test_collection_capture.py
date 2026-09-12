"""Evidence that was not retained cannot be re-examined.

MEASURED 2026-09-11 against the installer at
66772b3a9de7779087c2380e865194489c91a8644427ebfbfb35bd4a2e96bd7a: collection
runs with `capture_output=True`, the bytes exist, and they are discarded when
the subprocess result leaves scope. Three call sites -- baseline, prospective
inside rollback, apply recheck -- and `run_gate` has the same shape.

The consequence is already on the record: the unit-O interpretation migration
had to be RECONSTRUCTED from ten identities recorded in prose, because the raw
collection output from f808944 was never kept.

Every control below runs a REAL subprocess. A capture component tested only
against synthetic byte strings would not establish that it captures anything.

Author: Monzia Moodie
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.operations.collection_capture import (
    ArtifactTooLarge,
    ArtifactUnavailable,
    CaptureError,
    DirectoryReader,
    CollectionPhase,
    DirectorySink,
    RetentionFailed,
    Subject,
    SubjectKind,
    capture_collection,
    decode_retained,
)

COMMIT = "a" * 40


def emit(stdout: bytes = b"", stderr: bytes = b"", code: int = 0):
    """An argv that writes exactly the given BYTES and exits with `code`.

    Writes through `sys.stdout.buffer`, NOT `sys.stdout`.

    MEASURED 2026-09-11 at the acceptance gate on Windows: the first version
    used `sys.stdout.write("...\\n")`, and four controls failed --

        assert b't.py::test_a\\r\\n' == b't.py::test_a\\n'

    -- because Python's stdout is a TEXT stream and text mode translates "\\n"
    to "\\r\\n" on write. THE COMPONENT WAS CORRECT: it retained exactly the
    bytes the child produced, carriage return included. The fixture demanded
    POSIX bytes from a Windows child.

    That is also the property the capture exists to preserve. Had it
    normalised, the failure would never have appeared and the retained
    evidence would have differed from what was observed.

    The buffer form emits precisely the bytes named on every platform, which
    removes the dependence from the FIXTURE rather than accommodating it in
    the assertions.
    """
    return [sys.executable, "-c",
            "import sys;sys.stdout.buffer.write({!r});"
            "sys.stderr.buffer.write({!r});sys.exit({})".format(
                stdout, stderr, code)]


@pytest.fixture
def sink(tmp_path):
    return DirectorySink(tmp_path / "evidence")


def committed():
    return Subject(kind=SubjectKind.COMMITTED_TREE, commit=COMMIT)


# ---------------------------------------------------------------------------
# 1. The bytes are retained, and retention is verified rather than assumed
# ---------------------------------------------------------------------------

def test_the_exact_stdout_bytes_reach_disk(tmp_path, sink):
    document, out = capture_collection(
        argv=emit(stdout=b"t.py::test_a\n"), cwd=tmp_path, env=None, timeout=60,
        phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    assert out == b"t.py::test_a\n"
    retained = (tmp_path / "evidence" / document["retained"]["stdout_path"])
    assert retained.read_bytes() == out
    assert document["stdout"]["sha256"] == hashlib.sha256(out).hexdigest()


def test_stderr_is_retained_separately_from_stdout(tmp_path, sink):
    """Separate pipes do not preserve interleaving, so concatenating them and
    claiming they do would be a fabricated ordering."""
    document, _ = capture_collection(
        argv=emit(stdout=b"out\n", stderr=b"err\n"), cwd=tmp_path, env=None,
        timeout=60, phase=CollectionPhase.BASELINE_COLLECTION,
        subject=committed(), interpreter=sys.executable, sink=sink)
    names = document["retained"]
    assert names["stdout_path"] != names["stderr_path"]
    directory = tmp_path / "evidence"
    assert (directory / names["stdout_path"]).read_bytes() == b"out\n"
    assert (directory / names["stderr_path"]).read_bytes() == b"err\n"


def test_a_sink_that_cannot_retain_raises_rather_than_returning(tmp_path):
    """A successful collection whose evidence could not be stored is not an
    acceptable input to acceptance."""

    class Broken(DirectorySink):
        def publish_completed(self, record, stdout, stderr):
            raise RetentionFailed("simulated storage failure")

    with pytest.raises(RetentionFailed):
        capture_collection(
            argv=emit(stdout=b"x\n"), cwd=tmp_path, env=None, timeout=60,
            phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
            interpreter=sys.executable, sink=Broken(tmp_path / "nowhere"))


# ---------------------------------------------------------------------------
# 2. Retention precedes interpretation and the exit-status refusal
# ---------------------------------------------------------------------------

def test_a_failing_collection_still_retains_its_evidence(tmp_path, sink):
    """MEASURED: the installed collect_output prints the tail of STDOUT on a
    nonzero exit and discards STDERR -- where a collection error is written.
    The refusal shows a truncated listing while the cause goes unread."""
    document, _ = capture_collection(
        argv=emit(stderr=b"collection error\n", code=2), cwd=tmp_path, env=None,
        timeout=60, phase=CollectionPhase.APPLY_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    assert document["returncode"] == 2
    cause = tmp_path / "evidence" / document["retained"]["stderr_path"]
    assert cause.read_bytes() == b"collection error\n"


def test_capture_does_not_judge_the_exit_status(tmp_path, sink):
    """A nonzero exit is when the evidence matters most, so this boundary
    records it and leaves the refusal to the caller."""
    document, _ = capture_collection(
        argv=emit(code=3), cwd=tmp_path, env=None, timeout=60,
        phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    assert document["returncode"] == 3


# ---------------------------------------------------------------------------
# 3. The subject cannot be misdescribed by omission
# ---------------------------------------------------------------------------

def test_a_transaction_state_must_carry_a_state_reference():
    """Two of the installer's three collections run over a MODIFIED working
    tree. The predecessor commit does not describe those files."""
    with pytest.raises(CaptureError, match="state_reference"):
        Subject(kind=SubjectKind.MEASURED_TRANSACTION_STATE)


def test_a_committed_tree_may_not_also_carry_a_state_reference():
    with pytest.raises(CaptureError, match="may not also carry"):
        Subject(kind=SubjectKind.COMMITTED_TREE, commit=COMMIT,
                state_reference="a measurement")


def test_a_committed_tree_must_name_its_commit():
    with pytest.raises(CaptureError, match="must name its commit"):
        Subject(kind=SubjectKind.COMMITTED_TREE)


def test_a_transaction_state_subject_is_accepted_with_its_measurement():
    subject = Subject(kind=SubjectKind.MEASURED_TRANSACTION_STATE,
                      state_reference="tree-measurement-7f2a")
    assert subject.commit == ""


@pytest.mark.parametrize("bad", ["baseline_collection", None, 1, True])
def test_a_phase_that_is_not_the_enum_is_refused(tmp_path, sink, bad):
    """Annotations enforce nothing, as the transition owner already found."""
    with pytest.raises(CaptureError, match="CollectionPhase"):
        capture_collection(
            argv=emit(), cwd=tmp_path, env=None, timeout=60, phase=bad,
            subject=committed(), interpreter=sys.executable, sink=sink)


def test_a_subject_that_is_not_the_type_is_refused(tmp_path, sink):
    with pytest.raises(CaptureError, match="must be a Subject"):
        capture_collection(
            argv=emit(), cwd=tmp_path, env=None, timeout=60,
            phase=CollectionPhase.BASELINE_COLLECTION,
            subject={"kind": "committed_tree", "commit": COMMIT},
            interpreter=sys.executable, sink=sink)


# ---------------------------------------------------------------------------
# 4. The decoding procedure is declared, and its loss is demonstrable
# ---------------------------------------------------------------------------

def test_the_declared_decoding_procedure_is_recorded(tmp_path, sink):
    document, _ = capture_collection(
        argv=emit(stdout=b"t.py::test_a\n"), cwd=tmp_path, env=None, timeout=60,
        phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    assert document["decode_procedure"] == "utf-8/replace"


def test_replacement_decoding_maps_distinct_bytes_to_one_text():
    """NOT a defect being introduced -- the INSTALLED behaviour, demonstrated.

    Two different invalid sequences decode to the same text, so two distinct
    collections can yield one identity. Retaining the bytes is what allows that
    to be seen: their digests differ even when their texts do not. This is why
    the capture preserves the decoder rather than improving it -- changing it
    is an interpretation change that must be measured.
    """
    first, second = b"t.py::test_x[\xff]\n", b"t.py::test_x[\xfe]\n"
    assert first != second
    assert decode_retained(first) == decode_retained(second)
    assert hashlib.sha256(first).hexdigest() != \
        hashlib.sha256(second).hexdigest()


# ---------------------------------------------------------------------------
# 5. Positive control
# ---------------------------------------------------------------------------

def test_an_ordinary_collection_is_captured_end_to_end(tmp_path, sink):
    """Every test above is a refusal or a property check. Without this one, a
    component that captured nothing could satisfy them all."""
    listing = "t.py::test_a\nt.py::test_b\n\n2 tests collected in 0.01s\n"
    document, out = capture_collection(
        argv=emit(stdout=listing.encode("utf-8")), cwd=tmp_path, env=None, timeout=60,
        phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    assert decode_retained(out) == listing
    assert document["phase"] == "baseline_collection"
    assert document["subject"]["kind"] == "committed_tree"
    assert document["subject"]["commit"] == COMMIT
    assert document["stdout"]["bytes"] == len(listing.encode("utf-8"))
    assert document["schema"] == "gvc.collection-capture"
    assert "does_not_establish" in document


# ---------------------------------------------------------------------------
# 6. Attempt isolation -- added 2026-09-12
#
# MEASURED 2026-09-12 against the sink installed at d181e739: two captures with
# the same phase and IDENTICAL STDOUT but different stderr resolved to one
# filename stem, `phase-stdout_sha256[:16]`, and every write overwrote --
#
#     same retained stderr path : True
#     first attempt's stderr file now contains: b'second\n'
#     first attempt's evidence still matches   : False
#
# No SHA-256 collision is involved. Identical stdout across repeated collection
# is ORDINARY: a baseline collected twice produces it.
#
# THE SEVENTEEN TESTS ABOVE PASSED AGAINST THAT DEFECT AND AGAINST ITS REPAIR.
# They never exercised repeated capture, so the suite could not distinguish the
# two implementations. These controls exist because passing tests are not
# coverage.
# ---------------------------------------------------------------------------

def capture_twice(tmp_path, sink, first_stderr, second_stderr):
    """Two real captures through ONE sink, with byte-identical stdout."""
    documents = []
    for stderr in (first_stderr, second_stderr):
        document, _ = capture_collection(
            argv=emit(stdout=b"identical\n", stderr=stderr), cwd=tmp_path,
            env=None, timeout=60, phase=CollectionPhase.BASELINE_COLLECTION,
            subject=committed(), interpreter=sys.executable, sink=sink)
        documents.append(document)
    return documents


def test_a_repeated_capture_does_not_overwrite_the_earlier_attempt(
        tmp_path, sink):
    """THE DEFECT. Content digests identify artifacts; attempt identifiers
    distinguish executions. Conflating them lost an earlier attempt."""
    first, second = capture_twice(tmp_path, sink, b"first\n", b"second\n")
    assert first["attempt_id"] != second["attempt_id"]
    retained = tmp_path / "evidence" / first["retained"]["stderr_path"]
    assert retained.read_bytes() == b"first\n"
    assert hashlib.sha256(retained.read_bytes()).hexdigest() == \
        first["stderr"]["sha256"]


def test_both_attempts_remain_separately_retrievable(tmp_path, sink):
    first, second = capture_twice(tmp_path, sink, b"first\n", b"second\n")
    directory = tmp_path / "evidence"
    assert (directory / first["retained"]["stderr_path"]).read_bytes() == \
        b"first\n"
    assert (directory / second["retained"]["stderr_path"]).read_bytes() == \
        b"second\n"


def test_identical_captures_are_still_distinct_attempts(tmp_path, sink):
    """Even when BOTH streams match, two executions are two attempts."""
    first, second = capture_twice(tmp_path, sink, b"same\n", b"same\n")
    assert first["attempt_id"] != second["attempt_id"]
    assert first["retained"]["stdout_path"] != second["retained"]["stdout_path"]


def test_an_artifact_may_not_be_overwritten(tmp_path, sink):
    """Exclusive creation, so a replacement is a refusal rather than a loss."""
    document, _ = capture_collection(
        argv=emit(stdout=b"x\n"), cwd=tmp_path, env=None, timeout=60,
        phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    existing = tmp_path / "evidence" / document["retained"]["stdout_path"]
    with pytest.raises(RetentionFailed, match="may not overwrite"):
        DirectorySink._write_new(existing, b"replacement")


def test_the_manifest_is_published_after_the_streams(tmp_path, sink):
    """Its ABSENCE is how a reader determines an attempt did not complete.
    Publishing it first would make an interrupted attempt look finished."""
    document, _ = capture_collection(
        argv=emit(stdout=b"x\n"), cwd=tmp_path, env=None, timeout=60,
        phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
        interpreter=sys.executable, sink=sink)
    directory = tmp_path / "evidence"
    manifest = directory / document["manifest"]["path"]
    assert manifest.is_file()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == \
        document["manifest"]["sha256"]
    assert manifest.name == DirectorySink.MANIFEST_NAME


def test_a_retained_artifact_is_checked_for_size_as_well_as_digest(tmp_path):
    """A truncated artifact whose prefix coincides is not the artifact."""

    class Truncating(DirectorySink):
        @staticmethod
        def _write_new(path, raw):
            DirectorySink._write_new(path, raw[:-1] if raw else raw)

    with pytest.raises(RetentionFailed, match="bytes, not"):
        capture_collection(
            argv=emit(stdout=b"abcdef\n"), cwd=tmp_path, env=None, timeout=60,
            phase=CollectionPhase.BASELINE_COLLECTION, subject=committed(),
            interpreter=sys.executable,
            sink=Truncating(tmp_path / "truncating"))


# ---------------------------------------------------------------------------
# 7. The reader -- added 2026-09-12
#
# An in-memory reader cannot qualify filesystem retention. These controls run
# against a real directory: retrieval, limits, absence, and the root as a
# boundary.
#
# The reader is deliberately narrow -- one read operation. Not list, not
# delete, not upload, not "latest", not retention. A wider interface would have
# to be qualified wider.
# ---------------------------------------------------------------------------

@pytest.fixture
def reader_root(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "a.txt").write_bytes(b"hello\n")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_bytes(b"nested\n")
    (root / "empty").write_bytes(b"")
    (tmp_path / "outside.txt").write_bytes(b"secret\n")
    # A SIBLING whose name shares the root's prefix. Containment must be
    # established by resolved path, never by string prefix: "evidence2" starts
    # with "evidence" as text while being no descendant of it.
    (tmp_path / "evidence2").mkdir()
    (tmp_path / "evidence2" / "c.txt").write_bytes(b"sibling\n")
    return root


def test_the_reader_returns_retained_bytes(reader_root):
    reader = DirectoryReader(reader_root)
    assert reader.read("a.txt", max_bytes=64) == b"hello\n"
    assert reader.read("sub/b.txt", max_bytes=64) == b"nested\n"


def test_a_zero_length_artifact_is_legitimate_not_absence(reader_root):
    """Empty bytes can be a real artifact -- an empty stderr, for instance.
    Absence must have a separate representation, and it does."""
    assert DirectoryReader(reader_root).read("empty", max_bytes=64) == b""


def test_a_missing_artifact_is_unavailable_not_a_failure(reader_root):
    with pytest.raises(ArtifactUnavailable, match="no artifact is retained"):
        DirectoryReader(reader_root).read("nope", max_bytes=64)


def test_a_directory_key_is_refused(reader_root):
    with pytest.raises(ArtifactUnavailable, match="names a directory"):
        DirectoryReader(reader_root).read("sub", max_bytes=64)


def test_the_read_limit_is_enforced(reader_root):
    reader = DirectoryReader(reader_root)
    with pytest.raises(ArtifactTooLarge, match="exceeds the"):
        reader.read("a.txt", max_bytes=3)
    # AT the boundary, not one byte late.
    assert reader.read("a.txt", max_bytes=6) == b"hello\n"


def test_a_negative_limit_is_refused(reader_root):
    with pytest.raises(CaptureError, match="nonnegative"):
        DirectoryReader(reader_root).read("a.txt", max_bytes=-1)


#: EXPLICIT IDS. MEASURED 2026-09-12: with generated ids this parametrisation
#: produced ONE identity containing a backslash --
#:
#:     test_the_root_is_the_boundary[sub\\b.txt-a backslash path]
#:
#: -- which is exactly the character whose interpretation changed at 07bc7a5.
#: The PARAMETER VALUES are unchanged, so the coverage is unchanged; only the
#: identities become parser-independent. Without this, a unit declaring these
#: tests would carry an identity that two interpretations read differently.
@pytest.mark.parametrize("key", [
    pytest.param("../outside.txt", id="parent_traversal"),
    pytest.param("sub/../../outside.txt", id="traversal_through_a_real_dir"),
    pytest.param("../evidence2/c.txt", id="sibling_sharing_the_name_prefix"),
    pytest.param("/etc/passwd", id="absolute_posix_path"),
    pytest.param("C:/Windows/x", id="drive_letter"),
    pytest.param("sub\\b.txt", id="backslash_path"),
    pytest.param("https://example.test/x", id="url"),
    pytest.param("", id="empty_key"),
])
def test_the_root_is_the_boundary(reader_root, key):
    """Evidence must never nominate a location for the reader to open --
    the same principle as a manifest never nominating executable code."""
    with pytest.raises(ArtifactUnavailable):
        DirectoryReader(reader_root).read(key, max_bytes=64)


def test_the_reader_reads_what_the_sink_wrote(tmp_path):
    """The two halves of retention, exercised against each other."""
    root = tmp_path / "evidence"
    sink = DirectorySink(root / "capture")
    document, _ = capture_collection(
        argv=emit(stdout=b"t.py::test_a\n"), cwd=tmp_path, env=None,
        timeout=60, phase=CollectionPhase.BASELINE_COLLECTION,
        subject=committed(), interpreter=sys.executable, sink=sink)
    reader = DirectoryReader(root)
    key = "capture/" + document["manifest"]["path"]
    raw = reader.read(key, max_bytes=1 << 20)
    assert hashlib.sha256(raw).hexdigest() == document["manifest"]["sha256"]


def test_a_permissions_denial_is_a_failure_not_an_absence(reader_root,
                                                          monkeypatch):
    """The distinction the gate failure nearly destroyed.

    MEASURED 2026-09-12: opening a directory raises IsADirectoryError on POSIX
    and PermissionError on Windows. Catching PermissionError to fix that would
    have mapped a REAL denial onto "names a directory". The directory case is
    now TESTED rather than inferred, and a denial stays a retrieval FAILURE.

    monkeypatch is used because a denial cannot be produced portably: a
    read-only file is still readable, and a process running as root ignores the
    mode entirely.
    """
    reader = DirectoryReader(reader_root)
    real_open = Path.open

    def denied(self, *args, **kwargs):
        if self.name == "a.txt":
            raise PermissionError(13, "Permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(CaptureError, match="could not be read"):
        reader.read("a.txt", max_bytes=64)
    # And it is NOT the unavailable family, which would mean absence.
    monkeypatch.setattr(Path, "open", denied)
    try:
        reader.read("a.txt", max_bytes=64)
    except ArtifactUnavailable:                      # pragma: no cover
        pytest.fail("a permissions denial must not be reported as absence")
    except CaptureError:
        pass
