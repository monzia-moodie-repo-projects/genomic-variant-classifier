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

import pytest

from genomic_variant_classifier.operations.collection_capture import (
    CaptureError,
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
