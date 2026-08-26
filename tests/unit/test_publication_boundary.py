"""Evidence reaches disk one way, and that way validates.

PENDING-ATTESTATION-BYPASSES-SCHEMA-VALIDATION-1. Created 2026-08-26.

WHAT THIS GUARDS
----------------
`AttestationDocument` validated and serialised but did not WRITE, so every
caller opened a file itself and two paths diverged:

    success   payload -> AttestationDocument -> to_json -> caller writes
    pending   payload -> json.dumps -------------------> caller writes

MEASURED 2026-08-26 across THIRTY-THREE delivered installers: every single
`except PublicationPending` handler took the second path. Not twenty-two, as a
stale census claimed -- and the count included four installers written the day
before by an author who had just applied the opposing rule to the transition
record and the target record in those same files.

THE PENDING STATE WAS ALWAYS VALIDATABLE. `InstallStatus` declares
INSTALL_APPLIED_PUBLICATION_PENDING, `validate` requires `publication_error`
exactly when the status is pending, and nothing constrains `post_head`'s VALUE.
The schema anticipated the state; the pending path never used it.

THE STATIC GUARD FOLLOWS A PROVEN PATTERN
-----------------------------------------
`tests/unit/test_attestation_projection.py` established it on 2026-08-25: a
structural search, PARSED rather than grepped, plus a case proving the search
can find a PLANTED offender. A search that matched nothing would report green
forever -- the vacuous-iterator shape this repository keeps finding.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.transactions.install_attestation import (
    SCHEMA,
    SCHEMA_VERSION,
    AttestationDocument,
    AttestationSchemaError,
    PublicationError,
    publish,
)

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "genomic_variant_classifier"

#: The module that OWNS publication. Every other module is an offender if it
#: serialises an attestation itself.
OWNER = "install_attestation.py"


def a_payload(**over):
    """A minimal well-formed version-2 attestation.

    Real-shaped rather than minimal-shaped: `validate` enforces cross-field
    consistency, so a payload that merely has the right keys would be refused
    for the wrong reason and prove nothing.
    """
    payload = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "unit": "TEST-UNIT", "plan_digest": "a" * 64, "status": "PUBLISHED",
        "started_at": "2026-08-26T00:00:00Z",
        "finished_at": "2026-08-26T00:01:00Z",
        "python": "3.12.10", "platform": "test",
        "repository": {
            "pre_head": "aaaaaaa", "post_head": "bbbbbbb",
            "pre_head_oid": "a" * 40, "post_head_oid": "b" * 40,
        },
        "counter": {"scope": "tests", "before": 1, "after": 2},
        "acceptance": {
            "scope": "tests", "returncode": 0, "passed": 2, "skipped": 0,
            "failed": 0, "errors": 0, "xfailed": 0, "xpassed": 0,
            "warnings": 0, "summary": "2 passed", "seconds": 1.0,
            "measured_by": {}},
        "targets": [{"path": "x", "action": "create",
                     "post_sha256": "b" * 64, "post_size": 1}],
        "suite_transition": {
            "kind": "addition",
            "expected_added_nodeids": ["t::a"], "expected_removed_nodeids": [],
            "observed_added_nodeids": ["t::a"], "observed_removed_nodeids": [],
            "before_count": 1, "after_count": 2,
            "before_digest": "c" * 64, "after_digest": "d" * 64},
    }
    payload.update(over)
    return payload


def a_pending_payload():
    """The state that never reached the validator."""
    return a_payload(
        status="INSTALL_APPLIED_PUBLICATION_PENDING",
        publication_error="git add exited 128: index.lock exists",
        repository={"pre_head": "aaaaaaa", "post_head": None,
                    "pre_head_oid": "a" * 40, "post_head_oid": None})


# ---------------------------------------------------------------------------
# 1. THE PENDING STATE VALIDATES -- IT ALWAYS COULD
# ---------------------------------------------------------------------------

def test_a_pending_attestation_is_a_valid_version_2_document():
    """The finding, stated as a test.

    If this failed, the repair would have to be "give the pending state a
    shape that CAN be validated" -- a materially different unit. It passes,
    so the pending path simply never used the schema that was waiting for it.
    """
    doc = AttestationDocument(payload=a_pending_payload())
    assert doc.payload["status"] == "INSTALL_APPLIED_PUBLICATION_PENDING"


def test_a_pending_attestation_may_carry_a_null_post_head():
    """A pending install has no post-commit head because it never committed,
    and recording null is the truthful thing to do.

    Version 3 records BOTH the abbreviation and the full object identifier, so
    BOTH are null here -- and `_check_repository` refuses one of each, because
    a half-committed install is a state that cannot exist.
    """
    doc = AttestationDocument(payload=a_pending_payload())
    assert doc.payload["repository"]["post_head"] is None
    assert doc.payload["repository"]["post_head_oid"] is None


def test_a_pending_attestation_without_its_error_is_refused():
    """The cross-field rule that makes the state meaningful."""
    payload = a_pending_payload()
    del payload["publication_error"]
    with pytest.raises(AttestationSchemaError) as exc:
        AttestationDocument(payload=payload)
    assert "publication_error" in str(exc.value)


def test_a_published_attestation_WITH_an_error_is_refused():
    """The same rule, in the other direction."""
    with pytest.raises(AttestationSchemaError) as exc:
        AttestationDocument(payload=a_payload(publication_error="boom"))
    assert "publication_error" in str(exc.value)


def test_a_pending_attestation_may_record_a_nonzero_gate():
    """Line 221 refuses a nonzero return code only when PUBLISHED.

    A pending install may legitimately have failed its gate; forbidding that
    would make the state unusable for the case it exists to describe.
    """
    payload = a_pending_payload()
    payload["acceptance"] = dict(payload["acceptance"], returncode=1)
    assert AttestationDocument(payload=payload)


# ---------------------------------------------------------------------------
# 2. ONE PATH TO DISK
# ---------------------------------------------------------------------------

def test_publish_writes_bytes_that_re_validate(tmp_path):
    """Rendering and re-reading proves the BYTES are a valid attestation, not
    merely the object that produced them."""
    doc = AttestationDocument(payload=a_payload())
    out = publish(doc, tmp_path / "a.json")
    assert out.is_file()
    reread = AttestationDocument.from_json(out.read_text(encoding="utf-8"))
    assert reread.payload == doc.payload


def test_publish_writes_line_feed_endings_only(tmp_path):
    """The preserved artifacts are LF. A platform-default newline would make
    the same document differ by operating system.

    THIS GUARD IS PLATFORM-ASYMMETRIC, and saying so is part of the record.
    MEASURED 2026-08-26: on Linux `os.linesep` is a line feed, so
    `newline=None` and `newline="\n"` produce IDENTICAL bytes and no test in
    that environment can distinguish them. On Windows -- where this repository
    is developed and where every attestation to date was written --
    `newline=None` emits CRLF and this assertion fires.

    So a sabotage of the newline argument shows NOTHING FAILED on Linux. That
    is a limit of the environment, not a toothless guard, and the distinction
    matters: this repository deletes defence that CANNOT fire, and keeps
    defence that fires where the risk lives.
    """
    out = publish(AttestationDocument(payload=a_payload()), tmp_path / "a.json")
    assert b"\r\n" not in out.read_bytes()


def test_a_document_mutated_after_construction_is_refused(tmp_path):
    """The re-parse, shown firing on a REACHABLE case.

    `AttestationDocument` is a frozen dataclass -- but `payload` is a dict, and
    a dict is MUTABLE. MEASURED 2026-08-26. So a document that validated at
    construction can be altered afterwards, and `to_json` would faithfully
    render the alteration.

    Re-reading the rendered text is what catches it. Without this case the
    re-parse would be defence nobody had seen fire, which this repository
    treats as worse than absence: `suite_transition.py` deleted three such
    checks for exactly that reason.
    """
    doc = AttestationDocument(payload=a_payload())
    doc.payload["suite_transition"]["observed_added_nodeids"] = []
    with pytest.raises(AttestationSchemaError):
        publish(doc, tmp_path / "a.json")
    assert not (tmp_path / "a.json").exists(), (
        "the refusal left a partial artifact on disk")


def test_the_pending_state_publishes_through_the_SAME_function(tmp_path):
    """There is no second path. That is the entire repair."""
    doc = AttestationDocument(payload=a_pending_payload())
    out = publish(doc, tmp_path / "pending.json")
    back = json.loads(out.read_text(encoding="utf-8"))
    assert back["status"] == "INSTALL_APPLIED_PUBLICATION_PENDING"
    assert back["repository"]["post_head"] is None
    assert back["repository"]["post_head_oid"] is None
    assert back["publication_error"]


def test_a_raw_dict_cannot_be_published(tmp_path):
    """The TYPE is the proof that validation happened.

    Accepting a payload here would restore the bypass in a single argument.
    """
    with pytest.raises(PublicationError) as exc:
        publish(a_payload(), tmp_path / "a.json")
    assert "AttestationDocument" in str(exc.value)


def test_publishing_over_an_existing_artifact_is_refused(tmp_path):
    """Evidence is written once. Overwriting would destroy the record of what
    an earlier install claimed."""
    doc = AttestationDocument(payload=a_payload())
    publish(doc, tmp_path / "a.json")
    with pytest.raises(PublicationError) as exc:
        publish(doc, tmp_path / "a.json")
    assert "already exists" in str(exc.value)


def test_a_missing_parent_directory_is_refused(tmp_path):
    """Creating directories to store evidence hides a misconfigured path until
    an audit cannot find the artifact."""
    doc = AttestationDocument(payload=a_payload())
    with pytest.raises(PublicationError) as exc:
        publish(doc, tmp_path / "absent" / "a.json")
    assert "does not exist" in str(exc.value)
    assert not (tmp_path / "absent").exists(), (
        "the refusal created the directory it was refusing to use")


# ---------------------------------------------------------------------------
# 3. NO MODULE MAY SERIALISE EVIDENCE ITSELF
# ---------------------------------------------------------------------------

def _package_sources():
    assert PACKAGE.is_dir(), PACKAGE
    return sorted(PACKAGE.rglob("*.py"))


def _serialises_attestation(tree, text):
    """`json.dumps(...)` in a module that also names an attestation type.

    Parsed, not grepped: this file's own docstring names `json.dumps` several
    times, and a substring search cannot tell narration from code. That
    distinction has been the difference four times in this programme.
    """
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in ("dumps", "dump"):
            continue
        if ast.unparse(fn.value).split(".")[-1] != "json":
            continue
        hits.append(node.lineno)
    return hits


def test_no_module_outside_the_owner_serialises_an_attestation():
    """One authority for evidence on disk.

    The owner is exempt because it IS the owner. Any other module that both
    names an attestation type and calls `json.dumps` is publishing on its own
    authority, which is the defect this unit closes.
    """
    offenders = []
    for path in _package_sources():
        if path.name == OWNER:
            continue
        text = path.read_text(encoding="utf-8")
        if "AttestationDocument" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:                       # pragma: no cover
            continue
        for line in _serialises_attestation(tree, text):
            offenders.append("{}:{}".format(path.name, line))
    assert not offenders, (
        "these modules serialise an attestation instead of calling publish(): "
        "{}".format(offenders))


def test_the_static_guard_can_actually_find_an_offender():
    """Guards the guard.

    A structural search that matched nothing would pass over an empty result
    and report green forever. Proven on a synthetic offender so its silence on
    the real package means something.
    """
    source = (
        "import json\n"
        "def emit(doc: 'AttestationDocument') -> None:\n"
        "    open('x.json', 'w').write(json.dumps(doc.payload, indent=2))\n")
    found = _serialises_attestation(ast.parse(source), source)
    assert found, "the structural predicate matched nothing on a real offender"


def test_the_owner_module_does_serialise_and_is_exempt():
    """The exemption is real, not decorative.

    If the owner stopped serialising, `publish` would not be writing anything
    and this whole file would be guarding an empty contract.
    """
    owner = [p for p in _package_sources() if p.name == OWNER]
    assert len(owner) == 1, owner
    text = owner[0].read_text(encoding="utf-8")
    assert _serialises_attestation(ast.parse(text), text), (
        "the owner module no longer serialises anything")
