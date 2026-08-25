"""The committed archive must agree with the committed manifest.

ADR-0004. Created 2026-08-23.

WHY THIS EXISTS SEPARATELY FROM test_archive_manifest.py
========================================================
That file proves the manifest TYPE refuses malformed input. This one proves the
manifest ON DISK describes the artifacts ON DISK. A type that cannot be
misused is not the same as an archive that is correct, and a manifest agreeing
only with itself is the failure mode this project has repaired twice already --
a README stating a feature count in nine places with four values, and nine
attestations in three shapes under one schema version.

WHY THE CITATION SET IS COMMITTED, NOT DERIVED
==============================================
Every assertion below reads the manifest and the filesystem. NONE walks git
history.

`actions/checkout` defaults to `fetch-depth: 1` -- measured 2026-08-22 across
ten invocations in four workflows, none declaring a depth. A test collecting
`ATTESTATION:` trailers from `git log --all` would see ONE commit in continuous
integration, collect one citation, assert it resolves, and pass. It would shrink
its own obligation until it succeeded.

So full-history reconciliation is MIGRATION evidence, performed once by a probe
where history exists, and the resulting alias set is committed. These tests are
non-vacuous in any checkout, shallow or complete.

WHY THE ARCHIVE IS ALWAYS ONE BEHIND
====================================
The unit that preserves N attestations writes its own attestation AFTER its
commit, so that document cannot be inside the archive it creates. That is
inherent, not a loss: `genesis_cardinality` is what existed when the archive was
born. These tests therefore assert `>=`, never `==`.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from genomic_variant_classifier.repository_records.archive_manifest import (
    ArchiveManifest,
)
from genomic_variant_classifier.repository_records.roles import (
    ArtifactRole,
    canonical_root,
    is_within_canonical_root,
)
from genomic_variant_classifier.repository_records.validation import (
    validate_authored_text,
)

ROOT = Path(str(canonical_root(ArtifactRole.INSTALLATION_ATTESTATION)))
MANIFEST = ROOT / "manifest.json"
#: `ARTIFACTS = ROOT / "artifacts"` was removed on 2026-08-25 with the
#: assertion that used it. The archive is reconciled RECURSIVELY over the
#: role root now, so naming one subdirectory would reinstate the closed-world
#: assumption this repair removed -- and an unused constant is an invitation
#: to reinstate it.


@pytest.fixture(scope="module")
def manifest() -> ArchiveManifest:
    if not MANIFEST.is_file():
        pytest.fail(
            "{} does not exist. The archive index is what makes a historical "
            "citation resolvable; without it the artifacts are bytes nobody "
            "can find.".format(MANIFEST.as_posix()))
    return ArchiveManifest.parse(MANIFEST.read_bytes())


def test_the_manifest_parses_and_is_byte_stable_on_disk(manifest):
    """Committed bytes must equal what render() produces.

    If they differ, someone edited the index by hand -- and a hand-edited
    machine record is how ATTESTATION-SCHEMA-DRIFT-1 began.
    """
    assert MANIFEST.read_bytes() == manifest.render(), (
        "the committed manifest is not what the typed renderer produces; it "
        "was edited outside the type that owns it")


def test_the_manifest_is_authored_and_ends_with_a_newline():
    """The authoring half of ADR-0004 section C's asymmetry."""
    raw = MANIFEST.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert not any(b > 0x7F for b in raw), "ensure_ascii should hold"


def test_every_manifest_entry_exists_on_disk_with_the_recorded_bytes(manifest):
    """The load-bearing assertion. Digest AND size, not existence alone."""
    problems = []
    for entry in manifest.entries:
        p = Path(entry.identity.instance.canonical_path)
        if not p.is_file():
            problems.append("{}: ABSENT".format(p.as_posix()))
            continue
        raw = p.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry.identity.instance.content_sha256:
            problems.append("{}: digest {} != recorded {}".format(
                p.as_posix(), actual[:16],
                entry.identity.instance.content_sha256[:16]))
        if len(raw) != entry.identity.instance.size_bytes:
            problems.append("{}: {} bytes != recorded {}".format(
                p.as_posix(), len(raw), entry.identity.instance.size_bytes))
    assert not problems, "\n  ".join([""] + problems)


def test_every_record_on_disk_is_in_the_manifest(manifest):
    """The other direction. An unindexed record is a file nobody can find,
    and its presence would make the index silently incomplete.

    RECURSIVE OVER THE ROLE ROOT, not `artifacts/` alone.

    INSTALLATION-ARCHIVE-BINDING-HARDCODES-ARTIFACTS-DIR-1, measured
    2026-08-25. This assertion previously read `ARTIFACTS.iterdir()`, which is
    both hardcoded to one subdirectory and non-recursive. The type says the
    archive may grow; this test quietly said it may grow only with objects in
    one directory. The first authored reconstruction, filed under
    `reconstructions/`, entered `indexed` and never `on_disk` -- failing while
    the manifest was correct.

    Enumerating two named directories instead would be tomorrow's duplication.
    Deriving from the role root also REFUSES an unindexed record appearing in
    some third subtree later, which neither form did.
    """
    if not ROOT.is_dir():
        pytest.fail("{} does not exist".format(ROOT.as_posix()))
    on_disk = {p.as_posix() for p in sorted(ROOT.rglob("*.json"))
               if p != MANIFEST}
    indexed = {e.identity.instance.canonical_path for e in manifest.entries}
    assert on_disk == indexed, (
        "on disk but NOT indexed: {}\nindexed but NOT on disk: {}".format(
            sorted(on_disk - indexed), sorted(indexed - on_disk)))


def test_the_preserved_artifacts_are_NOT_authored_files(manifest):
    """PRESERVATION, not authoring. Measured 2026-08-23: every install
    attestation ends WITHOUT a newline because json.dumps does not append one.

    If one ever ends WITH a newline, either an artifact was mutated on the way
    in or the emitting installer changed -- and both matter.

    SCOPED TO PRESERVED RECORDS. This loop previously ran over EVERY entry,
    which was correct for the corpus it was born with and false the moment an
    authored record joined it: `validate_authored_text` REQUIRES a trailing
    newline, so the two rules would contradict each other.

    Scoped by PROVENANCE, never by filename or directory. ADR-0004: provenance
    is a fact about the artifact, and encoding it in a path loses it the moment
    the artifact moves. A `"reconstruction-" in p.name` test would make the
    filename the author of semantics.
    """
    ending = {}
    for entry in manifest.entries:
        if entry.reconstructs_missing_artifact:
            continue
        raw = Path(entry.identity.instance.canonical_path).read_bytes()
        ending[entry.identity.basename] = raw.endswith(b"\n")
    assert ending, (
        "no preserved artifact was examined. If every entry is now a "
        "reconstruction this assertion has become vacuous, which is worse "
        "than absent -- investigate rather than delete.")
    offenders = sorted(k for k, v in ending.items() if v)
    assert not offenders, (
        "these preserved artifacts end with a newline: {}\nThe authoring "
        "predicate demands one; the preservation predicate forbids ADDING "
        "one. A preserved artifact that gained a newline was mutated."
        .format(offenders))


def test_the_reconstructions_ARE_authored_files(manifest):
    """The complementary half, so the archive PROVES both policies.

    A reconstruction is written fresh by this repository, so ADR-0004
    section C's AUTHORING predicate applies to it in full: no byte-order mark,
    no CRLF, pure ASCII, a trailing newline.

    Without this the repair would merely EXEMPT reconstructions from the
    preservation rule and assert nothing in its place -- an exemption is not a
    contract.
    """
    reconstructed = [e for e in manifest.entries
                     if e.reconstructs_missing_artifact]
    if not reconstructed:
        pytest.skip("no reconstruction is indexed yet")
    for entry in reconstructed:
        path = entry.identity.instance.canonical_path
        validate_authored_text(path, Path(path).read_bytes())


def test_every_genesis_alias_still_resolves(manifest):
    """Historical citation compatibility, as a repository property.

    Sixteen or more commit messages name these files. Basename retention is a
    NECESSARY CONDITION for resolution and is not resolution: git does not turn
    a filename in a commit message into a locator. This does.
    """
    unresolved = [a for a in manifest.genesis_aliases
                  if manifest.resolve(a) is None]
    assert not unresolved, (
        "genesis alias(es) that no longer resolve: {}\nA historical citation "
        "that stops resolving is evidence lost.".format(unresolved))


def test_the_archive_has_not_shrunk_below_its_genesis(manifest):
    """`>=`, never `==`. The archive may grow; it may never shrink, and the
    preserving unit's own attestation is necessarily outside it."""
    assert len(manifest.entries) >= manifest.genesis_cardinality


def test_every_entry_lies_beneath_the_canonical_root_for_its_role(manifest):
    """Placement follows ROLE. Filing a record elsewhere is how six evidence
    locations happened."""
    stray = [e.identity.instance.canonical_path for e in manifest.entries
             if not is_within_canonical_root(
                 e.identity.instance.canonical_path, e.identity.role)]
    assert not stray, stray


def test_no_artifact_was_normalised_on_checkout(manifest):
    """RECORDS-EOL-NORMALIZATION-1, as a standing guard.

    `.gitattributes` sets `-text` on the artifacts directory so preserved bytes
    survive checkout. If that rule were removed or reordered, a carriage return
    would appear here on a Windows working tree -- and the digest assertion
    above would fail first. This names the cause explicitly so a future reader
    is not left guessing why digests drifted.
    """
    for entry in manifest.entries:
        raw = Path(entry.identity.instance.canonical_path).read_bytes()
        assert b"\r\n" not in raw, (
            "{} contains CRLF. Either .gitattributes lost its `-text` rule for "
            "the artifacts directory, or the general `records/**/*.json` rule "
            "was moved AFTER it -- later rules win."
            .format(entry.identity.basename))


def test_the_plane_root_readme_exists_and_is_authored():
    """ADR-0004 section I: one README at the PLANE root, not per family.

    A README inside each family directory would establish per-family
    documentation as the norm and quietly recreate the scatter this plane
    exists to end.
    """
    plane = Path("records/README.md")
    assert plane.is_file(), "the plane root has no README"
    raw = plane.read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert not (ROOT / "README.md").exists(), (
        "a per-family README would recreate the scatter this plane ends")
