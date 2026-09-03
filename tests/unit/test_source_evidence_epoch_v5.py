"""v4/schema3 retired; v5/schema5 introduced. Exactly that, and nothing else.

Phase 1C Unit 3A++.2. Created 2026-09-02.

THE DEFECT RETIRED
------------------
`EVIDENCE-DOMAIN-V4-PAYLOAD-SCHEMA3-1`. The domain read
`drift-source-evidence-manifest-v4` while the payload it digested carried
`"schema_version": 3`. `product` was added to `SourceArtifactKey` on
2026-09-01, changing key equality AND the canonical record shape; the domain
was bumped and the embedded literal was not.

Two writable declarations represented ONE semantic version, which permits
divergence by construction.

WHY v5/schema5 AND NOT v4/schema4
---------------------------------
The v4 epoch HISTORICALLY described records carrying schema_version 3. That is
a fact in the evidence trail, frozen at
`tests/fixtures/source_evidence_epoch_v4/epoch.json`.

Correcting the literal to 4 while keeping the v4 domain would make ONE nominal
domain describe TWO different canonical schemas -- exactly what domain
versioning exists to prevent. So v4 keeps meaning what it historically meant,
and the repaired schema is a NEW epoch. The historical inconsistency becomes
documented rather than rewritten.

WHAT THIS UNIT IS AUTHORIZED TO CHANGE
--------------------------------------
    domain
    embedded schema version
    resulting evidence digest

WHAT IT IS NOT AUTHORIZED TO CHANGE
-----------------------------------
    dependency set          dependency order
    source keys             source identities
    roles                   coordinate contexts
    release identifiers     artifact digests
    the TRANSFORMATION digest, which is a different identity family

Both frozen corpora police that boundary. Neither is regenerated: a corpus
rewritten to match new behaviour is not a witness.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from genomic_variant_classifier.provenance import (
    EVIDENCE_DOMAIN,
    SourceEvidenceManifest,
)
from genomic_variant_classifier.provenance.source import SOURCE_EVIDENCE_SCHEMA

_FIX = (Path(__file__).resolve().parents[2] / "tests" / "fixtures"
        / "source_evidence_epoch_v4" / "epoch.json")

pytestmark = pytest.mark.skipif(
    not _FIX.is_file(),
    reason="the frozen v4 epoch is absent; this unit's whole claim is a "
           "comparison against it -- this must not skip in CI")


def _v4() -> dict:
    return json.loads(_FIX.read_text(encoding="utf-8"))


def _names() -> list:
    return sorted(_v4()["cases"]) if _FIX.is_file() else []


def _strip_epoch(record: dict) -> dict:
    """The record WITHOUT its epoch metadata.

    Section 11: prove that the body is identical and only the epoch moved.
    """
    return {k: v for k, v in record.items() if k != "schema_version"}


def _rebuild(case: dict) -> SourceEvidenceManifest:
    """Reconstruct the manifest a frozen case describes, from its own record.

    Reconstruction rather than reuse: section 39 requires identity types to be
    exercised with equal-but-DISTINCT objects, so a test cannot pass by
    accident of interning, constant folding or a shared instance.

    Built from `canonical_record["dependencies"]`, which carries the identity
    AND the roles together and is already in canonical order. An earlier draft
    matched roles to identities via `dependency_order` and failed on every
    case: that field is `SourceDependency.canonical_key`, which is SIX fields
    -- source, kind, product, release, assembly, digest -- not the three of a
    `SourceArtifactKey`. Reading the frozen record settled it.
    """
    from genomic_variant_classifier.provenance import (
        ArtifactKind, CoordinateContext, CoordinateContextKind,
        SourceArtifactIdentity, SourceArtifactKey, SourceDependency, SourceRole)

    deps = []
    for dep in case["canonical_record"]["dependencies"]:
        record = dep["identity"]
        ctx = record["coordinate_context"]
        kind = CoordinateContextKind(ctx["kind"])
        context = (CoordinateContext.assembly(ctx["identifier"])
                   if kind is CoordinateContextKind.GENOMIC_ASSEMBLY
                   else CoordinateContext.build_independent())
        key = record["key"]
        identity = SourceArtifactIdentity(
            key=SourceArtifactKey(key["source"],
                                  ArtifactKind(key["artifact_kind"]),
                                  key["product"] or None),
            release_id=record["release_id"],
            coordinate_context=context,
            artifact_sha256=record["artifact_sha256"])
        deps.append(SourceDependency(
            identity=identity,
            roles=frozenset(SourceRole(r) for r in dep["roles"])))
    return SourceEvidenceManifest.of(tuple(deps))


# ---------------------------------------------------------------------------
# 1. the epoch moved, and moved completely
# ---------------------------------------------------------------------------

def test_the_live_epoch_is_v5_schema5():
    assert SOURCE_EVIDENCE_SCHEMA.family == "drift-source-evidence-manifest"
    assert SOURCE_EVIDENCE_SCHEMA.version == 5
    assert EVIDENCE_DOMAIN == SOURCE_EVIDENCE_SCHEMA.domain
    assert EVIDENCE_DOMAIN == "drift-source-evidence-manifest-v5"
    assert SOURCE_EVIDENCE_SCHEMA.record(dependencies=[])["schema_version"] == 5


def test_the_two_numbers_can_no_longer_DIVERGE():
    """The defect was architectural, not arithmetical.

    A second writable declaration is what permitted `v4` beside `schema 3`.
    Both now derive from `version`, so no edit can separate them.
    """
    assert SOURCE_EVIDENCE_SCHEMA.domain.endswith(
        "-v{}".format(SOURCE_EVIDENCE_SCHEMA.version))
    assert (SOURCE_EVIDENCE_SCHEMA.record()["schema_version"]
            == SOURCE_EVIDENCE_SCHEMA.version)


def test_the_v4_epoch_is_RETIRED_not_corrected():
    """v4 keeps meaning what it historically meant.

    Had the embedded literal been corrected to 4 under the v4 domain, one
    nominal domain would describe two canonical schemas.
    """
    frozen = _v4()
    assert frozen["domain"] == "drift-source-evidence-manifest-v4"
    assert frozen["embedded_schema_version"] == 3
    assert EVIDENCE_DOMAIN != frozen["domain"]
    assert SOURCE_EVIDENCE_SCHEMA.version != frozen["embedded_schema_version"]


def test_the_module_declares_no_literal_domain_or_schema_version():
    """One authority means one declaration."""
    import ast

    src = (Path(__file__).resolve().parents[2] / "src"
           / "genomic_variant_classifier" / "provenance" / "source.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) \
                and getattr(node.func, "id", None) == "domain_digest":
            raise AssertionError(
                "source.py calls domain_digest directly at line {}"
                .format(node.lineno))
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.startswith("drift-source-evidence-manifest-v")]
    assert not literals, literals


# ---------------------------------------------------------------------------
# 2. THE DIFFERENTIAL -- only the authorized things changed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", _names())
def test_the_record_BODY_is_unchanged_from_v4(name):
    """Section 11. Strip the epoch metadata and the records must be EQUAL.

    Dependency set, dependency order, source keys, identities, roles,
    coordinate contexts, release identifiers and artifact digests are all in
    that body. If any moved, this fails and the migration was not what it
    claimed.
    """
    case = _v4()["cases"][name]
    live = _rebuild(case)
    live_record = SOURCE_EVIDENCE_SCHEMA.record(
        dependencies=[d.as_record() for d in live.dependencies])
    assert _strip_epoch(live_record) == _strip_epoch(case["canonical_record"])


@pytest.mark.parametrize("name", _names())
def test_the_epoch_metadata_DID_change(name):
    """The other half. A migration that changed nothing would also pass the
    body comparison above."""
    case = _v4()["cases"][name]
    live_record = SOURCE_EVIDENCE_SCHEMA.record(dependencies=[])
    assert case["canonical_record"]["schema_version"] == 3
    assert live_record["schema_version"] == 5


@pytest.mark.parametrize("name", _names())
def test_the_evidence_DIGEST_moved(name):
    """Law VI: a v4 digest must not silently equal a v5 digest."""
    case = _v4()["cases"][name]
    live = _rebuild(case)
    assert live.digest != case["digest"], (
        "{}: the digest did NOT move. Either the epoch did not change, or "
        "the domain is not participating in the digest.".format(name))


@pytest.mark.parametrize("name", _names())
def test_the_dependency_ORDER_and_KEYS_are_unchanged(name):
    case = _v4()["cases"][name]
    live = _rebuild(case)
    assert [list(d.canonical_key) for d in live.dependencies] \
        == case["dependency_order"]
    assert [list(k.canonical_key) for k in live.keys] == case["keys_in_order"]


@pytest.mark.parametrize("name", _names())
def test_every_source_IDENTITY_is_unchanged(name):
    case = _v4()["cases"][name]
    live = _rebuild(case)
    assert [d.identity.as_record() for d in live.dependencies] \
        == case["identities"]


def test_the_digest_still_DISCRIMINATES_between_manifests():
    """The property a per-case comparison cannot see.

    MEASURED BY SABOTAGE 2026-09-02: replacing the payload with
    `digest(dependencies=[])` made EVERY manifest digest identically, and no
    test noticed. Each per-case assertion only asked "did this digest move
    from its v4 value", and an empty payload moves it too.

    A digest that ignores its content is not an identity. MEASURED: the
    thirteen frozen cases hold TWELVE distinct digests, with exactly one
    deliberate pair -- `same_key_different_release` was built from the SAME
    identity as `clinvar_grch38`. The live epoch must preserve that partition
    exactly: same partition, different values.
    """
    frozen = _v4()["cases"]
    live = {name: _rebuild(case).digest for name, case in frozen.items()}

    assert len(live) == 13
    assert len(set(live.values())) == 12, (
        "the live digests collapse to {} distinct values; a digest that does "
        "not discriminate is not an identity".format(len(set(live.values()))))

    def partition(mapping):
        groups = {}
        for name, value in mapping.items():
            groups.setdefault(value, []).append(name)
        return sorted(sorted(names) for names in groups.values())

    assert partition(live) == partition(
        {n: c["digest"] for n, c in frozen.items()}), (
        "the equivalence classes changed; the migration altered WHICH "
        "manifests are the same, not merely their digest values")
    assert live["same_key_different_release"] == live["clinvar_grch38"]
    assert live["clinvar_grch37"] != live["clinvar_grch38"]
    assert live["same_key_different_digest"] != live["clinvar_grch38"]


def test_removing_a_dependency_CHANGES_the_digest():
    """The narrowest form of the same property.

    An empty or truncated payload must not produce the same identity as the
    full one.
    """
    case = _v4()["cases"]["multiple_sources"]
    full = _rebuild(case)
    assert len(full.dependencies) == 2
    partial = SourceEvidenceManifest.of(full.dependencies[:1])
    assert partial.digest != full.digest
    assert SOURCE_EVIDENCE_SCHEMA.digest(dependencies=[]) != full.digest


# ---------------------------------------------------------------------------
# 3. identity-family orthogonality
# ---------------------------------------------------------------------------

def test_the_TRANSFORMATION_digest_did_NOT_move():
    """Section 12. A source-evidence migration must not touch another family.

    The witness predates this unit: `semantic.json` was frozen at `2d90c23`,
    before `CanonicalDigestSchema` and before v5.
    """
    import pickle

    fix = (Path(__file__).resolve().parents[2] / "tests" / "fixtures"
           / "provenance_migration_v1")
    frozen = json.loads((fix / "semantic.json").read_text(encoding="utf-8"))
    entry = frozen["transformation_all_component_kinds"]
    obj = pickle.loads((fix / "transformation_all_component_kinds.pickle"
                        ).read_bytes())
    assert obj.digest == entry["digest"]
    assert obj.digest == (
        "eda4cf34c0bf866342edee305852c08043adb6d0fb2b6cfc798cd9b891c9df4f")


def test_the_migration_corpus_EVIDENCE_entries_are_expected_to_differ():
    """The frozen corpus is NOT regenerated. It records the pre-v5 state.

    A corpus rewritten to match new behaviour is not a witness. Exactly three
    entries carry a source-evidence digest and all three must now differ;
    everything else in that corpus must still match.
    """
    import pickle

    fix = (Path(__file__).resolve().parents[2] / "tests" / "fixtures"
           / "provenance_migration_v1")
    frozen = json.loads((fix / "semantic.json").read_text(encoding="utf-8"))
    moved, same = [], []
    for name, entry in sorted(frozen.items()):
        obj = pickle.loads((fix / "{}.pickle".format(name)).read_bytes())
        if not hasattr(obj, "digest"):
            continue
        (moved if obj.digest != entry["digest"] else same).append(name)
    assert moved == ["evidence_multi_authority",
                     "evidence_three_gencode_products",
                     "manifest_clinvar"], moved
    assert same == ["transformation_all_component_kinds"], same
