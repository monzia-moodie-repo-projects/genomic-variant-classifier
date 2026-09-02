"""One canonical identity epoch, one version authority.

Phase 1C Unit 3A++.1. Created 2026-09-02.

WHAT THIS TYPE PREVENTS
-----------------------
MEASURED 2026-09-02 in `provenance/source.py`: line 105 declares the domain
`drift-source-evidence-manifest-v4` while line 486 digests a payload carrying
`"schema_version": 3`. `product` was added on 2026-09-01, changing key equality
and the record shape; the domain was bumped and the literal was not.

Two writable declarations represented ONE semantic version. That architecture
permits divergence BY CONSTRUCTION -- no amount of care removes it, because
nothing binds the numbers together.

`CanonicalDigestSchema` binds them. The invalid state

    domain v5, record schema 4

cannot be expressed through the API.

WHY TRANSFORMATION IS THE CONTROL
---------------------------------
Transformation identity was already coherent: domain `...-v1`, payload
`schema_version: 1`. Converting it to `CanonicalDigestSchema` must therefore
change NOTHING, which separates two experiments the design authority insists
must not be conflated:

    introducing the abstraction        must be semantic-zero
    migrating source evidence          is a DELIBERATE semantic change

MEASURED: twelve transformation digests, before and after, all identical. And
a pickle frozen at `2d90c23` -- before this type existed -- loads under the
converted module and produces digest
`eda4cf34c0bf866342edee305852c08043adb6d0fb2b6cfc798cd9b891c9df4f`, exactly as
`semantic.json` recorded it.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation.

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.provenance.digest_schema import (
    CanonicalDigestSchema,
    DigestSchemaError,
)
from genomic_variant_classifier.provenance.serialization import (
    canonical_json,
    domain_digest,
)
from genomic_variant_classifier.provenance.transformation import (
    TRANSFORMATION_DOMAIN,
    TRANSFORMATION_SCHEMA,
    TransformationComponent,
    TransformationComponentKind,
    TransformationIdentity,
)


# ---------------------------------------------------------------------------
# 1. the two numbers cannot drift apart
# ---------------------------------------------------------------------------

def test_the_domain_is_DERIVED_from_family_and_version():
    s = CanonicalDigestSchema(family="example-family", version=7)
    assert s.domain == "example-family-v7"
    assert s.record()["schema_version"] == 7
    assert s.describe() == "example-family-v7 (schema 7)"


def test_the_record_version_and_the_domain_version_ALWAYS_agree():
    """The invalid state `domain v5, record schema 4` is unrepresentable."""
    for version in (1, 2, 3, 4, 5, 99):
        s = CanonicalDigestSchema(family="example-family", version=version)
        assert s.domain.endswith("-v{}".format(version))
        assert s.record(payload=[])["schema_version"] == version


@pytest.mark.parametrize(
    "family",
    ["example-v1", "example-v12", "drift-source-evidence-manifest-v4"],
    ids=["v1", "v12", "the-real-one"])
def test_a_VERSIONED_family_is_refused(family):
    """A version in the family name would be a SECOND place to change."""
    with pytest.raises(DigestSchemaError) as exc:
        CanonicalDigestSchema(family=family, version=1)
    assert "VERSIONED" in str(exc.value) or "unversioned" in str(exc.value)


@pytest.mark.parametrize("family", ["", "   ", None, 7, b"x"],
                         ids=["empty", "spaces", "none", "int", "bytes"])
def test_a_family_that_cannot_BE_one_is_refused(family):
    with pytest.raises(DigestSchemaError):
        CanonicalDigestSchema(family=family, version=1)


def test_a_family_containing_the_SEPARATOR_byte_is_refused():
    """The NUL byte marks the domain/payload boundary in `domain_digest`."""
    with pytest.raises(DigestSchemaError) as exc:
        CanonicalDigestSchema(family="bad\x00family", version=1)
    assert "separator" in str(exc.value)


@pytest.mark.parametrize("version", [0, -1, 1.0, "1", None, True, False],
                         ids=["zero", "negative", "float", "str", "none",
                              "true", "false"])
def test_a_version_that_cannot_BE_one_is_refused(version):
    """`True` is refused EXPLICITLY: `True == 1` in Python, so a boolean
    would silently become schema version 1."""
    with pytest.raises(DigestSchemaError):
        CanonicalDigestSchema(family="example-family", version=version)


def test_a_caller_cannot_SUPPLY_schema_version():
    """Refused, not overridden. Silently ignoring it would let a caller
    believe they had set something."""
    s = CanonicalDigestSchema(family="example-family", version=2)
    with pytest.raises(DigestSchemaError) as exc:
        s.record(schema_version=9, payload=[])
    assert "owned by CanonicalDigestSchema" in str(exc.value)
    with pytest.raises(DigestSchemaError):
        s.digest(schema_version=9)


def test_the_family_is_stripped_and_frozen():
    s = CanonicalDigestSchema(family="  example-family  ", version=1)
    assert s.family == "example-family"
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.version = 2


def test_two_schemas_of_the_same_family_and_version_are_EQUAL():
    a = CanonicalDigestSchema(family="example-family", version=3)
    b = CanonicalDigestSchema(family="example-family", version=3)
    assert a == b and a is not b
    assert a != CanonicalDigestSchema(family="example-family", version=4)


# ---------------------------------------------------------------------------
# 2. the digest it produces is the digest domain_digest would have produced
# ---------------------------------------------------------------------------

def test_the_digest_equals_the_HAND_BUILT_equivalent():
    """The abstraction must add no encoding of its own."""
    s = CanonicalDigestSchema(family="example-family", version=3)
    payload = {"items": [{"a": 1}, {"b": 2}]}
    assert s.digest(**payload) == domain_digest(
        "example-family-v3", {"schema_version": 3, "items": payload["items"]})


def test_the_stamped_record_is_canonical_json_serialisable():
    s = CanonicalDigestSchema(family="example-family", version=1)
    rec = s.record(items=[3, 1, 2])
    once = canonical_json(rec)
    assert once == canonical_json(json.loads(once.decode("ascii")))
    assert json.loads(once.decode("ascii"))["schema_version"] == 1


def test_a_version_change_CHANGES_the_digest():
    """Law VI: a vN digest cannot silently equal a vN+1 digest."""
    a = CanonicalDigestSchema(family="example-family", version=1)
    b = CanonicalDigestSchema(family="example-family", version=2)
    assert a.digest(items=[]) != b.digest(items=[])
    assert a.domain != b.domain
    assert a.record(items=[]) != b.record(items=[])


def test_a_family_change_CHANGES_the_digest():
    a = CanonicalDigestSchema(family="family-one", version=1)
    b = CanonicalDigestSchema(family="family-two", version=1)
    assert a.digest(items=[]) != b.digest(items=[])


# ---------------------------------------------------------------------------
# 3. THE SEMANTIC-ZERO CONTROL -- transformation identity
# ---------------------------------------------------------------------------

def test_transformation_uses_ONE_authority():
    assert TRANSFORMATION_SCHEMA.family == "drift-transformation-identity"
    assert TRANSFORMATION_SCHEMA.version == 1
    assert TRANSFORMATION_DOMAIN == TRANSFORMATION_SCHEMA.domain
    assert TRANSFORMATION_DOMAIN == "drift-transformation-identity-v1"


def test_the_transformation_digest_is_UNCHANGED_by_the_conversion():
    """The whole point of choosing an already-coherent family.

    Reconstructing the pre-conversion computation by hand must reproduce the
    live digest exactly. If it does not, the abstraction altered identity and
    the later source-evidence migration could not be attributed cleanly.
    """
    comps = tuple(
        TransformationComponent(kind=k, schema_version=1,
                                fingerprint=str(i) * 64)
        for i, k in enumerate(TransformationComponentKind))
    identity = TransformationIdentity.of(comps)
    hand_built = domain_digest(
        "drift-transformation-identity-v1",
        {"schema_version": 1,
         "components": [c.as_record() for c in identity.components]})
    assert identity.digest == hand_built


def test_the_PER_COMPONENT_schema_version_is_a_different_concept():
    """`TransformationComponent.schema_version` versions ONE component's own
    record. `CanonicalDigestSchema.version` versions the identity EPOCH. The
    conversion must not have merged them."""
    a = TransformationIdentity.of((
        TransformationComponent(kind=TransformationComponentKind.MISSINGNESS,
                                schema_version=1, fingerprint="1" * 64),))
    b = TransformationIdentity.of((
        TransformationComponent(kind=TransformationComponentKind.MISSINGNESS,
                                schema_version=7, fingerprint="1" * 64),))
    assert a.digest != b.digest, "a component's own version must matter"
    assert TRANSFORMATION_SCHEMA.version == 1, (
        "the EPOCH version must not follow a component's version")
    assert a.components[0].as_record()["schema_version"] == 1
    assert b.components[0].as_record()["schema_version"] == 7


def test_the_module_no_longer_calls_domain_digest_DIRECTLY():
    """One authority means one call site.

    A module that still reaches past its schema object could reintroduce a
    second version literal without the schema object noticing.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "src"
           / "genomic_variant_classifier" / "provenance" / "transformation.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    called = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "domain_digest"]
    assert not called, "transformation.py calls domain_digest directly"
    literals = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.startswith("drift-transformation-identity-v")]
    assert not literals, (
        "a literal versioned domain remains: {}".format(
            [n.value for n in literals]))
