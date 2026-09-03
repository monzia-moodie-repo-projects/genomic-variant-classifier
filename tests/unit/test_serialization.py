"""The digest primitive, tested where it lives.

Phase 1C Unit 3A++.3. Created 2026-09-03.

WHY THIS FILE EXISTS
--------------------
MEASURED BY SABOTAGE 2026-09-03, against the full 405-identity provenance
suite: three guarantees stated by `serialization.py` had NO test at all.

    ensure_ascii disabled                 405 passed -- NOTHING detected
    an empty domain admitted              405 passed -- NOTHING detected
    the separator-in-domain check dropped 405 passed -- NOTHING detected

The module's only direct coverage was four assertions inside
`test_source_release.py`, which check domain separation, version separation,
the version-suffix requirement and key sorting. Those remain: they assert that
the SOURCE-RELEASE digest path is domain-separated, which is an integration
claim at that level. This file tests the PRIMITIVE exhaustively.

THE ASCII INVARIANT WAS ASSERTED IN PROSE ONLY
----------------------------------------------
`serialization.py` line 54 states that `ensure_ascii=True` exists "so the bytes
cannot depend on a locale or a filesystem encoding". MEASURED: not one byte of
non-ASCII appears in any test file, and every JSON fixture in
`tests/fixtures/` is pure ASCII -- the only high bytes in the corpora are
PICKLE FRAMING, not payload. So disabling the flag changed nothing.

It is not a no-op. For a source spelled with a non-ASCII character:

    ensure_ascii=True    {"source":"clinv\\u00e5r"}     pure ASCII bytes
    ensure_ascii=False   {"source":"clinv\\xc3\\xa5r"}   UTF-8 bytes

A source name or release identifier CAN carry a non-ASCII character. Under the
second form the digest of the same scientific evidence could differ between
machines -- a digest changing for a reason having nothing to do with the
evidence, which is the failure this subsystem exists to prevent.

`ASCII-INVARIANT-ASSERTED-IN-PROSE-ONLY-1`, closed here.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit; JSON = JavaScript Object
Notation; ASCII = American Standard Code for Information Interchange;
UTF-8 = Unicode Transformation Format, 8-bit; NUL = the zero byte.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json

import pytest

from genomic_variant_classifier.provenance import canonical_json, domain_digest
from genomic_variant_classifier.provenance.serialization import (
    DigestDomainError,
    ParsedDigestDomain,
    parse_versioned_domain,
)

#: A source spelling that is valid, plausible and NOT ASCII. Written as an
#: escape so this file itself stays pure ASCII -- the repository convention --
#: while still exercising a non-ASCII code point.
NON_ASCII = "clinv\u00e5r"
ASTRAL = "\U0001f9ec"          # a code point outside the Basic Multilingual Plane


# ---------------------------------------------------------------------------
# 1. THE ASCII INVARIANT -- the gap sabotage exposed
# ---------------------------------------------------------------------------

def test_canonical_json_is_ASCII_even_for_non_ascii_input():
    """The invariant `serialization.py` states and nothing tested.

    Under `ensure_ascii=False` this returns UTF-8 bytes instead, and the same
    scientific evidence could digest differently across machines.
    """
    out = canonical_json({"source": NON_ASCII})
    assert all(b < 128 for b in out), out
    assert out == b'{"source":"clinv\\u00e5r"}'


def test_a_non_ascii_payload_still_ROUND_TRIPS():
    """Escaping must not lose the character."""
    out = canonical_json({"source": NON_ASCII})
    assert json.loads(out.decode("ascii"))["source"] == NON_ASCII


def test_an_ASTRAL_code_point_is_also_escaped():
    """Beyond the Basic Multilingual Plane, where surrogate pairs appear."""
    out = canonical_json({"emoji": ASTRAL})
    assert all(b < 128 for b in out), out
    assert json.loads(out.decode("ascii"))["emoji"] == ASTRAL


def test_the_DIGEST_of_non_ascii_evidence_is_stable():
    """The scientific consequence, stated as a value.

    If this digest ever changes, the serialisation changed -- and every
    manifest naming a non-ASCII source would silently acquire a new identity.
    """
    got = domain_digest("probe-family-v1", {"source": NON_ASCII})
    expected = hashlib.sha256(
        b"genomic-variant-classifier:probe-family-v1\x00"
        b'{"source":"clinv\\u00e5r"}').hexdigest()
    assert got == expected


def test_two_spellings_that_NORMALIZE_alike_are_NOT_the_same_identity():
    """Composed and decomposed forms are different code point sequences.

    `serialization.py` performs no Unicode normalisation, and this test pins
    that: a caller must canonicalise a source name BEFORE it becomes an
    identity, not hope the digest layer does it. Silently normalising would
    merge two spellings a registry may deliberately distinguish.
    """
    composed = "\u00e5"                 # a with ring above, one code point
    decomposed = "a\u030a"              # a + combining ring above
    assert composed != decomposed
    assert canonical_json({"s": composed}) != canonical_json({"s": decomposed})


# ---------------------------------------------------------------------------
# 2. THE DOMAIN GUARDS -- untested until now
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["", None, 0, b"x-v1", []],
                         ids=["empty", "none", "int", "bytes", "list"])
def test_a_domain_that_cannot_BE_one_is_refused(domain):
    with pytest.raises(ValueError):
        domain_digest(domain, {"a": 1})


def test_an_EMPTY_FAMILY_is_refused():
    """MEASURED BY SABOTAGE 2026-09-03: widening the grammar's family group
    from `.+` to `.*` changed NO test, because the explicit empty-family guard
    still fired -- a no-op, but one that left that guard untested. A domain
    that declares only a version identifies nothing."""
    with pytest.raises(DigestDomainError):
        parse_versioned_domain("-v1")
    with pytest.raises(DigestDomainError):
        domain_digest("-v1", {"a": 1})


@pytest.mark.parametrize(
    "domain",
    ["famil\u00e5-v1", "\u00c5ngstrom-v2", "family-v1\u00e5"],
    ids=["accented-family", "leading-accent", "accented-tail"])
def test_a_NON_ASCII_domain_is_refused(domain):
    """The gap sabotage exposed.

    MEASURED 2026-09-03: removing the ASCII guard changed NO test. Every
    Unicode test in this file passes a non-ASCII PAYLOAD; none passed a
    non-ASCII DOMAIN, and the grammar alone happily matches one -- `.+`
    accepts any character.

    Without the guard the failure surfaces later as a `UnicodeEncodeError`
    from `.encode("ascii")`: a different exception type, from a different
    line, carrying no explanation of what the caller did wrong.
    """
    with pytest.raises(DigestDomainError) as exc:
        parse_versioned_domain(domain)
    assert "ASCII" in str(exc.value)
    with pytest.raises(DigestDomainError):
        domain_digest(domain, {"a": 1})


def test_the_ASCII_domain_guard_precedes_the_ENCODE():
    """The refusal must be a DigestDomainError, never a UnicodeEncodeError.

    A caller catching the documented error type would otherwise miss it.
    """
    with pytest.raises(DigestDomainError):
        domain_digest("famil\u00e5-v1", {"a": 1})
    try:
        domain_digest("famil\u00e5-v1", {"a": 1})
    except UnicodeEncodeError:                              # pragma: no cover
        pytest.fail("the guard did not precede the ASCII encode")
    except DigestDomainError:
        pass


def test_a_domain_containing_the_NUL_SEPARATOR_is_refused():
    """The NUL byte marks where the domain ends and the payload begins.

    A domain carrying one would make that boundary ambiguous, so two different
    (domain, payload) pairs could produce identical prefixed bytes.
    """
    with pytest.raises(ValueError) as exc:
        domain_digest("bad\x00domain-v1", {"a": 1})
    assert "separator" in str(exc.value)


@pytest.mark.parametrize("domain", ["no-version", "trailing-v", "v1-leading",
                                    "family-v", "family-1"],
                         ids=["none", "bare-v", "leading", "v-no-digits",
                                "digits-no-v"])
def test_a_domain_without_a_VERSION_SUFFIX_is_refused(domain):
    with pytest.raises(ValueError) as exc:
        domain_digest(domain, {"a": 1})
    assert "version" in str(exc.value)


@pytest.mark.parametrize("domain", ["family-v1", "family-v12", "a-b-c-v99"],
                         ids=["v1", "v12", "hyphens"])
def test_a_WELL_FORMED_domain_is_accepted(domain):
    """The sensitivity half: the guard must not refuse valid domains."""
    got = domain_digest(domain, {"a": 1})
    assert len(got) == 64 and all(c in "0123456789abcdef" for c in got)


# ---------------------------------------------------------------------------
# 2b. THE EPOCH GRAMMAR -- one spelling per version, over a complete space
# ---------------------------------------------------------------------------

def test_the_small_epoch_LANGUAGE_has_ONE_spelling_per_version():
    """A property over a complete small space, not three chosen examples.

    Section XI of the design authority. If `v1`, `v01` and `v001` were all
    accepted, one numerical epoch would have three byte-distinct namespaces --
    three digests for one meaning, which is the invariant this subsystem
    exists to enforce.
    """
    suffixes = ("", "0", "00", "01", "001", "0100",
                "1", "2", "9", "10", "11", "99", "100")
    accepted = {}
    for suffix in suffixes:
        domain = "family-v{}".format(suffix)
        try:
            parsed = parse_versioned_domain(domain)
        except DigestDomainError:
            continue
        accepted.setdefault(parsed.version, []).append(domain)
    assert accepted == {
        1: ["family-v1"], 2: ["family-v2"], 9: ["family-v9"],
        10: ["family-v10"], 11: ["family-v11"], 99: ["family-v99"],
        100: ["family-v100"],
    }, accepted


def test_the_parser_returns_the_TWO_things_a_domain_declares():
    parsed = parse_versioned_domain("drift-source-evidence-manifest-v5")
    assert isinstance(parsed, ParsedDigestDomain)
    assert parsed.family == "drift-source-evidence-manifest"
    assert parsed.version == 5
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.version = 6


def test_a_family_may_itself_CONTAIN_hyphens_and_digits():
    """The family is everything before the LAST `-vN`."""
    parsed = parse_versioned_domain("a-b-2-c-v7")
    assert parsed.family == "a-b-2-c" and parsed.version == 7


def test_DigestDomainError_is_a_ValueError():
    """So every existing `pytest.raises(ValueError)` still catches it."""
    assert issubclass(DigestDomainError, ValueError)


# ---------------------------------------------------------------------------
# 2c. NON-FINITE NUMBERS -- refused, and semantic-zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")],
                         ids=["nan", "inf", "-inf"])
def test_a_NON_FINITE_number_is_refused(value):
    """`json.dumps` emits the non-standard tokens NaN, Infinity and -Infinity
    unless told otherwise, and no conforming reader accepts them.

    MEASURED 2026-09-03 before this was enabled: the evidence payload holds
    only NoneType, int and str, and the transformation payload only int and
    str. No live canonical record contains a float, so the refusal moved no
    digest.
    """
    with pytest.raises(ValueError):
        canonical_json({"value": value})
    with pytest.raises(ValueError):
        domain_digest("family-v1", {"value": value})


def test_a_non_finite_number_NESTED_deeply_is_also_refused():
    with pytest.raises(ValueError):
        canonical_json({"a": {"b": [1, {"c": float("nan")}]}})


def test_ORDINARY_numbers_are_unaffected():
    """The sensitivity half."""
    assert canonical_json({"a": 1, "b": 1.5, "c": -0.25, "d": 0}) \
        == b'{"a":1,"b":1.5,"c":-0.25,"d":0}'


# ---------------------------------------------------------------------------
# 2d. CONTENT DISCRIMINATION -- the digest must incorporate its payload
# ---------------------------------------------------------------------------

def test_the_digest_DISCRIMINATES_over_payloads():
    """Section XIII. Generalised from a measured defect.

    Sabotage on 2026-09-02 replaced an evidence payload with an empty list:
    every manifest then digested identically, and no test noticed, because
    each assertion only asked whether a digest had MOVED. A digest that
    ignores its content is not an identity.
    """
    values = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": [1]}, {"x": [1, 2]},
              {"y": 1}, {}, {"x": None}, {"x": "1"}]
    digests = {domain_digest("test-v1", v) for v in values}
    assert len(digests) == len(values)


def test_a_TYPE_change_alone_changes_the_digest():
    """`1` and `"1"` are different scientific values."""
    assert domain_digest("test-v1", {"x": 1}) \
        != domain_digest("test-v1", {"x": "1"})
    assert domain_digest("test-v1", {"x": None}) \
        != domain_digest("test-v1", {"x": "null"})


# ---------------------------------------------------------------------------
# 3. domain separation, restated at the primitive
# ---------------------------------------------------------------------------

def test_the_domain_PARTICIPATES_in_the_digest():
    payload = {"a": 1}
    assert domain_digest("one-v1", payload) != domain_digest("two-v1", payload)
    assert domain_digest("one-v1", payload) != domain_digest("one-v2", payload)


def test_the_domain_is_PREFIXED_not_appended():
    """Order matters: a suffix would let a crafted payload imitate a domain."""
    got = domain_digest("family-v1", {"a": 1})
    prefix = b"genomic-variant-classifier:family-v1\x00"
    assert got == hashlib.sha256(prefix + b'{"a":1}').hexdigest()
    assert got != hashlib.sha256(b'{"a":1}' + prefix).hexdigest()


def test_the_NAMESPACE_participates():
    """Without it, another project's digest of the same record would collide."""
    assert domain_digest("family-v1", {"a": 1}) != hashlib.sha256(
        b"family-v1\x00" + b'{"a":1}').hexdigest()


# ---------------------------------------------------------------------------
# 4. canonical form
# ---------------------------------------------------------------------------

def test_key_ORDER_cannot_alter_identity():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_WHITESPACE_cannot_alter_identity():
    """No spaces after separators, at any depth."""
    out = canonical_json({"a": [1, 2], "b": {"c": 3}})
    assert out == b'{"a":[1,2],"b":{"c":3}}'
    assert b" " not in out


def test_NESTING_depth_does_not_need_a_new_rule():
    """The reason canonical JSON replaced a flat separator scheme."""
    deep = {"a": {"b": {"c": [{"d": 1}, {"e": 2}]}}}
    once = canonical_json(deep)
    assert once == canonical_json(json.loads(once.decode("ascii")))


def test_an_EMPTY_payload_still_digests():
    assert len(domain_digest("family-v1", {})) == 64
    assert canonical_json({}) == b"{}"


def test_a_LIST_and_a_DICT_are_different_identities():
    assert canonical_json([1, 2]) != canonical_json({"0": 1, "1": 2})
