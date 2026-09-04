"""Shared identity-law assertions over already-derived identity tokens.

Phase 1C Unit 3A++.4b. Created 2026-09-03.

A helper belongs here only when it:

    1. expresses a semantic relation rather than merely shortening syntax;
    2. has at least one current consumer;
    3. replaces existing inline reasoning at introduction;
    4. performs no domain normalisation;
    5. operates on already-derived identity tokens;
    6. has been shown to catch a real or sabotaged defect.

Do not add helpers speculatively for future identity families.

WHY THIS MODULE EXISTS RATHER THAN REUSING AN EXISTING ONE
----------------------------------------------------------
MEASURED 2026-09-03, law-authority census over 234 source, 365 test and 452
script modules: 77 candidate equality or grouping functions, and NOT ONE has
the semantics scientific identity requires.

    partitions_equivalent        collapses None, "", whitespace and NaN, and
                                 treats cluster labels as interchangeable
    evaluate_partition_agreement returns a metric, not a relation
    exact_duplicate_groups       operates on pandas columns
    legacy_values_equal          treats NaN as equal to NaN
    assert_same_representation   representation-specific

Reusing any of them would let domain normalisation leak into scientific
identity. For provenance, `None != ""` and `"GRCh38" != "grch38"` unless a
canonical schema explicitly says otherwise -- and any such normalisation
belongs at an ADMISSION BOUNDARY, before an identity is minted, never inside a
comparison.

THE CENSUS ALSO EXPOSED A LIMIT OF ITS OWN METHOD.
`partitions_equivalent` contains none of the normalisation tokens the scan
searched for, because the normalisation happens in a helper it calls. Lexical
inspection of one function body does not establish its semantics
(`LAW-AUTHORITY-CENSUS-BODY-SCAN-MISSES-DELEGATED-NORMALISATION-1`).

WHAT THIS MODULE IS NOT
-----------------------
It is the authority for HOW WE VERIFY mathematical properties of identity
outputs. It is NOT the authority for WHAT CONSTITUTES scientific identity --
that lives in `provenance/source.py`, `provenance/transformation.py`, and
whatever families follow.

It imports nothing but the Python standard library: no production module, no
NumPy, no pandas. Dependencies run tests -> production, never the reverse.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Tuple


IdentityToken = str
IdentityCases = Mapping[str, IdentityToken]
IdentityFamilies = FrozenSet[str]


def _validate_cases(
        cases: IdentityCases,
        *,
        where: str) -> Tuple[Tuple[str, IdentityToken], ...]:
    """Snapshot and validate labelled, already-derived identity tokens."""
    if not isinstance(cases, Mapping):
        raise TypeError(
            "{} must be a mapping of label -> identity token, got {}"
            .format(where, type(cases).__name__))

    if not cases:
        raise ValueError("{} must not be empty".format(where))

    items = tuple(cases.items())

    for label, token in items:
        if not isinstance(label, str) or not label:
            raise TypeError(
                "{} case labels must be non-empty strings; got {!r}"
                .format(where, label))

        if not isinstance(token, str) or not token:
            raise TypeError(
                "{}[{!r}] must be a non-empty identity token string; "
                "got {!r}. The laws compare already-derived identity "
                "outputs and will not coerce a value into one."
                .format(where, label, token))

    return items


def _pairwise_relation(
        cases: IdentityCases,
        *,
        where: str = "cases",
) -> Dict[Tuple[str, str], bool]:
    """Return equality for every unordered pair of labelled cases.

    NO NORMALISATION. Comparison is `==`, never `is`. This project has three
    times found a defect hidden by object identity holding accidentally: a
    reused singleton, a short interned literal, and a constant-folded
    `"a" * 64`.
    """
    items = _validate_cases(cases, where=where)
    relation: Dict[Tuple[str, str], bool] = {}

    for index, (left_label, left_token) in enumerate(items):
        for right_label, right_token in items[index + 1:]:
            relation[(left_label, right_label)] = (
                left_token == right_token
            )

    return relation


def assert_identity_equivalence_preserved(
        before: IdentityCases,
        after: IdentityCases) -> None:
    """Assert that a transition preserves the complete equality relation."""
    before_items = _validate_cases(before, where="before")
    after_items = _validate_cases(after, where="after")

    before_labels = tuple(label for label, _ in before_items)
    after_labels = tuple(label for label, _ in after_items)

    if before_labels != after_labels:
        raise AssertionError(
            "identity case population changed\n"
            "  before: {!r}\n"
            "  after:  {!r}\n"
            "A claim about equivalence is a claim about a NAMED "
            "POPULATION of cases."
            .format(before_labels, after_labels))

    before_relation = _pairwise_relation(before, where="before")
    after_relation = _pairwise_relation(after, where="after")

    changed_pairs = {
        pair: (before_relation[pair], after_relation[pair])
        for pair in before_relation
        if before_relation[pair] != after_relation[pair]
    }

    if changed_pairs:
        lines = [
            "  {} {} {} became {}".format(
                left,
                "==" if was_equal else "!=",
                right,
                "==" if is_equal else "!=",
            )
            for (left, right), (was_equal, is_equal)
            in sorted(changed_pairs.items())
        ]

        raise AssertionError(
            "identity equivalence relation changed:\n"
            + "\n".join(lines))


def assert_all_identities_distinct(
        cases: IdentityCases) -> None:
    """Assert injectivity over caller-declared scientifically distinct cases."""
    relation = _pairwise_relation(cases, where="cases")
    collisions = sorted(
        pair for pair, same in relation.items() if same
    )

    if collisions:
        raise AssertionError(
            "distinct identity cases collided:\n"
            + "\n".join(
                "  {} == {}".format(left, right)
                for left, right in collisions
            ))


def assert_orthogonal_change(
        *,
        before: IdentityCases,
        after: IdentityCases,
        changed: IdentityFamilies) -> None:
    """Assert that exactly the declared identity families changed.

    `changed` is an EXACT expectation, not an allow-list: every family named
    must move and every family not named must hold. An empty frozenset is
    valid and means that no family may move.

    THIS FUNCTION CARRIES THE SECOND EQUALITY SITE IN THE KERNEL. Orthogonality
    is a different mathematical statement from the pairwise relation and is
    deliberately not routed through `_pairwise_relation`, so the comparison
    below needs its own object-identity protection. MEASURED 2026-09-03:
    mutating it from `!=` to `is not` passed all 24 tests of the original
    suite, because every fixture used short interned literals where `is` and
    `==` coincide.
    """
    before_items = _validate_cases(before, where="before")
    after_items = _validate_cases(after, where="after")

    before_labels = tuple(label for label, _ in before_items)
    after_labels = tuple(label for label, _ in after_items)

    if before_labels != after_labels:
        raise AssertionError(
            "identity-family population changed\n"
            "  before: {!r}\n"
            "  after:  {!r}"
            .format(before_labels, after_labels))

    if not isinstance(changed, frozenset):
        raise TypeError(
            "changed must be a frozenset of non-empty identity-family "
            "names, got {}".format(type(changed).__name__))

    for family in changed:
        if not isinstance(family, str) or not family:
            raise TypeError(
                "changed family names must be non-empty strings; got {!r}"
                .format(family))

    known = frozenset(before_labels)
    unknown = sorted(changed - known)

    if unknown:
        raise ValueError(
            "changed names unknown identity families: {!r}"
            .format(unknown))

    before_token = dict(before_items)
    after_token = dict(after_items)

    observed = frozenset(
        label
        for label in before_labels
        if before_token[label] != after_token[label]
    )

    if observed != changed:
        unexpected = sorted(observed - changed)
        absent = sorted(changed - observed)

        raise AssertionError(
            "identity-family orthogonality violated\n"
            "  expected to change : {!r}\n"
            "  observed changed   : {!r}\n"
            "  changed but should NOT have : {!r}\n"
            "  should have changed but did NOT : {!r}"
            .format(
                sorted(changed),
                sorted(observed),
                unexpected,
                absent,
            ))
