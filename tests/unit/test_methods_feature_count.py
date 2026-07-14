"""METHODS.md must state the real feature count. Enforced, not trusted.

Created 2026-07-13 (roadmap 6.22).

WHY
---
METHODS.md is the scientific description of this model -- the document a reviewer, a
collaborator, or a regulator reads to understand what was actually built. On 2026-07-13 it
said:

    "A total of **64 tabular features** were derived from raw annotations."

The real figure was **95**. The group table beneath it summed to **62**, agreeing with neither.
And it listed **HGMD Professional** as data source 12, supplying a "disease mutation flag" and
a "report count" -- a source whose licence was never obtained, whose connector was never wired,
and whose two columns were CONSTANT ZERO for the entire life of the project.

So the methods document described a model that did not exist, in three separate ways, and
nothing anywhere checked. It could not have been caught by a code review; it is a markdown
file. It could only be caught by re-deriving it.

That is root pattern (a) -- a number written down once and never re-derived becomes a lie on a
schedule -- and it is worse here than in code, because a wrong number in a methods document is
the one that ends up in a paper.

    A COMMENT DOES NOT ENFORCE ITSELF. A DOCUMENT ENFORCES ITSELF EVEN LESS.
    MAKE FORGETTING FAIL.

This test re-derives the agreement on every run. Add a feature and forget METHODS.md, and the
suite goes red, naming both numbers.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    EXPECTED_TABULAR_FEATURE_COUNT,
    TABULAR_FEATURES,
)

METHODS = Path("METHODS.md")


@pytest.fixture(scope="module")
def methods_text() -> str:
    if not METHODS.is_file():
        pytest.fail(f"METHODS.md not found at {METHODS.resolve()}")
    return METHODS.read_text(encoding="utf-8")


def test_methods_states_the_real_feature_count(methods_text):
    """The headline sentence must carry the true number."""
    m = re.search(r"A total of \*\*(\d+) tabular features\*\*", methods_text)
    assert m, (
        "Could not find the sentence 'A total of **N tabular features**' in METHODS.md. "
        "If it was reworded, update this test -- do NOT delete it. It is the only thing "
        "standing between the methods document and a number that quietly goes stale, which "
        "is exactly what happened between the 64-feature era and the 95-feature one."
    )

    stated = int(m.group(1))
    assert stated == EXPECTED_TABULAR_FEATURE_COUNT == len(TABULAR_FEATURES), (
        f"METHODS.md says {stated} tabular features.\n"
        f"EXPECTED_TABULAR_FEATURE_COUNT is {EXPECTED_TABULAR_FEATURE_COUNT}.\n"
        f"len(TABULAR_FEATURES) is {len(TABULAR_FEATURES)}.\n"
        f"\n"
        f"These must agree. METHODS.md is the scientific description of this model; a wrong "
        f"number there is the one that ends up in a paper. Update METHODS.md IN THE SAME "
        f"COMMIT as the feature change."
    )


def test_the_methods_group_table_sums_to_the_feature_count(methods_text):
    """The table must not merely LOOK right -- it must add up.

    The pre-2026-07-13 table summed to 62 while the prose claimed 64 and the code held 95:
    three numbers, no two of which agreed, and nothing that ever added the column up.
    """
    # The group table rows look like: | Group name | 6 | description |
    # The bolded Total row is excluded -- it is the assertion, not a member of the sum.
    rows = re.findall(r"^\|\s*(?!\*\*Total)([^|]+?)\s*\|\s*(\d+)\s*\|", methods_text, re.M)
    assert rows, "No feature-group table rows found in METHODS.md."

    total = sum(int(n) for _, n in rows)
    assert total == EXPECTED_TABULAR_FEATURE_COUNT, (
        f"The METHODS.md feature-group table sums to {total}, but the contract holds "
        f"{EXPECTED_TABULAR_FEATURE_COUNT} features.\n"
        f"\n"
        f"Groups found ({len(rows)}):\n"
        + "\n".join(f"    {g:44s} {n:>3s}" for g, n in rows)
    )


def test_methods_does_not_claim_hgmd_as_a_data_source(methods_text):
    """HGMD was never integrated. The document must not say it was.

    It listed HGMD Professional as source 12 -- "Disease mutation flag, report count" -- and
    counted its two features in the Gene-disease group. The licence was never obtained, the
    connector was never wired, and both columns were constant zero throughout. A methods
    document that credits a source the model never had is not a stale detail; it is a false
    claim about the science.

    The document may (and does) DISCUSS HGMD -- explaining why it is absent and why it must
    never return as a variant-level feature. What it must not do is list it as a source that
    supplied features.
    """
    source_table = re.findall(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|", methods_text, re.M)
    hgmd_sources = [s for s in source_table if "hgmd" in s.lower()]
    assert not hgmd_sources, (
        f"METHODS.md lists HGMD as a numbered data source: {hgmd_sources}. "
        f"It never supplied a single non-zero value. Removed from the feature contract "
        f"2026-07-13; it must not be credited as a source."
    )

    assert "hgmd_is_disease_mutation" not in TABULAR_FEATURES
    assert "hgmd_n_reports" not in TABULAR_FEATURES
