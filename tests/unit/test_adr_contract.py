"""Architecture decision records are a contract, not a naming convention.

Created 2026-08-22. ADR-0003 section 3 and section 2.

WHY
---
ADR-0003 reserved `docs/architecture/decisions/` as the canonical location for
accepted architecture decision records and rejected two alternatives, so that a
normative document class has one home rather than a runtime preference. A
directory convention that only prose enforces is not a contract.

The need was demonstrated by the records themselves. Measured on 2026-08-22
across all three accepted records:

    field                ADR-0001   ADR-0002   ADR-0003
    Status               PRESENT    PRESENT    PRESENT
    Date                 PRESENT    PRESENT    PRESENT
    Authority            PRESENT    PRESENT    PRESENT
    Measured at commit   PRESENT    PRESENT    PRESENT
    Domains              ABSENT     PRESENT    PRESENT     <-- ADR-METADATA-INCOMPLETE-1

ADR-0001 is the record that INTRODUCES the domain concept, and it declared no
domains of its own. Nobody noticed until the three headers were placed side by
side, which is the argument for measuring rather than reading.

It is amended in the same commit as this file, to `meta`: ADR-0001 does not
govern one domain, it defines the lattice by which every domain is assigned.
The amendment and this checker are one unit because separating them would leave
the suite red in between.

TWO HEADER SHAPES, BOTH CORRECT
-------------------------------
    **Author: Monzia Moodie**       the mandated byline -- name INSIDE the bold
    **Status:** accepted            metadata -- key inside, value outside

The byline form is fixed by the project's authorship rule. Only two forms are
acceptable anywhere: "Written by Monzia Moodie" and "Author: Monzia Moodie".
This file therefore checks the byline as a byline and the metadata as metadata,
rather than flattening both into one permissive pattern that would accept drift
in either.

WHAT IS DELIBERATELY NOT CHECKED
--------------------------------
Content. A record's reasoning, its rulings and its consequences are the author's
work and no test can validate them. This file checks that a record is
DISCOVERABLE, IDENTIFIABLE, and SELF-DESCRIBING: that it lives where accepted
records live, that its identifier is unique, that it declares its status,
authority, domains and the commit it was measured against, and that a
superseded record says what superseded it.

Five of the twelve tests are negative controls. They exist to prove the parser
and the vocabulary can REJECT, and they call the same functions the real
assertions call, so a control cannot pass against a broken checker.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ADR_DIR = Path("docs/architecture/decisions")

#: Every metadata key an accepted record must declare.
REQUIRED_FIELDS = frozenset({
    "Status",
    "Date",
    "Authority",
    "Domains",
    "Measured at commit",
})

#: The authority domains ADR-0001 defines, plus `meta` for the record that
#: defines the lattice itself and therefore governs no single domain.
DOMAIN_VOCABULARY = frozenset({
    "meta",
    "execution",
    "data_schema",
    "scientific_policy",
    "project_state",
    "historical_repository_record",
    "execution_evidence",
    "public_project_identity",
    "current_program_state",
    "development_notebook",
    "architectural_decision",
    "measured_execution_evidence",
})

VALID_STATUS = frozenset({"draft", "accepted", "superseded", "rejected"})

BYLINE = "**Author: Monzia Moodie**"
FILENAME = re.compile(r"^ADR-(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
FIELD = re.compile(r"^\*\*([A-Z][A-Za-z ]*):\*\*\s*(\S.*)$")

#: The generated index. It ENUMERATES the records, which makes it a second copy
#: of the record list -- precisely the shape that made README.md load-bearing
#: and produced a feature count stated in nine places with four values. It is
#: therefore bound to the directory by test_the_index_lists_exactly_the_records
#: below. A list nobody checks is a list that goes stale on a schedule.
INDEX = ADR_DIR / "README.md"
INDEX_ENTRY = re.compile(r"ADR-\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md")

#: How far into a record the header may extend. A metadata block that has
#: drifted into the body is not a header.
HEADER_LINES = 12


# ---------------------------------------------------------------------------
# Parsing. Factored out so the negative controls exercise the SHIPPING code.
# ---------------------------------------------------------------------------

def parse_header(text: str) -> dict[str, str]:
    """Return the metadata block of a record. Byline excluded -- it is not a field.

    A field is `**Key:** value` with the colon INSIDE the bold and a non-empty
    value OUTSIDE it. `**Author: Monzia Moodie**` deliberately does not match:
    the byline is checked separately, as a byline.
    """
    found: dict[str, str] = {}
    for line in text.split("\n")[:HEADER_LINES]:
        m = FIELD.match(line)
        if m:
            found[m.group(1).strip()] = m.group(2).strip()
    return found


def split_domains(value: str) -> list[str]:
    return [d.strip() for d in value.split(",") if d.strip()]


def unknown_domains(value: str) -> list[str]:
    return sorted(d for d in split_domains(value) if d not in DOMAIN_VOCABULARY)


def record_number(name: str) -> int | None:
    m = FILENAME.match(name)
    return int(m.group(1)) if m else None


def index_entries(text: str) -> frozenset[str]:
    """Every record filename the index mentions, however it is formatted."""
    return frozenset(INDEX_ENTRY.findall(text))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records() -> dict[str, str]:
    if not ADR_DIR.is_dir():
        pytest.fail(
            f"{ADR_DIR} does not exist. ADR-0003 reserves it as the canonical "
            f"location for accepted architecture decision records; if it moved, "
            f"amend ADR-0003 and this locator together."
        )
    found = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(ADR_DIR.iterdir())
        if p.is_file() and p.name != "README.md"
    }
    assert found, (
        f"no records found in {ADR_DIR}. Three are accepted as of 2026-08-22; "
        f"a checker with nothing to check is not a checker."
    )
    return found


# ---------------------------------------------------------------------------
# 1. Discoverability and identity
# ---------------------------------------------------------------------------

def test_every_record_matches_the_canonical_filename_pattern(records):
    """ADR-NNNN-lowercase-hyphenated-slug.md, with no exceptions."""
    bad = sorted(n for n in records if not FILENAME.match(n))
    assert not bad, (
        f"these files do not match ADR-NNNN-slug.md: {bad}\n\n"
        f"The pattern is what makes a record findable by identifier rather than "
        f"by remembering its title. If a file in this directory is not a "
        f"record, it does not belong here."
    )


def test_record_identifiers_are_unique_and_contiguous(records):
    """Duplicate identifiers break provenance; gaps hide a deleted record."""
    numbers = sorted(record_number(n) for n in records)
    assert len(numbers) == len(set(numbers)), (
        f"duplicate record identifiers: {numbers}"
    )
    expected = list(range(1, len(numbers) + 1))
    assert numbers == expected, (
        f"record identifiers are {numbers}; expected {expected}.\n\n"
        f"A gap means a record was removed rather than superseded. A record is "
        f"superseded by a later record that says so -- never by deletion."
    )


def test_no_accepted_record_lives_outside_the_canonical_directory():
    """ADR-0003 rejected docs/adr/ and dated strategy subdirectories."""
    strays = [
        str(p) for p in Path("docs").rglob("ADR-*.md")
        if ADR_DIR not in p.parents
    ]
    assert not strays, (
        f"records outside {ADR_DIR}: {strays}\n\n"
        f"ADR-0003 rejected docs/adr/ (flattens records into a top-level "
        f"special case) and docs/strategy/<date>/ (an accepted record must "
        f"outlive the session that produced it). One normative document class, "
        f"one canonical location."
    )


# ---------------------------------------------------------------------------
# 2. Self-description
# ---------------------------------------------------------------------------

def test_every_record_carries_the_mandated_byline(records):
    """Exactly `**Author: Monzia Moodie**`. Never `Written FOR`."""
    missing = sorted(n for n, t in records.items()
                     if BYLINE not in t.split("\n")[:HEADER_LINES])
    assert not missing, (
        f"these records do not carry the byline {BYLINE!r} in their first "
        f"{HEADER_LINES} lines: {missing}"
    )


def test_every_record_declares_the_required_metadata(records):
    """ADR-METADATA-INCOMPLETE-1, measured 2026-08-22.

    ADR-0001 declared no Domains -- and ADR-0001 is the record that introduces
    the domain concept. Amended to `meta` in the same commit as this file:
    it does not govern one domain, it defines the lattice.
    """
    problems = []
    for name, text in sorted(records.items()):
        header = parse_header(text)
        absent = sorted(REQUIRED_FIELDS - set(header))
        if absent:
            problems.append(f"    {name}: missing {absent}")
    assert not problems, (
        "these records do not declare the required metadata:\n"
        + "\n".join(problems)
        + f"\n\nRequired: {sorted(REQUIRED_FIELDS)}\n"
          f"Each must appear as `**Key:** value` within the first "
          f"{HEADER_LINES} lines. Do not relax this to make a record pass -- "
          f"add the field to the record."
    )


def test_every_declared_status_is_in_the_vocabulary(records):
    wrong = {n: parse_header(t)["Status"] for n, t in records.items()
             if parse_header(t).get("Status") not in VALID_STATUS}
    assert not wrong, (
        f"unrecognised status values: {wrong}. Valid: {sorted(VALID_STATUS)}"
    )


def test_every_declared_domain_is_in_the_vocabulary(records):
    """A free-form domain string is a second vocabulary nobody agreed to."""
    problems = []
    for name, text in sorted(records.items()):
        value = parse_header(text).get("Domains", "")
        unknown = unknown_domains(value)
        if unknown:
            problems.append(f"    {name}: {unknown}  (declared: {value!r})")
    assert not problems, (
        "these records declare domains outside the vocabulary:\n"
        + "\n".join(problems)
        + f"\n\nVocabulary: {sorted(DOMAIN_VOCABULARY)}\n"
          f"If a genuinely new domain is needed, add it here AND to ADR-0001 "
          f"in the same commit -- not to one of them."
    )


def test_a_superseded_record_says_what_superseded_it(records):
    """Superseded without a successor is an orphan, not a decision."""
    orphans = [n for n, t in records.items()
               if parse_header(t).get("Status") == "superseded"
               and "Superseded by" not in parse_header(t)]
    assert not orphans, (
        f"these records are marked superseded but name no successor: {orphans}"
    )


# ---------------------------------------------------------------------------
# 3. Negative controls -- proof that the checks can REJECT
# ---------------------------------------------------------------------------

def test_the_parser_rejects_the_byline_as_a_metadata_field():
    """The byline and a field are different shapes, and must stay different.

    `**Author: Monzia Moodie**` puts the colon INSIDE the bold. If the parser
    accepted it as a field, a record could satisfy the metadata contract with a
    byline alone.
    """
    assert parse_header(BYLINE) == {}, (
        "the parser accepted the byline as a metadata field; the two shapes "
        "have been flattened and drift in either is now invisible."
    )


def test_the_parser_rejects_a_field_with_an_empty_value():
    assert parse_header("**Status:**") == {}, (
        "a key with no value was accepted as a declaration."
    )
    assert parse_header("**Status:**   ") == {}, (
        "a key with only whitespace was accepted as a declaration."
    )


def test_the_parser_ignores_metadata_that_has_drifted_out_of_the_header():
    """A field on line 40 is body text, not a header declaration."""
    text = "\n".join(["# title"] + [""] * HEADER_LINES + ["**Status:** accepted"])
    assert "Status" not in parse_header(text), (
        f"a field beyond line {HEADER_LINES} was accepted as header metadata."
    )


def test_the_domain_check_rejects_an_unknown_domain():
    assert unknown_domains("execution, not_a_real_domain") == \
        ["not_a_real_domain"], (
        "the domain vocabulary check failed to reject an unknown domain."
    )
    assert unknown_domains("meta, execution") == [], (
        "the domain vocabulary check rejected two legitimate domains."
    )


def test_the_filename_check_rejects_near_misses():
    """Each of these is wrong in exactly one way."""
    for bad in ("ADR-1-short-number.md",
                "ADR-0004-Has-Capitals.md",
                "ADR-0005_underscored.md",
                "ADR-0006-trailing-.md",
                "adr-0007-lowercase-prefix.md",
                "ADR-0008-no-extension"):
        assert FILENAME.match(bad) is None, (
            f"the filename check accepted {bad!r}, which it must reject."
        )
    assert FILENAME.match("ADR-0009-a-valid-slug.md") is not None, (
        "the filename check rejected a valid name."
    )


# ---------------------------------------------------------------------------
# 4. The index is bound to the directory, not merely written beside it
# ---------------------------------------------------------------------------

def test_the_index_lists_exactly_the_records_present(records):
    """A generated index that nobody checks is a second unenforced authority.

    ADR-0003 makes the point in general terms: derived presentation is not the
    source of truth, and a count written down twice goes stale. The index is
    derived presentation over the directory. This binds it.
    """
    assert INDEX.is_file(), (
        f"{INDEX} does not exist. The index carries the canonical convention "
        f"and enumerates the accepted records; without it, the directory is a "
        f"folder rather than a documented contract."
    )
    listed = index_entries(INDEX.read_text(encoding="utf-8"))
    present = frozenset(records)
    missing = sorted(present - listed)
    phantom = sorted(listed - present)
    assert not missing and not phantom, (
        f"THE INDEX DISAGREES WITH {ADR_DIR}.\n"
        f"  present but NOT listed ({len(missing)}): {missing}\n"
        f"  listed but NOT present ({len(phantom)}): {phantom}\n\n"
        f"Add the record to the index in the same commit that adds the record. "
        f"Do not delete this assertion: an index that has drifted from the "
        f"directory is worse than no index, because it looks authoritative."
    )


def test_the_index_check_detects_a_missing_and_a_phantom_entry():
    """NEGATIVE CONTROL, in both directions."""
    present = frozenset({"ADR-0001-alpha.md", "ADR-0002-beta.md"})
    listed = index_entries(
        "- [ADR-0001-alpha.md](ADR-0001-alpha.md)\n"
        "- [ADR-0099-ghost.md](ADR-0099-ghost.md)\n"
    )
    assert sorted(present - listed) == ["ADR-0002-beta.md"], (
        "the index check failed to detect a record absent from the index."
    )
    assert sorted(listed - present) == ["ADR-0099-ghost.md"], (
        "the index check failed to detect an index entry with no record."
    )
