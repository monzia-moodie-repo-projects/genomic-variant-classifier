"""Tests for the single ClinVar review-status tier map and its detector.

WHAT EACH TEST EXISTS TO CATCH, in the order the defects actually occurred
-------------------------------------------------------------------------
  1. A map key that stops resolving. The dead "no classification for the
     individual variant" key sat unmatched from at least 2026-05-08 because
     ClinVar had renamed it to "single variant" -- nothing failed, rows were
     silently demoted.
  2. A value present in the real cohort that the map does not cover. Measured
     2026-07-24 across both review-status columns: eleven distinct values, nine
     statuses and two missing markers.
  3. Silence on an unrecognised value. The whole incident chain is lookup misses
     becoming quality judgements.
  4. An escape hatch that is silent. Opting into the old behaviour must warn.
  5. A lookup that does not normalise, and the OPPOSITE error of normalising the
     clinical-significance path, which is exact and case-sensitive.
  6. The three resolution paths becoming indistinguishable. TIER_MISSING and
     TIER_UNMATCHED are both 5, so only TierResolution.path can tell them apart.
  7. Two implementations of one lookup drifting.
  8. A SECOND tier map appearing anywhere in the tree, under ANY name.
  9. The suite-size ratchet being guessed rather than measured.

WHY THE COLLECTION COUNT IS MEASURED IN A SUBPROCESS
----------------------------------------------------
The previous version of this file asserted

    EXPECTED_COLLECTED == len(REVIEW_STATUS_TIER) + len(OBSERVED_2026_07_24) + 8

while EXPECTED_COLLECTED was DEFINED as the left-hand side of that same sum. It
was a tautology, it could not fail, and its docstring claimed it made the ratchet
number impossible to guess wrong. It did nothing of the kind. This version asks
pytest, in a subprocess, how many items it actually collects from this file.
Collection does not execute tests, so there is no recursion.

Acronyms on first use. AST = abstract syntax tree. ClinVar = the National Center
for Biotechnology Information's Clinical Variation archive.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from genomic_variant_classifier.data.review_status import (
    MISSING_TOKENS,
    OBSERVED_2026_07_24,
    REVIEW_STATUS_SEMANTICS,
    REVIEW_STATUS_TIER,
    ReviewStatusSemanticClass,
    TIER_MISSING,
    TIER_UNMATCHED,
    TierResolutionPath,
    UnmatchedReviewStatusError,
    normalise,
    resolve,
    tier_of,
    tiers_for,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _review_map_detector as detector          # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_FILE = Path(__file__).resolve()

#: Every tier-map definition in the repository as of 2026-07-24, keyed on
#: path::NAME and NOT on line number, which shifts on unrelated edits.
#:
#: THIS IS A RATCHET, not a snapshot. Adding a map turns the suite red. REMOVING
#: one also turns it red, which forces this tuple to be edited in the same commit
#: that rewires a consumer -- so the inventory cannot drift in either direction.
#: It shrinks to four at Step 1b, to one at Step 1c, and is then replaced by an
#: exactly-once assertion.
#:
#: The eighth entry is the one a name-scoped guard could not see.
KNOWN_TIER_MAP_DEFINITIONS: tuple[str, ...] = (
    "scripts/augment_reviewstatus.py::REVIEW_STATUS_TIER",
    "scripts/clean_cohort.py::REVIEW_STATUS_TIER",
    "scripts/probe_review_status.py::REVIEW_STATUS_TIER",
    "scripts/probe_reviewstatus_gaps.py::REVIEW_STATUS_TIER",
    "scripts/probe_tier_filter_impact.py::REVIEW_STATUS_TIER",
    "src/genomic_variant_classifier/data/real_data_prep.py::REVIEW_STATUS_TIER",
    "src/genomic_variant_classifier/data/review_status.py::REVIEW_STATUS_TIER",
    "src/genomic_variant_classifier/monitoring/clinvar_tracker.py::REVIEW_TIER",
)


# --------------------------------------------------------------------------
# 1-2. Coverage
# --------------------------------------------------------------------------
@pytest.mark.parametrize("status,expected", sorted(REVIEW_STATUS_TIER.items()))
def test_every_map_key_resolves_to_its_own_tier(status: str, expected: int) -> None:
    """A key that stops resolving is exactly how the dead 'individual variant' key hid."""
    assert tier_of(status) == expected


@pytest.mark.parametrize("status", OBSERVED_2026_07_24)
def test_every_value_measured_in_the_cohort_resolves(status: str) -> None:
    """The complete distinct set of both review-status columns, 2026-07-24.

    The cohort is gitignored, so Continuous Integration cannot read it. Freezing
    the measured values means a map regression turns the suite red without it.
    """
    result = resolve(status)
    assert isinstance(result.tier, int)
    assert 1 <= result.tier <= 5
    if normalise(status) in MISSING_TOKENS:
        assert result.path is TierResolutionPath.MISSING_TOKEN
    else:
        assert result.path is TierResolutionPath.EXPLICIT_STATUS


def test_the_semantics_map_covers_exactly_the_tier_map() -> None:
    """Two parallel maps that may not drift apart."""
    assert REVIEW_STATUS_SEMANTICS.keys() == REVIEW_STATUS_TIER.keys()


def test_every_no_usable_classification_status_is_tier_five() -> None:
    """Statuses grouped as the same semantic class must share a tier."""
    statuses = {s for s, c in REVIEW_STATUS_SEMANTICS.items()
                if c is ReviewStatusSemanticClass.NO_USABLE_CLASSIFICATION}
    assert statuses, "the semantic class must be populated"
    assert {REVIEW_STATUS_TIER[s] for s in statuses} == {5}


def test_the_unflagged_records_status_is_an_explicit_key_not_a_missing_token() -> None:
    """Approved 2026-07-24. Known vocabulary, so it must not use either escape."""
    status = "no classifications from unflagged records"
    assert status in REVIEW_STATUS_TIER
    assert status not in MISSING_TOKENS
    result = resolve(status)
    assert result.tier == 5
    assert result.path is TierResolutionPath.EXPLICIT_STATUS


# --------------------------------------------------------------------------
# 3-4. Silence
# --------------------------------------------------------------------------
def test_an_unrecognised_status_raises_and_names_it() -> None:
    """Silence here is the defect. The message must make the fix obvious."""
    bogus = "criteria provided, some future clinvar wording"
    with pytest.raises(UnmatchedReviewStatusError) as exc:
        tier_of(bogus)
    message = str(exc.value)
    assert bogus in message, "the offending value must appear in the message"
    assert "REVIEW_STATUS_TIER" in message, "the message must name what to edit"
    assert "REVIEW_STATUS_SEMANTICS" in message, "both maps must be named"
    assert "allow_unmatched" in message, "the message must name the escape hatch"


def test_allow_unmatched_returns_the_fallback_and_warns(caplog) -> None:
    """Opting into the old behaviour is permitted. Doing it silently is not."""
    with caplog.at_level(logging.WARNING,
                         logger="genomic_variant_classifier.data.review_status"):
        result = resolve("an unknown status", allow_unmatched=True)
    assert result.tier == TIER_UNMATCHED
    assert result.path is TierResolutionPath.UNMATCHED_FALLBACK
    assert any("an unknown status" in r.getMessage() for r in caplog.records), \
        "the warning must name the value it accepted"


# --------------------------------------------------------------------------
# 5. Normalisation -- and the contract that must NOT be normalised
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    "Criteria_Provided,_Single_Submitter",
    "  criteria provided, single submitter  ",
    "CRITERIA PROVIDED, SINGLE SUBMITTER",
    "criteria  provided,   single   submitter",
])
def test_normalisation_is_applied_before_lookup(raw: str) -> None:
    """Review-status lookup folds case, underscores and whitespace.

    Not new: scripts/clean_cohort.py:150 has folded case on this path for as long
    as it has existed. The clinical-significance labelling path is the opposite
    contract and is asserted separately below.
    """
    assert tier_of(raw) == 3


@pytest.mark.parametrize("raw,expected", [
    (None, ""),
    ("", ""),
    ("   ", ""),
    ("_", ""),
    ("-", "-"),
    ("NaN", "nan"),
    ("<NA>", "<na>"),
])
def test_normalise_handles_every_absent_representation(raw: object,
                                                       expected: str) -> None:
    """str(pandas.NA) is '<NA>' and str(float('nan')) is 'nan'; both must resolve."""
    assert normalise(raw) == expected
    assert resolve(raw).path is TierResolutionPath.MISSING_TOKEN
    assert resolve(raw).tier == TIER_MISSING


def test_the_clinical_significance_contract_is_not_this_one() -> None:
    """A guard against a future maintainer merging two deliberately different rules.

    real_data_prep.py labels from clinical_sig with an EXACT, CASE-SENSITIVE
    membership test after fillna and strip. Its term sets are Title-case. If this
    module's normalisation were ever applied there, every label would still match
    -- but the reverse, applying the label contract here, would break every
    lowercase review status in the cohort. The asymmetry is the point.
    """
    source = (_REPO_ROOT / "src/genomic_variant_classifier/data/real_data_prep.py"
              ).read_text(encoding="utf-8")
    assert 'df["clinical_sig"].fillna("").str.strip()' in source, \
        "the clinical-significance contract changed; re-read the module docstring"
    assert ".lower()" not in source.split("PATHOGENIC_TERMS")[0][-400:], \
        "case folding appeared on the labelling path; the two contracts merged"


# --------------------------------------------------------------------------
# 6-7. Paths and single implementation
# --------------------------------------------------------------------------
def test_missing_and_unmatched_are_distinct_paths_despite_equal_tiers(
        monkeypatch) -> None:
    """TIER_MISSING and TIER_UNMATCHED are both 5. Only the path separates them.

    Repointing TIER_UNMATCHED to a sentinel proves the two branches are distinct
    code, not one branch reached two ways.
    """
    from genomic_variant_classifier.data import review_status as rs

    monkeypatch.setattr(rs, "TIER_UNMATCHED", 99)
    assert rs.resolve("-").tier == TIER_MISSING
    assert rs.resolve("-").path is TierResolutionPath.MISSING_TOKEN
    assert rs.resolve("unknown vocabulary", allow_unmatched=True).tier == 99
    assert rs.resolve("unknown vocabulary",
                      allow_unmatched=True).path is TierResolutionPath.UNMATCHED_FALLBACK


def test_missing_tokens_are_tested_before_the_map() -> None:
    """Ordering is load-bearing: 669,664 rows depend on it.

    '' and '-' account for 424,516 and 245,148 rows. If the map were consulted
    first they would raise, and every production run would abort.
    """
    for token in ("", "-"):
        assert resolve(token).path is TierResolutionPath.MISSING_TOKEN


def test_tier_of_and_tiers_for_both_delegate_to_resolve() -> None:
    """One lookup implementation, three entry points. Closes the drift defect."""
    values = list(OBSERVED_2026_07_24)
    assert [tier_of(v) for v in values] == [resolve(v).tier for v in values]
    assert tiers_for(values) == [resolve(v).tier for v in values]


def test_tiers_for_preserves_order_and_length() -> None:
    values = ["practice guideline", "-", "no assertion criteria provided", ""]
    assert tiers_for(values) == [1, TIER_MISSING, 4, TIER_MISSING]


def test_tiers_for_collects_every_unknown_before_raising() -> None:
    """Raising on the first miss turns a one-edit fix into an N-run bisection."""
    values = ["alpha unknown", "practice guideline", "beta unknown",
              "alpha unknown", "gamma unknown"]
    with pytest.raises(UnmatchedReviewStatusError) as exc:
        tiers_for(values)
    message = str(exc.value)
    for name in ("alpha unknown", "beta unknown", "gamma unknown"):
        assert name in message, f"{name!r} must be named in one message"
    assert "3 unrecognised" in message, "the count of distinct unknowns"
    assert "4 row(s)" in message, "the total row count, with alpha counted twice"


def test_tiers_for_warns_with_counts_when_unmatched_is_allowed(caplog) -> None:
    with caplog.at_level(logging.WARNING,
                         logger="genomic_variant_classifier.data.review_status"):
        result = tiers_for(["zeta unknown", "zeta unknown", "practice guideline"],
                           allow_unmatched=True)
    assert result == [TIER_UNMATCHED, TIER_UNMATCHED, 1]
    assert any("zeta unknown" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# 8. The detector, and its sabotage battery
# --------------------------------------------------------------------------
def test_the_tier_map_definition_inventory_is_exactly_as_frozen() -> None:
    """A ratchet on tier-map definitions, modelled on the suite-size ratchet.

    Adding a map fails. Removing one ALSO fails, so the inventory must be edited
    in the same commit that rewires a consumer.
    """
    found = detector.find_tier_map_definitions(_REPO_ROOT)
    unparseable = detector.unparseable_files(_REPO_ROOT)
    assert not unparseable, (
        f"{len(unparseable)} file(s) could not be parsed and were therefore not "
        f"scanned: {list(unparseable)}. A scan gap must never be silent."
    )
    if found != KNOWN_TIER_MAP_DEFINITIONS:
        located = {f"{r}::{n}": ln for r, n, ln in
                   detector.find_with_line_numbers(_REPO_ROOT)}
        added = [f"{d} (line {located[d]})" for d in found
                 if d not in KNOWN_TIER_MAP_DEFINITIONS]
        removed = [d for d in KNOWN_TIER_MAP_DEFINITIONS if d not in found]
        pytest.fail(
            f"The tier-map inventory changed.\n"
            f"  ADDED   ({len(added)}): {added}\n"
            f"  REMOVED ({len(removed)}): {removed}\n"
            f"  Expected {len(KNOWN_TIER_MAP_DEFINITIONS)}, found {len(found)}.\n"
            f"  If you rewired a consumer, edit KNOWN_TIER_MAP_DEFINITIONS in this\n"
            f"  file IN THE SAME COMMIT. If a new map appeared, it must not exist:\n"
            f"  import it from data/review_status.py instead."
        )


#: (label, source, must_be_detected). Each case is written to a temporary tree
#: and scanned. A guard is not trusted until every constructed failure is caught
#: by the gate meant to catch it, AND every constructed non-failure is not.
_SABOTAGE: tuple[tuple[str, str, bool], ...] = (
    ("a map under a different binding name", '''
        REVIEW_TIER = {
            "practice guideline": 1,
            "reviewed by expert panel": 1,
            "criteria provided, single submitter": 3,
        }
    ''', True),
    ("a map as a class attribute", '''
        class Tracker:
            REVIEW_TIER = {
                "practice guideline": 1,
                "reviewed by expert panel": 1,
                "criteria provided, single submitter": 3,
            }
    ''', True),
    ("a map inside a function body", '''
        def build():
            local_map = {
                "practice guideline": 1,
                "reviewed by expert panel": 1,
                "criteria provided, single submitter": 3,
            }
            return local_map
    ''', True),
    ("a map with an annotated assignment", '''
        TIERS: dict[str, int] = {
            "practice guideline": 1,
            "reviewed by expert panel": 1,
            "criteria provided, single submitter": 3,
        }
    ''', True),
    ("review-status strings in a docstring", '''
        """Discusses practice guideline, reviewed by expert panel and
        criteria provided, single submitter at length."""
        X = 1
    ''', False),
    ("an IMPORT of a map, which is a reference not a definition", '''
        from genomic_variant_classifier.data.review_status import (
            REVIEW_STATUS_TIER as T,
        )
        VALUE = T["practice guideline"]
    ''', False),
    ("a dict with review-status keys and STRING values", '''
        LABELS = {
            "practice guideline": "gold",
            "reviewed by expert panel": "silver",
            "criteria provided, single submitter": "bronze",
        }
    ''', False),
    ("a dict with only TWO recognised keys, below the threshold", '''
        PARTIAL = {
            "practice guideline": 1,
            "reviewed by expert panel": 1,
            "something else entirely": 9,
        }
    ''', False),
    ("a dict with review-status keys and BOOLEAN values", '''
        FLAGS = {
            "practice guideline": True,
            "reviewed by expert panel": False,
            "criteria provided, single submitter": True,
        }
    ''', False),
)


@pytest.mark.parametrize("label,source,must_detect",
                         [(a, b, c) for a, b, c in _SABOTAGE],
                         ids=[a.replace(" ", "_") for a, _, _ in _SABOTAGE])
def test_the_detector_fires_on_exactly_the_right_shapes(
        tmp_path: Path, label: str, source: str, must_detect: bool) -> None:
    """Every constructed failure caught; every constructed non-failure ignored."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "planted.py").write_text(
        textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    found = detector.find_tier_map_definitions(tmp_path)
    if must_detect:
        assert len(found) == 1, f"{label}: expected 1 detection, got {list(found)}"
    else:
        assert not found, f"{label}: must NOT be detected, got {list(found)}"


def test_an_unparseable_file_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    """A scan that cannot read a file must say so, not count it as clean."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "broken.py").write_text("def f(\n", encoding="utf-8")
    reported = detector.unparseable_files(tmp_path)
    assert len(reported) == 1
    assert reported[0].startswith("src/broken.py:"), reported
    assert "line" in reported[0], "the reason must name the line, not just the file"
    assert detector.find_tier_map_definitions(tmp_path) == ()


# --------------------------------------------------------------------------
# 9. The ratchet number, measured
# --------------------------------------------------------------------------
def test_the_collected_count_is_measured_by_pytest_not_computed() -> None:
    """Ask pytest how many items this file yields. Do not derive it.

    The previous version asserted a value against its own definition and could
    not fail. This runs collection in a subprocess -- collection does not execute
    tests, so there is no recursion -- and parses the count pytest reports.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(_THIS_FILE),
         "--collect-only", "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", proc.stdout)
    assert match, (
        "could not read a collection count from pytest.\n"
        f"stdout tail: {proc.stdout[-600:]}\nstderr tail: {proc.stderr[-600:]}"
    )
    measured = int(match.group(1))
    # Written out term by term so a future reader can check it against the file
    # rather than trust it. Corrected 2026-07-24: the first version of this
    # expression summed to 49 while pytest collected 57, and this test caught it
    # on its first run -- which is the entire reason it measures instead of
    # deriving.
    parametrised = (
        len(REVIEW_STATUS_TIER)          # every map key resolves            -> 11
        + len(OBSERVED_2026_07_24)       # every measured cohort value       -> 11
        + 4                              # normalisation is applied          ->  4
        + 7                              # every absent representation       ->  7
        + len(_SABOTAGE)                 # the detector sabotage battery     ->  9
    )
    plain = 15                           # non-parametrised test functions
    derived = parametrised + plain
    assert measured == derived, (
        f"pytest collects {measured} item(s) from this file; the arithmetic in "
        f"this test says {derived}. One of them is wrong, and the MEASURED "
        f"number is the authority. Update the arithmetic, then update "
        f"tests/EXPECTED_SUITE_SIZE and the README badge together."
    )
