"""ClinVar review status -> review tier. THE single source of truth.

Before this module existed there were three maps, and a fourth nobody had
counted. Measured 2026-07-24:

    src/genomic_variant_classifier/data/real_data_prep.py:132   substring lookup,
        unmatched default 5
    scripts/clean_cohort.py:126                                 exact lookup after
        normalisation, unmatched default 6
    scripts/augment_reviewstatus.py:21                          substring lookup,
        unmatched default 5
    src/genomic_variant_classifier/monitoring/clinvar_tracker.py:160
        REVIEW_TIER, a six-key class attribute -- invisible to any guard that
        searches for the NAME `REVIEW_STATUS_TIER`

The first three disagreed on six of ten keys. The cohort you got depended on
which module answered the question, and nothing in the code made that visible.
See docs/measurements/MEASUREMENT_2026-07-24_review-tier-map-divergence.md.

TIER SEMANTICS -- lower is more authoritative
---------------------------------------------
The scale mirrors ClinVar's star rating, verified 2026-07-24 against ClinVar
documentation and a live record, with 4 and 3 stars collapsed into tier 1:

    tier 1  practice guideline (4 stars); reviewed by expert panel (3 stars)
    tier 2  criteria provided, multiple submitters, no conflicts (2 stars)
    tier 3  criteria provided, single submitter (1 star)
            criteria provided, conflicting classifications (1 star)
    tier 4  no assertion criteria provided (0 stars) -- the last tier that is
            still an ASSERTION. A submitter classified the variant and
            documented no criteria.
    tier 5  the three statuses that are not assertions at all:
                no classification provided
                no classification for the single variant
                no classifications from unflagged records

Two decisions are recorded in
docs/measurements/DECISION_2026-07-24_review-tier-scale_R2.md and implemented
here: `no assertion criteria provided` is tier 4, not 5, because ClinVar's own
aggregation precedence ranks it last among assertions and does not rank it
against "no classification", which is not an assertion; and `criteria provided,
conflicting classifications` is tier 3, not 4, because ClinVar rates it at one
star, the same as a single submitter, and real_data_prep already has a separate
exclude_conflicting mechanism, so encoding conflict a second time in the tier
double-counts it.

A third decision, approved 2026-07-24: `no classifications from unflagged
records` is tier 5, as an EXPLICIT map key -- not a missing token, and not the
unmatched fallback. ClinVar received submissions, but after excluding flagged
records no usable classification remains. That is absence of admissible
classification evidence, which is what tier 5 means. It is not a serialization
marker, so it does not belong in MISSING_TOKENS, and it is known vocabulary, so
it must not reach the fallback. Measured in the cohort on 2026-07-24: 121 rows
through the legacy source, 133 through the repaired one, the legacy rows a
strict subset of the repaired ones, union 133. The source transition table
showed those rows UNCHANGED across both sources, so the status is stable
vocabulary rather than an artefact of one source.

TWO NORMALISATION CONTRACTS EXIST IN THIS PROJECT. DO NOT MERGE THEM.
---------------------------------------------------------------------
    CLINICAL SIGNIFICANCE, for labelling -- real_data_prep.py:510-513
        fillna("") -> strip -> EXACT, CASE-SENSITIVE membership test
        Verified 2026-07-24 across all three cohort artifacts: zero divergence.
        docs/measurements/LABEL_COLUMN_TERMS_2026-07-24.txt

    REVIEW STATUS, for tiering -- this module
        str -> lowercase -> underscores to spaces -> collapse whitespace ->
        strip -> exact lookup
        This is not new. scripts/clean_cohort.py:150 has folded case for the
        review-status path for as long as it has existed.

They are different because the sources are different, and a future maintainer
who "standardises" them into one will break whichever one they move.

WHY AN UNMATCHED VALUE RAISES
------------------------------
Every defect in this chain had one shape: a lookup miss silently becoming a
quality judgement. The blank ReviewStatus that became tier 5 and censored 98.834
percent of deletions. The dead "no classification for the individual variant"
key that never matched because ClinVar says "single variant". The `criteria
provided, conflicting classifications` value, absent from two of three maps,
silently demoted from one star to the worst tier.

The concrete mechanism, at scripts/clean_cohort.py:284-285:

    norm = norm.where(~norm.isin(MISSING_TOKENS), other=pd.NA)
    return norm.map(REVIEW_STATUS_TIER).fillna(TIER_UNMATCHED).astype(int)

A recognised missing token and an unsupported new status both become NA, and
both then become TIER_UNMATCHED. Absence and ignorance are indistinguishable in
the output. New ClinVar vocabulary passes as ordinary absence. This module
preserves three distinct paths and production raises on the third.

ClinVar renamed "conflicting interpretations" to "conflicting classifications"
and "...individual variant" to "...single variant" within the last two years.
Both renames are already in this cohort and both were absorbed silently. A raise
would have caught them on the day they appeared.

The precedent is already in real_data_prep.py:545-553, which raises rather than
filtering when the ReviewStatus COLUMN is absent, on the grounds that silence
"would silently keep all review levels". A missing VALUE misrepresents the
result identically, for the rows it touches, and line 551 already offers an
explicit escape hatch. This module offers the same one.

Acronyms on first use. ClinVar = the National Center for Biotechnology
Information's Clinical Variation archive. VCF = variant call format.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

logger = logging.getLogger(__name__)

__all__ = [
    "REVIEW_STATUS_TIER",
    "REVIEW_STATUS_SEMANTICS",
    "ReviewStatusSemanticClass",
    "TierResolution",
    "TierResolutionPath",
    "TIER_UNMATCHED",
    "TIER_MISSING",
    "MISSING_TOKENS",
    "OBSERVED_2026_07_24",
    "UnmatchedReviewStatusError",
    "normalise",
    "resolve",
    "tier_of",
    "tiers_for",
]


class ReviewStatusSemanticClass(str, Enum):
    """Why a status has the tier it has.

    Several statuses share tier 5. Without this, the numerical equality erases
    the reason, and a reader cannot tell an absent assertion from a suppressed
    one. Kept as a PARALLEL map rather than folded into a richer registry: the
    tier map must remain a dictionary literal of string keys and integer values,
    because that is the shape tests/unit/_review_map_detector.py keys on, and
    that detector is what found the fourth map nobody had counted.
    """

    PRACTICE_GUIDELINE = "practice_guideline"
    EXPERT_PANEL = "expert_panel"
    MULTIPLE_SUBMITTERS_NO_CONFLICT = "multiple_submitters_no_conflict"
    SINGLE_SUBMITTER_WITH_CRITERIA = "single_submitter_with_criteria"
    CONFLICTING_CLASSIFICATIONS = "conflicting_classifications"
    NO_ASSERTION_CRITERIA = "no_assertion_criteria"
    NO_USABLE_CLASSIFICATION = "no_usable_classification"


#: THE authoritative map. Keys are SPACE-form and lowercase; every lookup
#: normalises first. Eleven keys: nine observed in the cohort on 2026-07-24 plus
#: two spellings ClinVar has renamed, retained because older releases still
#: carry them and a dead key is how the original censoring defect began.
REVIEW_STATUS_TIER: dict[str, int] = {
    "practice guideline": 1,
    "reviewed by expert panel": 1,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 3,
    "criteria provided, conflicting classifications": 3,
    "criteria provided, conflicting interpretations": 3,   # the pre-2024 spelling
    "no assertion criteria provided": 4,
    "no classification provided": 5,
    "no classification for the single variant": 5,         # the spelling ClinVar uses
    "no classification for the individual variant": 5,     # retained: older releases
    "no classifications from unflagged records": 5,        # approved 2026-07-24
}

#: Parallel to REVIEW_STATUS_TIER, keyed identically. The key sets are asserted
#: equal by the test suite, so the two cannot drift apart silently.
REVIEW_STATUS_SEMANTICS: dict[str, ReviewStatusSemanticClass] = {
    "practice guideline": ReviewStatusSemanticClass.PRACTICE_GUIDELINE,
    "reviewed by expert panel": ReviewStatusSemanticClass.EXPERT_PANEL,
    "criteria provided, multiple submitters, no conflicts":
        ReviewStatusSemanticClass.MULTIPLE_SUBMITTERS_NO_CONFLICT,
    "criteria provided, single submitter":
        ReviewStatusSemanticClass.SINGLE_SUBMITTER_WITH_CRITERIA,
    "criteria provided, conflicting classifications":
        ReviewStatusSemanticClass.CONFLICTING_CLASSIFICATIONS,
    "criteria provided, conflicting interpretations":
        ReviewStatusSemanticClass.CONFLICTING_CLASSIFICATIONS,
    "no assertion criteria provided":
        ReviewStatusSemanticClass.NO_ASSERTION_CRITERIA,
    "no classification provided":
        ReviewStatusSemanticClass.NO_USABLE_CLASSIFICATION,
    "no classification for the single variant":
        ReviewStatusSemanticClass.NO_USABLE_CLASSIFICATION,
    "no classification for the individual variant":
        ReviewStatusSemanticClass.NO_USABLE_CLASSIFICATION,
    "no classifications from unflagged records":
        ReviewStatusSemanticClass.NO_USABLE_CLASSIFICATION,
}

#: The fallback, used ONLY when allow_unmatched is explicitly True.
TIER_UNMATCHED = 5

#: A row with no review status at all. Numerically equal to TIER_UNMATCHED and
#: semantically distinct from it: absence is a known state, an unknown string is
#: not. The distinction is enforced by TierResolution.path, not by the integer.
TIER_MISSING = 5

#: Every representation of "absent" seen in this cohort. '-' is the marker
#: metadata.review_status uses (245,148 rows, 2026-07-24); the empty string is
#: what the VCF join produced on a miss (424,516 rows).
MISSING_TOKENS = frozenset({"", "-", ".", "na", "nan", "none", "null", "<na>"})

#: The COMPLETE set of distinct values measured in both review-status columns of
#: data/processed/clinvar_grch38_clean.parquet on 2026-07-24 -- nine real
#: statuses plus the two missing markers. This replaces an earlier eight-value
#: list that was documented as "the top eight of the blank-ReviewStatus subset",
#: i.e. a partial view of one column. Evidence:
#: docs/measurements/COHORT_DELTA_FORENSICS_2026-07-24.txt.
#: Freezing it means a map regression turns the suite red without needing the
#: cohort, which is gitignored and absent from Continuous Integration.
OBSERVED_2026_07_24: tuple[str, ...] = (
    "",
    "-",
    "criteria provided, single submitter",
    "criteria provided, multiple submitters, no conflicts",
    "criteria provided, conflicting classifications",
    "no assertion criteria provided",
    "reviewed by expert panel",
    "no classification provided",
    "no classification for the single variant",
    "no classifications from unflagged records",
    "practice guideline",
)

_WHITESPACE = re.compile(r"\s+")


class UnmatchedReviewStatusError(ValueError):
    """Raised when a review status is present but recognised by no map key."""


class TierResolutionPath(str, Enum):
    """How a tier was arrived at. Observable because the integers collide."""

    EXPLICIT_STATUS = "explicit_status"
    MISSING_TOKEN = "missing_token"
    UNMATCHED_FALLBACK = "unmatched_fallback"


@dataclass(frozen=True)
class TierResolution:
    """A tier and the route taken to it.

    TIER_MISSING and TIER_UNMATCHED are both 5, so a bare integer cannot
    distinguish "this row had no review status" from "this row had a status this
    project does not recognise". Those are different facts requiring different
    responses, and returning the path makes the difference assertable in a test
    without monkeypatching anything.
    """

    tier: int
    path: TierResolutionPath
    normalised_value: str


def normalise(value: object) -> str:
    """Canonicalise a review status for lookup.

    Lowercase, underscores to spaces, collapse internal whitespace, strip. None
    becomes the empty string.

    THIS CONTRACT DIFFERS DELIBERATELY from the clinical-significance labelling
    path, which is an exact, case-sensitive membership test after
    `.fillna("").str.strip()` and nothing else. See the module docstring.

    Case folding is not new: scripts/clean_cohort.py:150 has folded case on the
    review-status path for as long as it has existed. Underscore folding is
    retained as a defensive compatibility normalisation for supported ClinVar
    vocabulary; it is NOT justified by the currently measured cohort, in which
    all nine real values in both review-status columns are space-form. An
    earlier version of this docstring claimed otherwise, importing evidence that
    belongs to the `pathogenicity` column.
    """
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value).lower().replace("_", " ")).strip()


def resolve(value: object, *, allow_unmatched: bool = False) -> TierResolution:
    """THE resolver. Every other entry point in this module delegates here.

    Order matters and is asserted by the test suite: missing tokens are tested
    BEFORE the map, so a serialization marker never reaches the map and never
    raises. That ordering is why '' (424,516 rows) and '-' (245,148 rows) do not
    abort a production run.
    """
    key = normalise(value)
    if key in MISSING_TOKENS:
        return TierResolution(TIER_MISSING, TierResolutionPath.MISSING_TOKEN, key)
    tier = REVIEW_STATUS_TIER.get(key)
    if tier is not None:
        return TierResolution(tier, TierResolutionPath.EXPLICIT_STATUS, key)
    if not allow_unmatched:
        raise UnmatchedReviewStatusError(
            f"Unrecognised ClinVar review status: {key!r}.\n"
            f"  It is not in REVIEW_STATUS_TIER and it is not a missing token.\n"
            f"  Assigning it a default tier would turn a lookup miss into a quality\n"
            f"  judgement, which is the defect this map exists to prevent.\n"
            f"  Fix: add it to REVIEW_STATUS_TIER and REVIEW_STATUS_SEMANTICS in\n"
            f"  src/genomic_variant_classifier/data/review_status.py, or pass\n"
            f"  allow_unmatched=True to accept tier {TIER_UNMATCHED} explicitly."
        )
    logger.warning(
        "Unrecognised review status %r mapped to tier %d because "
        "allow_unmatched=True. Add it to REVIEW_STATUS_TIER.", key, TIER_UNMATCHED
    )
    return TierResolution(TIER_UNMATCHED, TierResolutionPath.UNMATCHED_FALLBACK, key)


def tier_of(value: object, *, allow_unmatched: bool = False) -> int:
    """Tier for one review status. A thin wrapper over resolve()."""
    return resolve(value, allow_unmatched=allow_unmatched).tier


def tiers_for(values: Iterable[object], *, allow_unmatched: bool = False) -> list[int]:
    """Tiers for many review statuses, order and length preserved.

    Collects EVERY unrecognised value before raising, with its row count, so the
    fix is one map edit rather than a sequence of them. Raising on the first miss
    would make a cohort with four unknown statuses take four runs to diagnose --
    over 4.4 million rows that is a bisection, not a fix.

    The pre-pass decides WHETHER to raise. The values themselves are then routed
    through resolve(), so there is exactly one lookup implementation in this
    module and the two cannot drift.
    """
    keys = [normalise(v) for v in values]
    unknown = Counter(
        k for k in keys if k not in MISSING_TOKENS and k not in REVIEW_STATUS_TIER
    )
    if unknown and not allow_unmatched:
        lines = "\n".join(f"    {n:>9,}  {k!r}" for k, n in unknown.most_common())
        raise UnmatchedReviewStatusError(
            f"{len(unknown)} unrecognised ClinVar review status value(s), "
            f"{sum(unknown.values()):,} row(s) total:\n"
            f"{lines}\n"
            f"  None is in REVIEW_STATUS_TIER and none is a missing token.\n"
            f"  Assigning a default tier would turn a lookup miss into a quality\n"
            f"  judgement, which is the defect this map exists to prevent.\n"
            f"  Fix: add them to REVIEW_STATUS_TIER and REVIEW_STATUS_SEMANTICS in\n"
            f"  src/genomic_variant_classifier/data/review_status.py, or pass\n"
            f"  allow_unmatched=True to accept tier {TIER_UNMATCHED} explicitly."
        )
    if unknown:
        logger.warning(
            "allow_unmatched=True: %d unrecognised review status value(s), %d row(s), "
            "mapped to tier %d: %s",
            len(unknown), sum(unknown.values()), TIER_UNMATCHED,
            ", ".join(repr(k) for k, _ in unknown.most_common()),
        )
    return [resolve(k, allow_unmatched=True).tier for k in keys]
