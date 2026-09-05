"""`CLAUDE.md` must not lie about numbers the code already knows. Enforced, not trusted.

Created 2026-09-04. Modelled on `tests/unit/test_readme_claims.py` (2026-07-14) and
`tests/unit/test_roadmap_claims.py` (2026-08-23), whose lessons are applied here rather
than re-learned.

WHY
---
MEASURED 2026-09-04 by parsing every tracked Python file:

    src/genomic_variant_classifier/models/variant_ensemble.py:191
        EXPECTED_TABULAR_FEATURE_COUNT = 95
    src/genomic_variant_classifier/models/variant_ensemble.py:325
        TABULAR_FEATURES has 95 literal elements

`CLAUDE.md` line 209 said **97**. It had said 97 since the contract moved 97 -> 95 on
2026-07-13, when HGMD was removed from `TABULAR_FEATURES` for label leakage -- so the
number was wrong for **seven weeks**, inside the document line 3 calls "the operating
manual" and instructs be read first, every session.

That is roadmap section 7, root pattern (a) -- *a number written down once and never
re-derived becomes a lie on a schedule* -- sitting in section 4 of the document that
DEFINES root pattern (a) in section 2.

`README.md` and `docs/ROADMAP.md` were already bound and were both CORRECT at 95.
`CLAUDE.md` was not bound, and it drifted. The difference is the binding, not the care.

    A COMMENT DOES NOT ENFORCE ITSELF. AN OPERATING MANUAL ENFORCES ITSELF NO BETTER.
    MAKE FORGETTING FAIL.

WHY ENUMERATED SITES RATHER THAN A SWEEP
----------------------------------------
The same reason `test_readme_claims.py` gives, and it was learned the hard way there: a
sweep that cannot distinguish "the contract IS n" from "the contract WAS 97 before HGMD
was removed" is not a count test -- it is a ban on writing history down, in a project
whose entire method is writing history down. `CLAUDE.md` is full of true historical
numbers: a pytest floor that rotted 1485 -> 1805 -> 1842 -> 1850 -> 1853, a suite of
1,815, a 0.855 measured delta, `EXPECTED_SCHEMA_COLS = 87`. Every one must remain
sayable.

So the claim sites are ENUMERATED. Each MUST be present -- a vanished pattern FAILS,
so this file cannot silently go vacuous if the document is restructured -- and each
MUST equal its live source.

**If you add a NEW place in `CLAUDE.md` that asserts a live number, ADD IT HERE.** That
is the one manual step, and it is the honest one: the alternative is a regular
expression that either misses claims or forbids prose.

WHY NOT AN ABSENCE TEST
-----------------------
The alternative considered and rejected: assert that `CLAUDE.md` states no count at all,
forcing a reader to the constant. Its pass condition would be MATCHING NOTHING, so a
pattern that quietly stopped matching would be indistinguishable from success -- the
"check that cannot fail" this project exists to hunt. Bind, do not ban.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    EXPECTED_TABULAR_FEATURE_COUNT,
    TABULAR_FEATURES,
)

CLAUDE_MD = Path("CLAUDE.md")

#: Every place `CLAUDE.md` makes a CLAIM about a number the code already knows, with the
#: live source each must equal. One capture group per pattern, holding the digits.
#:
#: `feature-count convention` is section 4's instruction to bump the guard when a real
#: feature is added. The parenthetical exists to save a reader a lookup, and on
#: 2026-09-04 it had cost more than it saved: it read 97 against a contract of 95.
CONTRACT_CLAIM_PATTERNS: dict[str, str] = {
    "feature-count convention": r"bump `EXPECTED_TABULAR_FEATURE_COUNT`\*\*\s*\(currently\s*\*\*(\d+)\*\*\)",
}


@pytest.fixture(scope="module")
def claude_md() -> str:
    if not CLAUDE_MD.is_file():
        pytest.fail(f"CLAUDE.md not found at {CLAUDE_MD.resolve()}")
    return CLAUDE_MD.read_text(encoding="utf-8")


def test_every_claim_site_is_still_present(claude_md):
    """A claim site that vanishes must FAIL, not quietly stop being checked.

    Either `CLAUDE.md` was restructured -- fix the pattern -- or the claim was dropped.
    Deleting the entry to go green is the defect this file prevents.
    """
    missing = [site for site, pattern in CONTRACT_CLAIM_PATTERNS.items()
               if re.search(pattern, claude_md) is None]
    assert not missing, (
        f"these claim sites are no longer found in CLAUDE.md: {missing}\n\n"
        f"Either CLAUDE.md was restructured -- in which case FIX THE PATTERN in "
        f"CONTRACT_CLAIM_PATTERNS -- or the claim was removed. Do not delete the entry "
        f"to make this pass: a check that no longer checks anything is exactly the "
        f"defect this whole file exists to prevent."
    )


def test_every_claim_equals_its_live_source(claude_md):
    """EQUALITY, with no tolerance anywhere.

    `test_roadmap_claims.py` records why: the README binding's first version used
    `assert collected - n <= 50` and let a real 17-test drift pass while reporting
    green. A tolerance on a number that CAN be exact is not engineering judgement; it
    is a place for rot to live.
    """
    wrong = []
    for site, pattern in CONTRACT_CLAIM_PATTERNS.items():
        m = re.search(pattern, claude_md)
        assert m is not None, site      # covered above; belt and braces
        claimed = int(m.group(1).replace(",", ""))
        if claimed != EXPECTED_TABULAR_FEATURE_COUNT:
            wrong.append((site, claimed, EXPECTED_TABULAR_FEATURE_COUNT))
    assert not wrong, (
        f"CLAUDE.md states {len(wrong)} wrong value(s):\n"
        + "\n".join(f"    {s:32s} says {c:>6}   live source says {a:>6}"
                    for s, c, a in wrong)
        + "\n\nUpdate EVERY site, in the same commit as the change that moved the "
          "number. On 2026-09-04 this site read 97 against a contract of 95, seven "
          "weeks after HGMD was removed."
    )


def test_the_constant_equals_the_list_it_describes():
    """A constant that has drifted from its own list is the same defect one level down,
    and it would make the claim above agree with a wrong number.

    Deliberately duplicated from `test_roadmap_claims.py`: this file must not depend on
    another file's coverage to be sound, and the assertion is one line.
    """
    assert EXPECTED_TABULAR_FEATURE_COUNT == len(TABULAR_FEATURES), (
        f"EXPECTED_TABULAR_FEATURE_COUNT is {EXPECTED_TABULAR_FEATURE_COUNT} but "
        f"len(TABULAR_FEATURES) is {len(TABULAR_FEATURES)}"
    )
