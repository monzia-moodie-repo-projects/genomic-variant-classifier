"""METHODS.md may not present a retired architecture as the current one.

Created 2026-08-24.

WHAT THIS GUARDS
----------------
`METHODS.md` is the scientific description of this model -- in the words of the
test that guards its feature count, "the document a reviewer, a collaborator, or
a journal sees". Until 2026-08-24 its section 3.1 stated, in the present tense:

    Four tabular base models were trained on the 64-feature matrix:

against a tabular feature contract of 95 and a thirteen-model ensemble. Section
2 of the same document already carried a 2026-07-13 correction for the identical
class of defect, ending: "Restating the number by hand would only reset the
clock on the same defect."

The exact run identity of that four-estimator configuration is NOT established
by the committed evidence. So the repair states that, quotes the former text as
history, and substitutes no corrected figure -- the same resolution BASELINE-1
received on the same day, for the same reason: no attributable value exists.

TWO FINDINGS, NOT ONE
---------------------
    METHODS-CURRENT-ARCHITECTURE-STALE-1              closed by this unit
    METHODS-HISTORICAL-CONFIGURATION-UNATTRIBUTED-1   OPEN

The second remains until the four-estimator, 64-feature state can be attributed
to a run from machine artefacts -- or until it is explicitly ruled
unrecoverable. A documentation correction must not pretend to establish
historical provenance.

THIS GUARD IS TRANSITIONAL, AND SAYS SO
---------------------------------------
A forbidden-string assertion is the weakest form of this check. The durable
version is a PROJECTION from a declarative estimator specification -- but
`ESTIMATOR_SPECS` does not exist: MEASURED 2026-08-24, the roster is BUILT by
`_build_estimators` and obtained from `len(VariantEnsemble().base_estimators)`
on a live instance. Until that specification exists, this narrow guard holds the
line, and it should be DELETED and replaced when the projection lands rather
than accumulated beside it.

Author: Monzia Moodie
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

METHODS = Path(__file__).resolve().parents[2] / "METHODS.md"

#: The exact sentence that made the false present-tense claim.
RETIRED_CLAIM = "Four tabular base models were trained on the 64-feature matrix"

#: Corrections this document has earned. Deleting one erases the record that a
#: defect was found, which is worse than the defect.
REQUIRED_CORRECTIONS = (
    "**Correction, 2026-07-13.**",
    "**Correction, 2026-08-24.**",
    "**Historical-configuration notice, 2026-08-24.**",
)


@pytest.fixture(scope="module")
def methods() -> str:
    assert METHODS.is_file(), f"{METHODS} is absent"
    return METHODS.read_text(encoding="utf-8")


def _bare_lines(text: str, needle: str) -> list:
    """Lines stating `needle` OUTSIDE a blockquote.

    A blockquote QUOTES; body text ASSERTS. The distinction is the whole
    repair: the former table is preserved as history and must not be deleted,
    but it may not be read as a current claim.
    """
    return [line for line in text.split("\n")
            if needle in line and not line.lstrip().startswith(">")]


def test_the_retired_architecture_is_not_asserted_as_current(methods):
    bare = _bare_lines(methods, RETIRED_CLAIM)
    assert not bare, (
        "METHODS.md asserts the retired four-estimator, 64-feature "
        f"configuration as current:\n" + "\n".join(f"  {b}" for b in bare)
        + "\n\nThe contract is 95 features and the ensemble has 13 models."
    )


def test_the_retired_architecture_survives_as_quoted_history(methods):
    """Preserved, not erased.

    A document that deletes its own former claims cannot be audited, and the
    2026-07-13 correction in section 2 exists precisely because the record of a
    defect is worth keeping.
    """
    quoted = [line for line in methods.split("\n")
              if RETIRED_CLAIM in line and line.lstrip().startswith(">")]
    assert quoted, (
        "the retired claim was DELETED rather than quoted. Erasing a former "
        "claim removes the evidence that it was ever made."
    )


def test_no_bare_sentence_states_the_retired_feature_count(methods):
    """`64-feature matrix` anywhere outside a quotation is the same claim."""
    bare = _bare_lines(methods, "64-feature matrix")
    assert not bare, (
        "METHODS.md states '64-feature matrix' outside a blockquote:\n"
        + "\n".join(f"  {b}" for b in bare)
    )


def test_the_document_does_not_assign_the_configuration_to_a_named_run(methods):
    """The run identity is NOT established, and inventing one is worse than
    leaving it open.

    BASELINE-1 established that `0.9847` could not be attributed and recorded
    exactly that. The same discipline applies here: a correction may not
    manufacture lineage to look complete.
    """
    window = methods[methods.index("### 3.1"):methods.index("### 3.2")]

    # NO RUN REFERENCE AT ALL in body text. An earlier draft used a LOOKAHEAD
    # requiring the configuration keywords to FOLLOW the run name, and so
    # missed "That four-estimator, 64-feature state was Run 8" -- the same
    # claim with the clause order reversed. Found by sabotage, which reported
    # NOTHING FAILED. A directional pattern is not a predicate.
    #
    # Quoted history may mention a run; asserted body text may not, because
    # section 3.1 exists to record that the identity is NOT established.
    attributions = [
        (line, match)
        for line in window.split("\n")
        if not line.lstrip().startswith(">")
        for match in re.findall(r"\bRun\s+\d+[a-z]?\b", line)
    ]
    assert not attributions, (
        "section 3.1 attributes the retired configuration to "
        f"{sorted({m for _l, m in attributions})} in body text:\n"
        + "\n".join(f"  {line.strip()}" for line, _m in attributions)
        + "\n\nThe committed evidence does not establish which run it was."
    )


@pytest.mark.parametrize("correction", REQUIRED_CORRECTIONS,
                         ids=("2026-07-13", "2026-08-24", "historical-notice"))
def test_every_earned_correction_survives(methods, correction):
    assert correction in methods, (
        f"the correction {correction!r} was removed. A document must be able "
        "to record its own repairs."
    )
