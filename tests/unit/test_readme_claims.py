"""The README must not lie about numbers the code already knows. Enforced, not trusted.

Created 2026-07-14 (roadmap 6.23). Full diagnosis: docs/audits/README_AUDIT_2026-07-14.md

WHY
---
The README is the first thing a collaborator, a reviewer, or a regulator reads. On 2026-07-14
an audit found that it stated:

  * the FEATURE COUNT in NINE places, with FOUR different values -- 80 (x6), 78, 79 -- against
    a true contract of 95;
  * the TEST COUNT in THREE places, with THREE different values -- 862 (badge), 501, 501 --
    against a true suite of 1,926 passing / 1,933 collected;
  * that the message-bus suite passed on Python 3.14.3 in one place and 3.12.10 in another
    (the project runs 3.11 and 3.12; 3.14 is the version under which requirements.txt was
    mis-compiled, silently dropping the entire torch stack -- roadmap 6.18);
  * HGMD Professional as an integrated data source, in three places, with two features -- a
    source whose licence was never obtained, whose connector was never wired, and whose two
    columns were CONSTANT ZERO for the life of the project;
  * a training quickstart using `--parquet`, a flag that HAS NEVER EXISTED (the script takes
    `--clinvar`). Anyone who copied it got an argparse error.

Not one of those numbers was ever re-derived. Every one was transcribed, and then transcribed
again, and then went stale. That is roadmap section 7, root pattern (a): a number written down
once and never re-derived becomes a lie on a schedule.

THE FIX IS NOT TO CORRECT THE NUMBERS. It is to stop keeping a second copy of them.

This file re-derives the agreement on every test run. Change the feature contract, add tests,
add an exit code, or reinstate HGMD, and forget the README -- and the suite goes RED, naming
both values. The README can no longer rot, because rotting now fails.

    A COMMENT DOES NOT ENFORCE ITSELF. A README ENFORCES ITSELF EVEN LESS.
    MAKE FORGETTING FAIL.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from genomic_variant_classifier.models.variant_ensemble import (
    EXPECTED_TABULAR_FEATURE_COUNT,
    TABULAR_FEATURES,
)

README = Path("README.md")
SUITE_SIZE_FILE = Path("tests/EXPECTED_SUITE_SIZE")


@pytest.fixture(scope="module")
def readme() -> str:
    if not README.is_file():
        pytest.fail(f"README.md not found at {README.resolve()}")
    return README.read_text(encoding="utf-8")


def _expected_suite_size() -> int:
    """The ratchet's number -- the SAME file tests/conftest.py reads. Never a second copy."""
    for line in SUITE_SIZE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return int(line)
    raise AssertionError(f"no bare integer found in {SUITE_SIZE_FILE}")


# ---------------------------------------------------------------------------
# 1. THE FEATURE COUNT -- stated nine times, four different values, all wrong
# ---------------------------------------------------------------------------

#: Every place the README makes a CLAIM ABOUT THE CURRENT FEATURE CONTRACT.
#:
#: WHY THESE ARE EXPLICIT AND NOT A REGEX SWEEP
#: -------------------------------------------
#: The first version of this test swept the whole document for `<N> features` / `<N>-feature` /
#: `<N> dimensions`. It failed immediately -- and it was RIGHT to fail, but for the wrong
#: reason. It had matched:
#:
#:     78  x4  -- "36 of its 78 features CONSTANT ZERO", "across all 78 features"
#:     80      -- "not the 78 or 80 the document claimed"
#:     38      -- "the 38 features that were real"
#:      6      -- "GTEx (6 features)"
#:
#: Every one of those is a HISTORICAL statement about the Run-15 matrix, and every one is TRUE
#: and necessary. A sweep that cannot distinguish "the model HAS n features" from "36 of its 78
#: features WERE zero" is not a feature-count test -- it is a ban on writing history down, in a
#: project whose entire method is writing history down.
#:
#: So the claim sites are enumerated. Each MUST be present (a missing pattern fails the test, so
#: the check cannot silently go vacuous if the README is restructured), and each MUST say 95.
#:
#: If you add a NEW place that asserts the feature count, ADD IT HERE. That is the one manual
#: step, and it is the honest one: the alternative is a regex that either misses claims or
#: forbids prose.
#: REPOINTED 2026-07-15. The README was restored to its pre-2026-07-14 state and rewritten
#: clean (see tests/EXPECTED_SUITE_SIZE, entry 1966). The old sites lived in sections that no
#: longer exist -- the ASCII pipeline diagram, the /info endpoint gloss and the quickstart
#: comment all carried a hand-kept copy of the count, which is root pattern (a) with extra
#: steps. The rewrite states the count in FOUR places instead of seven, and the last of them
#: is a table that must SUM to it. These patterns point at what the document now has.
#:
#: This is the branch this file's own failure message demands -- "Either the README was
#: restructured -- in which case FIX THE PATTERN" -- and it is the opposite of deleting an
#: entry to go green.
CONTRACT_CLAIM_PATTERNS: dict[str, str] = {
    "shields.io badge":         r"tabular%20features-(\d+)",
    "headline sentence":        r"\*\*(\d+)-feature\*\* matrix",
    "feature-set heading":      r"## Feature set \((\d+) tabular features\)",
    "feature-table total row":  r"\|\s*\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*",
}


def test_readme_never_states_a_wrong_feature_count(readme):
    """Every CLAIM about the current feature contract must say 95.

    Historical statements ("36 of its 78 features were constant zero") are legitimate and are
    NOT policed here -- see the note on CONTRACT_CLAIM_PATTERNS above for why that distinction
    is the whole point.

    Before 2026-07-14 this README stated the feature count in NINE places with FOUR different
    values (80 x6, 78, 79) against a true contract of 95.
    """
    found: dict[str, int] = {}
    missing: list[str] = []

    for site, pattern in CONTRACT_CLAIM_PATTERNS.items():
        m = re.search(pattern, readme)
        if m is None:
            missing.append(site)
        else:
            found[site] = int(m.group(1))

    # Guard the guard, in BOTH directions.
    # A claim site that has vanished means either the README was restructured (fix the pattern)
    # or the claim was silently dropped. Either way this test must not quietly stop checking it.
    assert not missing, (
        f"these feature-count claim sites are no longer found in README.md: {missing}\n"
        f"\n"
        f"Either the README was restructured -- in which case FIX THE PATTERN in "
        f"CONTRACT_CLAIM_PATTERNS -- or the claim was removed. Do not delete the entry to make "
        f"this pass: a check that no longer checks anything is exactly the defect this whole "
        f"file exists to prevent."
    )

    wrong = {site: n for site, n in found.items() if n != EXPECTED_TABULAR_FEATURE_COUNT}
    assert not wrong, (
        f"README.md asserts the wrong feature count in {len(wrong)} place(s):\n"
        + "\n".join(f"    {site:32s} says {n}" for site, n in wrong.items())
        + f"\n\nThe contract (EXPECTED_TABULAR_FEATURE_COUNT, variant_ensemble.py) is "
          f"{EXPECTED_TABULAR_FEATURE_COUNT}. Update EVERY site, in the same commit as the "
          f"feature change."
    )


def _feature_set_section(readme: str) -> str:
    """Just the '## Feature set' section -- NOT the whole README.

    Scoped deliberately. A naive `^\\|.*\\|\\s*(\\d+)\\s*\\|` sweep over the whole document would
    also hoover up any other table that happens to have a number in its second column, and the
    sum would then be meaningless while still looking authoritative. A test that adds up the
    wrong rows and passes is worse than no test.
    """
    m = re.search(r"^## Feature set\b.*?(?=^## )", readme, re.M | re.S)
    assert m, (
        "could not find the '## Feature set' section in README.md. If it was renamed, update "
        "this locator -- do NOT loosen it to search the whole document."
    )
    return m.group(0)


def test_readme_feature_table_sums_to_the_contract(readme):
    """The table must not merely LOOK right -- it must add up.

    The pre-2026-07-14 table summed to exactly 80, agreeing with the (wrong) prose and with
    nothing else. It reached 80 partly by counting two features -- uncertainty_epistemic and
    uncertainty_aleatoric -- that live in PHASE_4_FEATURES and are NOT in the trained contract,
    and two more (hgmd_*) that were constant zero.
    """
    section = _feature_set_section(readme)
    rows = re.findall(r"^\|\s*(?!\*\*Total)([^|]+?)\s*\|\s*(\d+)\s*\|", section, re.M)
    assert rows, "no feature-group table rows found in the '## Feature set' section"

    total = sum(int(n) for _, n in rows)
    assert total == EXPECTED_TABULAR_FEATURE_COUNT, (
        f"the README feature-group table sums to {total}; the contract is "
        f"{EXPECTED_TABULAR_FEATURE_COUNT}.\n\nGroups found ({len(rows)}):\n"
        + "\n".join(f"    {g:44s} {n:>3s}" for g, n in rows)
    )


# ---------------------------------------------------------------------------
# 2. THE TEST COUNT -- bound to the ratchet, not to a badge someone remembered
# ---------------------------------------------------------------------------

#: Every place the README CLAIMS the suite size. Enumerated for the same reason
#: CONTRACT_CLAIM_PATTERNS is: a blanket sweep cannot tell a CLAIM from a HISTORICAL NOTE.
#:
#: This was learned twice, in the same file, one turn apart:
#:   * The feature-count sweep matched "36 of its 78 features CONSTANT ZERO" and "the 38
#:     features that were real" -- true, necessary history. Fixed by enumerating claim sites.
#:   * The test-count check was then written as a blanket ban on the string "1,926" ... and it
#:     fired on the CORRECTION NOTE explaining that "1,926 tests passing" was stale. The test
#:     forbade the document from recording its own repair.
#:
#: A gate that cannot distinguish "the suite HAS n tests" from "this README used to SAY n" is
#: not a test-count gate; it is a ban on writing history down, in a project whose entire
#: method is writing history down.
#: REPOINTED 2026-07-15. The 2026-07-15 rewrite states the suite size ONCE, in the badge --
#: deliberately. The old README stated it in three places and they disagreed (862 in a badge,
#: 501/501 twice, against a true 1,926). One claim site cannot contradict itself.
#:
#: The single entry is not a weakening: the site that remains is still bound by `==` to
#: EXPECTED_SUITE_SIZE, which is itself bound by `--assert-suite-size` to the COLLECTED count.
#: The chain from badge to reality is unbroken; it is just shorter.
TEST_COUNT_CLAIM_PATTERNS: dict[str, str] = {
    "shields.io badge": r"tests-([\d,]+)-success",
}


def test_readme_test_count_equals_the_suite_size_ratchet_exactly(readme):
    """The README's test count must EQUAL tests/EXPECTED_SUITE_SIZE. No tolerance.

    THIS TEST WAS REBUILT ON 2026-07-14 BECAUSE ITS FIRST VERSION LET A STALE NUMBER THROUGH.
    ---------------------------------------------------------------------------------------
    The first version allowed the README to quote a PASSING count and checked it with a
    tolerance:

        assert n <= collected
        assert collected - n <= 50      # <-- let a 17-test drift pass, silently

    The README said "1,926 tests passing" while the suite had grown to 1,943 collected. The
    gap was 17. The tolerance was 50. **The test passed and the README was wrong.**

    The reasoning behind the tolerance was that a passing count is environment-dependent, so it
    cannot be asserted exactly -- which is TRUE, and is exactly why the README must not quote
    one. The right answer was already written down, in the header of the very file this test
    reads (tests/EXPECTED_SUITE_SIZE):

        WHY *COLLECTED* AND NOT *PASSED*
        The collected count is ENVIRONMENT-INDEPENDENT. The passed/skipped split is not:
            Windows + full data:   1863 passed +  7 skipped              = 1870
            Linux CI runner:       1856 passed + 13 skipped + 1 xfailed  = 1870
        Same collection, different outcomes ... Asserting `passed` would force two numbers and
        re-create the divergence this file exists to kill. Asserting `collected` gives ONE
        number that is true everywhere.

    That paragraph was read, and entries were written beneath it, and then the README was made
    to quote a passing count anyway -- with a tolerance to hide the consequence. A tolerance on
    a number that CAN be exact is not engineering judgement; it is a place for rot to live.

    So: the README states COLLECTED, and this asserts EQUALITY. One number, true on every
    machine, checked with `==`.
    """
    collected = _expected_suite_size()
    found: dict[str, int] = {}
    missing: list[str] = []

    for site, pattern in TEST_COUNT_CLAIM_PATTERNS.items():
        m = re.search(pattern, readme)
        if m is None:
            missing.append(site)
        else:
            found[site] = int(m.group(1).replace(",", ""))

    assert not missing, (
        f"these test-count claim sites are no longer found in README.md: {missing}\n"
        f"\n"
        f"Either the README was restructured -- in which case FIX THE PATTERN in "
        f"TEST_COUNT_CLAIM_PATTERNS -- or the claim was silently dropped. Do not delete the "
        f"entry to make this pass."
    )

    wrong = {site: n for site, n in found.items() if n != collected}
    assert not wrong, (
        f"README.md states the wrong test count in {len(wrong)} place(s):\n"
        + "\n".join(f"    {site:32s} says {n}" for site, n in wrong.items())
        + f"\n\ntests/EXPECTED_SUITE_SIZE says {collected}. These must be EQUAL -- no "
          f"tolerance. The collected count is environment-independent and is written down in "
          f"exactly one place; the README must re-derive it, not approximate it."
    )


def test_readme_does_not_quote_an_environment_dependent_passing_count(readme):
    """The README must not quote a PASSING count at all -- only COLLECTED.

    Separate from the equality test above, and deliberately so. That test checks the number is
    right; this one checks the README is quoting the RIGHT KIND OF NUMBER.

    A passing count is `collected - skipped`, and the skip set differs by machine: 7 skips on
    Windows with the full cohort, 13 plus an xfail on a hosted Linux runner, from the SAME
    collection. Any passing count the README states is therefore true on at most one machine
    and false on the other -- and it cannot be gated exactly, which is what pushed the first
    version of this test into a 50-wide tolerance that then hid a real 17-test drift.

    The message-bus suite's own count ("35/35 tests passing") is exempt: it is a fixed,
    fully-deterministic file with no environment-dependent skips.

    BLOCKQUOTES ARE EXEMPT, AND THAT EXEMPTION IS THE POINT.
    -------------------------------------------------------
    The first version of this test banned the string outright and FAILED on the README's own
    correction note -- the blockquote that says *"This README said '1,926 tests passing' until
    2026-07-14"*. It forbade the document from recording its own repair.

    That is the identical mistake the feature-count sweep made one turn earlier, in this same
    file, and which was fixed there by enumerating claim sites. Twice in two turns.

    So: lines beginning with `>` are COMMENTARY -- the document explaining its own history --
    and are not policed. Body text is CLAIMS, and is. That is a real, checkable distinction,
    and it is the convention this README already follows for every correction note in it.
    """
    body = "\n".join(
        line for line in readme.splitlines() if not line.lstrip().startswith(">")
    )
    offenders = [
        m.group(0)
        for m in re.finditer(r"\b\d[\d,]{2,6}\s*(?:tests?\s+)?passing", body)
        if "/" not in m.group(0)          # allow "35/35 passing" -- see docstring
    ]
    assert not offenders, (
        f"README.md quotes a PASSING test count: {offenders}\n"
        f"\n"
        f"Quote the COLLECTED count instead. Passing is `collected - skipped`, and the skip set "
        f"is environment-dependent (7 on Windows, 13 + 1 xfail on the Linux runner, from the "
        f"same collection) -- so a passing figure is true on at most one machine, and cannot be "
        f"asserted exactly. tests/EXPECTED_SUITE_SIZE's own header has said so since it was "
        f"written; this README ignored it and went stale by ten within a day."
    )


# ---------------------------------------------------------------------------
# 3. HGMD -- removed from the contract; must not creep back into the prose
# ---------------------------------------------------------------------------

def test_readme_does_not_present_hgmd_as_a_feature_or_a_source(readme):
    """HGMD may be DISCUSSED. It may not be CLAIMED.

    Until 2026-07-13 the README listed HGMD Professional in the source diagram, among the
    "gene-disease knowledge bases", and as a two-feature row in the feature table -- a source
    whose licence was never obtained, whose connector was never wired, and whose two columns
    were constant zero across all 1,038,974 variants of Run 15.

    The document must still be free to EXPLAIN that -- and it does, at length, because the
    reason HGMD must never return as a variant-level feature (it is the ClinVar-Pathogenic
    label under another vendor's name) is scientifically important. What it must not do is
    present HGMD as something the model uses.
    """
    assert "hgmd_is_disease_mutation" not in TABULAR_FEATURES
    assert "hgmd_n_reports" not in TABULAR_FEATURES

    # A table row of the form `| HGMD | 2 | ... |` -- i.e. HGMD claimed as a feature group.
    claimed_as_group = re.findall(r"^\|\s*HGMD[^|]*\|\s*\d+\s*\|", readme, re.M | re.I)
    assert not claimed_as_group, (
        f"README.md presents HGMD as a feature group with a count: {claimed_as_group}. "
        f"It supplied zero non-zero values and was removed from the contract 2026-07-13. "
        f"If the licence has been obtained, wire it GENE-LEVEL and LEAVE-ONE-OUT "
        f"(n_hgmd_dm_in_gene, excluding the variant being scored) -- never as the variant-level "
        f"flag, which is the training label wearing a different badge."
    )

    # The README must not be able to drop the explanation either: if HGMD is mentioned at all,
    # the reason for its absence must be present. Otherwise a future edit could trim the
    # caveat and leave a bare mention that reads like an endorsement.
    if re.search(r"\bHGMD\b", readme):
        assert re.search(r"HGMD is NOT (a source|in the feature set)", readme, re.I), (
            "README.md mentions HGMD but no longer states that it is NOT a source / NOT in the "
            "feature set. A bare mention reads as an endorsement. Keep the explanation."
        )


# ---------------------------------------------------------------------------
# 3b. THE BASE-MODEL ROSTER -- 13, and the README must not say 12
# ---------------------------------------------------------------------------

def test_readme_base_model_roster_matches_the_ensemble(readme, tmp_path):
    """The README's base-model table must equal VariantEnsemble's actual roster.

    THIS IS NOT A COSMETIC COUNT. Until 2026-07-14 the README said the ensemble had **twelve**
    base classifiers. The real roster is THIRTEEN. The old list made two compounding errors that
    nearly cancelled:

      * it OMITTED `svm` and `svm_bagged_rbf` -- two real base classifiers  (-2)
      * it COUNTED the Graph Attention Network as a base classifier          (+1)

    The GAT is not in the roster. It produces `gnn_score`, a FEATURE, and contributes no
    out-of-fold column. 13 - 2 + 1 = 12: two mistakes landing on a plausible number.

    WHY IT MATTERS. Roadmap 6.6a is the defect in which a 13-model ensemble SILENTLY BECAME A
    12-MODEL ENSEMBLE -- the Kolmogorov-Arnold Network's out-of-fold step raised, a bare
    `except Exception` swallowed it, and the model vanished from `trained_models_`, from the
    blend, and from every cross-algorithm comparison artifact, while the run reported normal
    metrics and looked healthy.

    Anyone checking a run's TWELVE models against a README that said TWELVE would have concluded
    the ensemble was complete. **The document would have concealed the exact defect it took weeks
    to find.** A roster count that disagrees with the code is not a typo in a doc; it is a
    disabled alarm.

    The roster is read from a REAL VariantEnsemble instance -- not from a regex over the source
    -- so the conditional members (`kan` behind `_KAN_AVAILABLE and not cfg.skip_kan`,
    `catboost`, the SVMs, `mc_dropout`) are resolved exactly as a run resolves them.
    """
    from genomic_variant_classifier.models.variant_ensemble import (
        EnsembleConfig,
        VariantEnsemble,
    )

    ens = VariantEnsemble(EnsembleConfig(model_dir=str(tmp_path)))

    # `base_estimators` is THE roster -- it is exactly what fit() writes into
    # `ensemble_completeness_["roster"]` (variant_ensemble.py:2286). Read it from a live
    # instance, not with a regex over the source: `kan` is added via dict-unpacking behind
    # `_KAN_AVAILABLE and not cfg.skip_kan`, and a naive line-regex MISSES IT -- which is
    # precisely the mistake that produced the wrong count in the first place.
    roster = set(ens.base_estimators)

    assert roster, "VariantEnsemble built no base estimators at all -- the ensemble is empty."

    # The README table rows look like:  | 11 | `kan` | Kolmogorov-Arnold Network |
    documented = set(re.findall(r"^\|\s*\d+\s*\|\s*`([a-z_0-9]+)`\s*\|", readme, re.M))
    assert documented, (
        "no base-model roster table found in README.md (expected rows like "
        "'| 11 | `kan` | ... |'). If the table was restructured, FIX THIS PATTERN -- do not "
        "delete the assertion."
    )

    missing = sorted(roster - documented)     # in the code, absent from the README
    extra = sorted(documented - roster)       # in the README, absent from the code

    assert not missing and not extra, (
        f"THE README'S BASE-MODEL ROSTER DISAGREES WITH VariantEnsemble.\n"
        f"\n"
        f"  in the ensemble but NOT documented ({len(missing)}): {missing}\n"
        f"  documented but NOT in the ensemble ({len(extra)}): {extra}\n"
        f"\n"
        f"  actual roster ({len(roster)}): {sorted(roster)}\n"
        f"  README says   ({len(documented)}): {sorted(documented)}\n"
        f"\n"
        f"This is roadmap 6.6a territory. A README that undercounts the roster would let a "
        f"SILENTLY DROPPED base model look like normal operation -- which is exactly how a "
        f"13-model ensemble became a 12-model ensemble and published a cross-algorithm "
        f"comparison with a headline model missing."
    )


# ---------------------------------------------------------------------------
# 3c. THE AGENT ROSTER -- 22, and the README must not say 13
# ---------------------------------------------------------------------------

#: Every place the README CLAIMS the agent count. Enumerated, not swept -- see
#: CONTRACT_CLAIM_PATTERNS for why a blanket regex cannot tell a claim from a history note.
#: REPOINTED 2026-07-15. The rewrite states the agent count in three places rather than six,
#: and the roster table below carries every class name -- which
#: test_readme_agent_roster_matches_the_orchestrator_registry checks against a LIVE
#: Orchestrator._register_agents(). The count and the names are therefore both bound, and the
#: names are the stronger binding: the pre-2026-07-14 README said 13, got EIGHT of the
#: thirteen names wrong, and omitted NINE agents entirely.
AGENT_COUNT_CLAIM_PATTERNS: dict[str, str] = {
    "shields.io badge":    r"autonomous%20agents-(\d+)",
    "opening paragraph":   r"autonomous layer of \*\*(\d+) specialised agents\*\*",
    "agent-layer heading": r"## Autonomous agent layer \((\d+) agents\)",
}


def test_readme_agent_roster_matches_the_orchestrator_registry(readme, tmp_path):
    """The README's agent table must equal Orchestrator's actual registry.

    THE README SAID THIRTEEN. THERE ARE TWENTY-TWO. A 41% UNDERCOUNT of the supervisory layer
    the document calls the system's defining feature -- and EIGHT of the thirteen names it did
    give were wrong.

    Verified 2026-07-14 by three independent methods that agree exactly:
      * abstract-syntax-tree inheritance analysis -> 22 concrete subclasses of BaseAgent
        (NOTE: a first pass returned 13 because it filtered on `"Agent" in base_name`, which
        MISSES every agent inheriting from `DriftMonitorBase` -- a base class whose name
        contains no "Agent". It returned exactly the README's wrong number, which is precisely
        how a malformed search launders a wrong answer into a confirmation.)
      * Orchestrator._register_agents() -> 22 entries
      * scripts/check_agents_active.py -> "22 agents (registered=22, scheduled=22)"

    The wrong names mattered as much as the count. The README listed `SchemaDriftAgent`, and a
    class by that name EXISTS -- but it is the schema-drift DETECTOR used by
    run_schema_drift_check.py, not an agent: it does not descend from BaseAgent and is not in
    the registry. A reader auditing the agent layer against that table would have gone looking
    for the wrong object.

    This reads the registry from a LIVE orchestrator instance rather than parsing the source,
    for the same reason the base-model roster test does: the registry is built inside a method,
    its values are `_Lazy("module:Class")` strings, and a regex over the file is exactly the
    kind of thing that produced the wrong count in the first place.
    """
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator

    # __new__ + _register_agents(): the registry is built from string literals and _Lazy
    # wrappers only, so it needs no __init__ state -- and __init__ would touch shared_state on
    # disk, which a unit test must not do.
    orch = Orchestrator.__new__(Orchestrator)
    orch._register_agents()
    registry = set(orch._agent_registry)

    assert registry, "Orchestrator registered NO agents -- the registry is empty."

    # Rows look like:  | `DataFreshnessAgent` | Polls ClinVar, ... |
    documented = set(re.findall(r"^\|\s*`([A-Za-z_0-9]+Agent)`\s*\|", readme, re.M))
    assert documented, (
        "no agent roster table found in README.md (expected rows like "
        "'| `DataFreshnessAgent` | ... |'). If the table was restructured, FIX THIS PATTERN -- "
        "do not delete the assertion."
    )

    missing = sorted(registry - documented)     # registered but undocumented
    extra = sorted(documented - registry)       # documented but not registered

    assert not missing and not extra, (
        f"THE README'S AGENT ROSTER DISAGREES WITH Orchestrator._register_agents().\n"
        f"\n"
        f"  registered but NOT documented ({len(missing)}): {missing}\n"
        f"  documented but NOT registered ({len(extra)}): {extra}\n"
        f"\n"
        f"  actual registry ({len(registry)}): {sorted(registry)}\n"
        f"  README documents ({len(documented)}): {sorted(documented)}\n"
        f"\n"
        f"Cross-check with the project's own liveness checker, which has been able to answer "
        f"this the whole time:\n"
        f"    python scripts/check_agents_active.py"
    )


def test_readme_never_states_a_wrong_agent_count(readme, tmp_path):
    """Every CLAIM about the agent count must equal the registry's size.

    Six claim sites: a badge, the opening paragraph, the architecture diagram, a key-properties
    bullet, the section heading, and the repository-structure listing. ALL SIX said 13. The
    README stated this number six times and was wrong six times -- which is what a number
    written down six times and never re-derived does.
    """
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)
    orch._register_agents()
    n_agents = len(orch._agent_registry)

    found: dict[str, int] = {}
    missing: list[str] = []
    for site, pattern in AGENT_COUNT_CLAIM_PATTERNS.items():
        m = re.search(pattern, readme)
        if m is None:
            missing.append(site)
        else:
            found[site] = int(m.group(1))

    assert not missing, (
        f"these agent-count claim sites are no longer found in README.md: {missing}\n"
        f"Either the README was restructured (FIX THE PATTERN) or the claim was dropped. Do "
        f"not delete the entry to make this pass."
    )

    wrong = {site: n for site, n in found.items() if n != n_agents}
    assert not wrong, (
        f"README.md asserts the wrong agent count in {len(wrong)} place(s):\n"
        + "\n".join(f"    {site:26s} says {n}" for site, n in wrong.items())
        + f"\n\nOrchestrator._register_agents() registers {n_agents}."
    )


# ---------------------------------------------------------------------------
# 4. THE DRIFT-MONITOR EXIT CODES -- 4 exists, and 4 means NOT CHECKED
# ---------------------------------------------------------------------------

def test_readme_documents_the_not_checked_exit_code(readme):
    """Exit 4 = NOT CHECKED. The README said 0/1/2/3 -- and 4 is the whole point.

    The scheduled monitor's original defect was that a run which measured NOTHING exited 0, and
    0 means "no drift" -- so it reported a clean bill of health every month having never read a
    row of data (roadmap 6.20). The obvious fix -- make those paths exit 3 -- is the same bug in
    the opposite costume, because 3 means urgent_retrain and would fire a SEVERE DRIFT alarm on
    a healthy model.

    So "I could not look" got its own code. A README that documents 0/1/2/3 hides the single
    most important thing about how this monitor now behaves.
    """
    assert re.search(r"0\s*/\s*1\s*/\s*2\s*/\s*3\s*/\s*4", readme), (
        "README.md does not document run_drift_monitor.py's exit code 4 (NOT CHECKED). It "
        "previously documented '0/1/2/3', which omits the code that distinguishes 'I looked and "
        "saw nothing' from 'I could not look'. That distinction is the entire fix for roadmap "
        "6.20."
    )

    # And the exit code must actually EXIST in the script -- the README must not document a
    # capability the code does not have. (That is how this whole audit started.)
    script = Path("scripts/run_drift_monitor.py").read_text(encoding="utf-8")
    assert "EXIT_NOT_CHECKED = 4" in script, (
        "README.md documents exit code 4, but scripts/run_drift_monitor.py no longer defines "
        "EXIT_NOT_CHECKED = 4. The document and the code have diverged -- and the document is "
        "now claiming a behaviour the code does not have."
    )


# ---------------------------------------------------------------------------
# 5. THE QUICKSTART MUST BE RUNNABLE
# ---------------------------------------------------------------------------

def _commands_in_readme(readme: str) -> list[tuple[str, set[str]]]:
    """Extract every `python scripts/<x>.py` invocation and ITS OWN flags.

    Parsed PER COMMAND, honouring backslash line-continuations. The first version of this test
    filtered at CODE-BLOCK level -- "does this block mention run_phase2_eval.py?" -- and then
    harvested every flag in the block. The README's quickstart is a SINGLE block containing four
    commands, so that test attributed `--port` (uvicorn) and `--reference-splits`,
    `--new-clinvar`, `--old-clinvar`, `--output-dir`, `--auto-retrain` (run_drift_monitor.py) to
    run_phase2_eval.py, and reported six false failures.

    All six flags were perfectly valid -- for the scripts they actually belonged to. The test
    was measuring the wrong thing while looking authoritative, which is the exact failure mode
    this file exists to catch. Fixed by attributing each flag to the command it is written on.
    """
    commands: list[tuple[str, set[str]]] = []

    for block in re.findall(r"```[^\n]*\n(.*?)```", readme, re.S):
        lines = block.splitlines()
        i = 0
        while i < len(lines):
            m = re.search(r"python\s+(scripts/[\w./-]+\.py)", lines[i])
            if not m:
                i += 1
                continue

            script = m.group(1)
            body = [lines[i]]
            # Follow backslash continuations to the end of THIS command.
            while body[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                body.append(lines[i])

            # Drop comment lines: the README deliberately lists further flags in comments
            # (e.g. "# ... plus --gnomad, --spliceai, ..."), and those are illustrative, not
            # part of the runnable command.
            runnable = "\n".join(ln for ln in body if not ln.strip().startswith("#"))
            flags = set(re.findall(r"(--[a-zA-Z0-9][a-zA-Z0-9-]*)", runnable))
            commands.append((script, flags))
            i += 1

    return commands


def test_readme_quickstart_uses_no_flag_that_does_not_exist(readme):
    """Every flag the README hands you must exist in the script it hands it to.

    `--parquet` sat in the training quickstart until 2026-07-14. It has NEVER been a flag --
    `run_phase2_eval.py` takes `--clinvar`. Anyone who copied the command out of the README of
    this public repository got an argparse error. That is not staleness; it is a command that
    could never have worked, published as the way to train the model.

    Checked for EVERY script the README invokes, not just that one -- because the next wrong
    flag will be somewhere else.
    """
    commands = _commands_in_readme(readme)

    assert commands, (
        "no `python scripts/*.py` invocations found in README.md. Either the quickstart was "
        "removed, or this parser has stopped matching. A test that checks nothing is not a "
        "test -- fix the parser, do not delete the assertion."
    )

    problems: list[str] = []
    for script, used in commands:
        path = Path(script)
        if not path.is_file():
            problems.append(
                f"{script}: THE SCRIPT DOES NOT EXIST. The README tells the reader to run a "
                f"file that is not in the repository."
            )
            continue

        source = path.read_text(encoding="utf-8")
        defined = set(re.findall(r"['\"](--[a-zA-Z0-9][a-zA-Z0-9-]*)['\"]", source))
        if not defined:
            problems.append(f"{script}: could not parse any flags out of the script itself")
            continue

        unknown = sorted(used - defined)
        if unknown:
            problems.append(
                f"{script}: uses flag(s) it does NOT define: {unknown}\n"
                f"        flags it DOES define: {sorted(defined)}"
            )

    assert not problems, (
        "README.md documents commands that cannot run:\n\n"
        + "\n\n".join(f"  {p}" for p in problems)
        + "\n\nUntil 2026-07-14 the training quickstart used `--parquet`, which has never "
          "existed. A README that tells you to run something that cannot run is worse than a "
          "README with no quickstart at all."
    )


# ---------------------------------------------------------------------------
# 6. THE PERFORMANCE-FIGURE BAN WAS DELETED ON 2026-07-15, DELIBERATELY
# ---------------------------------------------------------------------------
#
# `test_readme_publishes_no_performance_figures_pending_run17` lived here. It banned any
# bare four-decimal figure in the 0.9x range from README.md, on the grounds that every
# performance number this project had reported came from Run 15 -- whose feature space was
# 46% constant zero (roadmap 6.21), so its headline 0.9984 was produced by the 38 features
# that were real out of a claimed 80.
#
# Its own docstring said: "This test keeps them withdrawn until someone deliberately
# deletes it -- which is the point: reinstating a headline metric should be a decision,
# not an oversight."
#
# THE OWNER MADE THAT DECISION ON 2026-07-15. The test is deleted, not disabled, because
# that is the mechanism it described. The README now carries the Run 15 AUROC under an
# "Early results" heading which states, in the document itself, that the figure describes
# an earlier and narrower configuration of the system, that the feature space, model
# roster, split protocol and data-integrity gates have all changed since, and that a
# like-for-like table will be published from the next full run.
#
# That is the honest disposition, and it is a better one than silence: the project has a
# large planned build-out ahead of it, the number WILL change, and a README that shows an
# early waypoint and says so is more informative than a README that shows nothing.
#
# The remaining tests in this file still bind the README's COUNTS -- features, agents,
# models, tests -- to the code. They are the part worth keeping: a wrong count is a wrong
# claim about what the system IS, whereas a stale metric, labelled stale, is a dated
# measurement. The two are not the same kind of error.
