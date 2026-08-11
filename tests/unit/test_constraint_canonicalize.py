"""Tests for the gnomAD constraint canonicaliser -- DUPLICATE-1A.

Every fixture below reproduces a structure MEASURED in
`gnomad.v4.1.constraint_metrics.tsv` on 2026-08-09, not an invented one. Where
a number appears it is the number that was observed.

Imports are package-qualified, so this file is repository-shaped as written and
needs no transformation at install time. An earlier version used a bare module
import, which passed in isolation and failed the moment it was placed under
tests/unit -- the installer rolled the whole placement back, correctly.

Author: Monzia Moodie
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import genomic_variant_classifier.data.constraint_canonicalize as cc
from genomic_variant_classifier.data.constraint_canonicalize import (
    ConstraintSourceError, GeneIdNamespace, NAMESPACE_ATOL, OE_ARITHMETIC_ATOL,
    TranscriptSelectionTier, TranscriptSourceFacts, assert_row_conservation,
    canonicalize_mane_constraint, describe_transcript_source, to_numeric_strict,
    derive_gene_is_constrained, exact_duplicate_groups, namespace_of,
    select_constraint_transcripts, validate_published_oe,
)


def _row(gene, gene_id, transcript, obs, exp, oe, loeuf, mane="true",
         canonical=None):
    if canonical is None:
        canonical = mane
    return {"gene": gene, "gene_id": gene_id, "transcript": transcript,
            "mane_select": mane, "canonical": canonical,
            "lof.obs": obs, "lof.exp": exp,
            "lof.oe": oe, "lof.oe_ci.upper": loeuf}


def measured_source() -> pd.DataFrame:
    """The four pairing shapes measured in the real file, in miniature."""
    rows = [
        # 1. An ordinary pair sharing one symbol (17,468 of these).
        _row("ZZZ3", "26009", "NM_015534.6", 40.0, 80.0, 0.5, 0.72),
        _row("ZZZ3", "ENSG00000036549", "ENST00000370801", 40.0, 80.0, 0.5, 0.72),
        # 2. A pair split by SYMBOL DISAGREEMENT (5 of these). Identical
        #    metrics, different names -- RefSeq says SCHIP1, Ensembl says
        #    IQCJ-SCHIP1. Both must survive so either annotation joins.
        _row("SCHIP1", "29970", "NM_014575.4", 18.0, 41.823, 0.43038, 0.638),
        _row("IQCJ-SCHIP1", "ENSG00000283154", "ENST00000638749",
             18.0, 41.823, 0.43038, 0.638),
        # 3. A pair split because Ensembl has NO SYMBOL and RefSeq uses a
        #    provisional LOC* placeholder (8 of these).
        _row("LOC728392", "728392", "NM_001162371.3", 4.0, 3.2926, 1.2148, 1.896),
        _row(np.nan, "ENSG00000286190", "ENST00000568641",
             4.0, 3.2926, 1.2148, 1.896),
        # 4. A well-powered gene whose point estimate exceeds the reported
        #    upper bound (12 of these; DNMT3A is real, lof.exp 106.98).
        _row("DNMT3A", "1788", "NM_022552.5", 267.0, 106.98, 2.4957, 1.998),
        _row("DNMT3A", "ENSG00000119772", "ENST00000321117",
             267.0, 106.98, 2.4957, 1.998),
        # A non-MANE row that must never be selected.
        _row("ZZZ3", "ENSG00000036549", "ENST00000000001",
             1.0, 1.0, 1.0, 1.0, mane="false"),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def test_namespace_detection():
    assert namespace_of("ENSG00000036549") is GeneIdNamespace.ENSEMBL
    assert namespace_of("ENSG00000036549.12") is GeneIdNamespace.ENSEMBL
    assert namespace_of("26009") is GeneIdNamespace.NCBI_GENE
    assert namespace_of(26009) is GeneIdNamespace.NCBI_GENE


def test_canonical_index_is_one_row_per_symbol():
    out, audit = canonicalize_mane_constraint(measured_source())
    assert out["gene"].is_unique
    # ZZZ3, SCHIP1, IQCJ-SCHIP1, LOC728392, DNMT3A
    assert sorted(out["gene"]) == ["DNMT3A", "IQCJ-SCHIP1", "LOC728392",
                                   "SCHIP1", "ZZZ3"]
    assert audit.n_genes_canonical == 5


def test_ensembl_is_preferred_and_refseq_falls_back():
    out, audit = canonicalize_mane_constraint(measured_source())
    by = out.set_index("gene")
    assert by.loc["ZZZ3", "gene_id"] == "ENSG00000036549"
    assert by.loc["DNMT3A", "gene_id"] == "ENSG00000119772"
    # LOC728392 exists only as RefSeq once its symbol-less Ensembl twin is
    # excluded, so it must survive by explicit fallback rather than be dropped.
    assert by.loc["LOC728392", "gene_id"] == "728392"
    assert audit.n_genes_ensembl_preferred == 3
    assert audit.n_genes_refseq_fallback == 2


def test_null_symbol_rows_are_excluded_explicitly_not_silently():
    out, audit = canonicalize_mane_constraint(measured_source())
    assert audit.n_rows_null_symbol_excluded == 1
    assert audit.excluded_null_symbol_gene_ids == ("ENSG00000286190",)
    assert any("requires a gene symbol" in n for n in audit.notes)
    # The runtime must NOT claim to have proved the partner pairing.
    assert any("does not re-establish that pairing" in n for n in audit.notes)


def test_row_conservation_holds():
    """Grouped + explicitly excluded must equal the SELECTED row count.

    SELECTED, not MANE: the count includes the canonical-fallback tier. On the
    real source that tier contributes 696 genes, so a field named for MANE
    while holding the sum of two tiers is false in production.

    The real file, MANE tier: 34,954 grouped + 8 excluded = 34,962.
    """
    src = measured_source()
    out, audit = canonicalize_mane_constraint(src)
    assert audit.n_rows_grouped + audit.n_rows_null_symbol_excluded == audit.n_rows_selected
    tiers = dict(audit.tier_counts)
    assert audit.n_genes_canonical == (
        tiers["mane_select"] + tiers["canonical"]), (
        "the gene count does not reconcile against the tier counts")


def test_the_emitted_audit_is_IMMUTABLE():
    """A record that documents itself as evidence must be unable to change.

    An earlier version froze only `notes` while a comment called the whole
    record sealed. A probe then set n_genes_canonical = 999, a NEGATIVE tier
    count and a fabricated note -- all accepted. A half-frozen audit is worse
    than a mutable one, because a reader assumes guarantees that do not exist.
    """
    import dataclasses
    _, audit = canonicalize_mane_constraint(measured_source())
    for field_name, value in (("n_genes_canonical", 999),
                              ("n_rows_selected", -1),
                              ("notes", ("fabricated",)),
                              ("source_sha256", "0" * 64)):
        try:
            setattr(audit, field_name, value)
        except dataclasses.FrozenInstanceError:
            continue
        raise AssertionError(
            "the sealed audit accepted a write to {!r}".format(field_name))
    assert isinstance(audit.tier_counts, tuple), (
        "tier_counts is still a mutable mapping: {!r}".format(type(audit.tier_counts)))
    assert isinstance(audit.notes, tuple)
    assert isinstance(audit.excluded_null_symbol_gene_ids, tuple)


def test_non_mane_rows_are_never_selected():
    out, _ = canonicalize_mane_constraint(measured_source())
    assert "ENST00000000001" not in set(out["transcript"])


def test_oe_exceeds_upper_bound_is_recorded_and_value_untouched():
    """DNMT3A: lof.oe 2.4957 against a reported upper bound of 1.998.

    The observation is recorded. The value is NOT clipped -- min(oe, loeuf)
    would manufacture a statistic gnomAD never published.
    """
    out, audit = canonicalize_mane_constraint(measured_source())
    by = out.set_index("gene")
    assert bool(by.loc["DNMT3A", "oe_exceeds_reported_upper_bound"]) is True
    assert float(by.loc["DNMT3A", "lof.oe"]) == 2.4957
    assert audit.n_oe_exceeds_reported_upper_bound == 1
    assert bool(by.loc["ZZZ3", "oe_exceeds_reported_upper_bound"]) is False


# ---- the drift tripwire ---------------------------------------------------
def test_namespace_disagreement_raises():
    """MUST FAIL. Measured 2026-08-09: 0 of 17,486 genes disagree. If a future
    release ever does, the build stops instead of silently preferring one."""
    # ISOLATE the disagreement. Perturbing lof.oe would break the obs/exp
    # identity as well, and that check runs FIRST -- so the test would pass for
    # the wrong reason and prove nothing about namespace equivalence. Caught
    # 2026-08-09 by running it. lof.oe_ci.upper is not part of the arithmetic
    # identity, so perturbing it exercises exactly one invariant.
    src = measured_source()
    src.loc[(src["gene"] == "ZZZ3") & (src["gene_id"] == "26009"),
            "lof.oe_ci.upper"] = 0.90
    try:
        canonicalize_mane_constraint(src)
    except ConstraintSourceError as exc:
        assert "DISAGREE" in str(exc), str(exc)
        assert "ZZZ3" in str(exc), str(exc)
        assert "lof.oe_ci.upper" in str(exc), str(exc)
        return
    raise AssertionError("a namespace metric disagreement was NOT detected")


def test_arithmetic_invariant_fires_before_namespace_equivalence():
    """ORDERING, pinned. A row that breaks both must report the arithmetic
    failure, because obs/exp is the stronger and more specific claim. Without
    this pin, the previous disagreement test passed for the wrong reason."""
    src = measured_source()
    src.loc[(src["gene"] == "ZZZ3") & (src["gene_id"] == "26009"), "lof.oe"] = 0.9
    try:
        canonicalize_mane_constraint(src)
    except ConstraintSourceError as exc:
        assert "obs/exp identity" in str(exc), str(exc)
        return
    raise AssertionError("a row breaking both invariants raised neither")


def test_namespace_equivalence_admits_no_rounding_allowance():
    """NAMESPACE_ATOL is 0.0, and that is deliberate.

    The two rows are two ENCODINGS of one published record; no calculation
    happens between them, so nothing may differ. Inheriting the arithmetic
    tolerance of 5e-4 here would import a rounding allowance into a place
    where no rounding occurs. Measured 2026-08-09: 0 disagreements across
    17,486 gene symbols, so exact equality is what the source actually shows.
    """
    assert NAMESPACE_ATOL == 0.0
    assert OE_ARITHMETIC_ATOL > 0.0
    src = measured_source()
    src.loc[(src["gene"] == "ZZZ3") & (src["gene_id"] == "26009"),
            "lof.oe_ci.upper"] = 0.72 + 1e-9
    try:
        canonicalize_mane_constraint(src)
    except ConstraintSourceError as exc:
        assert "DISAGREE" in str(exc)
        return
    raise AssertionError(
        "a namespace difference of 1e-9 was tolerated; NAMESPACE_ATOL is not "
        "load-bearing")


def test_namespace_missingness_disagreement_raises():
    """MUST FAIL. One representation populated, the other missing.

    dropna() before comparing would call these equivalent -- and Ensembl is
    then preferred, so a real RefSeq measurement would be silently replaced
    by missing data.
    """
    src = measured_source()
    mask = (src["gene"] == "ZZZ3") & (src["gene_id"] == "ENSG00000036549")
    src.loc[mask, "lof.oe"] = np.nan
    try:
        canonicalize_mane_constraint(src)
    except ConstraintSourceError as exc:
        assert "missing while another is populated" in str(exc), str(exc)
        return
    raise AssertionError("a missingness disagreement was NOT detected")


def test_namespace_all_missing_is_not_a_disagreement():
    """Both representations missing the same metric is consistent, not a
    disagreement. Two of the eight LOC* pairs are metric-null on both sides."""
    src = measured_source()
    mask = src["gene"].isin(["ZZZ3"])
    src.loc[mask, "lof.oe_ci.upper"] = np.nan
    out, _ = canonicalize_mane_constraint(src)
    assert len(out) == 5


def test_unknown_gene_id_namespace_raises():
    for bad in ("ENST00000370801", "garbage", "", "  ", "ENSG_", "ENSG",
                "ENSGabc", "12a34"):
        try:
            namespace_of(bad)
        except ConstraintSourceError:
            continue
        raise AssertionError(
            "namespace_of({!r}) classified an unrecognised identifier instead "
            "of refusing".format(bad))


def test_missing_gene_id_is_diagnosed_as_MISSING_not_as_unrecognised():
    """The message is the point, not merely that something raised.

    A sabotage run on 2026-08-09 deleted the null/blank guard and every test
    still passed, because a null identifier fell through to the catch-all
    "unrecognised namespace" raise. Control flow was unchanged; DIAGNOSIS was
    not. A missing identifier is a data-completeness problem and garbage is a
    format problem, and an operator needs to be told which.
    """
    for bad in (None, float("nan"), "", "   ", "NaN", "<NA>"):
        try:
            namespace_of(bad)
        except ConstraintSourceError as exc:
            assert "missing, blank or null-like" in str(exc), (
                "namespace_of({!r}) raised {!r}, which diagnoses a FORMAT "
                "problem for what is a MISSING VALUE".format(bad, str(exc)))
            continue
        raise AssertionError(
            "namespace_of({!r}) fell through to NCBI_GENE".format(bad))


def test_garbage_gene_id_is_diagnosed_as_UNRECOGNISED_not_as_missing():
    for bad in ("ENST00000370801", "garbage", "ENSG_", "ENSGabc", "12a34"):
        try:
            namespace_of(bad)
        except ConstraintSourceError as exc:
            assert "unrecognised" in str(exc), (
                "namespace_of({!r}) raised {!r}, which diagnoses a MISSING "
                "VALUE for what is malformed".format(bad, str(exc)))
            continue
        raise AssertionError("namespace_of({!r}) was accepted".format(bad))


# ---- the Boolean must carry three states ---------------------------------
def test_missing_loeuf_does_not_mean_unconstrained():
    """The CONSTRAINTFILL-1 shape, one layer down. `np.nan < 0.35` is False."""
    got = derive_gene_is_constrained(pd.Series([0.20, 0.80, np.nan]),
                                     threshold=0.35)
    assert got.iloc[0] == 1
    assert got.iloc[1] == 0
    assert pd.isna(got.iloc[2]), (
        "a gene with no constraint data was recorded as NOT CONSTRAINED; "
        "absence of evidence is not evidence of tolerance")


def test_boolean_depends_on_loeuf_not_on_the_point_estimate():
    """The dependency-graph reversal, pinned.

    oe says constrained, loeuf says not, and vice versa. The Boolean must
    follow loeuf in both rows.
    """
    loeuf = pd.Series([0.50, 0.20])
    got = derive_gene_is_constrained(loeuf, threshold=0.35)
    assert got.tolist() == [0, 1]


def test_known_loeuf_with_missing_point_estimate_still_yields_a_boolean():
    got = derive_gene_is_constrained(pd.Series([0.20]), threshold=0.35)
    assert got.iloc[0] == 1


def test_known_point_estimate_with_missing_loeuf_leaves_the_boolean_na():
    got = derive_gene_is_constrained(pd.Series([np.nan]), threshold=0.35)
    assert pd.isna(got.iloc[0])


def test_two_ensembl_rows_for_one_symbol_raises():
    src = measured_source()
    extra = _row("ZZZ3", "ENSG00000099999", "ENST00000999999", 40.0, 80.0, 0.5, 0.72)
    src = pd.concat([src, pd.DataFrame([extra])], ignore_index=True)
    try:
        canonicalize_mane_constraint(src)
    except ConstraintSourceError as exc:
        assert "Ensembl MANE rows" in str(exc)
        return
    raise AssertionError("transcript ambiguity was resolved silently")


# ---- the arithmetic invariant --------------------------------------------
def test_oe_arithmetic_invariant_passes_on_measured_rounding():
    src = measured_source()
    v = validate_published_oe(src)
    assert v.n_failed == 0
    assert v.n_checked == len(src)


def test_oe_arithmetic_invariant_raises_on_a_real_break():
    src = measured_source()
    src.loc[0, "lof.oe"] = 0.9          # obs/exp is 0.5
    try:
        validate_published_oe(src)
    except ConstraintSourceError as exc:
        assert "obs/exp identity" in str(exc)
        return
    raise AssertionError("a broken obs/exp identity was NOT detected")


def test_oe_arithmetic_skips_unusable_rows_rather_than_passing_them():
    """exp <= 0 and non-finite triples are UNCHECKABLE, not passing."""
    src = measured_source()
    src.loc[0, "lof.exp"] = 0.0
    src.loc[1, "lof.oe"] = np.nan
    v = validate_published_oe(src)
    assert v.n_checked == len(src) - 2
    assert v.n_failed == 0


# ---- the duplicate gate ---------------------------------------------------
def test_duplicate_gate_detects_the_constraint_alias():
    """The exact defect: gene_constraint_oe bit-identical to loeuf."""
    df = pd.DataFrame({"loeuf": [0.1, 0.2, 0.9],
                       "gene_constraint_oe": [0.1, 0.2, 0.9],
                       "other": [1.0, 2.0, 3.0]})
    assert exact_duplicate_groups(df) == [("loeuf", "gene_constraint_oe")]


def test_two_different_constant_defaults_are_not_duplicate_signal():
    """A constant column is a VITALITY failure, not a duplicate-signal one.

    This is why the gate runs PRE-TRANSFORM: after standardisation both of
    these become 0.0 and would be reported as duplicates.
    """
    df = pd.DataFrame({"a": [0.0, 0.0, 0.0], "b": [5.0, 5.0, 5.0]})
    assert exact_duplicate_groups(df) == []


def test_duplicate_gate_is_clean_on_distinct_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.5]})
    assert exact_duplicate_groups(df) == []


def test_duplicate_gate_matches_on_null_position_not_only_values():
    a = pd.Series([1.0, np.nan, 3.0])
    b = pd.Series([1.0, np.nan, 3.0])
    c = pd.Series([1.0, 2.0, 3.0])
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    assert exact_duplicate_groups(df) == [("a", "b")]


def test_duplicate_gate_finds_a_three_way_group():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "z": [1.0, 2.0],
                       "w": [9.0, 8.0]})
    groups = exact_duplicate_groups(df)
    assert len(groups) == 1 and set(groups[0]) == {"x", "y", "z"}


def test_row_conservation_assertion_actually_raises():
    """The assertion itself, exercised directly.

    A sabotage run on 2026-08-09 disabled the conservation check and every
    test still passed, because a clean fixture never loses rows. An assertion
    that is never exercised is not a check.
    """
    assert_row_conservation(34954, 34954, 8, 34962)      # the measured real case
    cases = (
        (8, 9, 0, 9),          # groupby dropped one of the retained rows
        (34954, 34954, 0, 34962),
        (10, 10, 8, 9),
        # THE CANCELLATION CASE. Under a single identity 8 + 1 == 9 balanced
        # while a row had in fact been dropped inside groupby.
        (8, 9, 1, 10),
    )
    for grouped, retained, excluded, mane in cases:
        try:
            assert_row_conservation(grouped, retained, excluded, mane)
        except ConstraintSourceError as exc:
            assert "LOST rows" in str(exc)
            continue
        raise AssertionError(
            "conservation accepted grouped={} retained={} excluded={} "
            "mane={}".format(grouped, retained, excluded, mane))


def test_duplicate_gate_skips_two_IDENTICAL_dead_columns():
    """The skip is load-bearing only when the constants are the SAME.

    The earlier fixture used 0.0 and 5.0, whose hashes differ, so removing the
    degenerate-column skip changed nothing and a sabotage run went undetected.
    After standardisation every dead feature becomes 0.0, so THIS is the shape
    that would flood the gate with meaningless pairs.
    """
    df = pd.DataFrame({"dead_a": [0.0, 0.0, 0.0], "dead_b": [0.0, 0.0, 0.0],
                       "live": [1.0, 2.0, 3.0]})
    assert exact_duplicate_groups(df) == []


def test_duplicate_gate_requires_equality_not_only_a_hash_match():
    """Hash identifies CANDIDATES; equality establishes the FINDING.

    Forced by making every column share a fingerprint. Without the equality
    check two unrelated columns would be reported as duplicates -- untestable
    otherwise, since a real SHA-256 collision cannot be manufactured.
    """
    original = cc._fingerprint
    cc._fingerprint = lambda s: "collision"
    try:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [9.0, 8.0, 7.0]})
        assert exact_duplicate_groups(df) == [], (
            "a hash collision between DIFFERENT columns was reported as a "
            "duplicate; the equality check is not load-bearing")
        df2 = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [5.0, 6.0]})
        assert exact_duplicate_groups(df2) == [("a", "b")]
    finally:
        cc._fingerprint = original


# ---- the transcript-selection tier ladder --------------------------------
def _tiered_source():
    """Three genes, one per tier. MEASURED shape: 18,203 gene symbols in the
    source, 17,486 with a MANE Select transcript, so ~718 have none."""
    return pd.DataFrame([
        # MANE available -> MANE tier
        _row("HASMANE", "ENSG00000000001", "ENST1", 40.0, 80.0, 0.5, 0.72,
             mane="true", canonical="true"),
        _row("HASMANE", "1001", "NM_1", 40.0, 80.0, 0.5, 0.72,
             mane="true", canonical="true"),
        # No MANE, but canonical -> CANONICAL tier
        _row("CANONONLY", "ENSG00000000002", "ENST2", 10.0, 20.0, 0.5, 0.90,
             mane="false", canonical="true"),
        _row("CANONONLY", "ENSG00000000003", "ENST2b", 10.0, 20.0, 0.5, 0.90,
             mane="false", canonical="false"),
        # Neither -> UNSELECTED, retained only on request
        _row("NEITHER", "ENSG00000000004", "ENST3", 5.0, 10.0, 0.5, 1.10,
             mane="false", canonical="false"),
    ])


def test_tier_ladder_prefers_mane_then_canonical():
    frame, counts = select_constraint_transcripts(_tiered_source(),
                                                  allow_canonical_fallback=True)
    tiers = dict(zip(frame["gene"], frame["_tier"]))
    assert tiers["HASMANE"] == TranscriptSelectionTier.MANE_SELECT.value
    assert tiers["CANONONLY"] == TranscriptSelectionTier.CANONICAL.value
    # NEITHER has no declared transcript: recorded, never retained.
    assert "NEITHER" not in tiers
    assert counts[TranscriptSelectionTier.MANE_SELECT.value] == 1
    assert counts[TranscriptSelectionTier.CANONICAL.value] == 1
    assert counts["no_declared_transcript"] == 1


def test_a_gene_with_mane_is_never_taken_from_a_lower_tier():
    """The ladder must not mix tiers WITHIN a gene. HASMANE has a canonical
    flag too; it must still be selected as MANE."""
    frame, _ = select_constraint_transcripts(_tiered_source(), allow_canonical_fallback=True)
    rows = frame[frame["gene"] == "HASMANE"]
    assert set(rows["_tier"]) == {TranscriptSelectionTier.MANE_SELECT.value}


def test_a_gene_with_no_declared_transcript_is_never_retained():
    """UNSELECTED is an AUDIT category, never a production tier.

    Retaining an arbitrary transcript would state a biological measurement the
    source does not support. The gene carries missing constraint instead, and
    the count is recorded so the ~718 real cases cannot vanish unnoticed.
    """
    for fallback in (True, False):
        frame, counts = select_constraint_transcripts(
            _tiered_source(), allow_canonical_fallback=fallback)
        assert "NEITHER" not in set(frame["gene"])
        assert TranscriptSelectionTier.UNSELECTED.value not in set(frame["_tier"])
        assert counts["no_declared_transcript"] >= 1
        assert counts["canonical_fallback_enabled"] is fallback


def test_each_gene_belongs_to_exactly_one_tier():
    """Tier consistency asserted AT ITS SOURCE, where it is decidable.

    A downstream mixed-tier guard was unreachable -- selection assigns tier per
    gene, so a gene cannot span tiers -- and an unreachable guard is not a
    guard. This asserts the property that makes it unreachable.
    """
    src = pd.concat([_tiered_source(), _tiered_source()], ignore_index=True)
    frame, _ = select_constraint_transcripts(src, allow_canonical_fallback=True)
    per_gene = frame.dropna(subset=["gene"]).groupby("gene")["_tier"].nunique()
    bad = per_gene[per_gene > 1]
    assert bad.empty, "gene(s) span multiple tiers: {}".format(dict(bad))


def test_canonical_only_gene_REACHES_THE_FINAL_INDEX():
    """THE LOAD-BEARING TEST THAT WAS MISSING.

    On 2026-08-09 the tier ladder selected a canonical-tier gene and
    canonicalize_mane_constraint then discarded it again, because it called
    select_mane(). Thirty-six tests passed; none asked whether a canonical-tier
    gene reached the OUTPUT. The component was tested and the PATH was not.
    """
    out, audit = canonicalize_mane_constraint(_tiered_source(),
                                              allow_canonical_fallback=True)
    by = out.set_index("gene")
    assert "HASMANE" in by.index
    assert "CANONONLY" in by.index, (
        "a canonical-tier gene was selected and then dropped before the index "
        "was built; the ladder is not wired into canonicalisation")
    assert by.loc["HASMANE", "_tier"] == TranscriptSelectionTier.MANE_SELECT.value
    assert by.loc["CANONONLY", "_tier"] == TranscriptSelectionTier.CANONICAL.value
    assert "NEITHER" not in by.index
    assert dict(audit.tier_counts)["no_declared_transcript"] == 1


def test_disabling_the_fallback_removes_canonical_genes_from_the_index():
    """MANE-only remains available for percentile calibration: gnomAD's
    v4.1.1 threshold table is derived from 17,063 MANE Select transcripts, so
    calibration must not run on the fallback-extended population."""
    out, audit = canonicalize_mane_constraint(_tiered_source(),
                                              allow_canonical_fallback=False)
    assert set(out["gene"]) == {"HASMANE"}
    assert dict(audit.tier_counts)["canonical_fallback_enabled"] is False


def test_tier_ladder_never_uses_source_row_order():
    """Reversing the source must not change any gene's selected row.

    `drop_duplicates(keep='first')` -- the construct this replaces -- fails
    this outright. Measured on the real file, first-row selection disagrees
    with MANE Select for 31.3% of genes.
    """
    # Compare the full ROW SET, not a dict keyed by gene: HASMANE has two rows
    # (the RefSeq/Ensembl namespace pair) and dict(zip(...)) silently keeps only
    # the last, so reversal flipped the dict while the selection was identical.
    # The first version of this test failed for that reason and not because the
    # ladder was order-dependent.
    src = _tiered_source()
    a, ca = select_constraint_transcripts(src, allow_canonical_fallback=True)
    b, cb = select_constraint_transcripts(src.iloc[::-1].reset_index(drop=True),
                           allow_canonical_fallback=True)
    # itertuples RENAMES columns starting with an underscore to positional
    # names (_tier -> _5), so attribute access silently loses it. Use column
    # access; the first version of this line raised AttributeError and looked
    # like an ordering failure.
    ka = set(zip(a["gene"], a["gene_id"], a["_tier"]))
    kb = set(zip(b["gene"], b["gene_id"], b["_tier"]))
    assert ka == kb, "tier selection depends on source row order"
    assert ca == cb, "tier COUNTS depend on source row order"

    # The stronger property: the CANONICAL OUTPUT is order-invariant too, which
    # is what `drop_duplicates(keep="first")` could never guarantee.
    out_a, _ = canonicalize_mane_constraint(src)
    out_b, _ = canonicalize_mane_constraint(src.iloc[::-1].reset_index(drop=True))
    ra = set(zip(out_a["gene"], out_a["gene_id"]))
    rb = set(zip(out_b["gene"], out_b["gene_id"]))
    assert ra == rb, (
        "canonicalisation depends on source row order: {} vs {}".format(
            sorted(ra), sorted(rb)))


def test_null_symbol_mane_rows_survive_tier_selection():
    """The eight null-symbol MANE rows must reach the canonicaliser's explicit
    exclusion, not be dropped silently by the tier ladder -- otherwise row
    conservation is satisfied by two errors cancelling."""
    src = _tiered_source()
    src = pd.concat([src, pd.DataFrame([
        _row(np.nan, "ENSG00000000009", "ENST9", 1.0, 2.0, 0.5, 0.8)])],
        ignore_index=True)
    frame, _ = select_constraint_transcripts(src, allow_canonical_fallback=False)
    assert int(frame["gene"].isna().sum()) == 1


# ---- AUDITCOUNT-1: source facts must not depend on project policy --------
def test_source_facts_do_not_depend_on_selection_policy():
    """THE INVARIANT. Measured 2026-08-10: `no_declared_transcript` reported 1
    with the canonical fallback enabled and 2 without it, for the same source.

    A count named for a property of gnomAD moved when a project switch moved --
    and that switch exists for MANE-only threshold calibration, which is
    exactly the configuration in which the count would be read.
    """
    src = _tiered_source()
    _, a = canonicalize_mane_constraint(src, allow_canonical_fallback=True)
    _, b = canonicalize_mane_constraint(src, allow_canonical_fallback=False)

    assert a.source_facts == b.source_facts, (
        "a fact about the SOURCE changed when project POLICY changed: "
        "{} vs {}".format(a.source_facts, b.source_facts))
    assert a.selection != b.selection, (
        "the selection audit did NOT change when policy did; the split is "
        "recording the same thing twice rather than two different things")


def test_source_facts_are_computed_without_any_policy_argument():
    """describe_transcript_source takes no policy parameter, deliberately.
    A function that cannot see a policy cannot be contaminated by one."""
    import inspect
    params = list(inspect.signature(describe_transcript_source).parameters)
    assert params == ["raw"], (
        "describe_transcript_source accepts {}; any parameter beyond the source "
        "frame is a route for policy to reach a fact".format(params))


def test_the_facts_reconcile_against_the_source():
    """3 named genes: HASMANE (mane+canonical), CANONONLY (canonical only),
    NEITHER (neither flag)."""
    f = describe_transcript_source(_tiered_source())
    assert f.n_gene_symbols == 3
    assert f.n_with_mane_select == 1
    assert f.n_with_canonical == 2          # HASMANE and CANONONLY
    assert f.n_without_mane_select == 2     # CANONONLY and NEITHER
    assert f.n_without_declared_transcript == 1   # NEITHER only, both policies


def test_the_selection_audit_records_the_policy_it_ran_under():
    src = _tiered_source()
    _, a = canonicalize_mane_constraint(src, allow_canonical_fallback=True)
    _, b = canonicalize_mane_constraint(src, allow_canonical_fallback=False)
    assert a.selection.allow_canonical_fallback is True
    assert b.selection.allow_canonical_fallback is False
    assert a.selection.n_selected_canonical == 1
    assert b.selection.n_selected_canonical == 0


# ---- STRICTNUMERIC-1 ------------------------------------------------------
def test_missing_and_malformed_are_not_the_same_value():
    """`pd.to_numeric(errors="coerce")` maps "", None and "not_a_number" all to
    NaN, so an absent measurement and a corrupted field reach the model
    identically. That is the conflation this whole line of work removes."""
    ok = to_numeric_strict(pd.Series(["1.5", None, "", "2.5"]), column="x")
    assert ok.tolist()[0] == 1.5
    assert pd.isna(ok.tolist()[1]) and pd.isna(ok.tolist()[2])

    try:
        to_numeric_strict(pd.Series(["1.5", "not_a_number"]), column="loeuf")
    except ConstraintSourceError as exc:
        assert "PRESENT but do not parse" in str(exc)
        assert "loeuf" in str(exc)
        return
    raise AssertionError("a malformed value was silently coerced to NaN")


def test_already_missing_values_survive_strict_coercion():
    out = to_numeric_strict(pd.Series([np.nan, 1.0, None]), column="x")
    assert int(out.isna().sum()) == 2 and out.tolist()[1] == 1.0


# ---- PROVENANCE-ASSERT-1 --------------------------------------------------
def test_an_unverifiable_digest_is_recorded_as_UNVERIFIED():
    """A caller-supplied digest is a CLAIM. With no readable file behind it,
    the record must say so rather than presenting it as provenance."""
    _, audit = canonicalize_mane_constraint(
        measured_source(), source_path="", source_sha256="0" * 64)
    assert audit.source_sha256_verified is False


def test_a_wrong_caller_digest_RAISES_when_the_file_is_readable(tmp_path=None):
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "src.tsv")
        measured_source().to_csv(p, sep="\t", index=False)
        try:
            canonicalize_mane_constraint(
                measured_source(), source_path=p, source_sha256="0" * 64)
        except ConstraintSourceError as exc:
            assert "provenance mismatch" in str(exc)
            return
    raise AssertionError("a false provenance claim was accepted unverified")


def test_an_UNREADABLE_source_cannot_be_reported_as_verified():
    """The case `_verified = actual is not None` exists for.

    A path and a digest are supplied, but the file cannot be read -- so nothing
    was recomputed and nothing was checked. Reporting that as verified would be
    a provenance claim resting on an OSError. Sabotage forcing
    `_verified = True` went undetected until this test existed: the suite had a
    case for "no path" and one for "correct digest", and none for "the check
    could not run".
    """
    import os
    absent = os.path.join(os.sep, "no", "such", "directory", "gnomad.tsv")
    _, audit = canonicalize_mane_constraint(
        measured_source(), source_path=absent, source_sha256="a" * 64)
    assert audit.source_sha256_verified is False, (
        "an unreadable source was reported as having a VERIFIED digest")


def test_a_correct_caller_digest_is_VERIFIED():
    import tempfile, os
    from genomic_variant_classifier.data.constraint_canonicalize import sha256_file
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "src.tsv")
        measured_source().to_csv(p, sep="\t", index=False)
        _, audit = canonicalize_mane_constraint(
            measured_source(), source_path=p, source_sha256=sha256_file(p))
        assert audit.source_sha256_verified is True


# --------------------------------------------------------------------------
def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  {}".format(name))
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed, {} total".format(
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
