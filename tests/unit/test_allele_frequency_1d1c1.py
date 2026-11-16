"""Stage 1D-1c.1B (set "at"): the allele-frequency materialiser. Owner rulings 482cd291, 4bef3910 and b937681e
(2026-11-14). Synthetic witnesses; the real-data reconciliation is docs/census/AF_REPAIR_1D1c1.json. Author: Monzia Moodie.

Where a legacy behaviour exists, the matching test also shows the legacy code doing the wrong thing (the pilot
materialization.gnomad_source_table and DataPrepPipeline._join_gnomad's parse), so the witness fails against the defect.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import allele_frequency as A
from genomic_variant_classifier.information_sources import columns as K
from genomic_variant_classifier.information_sources import materialization as M
from genomic_variant_classifier.information_sources import observation as O

S = O.ObservationState
REGISTERED = ("gnomad_af", "gnomad_ymt_api")


def _source_bytes(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows, columns=["variant_id", "allele_freq"]).to_parquet(buf, index=False)
    return buf.getvalue()


def _parsed(rows):
    raw = _source_bytes(rows)
    import hashlib
    return M.Parsed(raw, hashlib.sha256(raw).hexdigest(), len(raw))


SOURCE_ROWS = [
    ("1:100:A:G", 0.0),                 # a reported zero (filter unknown)
    ("1:200:C:T", 0.25),
    ("2:300:G:A", None),                # a record without a value
    ("X:2781479:T:C", 1e-5),            # X, last base of PAR1
    ("X:5000000:G:C", 0.5),             # X non-PAR
    ("gnomad:Y:3000000:A:T", 0.125),    # Y, API (exome else genome)
    ("gnomad:MT:100:T:C", 0.0),         # MT, API homoplasmic fraction 0 (ambiguous zero)
    ("gnomad:MT:200:A:G", 0.375),
    ("9:999:A:C", 0.7),                 # a source record no cohort row requests
]
COHORT = [
    # variant_id, chrom, pos, ref, alt, label, reference_verified
    ("clinvar:1:100:A:G", "1", 100, "A", "G", 1, True),
    ("clinvar:1:200:C:T", "1", 200, "C", "T", 0, True),
    ("clinvar:2:300:G:A", "2", 300, "G", "A", 0, True),
    ("clinvar:3:400:C:G", "3", 400, "C", "G", 1, True),          # no row in the extract
    ("clinvar:X:2781479:T:C", "X", 2781479, "T", "C", 0, True),
    ("clinvar:X:5000000:G:C", "X", 5000000, "G", "C", 0, True),
    ("clinvar:4:500:A:ANNNA", "4", 500, "A", "ANNNA", 1, False),  # unknown bases in the inserted sequence
    ("clinvar:Y:3000000:A:T", "Y", 3000000, "A", "T", 0, True),
    ("clinvar:MT:100:T:C", "MT", 100, "T", "C", 1, True),
    ("clinvar:MT:300:G:A", "MT", 300, "G", "A", 0, True),        # MT, no API row
]


def cohort(rows=COHORT):
    return pd.DataFrame([r[:6] for r in rows], columns=["variant_id", "chrom", "pos", "ref", "alt", "label"])


def refs(rows=COHORT):
    return np.array([r[6] for r in rows], dtype=bool)


def run(rows=SOURCE_ROWS, frame=None):
    f = cohort() if frame is None else frame
    pop = M.population_from_frame(f, "pop:test")
    return A.materialize_allele_frequency(f, pop, _parsed(rows), registered_datasets=REGISTERED), pop, f


POLICY = A.AFSelectionPolicy(unknown_base_treatment=A.UnknownBaseTreatment.EXCLUDE_FROM_SEQUENCE_RESOLVED_PRIMARY)


# --------------------------------------------------------------------------------------------- the source grammar
def test_the_identifier_grammar_is_strict_and_the_legacy_parse_is_not():
    for bad in ("chr1:100:A:G", "1:0100:A:G", "1:0:A:G", "1:100:A", "1:100:A:G:T", "gnomad:X:100:A:G",
                "gnomad:1:100:A:G", "Y:100:A:G", "MT:100:A:G", "1:100:A:A", "1:100:a:G", "GRCh38:1:100:A:G",
                "1:100:A:<INS>", "1:-5:A:G", "23:100:A:G"):
        with pytest.raises(ValueError, match="identifier"):
            run([("1:200:C:T", 0.25), (bad, 0.5)])
    # The legacy parse (DataPrepPipeline._join_gnomad's, unchanged in the pilot) drops every X record and accepts an
    # unknown prefix: GNOMAD-CHRX-LOCUS-PARSE-1.
    assert M._parse_gnomad_locus("X:5000000:G:C") is None
    assert M._parse_gnomad_locus("GRCh38:1:100:A:G") == ("1", "100", "A", "G")
    r, _, _ = run([("X:5000000:G:C", 0.5), ("1:200:C:T", 0.25)])
    assert r.primary.values[5] == 0.5                      # the strict grammar keeps the X record


def test_a_malformed_position_is_refused_never_coerced_to_zero():
    raw = _source_bytes([("1:abc:A:G", 0.5), ("1:200:C:T", 0.25)])
    legacy = M.gnomad_source_table(raw)                  # the pilot's table: the bad position becomes 0
    assert (legacy["_pos"] == 0).sum() == 1
    with pytest.raises(ValueError, match="identifier"):
        run([("1:abc:A:G", 0.5), ("1:200:C:T", 0.25)])


@pytest.mark.parametrize("bad", [-1e-9, 1.0000001, float("inf"), float("-inf")])
def test_a_frequency_outside_the_unit_interval_is_refused_never_clipped(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run([("1:100:A:G", bad), ("1:200:C:T", 0.25)])


def test_a_non_numeric_or_boolean_frequency_column_is_refused():
    for col in (["0.5", "0.25"], [True, False]):
        buf = io.BytesIO()
        pd.DataFrame({"variant_id": ["1:100:A:G", "1:200:C:T"], "allele_freq": col}).to_parquet(buf, index=False)
        raw = buf.getvalue()
        import hashlib
        with pytest.raises(ValueError, match="floating-point"):
            A.materialize_allele_frequency(cohort(), M.population_from_frame(cohort(), "p"),
                                           M.Parsed(raw, hashlib.sha256(raw).hexdigest(), len(raw)),
                                           registered_datasets=REGISTERED)


def test_a_duplicated_source_key_is_refused_never_first_wins():
    for dup in ([("1:100:A:G", 0.1), ("1:100:A:G", 0.1)], [("1:100:A:G", 0.1), ("1:100:A:G", 0.2)],
                [("gnomad:MT:100:T:C", 0.0), ("gnomad:MT:100:T:C", 0.0)]):
        with pytest.raises(ValueError, match="more than once"):
            run(dup + [("1:200:C:T", 0.25)])


def test_both_logical_sources_must_be_registered_in_the_census():
    f = cohort()
    with pytest.raises(ValueError, match="registered"):
        A.materialize_allele_frequency(f, M.population_from_frame(f, "p"), _parsed(SOURCE_ROWS),
                                       registered_datasets=("gnomad_af",))


# --------------------------------------------------------------------------------------------- source observations
def test_source_states_are_source_facts_with_details():
    r, pop, _ = run()
    p = r.primary
    st = [K.STATE_CODES[c] for c in p.states]
    det = [p.detail_vocabulary[d] for d in p.details]
    assert st[:6] == [S.OBSERVED, S.OBSERVED, S.NOT_IN_SOURCE, S.NOT_IN_SOURCE, S.OBSERVED, S.OBSERVED]
    assert det[:6] == ["reported_zero_filter_unknown", "", "record_without_value", "no_row_in_extracted_table", "", ""]
    assert st[6] is S.UNSUPPORTED_VARIANT and det[6] == "allele_sequence_incomplete"
    assert st[7:] == [S.UNSUPPORTED_VARIANT] * 3 and set(det[7:]) == {"chromosome_outside_extract_scope"}
    assert p.values[0] == 0.0 and p.values[1] == 0.25 and np.isnan(p.values[2:4]).all()
    assert p.source.logical_id == "dataset:gnomad_af" and p.source.content_sha256 == r.receipt["content_sha256"]
    assert p.source.release is None                       # the release is not recorded by the extract
    assert p.feature == "af_raw" and "not_in_gnomad" not in p.detail_vocabulary


def test_y_and_mt_keep_their_own_statistics_in_their_own_columns():
    r, _, _ = run()
    y, mt = r.y_api, r.mt_api
    assert y.source.logical_id == mt.source.logical_id == "dataset:gnomad_ymt_api"
    assert r.statistics["af_raw"] == "gnomad_preprocessed_mixed_af_unfiltered_v1"
    assert r.statistics[y.feature] == "gnomad_api_y_exome_else_genome_af_v1"
    assert r.statistics[mt.feature] == "gnomad_api_mt_homoplasmic_fraction_v1"
    assert y.values[7] == 0.125 and np.isnan(y.values[:7]).all() and np.isnan(y.values[8:]).all()
    assert mt.values[8] == 0.0 and mt.detail_vocabulary[mt.details[8]] == "zero_ac_hom_or_missing_ac_hom"
    assert K.STATE_CODES[mt.states[9]] is S.NOT_IN_SOURCE and mt.detail_vocabulary[mt.details[9]] == "no_row_in_api_extract"
    assert K.STATE_CODES[y.states[0]] is S.UNSUPPORTED_VARIANT


def test_every_row_is_accounted_for_in_every_column():
    r, pop, _ = run()
    for col in (r.primary, r.y_api, r.mt_api):
        assert sum(col.counts().values()) == len(pop) == len(col.values)
    assert r.receipt["source_records"] == {"dataset:gnomad_af": 6, "dataset:gnomad_ymt_api": 3}
    assert r.receipt["source_records_requested_by_population"] == {"dataset:gnomad_af": 5, "dataset:gnomad_ymt_api": 2}


def test_the_cohort_identity_must_equal_its_coordinate_columns():
    f = cohort()
    f.loc[1, "pos"] = 201
    with pytest.raises(ValueError, match="variant_id"):
        A.materialize_allele_frequency(f, M.population_from_frame(f, "p"), _parsed(SOURCE_ROWS),
                                       registered_datasets=REGISTERED)


def test_an_unadjudicated_allele_form_is_refused():
    rows = list(COHORT)
    rows[3] = ("clinvar:3:400:C:<DEL>", "3", 400, "C", "<DEL>", 1, False)        # symbolic: contains no N base
    with pytest.raises(ValueError, match="allele"):
        run(frame=cohort(rows))


def test_materialisation_never_reads_labels():
    r1, _, f = run()
    g = f.copy()
    g["label"] = 1 - g["label"]
    r2, _, _ = run(frame=g)
    h = f.drop(columns=["label"])
    r3, _, _ = run(frame=h)
    for a, b in ((r1, r2), (r1, r3)):
        for c1, c2 in zip(a.columns(), b.columns()):
            assert c1.sha256 == c2.sha256
        assert a.receipt == b.receipt


# --------------------------------------------------------------------------------------------- selection and claim
def test_the_selection_policy_must_declare_the_unknown_base_treatment():
    with pytest.raises(ValueError, match="declare"):
        A.AFSelectionPolicy()
    with pytest.raises(ValueError, match="statistic"):
        A.AFSelectionPolicy(unknown_base_treatment=A.UnknownBaseTreatment.EXCLUDE_FROM_SEQUENCE_RESOLVED_PRIMARY,
                            statistic="nuclear_exome_af")


def test_selection_claim_and_value_are_separate_and_reconcile():
    r, pop, f = run()
    sel = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    assert [sel.claim_vocabulary[c] for c in sel.claim] == ["eligible"] * 6 + ["incomplete_allele"] + ["outside_domain"] * 3
    assert [sel.selection_vocabulary[c] for c in sel.selection] == [
        "selected", "selected", "no_qualified_observation", "no_qualified_observation", "selected", "selected",
        "allele_sequence_incomplete", "policy_out_of_scope", "policy_out_of_scope", "policy_out_of_scope"]
    assert sel.af_model[0] == 0.0 and sel.af_model[4] == 1e-5 and np.isnan(sel.af_model[[2, 3, 6, 7, 8, 9]]).all()
    c = sel.census()
    assert c["full"] == 10 and c["primary_domain"] == 7 and c["sequence_resolved_primary"] == 6
    assert c["selected"] == 4 and c["eligible_unselected"] == 2 and c["unknown_base"] == 1 and c["outside_domain"] == 3
    assert c["selected"] + c["eligible_unselected"] + c["unknown_base"] + c["outside_domain"] == c["full"]
    assert c["selected_reported_zero"] == 1


def test_y_and_mt_values_are_never_selected_even_when_observed():
    r, _, _ = run()
    sel = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    assert K.STATE_CODES[r.y_api.states[7]] is S.OBSERVED and np.isnan(sel.af_model[7])
    assert K.STATE_CODES[r.mt_api.states[8]] is S.OBSERVED and np.isnan(sel.af_model[8])


def test_eligibility_does_not_depend_on_frequency_availability():
    r, _, _ = run()
    a = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    r2, _, _ = run(rows=[("9:999:A:C", 0.7)])                       # no cohort row has a value now
    b = A.select_strict(r2, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    assert np.array_equal(a.claim, b.claim)


def test_reference_evidence_must_agree_with_the_allele_form():
    r, _, _ = run()
    bad = refs().copy()
    bad[1] = False                                                     # a fully specified allele not verified
    with pytest.raises(ValueError, match="reference"):
        A.select_strict(r, POLICY, reference_verified=bad, reference_evidence_sha256="e" * 64)
    bad = refs().copy()
    bad[6] = True                                                      # an N allele cannot be reference-verified
    with pytest.raises(ValueError, match="reference"):
        A.select_strict(r, POLICY, reference_verified=bad, reference_evidence_sha256="e" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="not-a-digest")


def test_sensitivity_arms_change_selection_never_the_source():
    r, _, _ = run()
    base = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    zero = A.select_strict(r, A.AFSelectionPolicy(unknown_base_treatment=POLICY.unknown_base_treatment,
                                                  arm=A.Arm.WITHHOLD_REPORTED_ZEROS),
                           reference_verified=refs(), reference_evidence_sha256="e" * 64)
    none = A.select_strict(r, A.AFSelectionPolicy(unknown_base_treatment=POLICY.unknown_base_treatment,
                                                  arm=A.Arm.WITHOUT_AF),
                           reference_verified=refs(), reference_evidence_sha256="e" * 64)
    assert zero.selection_vocabulary[zero.selection[0]] == "withheld_zero_sensitivity" and np.isnan(zero.af_model[0])
    assert zero.af_model[1] == 0.25
    assert np.isnan(none.af_model).all() and none.selection_vocabulary[none.selection[1]] == "withheld_af_ablation"
    assert K.STATE_CODES[r.primary.states[0]] is S.OBSERVED and r.primary.values[0] == 0.0      # the source is unchanged
    assert len({base.identity, zero.identity, none.identity}) == 3
    assert np.array_equal(base.claim, zero.claim) and np.array_equal(base.claim, none.claim)


def test_audit_details_cannot_reach_the_model_view():
    """Two source columns differing ONLY in an audit detail (record_without_value against no_row_in_extracted_table,
    both NOT_IN_SOURCE and NaN) give the same model values and the same selection codes."""
    r, pop, _ = run()
    p = r.primary
    voc = p.detail_vocabulary
    swapped = p.details.copy()
    swapped[2] = voc.index("no_row_in_extracted_table")
    p2 = K.ObservationColumn(p.feature, p.source, p.population, p.values.copy(), p.states.copy(), swapped, voc)
    r2 = A.AlleleFrequencyMaterialization(p2, r.y_api, r.mt_api, r.chromosome, r.allele_form, r.statistics, r.receipt)
    a = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    b = A.select_strict(r2, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    assert p2.sha256 != p.sha256
    assert np.array_equal(a.af_model, b.af_model, equal_nan=True) and np.array_equal(a.selection, b.selection)


def test_a_policy_or_evidence_change_changes_the_selection_identity():
    r, _, _ = run()
    a = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="e" * 64)
    b = A.select_strict(r, POLICY, reference_verified=refs(), reference_evidence_sha256="f" * 64)
    assert a.identity != b.identity
    for arr in (a.af_model, a.claim, a.selection):
        assert not arr.flags.writeable


def test_the_statistic_declaration_forbids_stronger_claims():
    d = A.STATISTICS["gnomad_preprocessed_mixed_af_unfiltered_v1"]
    assert d["primary_field"] == "AF" and d["fallback_field"] == "AF_genome"
    assert d["fallback_events_recorded"] is False and d["filter_policy"] == {"applied": False,
                                                                           "source_filter_values_available": False}
    assert d["claims"] == {"exome_only": False, "filter_qualified": False, "complete_gnomad_catalogue": False}
    assert A.STATISTICS["gnomad_api_mt_homoplasmic_fraction_v1"]["quantity"] == "ac_hom / an"
    for sid in A.STATISTICS:
        assert sid not in A.SELECTABLE_STATISTICS or sid == "gnomad_preprocessed_mixed_af_unfiltered_v1"


def test_the_rare_flag_audit_reports_beyond_auroc():
    s = A.audit_binary_flag(n_positive=210_549, n_negative=1_279_465, flagged_positive=478, flagged_negative=22)
    assert math.isclose(s["auroc"], 0.5 + 0.5 * (478 / 210_549 - 22 / 1_279_465))
    assert 0.50112 < s["auroc"] < 0.50113 and 131 < s["lr_plus"] < 133
    lo, hi = s["prevalence_95ci"]
    assert lo < 478 / 500 < hi
    z = A.audit_binary_flag(n_positive=210_549, n_negative=1_279_465, flagged_positive=0, flagged_negative=1_354)
    assert z["prevalence"] == 0 and z["prevalence_95ci"][0] == 0 and z["lr_plus"] == 0
    with pytest.raises(ValueError):
        A.audit_binary_flag(n_positive=0, n_negative=1, flagged_positive=0, flagged_negative=0)
