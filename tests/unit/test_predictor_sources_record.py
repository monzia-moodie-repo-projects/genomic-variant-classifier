"""Stage 1D-1c.2A (set "aw"): docs/census/PREDICTOR_SOURCES_1D1c2.json, measured on the real files by
scripts/measure_predictor_sources_1d1c2.py (owner ruling e755519d). The record is RE-MEASURED from the data on the owner's
machine by the installer's predictor_sources stage; here it is checked for its own arithmetic, its bindings to the code
that measured it (semantic-closure digests re-measured now), to the pilot record it corrects and to the census statements
that quote it. Author: Monzia Moodie."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "census" / "PREDICTOR_SOURCES_1D1c2.json"


def _record() -> dict:
    from genomic_variant_classifier.information_sources import census as C

    return C.strict_load(RECORD)


def test_record_is_canonical_and_bound_to_the_code_that_measured_it():
    from genomic_variant_classifier.information_sources import census as C

    raw = RECORD.read_bytes()
    rec = _record()
    assert raw == (json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    assert rec["schema"] == "gvc.predictor-sources.1d1c2/1" and rec["mode"].startswith("shadow:")
    assert sorted(rec["parts"]) == ["index-defaults", "mane", "pilot", "revel"]
    for sym, digest in rec["procedure"].items():
        assert C.symbol_closure_digest(REPO, sym) == digest, f"{sym} changed since the record was measured; re-measure"
    import importlib.util

    spec = importlib.util.spec_from_file_location("m", REPO / "scripts" / "measure_predictor_sources_1d1c2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert sorted(rec["procedure"]) == sorted(m.PROCEDURE)


def test_record_binds_each_part_by_the_digest_of_its_canonical_bytes():
    """Each part the record names is reproduced from the record's own copy of it: the part's canonical bytes (LF line ends
    on every platform) have exactly the SHA-256 the record binds."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("m", REPO / "scripts" / "measure_predictor_sources_1d1c2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rec = _record()
    for part, bound in rec["parts"].items():
        body = {"part": part, **rec[part.replace("-", "_")]}
        assert hashlib.sha256(m._canonical(body)).hexdigest() == bound["sha256"], part
        assert bound["file"] == part.replace("-", "_") + ".json"


def test_record_inputs_are_the_pilots_and_the_declared_sources():
    from genomic_variant_classifier.information_sources import census as C

    rec = _record()
    pilot = C.strict_load(REPO / "docs" / "census" / "PILOTS_1D1b2.json")
    for part in ("pilot", "mane", "revel"):
        assert rec[part]["inputs"]["cohort"] == {k: pilot["inputs"]["cohort"][k] for k in ("bytes", "file", "sha256")}
        assert rec[part]["inputs"]["dbnsfp_clinvar_index"]["sha256"] == pilot["inputs"]["dbnsfp"]["sha256"]
    assert rec["pilot"]["population"]["sha256_v1"] == pilot["population"]["sha256"]
    assert rec["mane"]["population_sha256_v1"] == rec["revel"]["population_sha256_v1"] == pilot["population"]["sha256"]
    frame = rec["pilot"]["frame"]["sha256"]
    assert rec["mane"]["inputs"]["pilot_frame"]["sha256"] == rec["revel"]["inputs"]["pilot_frame"]["sha256"] == frame
    rv = rec["revel"]["inputs"]
    assert rv["revel_archive"] == {"bytes": 667102707, "file": "revel-v1.3_all_chromosomes.zip",
                                   "md5": "3ea2bc33e6b5455fc7e9899da863b5fe", "member": "revel_with_transcript_ids",
                                   "sha256": "815801d832e4ab76922dddfd901d7f24a755df57078a7138aa14f615da86a8a2",
                                   "zenodo_record": "10.5281/zenodo.7072866"}
    assert rv["gencode_v50_gtf"]["sha256"] == rec["mane"]["inputs"]["gencode_v50_gtf"]["sha256"] == \
        "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453"
    assert rv["clinvar_variant_summary"]["sha256"] == "5fe4fe10783391d01dc414dc5583a3e63487b67f8cd3c8429d59227cd5f4f37f"


def test_record_arithmetic_reconciles():
    rec = _record()
    n = rec["pilot"]["population"]["rows"]
    assert n == 1490014
    rv = rec["revel"]
    assert sum(rv["historical_index_against_publisher"].values()) == n
    assert sum(rv["selection_mane_select"]["states"].values()) == n
    assert sum(rv["selection_clinvar_declared_substitution"]["states"].values()) == n
    assert sum(rv["selection_agreement"].values()) == n
    for sel in ("selection_mane_select", "selection_clinvar_declared_substitution"):
        by = rv[sel]["by_label_reported_after_selection"]
        assert sorted(by) == sorted(rv[sel]["states"])
        assert all(by[s]["pathogenic"] + by[s]["benign"] == c for s, c in rv[sel]["states"].items())
    sc = rv["source_census"]
    assert sc["rows"] == sum(rv["rows_by_chromosome"].values()) == sum(rv["transcripts_per_row"].values())
    assert sc["rows"] == sum(rv["score_decimal_places"].values()) + sc["score_missing"]
    ab = rv["absence_as_evidence_correction"]
    h = rv["historical_index_against_publisher"]
    # owner ruling ace92e09, section 3: three quantities, each from the partition, never interchangeable
    unsupported = ab["default_triggered_without_index_score_or_without_a_publisher_0_5_at_the_allele"]
    assert unsupported == ab["pilot_unobserved"] + ab["pilot_observed_at_0_5"] - ab["of_which_publisher_has_0_5_at_allele"]
    assert ab["default_triggered_positives"] == ab["pilot_unobserved"] + ab["pilot_observed_at_0_5"]
    assert ab["of_which_no_publisher_score_in_the_grch38_view"] == \
        h["hist_absent_pub_absent"] + h["hist_0_5_pub_absent__fabricated"]
    assert ab["of_which_a_publisher_score_exists_at_the_allele"] == \
        h["hist_absent_pub_present"] + h["hist_0_5_pub_other_values"]
    assert ab["of_which_no_publisher_score_in_the_grch38_view"] + ab["of_which_a_publisher_score_exists_at_the_allele"] \
        == unsupported != ab["of_which_no_publisher_score_in_the_grch38_view"]
    assert ab["nondefault_index_value_without_publisher_score_positivity_not_recorded"] == h["hist_value_pub_absent"]
    dom = rv["historical_index_against_publisher_by_domain"]
    assert sorted(dom) == sorted(h) and all(sum(dom[k].values()) == h[k] for k in h)
    assert sum(ab["no_publisher_score_in_the_grch38_view_by_domain"].values()) == \
        ab["of_which_no_publisher_score_in_the_grch38_view"]
    rescued = ab["no_publisher_score_in_the_grch38_view_but_a_publisher_row_at_the_grch37_allele_with_unresolved_grch38"]
    assert all(rescued[k][d] <= dom[k][d] for k in rescued for d in rescued[k])
    assert ab["of_which_the_grch37_row_has_a_score"] <= sum(sum(v.values()) for v in rescued.values())
    # the policy disagreements are different protein contexts, counted in the agreement table
    pd_ = rv["policy_disagreements"]
    assert pd_["rows"] == rv["selection_agreement"]["policy_disagreement_different_score"]
    assert pd_["amino_acid_change_differs"] <= pd_["rows"] and pd_["crosses_0_5"] <= pd_["rows"]
    for sel in ("selection_mane_select", "selection_clinvar_declared_substitution"):
        assert "CANDIDATE" in rv[sel]["policy"] and "observed" not in rv[sel]["states"]
    mane = rec["mane"]["pilot_missense_snv_rows_by_locus_mane"]
    assert sum(mane.values()) == rec["mane"]["pilot_missense_rows"] - rv["selection_mane_select"]["states"]["not_applicable_not_snv"]
    assert h["hist_0_5_pub_has_0_5__genuine"] == ab["of_which_publisher_has_0_5_at_allele"]
    assert h["hist_0_5_pub_absent__fabricated"] + h["hist_0_5_pub_has_0_5__genuine"] + h["hist_0_5_pub_other_values"] \
        == ab["pilot_observed_at_0_5"] == rec["pilot"]["per_predictor"]["revel_score"]["exactly_default"]
    assert ab["pilot_unobserved"] == h["hist_absent_pub_absent"] + h["hist_absent_pub_present"]


def test_record_agrees_with_the_pilot_record_it_corrects():
    from genomic_variant_classifier.information_sources import census as C

    rec = _record()
    pilot = C.strict_load(REPO / "docs" / "census" / "PILOTS_1D1b2.json")
    assert rec["pilot"]["revel"]["pilot_recorded_observed"] == pilot["revel"]["observed"]
    assert rec["pilot"]["revel"]["join_unobserved"] == pilot["revel"]["unobserved"] == \
        rec["revel"]["absence_as_evidence_correction"]["pilot_unobserved"]
    obs = rec["pilot"]["revel"]
    assert obs["observed_not_at_default_ge_0_5"] + rec["pilot"]["per_predictor"]["revel_score"]["exactly_default"] == \
        pilot["revel"]["observed_revel_pathogenic"]


def test_census_statements_quote_the_record():
    from genomic_variant_classifier.information_sources import census as C

    rec = _record()
    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    by_id = {d["id"]: d for d in ledger["registered_defects"]}
    ab = rec["revel"]["absence_as_evidence_correction"]
    st = by_id["ABSENCE-AS-EVIDENCE-1"]["statement"]
    dom = ab["no_publisher_score_in_the_grch38_view_by_domain"]
    rescued = ab["no_publisher_score_in_the_grch38_view_but_a_publisher_row_at_the_grch37_allele_with_unresolved_grch38"]
    for n in (ab["default_triggered_without_index_score_or_without_a_publisher_0_5_at_the_allele"],
              ab["default_triggered_positives"], ab["pilot_observed_at_0_5"],
              ab["of_which_no_publisher_score_in_the_grch38_view"], ab["of_which_a_publisher_score_exists_at_the_allele"],
              dom["not_single_nucleotide"], dom["single_nucleotide_other_consequence"], dom["single_nucleotide_missense"],
              dom["not_single_nucleotide"] + dom["single_nucleotide_other_consequence"],
              sum(sum(rescued[k].values()) for k in ("hist_absent_pub_absent", "hist_0_5_pub_absent__fabricated"))):
        assert f"{n:,}" in st, n
    assert f"at {ab['of_which_publisher_has_0_5_at_allele']};" in st and "NOT the number without a publisher score" in st
    st = by_id["DBNSFP-INDEX-DEFAULT-BAKED-1"]["statement"]
    for c in ("revel_score", "sift_score", "polyphen2_score", "cadd_phred"):
        assert f"{rec['index_defaults']['per_predictor'][c]['exactly_default']:,}" in st
    st = by_id["COHORT-TRANSCRIPT-UNDECLARED-1"]["statement"]
    cv = rec["revel"]["selection_clinvar_declared_substitution"]["states"]
    mn = rec["revel"]["selection_mane_select"]["states"]
    assert f"{cv['candidate']:,}" in st and f"{mn['candidate']:,}" in st and "CANDIDATE" in st
    locus = rec["mane"]["pilot_missense_snv_rows_by_locus_mane"]
    assert f"at {locus['one_mane_coding_sequence_of_another_gene']:,} single-nucleotide" in st
    several = locus["several_mane_coding_sequences_cohort_gene_among_them"] + locus["several_mane_coding_sequences_cohort_gene_not_among_them"]
    assert f"({several} are covered by several)" in st
    pd_ = rec["revel"]["policy_disagreements"]
    assert f"the other {pd_['rows']} are policy disagreements" in st and f"in all {pd_['amino_acid_change_differs']}" in st
    assert pd_["amino_acid_change_differs"] == pd_["rows"] and f"({pd_['crosses_0_5']} cross 0.5)" in st
    st = by_id["DBNSFP-INDEX-DEFAULT-BAKED-1"]["statement"]
    within = rec["pilot"]["per_predictor"]["revel_score"]["exactly_default_auroc_within_consequence_among_joined_rows"]
    for k, label in (("missense_variant", "missense"), ("nonsense", "nonsense"), ("intron_variant", "intron")):
        assert f"{label} {within[k]:.3f}" in st, k
    assert f"{rec['pilot']['per_predictor']['revel_score']['exactly_default_auroc_among_joined_rows']:.3f} over the joined" in st
    assert "non-commercial, confirmed by the owner" in ledger["datasets"]["revel_v1_3"]["access"]
    assert rec["mane"]["cohort_transcript_id_null_rows"] == rec["mane"]["cohort_protein_change_null_rows"] == 1490014
    for ds in ("revel_v1_3", "gencode_v50", "clinvar_variant_summary"):
        assert "record:docs/census/PREDICTOR_SOURCES_1D1c2.json" in ledger["datasets"][ds]["evidence_refs"]
    assert rec["revel"]["inputs"]["revel_archive"]["sha256"] in ledger["datasets"]["revel_v1_3"]["description"]
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest()       # the record is read whole
