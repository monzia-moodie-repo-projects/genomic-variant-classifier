"""Stage 1D-1b.2 (set "aq"): the real-data pilot record (docs/census/PILOTS_1D1b2.json) is internally consistent and
bound to the code that produced it; the measurement script's helpers are correct. The record itself was measured on the
real source files (scripts/measure_1d1b2_pilots.py); re-deriving it needs those files, which a clone does not hold.
Owner ruling 7f4b69d0 (2026-11-09), sections 5 and 19. Author: Monzia Moodie."""
from __future__ import annotations

import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import materialization as M

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs/census/PILOTS_1D1b2.json"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _script():
    spec = importlib.util.spec_from_file_location("measure_1d1b2_pilots", REPO / "scripts/measure_1d1b2_pilots.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def record():
    raw = RECORD.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    return json.loads(raw)


def test_every_receipt_reconciles_with_the_population(record):
    n = record["population"]["rows"]
    assert n == 1_490_014 and record["population"]["pathogenic"] + record["population"]["benign"] == n
    for name in ("eve", "allele_frequency"):
        r = record[name]["receipt"]
        assert r["n_population"] == n and sum(r["state_counts"].values()) == n, name
        assert sum(sum(d.values()) for d in r["detail_counts"].values()) == n, name
        assert r["population_sha256"] == record["population"]["sha256"], name
        assert HEX64.match(r["content_sha256"]) and HEX64.match(r["column_sha256"]), name
        assert r["state_counts"]["observed"] == record[name]["observed"], name


def test_the_record_is_bound_to_the_materialisers_as_they_are_now(record):
    for name, qual in (("eve", "materialize_eve"), ("allele_frequency", "materialize_gnomad_af")):
        symbol, digest = M.parser_identity(qual)
        assert record[name]["receipt"]["parser"] == symbol
        assert record[name]["receipt"]["parser_sha256"] == digest, (
            f"{qual} changed since the pilot was measured: re-run scripts/measure_1d1b2_pilots.py on the real files")


def test_every_input_is_identified_by_the_digest_of_the_bytes_read(record):
    for label, entry in record["inputs"].items():
        assert HEX64.match(entry["sha256"]) and entry["bytes"] > 0, label
        assert entry["logical_source"].startswith("dataset:"), label
    assert record["inputs"]["cohort"]["columns_not_read"] == ["metadata"]
    src = record["eve_source"]
    assert src["files"] == src["legacy_parse_equal_files"] == 3211        # every file parses as the connector parses it
    assert HEX64.match(src["directory_manifest_sha256"]) and src["directory_manifest_sha256"] == \
        record["eve"]["receipt"]["content_sha256"]


def test_the_allele_frequency_pilot_reproduces_the_current_join_exactly(record):
    af = record["allele_frequency"]
    assert af["legacy_equivalence"]["identical"] and af["legacy_equivalence"]["differ"] == 0
    assert af["observed"] + af["unobserved"] == record["population"]["rows"]
    # Every frequency the legacy join leaves empty is filled with 0.0 and so becomes "ultra-rare" and "absent".
    assert af["legacy_af_is_ultra_rare_set_by_absence"] == af["legacy_value_filled"]
    assert af["legacy_af_is_absent_set_by_absence"] == af["legacy_value_filled"]
    assert af["legacy_value_filled"] + af["unobserved_but_legacy_kept_a_first_record"] == af["unobserved"]
    assert af["legacy_af_is_absent"] >= af["legacy_value_filled"] + af["legacy_af_is_absent_observed_zero"]


def test_absence_arithmetic_for_eve_and_revel(record):
    eve, revel, n = record["eve"], record["revel"], record["population"]["rows"]
    assert eve["observed"] + eve["fabricated_0_5"] == n
    assert eve["legacy_coverage_count_score_ne_0_5"] == eve["observed"] - eve["observed_exactly_0_5"]
    assert revel["observed"] + revel["unobserved"] == n
    assert revel["unobserved_missense"] + revel["unobserved_non_missense"] == revel["unobserved"]
    assert revel["legacy_revel_pathogenic_set_by_absence"] == revel["unobserved"]
    assert revel["legacy_revel_pathogenic"] == revel["unobserved"] + revel["observed_revel_pathogenic"]
    assert record["is_snv"]["rows_with_absent_ref_or_alt"] == 0


def test_the_population_is_built_without_reading_the_unused_metadata_column(tmp_path):
    from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline

    frame = pd.DataFrame({
        "variant_id": [f"clinvar:1:{i}:A:G" for i in range(8)], "chrom": ["1"] * 8, "pos": list(range(8)),
        "ref": ["A"] * 8, "alt": ["G"] * 8, "gene_symbol": ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"],
        "clinical_sig": ["Pathogenic", "Benign", "Uncertain significance", "Likely benign", "Pathogenic",
                         "Conflicting classifications of pathogenicity", "Likely pathogenic", "Benign"],
        "ReviewStatus": ["criteria provided, single submitter"] * 4 + ["no assertion criteria provided"] * 2 +
                        ["reviewed by expert panel"] * 2,
        "metadata": [{"review_status": "x", "rs_id": i} for i in range(8)],
    })
    raw = io.BytesIO()
    frame.to_parquet(raw, index=False)
    pipe = DataPrepPipeline(DataPrepConfig(output_dir=tmp_path / "splits"))
    full = pipe._load_and_label(io.BytesIO(raw.getvalue()))
    lean, unread = _script().load_population(pipe, raw.getvalue())
    assert unread == ["metadata"] and "metadata" not in lean.columns
    assert full.drop(columns=["metadata"]).equals(lean)                 # the same rows, in the same order


def test_the_auroc_of_an_indicator():
    s = _script()
    assert s._auroc([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert s._auroc([0, 0, 1, 1], [1, 1, 0, 0]) == 0.0
    assert s._auroc([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5
    assert s._auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5                  # a constant indicator carries nothing
    assert s._auroc([1, 0], [1, 1]) is None
    by = s._by_label(np.array([True, False, True, True]), np.array([1, 1, 0, 0]))
    assert by["pathogenic"] == {"rows": 2, "count": 1, "fraction": 0.5} and by["benign"]["count"] == 2


def test_every_step_of_the_procedure_is_the_code_now(record):
    from genomic_variant_classifier.information_sources import census as C

    procedure = record["procedure"]
    assert sorted(procedure) == sorted(_script().PROCEDURE)
    for symbol, digest in procedure.items():
        assert C.symbol_closure_digest(REPO, symbol) == digest, (
            f"{symbol} changed since the pilot was measured: re-run scripts/measure_1d1b2_pilots.py on the real files")


def test_the_measured_defects_are_the_ones_the_census_registers(record):
    af, eve = record["allele_frequency"], record["eve"]
    assert af["by_chromosome_class"]["X"] == {"rows": 63_929, "observed": 0}            # GNOMAD-CHRX-LOCUS-PARSE-1
    assert af["by_chromosome_class"]["autosome"]["observed"] == 990_350                 # Run 15's logged count
    assert eve["legacy_equivalence"]["rows_matching_a_key_scored_more_than_once"] == 280
    assert eve["legacy_equivalence"]["on_rows_not_matching_a_duplicated_key"]["identical"]
    assert "1490294 rows for 1490014" in eve["legacy_equivalence"]["legacy_join_on_the_whole_population"]
    ids = {d["id"] for d in json.loads((REPO / "docs/census/FEATURE_CENSUS_1D1a.json").read_text())["registered_defects"]}
    assert {"ABSENCE-AS-EVIDENCE-1", "GNOMAD-CHRX-LOCUS-PARSE-1", "EVE-DUPLICATE-KEY-MULTIPLIES-ROWS-1"} <= ids
