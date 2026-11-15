"""Stage 1D-1c.1B (set "at"): the committed allele-frequency difference ledger docs/census/AF_REPAIR_1D1c1.json is internally
consistent, agrees with the sealed 1D-1b.2 pilot on the population and the source bytes, and is bound to the code as it is
now. Re-measured on the real files by the installer (scripts/measure_af_repair_1d1c1.py --check). Author: Monzia Moodie."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import allele_frequency as A
from genomic_variant_classifier.information_sources import census as C

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "census" / "AF_REPAIR_1D1c1.json"


def _script():
    spec = importlib.util.spec_from_file_location("measure_af_repair_1d1c1", REPO / "scripts" / "measure_af_repair_1d1c1.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_af_repair_1d1c1"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def record():
    return C.strict_load(RECORD)


@pytest.fixture(scope="module")
def pilot():
    return C.strict_load(REPO / "docs" / "census" / "PILOTS_1D1b2.json")


def test_the_record_is_in_canonical_form(record):
    assert RECORD.read_bytes() == C.ledger_bytes(record)
    assert record["schema"] == "gvc.af-repair-ledger.1d1c1/1"


def test_the_population_and_source_are_the_pilots(record, pilot):
    pop = record["population"]
    assert pop["rows"] == pilot["population"]["rows"] == 1_490_014
    assert pop["sha256_v1"] == pop["pilot_population_v1"] == pilot["population"]["sha256"]
    assert pop["pathogenic"] == pilot["population"]["pathogenic"] and pop["benign"] == pilot["population"]["benign"]
    assert record["inputs"]["gnomad"]["sha256"] == pilot["inputs"]["gnomad_af"]["sha256"]
    assert record["inputs"]["cohort"]["sha256"] == pilot["inputs"]["cohort"]["sha256"]
    assert record["receipt"]["population_sha256"] == pop["sha256_v2"] and record["receipt"]["n_population"] == pop["rows"]


def test_the_census_is_the_pre_registered_one_and_partitions_the_population(record):
    c = record["census"]
    assert c == record["expected"] == _script().EXPECTED
    assert c["selected"] + c["eligible_unselected"] + c["unknown_base"] + c["outside_domain"] == c["full"]
    assert c["sequence_resolved_primary"] == c["selected"] + c["eligible_unselected"]


def test_every_row_has_one_legacy_class_and_no_unexpected_difference(record):
    lc = record["legacy_comparison"]
    assert sum(lc.values()) == record["population"]["rows"]
    allowed = ("unchanged_value", "recovered_x", "withdrawn_policy_out_of_scope:", "still_unselected:")
    assert all(k.startswith(allowed) for k in lc), sorted(lc)
    assert lc["unchanged_value"] == 990_350 and lc["recovered_x"] == 37_874
    assert lc["withdrawn_policy_out_of_scope:MT"] == 1_674 and lc["withdrawn_policy_out_of_scope:Y"] == 20


def test_every_column_reconciles_and_the_sources_stay_typed(record):
    n = record["population"]["rows"]
    cols = record["receipt"]["columns"]
    assert set(cols) == {"af_raw", "af_y_api", "af_hom_mt_api"}
    for name, col in cols.items():
        assert sum(col["state_counts"].values()) == n, name
        assert sum(sum(d.values()) for d in col["detail_counts"].values()) == n, name
    assert cols["af_raw"]["logical_source"] == A.PRIMARY and cols["af_hom_mt_api"]["logical_source"] == A.YMT_API
    assert "not_in_gnomad" not in str(cols)
    assert record["statistics"] == {k: dict(v) for k, v in A.STATISTICS.items()}


def test_the_record_is_label_blind_and_the_arms_differ(record):
    assert record["label_blind"] is True
    ids = {v["identity"] for v in record["arms"].values()}
    assert len(ids) == 3 and record["arms"]["retain_reported_zeros"]["identity"] == record["selection_identity"]
    assert record["arms"]["without_af"]["census"]["selected"] == 0


def test_reference_evidence_covers_every_row(record):
    v = record["reference_evidence"]["verdicts"]
    assert sum(v.values()) == record["population"]["rows"] and v["False:non_acgt_allele"] == record["census"]["unknown_base"]


def test_the_rare_flags_are_consistent(record):
    for name, f in record["rare_flags"].items():
        assert f["n"] == f["pathogenic"] + f["benign"], name
        assert sum(s["n"] for s in f["by_split"].values()) == f["n"], name
        assert sum(s["pathogenic"] for s in f["by_split"].values()) == f["pathogenic"], name
    assert record["rare_flags"]["record_without_value"]["pathogenic"] == 0


def test_the_record_is_bound_to_the_code_as_it_is_now(record):
    for symbol, digest in record["procedure"].items():
        assert C.symbol_closure_digest(REPO, symbol) == digest, (
            f"{symbol} changed since the ledger was measured: re-run scripts/measure_af_repair_1d1c1.py on the real files")
    assert set(record["procedure"]) == set(_script().PROCEDURE)
