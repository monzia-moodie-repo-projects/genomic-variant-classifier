"""Set "av": the committed real-data record docs/census/QUALIFIED_PREPARATION_1D1c1C.json is internally consistent, agrees
with the allele-frequency ledger it was declared by (sources, population, census, selection and reference digests), and is
bound to the code as it is now. Re-measured on the real files by the installer
(scripts/measure_qualified_preparation_1d1c1c.py --check). Author: Monzia Moodie."""
from __future__ import annotations

from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import qualified_preparation as Q
from genomic_variant_classifier.information_sources import reference_status as R

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "docs" / "census" / "QUALIFIED_PREPARATION_1D1c1C.json"
AF_LEDGER = REPO / "docs" / "census" / "AF_REPAIR_1D1c1.json"


@pytest.fixture(scope="module")
def record():
    return C.strict_load(RECORD)


@pytest.fixture(scope="module")
def ledger():
    return C.strict_load(AF_LEDGER)


def test_the_record_is_in_canonical_form(record):
    assert RECORD.read_bytes() == C.ledger_bytes(record)
    assert record["schema"] == "gvc.qualified-preparation-record.1d1c1c/1"


def test_the_sources_and_population_are_the_af_ledgers(record, ledger):
    assert record["inputs"]["declared_by"]["sha256"] == C.sha256_bytes(AF_LEDGER.read_bytes())
    for key in ("cohort", "gnomad", "reference_evidence"):
        assert record["inputs"][key]["sha256"] == ledger["inputs"][key]["sha256"], key
    assert record["population"]["sha256_v1"] == ledger["population"]["sha256_v1"]
    assert record["population"]["sha256_v2"] == ledger["population"]["sha256_v2"]
    assert record["inputs"]["cohort"]["allele_freq_non_missing_in_population"] == 0


def test_the_population_reconciles_exactly(record, ledger):
    census = record["reconciliation"]["census"]
    assert census == ledger["expected"] == record["reconciliation"]["expected"]
    assert census["selected"] + census["eligible_unselected"] + census["unknown_base"] + census["outside_domain"] == \
        census["full"] == 1_490_014
    assert (census["selected"], census["eligible_unselected"], census["unknown_base"], census["outside_domain"]) == \
        (1_028_224, 459_512, 500, 1_778)
    assert census["sequence_resolved_primary"] == 1_487_736
    receipt = record["canonical_matrix"]["receipt"]
    assert receipt["selection"]["identity"] == ledger["selection_identity"]
    assert receipt["reference_status"]["result_sha256"] == ledger["reference_evidence"]["rows_sha256"]
    assert record["reference_status"]["payload"]["validation_result"]["verdicts"] == ledger["reference_evidence"]["verdicts"]


def test_the_af_family_is_missing_exactly_where_nothing_was_selected(record):
    census = record["reconciliation"]["census"]
    unselected = census["full"] - census["selected"]
    for name, f in record["af_family"].items():
        assert f["missing"] == unselected and f["missing_equals_unselected"], name
    assert record["af_family"]["af_is_observed_zero"]["ones"] == census["selected_reported_zero"]
    assert record["chunking"]["compared_with_one_call"]["identical"] is True


def test_the_identities_reproduce(record):
    cm = record["canonical_matrix"]
    assert C.sha256_bytes(C.canonical_bytes(cm["receipt"])) == cm["identity"]
    assert C.sha256_bytes(C.canonical_bytes(record["reference_status"]["payload"])) == record["reference_status"]["identity"]
    assert cm["receipt"]["reference_status"]["identity"] == record["reference_status"]["identity"]
    assert cm["receipt"]["roster"] == list(Q.canonical_roster()) and cm["receipt"]["training"] == Q.TRAINING


def test_the_record_is_bound_to_the_code_as_it_is_now(record):
    receipt = record["canonical_matrix"]["receipt"]
    for symbol, digest in {**receipt["procedure"], **record["procedure"]}.items():
        assert C.symbol_closure_digest(REPO, symbol) == digest, (
            f"{symbol} changed since the record was measured: re-run scripts/measure_qualified_preparation_1d1c1c.py")
    assert set(receipt["procedure"]) == set(Q.PROCEDURE)
    assert record["reference_status"]["payload"]["validation_rule"] == R.rule_identity()
    assert record["chunking"]["row_local_proof_sha256"] == Q.row_local_proof()


def test_the_exact_columns_reproduce_the_receipt_and_af_log10_is_certified_not_sealed(record):
    cm = record["canonical_matrix"]
    digests = cm["exact_columns"]
    assert sorted(digests) == sorted(c for c in Q.canonical_roster() if c not in Q.ENVIRONMENT_DEPENDENT)
    assert len(digests) == len(Q.canonical_roster()) - len(Q.ENVIRONMENT_DEPENDENT) == 88
    assert "af_log10" not in digests and Q._exact_digest(digests) == cm["receipt"]["exact_columns_sha256"]
    assert cm["receipt"]["environment_dependent"] == Q._environment_dependent_entry()
    assert cm["receipt"]["schema"] == Q.RECEIPT_SCHEMA
    # revision c: the declared GRCh38 pin, and the matrix's stated scope (allele-frequency family only)
    assert cm["receipt"]["reference_status"]["reference_pin_sha256"] == R.GRCH38_PRIMARY_ASSEMBLY.digest
    assert cm["receipt"]["reference_status"]["reference_assembly"] == R.GRCH38_PRIMARY_ASSEMBLY.as_dict()
    assert cm["receipt"]["scope"] == Q.QUALIFICATION_SCOPE
