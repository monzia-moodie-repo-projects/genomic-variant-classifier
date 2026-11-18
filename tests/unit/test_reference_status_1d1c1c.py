"""Set "av": the GRCh38 reference-identity authority (information_sources/reference_status.py). Owner ruling aa999b96
sections 14-15 and revised section 10: one validation rule; the cached cohort result reused only by exact source, rule,
reference and population identity; a novel variant validated against the pinned sequence, never by table membership.

Author: Monzia Moodie
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from genomic_variant_classifier.data.delta_window_builder import build_window
from genomic_variant_classifier.information_sources import materialization as M
from genomic_variant_classifier.information_sources import reference_status as R
from genomic_variant_classifier.information_sources.columns import Population
from tests.unit import _qualified_support as S


def _evidence(tmp_path, frame: pd.DataFrame) -> M.Parsed:
    path = tmp_path / "evidence.parquet"
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), buf)
    path.write_bytes(buf.getvalue())
    return M.read_verified_bytes(path)


def _status(tmp_path):
    cohort = S.cohort_frame()
    population = M.population_from_frame(cohort, S.POPULATION_ID)
    evidence = _evidence(tmp_path, S.evidence_frame(cohort))
    return population, evidence, R.materialize_reference_status(population, evidence, registered_datasets=S.REGISTRY)


def test_the_evidence_dataset_is_registered_in_the_census():
    M.check_registered(R.EVIDENCE_DATASET, S.REGISTRY)


def test_the_result_digest_is_the_af_ledgers_reference_digest_formula():
    # measure_af_repair_1d1c1.py binds sha256(canonical [[ok, reason], ...]); the AF ledger records it as rows_sha256.
    from genomic_variant_classifier.information_sources import census as C

    ok, reason = [True, False, True], ["", "ref_mismatch", ""]
    assert R.result_digest(ok, reason) == C.sha256_bytes(C.canonical_bytes([[o, r] for o, r in zip(ok, reason)]))


def test_the_status_is_aligned_to_the_population_and_reproduces_its_digest(tmp_path):
    population, _ev, status = _status(tmp_path)
    assert len(status.ok) == len(population) and not status.ok.flags.writeable
    assert status.result_sha256 == R.result_digest(status.ok.tolist(), status.reason)
    assert status.rule == R.rule_identity() and status.rule["normalization"].startswith("not_established")
    assert int((~status.ok).sum()) == 1 and status.reason[-3] == "non_acgt_allele"


def test_a_missing_or_duplicated_evidence_row_is_refused(tmp_path):
    cohort = S.cohort_frame()
    population = M.population_from_frame(cohort, S.POPULATION_ID)
    ev = S.evidence_frame(cohort)
    with pytest.raises(ValueError, match="have no reference evidence"):
        R.materialize_reference_status(population, _evidence(tmp_path, ev.iloc[1:]), registered_datasets=S.REGISTRY)
    with pytest.raises(ValueError, match="names a variant twice"):
        R.materialize_reference_status(population, _evidence(tmp_path, pd.concat([ev, ev.iloc[:1]])),
                                       registered_datasets=S.REGISTRY)
    bad = ev.assign(reason=np.where(ev["ok"], "x", ev["reason"]))
    with pytest.raises(ValueError, match="not 'ok with an empty reason'"):
        R.materialize_reference_status(population, _evidence(tmp_path, bad), registered_datasets=S.REGISTRY)


def test_an_unregistered_evidence_dataset_is_refused(tmp_path):
    cohort = S.cohort_frame()
    population = M.population_from_frame(cohort, S.POPULATION_ID)
    with pytest.raises(ValueError, match="not a dataset registered"):
        R.materialize_reference_status(population, _evidence(tmp_path, S.evidence_frame(cohort)), registered_datasets=())


def test_content_addressed_reuse_round_trips_and_is_idempotent(tmp_path):
    population, ev, status = _status(tmp_path)
    ident, payload = R.write_reference_status(status, tmp_path / "store")
    assert ident == status.identity
    assert R.write_reference_status(status, tmp_path / "store") == (ident, payload)    # same bytes: a verified no-op
    assert (tmp_path / "store" / ident / f"{payload}.parquet").is_file()              # <identity>/<payload>
    again = R.load_reference_status(tmp_path / "store", ident, payload, population=population, source_sha256=ev.sha256)
    assert again.identity == ident and again.reason == status.reason and np.array_equal(again.ok, status.ok)


def test_reuse_is_refused_for_another_source_population_or_rule(tmp_path, monkeypatch):
    population, ev, status = _status(tmp_path)
    ident, payload = R.write_reference_status(status, tmp_path / "store")
    with pytest.raises(ValueError, match="reference_source"):
        R.load_reference_status(tmp_path / "store", ident, payload, population=population, source_sha256="0" * 64)
    other = Population(population.population_id, population.row_ids[::-1])
    with pytest.raises(ValueError, match="different population"):
        R.load_reference_status(tmp_path / "store", ident, payload, population=other, source_sha256=ev.sha256)
    changed = dict(R.rule_identity(), implementation_closure_sha256="1" * 64)
    monkeypatch.setattr(R, "rule_identity", lambda repo=None: changed)
    with pytest.raises(ValueError, match="validation_rule"):
        R.load_reference_status(tmp_path / "store", ident, payload, population=population, source_sha256=ev.sha256)


def test_a_tampered_store_is_refused(tmp_path):
    population, ev, status = _status(tmp_path)
    store = tmp_path / "store"
    ident, payload = R.write_reference_status(status, store)
    table = store / ident / f"{payload}.parquet"
    original = table.read_bytes()
    table.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="not the one its record names"):
        R.load_reference_status(store, ident, payload, population=population, source_sha256=ev.sha256)
    table.write_bytes(original)
    receipt = store / ident / f"{payload}.receipt.json"
    with pytest.raises(FileExistsError, match="never replaced"):
        receipt.write_text(receipt.read_text().replace('"rows"', '"rows" '), encoding="ascii")
        R.write_reference_status(status, store)


# ------------------------------------------------------------------------------------------------- novel variants
def _genome():
    rng = np.random.default_rng(3)
    return {"1": "".join(rng.choice(list("ACGT"), 400)), "2": "".join(rng.choice(list("ACGT"), 250))}


def test_the_pinned_fetch_equals_a_python_slice_and_is_none_where_the_window_build_expects(tmp_path):
    seqs = _genome()
    path, pin = S.fasta(tmp_path, seqs)
    with R.PinnedReference(path, pin) as ref:
        for contig, start, length in (("1", 0, 101), ("1", 350, 101), ("2", 17, 5), ("2", 249, 3)):
            assert ref.fetch(contig, start, length) == seqs[contig][start:start + length]
        assert ref.fetch("1", -1, 5) is None and ref.fetch("chr1", 0, 5) is None


def test_a_novel_variant_is_validated_against_the_pinned_sequence_by_the_training_rule(tmp_path):
    seqs = _genome()
    path, pin = S.fasta(tmp_path, seqs)
    pos = 120
    ref_base = seqs["1"][pos - 1]
    alt = "A" if ref_base != "A" else "C"
    with R.PinnedReference(path, pin) as ref:
        good = R.validate_variant(ref, "1", pos, ref_base, alt)
        wrong = R.validate_variant(ref, "1", pos, "T" if ref_base != "T" else "G", alt)
        unknown = R.validate_variant(ref, "22", pos, ref_base, alt)
        n_allele = R.validate_variant(ref, "1", pos, "N", alt)
        fetch = ref.fetch
        assert (good.ok, good.reason) == R.verdict_of(build_window(fetch, "1", pos, ref_base, alt, 101))
    assert good.ok and good.reason == "" and good.reference_sha256 == pin.digest
    assert not wrong.ok and wrong.reason and not unknown.ok and (n_allele.ok, n_allele.reason) == (False, "non_acgt_allele")
    assert good.normalization.startswith("not_established")


def test_a_reference_whose_bytes_differ_from_the_pin_is_refused(tmp_path):
    path, pin = S.fasta(tmp_path, _genome())
    data = bytearray(path.read_bytes())
    data[10] = ord("A") if data[10] != ord("A") else ord("C")
    path.write_bytes(bytes(data))
    fai = path.with_name(path.name + ".fai")
    fai.write_bytes(fai.read_bytes())                 # the index newer than the edited FASTA, as pyfaidx expects
    with pytest.raises(ValueError, match="content SHA-256"):
        R.PinnedReference(path, pin)
    with R.PinnedReference(path, pin, verify_content=False) as unverified:
        with pytest.raises(PermissionError, match="content was verified"):
            R.validate_variant(unverified, "1", 5, "A", "C")


def test_the_grch38_pin_is_the_window_build_reference():
    pin = R.GRCH38_PRIMARY_ASSEMBLY
    assert pin.fasta_bytes == 3_151_425_851 and pin.assembly == "GRCh38"
    assert pin.window_manifest_signature == "234a0a170c3089b931aa6b462daa8fdc76774095b951c68ef1b7f911e74b91d6"
