"""Set "av" revision c (owner ruling 6fa240c0; audit AUDIT_OF_RULING_6fa240c0_2026-11-17.md): the trust boundaries of the
qualified preparation made exact. Each test re-creates one defect the ruling reproduced (or the audit added) and requires
its refusal, by its own message, so that each layer of defence is tested on its own:

  REFERENCE-VERDICT-ZIP-TRUNCATION-1   a reference digest covered only the first rows (ordinary zip);
  the reference pin                    a status under another pin was accepted;
  source bytes at consumption          a Parsed wrapper's digest was trusted without hashing its bytes;
  the live matrix                      an edit of the features changed predictions under an unchanged identity, and a
                                       matrix contradicting its receipt could be constructed;
  logical vs physical identity         a second certified af_log10 representation could not be stored;
  the existing-record fast path        success was reported with the payload missing or corrupt;
  concurrent publication               a fixed temporary name let one writer publish another's bytes;
  exact replay                         the binding named no representation and no scope.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import json
import os
import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import integrity as I
from genomic_variant_classifier.information_sources import materialization as M
from genomic_variant_classifier.information_sources import qualified_preparation as Q
from genomic_variant_classifier.information_sources import reference_status as R
from genomic_variant_classifier.information_sources.columns import Population
from tests.unit import _qualified_support as S

ROSTER = ["af_raw", "af_log10", "cadd_phred"]


class _Model:
    """A fitted model stand-in: a fixed logistic score of its columns (the binding, not the model, is under test)."""

    def __init__(self, columns):
        self.tabular_columns_ = list(columns)
        self.source_binding_ = None

    def predict_proba(self, X):
        z = np.nan_to_num(X.to_numpy(dtype=np.float64)).sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.c_[1.0 - p, p]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    return S.prepared(tmp_path_factory.mktemp("integrity") / "sources")


def _shifted(ulps: int):
    def formula(af):
        y = np.log10(af + 1e-8)
        for _ in range(ulps):
            y = np.nextafter(y, np.inf)
        return y
    return formula


@pytest.fixture(scope="module")
def two_representations(tmp_path_factory, prepared):
    """The same logical matrix on 'another machine': af_log10 two ULP away, within the registered bound."""
    import genomic_variant_classifier.models.feature_semantics as FS

    mp = pytest.MonkeyPatch()
    mp.setattr(FS, "formula_af_log10", _shifted(2))
    try:
        other, _ = S.prepared(tmp_path_factory.mktemp("integrity_b") / "sources")
    finally:
        mp.undo()
    return prepared[0], other


# ------------------------------------------------------------------------------ REFERENCE-VERDICT-ZIP-TRUNCATION-1
def test_a_reference_digest_refuses_verdicts_and_reasons_of_different_lengths():
    ok = np.array([True, True, False])
    with pytest.raises(ValueError, match="reason count differs"):
        R.result_digest(ok.tolist(), ("",))
    with pytest.raises(ValueError, match="reason count differs"):
        R.result_digest([True], ("", ""))
    with pytest.raises(ValueError, match="a bool and its reason a str"):
        R.result_digest([1, 0], ("", "x"))
    # well-formed verdicts keep the digest the AF ledger recorded (canonical [[ok, reason], ...])
    good = ([True, False, True], ["", "ref_mismatch", ""])
    assert R.result_digest(*good) == C.sha256_bytes(C.canonical_bytes([[o, r] for o, r in zip(*good)]))


def _status_args(status):
    return dict(population=status.population, ok=np.array(status.ok.tolist()), reason=status.reason,
                source_sha256=status.source_sha256, rule=dict(status.rule), reference=dict(status.reference),
                alignment=dict(status.alignment), result_sha256=status.result_sha256)


def test_a_reference_status_is_valid_by_construction(prepared):
    status = prepared[1]["status"]
    base = _status_args(status)
    n = len(status.population)
    cases = [
        (dict(reason=status.reason[:1]), "for a population of"),                       # the ruling's truncated reasons
        (dict(ok=base["ok"][:-1]), "for a population of"),
        (dict(ok=base["ok"].astype(np.int8)), "one-dimensional boolean"),
        (dict(ok=np.stack([base["ok"], base["ok"]])), "one-dimensional boolean"),
        (dict(reason=list(status.reason)), "tuple of str"),
        (dict(reason=("x",) + status.reason[1:]), "not 'ok with an empty reason'"),
        (dict(alignment={**base["alignment"], "population_sha256": "0" * 64}), "different population"),
        (dict(reference={**base["reference"], "fasta_sha256": "not-a-digest"}), "lower-case SHA-256"),
        (dict(result_sha256="1" * 64), "do not reproduce their result digest"),
        (dict(source_sha256="abc"), "lower-case SHA-256"),
    ]
    for change, message in cases:
        with pytest.raises((ValueError, TypeError), match=message):
            R.ReferenceStatus(**{**base, **change})
    rebuilt = R.ReferenceStatus(**base)
    assert rebuilt.identity == status.identity and len(rebuilt.ok) == n
    with pytest.raises(ValueError):
        rebuilt.ok.setflags(write=True)                                    # bytes-backed: read-only for good
    with pytest.raises(TypeError):
        rebuilt.rule["rule_id"] = "another"                               # the identity's mappings are read-only


def test_a_status_under_another_reference_pin_is_refused_by_the_preparation(prepared):
    cm, inp = prepared
    status = inp["status"]
    other_pin = R.ReferencePin("GRCh37", "another assembly", "1" * 64, 1, "2" * 64, "3" * 64)
    foreign = R.ReferenceStatus(**{**_status_args(status), "reference": other_pin.as_dict()})
    with pytest.raises(ValueError, match="another reference identity"):
        Q.prepare_qualified(inp["cohort"], inp["population"], **{**inp["args"], "reference": foreign})
    with pytest.raises(ValueError, match="another reference identity"):        # the declared pin decides, not a default
        Q.prepare_qualified(inp["cohort"], inp["population"], **{**inp["args"], "reference_pin": other_pin})
    with pytest.raises(TypeError, match="declared reference_status.ReferencePin"):
        Q.prepare_qualified(inp["cohort"], inp["population"], **{**inp["args"], "reference_pin": other_pin.as_dict()})
    assert cm.receipt["reference_status"]["reference_pin_sha256"] == R.GRCH38_PRIMARY_ASSEMBLY.digest


# -------------------------------------------------------------------------------- source bytes at consumption
def _forged(parsed: M.Parsed) -> M.Parsed:
    """Other bytes in a wrapper that states the original digest and size."""
    table = pq.read_table(pa.BufferReader(parsed.value)).to_pandas()
    column = "allele_freq" if "allele_freq" in table.columns else "reason"
    table.loc[table.index[2], column] = 0.25 if column == "allele_freq" else "ref_mismatch"
    if column == "reason":
        table.loc[table.index[2], "ok"] = False
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), buf)
    return M.Parsed(buf.getvalue(), parsed.sha256, parsed.n_bytes)


def test_a_wrapper_whose_bytes_are_not_its_digest_is_refused_where_it_is_consumed(prepared):
    _, inp = prepared
    with pytest.raises(ValueError, match="not the .* their wrapper states"):
        Q.prepare_qualified(inp["cohort"], inp["population"], **{**inp["args"], "gnomad": _forged(inp["gnomad"])})
    with pytest.raises(ValueError, match="not the .* their wrapper states"):
        R.materialize_reference_status(inp["population"], _forged(inp["evidence"]), registered_datasets=S.REGISTRY)
    short = M.Parsed(inp["gnomad"].value, inp["gnomad"].sha256, inp["gnomad"].n_bytes + 1)
    with pytest.raises(ValueError, match="their wrapper states"):
        I.require_verified_bytes(short, what="gnomad")
    with pytest.raises(ValueError, match="not the declared"):
        I.require_verified_bytes(inp["gnomad"], what="gnomad", declared_sha256="0" * 64)
    assert I.require_verified_bytes(inp["gnomad"], what="gnomad", declared_sha256=inp["gnomad"].sha256) == \
        inp["gnomad"].sha256


# ------------------------------------------------------------------------------------------------- the live matrix
def test_the_live_matrix_cannot_be_edited_and_a_copy_changes_nothing(prepared):
    cm = prepared[0]
    identity, representation = cm.identity, cm.representation
    before = cm.features["af_raw"].to_numpy().copy()
    j = list(cm.features.columns).index("af_raw")
    def through_a_series(f):
        column = f["af_raw"]                                   # held by a name: an edit of the column itself
        column.iloc[2] = 0.9
    for edit in (lambda f: f.iloc.__setitem__((2, j), 0.9), lambda f: f.loc.__setitem__((2, "af_raw"), 0.9),
                 lambda f: f["af_raw"].to_numpy().__setitem__(2, 0.9), through_a_series):
        with pytest.raises(ValueError, match="read-only"):
            edit(cm.features)
    view = cm.features
    view["af_raw"] = 0.5                                       # replaces the column in this view only
    assert np.array_equal(cm.features["af_raw"].to_numpy(), before, equal_nan=True)
    for vector in (cm.selection, cm.claim, cm.columns["af_raw"]):
        with pytest.raises(ValueError, match="read-only"):
            vector[0] = 1
        with pytest.raises(ValueError):
            vector.setflags(write=True)
    with pytest.raises(TypeError):
        cm.columns["af_raw"] = before                         # the column mapping is read-only
    receipt = cm.receipt
    receipt["training"] = "allowed"
    assert cm.receipt["training"] == Q.TRAINING              # a fresh copy each time
    assert cm.identity == identity and cm.representation == representation
    shares = np.shares_memory(cm.features["af_raw"].to_numpy(), cm.columns["af_raw"])
    assert shares                                             # the view is zero-copy: no 1 GB copy per access


def _forge(cm, field_name: str, consistent: bool):
    """An object.__setattr__ forgery of one stored array -- the only way left to change a constructed matrix -- and,
    when ``consistent``, of the representation too (recomputed from the forged arrays), so that only the receipt's
    digests can refuse it."""
    forged = object.__new__(Q.CanonicalMatrix)
    for name in ("population", "columns", "selection", "claim", "receipt_bytes", "representation", "certified_ulp"):
        object.__setattr__(forged, name, getattr(cm, name))
    if field_name in ("af_raw", "af_log10"):
        cols = dict(cm.columns)
        v = cols[field_name]
        cols[field_name] = np.where(np.isnan(v), np.nan, 0.9 if field_name == "af_raw" else np.nextafter(v, np.inf))
        object.__setattr__(forged, "columns", cols)
    else:
        codes = np.array(getattr(cm, field_name))
        codes[0] = (codes[0] + 1) % 3
        object.__setattr__(forged, field_name, codes)
    if consistent:
        object.__setattr__(forged, "representation",
                           forged._representation({c: Q._column_sha256(v) for c, v in forged.columns.items()}))
    return forged


@pytest.mark.parametrize("field_name, consistent, layer", [
    ("af_raw", False, "the exact columns"), ("selection", False, "the selection or claim codes"),
    ("claim", False, "the selection or claim codes"),
    ("af_log10", False, "the representation"),                 # within the certificate: only the representation sees it
    ("af_raw", True, "the exact columns"), ("selection", True, "the selection or claim codes")])
def test_a_forged_matrix_is_refused_at_bind_score_and_write(prepared, tmp_path, field_name, consistent, layer):
    cm = prepared[0]
    forged = _forge(cm, field_name, consistent)
    assert forged.identity == cm.identity                        # the forgery keeps the identity ...
    message = f"not the ones verified at its construction \\({layer}"
    with pytest.raises(ValueError, match=message):
        Q.bind_fit(_Model(ROSTER), forged)                        # ... and every consumer refuses it
    model = _Model(ROSTER)
    Q.bind_fit(model, cm)
    with pytest.raises(ValueError, match=message):
        Q.score_canonical(model, forged)
    with pytest.raises(ValueError, match=message):
        Q.write_canonical(forged, tmp_path / "store")
    assert not (tmp_path / "store").exists()


def test_a_code_change_is_refused_at_every_consumer(prepared, tmp_path, monkeypatch):
    cm = prepared[0]
    model = _Model(ROSTER)
    Q.bind_fit(model, cm)
    monkeypatch.setattr(Q, "symbol_closure_digest", lambda repo, symbol: "3" * 64)
    for consume in (lambda: Q.bind_fit(_Model(ROSTER), cm), lambda: Q.score_canonical(model, cm),
                    lambda: Q.write_canonical(cm, tmp_path / "store")):
        with pytest.raises(ValueError, match="code that produced this matrix has changed"):
            consume()


def _receipt_with(cm, **change) -> bytes:
    return C.canonical_bytes({**cm.receipt, **change})


def test_a_matrix_contradicting_its_receipt_cannot_be_constructed(prepared):
    cm = prepared[0]
    cols = dict(cm.columns)
    args = dict(population=cm.population, columns=cols, selection=cm.selection, claim=cm.claim,
                receipt_bytes=cm.receipt_bytes)
    assert Q.CanonicalMatrix(**args).representation == cm.representation
    writable = {c: np.array(v) for c, v in cols.items()}                  # the caller keeps writable arrays ...
    copied = Q.CanonicalMatrix(**{**args, "columns": writable})
    writable["af_raw"][:] = 0.5                                            # ... and edits them afterwards
    assert not np.shares_memory(copied.columns["af_raw"], writable["af_raw"])
    copied.verify()                                                        # the matrix stored its own copy
    raw = dict(cols, af_raw=np.where(np.isnan(cols["af_raw"]), np.nan, cols["af_raw"] * 2))
    far = dict(cols, af_log10=_shifted(9)(np.where(np.isnan(cols["af_raw"]), np.nan, cols["af_raw"])))
    selection = np.array(cm.selection)
    selection[0] = (selection[0] + 1) % 3
    claim = np.array(cm.claim)
    claim[0] = (claim[0] + 1) % 3
    reordered = {c: cols[c] for c in reversed(list(cols))}
    other = Population(cm.population.population_id, cm.population.row_ids[::-1])
    cases = [
        (dict(columns=raw), "exact-column digest"),
        (dict(columns=far), "beyond the certified bound"),
        (dict(selection=selection), "selection codes do not reproduce"),
        (dict(claim=claim), "claim codes do not reproduce"),
        (dict(columns=reordered), "receipt's roster, in its order"),
        (dict(columns=dict(cols, af_raw=cols["af_raw"].astype(np.float32))), "float64 array"),
        (dict(population=other), "another population"),
        (dict(receipt_bytes=cm.receipt_bytes + b" "), "canonical form"),
        (dict(receipt_bytes=_receipt_with(cm, schema="gvc.canonical-matrix-receipt.1d1c1c/2")), "schema"),
        (dict(receipt_bytes=_receipt_with(cm, environment_dependent={})), "another accuracy contract"),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            Q.CanonicalMatrix(**{**args, **change})


# ------------------------------------------------------------------------------- logical vs physical identity
def test_two_certified_representations_share_the_logical_identity_and_are_stored_apart(two_representations, tmp_path):
    here, there = two_representations
    assert here.identity == there.identity and here.representation != there.representation
    store = tmp_path / "store"
    a, b = Q.write_canonical(here, store), Q.write_canonical(there, store)
    assert a.identity == b.identity and a.payload_sha256 != b.payload_sha256
    assert I.stored_payloads(store, here.identity) == tuple(sorted([a.payload_sha256, b.payload_sha256]))
    assert Q.load_canonical(store, *a).representation == here.representation
    assert Q.load_canonical(store, *b).representation == there.representation
    assert Q.payload_for(store, here.identity, there.representation) == b.payload_sha256      # the replay selector
    with pytest.raises(ValueError, match="exactly one is required"):
        Q.payload_for(store, here.identity, "0" * 64)


def test_replay_is_exact_a_numerically_equivalent_representation_is_refused(two_representations):
    here, there = two_representations
    model = _Model(ROSTER)
    binding = Q.bind_fit(model, here)
    assert binding["scope"] == Q.BINDING_SCOPE and binding["representation_sha256"] == here.representation
    assert binding["schema"] == Q.BINDING_SCHEMA
    Q.score_canonical(model, here)
    with pytest.raises(PermissionError, match="representation_sha256"):
        Q.score_canonical(model, there)
    widened = {k: v for k, v in binding.items() if k != "digest"}
    widened["scope"] = "general_inference"
    model.source_binding_ = {**widened, "digest": C.sha256_bytes(C.canonical_bytes(widened))}
    with pytest.raises(PermissionError, match="exact-replay binding"):
        Q.score_canonical(model, here)


def test_a_record_that_names_another_representation_or_certificate_is_refused(prepared, tmp_path):
    cm = prepared[0]
    stored = Q.write_canonical(cm, tmp_path / "store")
    record = tmp_path / "store" / cm.identity / f"{stored.payload_sha256}.receipt.json"
    original = json.loads(record.read_text())
    for key, value, message in (("representation_sha256", "0" * 64, "record's representation"),
                                ("certificate", {**original["certificate"], "af_log10_max_ulp": 3.9}, "certificate")):
        record.write_text(json.dumps({**original, key: value}, indent=2, sort_keys=True) + "\n", encoding="ascii")
        with pytest.raises(ValueError, match=message):
            Q.load_canonical(tmp_path / "store", *stored)


# ---------------------------------------------------------------------------------- the existing-record fast path
def _writers(prepared):
    cm, inp = prepared
    return [("canonical", lambda d: Q.write_canonical(cm, d), cm.identity),
            ("reference", lambda d: R.write_reference_status(inp["status"], d), inp["status"].identity)]


@pytest.mark.parametrize("which", ["canonical", "reference"])
def test_an_existing_record_succeeds_only_with_its_payload_present_and_intact(prepared, tmp_path, which):
    _, write, logical = next(w for w in _writers(prepared) if w[0] == which)
    store = tmp_path / "store"
    first = write(store)
    payload = store / logical / f"{first[1]}.parquet"
    assert write(store) == first                                         # intact: a verified no-op
    original = payload.read_bytes()
    payload.unlink()
    with pytest.raises(FileNotFoundError, match="store is incomplete"):
        write(store)
    assert not payload.exists()                                          # refused, never repaired by a writer
    payload.write_bytes(original[:-5] + b"wrong")
    with pytest.raises(ValueError, match="store is corrupt"):
        write(store)
    payload.write_bytes(original)
    assert write(store) == first


def test_a_payload_without_its_record_is_completed_only_with_identical_bytes(prepared, tmp_path):
    cm = prepared[0]
    stored = Q.write_canonical(cm, tmp_path / "store")
    folder = tmp_path / "store" / cm.identity
    (folder / f"{stored.payload_sha256}.receipt.json").unlink()          # a writer stopped between its two files
    assert Q.write_canonical(cm, tmp_path / "store") == stored             # the same bytes: completed
    (folder / f"{stored.payload_sha256}.receipt.json").unlink()
    raw = (folder / f"{stored.payload_sha256}.parquet").read_bytes()
    (folder / f"{stored.payload_sha256}.parquet").write_bytes(raw + b"\0")
    with pytest.raises(FileExistsError, match="never replaced"):
        Q.write_canonical(cm, tmp_path / "store")
    assert not (folder / f"{stored.payload_sha256}.receipt.json").exists()   # no record over a wrong payload


# ------------------------------------------------------------------------------------------ concurrent publication
def test_concurrent_writers_of_one_and_of_two_representations_never_collide(two_representations, tmp_path):
    here, there = two_representations
    store = tmp_path / "store"
    barrier = threading.Barrier(12)
    results, errors = [], []

    def writer(cm):
        barrier.wait()
        try:
            results.append(Q.write_canonical(cm, store))
        except BaseException as exc:               # noqa: BLE001 -- every outcome is recorded and asserted
            errors.append(exc)
    threads = [threading.Thread(target=writer, args=(here if i % 2 else there,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(results) == 12 and len(set(results)) == 2
    folder = store / here.identity
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        f"{r.payload_sha256}{suffix}" for r in set(results) for suffix in (".parquet", ".receipt.json"))   # no leftovers
    for r in set(results):
        Q.load_canonical(store, *r)


def test_a_writer_racing_another_writers_bytes_is_refused_and_publishes_no_record(prepared, tmp_path, monkeypatch):
    """The interleaving the fixed temporary name allowed: between this writer's write and its publication, another
    process publishes OTHER bytes at the same name. Exclusive publication refuses instead of completing over them."""
    cm = prepared[0]
    real_link = os.link

    def racing_link(src, dst):
        if str(dst).endswith(".parquet"):
            with open(dst, "wb") as fh:            # the other writer wins the name with other bytes
                fh.write(b"another writer's bytes")
        return real_link(src, dst)
    monkeypatch.setattr(I.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="never replaced"):
        Q.write_canonical(cm, tmp_path / "store")
    folder = tmp_path / "store" / cm.identity
    assert [p.name for p in folder.iterdir() if p.name.endswith(".receipt.json")] == []     # no false completion
    assert [p.name for p in folder.iterdir() if p.name.endswith(".partial")] == []          # its temporary removed


def test_a_file_system_without_hard_links_is_refused_loudly(prepared, tmp_path, monkeypatch):
    def no_links(src, dst):
        raise PermissionError("operation not permitted")
    monkeypatch.setattr(I.os, "link", no_links)
    with pytest.raises(OSError, match="needs hard links"):
        Q.write_canonical(prepared[0], tmp_path / "store")


# --------------------------------------------------------------------------------------------- stated limitations
def test_the_receipt_states_its_scope_and_the_bounds_basis(prepared):
    receipt = prepared[0].receipt
    assert receipt["scope"] == Q.QUALIFICATION_SCOPE and "QUALIFIED-FRAME-PROVENANCE-1" in receipt["scope"]
    assert "not a source-qualified biological feature matrix" in receipt["scope"]
    basis = receipt["environment_dependent"]["af_log10"]["certificate"]["bound_basis"]
    assert "registered permissible error" in basis and "not a guarantee" in basis
    assert receipt["schema"] == Q.RECEIPT_SCHEMA == "gvc.canonical-matrix-receipt.1d1c1c/3"
