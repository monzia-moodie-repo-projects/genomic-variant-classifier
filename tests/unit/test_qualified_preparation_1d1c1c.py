"""Set "av": the qualified data preparation (information_sources/qualified_preparation.py) and the compiled rosters
(admission.compiled_roster). Owner rulings aa999b96 (revised section 10, points 1-10) and ff77015c (sections 3, 11 and
13: qualified AF preparation, refusal of unauthorized source substitution, canonical epoch-2 materialisation, exact
reconciliation, the LOEUF withholding through the admission authority).

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import admission as AD
from genomic_variant_classifier.information_sources import allele_frequency as A
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import qualified_preparation as Q
from genomic_variant_classifier.information_sources import reference_status as R
from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, FeatureSemantics, derive_family, tabular_features
from tests.unit import _qualified_support as S


@pytest.fixture(scope="module")
def ledger():
    raw = (S.REPO / C.LEDGER_PATH).read_bytes()
    return C.strict_loads(raw.decode("ascii")), C.sha256_bytes(raw)


# ------------------------------------------------------------------------------------------------- the explicit mode
@pytest.mark.parametrize("mode", [None, "qualified", Q.PreparationMode.HISTORICAL_REPLAY])
def test_only_the_explicit_qualified_mode_is_prepared(tmp_path, mode):
    with pytest.raises(ValueError, match="no mode is inferred"):
        S.prepared(tmp_path, mode=mode)


# -------------------------------------------------------------------------------- the three AF states, preserved
def test_an_observed_value_an_observed_zero_and_an_unobserved_value_are_kept_apart(tmp_path):
    cm, inp = S.prepared(tmp_path)
    af = cm.features["af_raw"].to_numpy()
    sel = cm.selection
    selected = sel == A.SELECTION_STATES.index("selected")
    assert np.isnan(af[~selected]).all() and not np.isnan(af[selected]).any()
    assert af[0] == 0.0 and cm.features["af_is_observed_zero"].iloc[0] == 1.0            # an observed zero
    assert np.isnan(af[1]) and sel[1] == A.SELECTION_STATES.index("no_qualified_observation")  # record without value
    assert (af[selected] > 0).any()
    assert "af_is_absent" not in cm.features.columns


def test_the_af_family_is_derive_familys_bit_for_bit_and_nan_wherever_unselected(tmp_path):
    cm, _ = S.prepared(tmp_path)
    expected = derive_family(AF_FAMILY, cm.features["af_raw"].to_numpy())
    for name, values in expected.items():
        got = cm.features[name].to_numpy()
        assert np.array_equal(np.isnan(got), np.isnan(values))
        assert np.array_equal(got[~np.isnan(got)].view(np.uint64), values[~np.isnan(values)].view(np.uint64))


def test_the_full_population_is_kept_and_eligibility_is_a_recorded_projection(tmp_path):
    cm, inp = S.prepared(tmp_path)
    census = cm.receipt["selection"]["census"]
    assert len(cm.features) == len(inp["population"]) == census["full"]
    assert census["selected"] + census["eligible_unselected"] + census["unknown_base"] + census["outside_domain"] == \
        census["full"]
    assert int(cm.sequence_resolved_primary.sum()) == census["sequence_resolved_primary"]
    n_row = inp["cohort"].index[inp["cohort"]["ref"] == "AN"][0]
    assert cm.claim[n_row] == A.CLAIM_STATES.index("incomplete_allele")         # kept, not deleted
    for chrom in ("Y", "MT"):
        row = inp["cohort"].index[inp["cohort"]["chrom"] == chrom][0]
        assert cm.claim[row] == A.CLAIM_STATES.index("outside_domain") and np.isnan(cm.features["af_raw"].iloc[row])
    x = inp["cohort"].index[inp["cohort"]["chrom"] == "X"][0]
    assert cm.features["af_raw"].iloc[x] == 0.0123                              # X: selected by the strict policy


def test_a_declared_census_must_reconcile_exactly(tmp_path):
    cm, _ = S.prepared(tmp_path)
    census = dict(cm.receipt["selection"]["census"])
    S.prepared(tmp_path / "again", expected_census=census)
    with pytest.raises(ValueError, match="investigated, never absorbed"):
        S.prepared(tmp_path / "off", expected_census={**census, "selected": census["selected"] + 1})


# ---------------------------------------------------------------------------------------------- no substitution
def test_a_source_other_than_the_declared_one_is_refused_by_name(tmp_path):
    _, inp = S.prepared(tmp_path)
    for logical in (Q.COHORT, A.PRIMARY, R.EVIDENCE_DATASET):
        declared = {**inp["declared"], logical: "0" * 64}
        with pytest.raises(ValueError, match=f"{logical}: the bytes supplied"):
            S.prepared(tmp_path / logical.replace(":", "_"), declared_sources=declared)
    with pytest.raises(ValueError, match="declare exactly the sources"):
        S.prepared(tmp_path / "fewer", declared_sources={Q.COHORT: inp["declared"][Q.COHORT]})


def test_a_reference_status_for_another_population_or_rule_is_refused(tmp_path, monkeypatch):
    cm, inp = S.prepared(tmp_path)
    other = S.cohort_frame(seed=0).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="another population"):
        Q.prepare_qualified(other, S.M.population_from_frame(other, S.POPULATION_ID), **inp["args"])
    changed = dict(R.rule_identity(), verdict_closure_sha256="2" * 64)
    monkeypatch.setattr(R, "rule_identity", lambda repo=None: changed)
    with pytest.raises(ValueError, match="another validation rule"):
        Q.prepare_qualified(inp["cohort"], inp["population"], **inp["args"])


def test_an_allele_frequency_already_in_the_frame_is_an_unqualified_source(tmp_path):
    _, inp = S.prepared(tmp_path)
    frame = inp["cohort"].copy()
    frame.loc[3, "allele_freq"] = 0.01
    with pytest.raises(ValueError, match="unqualified source"):
        Q.prepare_qualified(frame, inp["population"], **inp["args"])


@pytest.mark.parametrize("column", ["af_1kg_eur", "finngen_af_fin", "gnomad_af"])
def test_the_legacy_joins_population_frequency_columns_are_refused(tmp_path, column):
    _, inp = S.prepared(tmp_path)
    frame = inp["cohort"].assign(**{column: 0.0})
    with pytest.raises(ValueError, match="legacy gnomAD join"):
        Q.prepare_qualified(frame, inp["population"], **inp["args"])


def test_the_preparation_never_reads_labels_or_audit_only_columns(tmp_path):
    cm, inp = S.prepared(tmp_path)
    permuted = inp["cohort"].assign(label=np.random.default_rng(5).permutation(inp["cohort"]["label"].to_numpy()),
                                    source_db="another-audit-value")
    again = Q.prepare_qualified(permuted, inp["population"], **inp["args"])
    assert again.identity == cm.identity and again.receipt == cm.receipt


def test_the_label_derived_columns_are_not_in_the_canonical_matrix(tmp_path):
    cm, _ = S.prepared(tmp_path)
    assert list(cm.features.columns) == list(Q.canonical_roster())
    assert set(Q.SPLIT_DEPENDENT).isdisjoint(cm.features.columns)
    assert list(Q.canonical_roster()) == [n for n in tabular_features(FeatureSemantics.V2_OBSERVED_AF)
                                          if n not in Q.SPLIT_DEPENDENT]
    assert cm.receipt["training"].startswith("refused")


def test_the_receipt_binds_sources_policy_population_code_and_values(tmp_path):
    cm, inp = S.prepared(tmp_path)
    r = cm.receipt
    assert r["mode"] == "qualified" and r["semantics"] == "gvc.features/2"
    assert r["sources"] == {k: inp["declared"][k] for k in sorted(inp["declared"])}
    assert r["population"]["sha256"] == inp["population"].sha256
    assert r["reference_status"]["identity"] == inp["status"].identity
    assert set(r["procedure"]) == set(Q.PROCEDURE) and all(C.HEX64.match(d) for d in r["procedure"].values())
    arm_b = A.AFSelectionPolicy(unknown_base_treatment=S.POLICY.unknown_base_treatment, arm=A.Arm.WITHHOLD_REPORTED_ZEROS)
    other, _ = S.prepared(tmp_path / "b", policy=arm_b)
    assert other.identity != cm.identity and other.receipt["selection"]["policy"]["arm"] == "withhold_reported_zeros"


# --------------------------------------------------------------------------------------------------- persistence
def test_the_canonical_matrix_round_trips_content_addressed(tmp_path):
    cm, inp = S.prepared(tmp_path)
    stored = Q.write_canonical(cm, tmp_path / "store")
    assert stored.identity == cm.identity and Q.write_canonical(cm, tmp_path / "store") == stored
    payload = tmp_path / "store" / cm.identity / f"{stored.payload_sha256}.parquet"           # <logical>/<payload>
    assert C.sha256_bytes(payload.read_bytes()) == stored.payload_sha256
    back = Q.load_canonical(tmp_path / "store", *stored, population=inp["population"])
    assert back.identity == cm.identity and back.features.equals(cm.features)
    assert back.representation == cm.representation and back.certified_ulp == cm.certified_ulp
    assert np.array_equal(back.selection, cm.selection) and np.array_equal(back.claim, cm.claim)


def test_a_tampered_or_stale_canonical_matrix_is_refused(tmp_path, monkeypatch):
    cm, inp = S.prepared(tmp_path)
    store = tmp_path / "store"
    stored = Q.write_canonical(cm, store)
    record = store / stored.identity / f"{stored.payload_sha256}.receipt.json"
    data = json.loads(record.read_text())
    data["receipt"]["roster"] = data["receipt"]["roster"][::-1]
    record.write_text(json.dumps(data), encoding="ascii")
    with pytest.raises(ValueError, match="does not reproduce its identity"):
        Q.load_canonical(store, *stored)
    stored2 = Q.write_canonical(cm, tmp_path / "store2")
    with pytest.raises(ValueError, match="another population"):
        other = S.M.population_from_frame(inp["cohort"].iloc[::-1], S.POPULATION_ID)
        Q.load_canonical(tmp_path / "store2", *stored2, population=other)
    monkeypatch.setattr(Q, "symbol_closure_digest", lambda repo, symbol: "3" * 64)
    with pytest.raises(ValueError, match="code that produced this matrix has changed"):
        Q.load_canonical(tmp_path / "store2", *stored2)


# ------------------------------------------------------------------------------------------------ compiled rosters
def test_the_compiled_rosters_equal_the_shadow_admission_record(ledger):
    led, sha = ledger
    report = json.loads((S.REPO / AD.REPORT_PATH).read_text())
    for regime in AD.REGIMES:
        admitted = [r["name"] for r in report["rows"] if r["kind"] == "feature" and r["admission"][regime] == "admitted"]
        compiled = AD.compiled_roster(led, ledger_sha256=sha, regime=regime)
        assert compiled["names"] == admitted and compiled["withheld_closure"] == []
    assert len(AD.compiled_roster(led, ledger_sha256=sha, regime="strict")["names"]) == 55


def test_withholding_loeuf_removes_its_admitted_dependent_closure(ledger):
    led, sha = ledger
    for regime in AD.REGIMES:
        r = AD.compiled_roster(led, ledger_sha256=sha, regime=regime, withheld=frozenset({"loeuf"}))
        # Measured: gnn_score and hetero_gnn_score also depend on loeuf, but no regime admits them already (they depend on
        # the retired n_pathogenic_in_gene), so the closure that drops out today is the pair.
        assert r["withheld_closure"] == ["gene_is_constrained", "loeuf"]                 # contract order
        assert "loeuf" not in r["names"] and "gene_is_constrained" not in r["names"]
    deps = {e["name"]: e.get("depends_on", []) for e in led["entries"]}
    assert {n for n, d in deps.items() if "loeuf" in d} == {"gene_is_constrained", "gnn_score", "hetero_gnn_score"}


def test_withholding_uses_the_transitive_fixed_point_not_a_list(ledger):
    led, sha = ledger
    r = AD.compiled_roster(led, ledger_sha256=sha, regime="strict", withheld=frozenset({"af_raw"}))
    assert {"af_raw", "af_log10", "af_is_absent", "af_is_ultra_rare", "af_is_rare", "af_is_common"} <= \
        set(r["withheld_closure"])


def test_an_inert_unknown_or_malformed_withholding_is_refused(ledger):
    led, sha = ledger
    with pytest.raises(ValueError, match="changes nothing"):
        AD.compiled_roster(led, ledger_sha256=sha, regime="strict", withheld=frozenset({"n_pathogenic_in_gene"}))
    with pytest.raises(ValueError, match="not census entries"):
        AD.compiled_roster(led, ledger_sha256=sha, regime="strict", withheld=frozenset({"no_such_feature"}))
    with pytest.raises(TypeError, match="frozenset"):
        AD.compiled_roster(led, ledger_sha256=sha, regime="strict", withheld={"loeuf"})
    with pytest.raises(ValueError, match="unknown regime"):
        AD.compiled_roster(led, ledger_sha256=sha, regime="lenient")


def test_a_cyclic_lineage_is_refused_by_the_same_authority(ledger):
    led, sha = ledger
    cyclic = json.loads(json.dumps(led))
    entries = {e["name"]: e for e in cyclic["entries"]}
    entries["loeuf"]["depends_on"] = ["gene_is_constrained"]
    with pytest.raises(ValueError, match="cycl"):
        AD.compiled_roster(cyclic, ledger_sha256=sha, regime="strict")


def test_a_fit_roster_is_the_compiled_roster_in_epoch_2_names(ledger):
    led, sha = ledger
    names = Q.fit_roster(AD.compiled_roster(led, ledger_sha256=sha, regime="strict", withheld=frozenset({"loeuf"})))
    assert "af_is_observed_zero" in names and "af_is_absent" not in names and len(names) == 53
    with pytest.raises(ValueError, match="compiled_roster"):
        Q.fit_roster({"names": list(names)})


def test_the_builder_receives_only_the_columns_it_can_read_and_its_output_is_identical(tmp_path):
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    names = Q.builder_input_names()
    assert {"chrom", "ref", "alt", "consequence", "allele_freq", "cadd_phred", "gene_constraint_oe"} <= names
    assert not {"variant_id", "pos", "gene_symbol", "source_db", "fasta_seq", "clinical_sig"} & names
    frame = S.cohort_frame().drop(columns=["label"]).assign(allele_freq=np.linspace(0, 0.5, 320))
    full = engineer_features(frame, semantics=FeatureSemantics.V2_OBSERVED_AF)
    part = engineer_features(frame[[c for c in frame.columns if c in names]], semantics=FeatureSemantics.V2_OBSERVED_AF)
    assert full.equals(part)


def test_the_callers_frame_is_never_modified(tmp_path):
    cohort = S.cohort_frame()
    before = cohort.copy(deep=True)
    S.prepared(tmp_path, cohort=cohort)
    assert cohort.equals(before) and list(cohort.columns) == list(before.columns)


@pytest.mark.parametrize("chunk_rows", [7, 64, 319, 320, 10_000])
def test_chunked_building_is_bit_identical_to_one_call(tmp_path, chunk_rows):
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    frame = S.cohort_frame().drop(columns=["label"]).assign(
        allele_freq=np.where(np.arange(320) % 5 == 0, np.nan, np.linspace(0, 0.5, 320)))
    whole = engineer_features(frame, semantics=FeatureSemantics.V2_OBSERVED_AF)
    chunked = Q.build_in_chunks(frame, Q.canonical_roster(), chunk_rows=chunk_rows)
    for c in Q.canonical_roster():
        a, b = whole[c].to_numpy(dtype=np.float64), chunked[c].to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)].view(np.uint64),
                                                                            b[~np.isnan(b)].view(np.uint64)), c
    cm, _ = S.prepared(tmp_path, chunk_rows=chunk_rows)
    reference, _ = S.prepared(tmp_path / "one", chunk_rows=10_000)
    assert cm.identity == reference.identity


def test_the_row_local_proof_refuses_a_builder_that_combines_rows(monkeypatch):
    import genomic_variant_classifier.models.variant_ensemble as VE

    def engineer_features(df, *, semantics=None):
        feats = df[["cadd_phred"]].copy()
        feats["cadd_phred"] = df["cadd_phred"] - df["cadd_phred"].mean()
        return feats

    monkeypatch.setattr(VE, "engineer_features", engineer_features)
    with pytest.raises(AssertionError, match="combines rows"):
        Q.row_local_proof()
    with pytest.raises(ValueError, match="chunk_rows"):
        Q.build_in_chunks(pd.DataFrame({"a": [1.0]}), ("a",), chunk_rows=0)


_A_NAME = "cadd_phred"


def test_the_input_bound_refuses_a_builder_whose_column_names_escape_its_literals(monkeypatch):
    import genomic_variant_classifier.models.variant_ensemble as VE

    def engineer_features(df, *, semantics=None):
        return df.get(_A_NAME)

    monkeypatch.setattr(VE, "engineer_features", engineer_features)
    with pytest.raises(AssertionError, match="cannot be bounded"):
        Q.builder_input_names()

    def engineer_features(df, *, semantics=None):            # noqa: F811 -- a second shape of the same defect
        return df.loc[:, "cadd_phred"]

    monkeypatch.setattr(VE, "engineer_features", engineer_features)
    with pytest.raises(AssertionError, match="df.loc"):
        Q.builder_input_names()


# -------------------------------------------------------- the numerical environment (CANONICAL-IDENTITY-ENVIRONMENT-1)
# Measured 2026-11-17: numpy's log10 bits differ between processor dispatch paths and platforms (27,950 of 1,028,224 real
# rows, at most 2 ULP), so the identity seals every column but af_log10, and af_log10 is certified against a reference.
def test_the_environment_dependent_columns_are_proven_to_be_exactly_af_log10():
    assert Q.environment_dependent_columns() == frozenset(Q.ENVIRONMENT_DEPENDENT) == frozenset({"af_log10"})
    assert "src/genomic_variant_classifier/models/feature_semantics.py::formula_af_log10" in Q.PROCEDURE


def _exp_builder(df, *, semantics=None):
    feats = df[["cadd_phred"]].copy()
    feats["cadd_phred"] = df["cadd_phred"].astype(float)
    feats["revel_score"] = np.exp(df["revel_score"])
    return feats


def _apply_builder(df, *, semantics=None):
    feats = df[["cadd_phred"]].copy()
    other = df["revel_score"].apply(np.log1p)
    feats["revel_score"] = other
    return feats


def _power_builder(df, *, semantics=None):
    feats = df[["cadd_phred"]].copy()
    feats["cadd_phred"] = df["cadd_phred"] ** 3
    return feats


@pytest.mark.parametrize("builder, message", [(_exp_builder, None), (_apply_builder, "outside"), (_power_builder, None)])
def test_a_transcendental_operation_in_another_column_is_named_or_refused(monkeypatch, builder, message):
    import genomic_variant_classifier.models.variant_ensemble as VE

    monkeypatch.setattr(VE, "engineer_features", builder)
    if message:                                     # a value passed outside an assignment: no column can be named
        with pytest.raises(AssertionError, match="outside a 'feats"):
            Q.environment_dependent_columns()
    else:                                           # inside an assignment: that column is named, so it differs from af_log10
        found = Q.environment_dependent_columns()
        assert found != frozenset(Q.ENVIRONMENT_DEPENDENT) and found - {"af_log10"}


def _shifted_log10(ulps: int):
    def formula(af):
        y = np.log10(af + 1e-8)
        for _ in range(ulps):
            y = np.nextafter(y, np.inf)
        return y
    return formula


def test_the_identity_does_not_depend_on_the_last_bits_of_log10(tmp_path, monkeypatch):
    import genomic_variant_classifier.models.feature_semantics as FS

    cm, _ = S.prepared(tmp_path / "here")
    monkeypatch.setattr(FS, "formula_af_log10", _shifted_log10(2))         # another machine's log10, 2 ULP away
    other, _ = S.prepared(tmp_path / "there")
    a, b = cm.features["af_log10"].to_numpy(), other.features["af_log10"].to_numpy()
    observed = ~np.isnan(a)
    assert not np.array_equal(a[observed].view(np.uint64), b[observed].view(np.uint64))   # the bits differ ...
    assert other.identity == cm.identity and other.receipt == cm.receipt                  # ... the identity does not
    assert Q.exact_column_digests(other.features) == Q.exact_column_digests(cm.features)
    assert "af_log10" not in Q.exact_column_digests(cm.features)


def test_a_log10_beyond_the_certified_bound_or_missing_elsewhere_is_refused(tmp_path, monkeypatch):
    import genomic_variant_classifier.models.feature_semantics as FS

    monkeypatch.setattr(FS, "formula_af_log10", _shifted_log10(8))
    with pytest.raises(ValueError, match="beyond the certified bound of 4.0 ULP"):
        S.prepared(tmp_path / "far")
    af = np.array([0.0, 0.25, np.nan])
    y = np.log10(np.array([0.0, 0.25, 0.5]) + 1e-8)
    with pytest.raises(ValueError, match="not missing exactly where af_raw is missing"):
        Q.certify_log10(af, y)                      # a value where af_raw is missing
    with pytest.raises(ValueError, match="not missing exactly where af_raw is missing"):
        Q.certify_log10(af, np.array([np.nan, y[1], np.nan]))      # missing where af_raw is observed


def test_the_certificate_measures_ulps_correctly():
    rng = np.random.default_rng(3)
    af = np.concatenate([[0.0, 1.0, 1e-8], 10 ** rng.uniform(-7, 0, 500)])
    y = np.log10(af + 1e-8)
    assert 0.0 <= Q.certify_log10(af, y) <= Q.LOG10_ULP_BOUND
    shifted = y.copy()
    for _ in range(10):
        shifted = np.nextafter(shifted, np.inf)
    with pytest.raises(ValueError, match="ULP from log10") as info:
        Q.certify_log10(af, shifted)
    worst = float(str(info.value).split(" is ")[1].split(" ULP")[0])
    # 10 ULP shifted: measured as 10, give or take this machine's own distance from the reference (at most the bound)
    assert 10.0 - Q.LOG10_ULP_BOUND <= worst <= 10.0 + Q.LOG10_ULP_BOUND


def test_reuse_recertifies_af_log10_and_reseals_the_exact_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cm, _ = S.prepared(tmp_path)

    def rewritten(column: str, ulps: int, name: str):
        """The stored payload edited and re-published under its new SHA-256 with a consistent record, as a forger who
        controls the store would: only the meaning can refuse it."""
        store = tmp_path / name                 # short names: the Windows path budget (|basetemp| + 185)
        stored = Q.write_canonical(cm, store)
        folder = store / stored.identity
        tpath, rpath = folder / f"{stored.payload_sha256}.parquet", folder / f"{stored.payload_sha256}.receipt.json"
        table = pq.read_table(tpath)
        v = table[column].to_numpy(zero_copy_only=False).astype(np.float64)
        for _ in range(ulps):
            v = np.where(np.isnan(v), v, np.nextafter(v, np.inf))
        table = table.set_column(table.column_names.index(column), column, pa.array(v, type=pa.float64()))
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf, compression="zstd")
        raw = buf.getvalue().to_pybytes()
        record = json.loads(rpath.read_text())
        record["payload"] = {"file": f"{C.sha256_bytes(raw)}.parquet", "sha256": C.sha256_bytes(raw), "bytes": len(raw)}
        new = C.sha256_bytes(raw)
        (folder / f"{new}.parquet").write_bytes(raw)
        (folder / f"{new}.receipt.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="ascii")
        return store, stored.identity, new

    store, ident, payload = rewritten("af_log10", 1, "one")
    with pytest.raises(ValueError, match="representation"):                   # the record still names the old bits
        Q.load_canonical(store, ident, payload)
    store, ident, payload = rewritten("af_log10", 9, "nine")
    with pytest.raises(ValueError, match="beyond the certified bound"):
        Q.load_canonical(store, ident, payload)
    store, ident, payload = rewritten("af_raw", 1, "exact")
    with pytest.raises(ValueError, match="exact-column digest"):
        Q.load_canonical(store, ident, payload)


def test_the_preparation_refuses_when_the_proof_names_another_column(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "environment_dependent_columns", lambda: frozenset({"af_log10", "cadd_phred"}))
    with pytest.raises(AssertionError, match="not the declared"):
        S.prepared(tmp_path)
