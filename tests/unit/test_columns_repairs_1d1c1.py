"""Stage 1D-1c.1A (set "at"): executable witnesses for the four columns.py defects the owner's ruling 482cd291 named, written
before the repair. Each test FAILS on main 6f5ba69e and passes after it. Author: Monzia Moodie.

1. COLUMNS-POPULATION-DIGEST-COLLISION-1: the v1 population digest joins identities with newlines (so ("a\\nb",) and
   ("a", "b") collide) and does not bind the population identifier.
2. COLUMNS-MUTABLE-ARRAYS-1: a validated column's arrays could be written after validation, so its digest and its
   invariants could go stale.
3. COLUMNS-DERIVATION-LINEAGE-1: a derived column's identity named only "derived:<name>", not its inputs or its rule.
4. COLUMNS-CALLER-SUPPLIED-DERIVATION-1: ``derive`` took its function and propagation rule from the caller.
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import columns as K
from genomic_variant_classifier.information_sources import observation as O

SHA, SHB = "a" * 64, "b" * 64
S = O.ObservationState
POP = K.Population("pop:1", ("r1", "r2", "r3", "r4"))


def _col(feature, pairs, sha=SHA, pop=POP):
    return K.ObservationColumn.from_rows(feature, O.SourceIdentity(f"dataset:{feature}", sha), pop,
                                         dict(zip(pop.row_ids, pairs)))


# ----------------------------------------------------------------------------------- 1. population identity, version 2
def test_v2_population_identity_has_no_newline_collision_and_binds_the_identifier():
    assert K.Population("p", ("a\nb",)).sha256 != K.Population("p", ("a", "b")).sha256
    assert K.Population("p", ("a", "b")).sha256 != K.Population("q", ("a", "b")).sha256
    assert K.Population("p", ("a", "b")).sha256 != K.Population("p", ("b", "a")).sha256          # order still binds
    want = C.sha256_bytes(C.canonical_bytes({"schema": K.POPULATION_IDENTITY_SCHEMA, "population_id": "p",
                                             "row_ids": ["a", "b"]}))
    assert K.Population("p", ("a", "b")).sha256 == want
    assert K.POPULATION_IDENTITY_SCHEMA == "gvc.population-order/2"


def test_v1_population_digest_stays_verifiable_for_sealed_records_and_refuses_what_it_cannot_represent():
    pop = K.Population("p", ("a", "b"))
    assert pop.sha256_v1 == hashlib.sha256(b"a\nb\n").hexdigest()            # the 1D-1b.2 formula, unchanged
    assert pop.sha256_v1 == K.Population("other", ("a", "b")).sha256_v1        # v1 never bound the label
    with pytest.raises(ValueError, match="newline"):
        _ = K.Population("p", ("a\nb",)).sha256_v1


def test_row_identities_are_exact_strings():
    class Sub(str):
        pass
    with pytest.raises(ValueError, match="strings"):
        K.Population("p", (Sub("a"),))


# ---------------------------------------------------------------------------------------------- 2. immutable arrays
def test_a_column_owns_read_only_copies_of_its_arrays():
    v = np.array([0.5, np.nan, 0.0, np.nan])
    s = np.array([K.CODE[S.OBSERVED], K.CODE[S.NOT_IN_SOURCE], K.CODE[S.OBSERVED], K.CODE[S.NOT_IN_SOURCE]], np.uint8)
    d = np.zeros(4, np.uint16)
    col = K.ObservationColumn("eve_score", O.SourceIdentity("dataset:eve", SHA), POP, v, s, d, ("",))
    before = col.sha256
    for arr in (col.values, col.states, col.details):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 1
        with pytest.raises(ValueError):
            arr.flags.writeable = True
    v[0], s[1] = 9.0, K.CODE[S.OBSERVED]                        # the caller's arrays are not the column's
    assert col.values[0] == 0.5 and col.states[1] == K.CODE[S.NOT_IN_SOURCE] and col.sha256 == before


def test_non_array_inputs_are_refused_before_any_copy():
    s = np.full(4, K.CODE[S.NOT_IN_SOURCE], np.uint8)
    with pytest.raises(ValueError, match="float64"):
        K.ObservationColumn("eve_score", O.SourceIdentity("dataset:eve", SHA), POP, [np.nan] * 4, s)


# ---------------------------------------------------------------------------------- 3. derivation lineage in identity
def test_a_derived_column_binds_its_inputs_and_rule():
    a = _col("af_raw", [(S.OBSERVED, 0.0), (S.OBSERVED, 1e-5), (S.NOT_IN_SOURCE, None), (S.OBSERVED, 0.5)], SHA)
    b = _col("af_raw", [(S.OBSERVED, 0.0), (S.OBSERVED, 2e-5), (S.NOT_IN_SOURCE, None), (S.OBSERVED, 0.7)], SHB)
    da, db = K.derive("af_is_ultra_rare", [a]), K.derive("af_is_ultra_rare", [b])
    assert np.array_equal(da.column.values, db.column.values, equal_nan=True)     # equal values ...
    assert da.column.lineage != db.column.lineage and da.column.sha256 != db.column.sha256   # ... different lineage
    rare = K.derive("af_is_rare", [a])
    assert rare.column.lineage != da.column.lineage                               # the rule is bound
    assert da.column.lineage == K.derivation_identity("af_is_ultra_rare", [a])


def test_lineage_is_required_exactly_for_derived_sources():
    v, s = np.full(4, np.nan), np.full(4, K.CODE[S.INPUT_UNOBSERVED], np.uint8)
    with pytest.raises(ValueError, match="lineage"):
        K.ObservationColumn("x", O.SourceIdentity("derived:af_is_rare"), POP, v, s)
    with pytest.raises(ValueError, match="lineage"):
        K.ObservationColumn("eve_score", O.SourceIdentity("dataset:eve", SHA), POP, np.full(4, np.nan),
                            np.full(4, K.CODE[S.NOT_IN_SOURCE], np.uint8), lineage=SHA)


def test_a_non_derived_column_digest_is_unchanged_by_the_lineage_field():
    col = _col("eve_score", [(S.OBSERVED, 0.5), (S.NOT_IN_SOURCE, None), (S.OBSERVED, 0.0), (S.NOT_IN_SOURCE, None)])
    assert col.sha256_v1 == K.column_digest(col, population_sha256=col.population.sha256_v1)
    assert col.sha256 == K.column_digest(col, population_sha256=col.population.sha256)
    assert col.sha256 != col.sha256_v1                                            # only the population identity moved


# ------------------------------------------------------------------------------------ 4. derivation by registered name
def test_derive_takes_its_implementation_from_the_registry_only():
    params = list(inspect.signature(K.derive).parameters)
    assert params == ["name", "inputs"]
    af = _col("af_raw", [(S.OBSERVED, 0.0), (S.NOT_IN_SOURCE, None), (S.BAD_MAPPING, None), (S.OBSERVED, 1e-5)])
    d = K.derive("af_is_ultra_rare", [af])
    assert list(d.column.values[[0, 3]]) == [1.0, 1.0] and np.isnan(d.column.values[1:3]).all()
    with pytest.raises(ValueError, match="not a registered derived feature"):
        K.derive("no_such_feature", [af])
    with pytest.raises(ValueError, match="model output"):
        K.derive("gnn_score", [af])
    with pytest.raises(ValueError, match="retired"):
        K.derive("gene_has_known_disease", [_col("n_pathogenic_in_gene", [(S.OBSERVED, 1)] * 4)])
    with pytest.raises(ValueError, match="declared inputs"):
        K.derive("af_is_ultra_rare", [_col("cadd_phred", [(S.OBSERVED, 1)] * 4)])
    with pytest.raises(ValueError, match="declared inputs"):
        K.derive("af_is_ultra_rare", [af, af])


def test_every_registered_implementation_equals_the_production_formula_on_observed_values():
    """The production expressions, copied verbatim from models/variant_ensemble.py engineer_features (lines 795-874,
    927-929) and data/finngen.py (lines 149-151) at 6f5ba69e, evaluated on observed values only."""
    rng = np.random.default_rng(20261115)
    n = 4000
    pop = K.Population("pop:parity", tuple(f"r{i}" for i in range(n)))
    def col(feature, values):
        return K.ObservationColumn(feature, O.SourceIdentity(f"dataset:{feature}", SHA), pop,
                                   np.asarray(values, np.float64), np.full(n, K.CODE[S.OBSERVED], np.uint8))
    af = np.concatenate([np.zeros(500), 10 ** rng.uniform(-7, 0, n - 500)])
    af[:8] = [1e-4, 1e-3, 0.01, 0.0001 - 1e-12, 0.001 - 1e-12, 0.01 - 1e-12, 1.0, 0.0]   # the thresholds themselves
    expected = {
        "af_log10": np.log10(af + 1e-8), "af_is_absent": (af == 0).astype(int),
        "af_is_ultra_rare": (af < 0.0001).astype(int), "af_is_rare": ((af >= 0.0001) & (af < 0.001)).astype(int),
        "af_is_common": (af >= 0.01).astype(int)}
    for name, want in expected.items():
        got = K.derive(name, [col("af_raw", af)]).column.values
        assert np.array_equal(got, want.astype(np.float64)), name
    for name, feature, lo, hi, formula in (
            ("cadd_high", "cadd_phred", 0, 60, lambda v: v >= 20),
            ("sift_deleterious", "sift_score", 0, 1, lambda v: v < 0.05),
            ("polyphen_probably_damaging", "polyphen2_score", 0, 1, lambda v: v >= 0.908),
            ("revel_pathogenic", "revel_score", 0, 1, lambda v: v >= 0.5),
            ("gene_is_constrained", "loeuf", 0, 2, lambda v: v < 0.35)):
        v = rng.uniform(lo, hi, n)
        v[:3] = [20, 0.05, 0.908][:3] if feature != "loeuf" else [0.35, 0.3499999, 0.35000001]
        assert np.array_equal(K.derive(name, [col(feature, v)]).column.values, formula(v).astype(np.float64)), name
    ins, dele = rng.integers(0, 2, n).astype(float), rng.integers(0, 2, n).astype(float)
    got = K.derive("is_indel", [col("is_insertion", ins), col("is_deletion", dele)]).column.values
    assert np.array_equal(got, ((ins.astype(int) | dele.astype(int))).astype(np.float64))
    fin, nfe = rng.uniform(0, 0.01, n), np.concatenate([np.zeros(10), rng.uniform(0, 0.01, n - 10)])
    for name, a_, b_ in (("finngen_enrichment", "finngen_af_fin", "finngen_af_nfsee"),
                         ("finngen_r13_enrichment", "finngen_r13_af_fin", "finngen_r13_af_nfsee")):
        got = K.derive(name, [col(a_, fin), col(b_, nfe)]).column.values
        assert np.array_equal(got, np.clip(fin / (nfe + 1e-9), None, 1000.0)), name
    tools = [K.derive(t, [col(f, v)]).column for t, f, v in (
        ("cadd_high", "cadd_phred", rng.uniform(0, 60, n)), ("sift_deleterious", "sift_score", rng.uniform(0, 1, n)),
        ("polyphen_probably_damaging", "polyphen2_score", rng.uniform(0, 1, n)),
        ("revel_pathogenic", "revel_score", rng.uniform(0, 1, n)))]
    d = K.derive("n_tools_pathogenic", tools)
    assert np.array_equal(d.column.values, np.sum([t.values for t in tools], axis=0)) and (d.inputs_observed == 4).all()
    assert not d.inputs_observed.flags.writeable


def test_every_rule_has_a_module_level_implementation_bound_into_lineage():
    for name, rule in K.DERIVED_RULES.items():
        if rule.retired:
            continue
        impls = rule.implementation if isinstance(rule.implementation, tuple) else (rule.implementation,)
        for qual in impls:
            fn = getattr(K, qual)
            assert callable(fn) and fn.__module__ == K.__name__ and fn.__qualname__ == qual, (name, qual)
