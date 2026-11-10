"""Stage 1D-1b.2 (set "aq"): logical against materialised source identity, population-reconciled columns, and the
propagation rule of every derived feature. Owner ruling 7f4b69d0 (2026-11-09), sections 4 and 5. Author: Monzia Moodie."""
from __future__ import annotations

import fractions
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import columns as K
from genomic_variant_classifier.information_sources import observation as O

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 64
S = O.ObservationState
EVE = O.SourceIdentity("dataset:eve", SHA, "EVE_all_data")
EVE_ABSENT = O.SourceIdentity("dataset:eve")
POP = K.Population("pop:1", ("r1", "r2", "r3", "r4"))


# ------------------------------------------------------------------------------------------------ source identities
def test_the_logical_source_exists_without_bytes_and_the_digest_only_with_them():
    assert EVE_ABSENT.content_sha256 is None and not EVE_ABSENT.is_derived
    for bad in ("eve", "dataset:", "dataset:EVE", "file:/x", "derived:"):
        with pytest.raises(ValueError, match="logical source"):
            O.SourceIdentity(bad)
    for bad in ("A" * 64, "a" * 63, "x"):
        with pytest.raises(ValueError, match="content identity"):
            O.SourceIdentity("dataset:eve", bad)
    with pytest.raises(ValueError, match="derived value has no source bytes"):
        O.SourceIdentity("derived:revel_pathogenic", SHA)
    with pytest.raises(ValueError, match="release"):
        O.SourceIdentity("dataset:eve", SHA, " ")


def test_the_state_decides_the_digest_by_the_owners_table():
    assert set(O.SOURCE_RULE) == set(S) - {S.INPUT_UNOBSERVED}
    for state in (S.OBSERVED, S.NOT_IN_SOURCE, S.UNSUPPORTED_VARIANT):
        with pytest.raises(ValueError, match="must name the SHA-256"):
            O.NumericObservation("eve_score", "r1", state, 0.5 if state is S.OBSERVED else None, EVE_ABSENT)
    with pytest.raises(ValueError, match="asserts no bytes were read"):
        O.absent("eve_score", "r1", S.SOURCE_ABSENT, EVE)
    assert O.absent("eve_score", "r1", S.SOURCE_ABSENT, EVE_ABSENT).source.logical_id == "dataset:eve"
    for state in (S.BAD_MAPPING, S.COMPUTATION_FAILED):              # depends on the failure stage: either
        assert O.absent("eve_score", "r1", state, EVE).state is state
        assert O.absent("eve_score", "r1", state, EVE_ABSENT).state is state
    with pytest.raises(ValueError, match="derived state"):
        O.absent("eve_score", "r1", S.INPUT_UNOBSERVED, EVE)


def test_a_derived_value_takes_only_derived_states():
    d = O.SourceIdentity("derived:af_is_ultra_rare")
    assert O.observed("af_is_ultra_rare", "r1", 1, d).value == 1.0
    assert O.absent("af_is_ultra_rare", "r1", S.INPUT_UNOBSERVED, d).value is None
    for state in (S.NOT_IN_SOURCE, S.SOURCE_ABSENT, S.UNSUPPORTED_VARIANT, S.BAD_MAPPING):
        with pytest.raises(ValueError, match="derived value cannot be"):
            O.absent("af_is_ultra_rare", "r1", state, d)


def test_an_observed_value_is_a_finite_real_number_never_a_boolean_nor_a_sentinel_on_absence():
    for flag in (True, False, np.bool_(True)):
        with pytest.raises(ValueError, match="not a boolean"):
            O.observed("x", "r1", flag, EVE)
    for bad in (float("nan"), float("inf"), np.float64("-inf")):
        with pytest.raises(ValueError, match="finite"):
            O.observed("x", "r1", bad, EVE)
    for bad in ("0.5", complex(1, 0)):
        with pytest.raises(ValueError, match="real number"):
            O.observed("x", "r1", bad, EVE)
    a, b = O.observed("x", "r1", np.float32(0.25), EVE), O.observed("x", "r1", fractions.Fraction(1, 4), EVE)
    assert a == b and type(a.value) is float
    assert O.observed("x", "r1", 0, EVE).is_observed                   # a genuine zero is observed
    for state in set(S) - {S.OBSERVED, S.INPUT_UNOBSERVED}:
        for sentinel in (0.0, 0.5, -1):
            with pytest.raises(ValueError, match="no value"):
                O.NumericObservation("x", "r1", state, sentinel, EVE_ABSENT if state is S.SOURCE_ABSENT else EVE)
    with pytest.raises(ValueError, match="other than OBSERVED"):
        O.absent("x", "r1", S.OBSERVED, EVE)
    with pytest.raises(ValueError, match="non-empty"):
        O.observed("", "r1", 0.5, EVE)


# ------------------------------------------------------------------------------------------------------- populations
def test_a_population_is_ordered_and_unique():
    with pytest.raises(ValueError, match="1 duplicated"):
        K.Population("p", ("a", "b", "a"))
    with pytest.raises(ValueError, match="non-empty tuple"):
        K.Population("p", ())
    assert K.Population("p", ("a", "b")).sha256 != K.Population("p", ("b", "a")).sha256      # order is identity
    assert K.Population("p", ("a", "b")).sha256 == K.Population("q", ("a", "b")).sha256      # the rows, not the label


def test_a_column_reconciles_with_its_population_exactly():
    rows = {"r1": (S.OBSERVED, 0.5), "r2": (S.NOT_IN_SOURCE, None), "r3": (S.OBSERVED, 0.0), "r4": (S.BAD_MAPPING, None)}
    col = K.ObservationColumn.from_rows("eve_score", EVE, POP, rows)
    counts = col.counts()
    assert counts["observed"] == 2 and counts["not_in_source"] == 1 and sum(counts.values()) == 4
    assert col.values[0] == 0.5 and np.isnan(col.values[1])           # a genuine 0.5 and an absent one, told apart
    with pytest.raises(ValueError, match="1 missing, 0 extra"):
        K.ObservationColumn.from_rows("eve_score", EVE, POP, {k: v for k, v in rows.items() if k != "r4"})
    with pytest.raises(ValueError, match="0 missing, 1 extra"):
        K.ObservationColumn.from_rows("eve_score", EVE, POP, {**rows, "r9": (S.OBSERVED, 1.0)})


def test_a_column_refuses_every_contract_breach_vectorised():
    good_v = np.array([0.5, np.nan, 0.0, np.nan])
    good_s = np.array([K.CODE[S.OBSERVED], K.CODE[S.NOT_IN_SOURCE], K.CODE[S.OBSERVED], K.CODE[S.NOT_IN_SOURCE]], np.uint8)
    K.ObservationColumn("eve_score", EVE, POP, good_v, good_s)
    with pytest.raises(ValueError, match="sentinel"):
        K.ObservationColumn("eve_score", EVE, POP, np.array([0.5, 0.5, 0.0, np.nan]), good_s)
    with pytest.raises(ValueError, match="finite"):
        K.ObservationColumn("eve_score", EVE, POP, np.array([np.inf, np.nan, 0.0, np.nan]), good_s)
    with pytest.raises(ValueError, match="state code"):
        K.ObservationColumn("eve_score", EVE, POP, good_v, np.array([0, 2, 0, 99], np.uint8))
    with pytest.raises(ValueError, match="float64"):
        K.ObservationColumn("eve_score", EVE, POP, good_v.astype(np.float32), good_s)
    with pytest.raises(ValueError, match="must name the SHA-256"):        # observed rows need the bytes' digest
        K.ObservationColumn("eve_score", EVE_ABSENT, POP, good_v, good_s)
    absent_s = np.full(4, K.CODE[S.SOURCE_ABSENT], np.uint8)
    K.ObservationColumn("eve_score", EVE_ABSENT, POP, np.full(4, np.nan), absent_s)
    with pytest.raises(ValueError, match="asserts no bytes"):
        K.ObservationColumn("eve_score", EVE, POP, np.full(4, np.nan), absent_s)
    with pytest.raises(ValueError, match="detail vocabulary"):
        K.ObservationColumn("eve_score", EVE, POP, good_v, good_s, None, ("", "x"))
    with pytest.raises(ValueError, match="unknown detail code"):
        K.ObservationColumn("eve_score", EVE, POP, good_v, good_s, np.array([0, 3, 0, 0], np.uint16), ("", "x"))


def test_detail_counts_and_column_identity():
    s = np.array([K.CODE[S.OBSERVED], K.CODE[S.NOT_IN_SOURCE], K.CODE[S.NOT_IN_SOURCE], K.CODE[S.OBSERVED]], np.uint8)
    v = np.array([0.5, np.nan, np.nan, 0.2])
    d = np.array([0, 1, 2, 0], np.uint16)
    col = K.ObservationColumn("eve_score", EVE, POP, v, s, d, ("", "gene_not_modeled", "substitution_unscored"))
    assert col.detail_counts() == {"observed": {"": 2}, "not_in_source": {"gene_not_modeled": 1, "substitution_unscored": 1}}
    other = K.ObservationColumn("eve_score", EVE, POP, v.copy(), s.copy(), np.array([0, 2, 1, 0], np.uint16),
                                ("", "gene_not_modeled", "substitution_unscored"))
    assert col.sha256 != other.sha256                                  # the reasons are part of the identity


def test_a_million_row_column_validates_in_vectorised_form():
    n = 1_000_000
    pop = K.Population("pop:big", tuple(f"r{i}" for i in range(n)))
    states = np.where(np.arange(n) % 3 == 0, K.CODE[S.NOT_IN_SOURCE], K.CODE[S.OBSERVED]).astype(np.uint8)
    vals = np.where(states == K.CODE[S.OBSERVED], 0.25, np.nan)
    col = K.ObservationColumn("af_raw", O.SourceIdentity("dataset:gnomad_af", SHA), pop, vals, states)
    assert col.counts()["not_in_source"] == 333_334


# -------------------------------------------------------------------------------------------------------- derivation
def _col(name, pairs):
    return K.ObservationColumn.from_rows(name, O.SourceIdentity(f"dataset:{name}", SHA), POP, dict(zip(POP.row_ids, pairs)))


def test_n_tools_pathogenic_counts_only_observed_tools_and_says_how_many():
    revel = _col("dbnsfp", [(S.OBSERVED, 0.8), (S.NOT_IN_SOURCE, None), (S.NOT_IN_SOURCE, None), (S.OBSERVED, 0.2)])
    sift = _col("dbnsfp", [(S.OBSERVED, 0.01), (S.OBSERVED, 0.01), (S.NOT_IN_SOURCE, None), (S.OBSERVED, 0.9)])
    d = K.derive("n_tools_pathogenic", [revel, sift],
                 [lambda v: (v >= 0.5).astype(float), lambda v: (v < 0.05).astype(float)], K.Propagation.PARTIAL_COUNT)
    assert list(d.inputs_observed) == [2, 1, 0, 2] and d.n_inputs == 2
    assert d.column.values[0] == 2 and d.column.values[1] == 1 and np.isnan(d.column.values[2]) and d.column.values[3] == 0
    assert d.column.states[2] == K.CODE[S.INPUT_UNOBSERVED] and d.column.source.logical_id == "derived:n_tools_pathogenic"
    # The CURRENT builder fills an absent REVEL with 0.5 and calls it pathogenic: r2 would count 2 (not 1) and r3 1 (not
    # unobserved) -- ABSENCE-AS-EVIDENCE-1.
    with pytest.raises(ValueError, match="0 or 1"):
        K.derive("n", [revel], [lambda v: v], K.Propagation.PARTIAL_COUNT)


def test_strict_propagation_marks_any_unobserved_input():
    af = _col("gnomad_af", [(S.OBSERVED, 0.0), (S.NOT_IN_SOURCE, None), (S.BAD_MAPPING, None), (S.OBSERVED, 1e-5)])
    d = K.derive("af_is_ultra_rare", [af], lambda v: (v < 1e-4).astype(float), K.Propagation.STRICT)
    assert d.column.values[0] == 1 and d.column.values[3] == 1 and np.isnan(d.column.values[1:3]).all()
    other = K.ObservationColumn.from_rows("b", O.SourceIdentity("dataset:b", SHA), K.Population("p2", ("r1",)),
                                          {"r1": (S.OBSERVED, 1.0)})
    with pytest.raises(ValueError, match="different populations"):
        K.derive("x", [af, other], lambda a, b: a + b, K.Propagation.STRICT)
    with pytest.raises(ValueError, match="one function"):
        K.derive("x", [af], [lambda v: v], K.Propagation.STRICT)


def test_every_derived_feature_of_the_census_has_exactly_one_declared_rule():
    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    derived = {e["name"]: tuple(e["depends_on"]) for e in ledger["entries"] if e["depends_on"]}
    assert len(derived) == 17
    assert set(K.DERIVED_RULES) | K.MODEL_OUTPUTS == set(derived) and not set(K.DERIVED_RULES) & K.MODEL_OUTPUTS
    for name, rule in K.DERIVED_RULES.items():
        assert rule.inputs == derived[name], name            # the rule's inputs ARE the census's declared dependencies
    assert K.DERIVED_RULES["n_tools_pathogenic"].propagation is K.Propagation.PARTIAL_COUNT
    assert all(r.propagation is K.Propagation.STRICT for n, r in K.DERIVED_RULES.items() if n != "n_tools_pathogenic")


def test_the_scalar_contract_imports_without_numpy():
    code = ("import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in {'numpy', 'pandas', 'torch', 'sklearn', 'scipy'}:\n"
            "            raise ImportError('blocked: ' + name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "from genomic_variant_classifier.information_sources import observation as O, lineage as L\n"
            "o = O.observed('x', 'r', 0.5, O.SourceIdentity('dataset:eve', 'a' * 64))\n"
            "assert o.value == 0.5 and L.dependency_order({'a': ()}) == ('a',)\n"
            "print('ok')\n")
    env = {"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""}
    done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
