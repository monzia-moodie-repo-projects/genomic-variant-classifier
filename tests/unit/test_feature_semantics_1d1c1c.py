"""Stage 1D-1c.1C: feature-semantics epochs and the registered allele-frequency family (owner ruling aa999b96).

The epoch-2 builder must (1) keep every non-AF feature identical to epoch 1, (2) make the AF family three-valued in all
six features, (3) compute it with formulas proven identical -- in syntax tree and bit for bit -- to the REGISTERED
derivations (columns.DERIVED_RULES), and (4) refuse -- never repair -- a frequency that is absent, non-numeric,
infinite or outside [0, 1].

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import columns as K
from genomic_variant_classifier.models.feature_semantics import (
    AF_FAMILY,
    PROJECTABLE_FAMILIES,
    RENAMED_IN_V2,
    DerivationFamily,
    FeatureSemantics as FS,
    census_identity,
    derive_family,
    infer_semantics,
    roster_without_family,
    tabular_features,
)
from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

V1, V2 = FS.V1_LEGACY, FS.V2_OBSERVED_AF


def _raw(af) -> pd.DataFrame:
    n = len(af)
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "allele_freq": af, "ref": rng.choice(list("ACGT"), n), "alt": rng.choice(["A", "C", "GT", "T"], n),
        "consequence": rng.choice(["missense_variant", "stop_gained", "synonymous_variant"], n),
        "cadd_phred": rng.normal(15, 5, n), "loeuf": rng.random(n), "gene_constraint_oe": rng.random(n),
        "chrom": rng.choice(["1", "X", "MT"], n)})


def test_epoch_two_is_epoch_one_with_the_rename_in_place():
    v1, v2 = tabular_features(V1), tabular_features(V2)
    assert v1 == tuple(TABULAR_FEATURES) and len(v2) == len(v1)
    diff = [(a, b) for a, b in zip(v1, v2) if a != b]
    assert diff == [("af_is_absent", "af_is_observed_zero")]
    assert dict(RENAMED_IN_V2) == {"af_is_observed_zero": "af_is_absent"}


def test_census_identity_translates_explicitly_and_refuses_the_other_epochs_name():
    assert census_identity("af_is_observed_zero", V2) == "af_is_absent"
    assert census_identity("af_raw", V2) == "af_raw" and census_identity("af_is_absent", V1) == "af_is_absent"
    with pytest.raises(ValueError):
        census_identity("af_is_absent", V2)
    with pytest.raises(ValueError):
        census_identity("af_is_observed_zero", V1)
    with pytest.raises(ValueError):
        census_identity("not_a_feature", V1)


def test_infer_semantics_refuses_a_mixture():
    assert infer_semantics(["af_raw", "af_is_absent"]) is V1
    assert infer_semantics(["af_raw", "af_is_observed_zero"]) is V2
    assert infer_semantics(["x", "y"]) is None
    with pytest.raises(ValueError, match="mix"):
        infer_semantics(["af_is_absent", "af_is_observed_zero"])


def _returned_expression(fn) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    (body,) = tree.body[0].body                    # exactly one statement: the return
    assert isinstance(body, ast.Return)
    return ast.dump(body.value)


#: Every value class the formulas distinguish: zero, sub-normal, each threshold and its neighbours, one.
_EDGES = np.array([0.0, 5e-324, 1e-12, np.nextafter(1e-4, 0), 1e-4, np.nextafter(1e-4, 1), 5e-4,
                   np.nextafter(1e-3, 0), 1e-3, np.nextafter(1e-3, 1), 0.0099, np.nextafter(0.01, 0), 0.01,
                   np.nextafter(0.01, 1), 0.5, np.nextafter(1.0, 0), 1.0])


@pytest.mark.parametrize("family", [f for fams in PROJECTABLE_FAMILIES.values() for f in fams],
                         ids=lambda f: f.family_id)
def test_every_projectable_family_is_exactly_the_registered_strict_dependents_of_its_root(family):
    root = census_identity(family.root, family.semantics)
    registered = {k for k, r in K.DERIVED_RULES.items()
                  if r.inputs == (root,) and r.propagation is K.Propagation.STRICT and not r.retired}
    assert set(family.rule_map().values()) == registered
    rng = np.random.default_rng(11)
    values = np.concatenate([_EDGES, rng.random(10_000), rng.random(10_000) * 1e-3])
    for dependent, key in family.rule_map().items():
        registered_fn = getattr(K, K.DERIVED_RULES[key].implementation)
        formula = family.implementation(dependent)
        assert formula is not registered_fn
        assert _returned_expression(formula) == _returned_expression(registered_fn), dependent
        got, want = formula(values), registered_fn(values)
        assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), dependent
    assert PROJECTABLE_FAMILIES[V1] == () and PROJECTABLE_FAMILIES[V2] == (AF_FAMILY,)


def test_a_malformed_family_is_refused():
    with pytest.raises(ValueError, match="distinct"):
        DerivationFamily("bad", V2, "af_raw", ("af_raw", "af_log10"), {"af_log10": "af_log10"},
                         {"af_log10": "formula_af_log10"}, {"af_raw": "af_log10", "af_log10": "m_log"}, (0.0, 1.0))
    with pytest.raises(ValueError, match="not a formula"):
        DerivationFamily("bad", V2, "af_raw", ("af_raw", "af_log10"), {"af_log10": "af_log10"},
                         {"af_log10": "derive_family"}, {"af_raw": "m_raw", "af_log10": "m_log"}, (0.0, 1.0))
    with pytest.raises(ValueError, match="a rule and a formula"):
        DerivationFamily("bad", V2, "af_raw", ("af_raw", "af_log10"), {"af_log10": "af_log10"},
                         {}, {"af_raw": "m_raw", "af_log10": "m_log"}, (0.0, 1.0))
    with pytest.raises(ValueError, match="census identity"):
        DerivationFamily("bad", V2, "af_raw", ("af_raw", "af_log10"), {"af_log10": "cadd_high"},
                         {"af_log10": "formula_af_log10"}, {"af_raw": "m_raw", "af_log10": "m_log"}, (0.0, 1.0))


@pytest.mark.parametrize("value,flags", [(0.0, (1, 1, 0, 0)), (1e-12, (0, 1, 0, 0)), (9.99999e-5, (0, 1, 0, 0)),
                                         (1e-4, (0, 0, 1, 0)), (9.99e-4, (0, 0, 1, 0)), (1e-3, (0, 0, 0, 0)),
                                         (0.0099, (0, 0, 0, 0)), (0.01, (0, 0, 0, 1)), (1.0, (0, 0, 0, 1))])
def test_thresholds_are_exact(value, flags):
    d = derive_family(AF_FAMILY, np.array([value]))
    got = tuple(int(d[m][0]) for m in ("af_is_observed_zero", "af_is_ultra_rare", "af_is_rare", "af_is_common"))
    assert got == flags
    assert d["af_log10"][0] == np.log10(value + 1e-8)


def test_an_unobserved_frequency_is_nan_in_all_six():
    d = derive_family(AF_FAMILY, np.array([np.nan, 0.0]))
    assert all(np.isnan(d[m][0]) for m in AF_FAMILY.members)
    assert d["af_is_observed_zero"][1] == 1.0


@pytest.mark.parametrize("bad", [[np.inf], [-np.inf], [-1e-9], [1.0000001]])
def test_an_invalid_observed_frequency_is_refused_never_clipped(bad):
    with pytest.raises(ValueError):
        derive_family(AF_FAMILY, np.array(bad))


def test_epoch_two_builder_matches_epoch_one_everywhere_except_the_af_family():
    af = np.array([0.0, np.nan, 3e-5, 2e-4, 0.05, np.nan, 1.0])
    v1 = engineer_features(_raw(af))
    v2 = engineer_features(_raw(af), semantics=V2)
    assert tuple(v2.columns) == tabular_features(V2)
    family = list(AF_FAMILY.members)
    pd.testing.assert_frame_equal(v1.drop(columns=list(v1.columns[:6])), v2.drop(columns=family))
    observed = ~np.isnan(af)
    for c1, c2 in zip(v1.columns[:6], family):
        assert np.array_equal(v1[c1].to_numpy(dtype=float)[observed], v2[c2].to_numpy(dtype=float)[observed])
        assert v2[c2][~observed].isna().all()
    # the epoch-1 builder still asserts an observed zero for the unobserved rows: the defect epoch 2 removes
    assert (v1["af_is_absent"][~observed] == 1).all()


def test_epoch_one_default_is_byte_identical_to_explicit_epoch_one():
    raw = _raw(np.array([0.0, np.nan, 0.2]))
    pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(raw, semantics=V1))


@pytest.mark.parametrize("values", [["0.1", 0.2, None], [True, False, True]])
def test_the_epoch_two_builder_refuses_a_non_numeric_frequency(values):
    raw = _raw(np.zeros(3)).assign(allele_freq=pd.Series(values, dtype=object))
    with pytest.raises(TypeError):
        engineer_features(raw, semantics=V2)


def test_the_epoch_two_builder_refuses_an_absent_frequency_column():
    with pytest.raises(ValueError, match="allele_freq column"):
        engineer_features(_raw(np.zeros(3)).drop(columns="allele_freq"), semantics=V2)


def test_the_epoch_two_builder_reads_nullable_floats():
    raw = _raw(np.zeros(3)).assign(allele_freq=pd.array([0.1, None, 0.0], dtype="Float64"))
    v2 = engineer_features(raw, semantics=V2)
    assert np.isnan(v2["af_raw"][1]) and v2["af_is_observed_zero"][2] == 1.0


def test_the_builder_refuses_a_semantics_that_is_not_the_enum():
    with pytest.raises(TypeError):
        engineer_features(_raw(np.zeros(2)), semantics="gvc.features/2")


def test_the_family_is_closed_so_removing_it_leaves_no_dependent_behind():
    """The AF-free arm removes the family only. That is complete exactly when no registered derivation outside the
    family takes a family member as input -- measured here on the registry, not assumed."""
    family = {census_identity(m, V2) for m in AF_FAMILY.members}
    outside = {k: r.inputs for k, r in K.DERIVED_RULES.items() if k not in family and set(r.inputs) & family}
    assert outside == {}


def test_the_af_free_arm_removes_the_whole_family_and_nothing_else():
    roster = tabular_features(V2)
    kept = roster_without_family(roster, AF_FAMILY)
    assert set(roster) - set(kept) == set(AF_FAMILY.members)
    assert [n for n in roster if n in kept] == list(kept)
    with pytest.raises(ValueError, match="complete"):
        roster_without_family(tuple(n for n in roster if n != "af_is_rare"), AF_FAMILY)
