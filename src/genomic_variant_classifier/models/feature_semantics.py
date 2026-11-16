"""Stage 1D-1c.1C: feature-semantics epochs and the registered derivation families the member projection operates on.

Owner ruling aa999b96 (2026-11-16), audited in AUDIT_OF_RULING_1D1c1C_2026-11-16.md (refinements R2 and R6).

Two epochs, never mixed:

* ``FeatureSemantics.V1_LEGACY`` (``gvc.features/1``) -- the historical builder. A missing allele frequency is filled with 0.0
  BEFORE the derived flags are computed, so ``af_is_absent`` means "zero after filling" and an unobserved frequency is
  indistinguishable from an observed zero. Every artifact fitted before this stage has this identity; it is kept exactly
  for historical replay and is never re-labelled.
* ``FeatureSemantics.V2_OBSERVED_AF`` (``gvc.features/2``) -- the qualified builder. The allele-frequency family is
  three-valued: NaN wherever the frequency was not observed, in every member of the family. ``af_is_absent`` is renamed
  ``af_is_observed_zero`` because it says "observed, and zero". The rename is a versioned translation bound to the
  census identity (``census_identity``), so the sealed feature census is not regenerated and a legacy artifact cannot
  render epoch-2 input without its roster differing.

A ``DerivationFamily`` is one source quantity (its root) and the features registered as STRICT functions of it in
``information_sources.columns.DERIVED_RULES``. Its formulas are the ``formula_*`` functions below: the epoch-2 builder
(``derive_family``) and the member projection call the SAME functions, so production has one formula per feature. They
are not imported from columns.py, by measurement: the models layer is in the inference service's static import closure,
and importing columns there pulls 48 modules -- the whole data layer, through graph_identity.py and the sealed pilot code
of materialization.py -- into the API image (tests/unit/test_docker_image_covers_the_api.py). Nor can columns.py import
them from here: its implementations are sealed into every derivation identity. So, by the precedent columns.py itself
sets (its implementations copy the epoch-1 builder's formulas and a test proves them equal),
tests/unit/test_feature_semantics_1d1c1c.py proves each formula identical to its registered implementation in syntax
tree and in output, bit for bit, and the family equal to the registry's live STRICT dependents of ``af_raw``.

A family is projectable only in an epoch whose root is three-valued; today that is the allele-frequency family in epoch
2. The gene-constraint family (``loeuf`` -> ``gene_is_constrained``) is NOT projectable: the model feature ``loeuf`` is
filled with 1.0 by the builder and carries an upstream 1.0 sentinel (LOEUF-SENTINEL-1), so its root is not three-valued.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["AF_FAMILY", "DerivationFamily", "FeatureSemantics", "PROJECTABLE_FAMILIES", "RENAMED_IN_V2",
           "census_identity", "derive_family", "infer_semantics", "roster_without_family", "tabular_features"]


class FeatureSemantics(str, Enum):
    V1_LEGACY = "gvc.features/1"
    V2_OBSERVED_AF = "gvc.features/2"


#: Epoch-2 names that differ from epoch 1 -> the census identity (the epoch-1 name) each resolves to.
RENAMED_IN_V2: Mapping[str, str] = MappingProxyType({"af_is_observed_zero": "af_is_absent"})
_V1_TO_V2 = {v: k for k, v in RENAMED_IN_V2.items()}


def _legacy_features() -> tuple[str, ...]:
    from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES   # call time: no import cycle
    return tuple(TABULAR_FEATURES)


def tabular_features(semantics: FeatureSemantics) -> tuple[str, ...]:
    """The tabular contract of an epoch, in contract order. Epoch 2 is epoch 1 with the renamed names in place."""
    if not isinstance(semantics, FeatureSemantics):
        raise TypeError("semantics must be a FeatureSemantics")
    legacy = _legacy_features()
    if semantics is FeatureSemantics.V1_LEGACY:
        return legacy
    return tuple(_V1_TO_V2.get(name, name) for name in legacy)


def _census_key(name: str, semantics: FeatureSemantics) -> str:
    """The explicit translation alone (no roster lookup, so it is safe at import time)."""
    if semantics is FeatureSemantics.V2_OBSERVED_AF:
        if name in _V1_TO_V2:
            raise ValueError(f"{name!r} is an epoch-1 name; epoch 2 calls it {_V1_TO_V2[name]!r}")
        return RENAMED_IN_V2.get(name, name)
    if name in RENAMED_IN_V2:
        raise ValueError(f"{name!r} is an epoch-2 name; it has no meaning in {semantics.value}")
    return name


def census_identity(name: str, semantics: FeatureSemantics) -> str:
    """The census entry a feature name of ``semantics`` resolves to (the translation is explicit, never a guess)."""
    if not isinstance(semantics, FeatureSemantics):
        raise TypeError("semantics must be a FeatureSemantics")
    if name not in tabular_features(semantics):
        raise ValueError(f"{name!r} is not a tabular feature of {semantics.value}")
    return _census_key(name, semantics)


def infer_semantics(names: Iterable[str]) -> FeatureSemantics | None:
    """The epoch a set of column names can only belong to, or None when it names neither epoch's distinguishing feature.

    A frame holding BOTH an epoch-1-only name and an epoch-2-only name is refused: it is a mixture of epochs."""
    names = set(names)
    v1 = names & set(_V1_TO_V2)
    v2 = names & set(RENAMED_IN_V2)
    if v1 and v2:
        raise ValueError(f"the columns mix feature-semantics epochs: {sorted(v1)} (epoch 1) with {sorted(v2)} (epoch 2)")
    if v2:
        return FeatureSemantics.V2_OBSERVED_AF
    if v1:
        return FeatureSemantics.V1_LEGACY
    return None


# --------------------------------------------------------------------------------------------- derivation families
@dataclass(frozen=True)
class DerivationFamily:
    """A root quantity and its registered STRICT dependents, in an epoch where the root is three-valued."""
    family_id: str
    semantics: FeatureSemantics
    root: str
    members: tuple[str, ...]                 # canonical names, root first, in contract order
    rules: tuple[tuple[str, str], ...]       # (dependent canonical name, DERIVED_RULES key = its census identity)
    formulas: tuple[tuple[str, str], ...]    # (dependent canonical name, the function of this module computing it)
    model_names: tuple[tuple[str, str], ...] # (canonical name, model-projected name) (ruling aa999b96 revised s.4)
    domain: tuple[float, float]              # closed interval of a valid observed root value

    # Stored as sorted tuples of pairs (immutable and picklable: a fitted member serialises its families); read through
    # rule_map() and model_name_map().
    def __post_init__(self) -> None:
        for name in ("rules", "formulas", "model_names"):
            pairs = dict(getattr(self, name))
            object.__setattr__(self, name, tuple(sorted(pairs.items())))
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "domain", (float(self.domain[0]), float(self.domain[1])))
        rules, model_names = self.rule_map(), self.model_name_map()
        if self.members[0] != self.root or len(set(self.members)) != len(self.members):
            raise ValueError("a family lists its root first and each member once")
        formulas = dict(self.formulas)
        if set(rules) != set(self.members[1:]) or set(formulas) != set(rules) or set(model_names) != set(self.members):
            raise ValueError("a family names a rule and a formula for every dependent and a model name for every member")
        if len(set(model_names.values())) != len(self.members) or set(model_names.values()) & set(self.members):
            raise ValueError("model-projected names are distinct from each other and from the canonical names")
        for dependent, key in rules.items():
            if _census_key(dependent, self.semantics) != key:
                raise ValueError(f"{dependent}: its census identity is not {key!r}")
        for dependent, fn in formulas.items():
            if not callable(globals().get(fn)) or not fn.startswith("formula_"):
                raise ValueError(f"{dependent}: {fn!r} is not a formula of this module")

    def rule_map(self) -> dict[str, str]:
        return dict(self.rules)

    def model_name_map(self) -> dict[str, str]:
        return dict(self.model_names)

    def implementation(self, dependent: str):
        """The production function computing ``dependent`` from observed root values (proven equal, bitwise, to its
        registered implementation in columns.DERIVED_RULES by the suite)."""
        return globals()[dict(self.formulas)[dependent]]

    def implementation_names(self) -> dict[str, str]:
        return dict(self.formulas)

    def as_dict(self) -> dict:
        return {"family_id": self.family_id, "semantics": self.semantics.value, "root": self.root,
                "members": list(self.members), "registered_rules": self.rule_map(),
                "implementations": self.implementation_names(),
                "model_names": self.model_name_map(), "domain": list(self.domain)}


# The epoch-2 formulas: the expressions of columns.DERIVED_RULES' registered implementations (af_log10_of, af_is_absent_of,
# af_is_ultra_rare_of, af_is_rare_of, af_is_common_of), applied to observed values only. Equality is a test, bit for bit.
def formula_af_log10(af: np.ndarray) -> np.ndarray:
    return np.log10(af + 1e-8)


def formula_af_is_observed_zero(af: np.ndarray) -> np.ndarray:
    return (af == 0).astype(np.float64)


def formula_af_is_ultra_rare(af: np.ndarray) -> np.ndarray:
    return (af < 0.0001).astype(np.float64)


def formula_af_is_rare(af: np.ndarray) -> np.ndarray:
    return ((af >= 0.0001) & (af < 0.001)).astype(np.float64)


def formula_af_is_common(af: np.ndarray) -> np.ndarray:
    return (af >= 0.01).astype(np.float64)


AF_FAMILY = DerivationFamily(
    family_id="allele_frequency",
    semantics=FeatureSemantics.V2_OBSERVED_AF,
    root="af_raw",
    members=("af_raw", "af_log10", "af_is_observed_zero", "af_is_ultra_rare", "af_is_rare", "af_is_common"),
    rules={"af_log10": "af_log10", "af_is_observed_zero": "af_is_absent", "af_is_ultra_rare": "af_is_ultra_rare",
           "af_is_rare": "af_is_rare", "af_is_common": "af_is_common"},
    formulas={"af_log10": "formula_af_log10", "af_is_observed_zero": "formula_af_is_observed_zero",
              "af_is_ultra_rare": "formula_af_is_ultra_rare", "af_is_rare": "formula_af_is_rare",
              "af_is_common": "formula_af_is_common"},
    model_names={"af_raw": "af_model_raw", "af_log10": "af_model_log10", "af_is_observed_zero": "af_model_is_zero",
                 "af_is_ultra_rare": "af_model_is_ultra_rare", "af_is_rare": "af_model_is_rare",
                 "af_is_common": "af_model_is_common"},
    domain=(0.0, 1.0),
)

#: The families a member projection may impute jointly, per epoch. Epoch 1 has none: its AF family is zero-filled.
PROJECTABLE_FAMILIES: Mapping[FeatureSemantics, tuple[DerivationFamily, ...]] = MappingProxyType({
    FeatureSemantics.V1_LEGACY: (),
    FeatureSemantics.V2_OBSERVED_AF: (AF_FAMILY,),
})


def derive_family(family: DerivationFamily, root_values: np.ndarray) -> dict[str, np.ndarray]:
    """Every member of ``family`` from root values: the registered implementation on observed rows, NaN on every other.

    Refused, never repaired: a non-vector, an infinite value, and an observed value outside the family's domain."""
    if not isinstance(family, DerivationFamily):
        raise TypeError("family must be a DerivationFamily")
    a = np.asarray(root_values, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{family.root}: one value per row required")
    if np.isinf(a).any():
        raise ValueError(f"{family.root}: {int(np.isinf(a).sum())} infinite value(s); refused, never clipped")
    observed = ~np.isnan(a)
    lo, hi = family.domain
    bad = observed & ((a < lo) | (a > hi))
    if bad.any():
        raise ValueError(f"{family.root}: {int(bad.sum())} observed value(s) outside [{lo}, {hi}], e.g. "
                         f"{a[bad][:3].tolist()}; refused, never clipped")
    out = {family.root: a.copy()}
    for dependent in family.members[1:]:
        values = np.full(len(a), np.nan)
        if observed.any():
            values[observed] = np.asarray(family.implementation(dependent)(a[observed]), dtype=np.float64)
        out[dependent] = values
    return out


def roster_without_family(roster: Iterable[str], family: DerivationFamily) -> tuple[str, ...]:
    """The AF-free arm's roster (ruling aa999b96 decision A): the roster with the WHOLE family removed and every other
    name kept in order. A roster holding only part of the family is refused -- an ablation of a family that was never
    whole is not the declared experiment."""
    names = tuple(roster)
    present = set(family.members) & set(names)
    if present != set(family.members):
        raise ValueError(f"the {family.family_id} family is not complete in this roster (missing "
                         f"{sorted(set(family.members) - present)}); the AF-free arm removes a complete family only")
    return tuple(n for n in names if n not in present)
