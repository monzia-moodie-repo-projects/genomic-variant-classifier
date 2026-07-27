"""The controlled metadata vocabulary, and the accessors built on it.

WHY A CONTROLLED VOCABULARY
===========================
Free-form string keys drift. Without a single canonical spelling,
`population_scope`, `populationScope`, `population` and `scope` can all appear
across metrics and each looks correct in isolation. That is the same wording
drift that let one word name two estimands in the 2026-07-25 P6 audit, and that
let "explicit conflicts preserved" name a count of withheld-label states.

WHY PROPERTIES AND NOT CONSTRUCTOR FIELDS
------------------------------------------
Measured on 2026-07-27: 53 `MetricResult` construction sites, 39 in `src/`, of
which 35 are in `representation_geometry.py` and `norm_angle_probe.py` and use
POSITIONAL arguments -- `MetricResult(mean, MetricStatus.OK)`.

Those are mathematical probes over embedding spaces: effective rank, anisotropy,
angular concentration, hubness. "Population scope" has no epidemiological
meaning for a spectral effective-rank measurement, and "classes observed" none at
all. Making the fields mandatory would put ceremonial values in exactly the
places they cannot be checked, which is a WEAKER contract, not a stronger one.

So `MetricResult` remains a GENERIC result contract. The evaluation registry
requires these keys and fills them; representation probes do not, and the
accessors return None there. A stronger domain-specific contract can be layered
on later without touching any of the 53 sites.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.evaluation import (
    capabilities,
    norm_angle_probe,
    registry,
    representation_geometry,
)
from genomic_variant_classifier.evaluation.capabilities import (
    MetricMetadataKey,
    MetricResult,
    MetricStatus,
)
from genomic_variant_classifier.evaluation.registry import MetricContext, evaluate_registered

_SRC = Path(capabilities.__file__).parent


def _ctx(**kw):
    base = dict(y_true=np.array([0., 1., 0., 1.]), y_score=np.array([.2, .8, .3, .9]),
                y_prob=np.array([.2, .8, .3, .9]), population_scope="unit_test_cohort")
    base.update(kw)
    return MetricContext(**base)


# --------------------------------------------------------------------------- #
# 1. The vocabulary itself
# --------------------------------------------------------------------------- #
def test_the_key_enum_is_a_str_enum_not_a_StrEnum():
    """`StrEnum` arrived in Python 3.11; pyproject declares >= 3.10. Every other
    vocabulary enum in this project uses the same `str`-and-`Enum` pattern."""
    assert issubclass(MetricMetadataKey, str)
    assert MetricMetadataKey.POPULATION_SCOPE == "population_scope"


def test_members_interchange_with_their_values_as_dict_keys():
    """This is what lets the enum be the canonical spelling in code while
    serialized artifacts keep plain string keys, so every existing reader of
    metadata["population_scope"] keeps working. Verified rather than assumed."""
    k = MetricMetadataKey.POPULATION_SCOPE
    assert hash(k) == hash("population_scope")
    assert {"population_scope": 1}[k] == 1
    assert {k: 1}["population_scope"] == 1
    assert json.loads(json.dumps({k: 1})) == {"population_scope": 1}


def test_every_key_value_is_lower_snake_case_and_unique():
    values = [m.value for m in MetricMetadataKey]
    assert len(values) == len(set(values))
    for v in values:
        assert v == v.strip().lower() and " " not in v, v


# --------------------------------------------------------------------------- #
# 2. No spelling variants anywhere in the package
# --------------------------------------------------------------------------- #
# Only spellings that would mean THE SAME THING as the canonical key. The first
# draft of this list was wider and produced two false positives, both instructive:
#
#   prediction_artifacts.py "scope"  -- a COLUMN in a calibration-breakdown table
#                                       ("scope": "global"), not result metadata.
#   representation_geometry.py "n_rows" -- genuinely MetricResult metadata, but on
#                                       a Family B probe, where it means rows of an
#                                       EMBEDDING MATRIX, not observations in a
#                                       cohort. A different quantity, not a variant.
#
# Those two findings are why the two-family distinction is recorded rather than
# erased: forcing Family B probes onto Family A's vocabulary would have been the
# weaker contract.
_FORBIDDEN = {
    "population_scope": {"populationScope", "populationscope", "pop_scope",
                         "population_Scope"},
    "certification_eligible": {"certificationEligible", "cert_eligible",
                               "certificationeligible"},
    "n_observations": {"nObservations", "n_obs", "num_observations"},
    "n_classes_observed": {"nClassesObserved", "num_classes_observed"},
    "n_clusters": {"nClusters", "num_clusters", "n_cluster"},
}


@pytest.mark.parametrize("canonical,variants", sorted(_FORBIDDEN.items()))
def test_no_spelling_variant_of_a_canonical_key_is_used_as_a_metadata_key(canonical, variants):
    """Forbids the drift the enum exists to prevent, at the source level: a
    dictionary literal keyed by a near-miss spelling.

    Scanned on the abstract syntax tree, not by grepping text, because a textual
    search cannot distinguish a key from a docstring -- a distinction that broke
    an earlier guard in this same session."""
    offenders = []
    for path in _SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k in node.keys:
                    if isinstance(k, ast.Constant) and k.value in variants:
                        offenders.append(f"{path.name}:{k.lineno} -> {k.value!r}")
    assert not offenders, (
        f"spelling variants of {canonical!r} used as metadata keys: {offenders}. "
        f"Use MetricMetadataKey.{canonical.upper()}.")


# --------------------------------------------------------------------------- #
# 3. The accessors
# --------------------------------------------------------------------------- #
def test_registry_results_expose_every_accessor():
    r = evaluate_registered(_ctx())["auroc"]
    assert r.population_scope == "unit_test_cohort"
    assert r.certification_eligible is True
    assert r.n_observations == 4
    assert r.n_classes_observed == 2
    assert r.metric_name == "auroc"
    assert r.n_clusters is None, "no clusters supplied"


def test_cluster_count_appears_only_when_clusters_are_supplied():
    r = evaluate_registered(_ctx(clusters=np.array(["A", "A", "B", "B"])))["auroc"]
    assert r.n_clusters == 2


def test_accessors_return_None_for_a_purely_mathematical_probe():
    """A representation probe carries no population, and must not be forced to
    invent one. This is the measured reason the fields are properties rather
    than constructor arguments."""
    probe = MetricResult(0.87, MetricStatus.OK)
    assert probe.population_scope is None
    assert probe.certification_eligible is None
    assert probe.n_observations is None
    assert probe.n_clusters is None


@pytest.mark.parametrize("key,bad", [
    (MetricMetadataKey.POPULATION_SCOPE, 7),
    (MetricMetadataKey.CERTIFICATION_ELIGIBLE, "yes"),
    (MetricMetadataKey.N_OBSERVATIONS, "4"),
    (MetricMetadataKey.N_OBSERVATIONS, True),
    (MetricMetadataKey.N_CLUSTERS, 2.5),
])
def test_an_accessor_refuses_a_wrongly_typed_value_rather_than_returning_it(key, bad):
    """A wrongly typed value is not evidence. `True` is deliberately rejected for
    the counts: bool is a subclass of int, so a naive isinstance check would
    report n_observations == True."""
    r = MetricResult(0.5, MetricStatus.OK, metadata={key: bad})
    attr = key.value if key.value != "metric_name" else "metric_name"
    assert getattr(r, attr) is None


# --------------------------------------------------------------------------- #
# 4. The existing panels still work, unchanged
# --------------------------------------------------------------------------- #
def test_the_positional_constructor_still_works():
    """35 of the 39 production construction sites are positional. A change that
    broke them would have been a constructor redesign smuggled into an
    integration commit."""
    r = MetricResult(0.5, MetricStatus.OK)
    assert r.value == 0.5 and r.status is MetricStatus.OK and r.metadata == {}
    bad = MetricResult(float("nan"), MetricStatus.INSUFFICIENT_SUPPORT, "too few rows")
    assert bad.reason == "too few rows"


@pytest.mark.parametrize("module", [representation_geometry, norm_angle_probe])
def test_the_probe_modules_still_resolve_the_same_MetricResult(module):
    assert module.MetricResult is MetricResult


def test_the_registry_uses_the_enum_rather_than_string_literals():
    """The enum only prevents drift if the registry actually uses it."""
    src = Path(registry.__file__).read_text(encoding="utf-8")
    assert "MetricMetadataKey." in src
    tree = ast.parse(src)
    literal_keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value in {m.value for m in MetricMetadataKey}:
                    literal_keys.append(f"{k.lineno} -> {k.value!r}")
    assert not literal_keys, (
        f"registry.py uses canonical keys as string literals: {literal_keys}. "
        "Use MetricMetadataKey members so the spelling has one source.")
