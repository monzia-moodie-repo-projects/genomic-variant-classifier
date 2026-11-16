"""Fit contexts for unit tests (stage 1D-1c.1C, owner ruling aa999b96 decision B).

VariantEnsemble.fit requires an explicit FitContext. A unit test fits synthetic matrices to qualify code: that is an
ENGINEERING QUALIFICATION, never a scientific artifact, and it says so here once rather than at every call site.

Author: Monzia Moodie
"""
from __future__ import annotations

from genomic_variant_classifier.models.fit_context import FitContext
from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
from genomic_variant_classifier.models.member import ProjectionProtocol

#: Epoch-1 (legacy) synthetic qualification: the behaviour every pre-1D-1c.1C unit test exercised, unchanged.
EPOCH1_TEST_FIT = FitContext.engineering_qualification(
    FeatureSemantics.V1_LEGACY, ProjectionProtocol.LEGACY_UNPROJECTED,
    "unit-test engineering qualification on synthetic epoch-1 matrices")


def epoch2_test_fit(projection: ProjectionProtocol, reason: str = "unit-test engineering qualification") -> FitContext:
    """An epoch-2 synthetic qualification under a projection protocol."""
    return FitContext.engineering_qualification(FeatureSemantics.V2_OBSERVED_AF, projection, reason)
