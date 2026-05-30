"""AutoKernel-style correctness harness for the variant ensemble.

Public surface:
    from genomic_variant_classifier.agent_layer.harness import (
        run_correctness_harness, HarnessReport,
    )
"""
from __future__ import annotations

from genomic_variant_classifier.agent_layer.harness.correctness_harness import (
    KNOWN_ZERO_DEFAULT,
    HarnessReport,
    build_reference_slice,
    run_correctness_harness,
)

__all__ = [
    "run_correctness_harness",
    "HarnessReport",
    "build_reference_slice",
    "KNOWN_ZERO_DEFAULT",
]