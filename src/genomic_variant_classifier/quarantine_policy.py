"""The project's ONE reviewed quarantine policy -- the single definition of what is quarantined.

Dependency-free on purpose: the structural producers and real_data_prep import it, and
real_data_prep must stay importable with the heavy machine-learning stack blocked (a
suite-tested constraint). The feature-contract authority, variant_ensemble.TABULAR_FEATURES,
imports QUARANTINED_FEATURES from here and must stay disjoint from it, so there is exactly one list.

BASIS
  docs/CONTAINMENT_2026-07-24.md section 4 -- the four Phase D structural features are
    quarantined (protein_pipeline's data[0] resolver: 559,786 of 4,399,089 cohort variants
    affected, 220,590 receiving values that are wrong rather than absent).
  docs/CONTAINMENT_2026-07-24_R2.md section 1 -- scope measured: 225 artifacts, 2026-03-30 to
    2026-07-06, twelve runs.
  Monzia's ruling of 2026-09 (Option A) -- the four leave the active contract (95 -> 91) and BOTH
    structural producers are blocked, AlphaFoldConnector as well as ProteinStructurePipeline.

This policy CONTAINS; it does not repair. Restoration is the Phase 1 repair, whose target is the
seven feature-provenance states of CONTAINMENT_2026-07-24.md section 4. Restoring any name below
requires that repair and a new reviewed policy, never an edit to this list alone.
"""
from __future__ import annotations

QUARANTINED_FEATURES: tuple[str, ...] = (
    "alphafold_plddt",
    "solvent_accessibility",
    "secondary_structure_context",
    "dist_to_active_site",
)

# Fixed constants, used by each producer at its own reviewed entry points. A string supplied by an
# arbitrary caller does not authenticate code (containment boundary reference, README).
BLOCKED_PRODUCERS: tuple[str, ...] = (
    "AlphaFoldConnector",
    "ProteinStructurePipeline",
)

# AnnotationConfig fields that exist only to drive the blocked producers. Setting any of them is a
# request for quarantined features and is refused -- never silently ignored.
STRUCTURAL_CONFIG_FIELDS: tuple[str, ...] = (
    "alphafold_path",
    "alphafold_uniprot_index_path",
    "protein_cache_dir",
)
