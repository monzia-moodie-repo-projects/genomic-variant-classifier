"""Scientific information-source admission (package 1D, stage 1D-1a onward).

Stage 1D-1a (set "ao") records the census: what every declared tabular feature and every non-tabular information source
(graph relation, edge attribute, supervision label) MEANS, where it comes from, which surfaces consume it (measured from the
code, not asserted), and its ruled disposition. It changes no model behaviour. Stage 1D-1b builds the admission authority
on this census; stage 1D-1b.1 (set "ap") adds the typed observation contract (observation.py) and the shadow admission report
derived from the census (admission.py); stage 1D-1b.2 (set "aq") adds lineage integrity (lineage.py), columnar observations and
derived-feature propagation (columns.py), source materialisation with the EVE and allele-frequency pilots (materialization.py)
and the source-absence census (absence_census.py); stage 1D-1b.3 (set "ar") adds source-bound graph identity
(graph_identity.py: exact STRING pairs, threshold status, canonical HGNC mapping, computed graph materialisations, pathway
relations by global set size) and authorized labels (labels.py); stage 1D-1b.4 (set "as") adds population identity and fit
authorization (population.py: gene-set rows, variant-and-gene split components, the three-way split audit, the STRICT and
CONDITIONAL_COMPONENT authorities, tempered component weights, the component-disjoint checkpoint split), the claims authority
(claims.py), training labels that record every read (labels.TrainingLabels) and graph acceptance by re-derivation from the
pinned files with immutable materialisations (graph_identity.py). Nothing in production imports any of them yet.
"""
from __future__ import annotations
