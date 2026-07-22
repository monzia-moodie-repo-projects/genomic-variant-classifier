"""Typed output for the graph-neural-network forward pass.

WHY THIS EXISTS
===============
`models/gnn.py` computed the focal-node representation and threw it away:

    focal_embeddings = x[gene_idx]
    return self.classifier(focal_embeddings)   # the embedding is discarded

That discard is the sole reason Panel R's R3 through R7 are registered
NOT_IMPLEMENTED: every probe, whitening ladder, hubness and trajectory metric
needs a representation the model never handed out. This dataclass is the return
type that lets the model hand it out WITHOUT changing what any existing caller
receives.

THE CONTRACT
------------
`focal_embeddings` is the EXACT tensor consumed by the classifier -- not a copy,
not an adjacent layer, not a re-derivation. A forward-pre-hook on the classifier
must capture a tensor equal to this one bit-for-bit. That identity is what makes
the exported representation a legitimate scientific object rather than "something
shaped like the embedding".

It stays attached to autograd during training and must be detached explicitly
before persistence or cross-process analysis. The extraction boundary
(a separate, later commit) is responsible for that detachment; this type does
not silently detach, because a silent detach would hide a gradient-flow bug.

WHY OPT-IN, NOT ALWAYS-ON
-------------------------
Five call sites in this repository do `out = model(...)` then
`F.softmax(out, ...)`. Returning this dataclass unconditionally would break all
five in a single commit. Instead `forward(..., return_embeddings=False)` returns
the bare logits tensor exactly as before, and only `return_embeddings=True`
returns a GNNOutput. Every existing caller is untouched; the extraction path
opts in explicitly.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # torch is a hard dependency of the GNN branch, but this module must be
      # importable for type reference even where the annotation is a string.
    from torch import Tensor
except Exception:  # pragma: no cover - exercised only in a torch-free env
    Tensor = "Tensor"  # type: ignore


@dataclass(frozen=True)
class GNNOutput:
    """Outputs of a VariantGAT / VariantGATGPS forward pass.

    Attributes:
        logits: (n_focal, 2) class logits -- byte-identical to what forward()
            returns in its default (tensor) mode.
        focal_embeddings: (n_focal, out_channels) the exact pre-classifier
            representation, or None when the pass did not request it.

    Frozen so a caller cannot mutate the exported representation in place and
    then believe it is still what the model produced.
    """

    logits: "Tensor"
    focal_embeddings: Optional["Tensor"] = None

    @property
    def has_embeddings(self) -> bool:
        return self.focal_embeddings is not None
