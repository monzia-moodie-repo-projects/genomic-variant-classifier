"""Scientific information-source admission (package 1D, stage 1D-1a onward).

Stage 1D-1a (set "ao") records the census: what every declared tabular feature and every non-tabular information source
(graph relation, edge attribute, supervision label) MEANS, where it comes from, which surfaces consume it (measured from the
code, not asserted), and its ruled disposition. It changes no model behaviour. Stage 1D-1b builds the admission authority
on this census; stage 1D-1b.1 (set "ap") adds the typed observation contract (observation.py) and the shadow admission report
derived from the census (admission.py), which nothing in production imports yet.
"""
from __future__ import annotations
