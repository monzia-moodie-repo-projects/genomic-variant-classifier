"""Model introspection helpers for the run report (no heavy deps).

Kept dependency-light (numpy only) so it is unit-testable without importing the
full ensemble / torch stack.
Author: Monzia Moodie
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_WRAPPER_ATTRS = ("estimator_", "estimator", "_base", "base_estimator")


def model_input_width(model) -> Optional[int]:
    """Number of input features the fitted *model* actually consumes.

    For neural models carrying a fit-time variance mask this is the kept-column
    count (``feature_mask_.sum()``); otherwise ``n_features_in_``. The function
    unwraps the isotonic calibrator and the mc_dropout / deep_ensemble wrappers,
    so a calibrated TabularNN reports its masked width, not the full matrix.
    Returns ``None`` when neither attribute is exposed anywhere in the chain
    (e.g. the sequence CNN or the stacking meta-learner).
    """
    fallback: Optional[int] = None
    seen: set[int] = set()
    cur = model
    for _ in range(8):  # defensive unwrap depth cap
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))

        mask = getattr(cur, "feature_mask_", None)
        if mask is not None:
            try:
                return int(np.asarray(mask).sum())
            except Exception:
                pass

        if fallback is None:
            n = getattr(cur, "n_features_in_", None)
            if n is not None:
                try:
                    fallback = int(n)
                except Exception:
                    pass

        nxt = None
        for attr in _WRAPPER_ATTRS:
            v = getattr(cur, attr, None)
            if v is not None:
                nxt = v
                break
        if nxt is None:
            members = getattr(cur, "members_", None)
            if members:
                nxt = members[0]
        cur = nxt

    return fallback
