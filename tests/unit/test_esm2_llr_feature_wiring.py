"""Lockstep + no-clip contract for esm2_llr (Phase 1 wiring).

Guards the two findings from the wiring audit:
  1. esm2_llr is registered exactly once, adjacent to esm2_delta_norm, and the
     count constant tracks it (the tripwire stays green at 80);
  2. esm2_llr is assembled WITHOUT a lower clip -- the signed negative signal
     (negative = damaging) must survive feature engineering. A reintroduced
     .clip(lower=0.0) would silently zero the entire feature; this test fails
     loudly if that happens. Author: Monzia Moodie.
"""
from __future__ import annotations

import pandas as pd

from genomic_variant_classifier.models import variant_ensemble as VE


def test_esm2_llr_registered_once_after_delta():
    feats = VE.TABULAR_FEATURES
    assert feats.count("esm2_llr") == 1, "esm2_llr must appear exactly once"
    assert feats.index("esm2_llr") == feats.index("esm2_delta_norm") + 1, \
        "esm2_llr must sit immediately after esm2_delta_norm"


def test_count_constant_tracks_list():
    assert VE.EXPECTED_TABULAR_FEATURE_COUNT == len(VE.TABULAR_FEATURES)
    assert "esm2_llr" in VE.TABULAR_FEATURES


def test_esm2_llr_assembly_preserves_negative_signal():
    # the regression guard for the clip trap: negatives MUST survive
    df = pd.DataFrame({"esm2_llr": [-9.13, 0.0, 4.0, None]})
    out = VE.engineer_features(df)
    assert "esm2_llr" in out.columns
    vals = out["esm2_llr"].tolist()
    assert vals[0] == -9.13, f"negative LLR was altered (clip trap?): {vals[0]}"
    assert vals[1] == 0.0 and vals[2] == 4.0
    assert vals[3] == 0.0, "missing LLR must default to 0.0 (neutral)"


def test_esm2_delta_norm_still_clipped():
    # delta is a norm (>=0); its clip must remain intact (we only changed llr)
    df = pd.DataFrame({"esm2_delta_norm": [-1.0, 2.0]})
    out = VE.engineer_features(df)
    assert out["esm2_delta_norm"].tolist() == [0.0, 2.0], "delta_norm clip(lower=0) was lost"
