"""test_split_health_gate.py -- Author: Monzia Moodie
Validates the allowlist-aware split-health gate's pure core: hard vs soft
degeneracy, the three buckets, and the stale-split vs revived-split verdicts.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import split_health_gate as G  # noqa: E402


def test_hard_vs_soft_degeneracy():
    assert G.is_hard_degenerate("ALL_ZERO") is True
    assert G.is_hard_degenerate("CONSTANT;ALL_ZERO") is True
    assert G.is_hard_degenerate("ALL_NULL") is True
    assert G.is_hard_degenerate("NEAR_CONSTANT(0.9994)") is False   # soft
    assert G.is_hard_degenerate("") is False


def test_core_degenerate_is_no_go():
    res = G.classify({"alphamissense_score": "ALL_ZERO"})
    assert res["verdict"] == "NO_GO" and res["core_degenerate"] == ["alphamissense_score"]


def test_unexpected_degenerate_is_no_go():
    # gtex_* is NOT a known stub: if still dead after re-prep, that's a real failure
    res = G.classify({"gtex_max_tpm": "CONSTANT;ALL_ZERO"})
    assert res["verdict"] == "NO_GO" and res["unexpected_degenerate"] == ["gtex_max_tpm"]


def test_only_expected_stubs_is_go():
    res = G.classify({"eve_score": "CONSTANT", "phylop_score": "ALL_ZERO",
                      "hgmd_n_reports": "ALL_ZERO", "alphafold_plddt": "CONSTANT"})
    assert res["verdict"] == "GO"
    assert set(res["expected_degenerate"]) == {"eve_score", "phylop_score",
                                               "hgmd_n_reports", "alphafold_plddt"}


def test_near_constant_is_warning_not_failure():
    res = G.classify({"is_mitochondrial": "NEAR_CONSTANT(0.9994)"})
    assert res["verdict"] == "GO"
    assert res["near_constant_warnings"] == ["is_mitochondrial"]
    assert res["unexpected_degenerate"] == []


def test_stale_split_fails_revived_split_passes():
    # BEFORE re-prep: stale families dead -> unexpected -> NO_GO
    stale = {c: "ALL_ZERO" for c in
             ["gtex_max_tpm", "af_1kg_eur", "gene_constraint_oe", "maxentscan_score",
              "dbsnp_af", "finngen_af_fin"]}
    stale.update({"eve_score": "CONSTANT", "phylop_score": "ALL_ZERO"})  # stubs too
    assert G.classify(stale)["verdict"] == "NO_GO"
    # AFTER re-prep: stale families revived; only stubs remain dead -> GO
    revived = {"eve_score": "CONSTANT", "phylop_score": "ALL_ZERO",
               "omim_n_diseases": "ALL_ZERO", "esm2_delta_norm": "ALL_ZERO"}
    assert G.classify(revived)["verdict"] == "GO"


def test_curated_sets_are_disjoint():
    assert not (G.EXPECTED_ZERO & G.CORE_FEATURES)


def test_max_unexpected_tolerance():
    deg = {"some_new_feature": "ALL_ZERO"}
    assert G.classify(deg, max_unexpected=0)["verdict"] == "NO_GO"
    assert G.classify(deg, max_unexpected=1)["verdict"] == "GO"


# --- adapter seam: the bug was storing col_health()'s DICT instead of its string ---
def test_reason_from_health_extracts_string():
    # real col_health contract: dict with a "degenerate" key
    assert G.reason_from_health({"degenerate": "CONSTANT;ALL_ZERO"}) == "CONSTANT;ALL_ZERO"
    assert G.reason_from_health({"degenerate": ""}) == ""          # healthy column
    assert G.reason_from_health({"degenerate": "", "n_unique": 5}) == ""


def test_reason_from_health_rejects_bad_contract():
    import pytest
    # passing the raw object (the original bug) must fail LOUDLY, not silently
    with pytest.raises(TypeError):
        G.reason_from_health("ALL_ZERO")          # a bare string is NOT the contract
    with pytest.raises(TypeError):
        G.reason_from_health({"oops": "x"})       # dict without "degenerate"


def test_score_splits_uses_real_col_health(tmp_path):
    # end-to-end through the REAL feature_health.col_health on real parquets:
    # this is the seam the original 8 tests never exercised.
    import pandas as pd
    n = 50
    df = pd.DataFrame({
        "alphamissense_score": [i / n for i in range(n)],   # healthy (varied)
        "gtex_max_tpm": [0.0] * n,                           # ALL_ZERO -> degenerate
        "eve_score": [1.0] * n,                              # CONSTANT -> degenerate (stub)
    })
    sdir = tmp_path / "splits"
    sdir.mkdir()
    df.to_parquet(sdir / "X_train.parquet")
    df.to_parquet(sdir / "X_val.parquet")
    deg, present = G._score_splits(sdir, near_constant_frac=0.999)
    # values must be STRINGS (the bug stored dicts here)
    assert all(isinstance(v, str) for v in deg.values()), deg
    assert "alphamissense_score" not in deg
    assert "ALL_ZERO" in deg.get("gtex_max_tpm", "")
    assert "CONSTANT" in deg.get("eve_score", "")
    assert present == {"alphamissense_score", "gtex_max_tpm", "eve_score"}
    # full gate verdict: gtex dead = unexpected -> NO_GO
    res = G.classify(deg)
    assert res["verdict"] == "NO_GO"
    assert res["unexpected_degenerate"] == ["gtex_max_tpm"]
    assert res["expected_degenerate"] == ["eve_score"]


def test_score_splits_no_matching_files_is_loud(tmp_path):
    import pytest
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        G._score_splits(tmp_path / "empty", near_constant_frac=0.999)


# --- presence check: a silently-MISSING core feature must fail, with prep-only exemption ---
def test_missing_core_feature_is_no_go():
    # all core present except splice_ai_score -> NO_GO even with zero degeneracy
    present = G.CORE_FEATURES - {"splice_ai_score"}
    res = G.classify({}, present=present)
    assert res["verdict"] == "NO_GO"
    assert res["missing_core"] == ["splice_ai_score"]


def test_prep_only_exempts_gnn_stage_features():
    # prep.run() output legitimately lacks gnn_score/hetero_gnn_score
    present = G.CORE_FEATURES - G.GNN_STAGE_FEATURES
    assert G.classify({}, present=present, prep_only=True)["verdict"] == "GO"
    # but in FULL mode their absence IS a failure
    full = G.classify({}, present=present, prep_only=False)
    assert full["verdict"] == "NO_GO"
    assert set(full["missing_core"]) == G.GNN_STAGE_FEATURES


def test_present_none_skips_presence_check():
    # backward-compatible: no present set -> no presence gating
    assert G.classify({})["verdict"] == "GO"
    assert G.classify({})["missing_core"] == []
