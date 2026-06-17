"""test_run17_fullflag_smoke.py -- Monzia Moodie

The Run-17 full-flag laptop smoke must be ABLE to activate AND audit every no-defer feature together:
  gnn_score (--string-db auto), hetero_gnn_score (--hetero-gnn), af_1kg_* (--kg), reactome_pathway_count
  (--kg-edges reactome:...), plus the already-wired gnomAD/dbNSFP/LOVD features.

Two coupled pieces are exercised here:
  1. smoke_all_models._build_eval_cmd now forwards --kg / --hetero-gnn / --kg-edges (getattr-safe, so the
     existing gnn-epochs namespace still works).
  2. audit_smoke_feature_population.py gains a --run17 mode (EXPECT_RUN17) that fails loud if any FAIL-
     severity no-defer feature is dead (constant) in the engineered split matrix.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import smoke_all_models
import audit_smoke_feature_population as audit


# ----------------------------------------------------------------------------- forwarding (smoke side)
def _smoke_args(**kw):
    base = dict(string_db="auto", smoke_n=3000, n_folds=3, min_review_tier=3,
                gnomad=None, spliceai=None, alphamissense=None, seq_windows=None,
                gnomad_constraint=None, dbnsfp_path=None, lovd_path=None, gnn_epochs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_omits_run17_flags_by_default():
    # the legacy namespace has NO kg/hetero_gnn/kg_edges attrs -> getattr-safe -> none forwarded
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(), "eval.py", "c.parquet", "/tmp/o")
    assert "--kg" not in cmd
    assert "--hetero-gnn" not in cmd
    assert "--kg-edges" not in cmd
    assert "--reactome-path" not in cmd
    assert "--gtex-path" not in cmd


def test_forwards_kg():
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(kg="kg.parquet"), "eval.py", "c.parquet", "/tmp/o")
    assert "--kg" in cmd and cmd[cmd.index("--kg") + 1] == "kg.parquet"


def test_forwards_reactome_path():
    cmd = smoke_all_models._build_eval_cmd(
        _smoke_args(reactome_path="reactome.parquet"), "eval.py", "c.parquet", "/tmp/o")
    assert "--reactome-path" in cmd and cmd[cmd.index("--reactome-path") + 1] == "reactome.parquet"


def test_forwards_gtex_path():
    cmd = smoke_all_models._build_eval_cmd(
        _smoke_args(gtex_path="gtex.parquet"), "eval.py", "c.parquet", "/tmp/o")
    assert "--gtex-path" in cmd and cmd[cmd.index("--gtex-path") + 1] == "gtex.parquet"


def test_forwards_hetero_gnn():
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(hetero_gnn=True), "eval.py", "c.parquet", "/tmp/o")
    assert "--hetero-gnn" in cmd
    # store_true flag: no value follows it
    assert "--hetero-gnn" not in cmd[cmd.index("--hetero-gnn") + 1:]


def test_omits_hetero_gnn_when_false():
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(hetero_gnn=False), "eval.py", "c.parquet", "/tmp/o")
    assert "--hetero-gnn" not in cmd


def test_forwards_kg_edges():
    cmd = smoke_all_models._build_eval_cmd(
        _smoke_args(kg_edges=["reactome:r.gmt", "kegg:k.gmt"]), "eval.py", "c.parquet", "/tmp/o")
    i = cmd.index("--kg-edges")
    assert cmd[i + 1] == "reactome:r.gmt" and cmd[i + 2] == "kegg:k.gmt"


def test_full_flag_all_present_with_core():
    cmd = smoke_all_models._build_eval_cmd(
        _smoke_args(kg="kg.parquet", hetero_gnn=True, kg_edges=["reactome:r.gmt"]),
        "eval.py", "c.parquet", "/tmp/o")
    for f in ("--clinvar", "--string-db", "--kg", "--hetero-gnn", "--kg-edges"):
        assert f in cmd, f


def test_parser_accepts_run17_flags():
    # exercises the REAL extracted parser, not a rebuilt copy
    parsed = smoke_all_models.parse_args(
        ["--clinvar", "c.parquet", "--kg", "k.parquet", "--hetero-gnn",
         "--kg-edges", "reactome:r.gmt", "kegg:k.gmt"])
    assert parsed.kg == "k.parquet"
    assert parsed.hetero_gnn is True
    assert parsed.kg_edges == ["reactome:r.gmt", "kegg:k.gmt"]


def test_parser_run17_flags_default_off():
    parsed = smoke_all_models.parse_args(["--clinvar", "c.parquet"])
    assert parsed.kg is None
    assert parsed.hetero_gnn is False
    assert parsed.kg_edges is None
    assert parsed.reactome_path is None


def test_parser_accepts_reactome_path():
    parsed = smoke_all_models.parse_args(
        ["--clinvar", "c.parquet", "--reactome-path", "r.parquet"])
    assert parsed.reactome_path == "r.parquet"


def test_parser_accepts_gtex_path():
    parsed = smoke_all_models.parse_args(
        ["--clinvar", "c.parquet", "--gtex-path", "g.parquet"])
    assert parsed.gtex_path == "g.parquet"


# ----------------------------------------------------------------------------- EXPECT_RUN17 coverage
def test_expect_run17_covers_nodefer_set():
    cols = set(audit.EXPECT_RUN17)
    for c in ("af_log10", "gnn_score", "hetero_gnn_score", "reactome_pathway_count",
              "af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr",
              "cadd_phred", "sift_score", "revel_score", "n_tools_pathogenic", "lovd_variant_class"):
        assert c in cols, c
    # af_1kg_* and the score features are FAIL severity; lovd is WARN
    assert audit.EXPECT_RUN17["af_1kg_afr"][1] == "fail"
    assert audit.EXPECT_RUN17["hetero_gnn_score"][1] == "fail"
    assert audit.EXPECT_RUN17["lovd_variant_class"][1] == "warn"


# ----------------------------------------------------------------------------- audit --run17 end to end
def _write_splits(tmp_path, overrides=None):
    """Build a synthetic engineered matrix where, by default, EVERY run17 feature is populated."""
    n = 60
    import numpy as np
    rng = np.arange(n)
    df = pd.DataFrame({
        "af_log10": np.linspace(-6, -1, n),
        "gnn_score": np.linspace(0.1, 0.9, n),
        "hetero_gnn_score": np.linspace(0.2, 0.8, n),
        "reactome_pathway_count": (rng % 7).astype(float),
        "af_1kg_afr": np.where(rng % 3 == 0, 0.0, np.linspace(0.01, 0.4, n)),
        "af_1kg_eur": np.where(rng % 3 == 1, 0.0, np.linspace(0.01, 0.4, n)),
        "af_1kg_eas": np.where(rng % 4 == 0, 0.0, np.linspace(0.01, 0.4, n)),
        "af_1kg_sas": np.where(rng % 5 == 0, 0.0, np.linspace(0.01, 0.4, n)),
        "af_1kg_amr": np.where(rng % 2 == 0, 0.0, np.linspace(0.01, 0.4, n)),
        "cadd_phred": np.linspace(0, 35, n),          # nondefault vs 15.0
        "sift_score": np.linspace(0, 1, n),           # nondefault vs 0.5
        "revel_score": np.linspace(0, 1, n),          # nondefault vs 0.5
        "n_tools_pathogenic": (rng % 5).astype(float),# nondefault vs 0
        "lovd_variant_class": (rng % 3).astype(float),# nonzero
    })
    if overrides:
        for col, val in overrides.items():
            df[col] = val
    sp = tmp_path / "splits"
    sp.mkdir()
    df.to_parquet(sp / "X_train.parquet")
    return sp


def _run_audit(splits_dir):
    old = sys.argv
    try:
        sys.argv = ["audit", str(splits_dir), "--run17"]
        return audit.main()
    finally:
        sys.argv = old


def test_audit_run17_pass_all_populated(tmp_path):
    assert _run_audit(_write_splits(tmp_path)) == 0


def test_audit_run17_fail_af1kg_dead(tmp_path):
    # all five af_1kg_* dead (constant 0.0) -> FAIL
    sp = _write_splits(tmp_path, overrides={c: 0.0 for c in
                       ["af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr"]})
    assert _run_audit(sp) == 1


def test_audit_run17_fail_hetero_dead(tmp_path):
    sp = _write_splits(tmp_path, overrides={"hetero_gnn_score": 0.5})  # constant default
    assert _run_audit(sp) == 1


def test_audit_run17_lovd_only_dead_passes(tmp_path):
    sp = _write_splits(tmp_path, overrides={"lovd_variant_class": 0.0})  # warn-only -> still PASS
    assert _run_audit(sp) == 0
