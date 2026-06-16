"""test_gnn_epochs_flag.py -- Monzia Moodie

--gnn-epochs lets a full-flag laptop smoke train few epochs (e.g. 10) while the real run keeps the default
100 (byte-identical to before). run_phase2_eval exposes the flag and threads it through all GNN sites;
smoke_all_models forwards it ONLY when explicitly set, via the pure _build_eval_cmd helper. Not a deferral.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_phase2_eval
import smoke_all_models


def test_eval_parser_gnn_epochs_default_100():
    a = run_phase2_eval.parse_args(["--clinvar", "x.parquet"])
    assert a.gnn_epochs == 100


def test_eval_parser_gnn_epochs_override():
    a = run_phase2_eval.parse_args(["--clinvar", "x.parquet", "--gnn-epochs", "10"])
    assert a.gnn_epochs == 10


def _smoke_args(**kw):
    base = dict(string_db="auto", smoke_n=3000, n_folds=3, min_review_tier=3,
                gnomad=None, spliceai=None, alphamissense=None, seq_windows=None,
                gnomad_constraint=None, dbnsfp_path=None, lovd_path=None, gnn_epochs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_smoke_omits_gnn_epochs_by_default():
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(), "eval.py", "c.parquet", "/tmp/o")
    assert "--gnn-epochs" not in cmd


def test_smoke_forwards_gnn_epochs_when_set():
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(gnn_epochs=10), "eval.py", "c.parquet", "/tmp/o")
    assert "--gnn-epochs" in cmd and cmd[cmd.index("--gnn-epochs") + 1] == "10"


def test_smoke_cmd_has_core_flags():
    cmd = smoke_all_models._build_eval_cmd(_smoke_args(), "eval.py", "c.parquet", "/tmp/o")
    for f in ("--clinvar", "--string-db", "--max-train", "--n-folds", "--min-review-tier", "--output"):
        assert f in cmd
