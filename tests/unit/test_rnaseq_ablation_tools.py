"""Regression tests for the RNA-seq Run-17-scale ablation tools.
Author: Monzia Moodie."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_MAKER = _SCRIPTS / "make_rnaseq_ablation_parquet.py"
sys.path.insert(0, str(_SCRIPTS))
import aggregate_rnaseq_ablation as agg  # noqa: E402

_RN = ["rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
       "rnaseq_log2fc", "rnaseq_de_neglog10p"]


def _src(tmp_path, n=120):
    r = np.random.default_rng(0)
    df = pd.DataFrame({"gene_symbol": [f"G{i}" for i in range(n)],
                       **{c: r.random(n) for c in _RN}})
    p = tmp_path / "full.parquet"; df.to_parquet(p, index=False)
    return p


def _make(src, out, mode, seed=0):
    r = subprocess.run([sys.executable, str(_MAKER), "--src", str(src), "--out", str(out),
                        "--mode", mode, "--seed", str(seed)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return pd.read_parquet(out)


def test_drop_all_zeros_every_rnaseq_col(tmp_path):
    out = _make(_src(tmp_path), tmp_path / "da.parquet", "drop_all")
    assert (out[_RN].to_numpy() == 0).all()


def test_drop_de_zeros_only_de(tmp_path):
    out = _make(_src(tmp_path), tmp_path / "dde.parquet", "drop_de")
    assert (out[["rnaseq_log2fc", "rnaseq_de_neglog10p"]].to_numpy() == 0).all()
    assert out["rnaseq_mean_log_tpm"].sum() != 0


def test_gene_shuffle_seed_deterministic_and_sensitive(tmp_path):
    src = _src(tmp_path)
    a = _make(src, tmp_path / "s11a.parquet", "gene_shuffle", 11)
    b = _make(src, tmp_path / "s11b.parquet", "gene_shuffle", 11)
    c = _make(src, tmp_path / "s37.parquet", "gene_shuffle", 37)
    assert a.equals(b)
    assert not a["rnaseq_log2fc"].equals(c["rnaseq_log2fc"])


def test_full_is_unchanged(tmp_path):
    src = _src(tmp_path)
    out = _make(src, tmp_path / "f.parquet", "full")
    assert out[_RN].equals(pd.read_parquet(src)[_RN])


def test_parse_dir():
    assert agg._parse_dir("gene_shuffle_seed23") == ("gene_shuffle", 23)
    assert agg._parse_dir("full") == ("full", None)
    assert agg._parse_dir("drop_all") == ("drop_all", None)


def test_aggregator_runs_and_computes_retention(tmp_path, capsys):
    rr = tmp_path / "runs"
    def mj(name, t, v):
        d = rr / name; d.mkdir(parents=True); (d / "metrics.json").write_text(json.dumps({"auroc": t, "val_auroc": v}))
    mj("full", 0.94, 0.95); mj("drop_all", 0.93, 0.94)
    mj("gene_shuffle_seed11", 0.938, 0.948); mj("gene_shuffle_seed23", 0.939, 0.949)
    rc = agg.main(["--runs-root", str(rr)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "retention" in out.lower() and "[verdict]" in out


def test_aggregator_loud_on_missing_keys(tmp_path):
    rr = tmp_path / "runs"; d = rr / "full"; d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({"foo": 1}))
    with pytest.raises(KeyError) as e:
        agg.main(["--runs-root", str(rr)])
    assert "auroc" in str(e.value)
