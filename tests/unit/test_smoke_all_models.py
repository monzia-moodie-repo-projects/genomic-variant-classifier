"""Unit tests for scripts/smoke_all_models.py streaming (no more silent multi-hour smoke). Monzia Moodie."""
import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_all_models", _SCRIPTS / "smoke_all_models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SM = _load()


def test_stream_child_tees_to_file_and_returns_full_text(tmp_path):
    child = tmp_path / "child.py"
    child.write_text("for i in range(3):\n    print(f'[GNN-TRACE] line {i}', flush=True)\n", encoding="utf-8")
    log = tmp_path / "smoke.log"
    rc, text = SM._stream_child([sys.executable, str(child)], tmp_path, log)
    assert rc == 0
    assert text.count("[GNN-TRACE] line") == 3            # accumulated for the post-run assertions
    assert log.read_text().count("[GNN-TRACE] line") == 3  # AND written to the log file live


def test_stream_child_propagates_nonzero_exit(tmp_path):
    child = tmp_path / "boom.py"
    child.write_text("import sys\nprint('partial', flush=True)\nsys.exit(7)\n", encoding="utf-8")
    rc, text = SM._stream_child([sys.executable, str(child)], tmp_path, tmp_path / "l.log")
    assert rc == 7 and "partial" in text


def test_stream_child_creates_log_parent(tmp_path):
    child = tmp_path / "c.py"; child.write_text("print('x', flush=True)\n", encoding="utf-8")
    log = tmp_path / "nested" / "dir" / "smoke.log"  # parent does not exist yet
    rc, _ = SM._stream_child([sys.executable, str(child)], tmp_path, log)
    assert rc == 0 and log.exists()


def test_subset_clinvar_caps_rows_and_preserves_columns(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "variant_id": [f"v{i}" for i in range(5000)],
        "gene_symbol": [f"GENE{i % 300}" for i in range(5000)],
        "clnsig": [i % 2 for i in range(5000)],
    })
    src = tmp_path / "clinvar_full.parquet"; df.to_parquet(src, index=False)
    out = SM._subset_clinvar(str(src), 500, tmp_path)
    got = pd.read_parquet(out)
    assert len(got) == 500
    assert list(got.columns) == ["variant_id", "gene_symbol", "clnsig"]
    assert got["gene_symbol"].nunique() > 1


def test_subset_clinvar_returns_original_when_already_small(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"variant_id": ["a", "b"], "gene_symbol": ["G1", "G2"]})
    src = tmp_path / "small.parquet"; df.to_parquet(src, index=False)
    assert SM._subset_clinvar(str(src), 1000, tmp_path) == str(src)


def test_subset_clinvar_deterministic(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"variant_id": [f"v{i}" for i in range(2000)],
                       "gene_symbol": [f"G{i % 50}" for i in range(2000)]})
    src = tmp_path / "c.parquet"; df.to_parquet(src, index=False)
    da = tmp_path / "a"; da.mkdir(); db = tmp_path / "b"; db.mkdir()
    a = pd.read_parquet(SM._subset_clinvar(str(src), 200, da))
    b = pd.read_parquet(SM._subset_clinvar(str(src), 200, db))
    assert a["variant_id"].tolist() == b["variant_id"].tolist()
