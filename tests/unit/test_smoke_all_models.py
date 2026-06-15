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
