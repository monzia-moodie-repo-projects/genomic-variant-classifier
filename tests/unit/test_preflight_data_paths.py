#!/usr/bin/env python3
"""test_preflight_data_paths.py -- Monzia Moodie

Unit tests for scripts/preflight_check_data_paths.py: the data-path-health guard must classify a healthy dir,
a stray-file shadow, a DANGLING reparse point (broken symlink == dangling junction), a missing dir, and
missing/present critical assets, with the right exit codes (0 ok / 2 path / 3 asset).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "preflight_check_data_paths.py"


def _load():
    spec = importlib.util.spec_from_file_location("preflight_check_data_paths", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def test_healthy_dir_is_ok(tmp_path):
    (tmp_path / "data").mkdir()
    status, _ = mod.check_path_health(str(tmp_path / "data"))
    assert status == "ok"


def test_stray_file_is_not_a_dir(tmp_path):
    (tmp_path / "data").write_text("x")
    status, _ = mod.check_path_health(str(tmp_path / "data"))
    assert status == "not_a_dir"


def test_missing_is_missing(tmp_path):
    status, _ = mod.check_path_health(str(tmp_path / "nope"))
    assert status == "missing"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink stands in for a Windows dangling junction")
def test_dangling_symlink_is_dangling(tmp_path):
    link = tmp_path / "data"
    link.symlink_to(tmp_path / "absent_target")  # target does not exist -> dangling
    status, msg = mod.check_path_health(str(link))
    assert status == "dangling"
    assert "rmdir" in msg and "git checkout" in msg  # actionable remediation present


def test_not_writable(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    # force the write probe to fail without chmod games (portable)
    real_write = Path.write_text
    def boom(self, *a, **k):
        if ".preflight_write_test_" in self.name:
            raise OSError("read-only (simulated)")
        return real_write(self, *a, **k)
    monkeypatch.setattr(Path, "write_text", boom)
    status, _ = mod.check_path_health(str(d), must_be_writable=True)
    assert status == "not_writable"


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs").mkdir()
    # all healthy, no assets -> 0
    assert mod.main(["--dir", "data", "--dir", "outputs"]) == 0
    # missing asset -> 3
    assert mod.main(["--dir", "data", "--asset", "data/external/spliceai/spliceai_index.parquet"]) == 3
    # present asset -> 0
    p = tmp_path / "data" / "external" / "spliceai"
    p.mkdir(parents=True)
    (p / "spliceai_index.parquet").write_text("x")
    assert mod.main(["--dir", "data", "--asset", "data/external/spliceai/spliceai_index.parquet"]) == 0
    # a bad dir -> 2
    assert mod.main(["--dir", "definitely_absent"]) == 2
