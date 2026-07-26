"""Pin the native Arrow read that removes the teardown abort (2026-07-23).

WHY THIS EXISTS
===============
scripts/make_rnaseq_ablation_parquet.py used pandas.read_parquet. pandas hands
Arrow a Python file handle, which Arrow wraps in arrow::py::PyReadableFile. That
wrapper holds a Python object reference, so its destructor calls PyGILState_Ensure
to release it. When that destructor runs on an Arrow background thread after
interpreter finalisation has begun, CPython's take_gil (Python/ceval_gil.c:353)
kills the thread with pthread_exit; the forced unwind propagates through C++
destructor frames that cannot survive it, and libstdc++ calls std::terminate. The
process aborts with SIGABRT and prints "terminate called without an active
exception" AFTER its work has completed and its success line has been printed.

Continuous Integration run 29962715186 (run number 585, commit 821a990) failed
exactly this way. Twenty-seven core dumps from the diagnostic all carry the same
frame chain; PyThread_exit_thread, _Unwind_ForcedUnwind and std::terminate each
appear exactly once per core.

pq.read_table opens the file natively in C++, so no Python object is wrapped and
the aborting destructor is never constructed. Measured on the Continuous
Integration runner at five thousand executions per arm in a single run:
pandas.read_parquet aborted twenty-seven times, and the native read aborted zero
times in each of two independent arms.

These tests pin that fix. If someone reintroduces pandas.read_parquet, the abort
returns silently and rarely -- roughly one run in two hundred -- which is exactly
the kind of defect that survives for months. The suite says so immediately instead.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "make_rnaseq_ablation_parquet.py")
_RNASEQ_COLUMNS = (
    "rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
    "rnaseq_log2fc", "rnaseq_de_neglog10p",
)


def _source() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _make_source_parquet(path: pathlib.Path, n: int = 60) -> None:
    rng = np.random.default_rng(0)
    pd.DataFrame({"gene_symbol": [f"G{i}" for i in range(n)],
                  **{c: rng.random(n) for c in _RNASEQ_COLUMNS}}
                 ).to_parquet(path, index=False)


def test_the_script_does_not_call_pandas_read_parquet():
    """The faulting call must stay gone. A comment mentioning it is fine; a real
    call is not, so this walks the syntax tree rather than grepping the text."""
    offenders = []
    for node in ast.walk(ast.parse(_source())):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "read_parquet":
            owner = func.value
            name = owner.id if isinstance(owner, ast.Name) else ""
            offenders.append(f"{name}.read_parquet at line {node.lineno}")
    assert not offenders, (
        "pandas.read_parquet reintroduces arrow::py::PyReadableFile and with it a "
        f"rare SIGABRT at interpreter teardown: {offenders}")


def test_the_script_reads_through_pyarrow_natively():
    tree = ast.parse(_source())
    reads_natively = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_table"
        for node in ast.walk(tree))
    assert reads_natively, (
        "the script must read through pyarrow.parquet.read_table, which opens the "
        "file natively in C++ and never wraps a Python file object")


def test_the_native_read_matches_the_pandas_read(tmp_path):
    """The fix must not change the data. Both readers are compared frame to frame
    on the same file, so equivalence is demonstrated rather than assumed."""
    import pyarrow.parquet as pq
    src = tmp_path / "src.parquet"
    _make_source_parquet(src)
    via_pandas = pd.read_parquet(src)
    via_arrow = pq.read_table(src).to_pandas()
    pd.testing.assert_frame_equal(via_pandas, via_arrow)


@pytest.mark.parametrize("mode", ["full", "drop_all", "drop_de", "gene_shuffle"])
def test_every_mode_still_runs_and_exits_zero(tmp_path, mode):
    """End to end through the real script: the exit code must be a faithful signal
    of success, and the output must still carry every rnaseq column."""
    src = tmp_path / "src.parquet"
    _make_source_parquet(src)
    out = tmp_path / f"{mode}.parquet"
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--src", str(src), "--out", str(out),
         "--mode", mode, "--seed", "0"],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    frame = pd.read_parquet(out)
    assert len(frame) == 60
    for column in _RNASEQ_COLUMNS:
        assert column in frame.columns
