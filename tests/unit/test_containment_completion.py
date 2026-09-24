"""Containment completion (owner ruling 2026-09-23): the kernel's admission contract and denied execution.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from genomic_variant_classifier.containment import (
    ContainmentError,
    load_after_admission,
    require_execution_permitted,
)
from genomic_variant_classifier.quarantine_policy import HISTORICAL_EXECUTION_DENIED

REPO = Path(__file__).resolve().parents[2]


def _artifact(tmp_path):
    p = tmp_path / "model.pkl"
    p.write_bytes(pickle.dumps({"model": "synthetic"}))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


class TestAdmitMustRaiseToRefuse:
    @pytest.mark.parametrize("returned", [False, True, 0, "allow"])
    def test_a_returned_value_is_refused_before_the_file_is_opened(self, tmp_path, returned):
        path, digest = _artifact(tmp_path)
        deserialized = []
        with mock.patch.object(Path, "open", side_effect=AssertionError("opened")) as opened:
            with pytest.raises(ContainmentError, match="must raise to refuse"):
                load_after_admission(path, admit=lambda: returned, artifact_sha256=digest,
                                     deserialize=lambda fh: deserialized.append(1))
        opened.assert_not_called()
        assert deserialized == []

    def test_positive_control_none_admits_and_loads_the_verified_bytes(self, tmp_path):
        path, digest = _artifact(tmp_path)
        out = load_after_admission(path, admit=lambda: None, artifact_sha256=digest, deserialize=pickle.load)
        assert out == {"model": "synthetic"}


class TestRequireExecutionPermitted:
    def test_a_denied_script_refuses_with_its_reason(self):
        with pytest.raises(ContainmentError, match="Execution denied: scripts/x.py: because"):
            require_execution_permitted("scripts/x.py", {"scripts/x.py": "because"})

    def test_an_unlisted_script_is_not_refused(self):
        require_execution_permitted("scripts/y.py", {"scripts/x.py": "because"})

    def test_a_denial_without_a_reason_is_itself_refused(self):
        with pytest.raises(ContainmentError, match="stated reason"):
            require_execution_permitted("scripts/y.py", {"scripts/x.py": "  "})


def test_the_denied_set_is_exactly_the_two_scripts_ruled_on():
    assert set(HISTORICAL_EXECUTION_DENIED) == {
        "scripts/diagnose_phase2_prediction_reconstruction.py",
        "scripts/run10b_partial_phase2_eval_v2.py",
    }
    with pytest.raises(TypeError):
        HISTORICAL_EXECUTION_DENIED["scripts/z.py"] = "x"          # read-only policy


@pytest.mark.parametrize("script", sorted(HISTORICAL_EXECUTION_DENIED))
def test_the_guard_is_the_first_statement_after_future_imports(script):
    tree = ast.parse((REPO / script).read_text(encoding="utf-8"))
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    body = [n for n in body if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    first = [ast.unparse(n) for n in body[:3]]
    assert first[0] == "from genomic_variant_classifier.containment import require_execution_permitted"
    assert first[1] == "from genomic_variant_classifier.quarantine_policy import HISTORICAL_EXECUTION_DENIED"
    assert first[2] == f"require_execution_permitted({script!r}, HISTORICAL_EXECUTION_DENIED)"


@pytest.mark.parametrize("script", sorted(HISTORICAL_EXECUTION_DENIED))
def test_running_the_script_refuses_before_any_model_library_or_output(script, tmp_path):
    wrapper = (
        "import runpy, sys\n"
        "from genomic_variant_classifier.containment import ContainmentError\n"
        "try:\n"
        "    runpy.run_path(sys.argv[1], run_name='__main__')\n"
        "except ContainmentError as e:\n"
        "    print('REFUSED', 'joblib' in sys.modules, 'sklearn' in sys.modules, str(e))\n"
        "    sys.exit(0)\n"
        "print('NOT REFUSED'); sys.exit(1)\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", wrapper, str(REPO / script)], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("REFUSED False False Execution denied: " + script), r.stdout
    assert list(tmp_path.iterdir()) == [], "the refused script created files"
