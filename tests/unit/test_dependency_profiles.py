"""The training-and-test profile's committed locks, and the check that guards them (scripts/check_dependency_profiles.py).

Positive control: the REAL committed files satisfy the offline check. Negative controls: copies with one defect each are refused or
reported, so the positive result is not vacuous. Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from genomic_variant_classifier.deps.governance import ARTIFACTS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
# Every declared dependency file, from the ONE declaration (a second hand-kept list here missed the build constraints once).
FILES = tuple(a.path for a in ARTIFACTS)


def _module():
    spec = importlib.util.spec_from_file_location("_check_dependency_profiles", ROOT / "scripts" / "check_dependency_profiles.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def copy(tmp_path):
    for name in FILES:
        shutil.copy2(ROOT / name, tmp_path / name)
    return tmp_path


def test_the_committed_locks_and_sources_satisfy_the_offline_check():
    assert _module().offline(ROOT) == []


def test_the_locks_are_lf_ascii_and_name_their_own_files_in_their_headers():
    for lock in ("requirements-train-test-linux.lock", "requirements-train-test-windows.lock"):
        raw = (ROOT / lock).read_bytes()
        assert b"\r\n" not in raw and raw.isascii()
        assert "--output-file={} ".format(lock) in raw.decode() and "--constraint=constraints-observed-venv312.txt" in raw.decode()


def test_the_two_locks_agree_wherever_both_pin_a_distribution():
    # Measured 2026-11-01: all 152 Windows pins are in the Linux lock at the same version; Linux adds the CUDA runtime, triton,
    # nvidia-nccl-cu12, pybigwig and hf-xet (HF-XET-WINDOWS-MARKER-1: huggingface_hub's marker reads "amd64", Windows reports "AMD64").
    mod = _module()
    lin, win = mod._pins(ROOT / "requirements-train-test-linux.lock", "linux-3.12"), mod._pins(ROOT / "requirements-train-test-windows.lock", "windows-3.12")
    norm = lambda d: {re.sub(r"[-_.]+", "-", k).lower(): v for k, v in d.items()}  # noqa: E731
    lin, win = norm(lin), norm(win)
    assert set(win) <= set(lin) and all(lin[k] == win[k] for k in win)
    assert "hf-xet" in set(lin) - set(win) and "pybigwig" in set(lin) - set(win)


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


def test_a_lock_record_without_hashes_is_refused(copy):
    lock = copy / "requirements-train-test-windows.lock"
    text = lock.read_text()
    start = text.index("pillow==12.2.0 \\\n")
    end = text.index("\n", text.index("    # via", start))
    lock.write_text(text[:start] + "pillow==12.2.0\n" + text[end + 1:])
    with pytest.raises(_module().ContractRefusal, match="carries no hash"):
        _module().offline(copy)


def test_a_declaration_the_lock_does_not_satisfy_is_reported_for_every_target(copy):
    _edit(copy / "requirements.txt", "pillow==12.2.0", "pillow==12.1.1")
    lines = _module().offline(copy)
    assert {l.split(" [")[1].split("]")[0] for l in lines if "source_conflict: pillow" in l} == {"linux-3.11", "linux-3.12", "windows-3.12"}


def test_a_lock_that_departs_from_a_measured_constraint_is_reported(copy):
    _edit(copy / "constraints-observed-venv312.txt", "pillow==12.2.0", "pillow==12.1.1")
    assert any("constraint_violated: pillow" in l for l in _module().offline(copy))


def test_a_toolchain_lock_that_disagrees_with_its_source_is_reported(copy):
    _edit(copy / "requirements-generator-toolchain.in", "click==8.3.1", "click==8.5.0")
    assert any("requirements-generator-toolchain.txt pins" in l for l in _module().offline(copy))


def test_a_build_constraint_that_departs_from_the_lock_is_reported(copy):
    _edit(copy / "build-constraints-train-test.txt", "setuptools==70.2.0", "setuptools==84.2.1")
    lines = _module().offline(copy)
    assert any("differs from the generator toolchain" in l for l in lines) and any("requirements-train-test-linux.lock's 70.2.0" in l for l in lines)


def test_two_active_clauses_for_one_name_are_refused(copy):
    lock = copy / "requirements-train-test-windows.lock"
    text = lock.read_text()
    block = text[text.index("pillow==12.2.0 \\\n"):]
    block = block[:block.index("    # via")]
    lock.write_text(text + block.rstrip("\\\n") + "\n")
    with pytest.raises(_module().ContractRefusal, match="two clauses"):
        _module().offline(copy)


def _observation(tmp_path, rows, env):
    o = {"schema": "gvc.environment-observation/1", "distributions": rows, "marker_environment": env,
         "interpreter": {"is_virtual_environment": True, "pyvenv_cfg": "include-system-site-packages = false\n", "prefix": "/v"},
         "pip_check": {"exit_code": 0, "stdout": "No broken requirements found.\n"}}
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(o))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def test_environment_mode_judges_an_exact_windows_install_and_refuses_one_extra(tmp_path):
    mod = _module()
    pins = mod._pins(ROOT / "requirements-train-test-windows.lock", "windows-3.12")
    rows = [{"name": k, "version": v, "requires_dist": [], "provides_extra": [], "metadata_path": "/s/" + k} for k, v in pins.items()]
    rows += [{"name": "pip", "version": "26.1.2", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/pip"},
             {"name": "genomic-variant-classifier", "version": "0.1.0", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/g"}]
    p, h = _observation(tmp_path, rows, mod.TARGETS["windows-3.12"])
    assert mod.environment(ROOT, p, h, "windows-3.12") == []
    rows.append({"name": "Faker", "version": "40.15.0", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/faker"})
    p, h = _observation(tmp_path, rows, mod.TARGETS["windows-3.12"])
    assert mod.environment(ROOT, p, h, "windows-3.12") == ["unexpected: faker 40.15.0"]


def test_the_command_line_exit_codes(tmp_path, copy, capsys):
    mod = _module()
    assert mod.main(["--offline", "--root", str(ROOT)]) == 0
    _edit(copy / "requirements.txt", "pillow==12.2.0", "pillow==12.1.1")
    assert mod.main(["--offline", "--root", str(copy)]) == 1
    (copy / "requirements-train-test-linux.lock").write_text("not a requirement line ===\n")
    assert mod.main(["--offline", "--root", str(copy)]) == 2
    assert "REFUSED" in capsys.readouterr().out
