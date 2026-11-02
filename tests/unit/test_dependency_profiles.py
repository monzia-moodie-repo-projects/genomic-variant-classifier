"""The dependency profiles' committed locks, and the check that guards them (scripts/check_dependency_profiles.py).

Positive control: the REAL committed files satisfy the offline check. Negative controls: copies with one defect each are refused or
reported, so the positive result is not vacuous. Profiles: train-test (DEPENDENCY-PROFILES-1) and drift (DRIFT-PROFILE-LOCK).
Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.deps.governance import ARTIFACTS
from genomic_variant_classifier.deps.profiles import DRIFT, PROFILES, TRAIN_TEST, install_command

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


def _norm(pins: dict) -> dict:
    return {re.sub(r"[-_.]+", "-", k).lower(): v for k, v in pins.items()}


def test_the_committed_locks_and_sources_satisfy_the_offline_check():
    assert _module().offline(ROOT) == []


def test_the_locks_are_lf_ascii_and_name_their_own_files_and_constraint_in_their_headers():
    for p in PROFILES.values():
        for lock, _ in p.locks:
            raw = (ROOT / lock).read_bytes()
            assert b"\r\n" not in raw and raw.isascii(), lock
            text = raw.decode()
            assert "--output-file={} ".format(lock) in text and "--constraint={} ".format(p.constraints) in text, lock
            assert text.splitlines()[4].endswith(" " + " ".join(sorted(p.sources))), lock      # the command line names its sources


def test_the_two_train_test_locks_agree_wherever_both_pin_a_distribution():
    # Measured 2026-11-01: all 152 Windows pins are in the Linux lock at the same version; Linux adds the CUDA runtime, triton,
    # nvidia-nccl-cu12, pybigwig and hf-xet (HF-XET-WINDOWS-MARKER-1: huggingface_hub's marker reads "amd64", Windows reports "AMD64").
    mod = _module()
    lin = _norm(mod._pins(ROOT / "requirements-train-test-linux.lock", "linux-3.12"))
    win = _norm(mod._pins(ROOT / "requirements-train-test-windows.lock", "windows-3.12"))
    assert set(win) <= set(lin) and all(lin[k] == win[k] for k in win)
    assert "hf-xet" in set(lin) - set(win) and "pybigwig" in set(lin) - set(win)


def test_the_two_drift_locks_agree_wherever_both_pin_a_distribution():
    # Measured 2026-11-02 (block D1 on the owner's machine): 135 shared pins at equal versions; colorama is Windows-only (click,
    # pytest and rich-click require it on win32); nvidia-nccl-cu12 (required by xgboost on Linux) and uvloop (by uvicorn, except on
    # win32) are Linux-only.
    mod = _module()
    lin = _norm(mod._pins(ROOT / "requirements-drift-linux.lock", "linux-3.12"))
    win = _norm(mod._pins(ROOT / "requirements-drift-windows.lock", "windows-3.12"))
    assert all(lin[k] == win[k] for k in set(lin) & set(win)), {k: (lin[k], win[k]) for k in set(lin) & set(win) if lin[k] != win[k]}
    assert set(win) - set(lin) == {"colorama"} and set(lin) - set(win) == {"nvidia-nccl-cu12", "uvloop"}


def test_the_drift_profile_departs_from_training_exactly_where_declared():
    # Measured 2026-11-02: 74 distributions shared on Linux, equal except the three declared departures.
    mod = _module()
    for t in DRIFT.targets:
        drift = _norm(mod._pins(ROOT / DRIFT.lock_for(t), t))
        train = _norm(mod._pins(ROOT / TRAIN_TEST.lock_for(t), t))
        differ = {k for k in set(drift) & set(train) if drift[k] != train[k]}
        assert differ == {n for n, _ in DRIFT.departures}, (t, differ)
        assert not {re.sub(r"[-_.]+", "-", n).lower() for n, _ in DRIFT.forbidden} & set(drift), t
    assert {k: _norm(mod._pins(ROOT / "requirements-drift-linux.lock", "linux-3.12"))[k] for k in ("numpy", "pandas", "scikit-learn", "pyarrow")} == {
        "numpy": "2.4.4", "pandas": "2.3.3", "scikit-learn": "1.8.0", "pyarrow": "23.0.1"}


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


# ---- the drift profile (DRIFT-PROFILE-LOCK) ----

def test_the_drift_constraints_are_exactly_their_derivation_and_the_command_rewrites_them_byte_for_byte(copy):
    mod = _module()
    committed = (ROOT / "constraints-drift.txt").read_bytes()
    assert mod.derive_constraints(ROOT, DRIFT).encode("ascii") == committed
    (copy / "constraints-drift.txt").write_text("stale\n")
    assert mod.main(["--derive-constraints", "drift", "--root", str(copy)]) == 0
    assert (copy / "constraints-drift.txt").read_bytes() == committed
    lines = [l for l in committed.decode().splitlines() if not l.startswith("#")]
    assert len(lines) == 171 and not {"lightgbm", "plotly", "xgboost"} & {l.split("==")[0] for l in lines}


def test_an_edited_drift_constraint_is_reported_twice(copy):
    _edit(copy / "constraints-drift.txt", "\nnumpy==2.4.4\n", "\nnumpy==2.4.3\n")
    lines = _module().offline(copy)
    assert any("constraints-drift.txt is not the derivation" in l for l in lines)
    assert {l.split(" [")[1].split("]")[0] for l in lines if "constraint_violated: numpy" in l} == {"linux-3.11", "linux-3.12", "windows-3.12"}


def test_training_locks_that_disagree_with_each_other_cannot_be_derived_from(copy):
    _edit(copy / "requirements-train-test-linux.lock", "pyarrow==23.0.1 \\", "pyarrow==23.0.2 \\")
    with pytest.raises(_module().ContractRefusal, match="pyarrow differs across the train-test locks"):
        _module().offline(copy)


def test_a_departure_that_no_longer_departs_is_reported(copy):
    _edit(copy / "requirements-drift-linux.lock", "lightgbm==4.5.0 \\", "lightgbm==4.6.0 \\")
    lines = _module().offline(copy)
    assert {l.split(" [")[1].split("]")[0] for l in lines if "departure lightgbm does not depart" in l} == {"linux-3.11", "linux-3.12"}
    assert any("source_conflict: lightgbm" in l for l in lines)


def test_a_forbidden_distribution_in_a_drift_lock_is_reported(copy):
    lock = copy / "requirements-drift-linux.lock"
    lock.write_text(lock.read_text() + "pyspark==3.5.8 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    lines = _module().offline(copy)
    assert {l.split(" [")[1].split("]")[0] for l in lines if "forbidden: pyspark" in l} == {"linux-3.11", "linux-3.12"}


def _observation(tmp_path, rows, env):
    o = {"schema": "gvc.environment-observation/1", "distributions": rows, "marker_environment": env,
         "interpreter": {"is_virtual_environment": True, "pyvenv_cfg": "include-system-site-packages = false\n", "prefix": "/v"},
         "pip_check": {"exit_code": 0, "stdout": "No broken requirements found.\n"}}
    p = tmp_path / "obs.json"
    p.write_text(json.dumps(o))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(pins):
    rows = [{"name": k, "version": v, "requires_dist": [], "provides_extra": [], "metadata_path": "/s/" + k} for k, v in pins.items()]
    return rows + [{"name": "pip", "version": "26.1.2", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/pip"},
                   {"name": "genomic-variant-classifier", "version": "0.1.0", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/g"}]


def test_environment_mode_judges_an_exact_windows_install_and_refuses_one_extra(tmp_path):
    mod = _module()
    rows = _rows(mod._pins(ROOT / "requirements-train-test-windows.lock", "windows-3.12"))
    p, h = _observation(tmp_path, rows, mod.TARGETS["windows-3.12"])
    assert mod.environment(ROOT, "train-test", p, h, "windows-3.12") == []
    rows.append({"name": "Faker", "version": "40.15.0", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/faker"})
    p, h = _observation(tmp_path, rows, mod.TARGETS["windows-3.12"])
    assert mod.environment(ROOT, "train-test", p, h, "windows-3.12") == ["unexpected: faker 40.15.0"]


def test_environment_mode_judges_a_drift_install_against_the_drift_lock_and_refuses_pyspark(tmp_path):
    mod = _module()
    rows = _rows(mod._pins(ROOT / "requirements-drift-linux.lock", "linux-3.11"))
    p, h = _observation(tmp_path, rows, mod.TARGETS["linux-3.11"])
    assert mod.environment(ROOT, "drift", p, h, "linux-3.11") == []
    # The same environment is NOT the train-test lock: the profile decides which lock judges it.
    assert any(l.startswith("missing: torch ") for l in mod.environment(ROOT, "train-test", p, h, "linux-3.11"))
    rows.append({"name": "pyspark", "version": "3.5.8", "requires_dist": [], "provides_extra": [], "metadata_path": "/s/pyspark"})
    p, h = _observation(tmp_path, rows, mod.TARGETS["linux-3.11"])
    assert mod.environment(ROOT, "drift", p, h, "linux-3.11") == ["unexpected: pyspark 3.5.8", "forbidden: pyspark "]


def test_the_command_line_exit_codes(tmp_path, copy, capsys):
    mod = _module()
    assert mod.main(["--offline", "--root", str(ROOT)]) == 0
    _edit(copy / "requirements.txt", "pillow==12.2.0", "pillow==12.1.1")
    assert mod.main(["--offline", "--root", str(copy)]) == 1
    (copy / "requirements-train-test-linux.lock").write_text("not a requirement line ===\n")
    assert mod.main(["--offline", "--root", str(copy)]) == 2
    assert "REFUSED" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:          # the profile is never guessed: a target belongs to several locks
        mod.main(["--environment", str(tmp_path / "o.json"), "--target", "linux-3.11", "--observation-sha256", "0" * 64])
    assert exc.value.code == 2


# ---- the profile declaration and the installer (scripts/install_locked_profile.py) ----

def test_the_profile_declaration_imports_with_the_standard_library_alone():
    # The installer imports it with the BASE interpreter, before any environment exists.
    code = ("import sys; sys.path.insert(0, sys.argv[1]); before = set(sys.modules); "
            "import genomic_variant_classifier.deps.profiles as p; "
            "extra = sorted(m for m in set(sys.modules) - before if not m.startswith(('genomic_variant_classifier', '_')) "
            "and m.split('.')[0] not in sys.stdlib_module_names); print(extra)")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src")], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]", out


def test_every_file_a_profile_names_is_a_declared_dependency_artifact():
    declared = set(FILES)
    for p in PROFILES.values():
        named = {*p.sources, p.constraints, *(lock for lock, _ in p.locks)} | ({p.build_constraints} if p.build_constraints else set())
        assert named <= declared, (p.name, named - declared)
    assert install_command("drift").startswith("python scripts/install_locked_profile.py --profile drift --target TARGET --venv NEW_FOLDER")


def _installer():
    spec = importlib.util.spec_from_file_location("_install_locked_profile", ROOT / "scripts" / "install_locked_profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_installer_takes_the_pip_record_exactly_and_refuses_a_malformed_one():
    mod = _installer()
    text = (ROOT / "requirements-generator-toolchain.txt").read_text(encoding="utf-8")
    record = mod.pip_record(text)
    lines = record.splitlines()
    assert lines[0] == "pip==26.1.2 \\" and len(lines) >= 2 and all(l.strip().startswith("--hash=sha256:") for l in lines[1:])
    assert record in text
    with pytest.raises(ValueError, match="exactly one line"):
        mod.pip_record(text.replace("pip==26.1.2 \\", "pip==26.1.3 \\"))
    with pytest.raises(ValueError, match="exactly one line"):
        mod.pip_record(text + "\n" + record)
    with pytest.raises(ValueError, match="not a pin followed only by hashes"):
        mod.pip_record("pip==26.1.2 \\\n    --hash=md5:00\n")


def test_the_installer_refuses_an_existing_environment_before_creating_anything(tmp_path, capsys):
    mod = _installer()
    existing = tmp_path / "env"
    existing.mkdir()
    assert mod.main(["--profile", "drift", "--target", "linux-3.12", "--venv", str(existing)]) == 2
    assert "already exists" in capsys.readouterr().out and list(existing.iterdir()) == []
    (tmp_path / "env2.observation.json").write_text("{}")
    assert mod.main(["--profile", "drift", "--target", "linux-3.12", "--venv", str(tmp_path / "env2")]) == 2
    assert not (tmp_path / "env2").exists()


def test_the_installer_refuses_a_missing_lock_before_creating_anything(tmp_path, copy, capsys, monkeypatch):
    mod = _installer()
    (copy / "requirements-drift-linux.lock").unlink()
    monkeypatch.setattr(mod, "ROOT", copy)
    assert mod.main(["--profile", "drift", "--target", "linux-3.12", "--venv", str(tmp_path / "env")]) == 2
    assert "is missing" in capsys.readouterr().out and not (tmp_path / "env").exists()
