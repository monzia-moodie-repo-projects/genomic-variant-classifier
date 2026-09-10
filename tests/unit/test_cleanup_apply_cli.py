"""The cleanup_apply entry point: application is unavailable, before any work.

Author: Monzia Moodie

The decisive property is ORDER: `--apply` must refuse before repository
selection or discovery, so no code capable of mutation is reached and the
refusal can never be mistaken for a completed dry run.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: MEASURED 2026-09-10 by the pending-unit fit census: scripts/cleanup_apply.py
#: DOES NOT EXIST at 48913c1. The file is scripts/forensics/cleanup_apply.py --
#: named correctly in the deletion audit and in every earlier report, and
#: dropped when the unit's target list was transcribed. The wrong path had
#: propagated into the script's own bootstrap as well.
_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "forensics"
           / "cleanup_apply.py")
ENV = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)


@pytest.fixture(scope="module")
def adapter():
    spec = importlib.util.spec_from_file_location("_cleanup_apply", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], capture_output=True,
                   env=ENV, check=True)
    for pair in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(repo), "config", *pair],
                       capture_output=True, env=ENV, check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], capture_output=True,
                   env=ENV, check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "v1"],
                   capture_output=True, env=ENV, check=True)
    return repo


def test_apply_refuses_before_selection_or_discovery(adapter, monkeypatch,
                                                     capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not run for an application request")

    monkeypatch.setattr(adapter, "inspect_cleanup_candidates", forbidden)
    monkeypatch.setattr(adapter, "select_repository", forbidden)
    assert adapter.main(["--apply"]) == adapter.APPLICATION_UNAVAILABLE
    assert "No cleanup action was attempted" in capsys.readouterr().err


def test_apply_refusal_is_not_a_successful_dry_run(adapter, monkeypatch,
                                                   capsys):
    monkeypatch.setattr(adapter, "select_repository",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not run")))
    code = adapter.main(["--apply"])
    captured = capsys.readouterr()
    assert code != 0
    assert "inspection_status" not in captured.out


def test_the_script_has_no_route_to_a_destructive_module():
    """The adapter imports the read-only owners only. No executor exists to
    reach, dormant or otherwise."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert "cleanup_authorization" not in source
    assert "apply_plan" not in source
    assert "unlink" not in source


#: The modules this assertion MUST examine. Naming them turns "the glob
#: matched nothing" into a failure instead of a pass.
_REQUIRED_LIBRARY_MODULES = ("cleanup_categories.py", "repository_inspection.py")


def _library_package() -> Path:
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "src" / "genomic_variant_classifier"
                      / "repository_hygiene",
                      root / "genomic_variant_classifier"
                      / "repository_hygiene"):
        if candidate.is_dir():
            return candidate
    raise AssertionError(
        "the library package was not found under {}. A test that examines "
        "nothing must fail, not pass.".format(root))


def test_the_library_does_not_touch_sys_path_at_import_time():
    """MEASURED as a test-validity defect: the previous version globbed a
    directory that might not exist, examined nothing and passed."""
    package = _library_package()
    present = {module.name for module in package.glob("*.py")}
    missing = [name for name in _REQUIRED_LIBRARY_MODULES
               if name not in present]
    assert not missing, "expected modules absent from {}: {}".format(package,
                                                                     missing)
    examined = 0
    for name in _REQUIRED_LIBRARY_MODULES:
        assert "sys.path" not in (package / name).read_text(encoding="utf-8"), \
            package / name
        examined += 1
    assert examined == len(_REQUIRED_LIBRARY_MODULES)


def test_root_invocation_inspects_and_exits_zero(adapter, tmp_path, capsys):
    repo = _repo(tmp_path)
    assert adapter.main(["--repo-root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "application_status    : unavailable" in out
    assert "authorization_status  : not_established" in out
    assert "deletion_attempts     : 0" in out


def test_subdirectory_invocation_is_refused(adapter, tmp_path, capsys):
    repo = _repo(tmp_path)
    assert adapter.main(["--repo-root", str(repo / "sub")]) == \
        adapter.INSPECTION_FAILED
    assert "SUBDIRECTORY" in capsys.readouterr().err


def test_a_non_repository_exits_nonzero(adapter, tmp_path, capsys):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert adapter.main(["--repo-root", str(plain)]) == \
        adapter.INSPECTION_FAILED
    assert "Repository selection failed" in capsys.readouterr().err
