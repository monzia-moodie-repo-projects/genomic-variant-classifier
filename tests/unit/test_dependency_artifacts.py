"""Every dependency file and every install path is declared, and each declaration is true of the repository. Author: Monzia Moodie"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from genomic_variant_classifier.deps.governance import ARTIFACTS, INSTALL_CONSUMERS, artifact_for
from genomic_variant_classifier.deps.model import ArtifactAuthority, ArtifactPurpose
from genomic_variant_classifier.deps.profiles import PROFILES

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"^(requirements[^/]*\.(txt|in|lock)|(build-)?constraints[^/]*\.txt)$")


#: The one installer of a locked profile (DRIFT-PROFILE-LOCK). A file that calls it installs packages exactly as surely as a file that
#: runs `pip install`, so the census counts both; without this, migrating a workflow to the installer would make it vanish from the
#: census instead of being recorded as lock-qualified.
INSTALLER = "scripts/install_locked_profile.py"


def _install_lines(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.strip()
        if body.startswith("#") or body.startswith("echo"):
            continue
        if re.search(r"\bpip install\b", body) or INSTALLER in body:
            out.append(body)
    return out


def _consumer_files() -> set:
    candidates = list((ROOT / ".github" / "workflows").glob("*.yml")) + [ROOT / "Dockerfile"] + list((ROOT / "scripts").glob("*.sh"))
    return {str(p.relative_to(ROOT)).replace("\\", "/") for p in candidates if p.is_file() and _install_lines(p)}


def test_every_dependency_file_at_the_root_is_declared_exactly_once():
    on_disk = {p.name for p in ROOT.iterdir() if p.is_file() and PATTERN.match(p.name)}
    declared = [a.path for a in ARTIFACTS]
    assert len(declared) == len(set(declared))
    assert on_disk == set(declared), {"undeclared": sorted(on_disk - set(declared)), "missing": sorted(set(declared) - on_disk)}


def test_a_generated_artifact_names_declared_sources_and_an_installation_lock_is_hashed_throughout():
    for a in ARTIFACTS:
        assert all(artifact_for(s) is not None for s in a.generated_from), a.path
        if a.authority is ArtifactAuthority.GENERATED and a.purpose in (ArtifactPurpose.INSTALLATION, ArtifactPurpose.GENERATOR_TOOLCHAIN):
            text = (ROOT / a.path).read_text(encoding="utf-8")
            records = re.findall(r"(?m)^[A-Za-z0-9][^\n]*==[^\n]*$", text)
            assert records and all(r.rstrip().endswith("\\") for r in records), a.path       # every record continues into its hashes


def test_no_install_path_installs_from_a_reference_resolution():
    references = [a.path for a in ARTIFACTS if a.purpose is ArtifactPurpose.REFERENCE_RESOLUTION]
    for consumer in _consumer_files():
        for line in _install_lines(ROOT / consumer):
            assert not any(re.search(r"(?<![\w-])" + re.escape(ref) + r"\b", line) for ref in references), (consumer, line)


def test_every_install_path_has_a_declared_migration_status():
    declared = {c.path for c in INSTALL_CONSUMERS}
    found = _consumer_files()
    assert found == declared, {"undeclared": sorted(found - declared), "declared but installs nothing": sorted(declared - found)}
    assert all(c.status for c in INSTALL_CONSUMERS)


def _installer_calls():
    """(workflow, job, profile, targets, venv, later step runs) for every call of the installer in a workflow."""
    calls = []
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        for job_id, job in (doc.get("jobs") or {}).items():
            steps = job.get("steps") or []
            for i, step in enumerate(steps):
                run = step.get("run") or ""
                if INSTALLER not in run:
                    continue
                text = run.replace("\\\n", " ")
                m = re.search(r'--profile\s+(\S+)\s+--target\s+(?:"([^"]+)"|(\S+))\s+--venv\s+(?:"([^"]+)"|(\S+))', text)
                assert m, (wf.name, job_id, run)
                profile, target, venv = m.group(1), m.group(2) or m.group(3), m.group(4) or m.group(5)
                if "${{ matrix.python-version }}" in target:
                    versions = job["strategy"]["matrix"]["python-version"]
                    targets = [target.replace("${{ matrix.python-version }}", str(v)) for v in versions]
                else:
                    targets = [target]
                calls.append((wf.name, job_id, profile, targets, venv, [s.get("run") or "" for s in steps[i + 1:]]))
    return calls


def test_every_installer_call_names_a_declared_profile_and_targets_that_profile_has_a_lock_for():
    calls = _installer_calls()
    assert {(w, j) for w, j, *_ in calls} == {("ci.yml", "test"), ("ci.yml", "drift"), ("drift_monitor.yml", "feature-drift")}
    for wf, job, profile, targets, venv, _ in calls:
        assert profile in PROFILES, (wf, job, profile)
        assert targets and all(t in PROFILES[profile].targets for t in targets), (wf, job, targets)


def test_the_step_after_every_installer_call_proves_the_job_now_runs_that_environment():
    for wf, job, profile, targets, venv, later in _installer_calls():
        assert later, (wf, job)
        assert 'test "$(command -v python)" = "{}/bin/python"'.format(venv) in later[0], (wf, job, later[0])


def test_every_workflow_that_calls_the_installer_retries_a_dropped_download():
    # A dropped download of the 23 MB nannyml wheel failed ci.yml's drift job on 2026-07-25 (run 30178381817); the monthly drift
    # monitor installs the same lock since 2026-11-02, so it carries the same resilience (pip reads these for every invocation).
    for wf in sorted({w for w, *_ in _installer_calls()}):
        env = yaml.safe_load((ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")).get("env") or {}
        assert int(env.get("PIP_RETRIES", 0)) >= 3 and int(env.get("PIP_DEFAULT_TIMEOUT", 0)) >= 30, (wf, env)
