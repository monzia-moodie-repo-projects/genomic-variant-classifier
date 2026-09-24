"""CI wiring for release publication and candidate validation (owner decision 2026-09-23: publish on release).

Semantics are read from the PARSED workflow, not its text. Until 2026-09-23 push-ghcr could never run: it
needed docker-build, which is skipped on `release`, and GitHub skips a job whose needed job was skipped.
It also rebuilt the image, so the published bytes were never the smoke-tested ones.

Author: Monzia Moodie
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CI = REPO / ".github" / "workflows" / "ci.yml"
SMOKE = REPO / "scripts" / "smoke_test_api_image.sh"
LOCAL_TAG = "gvc-release-candidate:local"


@pytest.fixture(scope="module")
def jobs():
    with CI.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["jobs"]


def _runs_on_release(job) -> bool:
    condition = job.get("if")
    return condition is None or "release" in condition


def test_publication_runs_on_a_published_release_and_every_needed_job_can_run_then(jobs):
    push = jobs["push-ghcr"]
    assert push["if"] == "github.event_name == 'release' && github.event.action == 'published'"
    for needed in push["needs"]:
        assert _runs_on_release(jobs[needed]), f"push-ghcr needs {needed}, which never runs on a release"


def test_the_published_image_is_built_once_smoke_tested_then_pushed(jobs):
    steps = jobs["push-ghcr"]["steps"]
    runs = [s.get("run", "") for s in steps]
    uses = [s.get("uses", "") for s in steps]
    assert not any("build-push-action" in u for u in uses), "a second, untested build would be published"
    build = [i for i, r in enumerate(runs) if "docker build" in r]
    smoke = [i for i, r in enumerate(runs) if "smoke_test_api_image.sh" in r]
    push = [i for i, r in enumerate(runs) if "docker push" in r]
    assert len(build) == 1 and len(smoke) == 1 and len(push) == 1
    assert build[0] < smoke[0] < push[0]
    assert LOCAL_TAG in runs[build[0]] and LOCAL_TAG in runs[smoke[0]] and f"docker tag {LOCAL_TAG}" in runs[push[0]]


def test_candidates_get_the_real_image_build_and_startup_test(jobs):
    job = jobs["docker-build"]
    assert "github.event_name == 'pull_request'" in job["if"]
    assert any("smoke_test_api_image.sh" in s.get("run", "") for s in job["steps"])
    assert not any("docker push" in s.get("run", "") for s in job["steps"])


@pytest.mark.parametrize("name", ["test", "drift"])
def test_each_python_version_reports_for_itself(jobs, name):
    assert jobs[name]["strategy"]["fail-fast"] is False


def test_the_shared_script_holds_the_contract_and_always_cleans_up():
    text = SMOKE.read_text(encoding="utf-8")
    assert ".live == true" in text and ".ready == false" in text
    assert ".model_loaded == false" in text and '.status == "degraded"' in text
    assert "trap cleanup EXIT" in text and "set -euo pipefail" in text


POSIX = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                           reason="executes Linux-runner shell blocks; needs a POSIX bash (CI runs these)")


def _run(script, tmp_path, **env):
    stub = tmp_path / "bin"
    stub.mkdir(exist_ok=True)
    (stub / "docker").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$STUB_LOG"\n', encoding="utf-8")
    (stub / "docker").chmod(0o755)
    log = tmp_path / "docker.log"
    log.write_text("")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={**os.environ, "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}",
                            "STUB_LOG": str(log), **env})
    return r.returncode, log.read_text().splitlines()


@POSIX
def test_the_release_build_passes_each_label_as_one_argument(jobs, tmp_path):
    build = next(s["run"] for s in jobs["push-ghcr"]["steps"] if "docker build" in s.get("run", ""))
    code, calls = _run(build, tmp_path, LABELS="a=b\nc=with spaces = x\n")
    assert code == 0 and len(calls) == 1
    code, calls = _run(build, tmp_path, LABELS="")
    assert code == 0 and "--label" not in calls[0]


@POSIX
def test_the_release_push_refuses_an_empty_tag_list(jobs, tmp_path):
    push = next(s["run"] for s in jobs["push-ghcr"]["steps"] if "docker push" in s.get("run", ""))
    code, calls = _run(push, tmp_path, TAGS="")
    assert code == 1 and not any(c.startswith("push") for c in calls)
    code, calls = _run(push, tmp_path, TAGS="r/i:1\nr/i:sha-x\n")
    assert code == 0 and [c for c in calls if c.startswith(("tag", "push"))] == [
        f"tag {LOCAL_TAG} r/i:1", "push r/i:1", f"tag {LOCAL_TAG} r/i:sha-x", "push r/i:sha-x"]
