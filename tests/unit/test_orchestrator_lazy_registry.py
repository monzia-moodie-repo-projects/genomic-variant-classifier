"""test_orchestrator_lazy_registry.py -- Monzia Moodie

Phase 1 regression locks for the lazy agent registry. These guarantee the Data Freshness CI failure
cannot silently return: the eager registry pulled sklearn (via ModelInsightsAgent ->
model_insights_detector) and torch (via the EWC chain) at Orchestrator construction, which crashes in a
minimal environment that has neither. The lazy registry defers all agent imports to first use.

The "imports without sklearn" checks run in a SUBPROCESS. Blocking sklearn by mutating this
interpreter's sys.modules / builtins.__import__ would pollute module identity for every later test
(re-imported sklearn classes become new objects -> joblib.dump of sklearn-bearing models fails with
"not the same object"). A child process keeps the parent's module graph pristine.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _run_in_subprocess(body: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    code = textwrap.dedent(body)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p), **(extra_env or {})}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_orchestrator_constructs_without_sklearn(tmp_path):
    """The whole point of Phase 1: construct the Orchestrator with sklearn unavailable."""
    r = _run_in_subprocess(
        """
        import builtins
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        o = Orchestrator(SharedState(state_file=__import__("os").environ["GVC_TEST_STATE_FILE"]), dry_run=True)
        assert len(o._agent_registry) >= 1
        print("CONSTRUCT_NO_SKLEARN_OK", len(o._agent_registry))
        """,
        extra_env={"GVC_TEST_STATE_FILE": str(tmp_path / "state.json")},
    )
    assert "CONSTRUCT_NO_SKLEARN_OK" in r.stdout, (
        "Orchestrator must construct without sklearn (lazy registry).\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_orchestrator_constructs_without_sklearn_or_torch(tmp_path):
    """Stronger: neither sklearn NOR torch available at construction (full minimal-CI simulation)."""
    r = _run_in_subprocess(
        """
        import builtins
        real = builtins.__import__
        _BLOCKED = ("sklearn", "torch", "xgboost", "lightgbm", "catboost", "shap", "transformers")
        def blk(n, *a, **k):
            if any(n == b or n.startswith(b + ".") for b in _BLOCKED):
                raise ModuleNotFoundError(f"No module named '{n}' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        o = Orchestrator(SharedState(state_file=__import__("os").environ["GVC_TEST_STATE_FILE"]), dry_run=True)
        print("CONSTRUCT_NO_HEAVY_OK", len(o._agent_registry))
        """,
        extra_env={"GVC_TEST_STATE_FILE": str(tmp_path / "state.json")},
    )
    assert "CONSTRUCT_NO_HEAVY_OK" in r.stdout, (
        "Orchestrator must construct with all heavy ML deps unavailable (lazy registry).\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )


def test_ci_data_freshness_pipeline_runs_without_sklearn(tmp_path):
    """The exact CI guarantee: the database_monitor pipeline runs to completion with sklearn blocked.

    This is the red->green proof for the Data Freshness workflow: it constructs the Orchestrator and
    runs the freshness pipeline (whose agents do not need sklearn) in a sklearn-free interpreter.

    NO NETWORK (2026-11-05, TESTS-REACH-THE-NETWORK-1). MEASURED: the child's registry scan probed ClinVar's FTP server,
    storage.googleapis.com, alphafold.ebi.ac.uk, api.github.com and databases.lovd.nl, and the root conftest.py's network guard
    cannot see a child interpreter. The child now replaces the detector's probe, before the pipeline constructs the agent, with
    one that makes no call and reports each source unreachable -- the outcome a refused probe gave -- and the test requires that
    the scan went through it (the child prints how many sources it probed) and that the report records its answer.

    NO LOCAL DATA (2026-11-06, DATA-GATED-TESTS-1). The scan's local half (database_freshness_detector.check_local) looks for each
    source's local_path under the agent's root, and the orchestrator builds the agent with root PROJECT_ROOT -- the repository
    itself (DatabaseFreshnessMonitorAgent: root None -> PROJECT_ROOT). MEASURED by the data census: from the owner's checkout the
    child walked his real data/ (81 git-ignored paths: data/raw 65, data/external 13, data/processed 3; sizes of whole directories).
    MEASURED in the sandbox: running the child from an empty current folder changed nothing (24 git-ignored paths of the clone still
    looked at) -- the root is not the current folder. The child now checks every local source under an empty folder, records the
    root the agent passed, and finds exactly the one asset the test planted there.
    """
    (tmp_path / "empty_root").mkdir()
    r = _run_in_subprocess(
        """
        import builtins
        real = builtins.__import__
        def blk(n, *a, **k):
            if n == "sklearn" or n.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn' (blocked for test)")
            return real(n, *a, **k)
        builtins.__import__ = blk
        from genomic_variant_classifier.agent_layer.agents import database_freshness_detector as detector
        probed = []
        def offline_probe(source):
            probed.append(source.key)
            return (None, "offline under test (no probe is made)")
        detector._default_probe = offline_probe
        import os
        from pathlib import Path
        from genomic_variant_classifier.monitoring import registry as registry_
        empty = os.environ["GVC_TEST_EMPTY_ROOT"]
        # ONE planted asset under the empty root: the check must find it (proving it ran there) and nothing else.
        planted = next(s for s in registry_.all_sources() if s.local_path)
        (Path(empty) / planted.local_path).parent.mkdir(parents=True, exist_ok=True)
        (Path(empty) / planted.local_path).write_bytes(b"planted by the test")
        real_check_local = detector.check_local
        local_roots, local_results = [], []
        def check_local_under_an_empty_root(source, *, root="."):
            local_roots.append(str(root))
            result = real_check_local(source, root=empty)
            local_results.append(result)
            return result
        detector.check_local = check_local_under_an_empty_root
        from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
        from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT
        from genomic_variant_classifier.agent_layer.shared_state import SharedState
        o = Orchestrator(SharedState(state_file=__import__("os").environ["GVC_TEST_STATE_FILE"]), dry_run=True)
        results = o.run_pipeline("database_monitor")
        assert probed, "the registry scan probed no source"
        print("PROBED_OFFLINE", len(probed))
        assert local_roots and set(local_roots) == {str(PROJECT_ROOT)}, local_roots
        found = sorted(r.key for r in local_results if r.detail not in ("absent on disk", "no local_path declared"))
        assert found == [planted.key], (found, planted.key)
        print("LOCAL_CHECKED_UNDER_AN_EMPTY_ROOT", len(local_roots), "instead of", local_roots[0], "found only", found)
        assert "DatabaseFreshnessMonitorAgent" in results, results
        action = results["DatabaseFreshnessMonitorAgent"].get("action")
        assert action != "error", f"freshness agent errored under sklearn-block: {results}"
        print("CI_FRESHNESS_OK", action)
        """,
        # The freshness report is an ARTIFACT: GVC_ARTIFACT_ROOT keeps it out of the real reports/.
        extra_env={"GVC_ARTIFACT_ROOT": str(tmp_path), "GVC_TEST_STATE_FILE": str(tmp_path / "state.json"),
                   "GVC_TEST_EMPTY_ROOT": str(tmp_path / "empty_root")},
    )
    assert list((tmp_path / "reports" / "data_freshness").glob("FRESHNESS_*.md")), (
        "the freshness report did not land under GVC_ARTIFACT_ROOT")
    assert "CI_FRESHNESS_OK" in r.stdout, (
        "database_monitor pipeline must run clean without sklearn (the CI guarantee).\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    report = next((tmp_path / "reports" / "data_freshness").glob("FRESHNESS_*.md")).read_text(encoding="utf-8")
    assert "PROBED_OFFLINE" in r.stdout and "offline under test (no probe is made)" in report, report
    assert "LOCAL_CHECKED_UNDER_AN_EMPTY_ROOT" in r.stdout, r.stdout


def test_registry_values_are_lazy(tmp_path):
    """Registry values must be _Lazy (not eagerly-imported classes)."""
    from genomic_variant_classifier.agent_layer.orchestrator import Orchestrator
    from genomic_variant_classifier.agent_layer.shared_state import SharedState
    from genomic_variant_classifier.agent_layer._lazy_agent import _Lazy

    o = Orchestrator(SharedState(state_file=tmp_path / "state.json"), dry_run=True)
    assert o._agent_registry, "registry is empty"
    assert all(isinstance(v, _Lazy) for v in o._agent_registry.values()), (
        "every registry value must be a _Lazy descriptor"
    )
