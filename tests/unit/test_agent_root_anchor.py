"""Agent output roots must not depend on the current working directory.

AGENT-ROOT-ANCHOR-1
===================
Five agents write timestamped reports to `Path(self._root) / "reports" / ...`.
Four of them defaulted `root: str = "."`, so the destination depended on where
the process was launched.

That is not hypothetical. scripts/apply_data_readiness_root_fix.py diagnosed it
for DataReadinessAgent in exactly these terms:

    "it defaulted root='.' so it resolved registry.critical_assets()
     (repo-relative paths) against the CURRENT WORKING DIRECTORY. When the
     orchestrator is launched from src/.../agent_layer ... cwd was the
     agent_layer dir and every asset read as missing -> spurious NO_GO"

and three generated reports sit inside the source tree today as its residue:

    src/.../agent_layer/reports/agent_ops/OPS_2026-06-20.md
    src/.../agent_layer/reports/data_freshness/FRESHNESS_2026-06-20.md
    src/.../agent_layer/reports/data_readiness/READINESS_2026-06-20.md

WHY THIS TEST EXISTS AT ALL
Every one of the eight measured test call sites passes root=str(tmp_path):

    test_agent_ops_monitor_agent.py:18       test_agent_ops_wiring.py:38
    test_data_readiness_agent.py:19          test_data_readiness_wiring.py:43
    test_database_freshness_monitor_agent.py:16
    test_finops_wiring.py:38
    test_model_insights_agent.py:37          test_model_insights_wiring.py:46

The docstrings call root "injectable for hermetic tests", and the tests do
exactly that -- so the DEFAULT is never exercised, and changing it would have
passed the entire suite whether the change was right or wrong. This test drives
the default itself.

Author: Monzia Moodie
"""
from __future__ import annotations

import subprocess
import sys

import pytest

#: The SIX agents that resolve an output location from self._root.
#:
#: MEASURED 2026-08-14 by constructing each with a stub shared_state:
#:     AgentOpsMonitorAgent            _root = '.'
#:     DatabaseFreshnessMonitorAgent   _root = '.'
#:     FinOpsAdvisorAgent              _root = '.'
#:     ModelInsightsAgent              _root = '.'
#:     ProvisioningAgent               _root = '.'
#:     DataReadinessAgent              _root = PROJECT_ROOT   (already repaired)
#:
#: An earlier version of this list held five. A census across all of src and
#: scripts found provisioning_agent.py:45 carrying the same default -- it
#: writes via PD.write_provisioning_doc(self._root, event). Dropping an agent
#: from this list makes its defect invisible, which sabotage T5 confirmed.
AGENTS = (
    ("agent_ops_monitor_agent", "AgentOpsMonitorAgent"),
    ("database_freshness_monitor_agent", "DatabaseFreshnessMonitorAgent"),
    ("finops_advisor_agent", "FinOpsAdvisorAgent"),
    ("model_insights_agent", "ModelInsightsAgent"),
    ("provisioning_agent", "ProvisioningAgent"),
    ("data_readiness_agent", "DataReadinessAgent"),
)


@pytest.mark.parametrize("module,cls", AGENTS, ids=[c for _, c in AGENTS])
def test_the_default_root_is_NOT_the_working_directory(module, cls):
    """Constructed with no root, the agent must not resolve to ".".

    The instance is built with a stub shared_state so nothing runs; only the
    constructed default is inspected.
    """
    import importlib

    mod = importlib.import_module(
        "genomic_variant_classifier.agent_layer.agents." + module)
    agent_cls = getattr(mod, cls)

    class _StubState:
        def __getattr__(self, name):
            return lambda *a, **k: None

    agent = agent_cls(_StubState())
    assert agent._root not in (".", "", None), (
        "{} defaults its root to {!r} -- the destination depends on where the "
        "process was launched".format(cls, agent._root))


@pytest.mark.parametrize("module,cls", AGENTS, ids=[c for _, c in AGENTS])
def test_the_default_root_follows_PROJECT_ROOT(module, cls):
    """One anchor, not five conventions.

    PROJECT_ROOT is the anchor InterpretabilityAgent already uses via
    CHECKPOINT_DIR = PROJECT_ROOT / "models", and the one the earlier repair
    chose. A third convention would be the parallel-vocabulary failure this
    repository keeps eliminating.
    """
    import importlib

    from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT
    mod = importlib.import_module(
        "genomic_variant_classifier.agent_layer.agents." + module)

    class _StubState:
        def __getattr__(self, name):
            return lambda *a, **k: None

    agent = getattr(mod, cls)(_StubState())
    assert str(agent._root) == str(PROJECT_ROOT), (
        "{} anchors to {!r}, not PROJECT_ROOT {!r}".format(
            cls, agent._root, str(PROJECT_ROOT)))


@pytest.mark.parametrize("module,cls", AGENTS, ids=[c for _, c in AGENTS])
def test_root_remains_injectable(module, cls):
    """The hermetic-test seam the docstrings promise must survive the anchor."""
    import importlib

    mod = importlib.import_module(
        "genomic_variant_classifier.agent_layer.agents." + module)

    class _StubState:
        def __getattr__(self, name):
            return lambda *a, **k: None

    agent = getattr(mod, cls)(_StubState(), root="/injected/path")
    assert str(agent._root) == "/injected/path"


def test_the_default_TRACKS_the_environment_not_the_cwd():
    """Measured 2026-08-14: config.py reads GVC_PROJECT_ROOT at import time, so
    a fresh interpreter with the variable set gets a different anchor -- while
    changing the working directory does NOT move it.

    That distinction is the whole point: an ambient cwd is accidental, an
    explicit environment variable is a decision.
    """
    code = (
        "import os\n"
        "from genomic_variant_classifier.agent_layer.agents."
        "agent_ops_monitor_agent import AgentOpsMonitorAgent\n"
        "class S:\n"
        "    def __getattr__(self, n): return lambda *a, **k: None\n"
        "print(AgentOpsMonitorAgent(S())._root)\n"
    )
    import os
    env = {**os.environ, "GVC_PROJECT_ROOT": os.path.join(os.sep, "probe_anchor")}
    out = subprocess.run([sys.executable, "-B", "-c", code],
                         capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert "probe_anchor" in out.stdout, out.stdout.strip()
