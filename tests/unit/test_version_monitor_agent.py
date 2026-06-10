"""Smoke test: VersionMonitorAgent wraps the module-level watch targets and
surfaces a summary into its SharedState section (watch targets monkeypatched -- no network)."""
from __future__ import annotations
from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents import version_monitor_agent as vm


def test_version_monitor_surfaces_alerts(monkeypatch):
    monkeypatch.setattr(vm, "_run_watch_targets", lambda *, dry_run=False: {
        "literature_scout.last_run": "t",
        "literature_scout.alerts": ["[KAN] pykan 1.1 available"],
        "literature_scout.pykan_installed": "1.0",
        "literature_scout.pykan_latest": "1.1",
        "literature_scout.pykan_alert": True,
    })
    agent = vm.VersionMonitorAgent(SharedState())
    r = agent.run(dry_run=True)
    assert r["status"] == "ok"
    assert r["n_alerts"] == 1 and r["pykan_alert"] is True
    assert agent._get_section("version_monitor")["pykan_latest"] == "1.1"
