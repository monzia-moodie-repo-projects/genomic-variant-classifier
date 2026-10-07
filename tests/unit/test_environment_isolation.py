"""The network claim a replay may make -- version 2 (owner rulings 2026-10-07 section 6, 2026-10-07b section 4): the claim describes an
ENFORCEMENT MECHANISM's evidence; adapter observations and probe failures are never enforcement.

Author: Monzia Moodie
"""
from __future__ import annotations

import json

import pytest

from genomic_variant_classifier.environment_qualification.isolation import (
    HOST_PROCEDURE_REPLAY, LOCAL_ARTIFACT_REPLAY, OFFLINE_REPLAY, IsolationEvidence, classify_replay_claim)
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError

FAILED = {h: "failed: OSError: refused" for h in ("cloud.r-project.org", "bioconductor.org", "packagemanager.posit.co", "github.com")}
DOWN = (("Wi-Fi", "Disabled"), ("Ethernet", "Disconnected"), ("Hidden Virtual", "Not Present"))
WSB_OFF = b"<Configuration><Networking>Disable</Networking><ClipboardRedirection>Disable</ClipboardRedirection></Configuration>"
VM_OFF = json.dumps([{"Name": "Network Adapter", "SwitchName": None, "Connected": False}]).encode()


def _refused(evidence, probes, reason):
    with pytest.raises(AdmissionError) as error:
        classify_replay_claim(evidence, probes)
    assert str(error.value) == reason


def test_failed_probes_without_evidence_are_never_offline():
    """The owner's first counterexample: four refused probes previously produced "offline replay"."""
    result = classify_replay_claim(None, FAILED)
    assert result["claim"] == LOCAL_ARTIFACT_REPLAY and result["reachability"] == "unknown_without_enforced_isolation"


def test_reachable_probes_without_evidence_are_local_artifact():
    result = classify_replay_claim(None, dict(FAILED, **{"github.com": "connected"}))
    assert result["claim"] == LOCAL_ARTIFACT_REPLAY and result["reachable_hosts"] == ["github.com"]


def test_a_host_disconnection_procedure_is_never_an_offline_claim():
    result = classify_replay_claim(IsolationEvidence("host_disconnection_procedure", None, DOWN, DOWN), FAILED)
    assert result["claim"] == HOST_PROCEDURE_REPLAY and result["claim"] != OFFLINE_REPLAY


@pytest.mark.parametrize("status", ["Unknown", "up", "Up", "disabled", "Degraded", ""])
def test_adapter_states_outside_the_exact_allowlist_are_refused(status):
    """The owner's executed counterexamples: "Unknown", "Disconnected" and lower-case "up" were accepted by version 1."""
    rows = (("Wi-Fi", status),) if status else (("Wi-Fi", "Disabled"), ("x", ""))
    reason = "isolation.adapter_not_inactive:Wi-Fi" if status else "isolation.before.shape"
    _refused(IsolationEvidence("host_disconnection_procedure", None, rows, rows), FAILED, reason)


def test_windows_sandbox_with_networking_disabled_is_offline():
    result = classify_replay_claim(IsolationEvidence("windows_sandbox_networking_disabled", WSB_OFF), FAILED)
    assert result["claim"] == OFFLINE_REPLAY and len(result["configuration_sha256"]) == 64


def test_a_detached_virtual_machine_is_offline():
    assert classify_replay_claim(IsolationEvidence("virtual_machine_network_detached", VM_OFF), FAILED)["claim"] == OFFLINE_REPLAY


@pytest.mark.parametrize("evidence, probes, reason", [
    (IsolationEvidence("windows_sandbox_networking_disabled", b"<Configuration><Networking>Default</Networking></Configuration>"), FAILED,
     "isolation.configuration_does_not_disable_networking"),
    (IsolationEvidence("windows_sandbox_networking_disabled", b"<Configuration></Configuration>"), FAILED, "isolation.configuration_does_not_disable_networking"),
    (IsolationEvidence("windows_sandbox_networking_disabled", b"<Configuration><Networking>Disable"), FAILED, "isolation.sandbox_configuration_unparseable"),
    (IsolationEvidence("windows_sandbox_networking_disabled", None), FAILED, "isolation.configuration_required"),
    (IsolationEvidence("virtual_machine_network_detached", json.dumps([{"SwitchName": "Default Switch", "Connected": True}]).encode()), FAILED,
     "isolation.configuration_does_not_disable_networking"),
    (IsolationEvidence("virtual_machine_network_detached", b"not json"), FAILED, "isolation.vm_configuration_unparseable"),
    (IsolationEvidence("windows_sandbox_networking_disabled", WSB_OFF), dict(FAILED, **{"bioconductor.org": "connected"}),
     "isolation.contradicted_by_probe:bioconductor.org"),
    (IsolationEvidence("all network adapters disabled", None, DOWN, DOWN), FAILED, "isolation.unknown_mechanism"),
    (IsolationEvidence("host_disconnection_procedure", None, DOWN, DOWN[:2]), FAILED, "isolation.adapter_set_changed"),
    (IsolationEvidence("host_disconnection_procedure", None, (), DOWN), FAILED, "isolation.before.empty"),
    (IsolationEvidence("host_disconnection_procedure", None, (("Wi-Fi", "Disabled"), ("Wi-Fi", "Disabled")), DOWN), FAILED, "isolation.before.duplicate_adapter"),
    (IsolationEvidence("windows_sandbox_networking_disabled", WSB_OFF), {}, "isolation.probes_shape"),
])
def test_isolation_refusals(evidence, probes, reason):
    _refused(evidence, probes, reason)
