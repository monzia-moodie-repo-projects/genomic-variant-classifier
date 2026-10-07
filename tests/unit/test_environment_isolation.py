"""The network claim a replay may make (owner ruling 2026-10-07, section 6): probes are diagnostic; offline needs enforced isolation.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

from genomic_variant_classifier.environment_qualification.isolation import (
    LOCAL_ARTIFACT_REPLAY, OFFLINE_REPLAY, IsolationEvidence, classify_replay_claim)
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError

FAILED = {h: "failed: OSError: refused" for h in ("cloud.r-project.org", "bioconductor.org", "packagemanager.posit.co", "github.com")}
DOWN = (("Wi-Fi", "Disabled"), ("Ethernet", "Disconnected"))


def test_four_failed_probes_alone_are_not_an_offline_claim():
    """The owner's counterexample: four refused probes previously produced "offline replay"."""
    result = classify_replay_claim(None, FAILED)
    assert result["claim"] == LOCAL_ARTIFACT_REPLAY and result["reachability"] == "unknown_without_enforced_isolation"


def test_reachable_probes_without_isolation_are_local_artifact():
    result = classify_replay_claim(None, dict(FAILED, **{"github.com": "connected"}))
    assert result["claim"] == LOCAL_ARTIFACT_REPLAY and result["reachable_hosts"] == ["github.com"]


def test_enforced_isolation_with_failed_probes_is_offline():
    result = classify_replay_claim(IsolationEvidence("all network adapters disabled", DOWN, DOWN), FAILED)
    assert result["claim"] == OFFLINE_REPLAY and result["isolation"]["adapters"] == ["Ethernet", "Wi-Fi"]


@pytest.mark.parametrize("evidence, probes, reason", [
    (IsolationEvidence("adapters disabled", (("Wi-Fi", "Up"), ("Ethernet", "Disconnected")), DOWN), FAILED, "isolation.adapter_up:Wi-Fi"),
    (IsolationEvidence("adapters disabled", DOWN, (("Wi-Fi", "Up"), ("Ethernet", "Disconnected"))), FAILED, "isolation.adapter_up:Wi-Fi"),
    (IsolationEvidence("adapters disabled", DOWN, DOWN), dict(FAILED, **{"bioconductor.org": "connected"}), "isolation.contradicted_by_probe:bioconductor.org"),
    (IsolationEvidence(" ", DOWN, DOWN), FAILED, "isolation.method"),
    (IsolationEvidence("adapters disabled", (), DOWN), FAILED, "isolation.before.empty"),
    (IsolationEvidence("adapters disabled", DOWN, (("Wi-Fi", "Disabled"),)), FAILED, "isolation.adapter_set_changed"),
    (IsolationEvidence("adapters disabled", (("Wi-Fi", "Disabled"), ("Wi-Fi", "Disabled")), DOWN), FAILED, "isolation.before.duplicate_adapter"),
    (IsolationEvidence("adapters disabled", DOWN, DOWN), {}, "isolation.probes_shape"),
])
def test_isolation_refusals(evidence, probes, reason):
    with pytest.raises(AdmissionError) as error:
        classify_replay_claim(evidence, probes)
    assert str(error.value) == reason
