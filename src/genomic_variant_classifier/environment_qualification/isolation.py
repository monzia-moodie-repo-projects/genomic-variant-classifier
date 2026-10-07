"""The network claim a replay may make (owner ruling 2026-10-07, section 6).

Reachability probes are DIAGNOSTIC: several unreachable hosts do not prove that a process and its children lacked network access (the
owner executed the earlier classifier with four refused probes and it answered "offline replay" -- an invalid inference, reproduced
2026-10-07). An OFFLINE claim therefore requires ENFORCED, DOCUMENTED isolation: a named method and every network adapter measured NOT "Up"
both before and after the replay (adapter-level isolation covers every process on the machine during the window, R's children included).
A probe that connects while isolation is claimed CONTRADICTS the evidence and is refused, never overridden. Without isolation evidence the
claim is a local-artifact replay; the probes then describe reachability as observed-reachable or unknown, never as offline.

Windows adapter states measured with Get-NetAdapter: "Up", "Disconnected", "Disabled", "Not Present".

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, require

logger = logging.getLogger(__name__)

__all__ = ["OFFLINE_REPLAY", "LOCAL_ARTIFACT_REPLAY", "IsolationEvidence", "classify_replay_claim"]

OFFLINE_REPLAY = "offline replay"
LOCAL_ARTIFACT_REPLAY = "local-artifact replay"
_CONNECTED = "connected"


@dataclass(frozen=True)
class IsolationEvidence:
    method: str                                   # how isolation was enforced, e.g. "all network adapters disabled"
    adapters_before: tuple[tuple[str, str], ...]  # (adapter name, status) measured immediately before the replay
    adapters_after: tuple[tuple[str, str], ...]   # (adapter name, status) measured immediately after the replay


def _adapters(rows, reason: str) -> dict:
    require(type(rows) is tuple and len(rows) > 0, reason + ".empty")
    seen = {}
    for row in rows:
        require(type(row) is tuple and len(row) == 2 and all(type(x) is str and x for x in row), reason + ".shape")
        require(row[0] not in seen, reason + ".duplicate_adapter")
        seen[row[0]] = row[1]
    return seen


def classify_replay_claim(isolation: IsolationEvidence | None, probes: Mapping[str, str]) -> dict:
    """probes: host -> "connected" or a failure description (diagnostic only)."""
    require(isinstance(probes, Mapping) and len(probes) > 0 and all(type(k) is str and k and type(v) is str and v for k, v in probes.items()),
            "isolation.probes_shape")
    reachable = sorted(h for h, outcome in probes.items() if outcome == _CONNECTED)
    if isolation is None:
        return {"claim": LOCAL_ARTIFACT_REPLAY, "isolation": None, "reachable_hosts": reachable,
                "reachability": "observed_reachable" if reachable else "unknown_without_enforced_isolation"}
    require(type(isolation) is IsolationEvidence, "isolation.type")
    require(type(isolation.method) is str and isolation.method.strip() != "", "isolation.method")
    before = _adapters(isolation.adapters_before, "isolation.before")
    after = _adapters(isolation.adapters_after, "isolation.after")
    require(set(before) == set(after), "isolation.adapter_set_changed")
    up = sorted(name for name, status in list(before.items()) + list(after.items()) if status == "Up")
    if up:
        raise AdmissionError("isolation.adapter_up:" + ",".join(sorted(set(up))))
    if reachable:
        raise AdmissionError("isolation.contradicted_by_probe:" + ",".join(reachable))
    return {"claim": OFFLINE_REPLAY, "isolation": {"method": isolation.method, "adapters": sorted(before)}, "reachable_hosts": [],
            "reachability": "isolated_by_enforced_method"}
