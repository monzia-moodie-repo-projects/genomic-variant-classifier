"""The network claim a replay may make -- version 2 (owner rulings 2026-10-07 section 6, 2026-10-07b section 4).

The claim DESCRIBES the evidence an ENFORCEMENT MECHANISM supplies; it never manufactures enforcement from a method name or from
endpoint observations. Version 1 accepted any adapter status except the exact string "Up" ("Unknown", "Disconnected", "up" all became
"offline replay" -- the owner's executed counterexamples) and treated before/after adapter snapshots as enforcement, which they are not
(identical snapshots do not show an adapter stayed disabled throughout; Get-NetAdapter shows only VISIBLE adapters unless -IncludeHidden).

  no evidence                           -> "local-artifact replay"; probes describe reachability only (observed / unknown).
  host_disconnection_procedure          -> "local-artifact replay under a host disconnection procedure" -- a controlled operating
                                           PROCEDURE with observations, NEVER "offline": every adapter (hidden ones included) in an explicit
                                           inactive allowlist before AND after, and no probe connecting.
  windows_sandbox_networking_disabled   -> "offline replay" only if the Windows Sandbox CONFIGURATION is supplied and its
                                           <Networking> element is exactly "Disable"; its digest is recorded; no probe may connect.
  virtual_machine_network_detached      -> "offline replay" only if the VM's own network-adapter record (JSON) is supplied and shows every
                                           adapter with no switch and not connected; its digest is recorded; no probe may connect.
Any other mechanism is refused. A probe that connects contradicts any isolation claim and is refused, never overridden.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Mapping

from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, require

logger = logging.getLogger(__name__)

__all__ = ["OFFLINE_REPLAY", "LOCAL_ARTIFACT_REPLAY", "HOST_PROCEDURE_REPLAY", "INACTIVE_ADAPTER_STATES", "IsolationEvidence",
           "classify_replay_claim"]

OFFLINE_REPLAY = "offline replay"
LOCAL_ARTIFACT_REPLAY = "local-artifact replay"
HOST_PROCEDURE_REPLAY = "local-artifact replay under a host disconnection procedure"
INACTIVE_ADAPTER_STATES = frozenset({"Disabled", "Disconnected", "Not Present"})     # exact, case-sensitive; anything else refuses
_SANDBOX = "windows_sandbox_networking_disabled"
_VM = "virtual_machine_network_detached"
_HOST = "host_disconnection_procedure"
_CONNECTED = "connected"


@dataclass(frozen=True)
class IsolationEvidence:
    mechanism: str
    configuration: bytes | None = None                   # the .wsb file (sandbox) or the VM network-adapter record (JSON)
    adapters_before: tuple[tuple[str, str], ...] = ()    # (name, status) INCLUDING hidden adapters, measured before the replay
    adapters_after: tuple[tuple[str, str], ...] = ()     # the same, measured after


def _adapters(rows, reason: str) -> dict:
    require(type(rows) is tuple and len(rows) > 0, reason + ".empty")
    seen = {}
    for row in rows:
        require(type(row) is tuple and len(row) == 2 and all(type(x) is str and x for x in row), reason + ".shape")
        require(row[0] not in seen, reason + ".duplicate_adapter")
        seen[row[0]] = row[1]
    return seen


def _sandbox_networking_disabled(config: bytes) -> bool:
    try:
        root = ET.fromstring(config)
    except ET.ParseError:
        raise AdmissionError("isolation.sandbox_configuration_unparseable")
    require(root.tag == "Configuration", "isolation.sandbox_configuration_root")
    values = [(e.text or "").strip() for e in root.findall("Networking")]
    return values == ["Disable"]


def _vm_network_detached(config: bytes) -> bool:
    try:
        rows = json.loads(config)
    except ValueError:
        raise AdmissionError("isolation.vm_configuration_unparseable")
    rows = rows if isinstance(rows, list) else [rows]
    require(len(rows) > 0 and all(isinstance(r, dict) for r in rows), "isolation.vm_configuration_shape")
    return all(not r.get("SwitchName") and r.get("Connected") is False for r in rows)


def classify_replay_claim(isolation: IsolationEvidence | None, probes: Mapping[str, str]) -> dict:
    """probes: host -> "connected" or a failure description (diagnostic only)."""
    require(isinstance(probes, Mapping) and len(probes) > 0 and all(type(k) is str and k and type(v) is str and v for k, v in probes.items()),
            "isolation.probes_shape")
    reachable = sorted(h for h, outcome in probes.items() if outcome == _CONNECTED)
    if isolation is None:
        return {"claim": LOCAL_ARTIFACT_REPLAY, "mechanism": None, "reachable_hosts": reachable,
                "reachability": "observed_reachable" if reachable else "unknown_without_enforced_isolation"}
    require(type(isolation) is IsolationEvidence, "isolation.type")
    if reachable:
        raise AdmissionError("isolation.contradicted_by_probe:" + ",".join(reachable))
    if isolation.mechanism == _HOST:
        before = _adapters(isolation.adapters_before, "isolation.before")
        after = _adapters(isolation.adapters_after, "isolation.after")
        require(set(before) == set(after), "isolation.adapter_set_changed")
        active = sorted({name for name, status in list(before.items()) + list(after.items()) if status not in INACTIVE_ADAPTER_STATES})
        if active:
            raise AdmissionError("isolation.adapter_not_inactive:" + ",".join(active))
        return {"claim": HOST_PROCEDURE_REPLAY, "mechanism": _HOST, "reachable_hosts": [], "reachability": "procedure_observations_only",
                "adapters": sorted(before)}
    if isolation.mechanism in (_SANDBOX, _VM):
        require(type(isolation.configuration) is bytes and len(isolation.configuration) > 0, "isolation.configuration_required")
        enforced = _sandbox_networking_disabled(isolation.configuration) if isolation.mechanism == _SANDBOX else _vm_network_detached(isolation.configuration)
        if not enforced:
            raise AdmissionError("isolation.configuration_does_not_disable_networking")
        return {"claim": OFFLINE_REPLAY, "mechanism": isolation.mechanism, "configuration_sha256": hashlib.sha256(isolation.configuration).hexdigest(),
                "reachable_hosts": [], "reachability": "enforced_by_mechanism"}
    raise AdmissionError("isolation.unknown_mechanism")
