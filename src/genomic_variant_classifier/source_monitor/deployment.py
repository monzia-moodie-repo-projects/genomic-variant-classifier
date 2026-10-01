"""The source monitor's DEPLOYMENT identities (owner ruling 2026-10-01) -- one small, strictly validated configuration.

Production and the isolated qualification repository run the SAME reviewed implementation; only these identities differ.
The configuration is selected by the TRUSTED checkout (configs/source_monitor_deployment.json at the checked-out commit):
no report, receipt or manual input can choose the repository, destination or trusted author. It is passed EXPLICITLY to
the checker and the publisher -- never installed as module globals, never monkeypatched.

Refusals: unknown or missing keys; wrong types (booleans are never numbers); a repository name not "owner/name"; ANY
unresolved (null) identity -- the provisioning state before identities are discovered from GitHub, which must refuse
execution; a source workflow path other than the one the receipt protocol pins. publication_enabled=false makes the
publisher preview only (the commissioning state "publication disabled").

Standard library only (plus the dependency-light contract's strict JSON).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from genomic_variant_classifier.source_monitor import interpretation_contract as ic

CONFIG_PATH = "configs/source_monitor_deployment.json"
SCHEMA, SCHEMA_VERSION = "gvc.source-monitor-deployment", 1
#: The receipt protocol pins this path (c2_protocol.validate_payload); the qualification repository preserves it.
SOURCE_WORKFLOW_PATH = ".github/workflows/source_monitor.yml"
_REPO = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
_MAX_INT = 2 ** 53 - 1


class DeploymentError(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code if not detail else "{}: {}".format(code, detail))
        self.code = code


@dataclass(frozen=True)
class Deployment:
    repository: str
    repository_id: int
    source_workflow_path: str
    source_workflow_id: int
    branch: str
    events: tuple
    issue_number: int
    issue_id: int
    label: str
    trusted_author_id: int
    publication_enabled: bool
    sha256: str                                      # of the configuration file's exact bytes

    def admission(self) -> dict:
        """The deployment part of the effective verification policy."""
        return {"repository": self.repository, "repository_id": self.repository_id,
                "workflow_path": self.source_workflow_path, "workflow_id": self.source_workflow_id,
                "events": sorted(self.events), "branch": self.branch}


def _keys(value, keys, where):
    if type(value) is not dict or set(value) != set(keys):
        got = sorted(value) if type(value) is dict else type(value).__name__
        raise DeploymentError("deployment.shape", "{} must have exactly {}, got {}".format(where, sorted(keys), got))
    return value


def _positive(value, where, unresolved):
    if value is None:
        unresolved.append(where)
        return None
    if type(value) is not int or not 0 < value <= _MAX_INT:
        raise DeploymentError("deployment.type", "{} must be a positive integer or null, got {!r}".format(where, value))
    return value


def _text(value, where):
    if type(value) is not str or not value.strip() or value != value.strip():
        raise DeploymentError("deployment.type", "{} must be a nonempty string without surrounding space, got {!r}".format(where, value))
    return value


def parse(raw: bytes) -> Deployment:
    try:
        doc = ic.strict_json(raw)
    except ic.ContractError as exc:
        raise DeploymentError("deployment.json", str(exc)) from exc
    _keys(doc, {"schema", "schema_version", "repository", "repository_id", "source_workflow", "destination",
                "trusted_author_id", "publication_enabled"}, "the deployment")
    if doc["schema"] != SCHEMA or type(doc["schema_version"]) is not int or doc["schema_version"] != SCHEMA_VERSION:
        raise DeploymentError("deployment.schema", "{!r} version {!r}".format(doc["schema"], doc["schema_version"]))
    repository = _text(doc["repository"], "repository")
    if _REPO.fullmatch(repository) is None:
        raise DeploymentError("deployment.repository", "not owner/name: {!r}".format(repository))
    wf = _keys(doc["source_workflow"], {"path", "id", "branch", "events"}, "source_workflow")
    if wf["path"] != SOURCE_WORKFLOW_PATH:
        raise DeploymentError("deployment.workflow_path", "{!r} (the receipt protocol pins {!r})".format(wf["path"], SOURCE_WORKFLOW_PATH))
    events = wf["events"]
    if type(events) is not list or not events or any(type(e) is not str or not e for e in events) or len(set(events)) != len(events):
        raise DeploymentError("deployment.type", "source_workflow.events must be a nonempty list of distinct strings")
    dest = _keys(doc["destination"], {"issue_number", "issue_id", "label"}, "destination")
    if type(doc["publication_enabled"]) is not bool:
        raise DeploymentError("deployment.type", "publication_enabled must be a boolean")
    unresolved = []
    values = dict(repository_id=_positive(doc["repository_id"], "repository_id", unresolved),
                  source_workflow_id=_positive(wf["id"], "source_workflow.id", unresolved),
                  issue_number=_positive(dest["issue_number"], "destination.issue_number", unresolved),
                  issue_id=_positive(dest["issue_id"], "destination.issue_id", unresolved),
                  trusted_author_id=_positive(doc["trusted_author_id"], "trusted_author_id", unresolved))
    branch, label = _text(wf["branch"], "source_workflow.branch"), _text(dest["label"], "destination.label")
    if unresolved:
        raise DeploymentError("deployment.unresolved", "identities not yet discovered from GitHub: {}".format(", ".join(unresolved)))
    return Deployment(repository=repository, source_workflow_path=wf["path"], branch=branch, events=tuple(events), label=label,
                      publication_enabled=doc["publication_enabled"], sha256=hashlib.sha256(raw).hexdigest(), **values)


def load(root) -> Deployment:
    """The deployment of the TRUSTED checkout at `root`. Any failure refuses (raises DeploymentError)."""
    path = Path(root) / CONFIG_PATH
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DeploymentError("deployment.missing", "{}: {}".format(CONFIG_PATH, exc)) from exc
    return parse(raw)
