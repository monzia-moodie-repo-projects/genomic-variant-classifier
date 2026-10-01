"""The deployment configuration (owner ruling 2026-10-01): strict, explicit, selected by the trusted checkout.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor import deployment as dep

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = json.loads((ROOT / dep.CONFIG_PATH).read_text(encoding="utf-8"))


def _raw(doc):
    return json.dumps(doc).encode("ascii")


def test_the_production_configuration_carries_the_measured_identities():
    d = dep.load(ROOT)
    assert (d.repository, d.repository_id, d.source_workflow_id, d.issue_number, d.issue_id, d.label, d.trusted_author_id,
            d.publication_enabled) == ("monzia-moodie-repo-projects/genomic-variant-classifier", 1151261021, 359207377, 27,
                                       5600463137, "source-monitor-alert", 41898282, True)
    assert d.sha256 == hashlib.sha256((ROOT / dep.CONFIG_PATH).read_bytes()).hexdigest()


@pytest.mark.parametrize("mutate, code", [
    (lambda d: d.update(extra=1), "deployment.shape"),
    (lambda d: d.pop("trusted_author_id"), "deployment.shape"),
    (lambda d: d.update(schema_version=True), "deployment.schema"),
    (lambda d: d.update(repository="not a repository"), "deployment.repository"),
    (lambda d: d.update(repository_id=True), "deployment.type"),
    (lambda d: d.update(repository_id=0), "deployment.type"),
    (lambda d: d["source_workflow"].update(path=".github/workflows/other.yml"), "deployment.workflow_path"),
    (lambda d: d["source_workflow"].update(events=["schedule", "schedule"]), "deployment.type"),
    (lambda d: d["destination"].update(label=" padded "), "deployment.type"),
    (lambda d: d.update(publication_enabled=1), "deployment.type"),
    (lambda d: d["destination"].update(issue_id=None), "deployment.unresolved"),
])
def test_a_malformed_or_unresolved_configuration_refuses_with_its_reason(mutate, code):
    doc = json.loads(json.dumps(PRODUCTION))
    mutate(doc)
    with pytest.raises(dep.DeploymentError) as exc:
        dep.parse(_raw(doc))
    assert exc.value.code == code


def test_every_unresolved_identity_is_named():
    """The commissioning state before identities are discovered from GitHub: execution must refuse, naming each gap."""
    doc = json.loads(json.dumps(PRODUCTION))
    doc["repository_id"] = doc["source_workflow"]["id"] = doc["destination"]["issue_id"] = doc["trusted_author_id"] = None
    with pytest.raises(dep.DeploymentError) as exc:
        dep.parse(_raw(doc))
    assert str(exc.value) == ("deployment.unresolved: identities not yet discovered from GitHub: repository_id, source_workflow.id, "
                              "destination.issue_id, trusted_author_id")


def test_a_missing_configuration_refuses(tmp_path):
    with pytest.raises(dep.DeploymentError) as exc:
        dep.load(tmp_path)
    assert exc.value.code == "deployment.missing"


def test_the_digest_binds_the_exact_bytes():
    """Reformatting changes the digest -- the configuration is bound byte-for-byte, like the interpretation policy."""
    raw = (ROOT / dep.CONFIG_PATH).read_bytes()
    compact = json.dumps(json.loads(raw), separators=(",", ":")).encode("ascii")
    assert dep.parse(raw).repository == dep.parse(compact).repository and dep.parse(raw).sha256 != dep.parse(compact).sha256


def test_the_policy_binds_the_deployment_while_the_code_manifest_does_not():
    """The fidelity rule: implementation files must match across repositories; the deployment may differ, recorded."""
    from genomic_variant_classifier.source_monitor import report_verifier as rv
    from genomic_variant_classifier.source_monitor.run_monitor import REQUIRED_TARGETS
    current = rv.current_reconstruction(ROOT)
    doc = json.loads(json.dumps(PRODUCTION))
    doc["destination"]["label"] = "c2-qualification-destination"
    a = rv.build_checker_identity(root=ROOT, commit="a" * 40, current=current, required_targets=REQUIRED_TARGETS, deployment=dep.load(ROOT))
    b = rv.build_checker_identity(root=ROOT, commit="a" * 40, current=current, required_targets=REQUIRED_TARGETS,
                                  deployment=dep.parse(_raw(doc)))
    assert a["policy_sha256"] != b["policy_sha256"] and a["code_manifest_sha256"] == b["code_manifest_sha256"]


def test_an_unresolved_configuration_makes_the_checker_refuse_with_no_receipt(tmp_path, capsys):
    from tests.unit.test_report_verifier import RUN_ID, _cli
    doc = json.loads(json.dumps(PRODUCTION))
    doc["source_workflow"]["id"] = None
    (tmp_path / "configs").mkdir()
    (tmp_path / dep.CONFIG_PATH).write_bytes(_raw(doc))
    code = _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(tmp_path / "v.json"),
                        "--receipt", str(tmp_path / "r.json"), "--checker-commit", "a" * 40, "--evaluation-run-id", "1",
                        "--evaluation-attempt", "1"], fetch=lambda u, n: pytest.fail("no request may be made"),
                       deployment_root=tmp_path)
    err = capsys.readouterr().err
    assert code == 2 and not (tmp_path / "r.json").exists()
    assert "NO RECEIPT: DeploymentError: deployment.unresolved: identities not yet discovered from GitHub: source_workflow.id" in err
