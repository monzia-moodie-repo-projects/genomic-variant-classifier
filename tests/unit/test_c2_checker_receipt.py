"""The CHECKER's C2 receipt (owner ruling 2026-09-29): written by scripts/verify_monitor_run.py BEFORE its exit code.

Every outcome is checked through the real script with GitHub served offline from the run-8 fixtures (the helpers of
test_report_verifier.py): completed -> a receipt that c2_protocol.open_receipt accepts against INDEPENDENT bindings;
evidence unobtainable after GitHub's attempt record was obtained -> a checker-issued "unavailable" receipt (nulls,
checker.unavailable -- never six invented flags); no attempt record, or no effective policy -> NO receipt, stated.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.source_monitor import c2_protocol as c2
from genomic_variant_classifier.source_monitor import deployment as dep
from genomic_variant_classifier.source_monitor import report_verifier as rv
from tests.unit.test_report_verifier import CURRENT_V1, FIX, NOW, RUN, RUN_ID, _cli, _serve, read_blob

ROOT = Path(__file__).resolve().parents[2]
DEP = dep.load(ROOT)
EVALUATION = ["--checker-commit", "a" * 40, "--evaluation-run-id", "999", "--evaluation-attempt", "1"]


def _run(tmp_path, fetch, capsys, current=CURRENT_V1):
    receipt = tmp_path / "receipt.json"
    code = _cli().main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(tmp_path / "verdict.json"),
                        "--receipt", str(receipt)] + EVALUATION, fetch=fetch, read_blob=read_blob, now=NOW, current=current)
    return code, receipt, capsys.readouterr()


def _open(receipt, issuer="checker"):
    """Binds against an INDEPENDENTLY built identity (owner ruling 2026-10-01). The first version bound the receipt against
    its OWN checker block -- internal consistency only -- which let the unavailable-policy defect through."""
    assert receipt.is_file(), "the checker wrote NO receipt"         # absence is a stated failure, never a crash
    raw = receipt.read_bytes()
    subject = _cli().receipt_subject(RUN)                              # from GitHub's attempt record, never the report
    expected = rv.build_checker_identity(root=ROOT, commit="a" * 40, current=CURRENT_V1, required_targets=_targets(),
                                         deployment=DEP)
    try:
        return c2.open_receipt(raw, c2.Bindings(subject, expected, 999, 1, issuer), NOW)
    except c2.Refusal as exc:
        pytest.fail("the checker's receipt does not bind to the independently built identity: {}".format(exc.code))


def _targets():
    from genomic_variant_classifier.source_monitor.run_monitor import REQUIRED_TARGETS
    return REQUIRED_TARGETS


def test_a_completed_verification_writes_a_receipt_the_protocol_opens(tmp_path, capsys):
    code, receipt, out = _run(tmp_path, _serve(), capsys)
    p = _open(receipt)
    assert code == 0 and "RECEIPT written" in out.out
    assert p["subject"] == {"repository": DEP.repository, "repository_id": 1151261021, "workflow_id": 359207377,
                            "workflow_path": DEP.source_workflow_path, "run_id": RUN_ID, "run_number": 8, "attempt": 1,
                            "commit": RUN["head_sha"]}
    assert p["evidence"] == {"state": "complete", "artifact_id": 10924439843,
                             "archive_sha256": "838a35e1c9a2c4e9b389508a5891cfc170648fe770304e5440a77f8e86f0729e",
                             "report_sha256": "61656e1d3f5840e16b78b79da5d24f4debd309f6b063d5e84993a76dff288d21"}
    assert p["decision"]["reviews"] == [{"target": "gnomad-public-releases", "kind": "newer", "raw_prefix": "release/4.1.2/"}]
    assert (p["decision"]["status"], p["decision"]["verified"], p["decision"]["reasons"]) == ("completed", True, [])
    assert p["checker"]["policy_sha256"] == c2.digest("gvc.verification-policy/v1", rv.effective_policy(
        CURRENT_V1, __import__("genomic_variant_classifier.source_monitor.run_monitor", fromlist=["x"]).REQUIRED_TARGETS, DEP))
    assert c2.event_kind(p) == "review_required"


def test_evidence_lost_after_the_attempt_record_gives_a_checker_issued_unavailable_receipt(tmp_path, capsys):
    served = _serve()
    def fetch(url, limit):
        if "/artifacts" in url:
            raise OSError("simulated: the artifact listing could not be fetched")
        return served(url, limit)
    code, receipt, _ = _run(tmp_path, fetch, capsys)
    p = _open(receipt)
    assert code == 2
    assert p["decision"] == {"status": "unavailable", "verified": None, "flags": None, "reviews": [],
                             "reasons": [{"code": "checker.unavailable", "target": ""}]}
    assert p["evidence"] == {"state": "unavailable", "artifact_id": None, "archive_sha256": None, "report_sha256": None}
    assert c2.event_kind(p) == "verification_unavailable"
    assert any("simulated: the artifact listing could not be fetched" in d for d in p["diagnostics"])


def test_without_githubs_attempt_record_no_run_is_guessed(tmp_path, capsys):
    served = _serve()
    def fetch(url, limit):
        if "/attempts/" in url:
            raise OSError("simulated: the attempt record could not be fetched")
        return served(url, limit)
    code, receipt, out = _run(tmp_path, fetch, capsys)
    assert code == 2 and not receipt.exists()
    assert "NO RECEIPT: ValueError: GitHub's attempt record was not obtained; the subject is unknown, so no run is guessed" in out.err


def test_without_an_effective_policy_no_receipt_is_issued(tmp_path, capsys, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli.rv, "current_reconstruction", lambda root: (_ for _ in ()).throw(OSError("simulated")))
    receipt = tmp_path / "receipt.json"
    cli.main(["--run-id", str(RUN_ID), "--run-attempt", "1", "--verdict", str(tmp_path / "v.json"), "--receipt", str(receipt)]
             + EVALUATION, fetch=_serve(), read_blob=read_blob, now=NOW, current=None)
    err = capsys.readouterr().err
    assert not receipt.exists()
    assert ("NO RECEIPT: ValueError: the checker identity could not be reconstructed: ValueError: today's interpretation was not "
            "reconstructed") in err                              # detected BEFORE collection since the 2026-10-01 identity repair


def test_a_receipt_requires_all_three_identity_arguments(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _cli().main(["--run-id", "1", "--run-attempt", "1", "--verdict", str(tmp_path / "v.json"),
                     "--receipt", str(tmp_path / "r.json"), "--checker-commit", "a" * 40])
    assert exc.value.code == 2


def test_the_code_manifest_is_exactly_the_code_a_real_verification_loads(tmp_path):
    """MEASURED 2026-09-30: import alone loads 7 project modules; a REAL verification loads 15. A FRESH interpreter runs
    one offline verification that writes a receipt, then reports the project files it loaded; they must EQUAL the
    manifest's Python files -- no module missing, none listed that the checker never loads."""
    probe = r'''
import importlib.util, json, sys
from pathlib import Path
root, fix, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(root / "src"))
spec = importlib.util.spec_from_file_location("vmr", root / "scripts" / "verify_monitor_run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
api = "https://api.github.com/repos/" + __import__("genomic_variant_classifier.source_monitor.deployment", fromlist=["x"]).load(root).repository
art = json.loads((fix / "run8_artifacts.json").read_text(encoding="utf-8"))["artifacts"][0]["id"]
pages = {api + "/actions/runs/36300779115": (fix / "run8.json").read_bytes(),
         api + "/actions/runs/36300779115/attempts/1": (fix / "run8_attempt1.json").read_bytes(),
         api + "/actions/runs/36300779115/artifacts?per_page=100": (fix / "run8_artifacts.json").read_bytes(),
         api + "/actions/artifacts/{}/zip".format(art): (fix / "run8_source-monitor-report.zip").read_bytes()}
m.main(["--run-id", "36300779115", "--run-attempt", "1", "--verdict", str(out / "v.json"), "--receipt", str(out / "r.json"),
        "--checker-commit", "a" * 40, "--evaluation-run-id", "1", "--evaluation-attempt", "1"], fetch=lambda u, n: pages[u])
files = sorted(str(Path(x.__file__).resolve().relative_to(root)).replace("\\", "/") for n, x in list(sys.modules.items())
               if n.startswith("genomic_variant_classifier") and getattr(x, "__file__", None))
print(json.dumps({"files": files, "receipt": (out / "r.json").is_file()}))
'''
    r = subprocess.run([sys.executable, "-c", probe, str(ROOT), str(FIX), str(tmp_path)], capture_output=True, text=True,
                       timeout=300, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    measured = json.loads(r.stdout.strip().splitlines()[-1])
    assert measured["receipt"] is True
    listed = sorted(p for p in rv.CHECKER_SOURCES if p.startswith("src/"))
    assert measured["files"] == listed
    assert {"scripts/verify_monitor_run.py", "requirements-source-monitor.txt"} <= set(rv.CHECKER_SOURCES)
