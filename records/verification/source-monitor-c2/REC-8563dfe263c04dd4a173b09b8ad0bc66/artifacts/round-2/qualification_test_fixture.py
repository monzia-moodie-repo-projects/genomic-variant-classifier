"""The qualification fixture builder against the REAL run-8 report and the REAL committed checker: each scenario must hit
its INTENDED boundary (owner ruling 2026-10-01b), not merely run.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qualification"))
import fixture as fx  # noqa: E402

from genomic_variant_classifier.source_monitor import report_verifier as rv  # noqa: E402
from tests.unit import test_report_verifier as T  # noqa: E402

QID = "Q-0123456789abcdef"

with zipfile.ZipFile(io.BytesIO(T.ARCHIVE)) as _zf:
    RAW = _zf.read("report.json")


def _zip(raw, method):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=method) as zf:
        zf.writestr("report.json", raw)
    return buf.getvalue()


def test_normal_is_byte_identical_and_records_both_exit_values():
    out, rec = fx.build_fixture(RAW, scenario="normal", monitor_exit_status=1, qualification_id=QID)
    assert (rec["schema_version"], rec["qualification_id"]) == (2, QID)
    assert out == RAW and rec["transformation"] == "identity" and rec["compression_level"] == 6
    assert (rec["original_exit_code"], rec["fixture_exit_code"], rec["monitor_exit_status"]) == (1, 1, 1)


def test_claims_disagree_changes_only_the_exit_code_and_the_real_checker_says_claims_disagree():
    out, rec = fx.build_fixture(RAW, scenario="claims-disagree", monitor_exit_status=1, qualification_id=QID)
    original, mutated = json.loads(RAW), json.loads(out)
    assert {k for k in original.keys() | mutated.keys() if original.get(k) != mutated.get(k)} == {"exit_code"}
    assert (rec["original_exit_code"], rec["fixture_exit_code"], rec["monitor_exit_status"]) == (1, 0, 1)
    archive, arts = T._repack(mutated)
    v = T._verify(archive=archive, artifacts=arts)
    assert (v.verified, v.reasons) == (False, [{"code": "claims.disagree", "target": ""}])
    assert v.reviews == [{"target": "gnomad-public-releases", "kind": "newer", "raw_prefix": "release/4.1.2/"}]


def test_acquisition_limit_exceeds_the_ACQUISITION_budget_only_when_stored():
    """Stored (compression-level 0) -> the downloaded archive exceeds 1 MiB (unavailable). Deflated -> the SAME bytes pass
    download and fail ADMISSION (a completed artifact.invalid) -- the wrong exercise."""
    out, rec = fx.build_fixture(RAW, scenario="acquisition-limit", monitor_exit_status=1, qualification_id=QID)
    assert json.loads(out) == json.loads(RAW) and rec["compression_level"] == 0 and rec["output_bytes"] == 2 * fx.MIB
    stored, deflated = _zip(out, zipfile.ZIP_STORED), _zip(out, zipfile.ZIP_DEFLATED)
    assert len(stored) > rv.MAX_ARCHIVE_BYTES >= len(deflated)
    with pytest.raises(ValueError, match="report.json declares {} bytes".format(2 * fx.MIB)):
        rv.read_archive(deflated, {"digest": "sha256:" + hashlib.sha256(deflated).hexdigest(), "size_in_bytes": len(deflated)})


@pytest.mark.parametrize("raw, kwargs, code", [
    (b'{"exit_code": 1, "exit_code": 0}', {}, "fixture.duplicate_key:exit_code"),
    (b'{"exit_code": 1, "x": NaN}', {}, "fixture.non_json_number:NaN"),
    (b'{"exit_code": true}', {}, "fixture.exit_code"),
    (b'{"exit_code": 1}', {"scenario": "other"}, "fixture.scenario"),
    (b'{"exit_code": 1}', {"monitor_exit_status": 256}, "fixture.monitor_exit_status"),
    (b'{"exit_code": 1}', {"monitor_exit_status": True}, "fixture.monitor_exit_status"),
    (b'{"exit_code": 1}', {"qualification_id": "X-0123456789abcdef"}, "fixture.qualification_id"),
    (b'{"exit_code": 1}', {"qualification_id": "Q-0123456789ABCDEF"}, "fixture.qualification_id"),
    (b'{"exit_code": 1}', {"qualification_id": "Q-0123456789abcde"}, "fixture.qualification_id"),
    (b'{"exit_code": 1}', {"qualification_id": 7}, "fixture.qualification_id"),
])
def test_malformed_input_is_refused_with_its_reason(raw, kwargs, code):
    args = dict({"scenario": "normal", "monitor_exit_status": 1, "qualification_id": QID}, **kwargs)
    with pytest.raises(ValueError) as exc:
        fx.build_fixture(raw, **args)
    assert str(exc.value) == code


def test_the_command_line_writes_exactly_two_files(tmp_path, capsys):
    src = tmp_path / "in.json"
    src.write_bytes(RAW)
    assert fx.main(["--scenario", "claims-disagree", "--input", str(src), "--monitor-exit-status", "1", "--qualification-id", QID,
                    "--output-dir", str(tmp_path / "out")]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["fixture-record.json", "report.json"]
    assert json.loads((tmp_path / "out" / "fixture-record.json").read_text(encoding="ascii"))["scenario"] == "claims-disagree"
    assert "FIXTURE claims-disagree replace_exit_code" in capsys.readouterr().out
