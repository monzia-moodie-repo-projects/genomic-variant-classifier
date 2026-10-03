"""C2 qualification fixture builder -- QUALIFICATION-ONLY code (owner ruling 2026-10-01b). It never alters the checker, the
publisher or the scientific interpretation; it transforms the REAL monitor's report BEFORE upload, deterministically,
and records exactly what it did.

Scenarios (one enumerated input):
  normal             the report unchanged (positive control).
  claims-disagree    ONLY exit_code replaced: {0: 1, 1: 0, 2: 0}; the observation is preserved (test 2).
  acquisition-limit  the SAME JSON value padded with whitespace to 2x the archive budget, to be uploaded with
                     compression-level 0 so the DOWNLOADED ARCHIVE exceeds the checker's 1 MiB acquisition budget
                     (test 3, "acquisition-budget refusal"); compressed, it would hit the ADMISSION limit instead.

The record keeps the monitor's PROCESS exit status separate from the report's claimed exit code: the fixture changes a
claim, never the history of what the monitor executed. The original report is uploaded beside the fixture under its own
artifact name (qualification-original-report) for the recorder's PAIRED within-run baseline.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

MIB = 1024 * 1024
SCENARIOS = ("normal", "claims-disagree", "acquisition-limit")
COMPRESSION = {"normal": 6, "claims-disagree": 6, "acquisition-limit": 0}
#: The qualification CORRELATION id (owner ruling 2026-10-02): carried through the workflow input, the run name and this record,
#: so a collector can select its exercise's runs unambiguously, never "the first new run".
QUALIFICATION_ID = re.compile(r"Q-[0-9a-f]{16}")


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("fixture.duplicate_key:{}".format(key))
        result[key] = value
    return result


def _refuse_constant(value):
    raise ValueError("fixture.non_json_number:{}".format(value))


def parse_report(raw: bytes) -> dict:
    report = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_refuse_constant)
    if type(report) is not dict:
        raise ValueError("fixture.report_shape")
    if type(report.get("exit_code")) is not int or report["exit_code"] not in (0, 1, 2):
        raise ValueError("fixture.exit_code")
    return report


def build_fixture(raw: bytes, *, scenario: str, monitor_exit_status: int, qualification_id: str, archive_limit: int = MIB):
    if type(qualification_id) is not str or not QUALIFICATION_ID.fullmatch(qualification_id):
        raise ValueError("fixture.qualification_id")
    if scenario not in SCENARIOS:
        raise ValueError("fixture.scenario")
    if type(raw) is not bytes or not raw:
        raise ValueError("fixture.input")
    if type(archive_limit) is not int or archive_limit <= 0:
        raise ValueError("fixture.archive_limit")
    if type(monitor_exit_status) is not int or not 0 <= monitor_exit_status <= 255:
        raise ValueError("fixture.monitor_exit_status")
    original = parse_report(raw)
    if scenario == "normal":
        output, transformation = raw, "identity"
    elif scenario == "claims-disagree":
        modified = deepcopy(original)
        modified["exit_code"] = {0: 1, 1: 0, 2: 0}[original["exit_code"]]
        output = (json.dumps(modified, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        changed = {k for k in original.keys() | modified.keys() if original.get(k) != modified.get(k)}
        if changed != {"exit_code"} or parse_report(output) != modified:
            raise ValueError("fixture.unexpected_semantic_change")
        transformation = "replace_exit_code"
    else:
        target = max(len(raw) + 1, 2 * archive_limit)
        output = raw + b" " * (target - len(raw))           # whitespace: the JSON value is unchanged
        if parse_report(output) != original:
            raise ValueError("fixture.padding_changed_value")
        transformation = "append_json_whitespace"
    record = {"schema": "gvc.qualification-fixture", "schema_version": 2, "scenario": scenario, "qualification_id": qualification_id,
              "transformation": transformation, "input_sha256": digest(raw), "output_sha256": digest(output),
              "input_bytes": len(raw), "output_bytes": len(output), "original_exit_code": original["exit_code"],
              "fixture_exit_code": parse_report(output)["exit_code"], "monitor_exit_status": monitor_exit_status,
              "compression_level": COMPRESSION[scenario], "archive_limit_bytes": archive_limit}
    return output, record


def write_fixture(raw: bytes, *, scenario: str, monitor_exit_status: int, qualification_id: str, directory) -> dict:
    """report.json (the ONLY file for the source-monitor-report artifact) and fixture-record.json (uploaded separately --
    inside the report artifact it would cause an archive-layout rejection)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    output, record = build_fixture(raw, scenario=scenario, monitor_exit_status=monitor_exit_status, qualification_id=qualification_id)
    (directory / "report.json").write_bytes(output)
    (directory / "fixture-record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--input", required=True, help="the REAL monitor's report.json")
    parser.add_argument("--monitor-exit-status", required=True, type=int)
    parser.add_argument("--qualification-id", required=True, help="Q- followed by 16 lowercase hex digits (the correlation id)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    raw = Path(args.input).read_bytes()
    record = write_fixture(raw, scenario=args.scenario, monitor_exit_status=args.monitor_exit_status,
                           qualification_id=args.qualification_id, directory=args.output_dir)
    print("FIXTURE {} {} -> {} bytes (input {}), compression-level {}".format(record["scenario"], record["transformation"],
                                                                             record["output_bytes"], record["input_bytes"],
                                                                             record["compression_level"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
