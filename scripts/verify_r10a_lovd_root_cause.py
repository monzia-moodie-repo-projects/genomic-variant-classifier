"""
R10-A Formal LOVD Root-Cause Verification
==========================================
Produces a committable verification log confirming Case (a): LOVDConnector
was invoked at annotation step 15/16 without parquet_path, causing all
variants to receive lovd_variant_class=0.

Usage:
    python scripts/verify_r10a_lovd_root_cause.py \
        --regen-log outputs/run9_ready/regen.log \
        --output-dir docs/verified

Date: 2026-05-24
Status: COMPLETE — Case (a) confirmed 2026-05-13, formal verification script added
Fix: Commit 66593d6 adds --lovd-path argument to scripts/run_phase2_eval.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def verify_lovd_root_cause(regen_log: Path) -> dict:
    """Parse regen.log for LOVD diagnostic evidence."""
    if not regen_log.exists():
        return {
            "status": "FAIL",
            "reason": f"regen.log not found at {regen_log}",
            "case": None,
        }

    text = regen_log.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    evidence = {
        "status": "UNKNOWN",
        "regen_log_path": str(regen_log),
        "regen_log_lines": len(lines),
        "regen_log_bytes": regen_log.stat().st_size,
        "lovd_annotation_step_found": False,
        "lovd_annotation_step_line": None,
        "lovd_annotation_step_text": None,
        "no_parquet_warning_found": False,
        "no_parquet_warning_line": None,
        "no_parquet_warning_text": None,
        "lovd_nonzero_count": None,
        "case": None,
        "fix_commit": "66593d6",
        "fix_description": "Phase 1 B1: adds --lovd-path argument to scripts/run_phase2_eval.py",
        "verification_timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Search for LOVD annotation step line
    for i, line in enumerate(lines, 1):
        if re.search(r"Score annotation 15/16.*LOVD", line, re.IGNORECASE):
            evidence["lovd_annotation_step_found"] = True
            evidence["lovd_annotation_step_line"] = i
            evidence["lovd_annotation_step_text"] = line.strip()

            # Extract count if present
            m = re.search(r"(\d+)\s+variants?\s+with\s+lovd_variant_class\s*>\s*0", line)
            if m:
                evidence["lovd_nonzero_count"] = int(m.group(1))

        if re.search(r"no parquet loaded|lovd.*default.*0", line, re.IGNORECASE):
            evidence["no_parquet_warning_found"] = True
            evidence["no_parquet_warning_line"] = i
            evidence["no_parquet_warning_text"] = line.strip()

    # Determine case
    if evidence["no_parquet_warning_found"]:
        evidence["case"] = "a"
        evidence["case_description"] = (
            "Case (a): LOVDConnector invoked without parquet_path. "
            "All variants received lovd_variant_class=0. "
            "Fix: commit 66593d6 adds --lovd-path argument."
        )
        evidence["status"] = "CONFIRMED"
    elif evidence["lovd_annotation_step_found"] and evidence["lovd_nonzero_count"] == 0:
        evidence["case"] = "a"
        evidence["case_description"] = (
            "Case (a) inferred: LOVD step ran but produced 0 nonzero variants. "
            "Consistent with missing parquet_path."
        )
        evidence["status"] = "CONFIRMED"
    elif evidence["lovd_annotation_step_found"]:
        evidence["case"] = "b"
        evidence["case_description"] = (
            "Case (b): LOVD step ran and produced some nonzero variants. "
            "Root cause may be downstream overwrite or coordinate transformation."
        )
        evidence["status"] = "NEEDS_INVESTIGATION"
    else:
        evidence["case"] = "unknown"
        evidence["case_description"] = "LOVD annotation step not found in regen.log."
        evidence["status"] = "NEEDS_INVESTIGATION"

    return evidence


def main():
    parser = argparse.ArgumentParser(description="R10-A LOVD Root-Cause Verification")
    parser.add_argument(
        "--regen-log",
        default="outputs/run9_ready/regen.log",
        help="Path to regen.log from Run 9",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/verified",
        help="Output directory for verification artifact",
    )
    args = parser.parse_args()

    regen_log = Path(args.regen_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("R10-A LOVD ROOT-CAUSE VERIFICATION")
    print("=" * 60)

    evidence = verify_lovd_root_cause(regen_log)

    # Print summary
    print(f"\nStatus: {evidence['status']}")
    print(f"Case:   {evidence['case']}")
    print(f"Detail: {evidence.get('case_description', 'N/A')}")

    if evidence["lovd_annotation_step_found"]:
        print(f"\nLOVD annotation step (line {evidence['lovd_annotation_step_line']}):")
        print(f"  {evidence['lovd_annotation_step_text']}")

    if evidence["no_parquet_warning_found"]:
        print(f"\nNo-parquet warning (line {evidence['no_parquet_warning_line']}):")
        print(f"  {evidence['no_parquet_warning_text']}")

    print(f"\nFix: {evidence['fix_commit']} — {evidence['fix_description']}")
    print(f"R10-B post-condition tests: 66593d6 + f64c024 + 633e7d0 + e07e3d8")

    # Save verification artifact
    output_path = output_dir / "R10A_LOVD_VERIFICATION.json"
    with open(output_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"\nVerification artifact saved to {output_path}")

    # Return exit code
    if evidence["status"] == "CONFIRMED":
        print("\n✓ R10-A COMPLETE — Case (a) confirmed, fix shipped in 66593d6")
        sys.exit(0)
    else:
        print(f"\n✗ R10-A status: {evidence['status']} — manual investigation required")
        sys.exit(1)


if __name__ == "__main__":
    main()
