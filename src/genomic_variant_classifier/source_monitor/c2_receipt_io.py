"""Explicit-path receipt output (C2, owner ruling 2026-09-29; integrated from the reference receipt_io.py). Imports and test
calls never read GITHUB_* -- the protection built after CI run #896.
"""
from __future__ import annotations

from pathlib import Path
import base64
import os
import tempfile

from genomic_variant_classifier.source_monitor.c2_protocol import Refusal, digest, seal, strict_load, validate_payload


def write_receipt(payload, destination):
    """Ordinary verification rejection still writes a receipt and returns exit 2.

    The caller catches ordinary verification exceptions and constructs an
    unavailable decision. SIGKILL, runner loss and I/O failure require the
    downstream coordinator's separately identified unavailable receipt.
    """
    raw = seal(payload)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    # Local atomic visibility; no claim of survival after hosted-runner destruction.
    d = payload["decision"]
    return 0 if d["status"] == "completed" and d["verified"] else 2


def emit_job_output(receipt_path, output_path):
    """Run in its OWN conditional workflow step, including after checker exit 2."""
    raw = Path(receipt_path).read_bytes()
    doc = strict_load(raw)
    if type(doc) is not dict or set(doc) != {"payload", "receipt_sha256"}:
        raise Refusal("envelope.fields")
    validate_payload(doc["payload"])
    if doc["receipt_sha256"] != digest("gvc.receipt/v1", doc["payload"]):
        raise Refusal("receipt.checksum")
    encoded = base64.b64encode(raw).decode("ascii")
    with open(output_path, "a", encoding="ascii") as stream:
        stream.write("receipt_b64=" + encoded + "\n")
    # Base64 provides a single safe output line. It is NOT authentication.
    return len(encoded)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    emit_job_output(args.receipt, args.output)
