"""The qualification receipt: evidence bound to the EXACT candidate being admitted (owner ruling 2026-10-05b).

It prevents this sequence: qualify candidate A -> change a helper or the lockfile -> promote candidate B on A's receipt. Before
promotion, the identities are RECOMPUTED from the candidate actually being promoted; any difference makes the receipt
inapplicable. A checksum establishes content identity, not who produced it or whether the producer was trustworthy -- the
trusted launcher, the reviewed tests and independent checks carry that part of the argument.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import asdict, dataclass, fields

from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError, canonical, require

logger = logging.getLogger(__name__)

__all__ = ["CHECKPOINTS", "QualificationReceipt", "sha256_bytes", "require_applicable"]

CHECKPOINTS = ("runtime", "dependency")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TREE = re.compile(r"[0-9a-f]{40}")


def sha256_bytes(raw: bytes) -> str:
    require(type(raw) is bytes, "bytes_required")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QualificationReceipt:
    checkpoint: str
    repository_tree: str
    baseline_lock_sha256: str
    candidate_lock_sha256: str
    runtime_version: str
    runtime_release_status: str
    runtime_platform: str
    launcher_sha256: str
    qualification_code_sha256: str
    inventory_sha256: str
    required_test_report_sha256: str
    required_cases: int

    def __post_init__(self):
        require(self.checkpoint in CHECKPOINTS, "receipt_checkpoint")
        require(type(self.repository_tree) is str and bool(_TREE.fullmatch(self.repository_tree)), "receipt_repository_tree")
        for name in ("baseline_lock_sha256", "candidate_lock_sha256", "launcher_sha256", "qualification_code_sha256",
                     "inventory_sha256", "required_test_report_sha256"):
            value = getattr(self, name)
            require(type(value) is str and bool(_SHA256.fullmatch(value)), "receipt_" + name)
        require(type(self.runtime_version) is str and bool(self.runtime_version), "receipt_runtime_version")
        require(type(self.runtime_release_status) is str, "receipt_runtime_release_status")
        require(type(self.runtime_platform) is str and bool(self.runtime_platform), "receipt_runtime_platform")
        require(type(self.required_cases) is int and self.required_cases > 0, "receipt_required_cases")

    def render(self) -> bytes:
        return (canonical({"schema": "gvc.qualification-receipt/1", **asdict(self)}) + "\n").encode("ascii")


def require_applicable(receipt, current) -> None:
    """`current` is a QualificationReceipt RECOMPUTED from the candidate actually being promoted. Every bound identity must match."""
    require(type(receipt) is QualificationReceipt and type(current) is QualificationReceipt, "receipt_type")
    for f in fields(QualificationReceipt):
        if getattr(receipt, f.name) != getattr(current, f.name):
            raise AdmissionError("receipt_inapplicable:" + f.name)
