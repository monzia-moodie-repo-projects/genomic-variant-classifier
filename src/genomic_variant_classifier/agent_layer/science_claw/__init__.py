"""
science_claw -- artifact provenance ledger + deterministic policy gate.

Public API:
    ScienceClawLedger  append-only, hash-chained artifact ledger over SharedState
    evaluate           pure integrity+authorization gate (no I/O, deterministic)
    Verdict            frozen result of evaluate(): .allow, .reasons
    compute_sha256     caller-side file hashing helper
    LedgerError        raised on append-only / hash-chain violations
"""

from __future__ import annotations

from genomic_variant_classifier.agent_layer.science_claw.ledger import (
    ScienceClawLedger,
    evaluate,
    Verdict,
    compute_sha256,
    LedgerError,
)

__all__ = [
    "ScienceClawLedger",
    "evaluate",
    "Verdict",
    "compute_sha256",
    "LedgerError",
]