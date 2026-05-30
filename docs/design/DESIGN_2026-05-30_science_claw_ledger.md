# DESIGN 2026-05-30 -- ScienceClaw artifact ledger + deterministic policy gate (Task 3)

## Purpose
Provide artifact provenance (integrity) AND action authorization in a single,
deterministic gate layered on the existing SharedState substrate, and enforce it in
the orchestrator hot path before agents act.

## Components
- `science_claw/ledger.py`
  - `ScienceClawLedger(shared_state)` -- append-only, hash-chained record under the
    new `artifact_ledger` key in agent_state.json. Each row:
    {index, artifact_id, producer, sha256, uri, created_at, parent_ids, prev_hash, row_hash}.
    `prev_hash` chains the previous row's `row_hash`; `verify_chain()` recomputes the
    whole chain and raises `LedgerError` on any tamper. `record()` rejects duplicate
    artifact_id (append-only; ids unique).
  - `compute_sha256(path)` -- caller-side file hashing. The ONLY filesystem touch;
    keeps the gate pure.
  - `evaluate(ledger_entries, message, computed_sha) -> Verdict(allow, reasons)` --
    PURE: no I/O, no clock. Same inputs -> same Verdict. Enforces:
      INTEGRITY (only when payload carries BOTH artifact_id and artifact_sha256):
        deny if the artifact is missing from the ledger, if computed_sha is None,
        if recorded sha != computed_sha, or if message-declared sha != recorded sha.
      AUTHORIZATION: deny if `requires_approval` and `approved is not True`.
    When no artifact is referenced, integrity is a strict no-op (authorization alone
    decides), so existing message-bus behavior is unchanged.

## Message-bus integration
- New canonical subject `ARTIFACT_PUBLISHED`, added to `ALL_SUBJECTS` AND to
  `APPROVAL_REQUIRED_SUBJECTS` (requires human approval by default).
- Artifacts are referenced via payload keys `artifact_id` + `artifact_sha256`; any
  message (including existing CHECKPOINT_READY) can opt into integrity checking.

## Orchestrator enforcement (hot path)
- `Orchestrator.enforce_artifact_gate(agent_names)` runs inside `run_pipeline`
  immediately after `_deliver_pending_messages` and before the agent loop. For each
  agent's actionable, artifact-referencing message it computes the on-disk SHA from
  the ledger row's uri, calls `evaluate(...)`, and on DENY both rejects the message
  (so the agent will not act on it this run) AND adds a human-review item. DENY
  blocks. Messages without an artifact reference are untouched. No agent code changed.

## Determinism boundary
SHA-256 and timestamps are computed by callers and passed into `evaluate` as data.
The gate function reads no clock and performs no I/O, so it is fully unit-testable
with fixed inputs and yields identical verdicts for identical inputs.

## Tests
- `tests/unit/test_science_claw_ledger.py` -- 21 tests (subject wiring, append-only
  chain, tamper detection, compute_sha256, determinism, integrity, authorization,
  combined, no-op safety).
- `tests/unit/test_science_claw_orchestrator_gate.py` -- 7 tests (method exists,
  run_pipeline invokes gate, DENY blocks tampered/missing, ALLOW valid, no-op for
  non-artifact, ignores unapproved). Real fixtures, no mock patching.
- Full unit tree: 595 passed / 1 skipped.

## Known pre-existing issues (separate INCIDENTs, out of Task-3 scope)
- test_message_bus.py Group 4 stale patch-target (AttributeError on `agents.` path).
- test_message_bus.py "history ordering" timing flakiness (equal-microsecond ties).
Both proven pre-existing via stash test at 553d5b6.
