# INCIDENT 2026-05-30 -- test_message_bus.py "history ordering" timing flakiness (pre-existing)

## Status
RESOLVED 2026-07-01 (Unit 2). Production MessageBus.send already stamps a monotonic
per-message seq and MessageBus.history sorts by (timestamp, seq), so ordering is
deterministic under a timestamp tie. Both history-ordering tests (the strict-order
test and the frozen-clock tie test), renamed from the _test_ prefix to test_, now
pass under collected pytest at tests/unit/test_message_bus.py. Verified this run:
35/35 message-bus tests pass collected (exit 0); full suite 1545 passed, 7 skipped.
No production message_bus.py change was required to close this; the deterministic
seq tiebreak was already present.

## Summary
The message-bus self-suite (run via its __main__ harness,
`python src/genomic_variant_classifier/agent_layer/test_message_bus.py`) reports a
FAIL on Group 1 "history ordering" across repeated runs.

## Root cause
`_test_history_ordering` sends three messages back-to-back and asserts strict
most-recent-first ordering. `MessageBus.send` stamps each message with
`datetime.now(timezone.utc).isoformat()`, and `MessageBus.history` sorts solely on
that timestamp string. Three sends can land within the same microsecond, so the
sort key ties and the tie-break is unstable. The strict-order assertion then fails
non-deterministically. This is a timing-flaky test, not a defect in production code.

## Proof it is pre-existing (not caused by Task 3)
Reproduced 5x on the Task-3 working tree: "history ordering" FAILED all 5 runs.
Then all three Task-3 edits (message_bus.py, shared_state.py, orchestrator.py) were
stashed (tree returned to commit 553d5b6) and the suite was run 5x again:
"history ordering" FAILED all 5 stashed runs as well. Identical behavior with and
without the Task-3 changes proves the flakiness is independent of Task 3.

## Why the authoritative gate is unaffected
`pytest tests/unit/` collects 595 passed / 1 skipped. It does not collect the bare
`_test_*` functions in test_message_bus.py (custom `_run(group, name, fn)` harness),
so this only surfaces via the file's own __main__ runner.

## Planned fix (follow-up commit, out of Task-3 scope)
Make ordering deterministic: add a monotonic per-message sequence counter and sort
history by (timestamp, seq), or have the test assert a stable order via injected
timestamps. Then re-run the self-suite to confirm a stable pass under
PYTHONIOENCODING=utf-8.
