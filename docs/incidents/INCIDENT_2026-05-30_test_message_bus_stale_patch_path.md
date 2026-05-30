# INCIDENT 2026-05-30 -- test_message_bus.py Group 4 stale patch-target (pre-existing)

## Status
OPEN. Pre-existing; NOT introduced by Task 3. Fix deferred to a follow-up commit.

## Summary
Running the message-bus self-suite via its __main__ harness
(`python src/genomic_variant_classifier/agent_layer/test_message_bus.py`) reports
`32/34 passed (0 failed, 2 errors)`. Both errors are in Group 4 (DataFreshness):
  - "emits DATA_UPDATED on change"
  - "dry_run sends no message"

## Root cause
Both Group-4 tests call `patch("agents.data_freshness_agent.requests")` and
`patch("agents.data_freshness_agent.ftplib")`, using the legacy flat-layout import
path `agents.`. The module is now imported under the full package path
`genomic_variant_classifier.agent_layer.agents.data_freshness_agent`, and `requests`
is not bound as a top-level attribute reachable via the `agents.` alias, so
unittest.mock.patch raises at setup:

    AttributeError: <module 'agents.data_freshness_agent' ...> does not have the
    attribute 'requests'

This is a test-harness path-staleness issue, not a defect in production code.

## Proof it is pre-existing (not caused by Task 3)
With the Task-3 edits to message_bus.py and shared_state.py stashed (working tree
returned to commit 553d5b6), the same self-suite still reports `32/34 passed
(0 failed, 2 errors)` with the identical two Group-4 errors. Therefore the Task-3
changes (ARTIFACT_PUBLISHED subject + artifact_ledger state key) did not cause them.

## Why the full unit suite is unaffected
`tests/unit/` collects 588 passed / 1 skipped. pytest does not collect the bare
`_test_*` functions in test_message_bus.py by name (the file uses a custom
`_run(group, name, fn)` harness), so these two only surface via the file's own
__main__ runner. The authoritative regression gate (`pytest tests/unit/`) stays green.

## Planned fix (follow-up commit, out of Task-3 scope)
Update the Group-4 patch targets to the full module path, e.g.
`patch("genomic_variant_classifier.agent_layer.agents.data_freshness_agent.requests")`,
or patch `requests`/`ftplib` where they are actually bound. Then re-run the self-suite
to confirm 34/34 under PYTHONIOENCODING=utf-8.

