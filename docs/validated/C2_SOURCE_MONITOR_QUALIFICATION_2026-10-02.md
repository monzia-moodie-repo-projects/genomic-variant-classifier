# C2 source-monitor delivery -- isolated live qualification and closure (2026-10-02)

The machine evidence is `records/verification/source-monitor-c2/REC-8563dfe263c04dd4a173b09b8ad0bc66/`
(role VERIFICATION_RESULT): the exact bytes captured, indexed by a typed manifest
(`repository_records/qualification_manifest.py`, schema `gvc.source-monitor-c2-qualification` v1). This
document explains it; the manifest and its tests are the authority. The record is historical evidence and is
never a live posting authority.

## Process deviation

Production cutover preceded isolated live qualification. This is a process deviation. Subsequent isolated
qualification supplies compensating functional evidence.

## How it was qualified

A private repository (`genomic-variant-classifier-qualification`) ran production's exact checker, publisher and
verifier workflow -- the code manifest of each round equals production's -- under its own deployment
configuration, with a fixture producer in place of the monitor's schedule. Five exercises form the acceptance
contract: `normal`, `claims-disagree` (only the report's exit code altered), `acquisition-limit` (a report padded
with whitespace and uploaded uncompressed, so the archive exceeds the 1 MiB acquisition budget),
`duplicate-suppression` (re-running a verifier that had already posted) and `manual-preview`.

| Round | Runtime | Result |
|---|---|---|
| round-1 (2026-10-01 to 2026-10-02 morning) | C2 repairs 1 to 3 | All five passed; the first duplicate-suppression run exposed a history defect (below). |
| round-2 (2026-10-02 18:09-18:15 UTC) | final runtime, production tree `254c506f` | All five passed and were independently reconciled. |

Round 2 reconciliation: every transport archive's SHA-256 and size equal GitHub's own listing; every receipt is
accepted by `open_receipt` against the round's recorded checker identity and GitHub's run record; every delivery
key equals its outcome record and exactly one comment marker, whose body equals `render_comment`; the production
journal reader classifies every attempt correctly; the claims-disagree paired baseline isolates the mutation on
the identical observation; the padded report is the original plus spaces.

## Defects the qualification found -- each repaired, tested and merged before round 2

1. C2 repairs 2: the verifier's commit fetch failed on a private repository (anonymous fetch refused); a
   deployment refusal left no outcome record.
2. C2 repairs 3: after a re-run, an earlier attempt's artifacts were absent from every GitHub listing, so prior
   dispatch was blind to earlier attempts; the attempt journal moved to the publish job's log.
3. C2 repairs 4 (owner review): missing or malformed history was not always UNKNOWN; the reader is now
   conservative, and one action vocabulary is owned by the protocol.

## Retired claim

The publisher emits an intent before invoking the transport. A recovered intent is positive evidence that a POST
may have been issued. Missing or incomplete history remains unknown and blocks another POST. A printed line is
not remote durability. There is no unconditional exactly-once claim: if an acknowledgement and all discoverable
execution history are both gone, unresolved historical deliveries require manual reconciliation.

## Closure against the finite gates (C2 review, 2026-09-29)

| Gate | Evidence |
|---|---|
| 1 strict receipts, identity, distinct unavailable state | the test suite; every archived receipt validates |
| 2 one POST, no implicit retry, prior dispatch across restarts | round-2 duplicate-suppression: prior_dispatch from attempt 1's journal |
| 3 injected faults with exact refusal reasons | the offline fault-injection tests |
| 4 failed verification reaches the writer; manual paths cannot publish | round-2 claims-disagree, acquisition-limit, manual-preview |
| 5 protections ported; test identities compared | every C2 change, compared by node identity |
| 6 suite and installer; a designated test destination | installer validations; the isolated repository (with the deviation above) |
| 7 cutover; authentic failure and success alerts | production run #12; the owner's revised closure via round-2 rejection and unavailable |

## Limitations (also typed in the manifest)

- Round 1's artifacts were extracted, not preserved as transport archives; their transport digests cannot be
  recomputed. Round 1 selected runs by recency, without a correlation identifier.
- The private repository's tree is not in this repository; its deployment configuration and overlay files are.
- Operational qualification does not establish biological validity, calibration on real cohorts or clinical
  utility. Heartbeat monitoring and long-term delivery reconciliation are outside C2.

## Re-verify offline

`python -m pytest tests/unit/test_c2_qualification_record.py` -- stage 1 (inventory and exact bytes, required
cases from the contract) and stage 2 (the semantic replay), from a clean checkout without network access.
