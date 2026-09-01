# MEASUREMENT 2026-09-01 -- where Phase 1C would attach

**Author: Monzia Moodie**
**Measured at:** `11df0b5`
**Status:** MEASUREMENT ONLY. Nothing is built, and no design is committed.

---

## 0. Why this exists

`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` is the last open subsystem
finding, and item 5 of the adopted priority order is Phase 1C's production
`SourceEvidenceManifest` builder. The rulings are specific about its constraint:

> Do not do `for path in data_root.rglob("*"): add_to_source_evidence_manifest(path)`.
> That would turn inventory into provenance. Instrumentation should follow the
> actual computation.

That means the builder must attach where files are ACTUALLY OPENED. This
records where that is, before a line is written.

Four measurements, all taken 2026-09-01 at `11df0b5`. None has been acted on.

---

## 1. Ninety-three read sites across fifty-three modules

Every tracked file under `src/` was parsed and every call to a reader
(`read_parquet`, `read_csv`, `open`, `load`, and nine others) was judged BY ITS
ARGUMENT -- an extension, a path-like name -- not by the callee name.

That distinction is not cosmetic. The 2026-08-29 acquisition audit reported
1,453 sites when judging `get` by name and FIVE when judging by argument;
`open` and `load` are the same trap.

```
modules that open a data file   53
total sites                     93
```

The heaviest are `training/continual_trainer.py` (8), `data/real_data_prep.py`
(5), `data/dbnsfp.py` (4), `data/pipeline.py` (4), `models/gnn.py` (4).

**Ninety-three is too many to instrument individually.** A builder attaching at
each would be ninety-three edits and ninety-three chances to miss one silently.

---

## 2. FOUR loaders already compute a source digest

This is the finding that changes the design.

```
data/connectors/connector_gnomad_constraint.py:282
    source_sha256=sha256_file(str(self._tsv_path))

data/phylop.py:552
    source_digest = sha256_file(self._path)

data/constraint_canonicalize.py:242
    source_sha256: str = ""          (and source_sha256_verified: bool)

data/phylop_ingest.py:102,181,206,259,284
    source_sha256 threaded through ingestion
```

`SourceArtifactIdentity` requires exactly that field. The expensive part --
streaming a multi-gigabyte file to compute a digest -- is ALREADY DONE in these
paths. A builder does not need a new seam there; it needs to receive what they
already have.

---

## 3. `phylop.py` states ruling 20 independently, and paid for it

Lines 536 to 538:

> Identity here is the schema version and the SOURCE DIGEST. The path is
> deliberately excluded: the same source moved is the same source, and a
> different source at the same path is not.

The rulings say scientific identity must survive `C:\Projects\...`,
`G:\My Drive\...` and `/mnt/volume/...` moving underneath it. A data connector
reached that conclusion first, and lines 531 to 534 record why:

> The cache also carried no identity at all: a filename, and nothing that said
> which source it came from. CACHEIDENTITY-1 in the gnomAD constraint connector
> was exactly this, and a sidecar built by a defective parser was preferred to a
> repaired one because identity was the FILENAME.

`PhyloPIndex._load` builds a `CacheIdentity` from schema version plus source
digest, path excluded, and classifies the result `USABLE` / `STALE` / `CORRUPT`
-- distinguishing "the source changed" from "a cache claims to describe this
source and cannot be read".

`SourceArtifactIdentity` is that shape generalised.

---

## 4. `FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1`

```
data/constraint_canonicalize.py:325   sha256_file(path)
data/phylop_cache.py:158              sha256_file(path, *, chunk=1<<20)
agent_layer/science_claw/ledger.py:70 compute_sha256(path)
```

Two share a name in different modules; the third differs only in spelling.
`phylop.py` imports the second, `connector_gnomad_constraint.py` imports the
first, and neither knows about the third.

**MEASURED BY EXECUTION, not by reading.** All three were loaded and run
against `configs/data_manifest.yaml` and compared to an independently computed
reference:

```
reference                             740d98a10a51e000248ae789900760f3
constraint_canonicalize.py            740d98a10a51e000248ae789900760f3  AGREE
phylop_cache.py                       740d98a10a51e000248ae789900760f3  AGREE
science_claw/ledger.py                740d98a10a51e000248ae789900760f3  AGREE
```

Duplication, not disagreement. All three stream in 1 mebibyte chunks over raw
bytes.

`constraint_canonicalize.py` carries the reason to prefer it:

> Digest RAW BYTES. Never a parsed-and-reserialised object: normalisation would
> silently change the identity of the artefact being recorded.

Phase 1C must CONSUME one of these. Adding a fourth would be the defect this
project names elsewhere: a value stated independently of the thing it
describes.

---

## 5. `ScienceClawLedger` is a THIRD provenance system, and not this one

`agent_layer/science_claw/ledger.py`, 258 lines, read in full.

It records agent-PRODUCED artifacts -- things this project WRITES -- in an
append-only hash-chained log over `SharedState`, and gates message-bus
authorisation:

```
record(artifact_id, producer, sha256, uri, parent_ids)
verify_chain()   re-derives every row_hash; raises on tampering
evaluate(entries, message, computed_sha) -> Verdict(allow, reasons)
```

`SourceEvidenceManifest` answers the opposite question: what evidence a
computation CONSUMED. The ledger's `producer` has no counterpart in a source
manifest, and the manifest's `release_id`, `coordinate_context` and
`SourceRole` have none in the ledger.

The rulings name two systems -- inventory for estate governance, evidence for
experiment provenance. This is a THIRD: output provenance. Not a competitor,
and not a place to build Phase 1C.

Two things in it do belong to Phase 1C's design:

- `compute_sha256` is documented as "the ONLY place that touches the
  filesystem -- the gate never does, which is what makes the gate
  deterministic". That is the separation `SourceArtifactKey` already keeps.
- `_row_hash` uses `json.dumps(..., sort_keys=True, separators=(",", ":"))` so
  a row hash is independent of dict ordering. `domain_digest` in `_digest.py`
  solves the same problem; WHETHER THEY AGREE IS UNMEASURED.

---

## 6. What this does NOT decide

Where the builder attaches. Ninety-three sites is too many and four is
suspiciously few; whether the four digest-computing loaders cover the sources a
run actually consumes is unmeasured.

Whether `domain_digest` and `_row_hash` agree on canonical encoding.

Whether `sha256_file` should be consolidated. Three call sites and two names
is a real duplication, but moving a function used by two connectors and a
ledger is its own unit with its own preimages.

Whether `CacheIdentity` and `SourceArtifactIdentity` should share a base. They
were built for different purposes on the same principle, and merging types
because they rhyme is what the four unnecessary types of 2026-08-28 were.

---

## 7. Status

MEASUREMENT ONLY. `DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` remains
open, and no file in the repository is changed by this record.
