#!/usr/bin/env python3
"""Append dated status-update sections to the three Run-15 baseline-path incidents.

Idempotent (per-file marker check), additive (never edits existing content), newline/BOM-safe.
Grounded in facts confirmed 2026-05-31: the persistence test passes; cnn_1d is a sequence CNN
whose fasta_seq input is unpopulated upstream. Run from project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "## Status update (2026-05-31)"

UPDATES = {
    "docs/incidents/INCIDENT_2026-05-23_run10a-no-checkpoints.md": """

## Status update (2026-05-31): RESOLVED

Per-model incremental checkpointing is present in the base-model loop of
`variant_ensemble.py`: immediately after each model's OOF AUROC is logged it writes
`{name}.joblib`, `{name}_oof.npy`, `{name}_oof_indices.npy`, and `{name}_meta.json`
(with `saved_at_utc`) to `config.model_dir`, inside a log-but-do-not-abort try/except.
Locked by `tests/unit/test_ensemble_persistence.py::test_per_model_checkpoints_written`,
which fits a fast-tabular ensemble on a 300-row fixture and asserts the four-file quartet
plus OOF/index length parity for every base model. A regression that drops the emission
fails CI. Closed.
""",
    "docs/incidents/INCIDENT_2026-05-23_cnn1d-0.5-auroc.md": """

## Status update (2026-05-31): RECLASSIFIED -- missing-feature (Phase B), not a model defect

Confirmed by code read: `cnn_1d` (`CNN1DClassifier`, variant_ensemble.py) is a SEQUENCE CNN
over one-hot DNA windows; it consumes a `fasta_seq` string column. The pipeline never populates
`fasta_seq` (declared `df["fasta_seq"]=None  # populated by ETL enrichment` in
database_connectors.py; train.py and the run path fall back to a constant `"A"*101` poly-A), so
the model trains on constant input and OOF AUROC collapses to 0.5000. This is an absent upstream
FEATURE, not a defect in the model or its wrapper. Under the zero-known-bugs policy (deferral
allowed only for missing-feature scope), `cnn_1d` is honestly EXCLUDED from the Run-15 baseline
via `--skip-cnn`, and re-enabled in Phase B once `fasta_seq` (reference-genome 101-bp window
extraction) is populated -- which also unlocks the RNA splice pipeline that reads the same
column. Signed off as missing-feature scope.
""",
    "docs/incidents/INCIDENT_2026-05-24_cnn1d-cross-platform-unpickle.md": """

## Status update (2026-05-31): DEFERRED with cnn_1d (Phase B)

This incident concerns persisting/unpickling the `cnn_1d` model across platforms. Since `cnn_1d`
is excluded from the Run-15 baseline pending `fasta_seq` population (see
INCIDENT_2026-05-23_cnn1d-0.5-auroc.md), cross-platform unpickling of that model is not on the
baseline path and carries no baseline risk. It will be resolved together with the cnn_1d
re-enablement in Phase B (move the wrapper's model class to module level / persist via
state_dict). Tracked, not a baseline blocker.
""",
}


def main() -> int:
    rc = 0
    for rel, block in UPDATES.items():
        p = Path(rel)
        if not p.exists():
            print(f"SKIP (not found): {rel}")
            rc = 3
            continue
        raw = p.read_bytes()
        text = raw.decode("utf-8")
        nl = "\r\n" if b"\r\n" in raw else "\n"
        work = text.replace("\r\n", "\n")
        if MARKER in work:
            print(f"already updated (no-op): {rel}")
            continue
        work = work.rstrip("\n") + block
        p.write_bytes(work.replace("\n", nl).encode("utf-8"))
        print(f"appended status update: {rel}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
