"""Correction of measure_1d1_audit.json (EVE-COVERAGE-IDENTITY-1): writes docs/census/measure_1d1_correction.json.
Author: Monzia Moodie.

Usage, from the repository root (Run 15's files under outputs/run15_rerun_report; the 1D-1 audit's measurement file):
    python scripts/measure_1d1_correction.py --earlier measure_1d1_audit.json \
        --splits outputs/run15_rerun_report/full/splits --log outputs/run15_rerun_report/run15_baseline_master.log \
        --check docs/census/measure_1d1_correction.json        # or --out PATH (never overwrites)

The earlier sections `default_indicator_train` and `eve_gene_coverage_train` compared the persisted Run 15 matrices,
which are StandardScaler outputs (real_data_prep.py:1600), with RAW fill defaults (0.5, 15.0, ...). Those sections are
withdrawn, not re-derived: a numerical default can never establish source coverage, and the raw values are not persisted.

What replaces them is scale-invariant or read from the run's own log:
  * constant_columns: a column is constant after an affine map with nonzero scale iff it was constant before
    (StandardScaler maps a zero-variance column to 0 by setting scale_ = 1), so the set is identical for raw and scaled.
  * eve_source_join_run15: the Run 15 log line where the EVE connector reports that no source was configured.
The JSON is written with allow_nan=False (the earlier file carried a NaN, which strict JSON readers refuse).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

INVALID = ("default_indicator_train", "eve_gene_coverage_train")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def constant_columns(path: Path) -> dict:
    table = pq.read_table(path)
    out = {}
    for name in table.column_names:
        col = table.column(name)
        mm = pc.min_max(col)
        lo, hi = mm["min"].as_py(), mm["max"].as_py()
        out[name] = {"constant": lo == hi and col.null_count == 0, "nulls": col.null_count}
    return {"rows": table.num_rows, "columns": table.num_columns,
            "constant": sorted(k for k, v in out.items() if v["constant"]),
            "columns_with_nulls": sorted(k for k, v in out.items() if v["nulls"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--earlier", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path)
    target.add_argument("--check", type=Path, help="re-derive and byte-compare with this committed record")
    a = ap.parse_args(argv)
    if a.out is not None and a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    earlier = json.loads(a.earlier.read_text(encoding="utf-8"))
    missing = [k for k in INVALID if k not in earlier]
    if missing:
        raise SystemExit(f"earlier file lacks the sections being withdrawn: {missing}")
    kept = {k: v for k, v in earlier.items() if k not in INVALID}

    lines = a.log.read_text(encoding="utf-8").splitlines()
    eve_lines = [(i + 1, s) for i, s in enumerate(lines) if "EVEConnector" in s]
    if len(eve_lines) != 1 or "eve_path not set" not in eve_lines[0][1]:
        raise SystemExit(f"expected exactly one 'eve_path not set' EVEConnector line, found {eve_lines}")

    splits = {}
    for part in ("train", "val", "test"):
        p = a.splits / f"X_{part}.parquet"
        splits[part] = {"sha256": sha256(p), **constant_columns(p)}
    eve_const = {part: "eve_score" in splits[part]["constant"] for part in splits}
    if not all(eve_const.values()):
        raise SystemExit(f"eve_score is not constant in every split: {eve_const}")
    always = sorted(set.intersection(*(set(s["constant"]) for s in splits.values())))

    record = {
        "schema": "gvc.measure-1d1-correction/1",
        "corrects": {"file": a.earlier.name, "sha256": sha256(a.earlier)},
        "withdrawn_sections": {k: "EVE-COVERAGE-IDENTITY-1: compared StandardScaler output with a raw fill default; "
                                  "a numerical default cannot establish source coverage" for k in INVALID},
        "retained_sections": kept,
        "retained_sections_note": "auroc_decomposition_test is retained: a per-column StandardScaler map is affine "
                                  "and increasing, and AUROC is invariant to increasing maps.",
        "eve_source_join_run15": {
            "log": {"file": a.log.name, "sha256": sha256(a.log), "line": eve_lines[0][0], "text": eve_lines[0][1]},
            "finding": "the EVE connector ran with no source configured, so no variant was joined to an EVE score; "
                       "every row holds the fill default (raw 0.5, scaled 0.0). Run 15 EVE coverage = 0 variants, "
                       "by configuration. Gene-level and variant-level EVE coverage are therefore both unmeasured.",
            "eve_score_constant_in": eve_const,
        },
        "constant_columns": {"per_split": splits, "constant_in_every_split": always,
                             "count_constant_in_every_split": len(always)},
    }
    raw = (json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    print(f"withdrawn {list(INVALID)}; constant in every split: {len(always)} of {splits['train']['columns']}")
    if a.check is not None:
        if a.check.read_bytes() != raw:
            print(f"CORRECTION RECORD DIFFERS from the re-derivation: {a.check}")
            return 1
        print(f"correction record re-derived byte-identical: {a.check}")
        return 0
    a.out.write_bytes(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
