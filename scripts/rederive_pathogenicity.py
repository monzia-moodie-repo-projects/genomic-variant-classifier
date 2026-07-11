#!/usr/bin/env python
"""rederive_pathogenicity.py (2026-07-10)
Re-derive the 5-class 'pathogenicity' column of a processed ClinVar parquet from its EXISTING
'clinical_sig' column using the CORRECTED mapper (Conflicting classifications of pathogenicity
-> uncertain, not pathogenic). This is a LABEL-ONLY, deterministic correction: pathogenicity is
a pure function of clinical_sig, which is already present verbatim, so re-deriving is equivalent
to re-ingesting for this column -- but surgical and fully auditable.

SAFETY / RIGOR:
  * Writes a NEW artifact (--output), never mutates the input in place. Refuses to overwrite
    without --force.
  * Emits a provenance manifest: input path + MD5, output MD5, mapper version, UTC, and the
    EXACT per-transition counts (e.g. pathogenic->uncertain N).
  * Proves ONLY the pathogenicity column changed: asserts every OTHER column is byte-identical
    between input and output (via element-wise equality; dict-safe), and reports the full old->new pathogenicity
    transition matrix. Any change to a non-pathogenicity column ABORTS (fail loud).
  * The corrected mapper is embedded here identically to the connector/ingest fix (single source
    of truth for this script); it is unit-checked at startup against known strings.
Every acronym expanded on first use: Message-Digest-5 (MD5), ClinVar Variation Identifier
(VariationID), Variant of Uncertain Significance (VUS).
"""
import sys, argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
print("=== rederive_pathogenicity START ===", flush=True)
try:
    import pandas as pd
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

MAPPER_VERSION = "2026-07-10-conflicting-fix-v1"

def map_pathogenicity(sig) -> str:
    if not isinstance(sig, str) or not sig.strip():
        return "uncertain"
    s = sig.lower().strip()
    if s.startswith("conflicting"):
        return "uncertain"
    if s.startswith("pathogenic"):
        return "pathogenic"
    if s.startswith("benign"):
        return "benign"
    if "likely pathogenic" in s:
        return "likely_pathogenic"
    if "likely benign" in s:
        return "likely_benign"
    if "pathogenic" in s:
        return "pathogenic"
    if "benign" in s:
        return "benign"
    return "uncertain"

def _selfcheck():
    assert map_pathogenicity("Conflicting classifications of pathogenicity") == "uncertain"
    assert map_pathogenicity("Conflicting classifications of pathogenicity; risk factor") == "uncertain"
    assert map_pathogenicity("Pathogenic") == "pathogenic"
    assert map_pathogenicity("Pathogenic; drug response") == "pathogenic"
    assert map_pathogenicity("Benign") == "benign"
    assert map_pathogenicity("Uncertain significance") == "uncertain"
    assert map_pathogenicity("Likely pathogenic") == "likely_pathogenic"

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    _selfcheck()
    print("  mapper self-check PASSED", flush=True)

    inp, out = Path(a.input), Path(a.output)
    if not inp.exists():
        print(f"FATAL: input not found: {inp}", flush=True); return 2
    if out.exists() and not a.force:
        print(f"REFUSING to overwrite {out} without --force.", flush=True); return 3
    # Never write onto the input
    if inp.resolve() == out.resolve():
        print("FATAL: --output must differ from --input (never mutate in place).", flush=True); return 4

    df = pd.read_parquet(inp)
    print(f"  loaded {len(df):,} rows / {len(df.columns)} cols from {inp}", flush=True)
    if "clinical_sig" not in df.columns or "pathogenicity" not in df.columns:
        print("FATAL: expected clinical_sig + pathogenicity columns.", flush=True); return 5

    old_path = df["pathogenicity"].astype("string")
    other_cols = [c for c in df.columns if c != "pathogenicity"]
    # Snapshot the non-pathogenicity columns BEFORE re-derivation (deep copy so later assignment
    # to df cannot alias them). We prove invariance by DIRECT per-column equality, which handles
    # object columns containing unhashable cells (e.g. the 'metadata' dict column) natively --
    # unlike hash_pandas_object, which cannot hash dicts.
    before_other = df[other_cols].copy(deep=True)

    # re-derive pathogenicity from clinical_sig
    new_path = df["clinical_sig"].apply(map_pathogenicity)
    df["pathogenicity"] = new_path

    # invariance: every OTHER column must be element-wise identical
    changed_other = [c for c in other_cols if not before_other[c].equals(df[c])]
    if changed_other:
        print(f"FATAL: non-pathogenicity columns changed: {changed_other}. ABORTING.", flush=True)
        return 6
    print(f"  invariance PASSED: all {len(other_cols)} non-pathogenicity columns identical "
          f"(element-wise .equals, dict-safe)", flush=True)

    # transition matrix
    trans = Counter(zip(old_path.fillna("<NA>"), new_path.fillna("<NA>")))
    changed = {f"{o} -> {n}": c for (o, n), c in trans.items() if o != n}
    total_changed = sum(changed.values())
    print(f"\n  pathogenicity transitions (changed only):", flush=True)
    for k, v in sorted(changed.items(), key=lambda kv: -kv[1]):
        print(f"      {k:32s} {v:,}", flush=True)
    print(f"  total rows changed: {total_changed:,}", flush=True)
    print("\n  new pathogenicity distribution:", flush=True)
    for k, v in new_path.value_counts().items():
        print(f"      {k:20s} {v:,}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    manifest = {
        "tool": "rederive_pathogenicity.py",
        "mapper_version": MAPPER_VERSION,
        "utc": datetime.now(timezone.utc).isoformat(),
        "input": str(inp), "input_md5": _md5(inp),
        "output": str(out), "output_md5": _md5(out),
        "rows": int(len(df)),
        "pathogenicity_transitions_changed": changed,
        "total_rows_changed": int(total_changed),
        "non_pathogenicity_columns_invariant": True,
        "new_distribution": {k: int(v) for k, v in new_path.value_counts().items()},
    }
    mpath = out.with_suffix(out.suffix + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\n  output   : {out}  (MD5 {manifest['output_md5']})", flush=True)
    print(f"  manifest : {mpath}", flush=True)
    print("=== rederive_pathogenicity DONE ===", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
